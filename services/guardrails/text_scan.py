"""Prompt-injection scan for untrusted text: rules plus a classifier.

Two layers, because each misses what the other catches:
- Rules (regexes) are exact and explainable: "ignore previous instructions",
  fake chat markup, text addressed to "the AI", attempts to set the fraud score,
  tool names. They don't generalise to paraphrases.
- The classifier (ProtectAI's deberta-v3-base-prompt-injection-v2, Apache-2.0,
  run on ONNX Runtime) generalises to paraphrases but is a black box and can be
  fooled by unusual phrasings or other languages (it's English only).

Text is normalised first: NFKC folds full-width and other look-alike forms into
plain characters, and invisible characters are removed, so "ignore" written
in full-width letters or with a zero-width space inside matches like "ignore". Text hidden in the Unicode tag block
(U+E0000-E007F: invisible, but readable to some LLM tokenizers, "ASCII
smuggling") and long base64 runs are decoded and scanned too.

Two profiles decide what counts as flagged:
- "untrusted" (customer text, text in photos): a rule hit OR a classifier score
  at or above the threshold.
- "document" (policy wording from our own PDFs, pinned by SHA-256 at ingest):
  rule hits only; the score is still reported. Measured on the 221 policy
  chunks, the classifier flagged 9 (4%), up to 0.997 on the insuring agreement
  ("we agree with you as follows") and on a form number: legal text written in
  the second person reads like instructions to it. Withholding those would
  take the core coverage text away from the LLM on most claims.

A hit doesn't block anything by itself: the caller keeps flagged text away from
the LLM and shows it to a human (see services/agent/graph.py).
"""

import base64
import binascii
import os
import re
import threading
import unicodedata
from collections import OrderedDict
from pathlib import Path

import numpy as np

THRESHOLD = float(os.getenv("INJECTION_THRESHOLD", "0.5"))
# Threads per ONNX Runtime session; 0 = ORT's default, one per core of the
# *machine*. In a container with a CPU limit, set it to the limit (the Helm
# charts do): on a 20-core node, a pod limited to 2 CPUs would otherwise run
# 20 threads and spend most of each period throttled.
ORT_THREADS = int(os.getenv("ORT_THREADS", "0"))
PROFILES = ("untrusted", "document")
INFO_RULES = {"hidden_characters", "mixed_script"}
MAX_TOKENS = 512  # the model's limit, including [CLS] and [SEP]
STRIDE = (
    384  # windows overlap by 128 tokens, so no phrase is only ever seen cut in half
)

ZERO_WIDTH = "\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff"
INVISIBLE = re.compile(f"[{ZERO_WIDTH}\U000e0000-\U000e007f]")
TAG_CHARS = re.compile("[\U000e0020-\U000e007e]+")
BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")

# Cyrillic and Greek letters that look like Latin ones. NFKC leaves them alone
# (they're different letters), so "ignore" spelled with a Cyrillic o slips past
# every English rule and the classifier. They're mapped to Latin only inside
# words that also contain Latin letters (mixed script, the classic spoofing
# sign), so genuine Russian or Greek text isn't mangled.
CONFUSABLES = {
    # Cyrillic lower: a e o p c y x i j s d h l q w
    0x430: "a",
    0x435: "e",
    0x43E: "o",
    0x440: "p",
    0x441: "c",
    0x443: "y",
    0x445: "x",
    0x456: "i",
    0x458: "j",
    0x455: "s",
    0x501: "d",
    0x4BB: "h",
    0x4CF: "l",
    0x51B: "q",
    0x51D: "w",
    # Cyrillic upper
    0x410: "A",
    0x412: "B",
    0x415: "E",
    0x41A: "K",
    0x41C: "M",
    0x41D: "H",
    0x41E: "O",
    0x420: "P",
    0x421: "C",
    0x422: "T",
    0x425: "X",
    0x423: "Y",
    0x406: "I",
    0x408: "J",
    0x405: "S",
    # Greek
    0x3B1: "a",
    0x3BF: "o",
    0x3B9: "i",
    0x3BA: "k",
    0x3BD: "v",
    0x3C1: "p",
    0x3C4: "t",
    0x3C5: "u",
    0x3C7: "x",
    0x391: "A",
    0x392: "B",
    0x395: "E",
    0x396: "Z",
    0x397: "H",
    0x399: "I",
    0x39A: "K",
    0x39C: "M",
    0x39D: "N",
    0x39F: "O",
    0x3A1: "P",
    0x3A4: "T",
    0x3A5: "Y",
    0x3A7: "X",
}
WORD = re.compile(r"\w+")

AI = (
    r"(ai|a\.i\.|assistant|llm|language model|model|bot|chatbot|gpt|chatgpt|claude"
    r"|claims? (system|agent|assistant|bot|reviewer)|automated (system|reviewer))"
)
TOOLS = (
    "approve_claim|search_clauses|get_policy|check_coverage|estimate_repair_cost"
    "|find_duplicate_photos|get_claim_history"
)

# (rule id, pattern). Matched against lower-cased, normalised text that still
# has its line breaks (chat markup is only suspicious at the start of a line).
RULES = [
    (
        "override_instructions",
        # The qualifier is required: "the other driver ignored all traffic rules"
        # is a normal claim description.
        (
            r"\b(ignore|disregard|forget|override|bypass)\w*\b.{0,30}"
            r"\b(previous|prior|above|earlier|preceding|your|system|these|those)\b.{0,20}"
            r"\b(instructions?|prompts?|rules?|directions?|guidelines?|guardrails?|context)\b"
            r"|\b(ignore|disregard|forget)\b (everything|anything|all) (above|before|previously|you were told)"
        ),
    ),
    (
        "role_override",
        (
            r"\b(you are now|you're now|act as (an?|the) |pretend (to be|you are)|from now on,? you"
            r"|new (system )?instructions?\s*:|developer mode|jailbreak|do anything now)"
        ),
    ),
    (
        "chat_markup",
        # Only at the start of a line: "...windshield. System: none" is prose.
        (
            r"(^|\n)\s*(#+\s*)?(system|assistant|developer)( prompt)?\s*:"
            r"|<\|[a-z_]+\|>|\[/?(inst|sys)\]|<</?sys>>"
            r"|</?(claim_description|clause|claim|facts|text|tool|system)\b"
        ),
    ),
    (
        "prompt_leak",
        (
            r"\b(reveal|show|print|repeat|output|tell me|what (is|are))\b.{0,20}"
            r"\b(your|the)\s+(system\s+|initial\s+|hidden\s+)?(prompt|instructions)\b"
        ),
    ),
    (
        "addressed_to_ai",
        (
            rf"\b(note|message|instructions?|attention|reminder)\s+(to|for)\s+(the\s+|any\s+)?{AI}\b"
            r"|\b(dear|hey|hi|hello)\s+(ai|assistant|chatgpt|gpt|claude|llm|bot)\b"
        ),
    ),
    (
        "decision_override",
        # "risk" alone isn't enough: "it put my kids at risk to injury".
        (
            r"\b(set|mark|change|make|lower|reduce|put)\b.{0,25}\b(fraud|risk)\s*(score|level|flag|rating|assessment)\b"
            r"|\bauto[- ]?approve"
            r"|\b(approve|pay)\b.{0,40}\bwithout (any )?(review|checking|checks|verification|a human)"
            r"|\b(mark|classify|label|flag)\b.{0,25}\bas (not fraud|legitimate|verified|pre-?approved|low risk)"
            # An order to whoever processes the claim: "mark it covered".
            r"|\b(mark|record|treat|report)\s+(it|this|this claim|the claim|my claim)\s+(as\s+)?"
            r"(covered|approved|paid|legitimate|verified|genuine)\b"
        ),
    ),
    (
        "tool_call",
        (
            rf"\b(call|invoke|run|execute)\s+(the\s+)?(tool|function)\b|\b({TOOLS})\b"
            r'|"(name|function|tool_calls?)"\s*:'
        ),
    ),
]
# DOTALL: OCR text arrives one line per text region, and "ignore previous" /
# "instructions" on two lines of a sign is still one sentence.
COMPILED = [(rule, re.compile(pattern, re.DOTALL)) for rule, pattern in RULES]


def unmix_word(match):
    word = match.group(0)
    if any(ord(c) in CONFUSABLES for c in word) and any(
        "a" <= c.lower() <= "z" for c in word
    ):
        return word.translate(CONFUSABLES)
    return word


def mixed_script(text):
    return any(unmix_word(m) != m.group(0) for m in WORD.finditer(text))


def normalise(text):
    """NFKC, invisible characters removed, look-alike letters in mixed-script
    words mapped to Latin, spaces collapsed (line breaks kept)."""
    text = INVISIBLE.sub("", unicodedata.normalize("NFKC", text))
    text = WORD.sub(unmix_word, text)
    return re.sub(r"[^\S\n]+", " ", text).strip()


def reveal_tag_text(text):
    """Decode Unicode tag characters (each is an invisible copy of an ASCII one)."""
    return " ".join(
        "".join(chr(ord(c) - 0xE0000) for c in run) for run in TAG_CHARS.findall(text)
    )


def decode_base64_runs(text):
    """Base64 runs that decode to mostly printable text."""
    out = []
    for run in BASE64_RUN.findall(text):
        try:
            raw = base64.b64decode(run + "=" * (-len(run) % 4), validate=False)
            decoded = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            continue
        if (
            decoded
            and sum(c.isprintable() or c.isspace() for c in decoded) / len(decoded)
            > 0.9
        ):
            out.append(decoded)
    return out


def rule_hits(text):
    lowered = normalise(text).lower()
    return [rule for rule, pattern in COMPILED if pattern.search(lowered)]


class Classifier:
    """deberta-v3-base-prompt-injection-v2 on ONNX Runtime (no PyTorch).

    Texts longer than the model's 512 tokens are split into overlapping
    windows; a text's score is its highest window score."""

    def __init__(self, model_dir):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        options = ort.SessionOptions()
        options.intra_op_num_threads = ORT_THREADS
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.cls = self.tokenizer.token_to_id("[CLS]")
        self.sep = self.tokenizer.token_to_id("[SEP]")
        self.pad = self.tokenizer.token_to_id("[PAD]")

    def windows(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        size = MAX_TOKENS - 2
        starts = range(0, max(len(ids) - size, 0) + 1, STRIDE)
        out = [ids[s : s + size] for s in starts]
        if len(ids) > size and (len(ids) - size) % STRIDE:
            out.append(ids[-size:])  # make sure the tail is covered
        return [[self.cls, *w, self.sep] for w in out]

    def scores(self, texts):
        """Probability of INJECTION per text, scored in one batch."""
        windows, owner = [], []
        for i, text in enumerate(texts):
            for w in self.windows(text):
                windows.append(w)
                owner.append(i)
        width = max(len(w) for w in windows)
        ids = np.full((len(windows), width), self.pad, dtype=np.int64)
        mask = np.zeros_like(ids)
        for row, w in enumerate(windows):
            ids[row, : len(w)] = w
            mask[row, : len(w)] = 1
        logits = self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits[:, 1]) / np.exp(logits).sum(axis=1)
        best = [0.0] * len(texts)
        for i, p in zip(owner, probs, strict=True):
            best[i] = max(best[i], float(p))
        return best


class Scanner:
    CACHE_SIZE = 4096

    def __init__(self, classifier=None, threshold=THRESHOLD):
        self.classifier = classifier  # None: rules only (reported as the mode)
        self.threshold = threshold
        self._lock = threading.Lock()  # one batch at a time through the model
        # Policy clauses come back from search again and again: cache by text.
        self._cache = OrderedDict()

    def _classify(self, texts):
        if self.classifier is None or not texts:
            return [None] * len(texts)
        with self._lock:
            return self.classifier.scores(texts)

    def _scan_batch(self, texts):
        cleans = [normalise(t) for t in texts]
        # Hidden payloads are scanned as texts of their own, in the same batch.
        hidden = [
            [h for h in (reveal_tag_text(t), *decode_base64_runs(c)) if h]
            for t, c in zip(texts, cleans, strict=True)
        ]
        batch = [c for c in cleans if c] + [h for hs in hidden for h in hs]
        scores = iter(self._classify(batch))
        main = [next(scores) if c else 0.0 for c in cleans]
        results = []
        for text, score, hs in zip(texts, main, hidden, strict=True):
            rules = rule_hits(text)
            if INVISIBLE.search(text):
                rules.append("hidden_characters")
            if mixed_script(unicodedata.normalize("NFKC", text)):
                rules.append("mixed_script")
            smuggled = False
            for h in hs:
                s = next(scores)
                smuggled |= bool(rule_hits(h)) or (
                    s is not None and s >= self.threshold
                )
            if smuggled:
                rules.append("smuggled_payload")
            results.append(
                {"score": None if score is None else round(score, 4), "rules": rules}
            )
        return results

    def _flagged(self, result, profile):
        # Neither is an attack alone: emoji use zero-width joiners, and names
        # can mix scripts. They're reported; the other rules decide.
        by_rule = any(r not in INFO_RULES for r in result["rules"])
        if profile == "document":
            return by_rule
        score = result["score"]
        return by_rule or (score is not None and score >= self.threshold)

    def scan(self, texts, profile="untrusted"):
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}")
        todo = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if todo:
            for text, result in zip(todo, self._scan_batch(todo), strict=True):
                self._cache[text] = result
                if len(self._cache) > self.CACHE_SIZE:
                    self._cache.popitem(last=False)
        out = []
        for t in texts:
            # Look up again: a concurrent request may have evicted it.
            r = self._cache.get(t) or self._scan_batch([t])[0]
            out.append(
                {
                    "flagged": self._flagged(r, profile),
                    "score": r["score"],
                    "rules": list(r["rules"]),
                }
            )
        return out

    @property
    def mode(self):
        return "rules+classifier" if self.classifier else "rules-only"
