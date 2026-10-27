"""Split a policy wording into clause-sized chunks for retrieval.

Insurance wordings are structured: PART/SECTION headings, then ALL-CAPS
sub-headings (INSURING AGREEMENT, EXCLUSIONS, ...), then numbered clauses.
We keep that structure instead of cutting every N characters, so a chunk
never mixes two sections. Each chunk carries its heading path and page range,
which is what the agent cites ("PART D > EXCLUSIONS, p. 8").

Input is the text of each page as pypdf extracts it; no PDF parsing here, so
this module is unit-testable with plain strings. The rules are generic (running
headers, contents pages, split headings), not tuned per insurer.
"""

import re
from collections import Counter

MAX_WORDS = 280  # bge-small truncates at 512 tokens; ~280 words + heading stays under
OVERLAP_LINES = (
    2  # repeat the tail of the previous chunk so a clause split mid-way keeps context
)

NOISE = [
    re.compile(
        r"page\s+\d+\s+of\s+\d+", re.IGNORECASE
    ),  # "Page 8 of 12 ...", "A-30ME (08-11) Page 11 of 18"
    re.compile(r"^\s*(page\s*)?\d{1,3}\s*$", re.IGNORECASE),  # bare page number
    re.compile(r"\.{5,}"),  # contents-page dot leaders
    re.compile(r"copyright", re.IGNORECASE),
]
TOC_MIN_DOT_LINES = 5
TOP_LEVEL = re.compile(r"^(PART|SECTION)\s+[A-Z0-9IVX]+\b")
BARE_TOP_LEVEL = re.compile(r"^(PART|SECTION)\s+[A-Z0-9IVX]+$")
NUMBERED = re.compile(r"^\(?[0-9A-Za-z]{1,3}[.)]\s+")
# A heading line that starts or ends with one of these continues the previous/next line.
CONNECTORS = {"OR", "AND", "OF", "TO", "FOR", "THE", "IN", "UNDER", "WITH", "A", "AN"}
# Prefixes whose hyphen is real when a word is split across lines ("non-\nowned").
KEEP_HYPHEN = ("non", "self", "semi", "co")


def clean_text(text):
    # Progressive's PDF encodes its dashes in a font pypdf can't map: U+FFFD.
    text = text.replace("�", " - ").replace("–", "-").replace("—", "-")
    return text.replace("“", '"').replace("”", '"').replace("’", "'")


def normalise(line):
    return " ".join(line.split())


def is_heading(line):
    """Short ALL-CAPS line, optionally numbered ("1. NOTICE")."""
    letters = [c for c in NUMBERED.sub("", line) if c.isalpha()]
    return (
        3 <= len(letters)
        # Long enough for a heading joined from two lines (Progressive's
        # "EXCLUSIONS-READ THE FOLLOWING EXCLUSIONS CAREFULLY. IF AN EXCLUSION ...").
        and len(line) <= 140
        and sum(c.isupper() for c in letters) / len(letters) > 0.9
    )


def heading_label(line):
    """Keep the path readable: a heading's first sentence is enough to cite."""
    m = NUMBERED.match(line)
    number = m.group(0) if m else ""  # keep "1. " in "1. NOTICE"
    return number + line[len(number) :].split(". ", 1)[0].rstrip(".")


def running_headers(pages):
    """Lines that appear on most pages (form numbers, insurer names) are page furniture."""
    if len(pages) < 4:
        return set()
    counts = Counter(
        line
        for text in pages
        for line in {normalise(raw) for raw in clean_text(text).splitlines()}
    )
    return {line for line, n in counts.items() if line and n > len(pages) / 2}


def page_lines(pages):
    """Yield (page_number, line) with noise removed and split words rejoined."""
    repeated = running_headers(pages)
    pending = None  # (page, text) of a line ending in a hyphen
    for page_no, text in enumerate(pages, start=1):
        lines = [normalise(raw) for raw in clean_text(text).splitlines()]
        if sum(bool(NOISE[2].search(line)) for line in lines) >= TOC_MIN_DOT_LINES:
            continue  # contents page: its headings would pollute the heading path
        for line in lines:
            if not line or line in repeated or any(p.search(line) for p in NOISE):
                continue
            if pending:
                head = pending[1][:-1]
                if line[0].islower():  # "Declara-" + "tions"
                    prefix = head.rsplit(" ", 1)[-1].lower()
                    joiner = "-" if prefix in KEEP_HYPHEN else ""
                    first, _, rest = line.partition(" ")
                    yield pending[0], head + joiner + first
                    line = rest
                elif is_heading(pending[1]) and is_heading(
                    line
                ):  # "IF AN EX-" + "CLUSION APPLIES"
                    line = head + line
                else:
                    yield pending
                pending = None
                if not line:
                    continue
            if line.endswith("-"):
                pending = (page_no, line)
            else:
                yield page_no, line
    if pending:
        yield pending


def continues_heading(previous, line):
    """True if `line` is the second line of the heading `previous`."""
    if TOP_LEVEL.match(line) or NUMBERED.match(line):
        return False
    return previous.split()[-1] in CONNECTORS or line.split()[0] in CONNECTORS


def is_title_line(line):
    """A short Title Case line, like GEICO's "Physical Damage Coverages"."""
    return len(line) <= 60 and line[0].isupper() and not line.endswith(".")


def chunk_pages(pages, max_words=MAX_WORDS):
    """Return chunks: {"chunk_index", "section", "page_start", "page_end", "text"}."""
    chunks = []
    # Heading path, three levels: "PART D ..." > "CONDITIONS" > "1. NOTICE"
    path = {"top": "", "sub": "", "item": ""}
    last_heading = None  # (level, text) if no body line has followed it yet
    body = []  # [(page, line)] under the current heading

    def section():
        return " > ".join(s for s in path.values() if s) or "PREAMBLE"

    def flush():
        nonlocal body
        start = 0
        while start < len(body):
            end, words = start, 0
            while end < len(body) and (
                end == start or words + len(body[end][1].split()) <= max_words
            ):
                words += len(body[end][1].split())
                end += 1
            part = body[start:end]
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "section": section(),
                    "page_start": part[0][0],
                    "page_end": part[-1][0],
                    "text": "\n".join(line for _, line in part),
                }
            )
            if end == len(body):
                break
            start = max(end - OVERLAP_LINES, start + 1)
        body = []

    for page_no, line in page_lines(pages):
        if (
            last_heading
            and last_heading[0] == "top"
            and BARE_TOP_LEVEL.match(path["top"])
            and is_title_line(line)
        ):
            # "SECTION III" followed by its title: take the title into the path
            path["top"] += " - " + line.upper()
            continue
        if is_heading(line):
            if last_heading and continues_heading(last_heading[1], line):
                level = last_heading[0]
                path[level] += " " + line
                last_heading = (level, path[level])
                continue
            flush()
            line = heading_label(line)
            if TOP_LEVEL.match(line):
                path.update(top=line, sub="", item="")
                level = "top"
            elif NUMBERED.match(line):
                path["item"] = line
                level = "item"
            else:
                path.update(sub=line, item="")
                level = "sub"
            last_heading = (level, line)
        else:
            body.append((page_no, line))
            last_heading = None
    flush()
    return chunks


def embedding_text(chunk):
    """What gets embedded: the heading path gives the clause its context
    ("EXCLUSIONS" alone says nothing about which coverage it excludes)."""
    return f"{chunk['section']}\n{chunk['text']}"
