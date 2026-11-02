"""The LLM behind the agent: Ollama (Ollama Cloud by default, or a local Ollama).

Two calls are used:
- structured(): the answer must validate against a pydantic schema. Ollama
  constrains generation to the JSON schema (format=...), and we validate again
  anyway, since the model's output is untrusted. Not every cloud model honours
  format= (gemma4 on Ollama Cloud answers in prose, or wraps JSON in a code
  fence), so the fields are also listed in the system prompt and the JSON
  object is cut out of any fence before it's validated.
- chat(): free chat with tool definitions, for the bounded clause-search loop.

structured() can also take images, for the vision model that reviews photos
(VLM_MODEL, a second OllamaLLM). It only ever gets the redacted copies that
services/guardrails returns: faces and any text in the photo pixelated, no EXIF.

Temperature 0 so the same claim gives the same extraction where possible.

Every call is a Langfuse generation when tracing is on (tracing.py): prompt,
output, model and token counts; photos are replaced by a count.

Every call is also metered (usage.py): token budgets per run and per day, a
concurrency cap, Prometheus counts. Output is capped at LLM_MAX_OUTPUT_TOKENS
(reasoning included): a model stuck in a loop stops there, and its truncated
JSON fails validation, so the node falls back. A 429 or 5xx from the provider
is retried twice with backoff; anything else (auth, timeout) is not, since a
retry wouldn't help or would double a two-minute wait.
"""

import asyncio
import json
import os
import random
import time

import tracing
import usage

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
# gpt-oss takes a reasoning level; "low" keeps each call to a few seconds.
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "low")
# Vision model for photo_review. gemma4:31b is the only vision model on Ollama
# Cloud's free tier (checked 2026-10-29; qwen3-vl, kimi, minimax and mistral
# need paid usage). It has no reasoning level to set, unlike gpt-oss.
VLM_MODEL = os.getenv("VLM_MODEL", "gemma4:31b")
VLM_THINK = os.getenv("VLM_THINK", "")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "120"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_BACKOFF_S = (2.0, 6.0)  # one wait per retry, plus up to 1 s of jitter


class LLMError(RuntimeError):
    pass


def _status(error):
    """The HTTP status of an Ollama error, or None (network, timeout, ...)."""
    code = getattr(error, "status_code", None)
    return code if isinstance(code, int) and code > 0 else None


def field_list(schema):
    """The schema's fields in plain words, for the system prompt. Listing them
    works better than pasting the JSON schema: some models then answer with
    the schema's own shape ({"properties": {...}}). Nested objects get their
    own field list: without it, gemma4 names nested fields itself
    ("statement" for "text"), measured in services/eval's judge."""
    full = schema.model_json_schema()
    defs = full.get("$defs", {})

    def describe(p, indent):
        p = defs.get(p.get("$ref", "").rsplit("/", 1)[-1], p)
        items = defs.get(
            p.get("items", {}).get("$ref", "").rsplit("/", 1)[-1], p.get("items", {})
        )
        if "enum" in p:
            return " | ".join(json.dumps(v) for v in p["enum"])
        if p.get("type") == "array" and "enum" in items:
            return "list of " + " | ".join(json.dumps(v) for v in items["enum"])
        if p.get("type") == "array" and "properties" in items:
            return "list of objects, each with exactly these fields:\n" + fields(
                items, indent + 1
            )
        if p.get("type") == "array":
            return f"list of {items.get('type', 'string')}s"
        return p.get("type", "string")

    def fields(obj, indent=0):
        pad = "  " * indent
        return "\n".join(
            f'{pad}- "{name}": {describe(p, indent)}'
            for name, p in obj["properties"].items()
        )

    return (
        "Reply with only a JSON object (no prose, no code fences) with exactly "
        "these fields:\n" + fields(full)
    )


def json_object(text):
    """The JSON object in a reply, also when it's wrapped in a code fence or a
    sentence. The result is still validated against the schema."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start : end + 1])


class OllamaLLM:
    def __init__(
        self,
        host=OLLAMA_HOST,
        model=OLLAMA_MODEL,
        api_key=OLLAMA_API_KEY,
        think=OLLAMA_THINK,
    ):
        from ollama import AsyncClient

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self.client = AsyncClient(host=host, headers=headers, timeout=LLM_TIMEOUT_S)
        self.model = model
        self.think = think or None

    async def _call(self, messages, options, kwargs):
        """The provider call, retried on 429 and 5xx only."""
        for attempt in range(len(RETRY_BACKOFF_S) + 1):
            try:
                async with usage.slot():
                    return await self.client.chat(
                        model=self.model,
                        messages=messages,
                        think=self.think,
                        options=options,
                        **kwargs,
                    )
            except Exception as e:
                if _status(e) not in RETRY_STATUS or attempt == len(RETRY_BACKOFF_S):
                    raise
                await asyncio.sleep(RETRY_BACKOFF_S[attempt] + random.random())

    async def _chat(self, name, messages, **kwargs):
        options = {"temperature": 0, "num_predict": MAX_OUTPUT_TOKENS}
        params = {**options, "think": self.think}
        meter = usage.current()
        start = time.perf_counter()
        with tracing.generation(name, self.model, messages, params) as gen:
            refusal = await meter.refusal() if meter else None
            if refusal:
                gen.update(level="WARNING", status_message=refusal)
                usage.observe(self.model, name, "budget", 0)
                raise LLMError(f"LLM call refused: {refusal}")
            try:
                response = await self._call(messages, options, kwargs)
            except Exception as e:  # network, auth, rate limit: all mean "no answer"
                outcome = "rate_limited" if _status(e) == 429 else "error"
                usage.observe(self.model, name, outcome, time.perf_counter() - start)
                gen.update(level="ERROR", status_message=type(e).__name__)
                raise LLMError(f"LLM call failed: {type(e).__name__}") from e
            message = response.message
            tokens_in = response.prompt_eval_count or 0
            tokens_out = response.eval_count or 0
            usage.observe(
                self.model,
                name,
                "ok",
                time.perf_counter() - start,
                tokens_in,
                tokens_out,
            )
            if meter:
                await meter.add(self.model, tokens_in, tokens_out)
            gen.update(
                output={
                    "content": message.content,
                    "tool_calls": [
                        {
                            "name": c.function.name,
                            "arguments": dict(c.function.arguments),
                        }
                        for c in message.tool_calls or []
                    ],
                },
                usage_details={"input": tokens_in, "output": tokens_out},
                # So Langfuse shows a cost without a price table of its own.
                **(
                    {"cost_details": {"total": c}}
                    if (c := usage.cost(self.model, tokens_in, tokens_out))
                    else {}
                ),
            )
        return message

    async def structured(self, system, user, schema, images=None):
        prompt = {"role": "user", "content": user}
        if images:
            prompt["images"] = list(images)  # JPEG bytes
        message = await self._chat(
            schema.__name__,
            [
                {"role": "system", "content": f"{system}\n\n{field_list(schema)}"},
                prompt,
            ],
            format=schema.model_json_schema(),
        )
        try:
            return schema.model_validate(json_object(message.content or ""))
        except ValueError as e:
            raise LLMError("LLM output didn't match the schema") from e

    async def chat(self, messages, tools):
        """Returns {"content": str, "tool_calls": [{"name", "arguments"}]}."""
        message = await self._chat("chat", messages, tools=tools)
        return {
            "content": message.content or "",
            "tool_calls": [
                {"name": c.function.name, "arguments": dict(c.function.arguments)}
                for c in message.tool_calls or []
            ],
        }
