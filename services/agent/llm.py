"""The LLM behind the agent: Ollama (Ollama Cloud by default, or a local Ollama).

Two calls are used:
- structured(): the answer must validate against a pydantic schema. Ollama
  constrains generation to the JSON schema (format=...), and we validate again
  anyway, since the model's output is untrusted.
- chat(): free chat with tool definitions, for the bounded clause-search loop.

Temperature 0 so the same claim gives the same extraction where possible.
"""

import json
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
# gpt-oss takes a reasoning level; "low" keeps each call to a few seconds.
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "low")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "120"))


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, host=OLLAMA_HOST, model=OLLAMA_MODEL, api_key=OLLAMA_API_KEY):
        from ollama import AsyncClient

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self.client = AsyncClient(host=host, headers=headers, timeout=LLM_TIMEOUT_S)
        self.model = model

    async def _chat(self, messages, **kwargs):
        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                think=OLLAMA_THINK or None,
                options={"temperature": 0},
                **kwargs,
            )
        except Exception as e:  # network, auth, rate limit: all mean "no answer"
            raise LLMError(f"LLM call failed: {type(e).__name__}") from e
        return response.message

    async def structured(self, system, user, schema):
        message = await self._chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            format=schema.model_json_schema(),
        )
        try:
            return schema.model_validate(json.loads(message.content))
        except ValueError as e:
            raise LLMError("LLM output didn't match the schema") from e

    async def chat(self, messages, tools):
        """Returns {"content": str, "tool_calls": [{"name", "arguments"}]}."""
        message = await self._chat(messages, tools=tools)
        return {
            "content": message.content or "",
            "tool_calls": [
                {"name": c.function.name, "arguments": dict(c.function.arguments)}
                for c in message.tool_calls or []
            ],
        }
