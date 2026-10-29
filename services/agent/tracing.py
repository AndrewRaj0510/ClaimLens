"""Langfuse tracing for agent runs. Optional: off unless keys are set.

On when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set; LANGFUSE_BASE_URL
picks the server (Langfuse Cloud by default, or the self-hosted one in
docker-compose.tracing.yml). Off, every function here is a no-op, so tests and
keyless setups run unchanged and the SDK isn't even imported.

One trace per claim. The trace ID is derived from the claim ID, so the first
run, a retake, the adjuster's decision and a retry (separate runs, possibly days
apart and on another replica) all land in the same trace. Inside it:
- a span per run ("claim start", "claim approval", ...)
- a span per graph node, with the node's state update as its output
- a generation per LLM call, nested in its node: model, prompt, output,
  token counts, latency

Node spans are opened by graph.py around each node function, not by Langfuse's
LangChain callback handler: the handler records node spans but doesn't make
them current, so LLM calls inside a node ended up under the run instead
(seen while writing test_tracing.py,
which now checks the nesting).

Photo bytes are never sent: the vision call's input says how many photos were
attached instead. Tracing must never break a claim, so SDK errors are logged
and the run goes on (the SDK also exports in a background thread).
"""

import logging
import os
from contextlib import ExitStack, contextmanager

log = logging.getLogger("agent.tracing")

ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

_client = None


def client():
    """The Langfuse client, or None when tracing is off (or failed to start)."""
    global _client, ENABLED
    if ENABLED and _client is None:
        try:
            from langfuse import Langfuse

            _client = Langfuse()
        except Exception:
            log.exception("Langfuse failed to start; tracing off")
            ENABLED = False
    return _client


def without_images(messages):
    """A copy of the chat messages that's safe to log: photos replaced by a count."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("images"):
            m["images"] = f"[{len(m['images'])} redacted photo(s), not logged]"
        out.append(m)
    return out


class _Safe:
    """A span whose update() never raises (or a no-op one, span=None)."""

    def __init__(self, span=None):
        self.span = span

    def update(self, **kwargs):
        if self.span is None:
            return
        try:
            self.span.update(**kwargs)
        except Exception:
            log.exception("tracing: update failed")


@contextmanager
def _observe(**kwargs):
    """A Langfuse observation as the current span. If the SDK fails to open it,
    the block still runs, untraced: tracing must never fail a claim."""
    lf = client()
    if lf is None:
        yield _Safe()
        return
    with ExitStack() as stack:
        try:
            span = stack.enter_context(lf.start_as_current_observation(**kwargs))
        except Exception:
            log.exception("tracing: couldn't open %s", kwargs.get("name"))
            span = None
        yield _Safe(span)


@contextmanager
def claim_run(claim_id, kind, customer_id=None):
    """Wrap one graph run (start, retake, approval or retry)."""
    lf = client()
    if lf is None:
        yield
        return
    with ExitStack() as stack:
        try:
            from langfuse import propagate_attributes

            stack.enter_context(
                lf.start_as_current_observation(
                    trace_context={"trace_id": lf.create_trace_id(seed=claim_id)},
                    name=f"claim {kind}",
                    as_type="agent",
                    metadata={"claim_id": claim_id},
                )
            )
            # session_id groups the claim's runs in the UI too; user_id is the
            # customer ID (an opaque ID, not a name).
            stack.enter_context(
                propagate_attributes(
                    trace_name="claim",
                    session_id=claim_id,
                    user_id=customer_id,
                    tags=[kind],
                )
            )
        except Exception:
            log.exception("tracing: couldn't open the run span for %s", claim_id)
        yield


def node(name):
    """Wrap one graph node. Yields an object with .update(output=...)."""
    return _observe(name=name, as_type="span")


def generation(name, model, messages, parameters):
    """Wrap one LLM call. Yields an object with .update(output=..., usage_details=...)."""
    return _observe(
        name=name,
        as_type="generation",
        model=model,
        input=without_images(messages),
        model_parameters=parameters,
    )


def flush():
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            log.exception("tracing: flush failed")
