"""Prometheus metrics for a FastAPI service: request count and latency per
route, served at GET /metrics for Prometheus to scrape.

The same file is in gateway, agent and guardrails (each service builds from
its own folder). Labels use the route template ("/claims/{claim_id}"), never
the raw path, so random URLs can't create unbounded time series. /metrics
isn't on the frontend's /api allowlist, so it's reachable inside the cluster
only.
"""

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

REQUESTS = Counter(
    "http_requests_total", "HTTP requests handled", ["method", "route", "status"]
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "Time to handle an HTTP request",
    ["method", "route"],
    # Up to a minute: /detect waits for a vision worker, claims call an LLM.
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)


class _Middleware:
    """Plain ASGI, not BaseHTTPMiddleware, so streamed bodies pass untouched."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start, status = time.perf_counter(), 500

        async def send_status(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_status)
        finally:
            # The router puts the matched route into this same scope dict.
            route = getattr(scope.get("route"), "path", "unmatched")
            if route != "/metrics":
                method = scope["method"]
                REQUESTS.labels(method, route, str(status)).inc()
                LATENCY.labels(method, route).observe(time.perf_counter() - start)


def _metrics(request: Request):
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def instrument(app):
    app.add_middleware(_Middleware)
    app.add_route("/metrics", _metrics, include_in_schema=False)
