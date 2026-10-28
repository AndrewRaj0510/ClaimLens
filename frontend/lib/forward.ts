// Pure helpers for the /api route (app/api/[...path]/route.ts), kept here so
// they're unit-testable without a Next.js server.

// Only the gateway's public API is reachable through the proxy: not /docs,
// /openapi.json, /health or anything added to the gateway later by accident.
const ALLOWED_ROOTS = new Set(["auth", "me", "claims", "detect", "jobs"]);

// Request headers passed on. Everything else (X-Forwarded-*, Host, custom
// headers) is dropped, so a browser can't smuggle headers to the gateway.
export const REQUEST_HEADERS = ["accept", "content-type", "cookie"];
export const RESPONSE_HEADERS = ["cache-control", "content-type", "retry-after", "x-content-type-options"];

/** The gateway URL for these path segments, or null if it isn't allowed. */
export function upstreamUrl(segments: string[], search: string, base: string): URL | null {
  if (!segments.length || !ALLOWED_ROOTS.has(segments[0])) return null;
  // Segments arrive decoded: refuse anything that could change the path's shape.
  if (segments.some((s) => !s || s === "." || s === ".." || /[/\\?#%]/.test(s))) return null;
  const url = new URL(`/${segments.join("/")}`, base);
  url.search = search;
  return url;
}

export function pickHeaders(from: Headers, names: string[]): Headers {
  const out = new Headers();
  for (const name of names) {
    const value = from.get(name);
    if (value !== null) out.set(name, value);
  }
  return out;
}
