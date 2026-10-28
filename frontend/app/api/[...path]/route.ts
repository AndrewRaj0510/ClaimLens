// /api/* -> the gateway. A route handler rather than next.config rewrites,
// because standalone builds freeze rewrites at build time: here GATEWAY_URL is
// read at runtime, so one image works in Compose and Kubernetes. It also lets
// us choose exactly which paths and headers get through (lib/forward.ts).
//
// Request bodies (photo uploads) are streamed, not buffered.

import type { NextRequest } from "next/server";

import { REQUEST_HEADERS, RESPONSE_HEADERS, pickHeaders, upstreamUrl } from "@/lib/forward";

export const dynamic = "force-dynamic";

const GATEWAY_URL = process.env.GATEWAY_URL ?? "http://localhost:8000";

async function forward(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const url = upstreamUrl(path, req.nextUrl.search, GATEWAY_URL);
  if (!url) return Response.json({ detail: "Not found" }, { status: 404 });

  const hasBody = req.method !== "GET" && req.method !== "HEAD";
  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: req.method,
      headers: pickHeaders(req.headers, REQUEST_HEADERS),
      body: hasBody ? req.body : undefined,
      // Required by Node's fetch to stream a request body.
      ...(hasBody ? { duplex: "half" } : {}),
      redirect: "manual",
      cache: "no-store",
    } as RequestInit);
  } catch {
    return Response.json({ detail: "Gateway unavailable" }, { status: 502 });
  }

  const headers = pickHeaders(upstream.headers, RESPONSE_HEADERS);
  // Several Set-Cookie headers can't be joined into one, so copy each.
  for (const cookie of upstream.headers.getSetCookie()) headers.append("set-cookie", cookie);
  const empty = upstream.status === 204 || upstream.status === 304;
  return new Response(empty ? null : upstream.body, { status: upstream.status, headers });
}

export { forward as DELETE, forward as GET, forward as POST, forward as PUT };
