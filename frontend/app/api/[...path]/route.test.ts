// @vitest-environment node
import { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GET, POST } from "./route";

const ctx = (...path: string[]) => ({ params: Promise.resolve({ path }) });

describe("/api proxy", () => {
  const fetchMock = vi.fn();
  afterEach(() => {
    fetchMock.mockReset();
    vi.unstubAllGlobals();
  });

  it("forwards allowed headers only, and copies every Set-Cookie back", async () => {
    const headers = new Headers({ "content-type": "application/json", "x-internal": "secret" });
    headers.append("set-cookie", "a=1; HttpOnly");
    headers.append("set-cookie", "b=2; HttpOnly");
    fetchMock.mockResolvedValue(new Response('{"ok":true}', { status: 200, headers }));
    vi.stubGlobal("fetch", fetchMock);

    const req = new NextRequest("http://localhost:3000/api/auth/login?x=1", {
      method: "POST",
      headers: {
        cookie: "claimlens_session=t",
        "content-type": "application/json",
        "x-forwarded-for": "6.6.6.6",
        "x-adjuster-key": "guess",
      },
      body: '{"username":"sam"}',
    });
    const res = await POST(req, ctx("auth", "login"));

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("http://localhost:8000/auth/login?x=1");
    expect([...(init.headers as Headers).keys()].sort()).toEqual(["content-type", "cookie"]);
    expect(init.redirect).toBe("manual");
    expect(res.status).toBe(200);
    expect(res.headers.getSetCookie()).toEqual(["a=1; HttpOnly", "b=2; HttpOnly"]);
    expect(res.headers.get("x-internal")).toBeNull();
    expect(await res.json()).toEqual({ ok: true });
  });

  it("404s paths outside the gateway's public API without calling it", async () => {
    vi.stubGlobal("fetch", fetchMock);
    const res = await GET(new NextRequest("http://localhost:3000/api/docs"), ctx("docs"));
    expect(res.status).toBe(404);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("502s when the gateway is down", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));
    vi.stubGlobal("fetch", fetchMock);
    const res = await GET(new NextRequest("http://localhost:3000/api/claims"), ctx("claims"));
    expect(res.status).toBe(502);
  });

  it("passes 204 through with no body", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    const res = await POST(new NextRequest("http://localhost:3000/api/auth/logout", { method: "POST" }), ctx("auth", "logout"));
    expect(res.status).toBe(204);
    expect(await res.text()).toBe("");
  });
});
