// Next.js 16 "proxy" (what used to be middleware): runs before a page renders.
// Without a session cookie, send people to the login page straight away
// instead of flashing an empty page first. It only checks the cookie exists;
// whether the session is valid, and for which role, is the gateway's job.

import { NextResponse, type NextRequest } from "next/server";

export function proxy(request: NextRequest) {
  if (!request.cookies.has("claimlens_session")) {
    const login = new URL("/login", request.url);
    login.searchParams.set("next", request.nextUrl.pathname);
    return NextResponse.redirect(login);
  }
  return NextResponse.next();
}

export const config = { matcher: ["/claims/:path*", "/adjuster/:path*"] };
