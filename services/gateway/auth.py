"""Logins, sessions and roles.

    POST /auth/login     {username, password} -> sets the session cookie
    POST /auth/logout    ends the session
    GET  /auth/me        the logged-in user

Two roles. A customer sees and files only their own claims, on their own
policies. An adjuster sees every claim and is the only role that can decide.

Sessions are server-side: the cookie holds a random token, and Redis holds
the session under the token's SHA-256 (a Redis dump doesn't hand out working
cookies). Compared with a signed JWT, logging out ends the session right away,
and there's no signing key to manage. Every gateway replica reads the same
Redis, so any replica can serve any request.

The cookie is HttpOnly (page scripts can't read it, so injected JavaScript
can't steal it) and SameSite=Lax (browsers don't send it on cross-site POSTs,
which is the CSRF defence: every state change here is a POST, and GETs never
change anything).
"""

import asyncio
import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from passwords import DUMMY_HASH, MAX_PASSWORD_LEN, verify_password

COOKIE = "claimlens_session"
SESSION_TTL_S = int(os.getenv("SESSION_TTL_S", str(8 * 3600)))
# Secure cookies are only sent over HTTPS. Local compose runs plain HTTP, so it
# turns this off explicitly; the default is on.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true").lower() != "false"
# Failed logins per username before a lockout, and the window both last.
LOGIN_FAILURES = int(os.getenv("LOGIN_FAILURES", "5"))
LOGIN_WINDOW_S = 15 * 60


@dataclass(frozen=True)
class User:
    username: str
    role: Literal["customer", "adjuster"]
    name: str
    customer_id: str | None = None


class UserStore:
    """Reads logins and policies from Postgres. psycopg is used synchronously
    in a worker thread, as in the MCP servers: its async mode doesn't run on
    Windows' default event loop, where the tests run."""

    def __init__(self, pool):
        self.pool = pool  # psycopg_pool.ConnectionPool

    def _find(self, username):
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT username, role, display_name, customer_id, password_hash"
                " FROM auth.users WHERE username = %s",
                (username,),
            ).fetchone()
        if not row:
            return None
        return User(row[0], row[1], row[2], row[3]), row[4]

    def _policies(self, customer_id):
        with self.pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT p.policy_id, p.status, p.start_date, p.end_date,
                       p.collision, p.comprehensive, v.year, v.make, v.model
                FROM policies p JOIN vehicles v USING (vehicle_id)
                WHERE p.customer_id = %s
                ORDER BY p.policy_id
                """,
                (customer_id,),
            ).fetchall()
        return [
            {
                "policy_id": r[0],
                "status": r[1],
                "start_date": r[2].isoformat(),
                "end_date": r[3].isoformat(),
                "collision": r[4],
                "comprehensive": r[5],
                "vehicle": f"{r[6]} {r[7]} {r[8]}",
            }
            for r in rows
        ]

    async def find(self, username):
        """(User, password hash) or None."""
        return await asyncio.to_thread(self._find, username)

    async def policies(self, customer_id):
        return await asyncio.to_thread(self._policies, customer_id)

    async def owns_policy(self, customer_id, policy_id):
        return any(
            p["policy_id"] == policy_id for p in await self.policies(customer_id)
        )


class Sessions:
    def __init__(self, redis):
        self.redis = redis  # redis.asyncio.Redis, decode_responses=True

    @staticmethod
    def _key(token):
        return "session:" + hashlib.sha256(token.encode()).hexdigest()

    async def create(self, user):
        token = secrets.token_urlsafe(32)
        await self.redis.set(
            self._key(token), json.dumps(asdict(user)), ex=SESSION_TTL_S
        )
        return token

    async def get(self, token):
        raw = await self.redis.get(self._key(token))
        return User(**json.loads(raw)) if raw else None

    async def delete(self, token):
        await self.redis.delete(self._key(token))


class RateLimiter:
    """Fixed-window counters in Redis, shared by every gateway replica.

    Per-account limits live here; per-IP limits belong at the Ingress
    (NGINX limit_req, Phase 6), which sees the real client address."""

    def __init__(self, redis):
        self.redis = redis

    async def hit(self, key, window_s):
        """Count one event; returns the count in the current window."""
        pipe = self.redis.pipeline()
        pipe.incr(f"rl:{key}")
        # NX: only the first hit sets the expiry, so the window doesn't slide.
        # In the same MULTI as the INCR, so a key can't be left without a TTL.
        pipe.expire(f"rl:{key}", window_s, nx=True)
        count, _ = await pipe.execute()
        return count

    async def count(self, key):
        return int(await self.redis.get(f"rl:{key}") or 0)

    async def retry_after(self, key):
        return max(1, await self.redis.ttl(f"rl:{key}"))

    async def reset(self, key):
        await self.redis.delete(f"rl:{key}")


# Dependencies, so tests can swap in fakes via app.dependency_overrides.
def user_store(request: Request) -> UserStore:
    return request.app.state.users


def sessions(request: Request) -> Sessions:
    return request.app.state.sessions


def rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.limiter


async def current_user(
    request: Request, store: Annotated[Sessions, Depends(sessions)]
) -> User:
    token = request.cookies.get(COOKIE)
    user = await store.get(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="Log in first")
    return user


def require_role(role):
    async def check(user: Annotated[User, Depends(current_user)]) -> User:
        if user.role != role:
            raise HTTPException(status_code=403, detail=f"Only for {role}s")
        return user

    return check


CurrentUser = Annotated[User, Depends(current_user)]
Customer = Annotated[User, Depends(require_role("customer"))]
Adjuster = Annotated[User, Depends(require_role("adjuster"))]


class Login(BaseModel):
    username: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LEN)


router = APIRouter(prefix="/auth")


@router.post("/login")
async def login(
    body: Login,
    response: Response,
    users: Annotated[UserStore, Depends(user_store)],
    store: Annotated[Sessions, Depends(sessions)],
    limiter: Annotated[RateLimiter, Depends(rate_limiter)],
):
    username = body.username.strip().lower()
    key = f"login:{username}"
    # Locks the account for the rest of the window after too many failures.
    # The trade-off: anyone can lock an account for 15 minutes by failing on
    # purpose. That's the usual price for stopping password guessing.
    if await limiter.count(key) >= LOGIN_FAILURES:
        raise HTTPException(
            status_code=429,
            detail="Too many failed logins. Try again later.",
            headers={"Retry-After": str(await limiter.retry_after(key))},
        )
    found = await users.find(username)
    # Hash checked even for unknown users, so timing doesn't reveal who exists.
    stored = found[1] if found else DUMMY_HASH
    ok = await asyncio.to_thread(verify_password, body.password, stored)
    if not (found and ok):
        await limiter.hit(key, LOGIN_WINDOW_S)
        raise HTTPException(status_code=401, detail="Wrong username or password")
    await limiter.reset(key)
    user = found[0]
    response.set_cookie(
        COOKIE,
        await store.create(user),
        max_age=SESSION_TTL_S,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    return asdict(user)


@router.post("/logout", status_code=204)
async def logout(
    request: Request,
    response: Response,
    store: Annotated[Sessions, Depends(sessions)],
):
    token = request.cookies.get(COOKIE)
    if token:
        await store.delete(token)
    response.delete_cookie(COOKIE, path="/")


@router.get("/me")
async def me(user: CurrentUser):
    return asdict(user)
