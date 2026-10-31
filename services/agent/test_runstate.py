"""RunState against a real Redis: the Lua scripts and TTLs can't be faked.
Uses database 15 and flushes it: point TEST_REDIS_URL at a Redis you don't
mind emptying, e.g. redis://:<pw>@127.0.0.1:6380/15 (CI sets it)."""

import asyncio
import os

import pytest

from runstate import RunState

TEST_REDIS = os.getenv("TEST_REDIS_URL")
pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not TEST_REDIS, reason="set TEST_REDIS_URL to run"),
]
CLAIM = "a" * 32


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def fresh(lease_s=30):
    import redis.asyncio as redis

    r = redis.Redis.from_url(TEST_REDIS, decode_responses=True)
    await r.flushdb()
    return r, RunState(r, lease_s=lease_s)


async def test_only_one_replica_gets_the_lease():
    r, rs = await fresh()
    try:
        token = await rs.acquire(CLAIM)
        assert token and await rs.acquire(CLAIM) is None
        assert await rs.running([CLAIM, "b" * 32]) == {CLAIM}
        assert await rs.running([]) == set()
        # Someone else's token neither renews nor releases it.
        assert await rs.renew(CLAIM, "not-mine") is False
        await rs.release(CLAIM, "not-mine")
        assert await rs.running([CLAIM]) == {CLAIM}
        assert await rs.renew(CLAIM, token) is True
        await rs.release(CLAIM, token)
        assert await rs.running([CLAIM]) == set()
    finally:
        await r.aclose()


async def test_a_lease_nobody_renews_expires_and_can_be_taken_back():
    r, rs = await fresh(lease_s=1)
    try:
        token = await rs.acquire(CLAIM)
        await asyncio.sleep(1.2)  # the pod died: nobody renewed
        assert await rs.running([CLAIM]) == set()
        # The run was only cut off from Redis: renewing takes it back, as long
        # as nobody else started the claim in between.
        assert await rs.renew(CLAIM, token) is True
        assert await r.ttl(f"agent:run:{CLAIM}") > 0
        await rs.release(CLAIM, token)
        other = await rs.acquire(CLAIM)
        assert await rs.renew(CLAIM, token) is False
        await rs.release(CLAIM, other)
    finally:
        await r.aclose()


async def test_failures_and_owners():
    r, rs = await fresh()
    try:
        assert await rs.failure(CLAIM) is None
        await rs.set_failure(CLAIM, "RuntimeError: down")
        assert await rs.failure(CLAIM) == "RuntimeError: down"
        assert await r.ttl(f"agent:failure:{CLAIM}") > 0
        await rs.clear_failure(CLAIM)
        assert await rs.failure(CLAIM) is None

        await rs.set_owner(CLAIM, "C00001")
        assert await rs.owner(CLAIM) == "C00001"
        await rs.clear_owner(CLAIM)
        assert await rs.owner(CLAIM) is None
    finally:
        await r.aclose()
