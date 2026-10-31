"""Run state shared by every agent replica, in Redis.

The graph's state is in the Postgres checkpointer, so any replica can read
a claim. Three things aren't in a checkpoint and used to live in one
process's memory, which broke as soon as there were two replicas (a status
request on the other pod saw a running claim as "interrupted"):

    agent:run:<claim>      who is running it: a lease, SET NX with a TTL
    agent:failure:<claim>  why its last run failed
    agent:owner:<claim>    who filed it, until its first checkpoint exists

The lease is how one claim never runs twice: only one replica can SET NX it,
and the run renews it every LEASE_S / 3. If the pod dies (OOM kill, node
loss), the lease expires within LEASE_S and the claim shows as interrupted,
which POST /retry continues from the last checkpoint.
"""

import os
import uuid

LEASE_S = int(os.getenv("RUN_LEASE_S", "30"))
# A failed claim stays "failed" until retried. After this it would show as
# "interrupted" instead, which is still retryable; the TTL only stops keys
# piling up forever.
FAILURE_TTL_S = 30 * 24 * 3600
# Only needed until the run's first checkpoint, which takes a second or two.
OWNER_TTL_S = 24 * 3600

# Renew only our own lease; if it expired (e.g. a Redis blip longer than the
# TTL) and nobody else took it, take it back. 0 = someone else holds it.
_RENEW = """
local current = redis.call('GET', KEYS[1])
if current == ARGV[1] then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
  return 1
elseif not current then
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return 1
end
return 0
"""
# Release only our own lease, never one another replica took over.
_RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


def _run(claim_id):
    return f"agent:run:{claim_id}"


def _failure(claim_id):
    return f"agent:failure:{claim_id}"


def _owner(claim_id):
    return f"agent:owner:{claim_id}"


class RunState:
    def __init__(self, redis, lease_s=LEASE_S):
        self.r = redis  # redis.asyncio client with decode_responses=True
        self.lease_s = lease_s

    async def acquire(self, claim_id):
        """A lease token, or None if the claim is running somewhere already."""
        token = uuid.uuid4().hex
        ok = await self.r.set(_run(claim_id), token, nx=True, ex=self.lease_s)
        return token if ok else None

    async def renew(self, claim_id, token):
        """False if another replica holds the lease now."""
        return bool(await self.r.eval(_RENEW, 1, _run(claim_id), token, self.lease_s))

    async def release(self, claim_id, token):
        await self.r.eval(_RELEASE, 1, _run(claim_id), token)

    async def running(self, claim_ids):
        """The subset of claim_ids with a live lease (one round trip)."""
        claim_ids = list(claim_ids)
        if not claim_ids:
            return set()
        values = await self.r.mget([_run(c) for c in claim_ids])
        return {c for c, v in zip(claim_ids, values, strict=True) if v is not None}

    async def set_failure(self, claim_id, message):
        await self.r.set(_failure(claim_id), message, ex=FAILURE_TTL_S)

    async def failure(self, claim_id):
        return await self.r.get(_failure(claim_id))

    async def clear_failure(self, claim_id):
        await self.r.delete(_failure(claim_id))

    async def set_owner(self, claim_id, customer_id):
        await self.r.set(_owner(claim_id), customer_id, ex=OWNER_TTL_S)

    async def owner(self, claim_id):
        return await self.r.get(_owner(claim_id))

    async def clear_owner(self, claim_id):
        await self.r.delete(_owner(claim_id))
