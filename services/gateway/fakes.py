"""In-memory stand-ins for the gateway's Postgres and Redis pieces, for tests.
The real classes are tested against real Postgres and Redis in test_auth.py."""

import secrets

from auth import User
from passwords import hash_password

CUSTOMER = User("daniel.brown1@example.com", "customer", "Daniel Brown", "C00001")
OTHER_CUSTOMER = User("ann.lee2@example.com", "customer", "Ann Lee", "C00002")
ADJUSTER = User("sam", "adjuster", "Sam Adjuster")
PASSWORD = "correct horse battery staple"
# Hashed once per test session: scrypt is slow on purpose.
_HASH = hash_password(PASSWORD)


class FakeUsers:
    def __init__(self):
        self.users = {u.username: u for u in (CUSTOMER, OTHER_CUSTOMER, ADJUSTER)}
        self.lookups = []
        self.policy_rows = {
            "C00001": [
                {
                    "policy_id": "P00001",
                    "status": "active",
                    "start_date": "2025-09-21",
                    "end_date": "2027-09-21",
                    "collision": True,
                    "comprehensive": True,
                    "vehicle": "2020 Honda Civic",
                }
            ],
            "C00002": [],
        }

    async def find(self, username):
        self.lookups.append(username)
        user = self.users.get(username)
        return (user, _HASH) if user else None

    async def policies(self, customer_id):
        return self.policy_rows.get(customer_id, [])

    async def owns_policy(self, customer_id, policy_id):
        return any(
            p["policy_id"] == policy_id for p in await self.policies(customer_id)
        )


class FakeSessions:
    def __init__(self):
        self.sessions = {}

    async def create(self, user):
        token = secrets.token_urlsafe(16)
        self.sessions[token] = user
        return token

    async def get(self, token):
        return self.sessions.get(token)

    async def delete(self, token):
        self.sessions.pop(token, None)


class FakeLimiter:
    def __init__(self):
        self.counts = {}

    async def hit(self, key, window_s):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def count(self, key):
        return self.counts.get(key, 0)

    async def retry_after(self, key):
        return 60

    async def reset(self, key):
        self.counts.pop(key, None)
