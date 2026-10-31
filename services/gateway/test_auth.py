import asyncio
import os

import pytest
from fastapi.testclient import TestClient

import auth
import fakes
from auth import COOKIE, Sessions, User, rate_limiter, sessions, user_store
from main import app
from passwords import hash_password, verify_password


@pytest.fixture
def env():
    users, store, limiter = fakes.FakeUsers(), fakes.FakeSessions(), fakes.FakeLimiter()
    app.dependency_overrides[user_store] = lambda: users
    app.dependency_overrides[sessions] = lambda: store
    app.dependency_overrides[rate_limiter] = lambda: limiter
    # https, because the cookie is Secure and wouldn't be sent back over http.
    with_client = TestClient(app, base_url="https://testserver")
    yield with_client, users, store, limiter
    app.dependency_overrides.clear()


def login(client, username=fakes.CUSTOMER.username, password=fakes.PASSWORD):
    return client.post("/auth/login", json={"username": username, "password": password})


def test_password_hash_round_trip():
    stored = hash_password("hunter2")
    assert stored.startswith("scrypt$16384$8$5$")
    assert verify_password("hunter2", stored)
    assert not verify_password("hunter3", stored)
    assert hash_password("hunter2") != stored  # salted
    assert not verify_password("x", "not-a-hash")
    assert not verify_password("x", "bcrypt$1$2$3$4$5")


def test_login_sets_a_hardened_cookie_and_me_works(env):
    client, _, store, _ = env
    r = login(client)
    assert r.status_code == 200
    assert r.json() == {
        "username": "daniel.brown1@example.com",
        "role": "customer",
        "name": "Daniel Brown",
        "customer_id": "C00001",
    }
    cookie = r.headers["set-cookie"].lower()
    for attribute in ("httponly", "secure", "samesite=lax", "path=/", "max-age=28800"):
        assert attribute in cookie
    assert len(store.sessions) == 1
    assert client.get("/auth/me").json()["customer_id"] == "C00001"


def test_username_is_case_and_space_insensitive(env):
    client, users, _, _ = env
    assert login(client, "  Daniel.Brown1@Example.com ").status_code == 200
    assert users.lookups == ["daniel.brown1@example.com"]


@pytest.mark.parametrize(
    ("username", "password"),
    [("daniel.brown1@example.com", "wrong"), ("nobody@example.com", fakes.PASSWORD)],
)
def test_bad_logins_get_the_same_answer(env, username, password):
    client, _, store, _ = env
    r = login(client, username, password)
    assert r.status_code == 401
    assert r.json()["detail"] == "Wrong username or password"  # doesn't say which
    assert "set-cookie" not in r.headers
    assert store.sessions == {}


def test_unknown_user_still_checks_a_hash(env, monkeypatch):
    client, _, _, _ = env
    checked = []
    real = auth.verify_password

    def spy(password, stored):
        checked.append(stored)
        return real(password, stored)

    monkeypatch.setattr(auth, "verify_password", spy)
    login(client, "nobody@example.com")
    assert checked == [auth.DUMMY_HASH]  # same work as a wrong password


def test_repeated_failures_lock_the_account(env):
    client, _, _, _ = env
    for _ in range(auth.LOGIN_FAILURES):
        assert login(client, password="wrong").status_code == 401
    r = login(client)  # even the right password
    assert r.status_code == 429
    assert r.headers["retry-after"] == "60"
    # Other accounts aren't affected.
    assert login(client, fakes.ADJUSTER.username).status_code == 200


def test_success_resets_the_failure_count(env):
    client, _, _, limiter = env
    login(client, password="wrong")
    assert login(client).status_code == 200
    assert limiter.counts == {}


def test_logout_ends_the_session_server_side(env):
    client, _, store, _ = env
    login(client)
    token = client.cookies[COOKIE]
    assert client.post("/auth/logout").status_code == 204
    assert store.sessions == {}
    # Replaying the old cookie doesn't work: the session is gone, not just the cookie.
    client.cookies.set(COOKIE, token)
    assert client.get("/auth/me").status_code == 401


def test_no_or_forged_cookie_is_401(env):
    client, _, _, _ = env
    assert client.get("/auth/me").status_code == 401
    client.cookies.set(COOKIE, "made-up-token")
    assert client.get("/auth/me").status_code == 401


def test_overlong_inputs_are_rejected(env):
    client, users, _, _ = env
    assert login(client, password="x" * 257).status_code == 422
    assert login(client, "x" * 255).status_code == 422
    assert users.lookups == []


# ---- real Redis and Postgres ----

TEST_REDIS = os.getenv("TEST_REDIS_URL")
TEST_DB = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_REDIS, reason="set TEST_REDIS_URL to run")
def test_sessions_and_rate_limiter_on_redis():
    import redis.asyncio as redis

    async def scenario():
        r = redis.Redis.from_url(TEST_REDIS, decode_responses=True)
        await r.flushdb()
        store, limiter = Sessions(r), auth.RateLimiter(r)
        user = User("sam", "adjuster", "Sam")
        token = await store.create(user)
        keys = await r.keys("session:*")
        assert token not in keys[0]  # stored under a hash of the token
        assert 0 < await r.ttl(keys[0]) <= auth.SESSION_TTL_S
        assert await store.get(token) == user
        await store.delete(token)
        assert await store.get(token) is None

        assert [await limiter.hit("k", 30) for _ in range(3)] == [1, 2, 3]
        ttl = await r.ttl("rl:k")
        assert 0 < ttl <= 30  # set once, by the first hit
        assert await limiter.count("k") == 3
        await limiter.reset("k")
        assert await limiter.count("k") == 0
        await r.aclose()

    asyncio.run(scenario())


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_user_store_sql():
    from pathlib import Path

    import psycopg
    from psycopg_pool import ConnectionPool

    schema = Path(__file__).resolve().parents[2] / "db" / "schema.sql"
    with psycopg.connect(TEST_DB, autocommit=True) as conn:
        conn.execute(schema.read_text())
        conn.execute("TRUNCATE customers CASCADE")
        conn.execute("TRUNCATE auth.users")
        for cid in ("C00001", "C00002"):
            conn.execute(
                "INSERT INTO customers VALUES (%s,'A','B','1980-01-01','ME','a@example.com',"
                "'(207) 555-0100','2020-01-01')",
                (cid,),
            )
            conn.execute(
                "INSERT INTO vehicles VALUES (%s,%s,%s,'Honda','Civic',2020,'economy')",
                ("V" + cid[1:], cid, "VIN" + cid),
            )
            conn.execute(
                "INSERT INTO policies VALUES (%s,%s,%s,'w1','ME','2020-01-01','2026-12-31',"
                "'active',NULL,true,true,true,false,false,500,250,1500)",
                ("P" + cid[1:], cid, "V" + cid[1:]),
            )
        conn.execute(
            "INSERT INTO auth.users VALUES ('a@example.com', 'h', 'customer', 'A B', 'C00001')"
        )
        # A customer login must have a customer, an adjuster must not.
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                "INSERT INTO auth.users VALUES ('x', 'h', 'customer', 'X', NULL)"
            )

    pool = ConnectionPool(TEST_DB, min_size=1, max_size=2, open=True)
    store = auth.UserStore(pool)

    async def scenario():
        found = await store.find("a@example.com")
        assert found == (User("a@example.com", "customer", "A B", "C00001"), "h")
        assert await store.find("nobody") is None
        policies = await store.policies("C00001")
        assert policies[0]["policy_id"] == "P00001"
        assert policies[0]["vehicle"] == "2020 Honda Civic"
        assert await store.owns_policy("C00001", "P00001")
        assert not await store.owns_policy("C00001", "P00002")

    try:
        asyncio.run(scenario())
    finally:
        pool.close()


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_pool_survives_a_database_restart():
    """Every pooled connection dies when Postgres restarts. The pool must
    replace them, not hand them out (each one failed a login on kind)."""
    import psycopg

    from main import db_pool

    pool = db_pool(TEST_DB)
    pool.open(wait=True)
    try:
        with pool.connection() as conn:
            pid = conn.info.backend_pid
        # What a restart does to it: the server ends the connection.
        with psycopg.connect(TEST_DB, autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
        with pool.connection() as conn:
            assert conn.execute("SELECT 1").fetchone() == (1,)
            assert conn.info.backend_pid != pid
    finally:
        pool.close()
