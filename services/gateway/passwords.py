"""Password hashing with scrypt from the standard library.

Stored format: scrypt$<n>$<r>$<p>$<salt b64>$<hash b64>. The parameters are
stored with each hash so they can be raised later without breaking old ones.

scrypt is memory-hard: each guess needs n * r * 128 bytes (16 MiB here), which
makes GPU cracking expensive. n=2^14, r=8, p=5 is one of OWASP's recommended
scrypt settings; measured at about 0.2 s per login, on a laptop and in the
gateway container. Logins run it in a worker thread, so the event loop isn't
blocked meanwhile.

Stdlib only, so scripts/create_users.py imports this exact module instead of
keeping a copy that could drift.
"""

import base64
import hashlib
import hmac
import secrets

N, R, P = 2**14, 8, 5
KEY_LEN = 32
# Long inputs are cheap for scrypt, but there's no reason to accept them.
MAX_PASSWORD_LEN = 256


def _scrypt(password, salt, n, r, p):
    return hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=KEY_LEN,
        # Room for n * r * 128 bytes; the default 32 MiB is too tight for larger n.
        maxmem=2 * n * r * 128 + 1024 * 1024,
    )


def hash_password(password):
    salt = secrets.token_bytes(16)
    key = _scrypt(password, salt, N, R, P)
    b64 = base64.b64encode
    return f"scrypt${N}${R}${P}${b64(salt).decode()}${b64(key).decode()}"


def verify_password(password, stored):
    try:
        scheme, n, r, p, salt, key = stored.split("$")
        if scheme != "scrypt":
            return False
        actual = _scrypt(password, base64.b64decode(salt), int(n), int(r), int(p))
    except ValueError:
        return False
    return hmac.compare_digest(actual, base64.b64decode(key))


# Checked when the username doesn't exist, so a login for an unknown user takes
# as long as one with a wrong password: response times don't reveal which
# usernames exist.
DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
