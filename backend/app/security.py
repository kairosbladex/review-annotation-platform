from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError


PASSWORDS = PasswordHasher()


def hash_password(password: str) -> str:
    return PASSWORDS.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return PASSWORDS.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def issue_token(*, user_id: str, username: str, secret: str, expires_minutes: int) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    return jwt.encode(
        {"sub": user_id, "username": username, "exp": expires_at},
        secret,
        algorithm="HS256",
    )


def decode_token(token: str, secret: str) -> dict:
    return jwt.decode(token, secret, algorithms=["HS256"])
