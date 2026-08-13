"""Authentication: password hashing and access tokens (spec section 44)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from gauntlet.config import get_settings
from gauntlet.db.models import Candidate, User

ALGORITHM = "HS256"
# bcrypt silently truncates at 72 bytes; rejecting longer input is safer than letting
# two different passwords authenticate the same account.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 10


class AuthError(Exception):
    """Authentication or registration failed."""


@dataclass(frozen=True, slots=True)
class TokenPair:
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 0


def hash_password(password: str) -> str:
    validate_password(password)
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise AuthError("Password must be at most 72 bytes.")


def create_access_token(user_id: uuid.UUID, candidate_id: uuid.UUID | None) -> TokenPair:
    settings = get_settings()
    ttl = timedelta(minutes=settings.access_token_ttl_minutes)
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "cid": str(candidate_id) if candidate_id else None,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)
    return TokenPair(access_token=token, expires_in=int(ttl.total_seconds()))


def decode_access_token(token: str) -> dict[str, str]:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise AuthError("Invalid or expired token.") from exc


def register_user(session: Session, email: str, password: str, display_name: str) -> Candidate:
    normalised = email.strip().lower()
    if not normalised or "@" not in normalised:
        raise AuthError("A valid email address is required.")

    existing = session.scalar(select(User).where(User.email == normalised))
    if existing is not None:
        raise AuthError("An account with that email already exists.")

    user = User(email=normalised, password_hash=hash_password(password))
    candidate = Candidate(user=user, display_name=display_name.strip() or normalised.split("@")[0])
    session.add(user)
    session.add(candidate)
    session.flush()
    return candidate


def authenticate(session: Session, email: str, password: str) -> Candidate:
    user = session.scalar(select(User).where(User.email == email.strip().lower()))
    # Verify against a dummy hash when the user is missing so that a wrong email and a
    # wrong password take the same time to fail.
    if user is None:
        bcrypt.checkpw(b"timing-equalisation", bcrypt.hashpw(b"x", bcrypt.gensalt()))
        raise AuthError("Incorrect email or password.")
    if not user.is_active or not verify_password(password, user.password_hash):
        raise AuthError("Incorrect email or password.")

    candidate = session.scalar(select(Candidate).where(Candidate.user_id == user.id))
    if candidate is None:
        candidate = Candidate(user_id=user.id, display_name=user.email.split("@")[0])
        session.add(candidate)
        session.flush()
    return candidate
