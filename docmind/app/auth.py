"""API-key auth. Keys are random and high-entropy, so a plain SHA-256 hash is enough to store
(unlike passwords, they can't be guessed from a dictionary). The key is shown only once."""
import hashlib
import secrets

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import User, get_db

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)   # gives the "Authorize" button in /docs


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_api_key() -> str:
    return "dm_" + secrets.token_urlsafe(24)


def current_user(key: str | None = Security(api_key_header), db: Session = Depends(get_db)) -> User:
    if not key:
        raise HTTPException(401, "Missing X-API-Key header")
    user = db.scalar(select(User).where(User.api_key_hash == hash_key(key)))
    if not user:
        raise HTTPException(401, "Invalid API key")
    return user
