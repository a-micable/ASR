"""API key authentication for the Whisper ASR service."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_API_KEY_ENV = "WHISPER_API_KEY"


@dataclass
class APIKeyRecord:
    """Stored API key metadata."""

    key_hash: str
    label: str
    created_at: float = field(default_factory=time.time)
    last_used_at: float | None = None
    request_count: int = 0
    revoked: bool = False
    scopes: list[str] = field(default_factory=lambda: ["transcribe", "health"])


class APIKeyStore:
    """
    In-memory API key store with hash-based verification.

    For production, replace with a database-backed store (e.g. SQLAlchemy).
    Keys are stored as HMAC-SHA256 hashes — the plaintext is never stored.
    """

    def __init__(self, master_secret: str | None = None) -> None:
        """
        Initialize key store.

        Args:
            master_secret: Secret for HMAC hashing; falls back to env var
                           WHISPER_MASTER_SECRET or a random value.
        """
        self._secret = (
            master_secret
            or os.getenv("WHISPER_MASTER_SECRET")
            or secrets.token_hex(32)
        ).encode()
        self._keys: dict[str, APIKeyRecord] = {}

    def _hash_key(self, plaintext: str) -> str:
        """Compute deterministic HMAC-SHA256 of a key."""
        return hmac.new(self._secret, plaintext.encode(), hashlib.sha256).hexdigest()

    def create_key(self, label: str, scopes: list[str] | None = None) -> str:
        """
        Generate and register a new API key.

        Args:
            label: Human-readable label for the key.
            scopes: Allowed operation scopes.

        Returns:
            Plaintext API key (shown once; not stored).
        """
        plaintext = f"wsk_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(plaintext)
        self._keys[key_hash] = APIKeyRecord(
            key_hash=key_hash,
            label=label,
            scopes=scopes or ["transcribe", "health"],
        )
        logger.info("Created API key: label=%s scopes=%s", label, scopes)
        return plaintext

    def verify_key(self, plaintext: str, required_scope: str | None = None) -> APIKeyRecord:
        """
        Verify an API key and return its record.

        Args:
            plaintext: Key submitted by client.
            required_scope: If given, key must have this scope.

        Returns:
            APIKeyRecord.

        Raises:
            ValueError: If key is invalid, revoked, or missing required scope.
        """
        key_hash = self._hash_key(plaintext)
        record = self._keys.get(key_hash)
        if record is None:
            raise ValueError("Invalid API key")
        if record.revoked:
            raise ValueError("API key has been revoked")
        if required_scope and required_scope not in record.scopes:
            raise ValueError(f"API key lacks required scope: {required_scope}")

        record.last_used_at = time.time()
        record.request_count += 1
        return record

    def revoke_key(self, plaintext: str) -> bool:
        """
        Revoke an API key.

        Args:
            plaintext: Key to revoke.

        Returns:
            True if revoked, False if not found.
        """
        key_hash = self._hash_key(plaintext)
        if key_hash in self._keys:
            self._keys[key_hash].revoked = True
            logger.info("Revoked API key hash=%s", key_hash[:8])
            return True
        return False

    def list_keys(self) -> list[dict]:
        """Return sanitized list of registered keys (no hashes)."""
        return [
            {
                "label": r.label,
                "created_at": r.created_at,
                "last_used_at": r.last_used_at,
                "request_count": r.request_count,
                "revoked": r.revoked,
                "scopes": r.scopes,
            }
            for r in self._keys.values()
        ]


# Module-level singleton store populated from environment
_store: APIKeyStore | None = None


def get_key_store() -> APIKeyStore:
    """Return or initialize the module-level key store."""
    global _store
    if _store is None:
        _store = APIKeyStore()
        # Bootstrap a key from environment for zero-config deployments
        env_key = os.getenv(_API_KEY_ENV)
        if env_key:
            _store._keys[_store._hash_key(env_key)] = APIKeyRecord(
                key_hash=_store._hash_key(env_key),
                label="env_default",
                scopes=["transcribe", "health", "metrics"],
            )
            logger.info("Loaded API key from environment variable %s", _API_KEY_ENV)
    return _store


def require_api_key(
    api_key: Annotated[str | None, Security(_API_KEY_HEADER)] = None,
) -> APIKeyRecord:
    """
    FastAPI dependency: validates X-API-Key header.

    If WHISPER_API_KEY is not set in the environment, authentication is
    disabled to allow development use without a key.

    Args:
        api_key: Key from X-API-Key header.

    Returns:
        Validated APIKeyRecord.

    Raises:
        HTTPException 401 if key is missing or invalid.
    """
    env_key = os.getenv(_API_KEY_ENV)
    if not env_key:
        # Auth disabled in development
        return APIKeyRecord(key_hash="dev", label="dev_mode", scopes=["*"])

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    store = get_key_store()
    try:
        return store.verify_key(api_key, required_scope="transcribe")
    except ValueError as exc:
        logger.warning("Auth failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "ApiKey"},
        ) from exc
# hmac.new(secret, key.encode(), sha256) used; only digest stored in _keys
# required_scope checked after revocation check in verify_key()
# APIKeyHeader(name='X-API-Key') used as Security dependency in routes
