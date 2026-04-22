"""
Runtime safety controls for public API deployment.

These guards are intentionally lightweight and dependency-free so they work in
both local and container deployments.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Iterable

from fastapi import HTTPException, Request


DEFAULT_RATE_LIMIT_PER_MINUTE = 30
DEFAULT_RATE_LIMIT_BURST = 10
DEFAULT_MAX_CHAT_MESSAGE_CHARS = 4000
DEFAULT_MAX_CONVERSATION_MESSAGES = 12


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_public_api_key() -> str | None:
    value = os.getenv("PUBLIC_API_KEY")
    if not value:
        return None
    normalized = value.strip()
    if normalized in {"your_public_api_key_here", "change-me"}:
        return None
    return normalized


def is_public_api_key_required() -> bool:
    return get_public_api_key() is not None


def get_allowed_origins() -> list[str]:
    value = os.getenv("CORS_ORIGINS", "").strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_rate_limit_per_minute() -> int:
    return _get_int_env("API_RATE_LIMIT_PER_MINUTE", DEFAULT_RATE_LIMIT_PER_MINUTE)


def get_rate_limit_burst() -> int:
    return _get_int_env("API_RATE_LIMIT_BURST", DEFAULT_RATE_LIMIT_BURST)


def get_max_chat_message_chars() -> int:
    return _get_int_env("API_MAX_CHAT_MESSAGE_CHARS", DEFAULT_MAX_CHAT_MESSAGE_CHARS)


def get_max_conversation_messages() -> int:
    return _get_int_env("API_MAX_CONVERSATION_MESSAGES", DEFAULT_MAX_CONVERSATION_MESSAGES)


def trust_proxy_headers() -> bool:
    return _get_bool_env("TRUST_PROXY_HEADERS", False)


def get_client_ip(request: Request) -> str:
    if trust_proxy_headers():
        forwarded_for = request.headers.get("x-forwarded-for", "").strip()
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

    client = request.client
    if client and client.host:
        return client.host

    return "unknown"


def _normalize_token(token: str) -> str:
    return token.strip().lower().replace(" ", "_")


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    def __init__(self, rate_per_minute: int, burst: int) -> None:
        self._rate_per_second = max(rate_per_minute, 1) / 60.0
        self._capacity = max(burst, 1)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                self._buckets[key] = _Bucket(tokens=self._capacity - 1, last_refill=now)
                return True

            elapsed = now - bucket.last_refill
            bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._rate_per_second)
            bucket.last_refill = now

            if bucket.tokens < 1:
                return False

            bucket.tokens -= 1
            return True


_rate_limiter = RateLimiter(get_rate_limit_per_minute(), get_rate_limit_burst())


def check_public_api_key(request: Request) -> None:
    expected = get_public_api_key()
    if expected is None:
        return

    provided = request.headers.get("x-api-key")
    if provided is None:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()

    if provided != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


def check_rate_limit(request: Request, scope: str = "chat") -> None:
    client_ip = get_client_ip(request)
    key = f"{_normalize_token(scope)}:{client_ip}"
    if not _rate_limiter.allow(key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


def validate_chat_payload(message: str, conversation_messages: int) -> None:
    if len(message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if len(message) > get_max_chat_message_chars():
        raise HTTPException(status_code=413, detail="Message too long")

    if conversation_messages > get_max_conversation_messages():
        raise HTTPException(status_code=413, detail="Conversation history too long")
