"""
Shared OpenAI configuration for the Assistant chatbot.

All runtime paths use OpenAI directly.
"""

import os
from typing import Any, Sequence

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_MAX_OUTPUT_TOKENS = 64
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_HISTORY_MESSAGES = 2
DEFAULT_REASONING_EFFORT = "minimal"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 45.0


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_openai_api_key() -> str | None:
    value = os.getenv("OPENAI_API_KEY")
    if not value:
        return None

    normalized = value.strip()
    if normalized in {"your_api_key_here", "sk-your-key-here"}:
        return None

    return normalized


def has_openai_api_key() -> bool:
    return get_openai_api_key() is not None


def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", os.getenv("MODEL_NAME", DEFAULT_MODEL))


def get_active_model() -> str:
    return get_openai_model()


def get_max_output_tokens() -> int:
    return _get_int_env("OPENAI_MAX_OUTPUT_TOKENS", _get_int_env("MAX_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))


def get_temperature() -> float:
    return _get_float_env("OPENAI_TEMPERATURE", _get_float_env("TEMPERATURE", DEFAULT_TEMPERATURE))


def get_max_history_messages() -> int:
    value = os.getenv("OPENAI_MAX_HISTORY_MESSAGES")
    if value is None or value.strip() == "":
        return DEFAULT_MAX_HISTORY_MESSAGES
    try:
        return int(value)
    except ValueError:
        return DEFAULT_MAX_HISTORY_MESSAGES


def get_reasoning_effort() -> str:
    return os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip().lower()


def get_request_timeout_seconds() -> float:
    return _get_float_env("OPENAI_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS)


def supports_reasoning_effort(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def supports_custom_temperature(model: str) -> bool:
    # GPT-5 / reasoning-family models only accept the default temperature.
    return not supports_reasoning_effort(model)


def get_openai_client() -> OpenAI | None:
    api_key = get_openai_api_key()
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        timeout=get_request_timeout_seconds(),
        max_retries=1,
    )


def has_active_api_key() -> bool:
    return has_openai_api_key()


def get_active_client() -> OpenAI | None:
    return get_openai_client()


def supports_reasoning_for_active_model(model: str) -> bool:
    return supports_reasoning_effort(model)


def get_completion_token_param(model: str | None = None) -> dict[str, int]:
    # Reasoning models use max_completion_tokens. Keep max_tokens for older models.
    active_model = model or get_active_model()
    if supports_reasoning_effort(active_model):
        return {"max_completion_tokens": get_max_output_tokens()}
    return {"max_tokens": get_max_output_tokens()}


def get_temperature_param(model: str | None = None) -> dict[str, float]:
    active_model = model or get_active_model()
    if not supports_custom_temperature(active_model):
        return {}
    return {"temperature": get_temperature()}


def trim_conversation_history(messages: Sequence[Any], max_messages: int | None = None) -> list[Any]:
    limit = max_messages if max_messages is not None else get_max_history_messages()
    if limit <= 0:
        return list(messages)
    return list(messages)[-limit:]
