"""
Shared OpenAI configuration for the Assistant chatbot.

This keeps the API server and CLI aligned on the same low-cost defaults
and makes budget-related settings easy to override from the environment.
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


def supports_reasoning_effort(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def get_openai_client() -> OpenAI | None:
    api_key = get_openai_api_key()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def trim_conversation_history(messages: Sequence[Any], max_messages: int | None = None) -> list[Any]:
    limit = max_messages if max_messages is not None else get_max_history_messages()
    if limit <= 0:
        return list(messages)
    return list(messages)[-limit:]
