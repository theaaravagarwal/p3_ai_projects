#!/usr/bin/env python3
"""
Unit checks for OpenAI config helpers.
"""

from src.openai_config import (
    get_request_timeout_seconds,
    get_temperature_param,
    supports_custom_temperature,
)


def test_gpt5_models_skip_custom_temperature() -> None:
    assert supports_custom_temperature("gpt-5-nano") is False
    assert supports_custom_temperature("gpt-5.1") is False
    assert get_temperature_param("gpt-5-nano") == {}
    assert get_temperature_param("gpt-5.1") == {}


def test_older_models_keep_temperature() -> None:
    assert supports_custom_temperature("gpt-4o-mini") is True
    assert get_temperature_param("gpt-4o-mini") == {"temperature": 0.1}


def test_request_timeout_default_is_set() -> None:
    assert get_request_timeout_seconds() == 45.0
