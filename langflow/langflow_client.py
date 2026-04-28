from __future__ import annotations

import os
import json
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()


class LangflowClientError(RuntimeError):
    pass


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_langflow_api_key() -> str:
    langflow_api_key = os.getenv("LANGFLOW_API_KEY", "").strip()
    if langflow_api_key:
        return langflow_api_key

    legacy_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if legacy_api_key:
        return legacy_api_key

    raise RuntimeError(
        "Missing required environment variable: LANGFLOW_API_KEY "
        "(or legacy OPENAI_API_KEY)"
    )


def get_settings() -> dict[str, str]:
    base_url = require_env("LANGFLOW_BASE_URL").rstrip("/")
    flow_id = require_env("LANGFLOW_FLOW_ID")
    api_key = get_langflow_api_key()
    input_type = os.getenv("LANGFLOW_INPUT_TYPE", "chat").strip() or "chat"
    output_type = os.getenv("LANGFLOW_OUTPUT_TYPE", "chat").strip() or "chat"

    return {
        "base_url": base_url,
        "flow_id": flow_id,
        "openai_api_key": api_key,
        "input_type": input_type,
        "output_type": output_type,
    }


def build_run_url() -> str:
    settings = get_settings()
    return f"{settings['base_url']}/api/v1/run/{settings['flow_id']}"


def build_headers() -> dict[str, str]:
    settings = get_settings()
    headers = {
        "Content-Type": "application/json",
        "x-api-key": settings["openai_api_key"],
    }
    for variable_name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "WATSONX_API_KEY",
        "WATSONX_PROJECT_ID",
    ):
        value = os.getenv(variable_name, "").strip()
        if value:
            headers[f"X-LANGFLOW-GLOBAL-VAR-{variable_name}"] = value
    return headers


def build_tweaks() -> dict[str, Any]:
    raw_tweaks = os.getenv("LANGFLOW_TWEAKS_JSON", "").strip()
    if not raw_tweaks:
        return {}
    try:
        parsed = json.loads(raw_tweaks)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Invalid LANGFLOW_TWEAKS_JSON. Expected valid JSON for the Langflow "
            f"'tweaks' payload: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(
            "Invalid LANGFLOW_TWEAKS_JSON. Expected a JSON object for the Langflow "
            "'tweaks' payload."
        )
    return parsed


def build_payload(user_message: str) -> dict[str, Any]:
    settings = get_settings()
    payload: dict[str, Any] = {
        "input_value": user_message,
        "input_type": settings["input_type"],
        "output_type": settings["output_type"],
    }
    tweaks = build_tweaks()
    if tweaks:
        payload["tweaks"] = tweaks
    return payload


def call_langflow(user_message: str, timeout_seconds: int = 60) -> dict[str, Any]:
    settings = get_settings()
    run_url = build_run_url()

    try:
        response = requests.post(
            run_url,
            json=build_payload(user_message),
            headers=build_headers(),
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise LangflowClientError(
            "Could not connect to the Langflow backend. "
            f"Expected it at {settings['base_url']}. "
            "Start Langflow there or update LANGFLOW_BASE_URL in .env."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        response_text = ""
        if exc.response is not None:
            response_text = exc.response.text.strip()
        message = (
            f"Langflow API request failed with status {status_code} for {run_url}."
        )
        if response_text:
            message = f"{message} Response: {response_text[:300]}"
        if "Anthropic API key is required" in response_text:
            message = (
                f"{message} Set ANTHROPIC_API_KEY in this repo's .env so the client "
                "can pass it as X-LANGFLOW-GLOBAL-VAR-ANTHROPIC_API_KEY."
            )
        raise LangflowClientError(message) from exc
    except requests.exceptions.RequestException as exc:
        raise LangflowClientError(
            f"Langflow request failed for {run_url}: {exc}"
        ) from exc


TEXT_KEYS = (
    "text",
    "message",
    "content",
    "response",
    "answer",
    "output",
    "result",
)

CONTAINER_KEYS = (
    "outputs",
    "results",
    "artifacts",
    "data",
    "generations",
    "messages",
)


def _extract_from_string(value: str) -> str:
    text = value.strip()
    if not text:
        return ""

    if text[0] in "[{":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text

        extracted = _find_text(parsed, allow_plain_string=False)
        if extracted:
            return extracted

    return text


def _find_text(data: Any, *, allow_plain_string: bool) -> str:
    if isinstance(data, str):
        if allow_plain_string:
            return _extract_from_string(data)
        return ""

    if isinstance(data, list):
        for item in data:
            text = _find_text(item, allow_plain_string=False)
            if text:
                return text
        return ""

    if not isinstance(data, dict):
        return ""

    for key in TEXT_KEYS:
        if key not in data:
            continue

        value = data[key]
        text = _find_text(value, allow_plain_string=True)
        if text:
            return text

    for key in CONTAINER_KEYS:
        if key not in data:
            continue

        text = _find_text(data[key], allow_plain_string=False)
        if text:
            return text

    return ""


def extract_text(data: Any) -> str:
    text = _find_text(data, allow_plain_string=True)
    if text:
        return text

    if isinstance(data, (dict, list)):
        return json.dumps(data, indent=2)

    return str(data)
