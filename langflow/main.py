#!/usr/bin/env python3
"""
Project launcher for Assistant.

Run this with `uv run main.py` to see a small interactive launcher, or pass a
subcommand for a direct action:

  uv run main.py setup-env
  uv run main.py status
  uv run main.py bot
  uv run main.py api
  uv run main.py test
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from src.openai_config import (
    DEFAULT_MAX_HISTORY_MESSAGES,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    get_active_model,
    get_max_history_messages,
    get_max_output_tokens,
    get_request_timeout_seconds,
    get_temperature,
    has_active_api_key,
    supports_custom_temperature,
)


ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
MODEL_CHOICES = ("gpt-5-nano", "gpt-4o-mini")


def setup_env(force: bool = False) -> Path:
    if ENV_FILE.exists() and not force:
        print(f".env already exists at {ENV_FILE}")
        return ENV_FILE

    if not ENV_EXAMPLE.exists():
        raise FileNotFoundError(f"Missing template: {ENV_EXAMPLE}")

    shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
    print(f"Created {ENV_FILE} from .env.example")
    print("Add your real OPENAI_API_KEY, then rerun the app.")
    return ENV_FILE


def read_env_lines() -> list[str]:
    if not ENV_FILE.exists():
        return []
    return ENV_FILE.read_text().splitlines()


def write_env_lines(lines: list[str]) -> None:
    text = "\n".join(lines)
    if lines:
        text += "\n"
    ENV_FILE.write_text(text)


def set_model(model: str) -> None:
    if model not in MODEL_CHOICES:
        raise ValueError(f"Model must be one of: {', '.join(MODEL_CHOICES)}")

    lines = read_env_lines()
    updated = []
    found = False
    for line in lines:
        if line.startswith("OPENAI_MODEL="):
            updated.append(f"OPENAI_MODEL={model}")
            found = True
        else:
            updated.append(line)

    if not found:
        updated.append(f"OPENAI_MODEL={model}")

    write_env_lines(updated)
    print(f"Set OPENAI_MODEL={model} in {ENV_FILE}")


def print_status() -> None:
    print("\nProject Status")
    print("=" * 60)
    print(f"Project root: {ROOT}")
    print(f".env present: {'yes' if ENV_FILE.exists() else 'no'}")
    print(f"Active API key configured: {'yes' if has_active_api_key() else 'no'}")
    print(f"Active model: {get_active_model()}")
    print(f"Max output tokens: {get_max_output_tokens()}")
    print(f"Max history messages: {get_max_history_messages()}")
    print(f"Request timeout seconds: {get_request_timeout_seconds()}")
    if supports_custom_temperature(get_active_model()):
        print(f"Temperature: {get_temperature()}")
    else:
        print(f"Temperature: default only for {get_active_model()} (custom values ignored)")
    print(
        "Recommended default: "
        f"{DEFAULT_MODEL}, {DEFAULT_MAX_OUTPUT_TOKENS} tokens, "
        f"{DEFAULT_MAX_HISTORY_MESSAGES} history messages, "
        f"temperature {DEFAULT_TEMPERATURE} for older non-reasoning models"
    )
    print("=" * 60)


def run_bot() -> None:
    from src.bot import run_interactive_session

    run_interactive_session()


def run_api(reload: bool = False) -> None:
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("src.api:app", host=host, port=port, reload=reload)


def run_tests() -> int:
    from tests.test_prompts import run_tests as run_prompt_tests, test_conversation_flow
    from tests.test_openai_config import (
        test_gpt5_models_skip_custom_temperature,
        test_older_models_keep_temperature,
        test_request_timeout_default_is_set,
    )

    results = run_prompt_tests()
    flow_ok = test_conversation_flow()
    test_gpt5_models_skip_custom_temperature()
    test_older_models_keep_temperature()
    test_request_timeout_default_is_set()

    passed = sum(1 for item in results if item.get("status") == "PASSED")
    failed = sum(1 for item in results if item.get("status") == "FAILED")
    print(f"\nTest summary: {passed} passed, {failed} failed, flow={'ok' if flow_ok else 'failed'}")
    return 0 if failed == 0 and flow_ok else 1


def interactive_menu() -> int:
    while True:
        print("\nAssistant Launcher")
        print("1. Setup .env from .env.example")
        print("2. Show status")
        print("3. Run bot")
        print("4. Run API")
        print("5. Run tests")
        print("6. Choose model")
        print("q. Quit")

        choice = input("Select an option: ").strip().lower()
        if choice == "1":
            setup_env()
        elif choice == "2":
            print_status()
        elif choice == "3":
            run_bot()
        elif choice == "4":
            run_api(reload=True)
        elif choice == "5":
            return run_tests()
        elif choice == "6":
            choose_model_interactive()
        elif choice in {"q", "quit", "exit"}:
            return 0
        else:
            print("Unknown choice. Try again.")


def choose_model_interactive() -> None:
    print("\nChoose a model")
    for idx, model in enumerate(MODEL_CHOICES, 1):
        print(f"{idx}. {model}")

    choice = input("Select a model: ").strip().lower()
    if choice in {"1", MODEL_CHOICES[0]}:
        set_model(MODEL_CHOICES[0])
    elif choice in {"2", MODEL_CHOICES[1]}:
        set_model(MODEL_CHOICES[1])
    else:
        print("Unknown model choice.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assistant project launcher")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("setup-env", help="Create .env from .env.example")
    subparsers.add_parser("status", help="Show local project status")
    subparsers.add_parser("bot", help="Run the interactive chat bot")
    api_parser = subparsers.add_parser("api", help="Run the FastAPI server")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    subparsers.add_parser("test", help="Run prompt and flow tests")
    model_parser = subparsers.add_parser("model", help="Set the default OpenAI model in .env")
    model_parser.add_argument("model", choices=MODEL_CHOICES, help="Model to set as default")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "setup-env":
        setup_env()
        return 0
    if args.command == "status":
        print_status()
        return 0
    if args.command == "bot":
        run_bot()
        return 0
    if args.command == "api":
        run_api(reload=args.reload)
        return 0
    if args.command == "test":
        return run_tests()
    if args.command == "model":
        set_model(args.model)
        return 0

    if sys.stdin.isatty():
        print_status()
        return interactive_menu()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
