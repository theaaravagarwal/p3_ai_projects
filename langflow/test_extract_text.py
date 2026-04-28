from __future__ import annotations

from langflow_client import extract_text


def test_extracts_nested_langflow_message() -> None:
    data = {
        "outputs": [
            {
                "outputs": [
                    {
                        "results": {
                            "message": {
                                "text": "Hello from Langflow",
                                "sender": "Machine",
                            }
                        }
                    }
                ]
            }
        ]
    }

    assert extract_text(data) == "Hello from Langflow"


def test_skips_unrelated_nested_dicts_before_message() -> None:
    data = {
        "outputs": [
            {"component_display_name": "Debug component"},
            {"results": {"message": {"text": "Actual reply"}}},
        ]
    }

    assert extract_text(data) == "Actual reply"


def test_extracts_stringified_json_reply() -> None:
    data = {
        "outputs": [
            {
                "results": {
                    "message": {
                        "text": '{"answer": "Parsed reply", "sources": ["doc"]}'
                    }
                }
            }
        ]
    }

    assert extract_text(data) == "Parsed reply"


def test_formats_unrecognized_json_as_json() -> None:
    data = {"unexpected": {"shape": True}}

    assert extract_text(data) == '{\n  "unexpected": {\n    "shape": true\n  }\n}'
