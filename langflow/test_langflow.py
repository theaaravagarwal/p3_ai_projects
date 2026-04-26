from __future__ import annotations

from pprint import pprint

from langflow_client import call_langflow, extract_text


def main() -> int:
    data = call_langflow("Hello")
    print("Raw response:")
    pprint(data)
    print("\nExtracted text:")
    print(extract_text(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
