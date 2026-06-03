"""Quick connectivity check for the Qwen LLM API -- no audio / no GPU needed.

Run this first on the server to confirm the endpoint + subscription key work:
    python check_llm.py
"""

import sys

from src.config import Config
from src.llm import LLMClient, LLMError


def main() -> None:
    config = Config()
    if not config.llm_api_key:
        print("ERROR: LLM_API_KEY is not set (.env or environment).", file=sys.stderr)
        sys.exit(1)

    print(f"Endpoint: {config.llm_api_url}")
    try:
        reply = LLMClient(config).chat("What is the most powerful version of Qwen?")
    except LLMError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    print("OK. Model replied:\n")
    print(reply)


if __name__ == "__main__":
    main()
