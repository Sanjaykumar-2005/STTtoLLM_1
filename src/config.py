"""Central configuration. Values come from environment variables (or a .env file).

Nothing secret is hard-coded here -- the LLM subscription key must be supplied
via LLM_API_KEY so it never lands in source control.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; real env vars still work without it.
    pass


def _str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _float_opt(name: str) -> Optional[float]:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else None


def _str_opt(name: str) -> Optional[str]:
    raw = os.getenv(name)
    return raw if raw not in (None, "") else None


@dataclass
class Config:
    # --- STT (faster-whisper) ---
    whisper_model: str = field(default_factory=lambda: _str("WHISPER_MODEL", "large-v3"))
    whisper_device: str = field(default_factory=lambda: _str("WHISPER_DEVICE", "auto"))
    whisper_compute_type: str = field(default_factory=lambda: _str("WHISPER_COMPUTE_TYPE", "auto"))
    whisper_language: Optional[str] = field(default_factory=lambda: _str_opt("WHISPER_LANGUAGE"))
    whisper_beam_size: int = field(default_factory=lambda: _int("WHISPER_BEAM_SIZE", 5))

    # --- LLM API (Qwen-32B via Azure API Management) ---
    llm_api_url: str = field(
        default_factory=lambda: _str(
            "LLM_API_URL", "https://ltceip4prod.azure-api.net/qwen32b/chat/completions"
        )
    )
    llm_api_key: str = field(default_factory=lambda: _str("LLM_API_KEY", ""))
    llm_system_prompt: str = field(
        default_factory=lambda: _str("LLM_SYSTEM_PROMPT", "You are a helpful assistant.")
    )
    llm_max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 5000))
    llm_temperature: Optional[float] = field(default_factory=lambda: _float_opt("LLM_TEMPERATURE"))
    llm_timeout: int = field(default_factory=lambda: _int("LLM_TIMEOUT", 120))
