"""Client for the Qwen-32B chat/completions API (Azure API Management).

OpenAI-compatible request/response shape. Auth is the `ocp-apim-subscription-key`
header, matching the working curl call.
"""

from typing import Optional

import requests

from .config import Config


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: Config):
        self.config = config

    def chat(self, user_content: str, system_prompt: Optional[str] = None) -> str:
        cfg = self.config
        if not cfg.llm_api_key:
            raise LLMError("LLM_API_KEY is not set (the ocp-apim-subscription-key).")

        headers = {
            "Content-Type": "application/json",
            "ocp-apim-subscription-key": cfg.llm_api_key,
        }
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt or cfg.llm_system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": cfg.llm_max_tokens,
        }
        if cfg.llm_temperature is not None:
            payload["temperature"] = cfg.llm_temperature

        try:
            resp = requests.post(
                cfg.llm_api_url, headers=headers, json=payload, timeout=cfg.llm_timeout
            )
        except requests.RequestException as exc:
            raise LLMError(f"Request to LLM API failed: {exc}") from exc

        if resp.status_code != 200:
            raise LLMError(
                f"LLM API returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMError(
                f"Unexpected LLM API response shape: {resp.text[:500]}"
            ) from exc
