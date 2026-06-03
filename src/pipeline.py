"""Orchestrates the full pipeline: audio file -> transcript -> LLM response."""

import os
import time
from typing import Optional

from .config import Config
from .llm import LLMClient
from .stt import Transcriber


class STTtoLLMPipeline:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        # Heavy: loads the Whisper model into (GPU) memory once.
        self.transcriber = Transcriber(self.config)
        self.llm = LLMClient(self.config)

    def run(
        self,
        audio_path: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> dict:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        t0 = time.time()
        transcript, info = self.transcriber.transcribe(audio_path)
        t1 = time.time()

        # By default the transcript itself is the user message. An optional
        # instruction lets you tell the LLM what to do with it.
        user_content = transcript
        if instruction:
            user_content = f"{instruction}\n\n{transcript}"

        response = self.llm.chat(user_content, system_prompt=system_prompt)
        t2 = time.time()

        return {
            "audio_path": audio_path,
            "transcript": transcript,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "response": response,
            "stt_seconds": round(t1 - t0, 2),
            "llm_seconds": round(t2 - t1, 2),
            "device": self.transcriber.device,
        }
