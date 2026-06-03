"""STTtoLLM: audio file -> faster-whisper transcription -> Qwen LLM API."""

from .config import Config
from .pipeline import STTtoLLMPipeline

__all__ = ["Config", "STTtoLLMPipeline"]
