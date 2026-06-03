"""Speech-to-text using faster-whisper (CTranslate2 backend, GPU-accelerated)."""

from typing import Tuple

from faster_whisper import WhisperModel

from .config import Config


def _resolve_device(config: Config) -> Tuple[str, str]:
    """Pick (device, compute_type), auto-detecting CUDA when requested."""
    device = config.whisper_device
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    compute_type = config.whisper_compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


class Transcriber:
    """Loads a Whisper model once and transcribes audio files."""

    def __init__(self, config: Config):
        self.config = config
        self.device, self.compute_type = _resolve_device(config)
        self.model = WhisperModel(
            config.whisper_model,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio_path: str):
        """Return (transcript_text, info) for the given audio file."""
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=self.config.whisper_beam_size,
            language=self.config.whisper_language,
        )
        # segments is a generator; consume it to build the full transcript.
        text = "".join(segment.text for segment in segments)
        return text.strip(), info
