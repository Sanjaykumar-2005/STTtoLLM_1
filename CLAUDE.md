# STTtoLLM_1 — working notebook

Purpose: CLI pipeline that takes an **audio file → faster-whisper STT (GPU) → Qwen-32B LLM API → response**.
Target runtime: RHEL host with an **NVIDIA H200** GPU. Dev machine is Windows.

## Files
- `app.py` — Gradio web UI; loads pipeline once at startup, audio upload/mic → transcript + LLM response. Serves on 0.0.0.0:7860 (UI_PORT / UI_SHARE env).
- `main.py` — CLI entrypoint; parses args, runs the pipeline, prints transcript + LLM reply.
- `check_llm.py` — standalone Qwen API connectivity test (no audio/GPU).
- `src/config.py` — dataclass `Config`, all settings from env / `.env` (no secrets in code).
- `src/stt.py` — `Transcriber` wrapping `faster_whisper.WhisperModel`; auto-detects CUDA via `ctranslate2.get_cuda_device_count()`.
- `src/llm.py` — `LLMClient` POSTing to the Qwen `chat/completions` endpoint with `ocp-apim-subscription-key` header (OpenAI-compatible shape).
- `src/pipeline.py` — `STTtoLLMPipeline.run()` orchestrates STT→LLM, returns dict with transcript, response, timings, device.

## Key decisions
- LLM is the user's **hosted Qwen-32B API**, not a local model. Key supplied via `LLM_API_KEY` env (header `ocp-apim-subscription-key`).
- STT = faster-whisper, default model `large-v3`, `float16` on GPU.
- Secrets via `.env` (git-ignored); `.env.example` is the template.

## Run
`pip install -r requirements.txt` → `cp .env.example .env` (set key) → `python check_llm.py` → `python main.py --audio input.wav`

## Recent changes
- 2026-06-03: Initial scaffold — created full pipeline (config/stt/llm/pipeline), CLI, LLM connectivity checker, README, requirements, .env.example, .gitignore.
- 2026-06-03: Added 5 TTS test WAVs in samples/ (Windows System.Speech generator).
- 2026-06-03: Added Gradio web UI (app.py) — audio upload/mic → transcript + LLM response; added gradio to requirements.
