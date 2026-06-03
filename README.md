# STTtoLLM_1

Audio file → **Speech-to-Text** (faster-whisper, GPU) → **LLM** (Qwen-32B API) → response.

A small CLI pipeline: give it an audio file, it transcribes it with Whisper on the
GPU, then sends the transcript to your hosted Qwen-32B endpoint and prints the reply.

## Flow

```
input.wav ──► faster-whisper (H200 GPU) ──► transcript ──► Qwen-32B API ──► response
```

## Layout

| File / dir          | Purpose                                                        |
| ------------------- | ------------------------------------------------------------- |
| `app.py`            | **Web UI** — upload audio, see transcript + LLM response      |
| `main.py`           | CLI entrypoint (`python main.py --audio input.wav`)           |
| `check_llm.py`      | Standalone LLM connectivity test (no audio/GPU needed)        |
| `src/config.py`     | Env-based configuration (`.env`)                              |
| `src/stt.py`        | faster-whisper transcription + CUDA auto-detect               |
| `src/llm.py`        | Qwen-32B `chat/completions` client                            |
| `src/pipeline.py`   | Orchestrates STT → LLM and times each stage                   |
| `requirements.txt`  | Python dependencies                                           |
| `.env.example`      | Template for secrets/config — copy to `.env`                  |

## Setup (on the RHEL / H200 server)

```bash
# 1. Python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Config — copy the template and add your real subscription key
cp .env.example .env
#   then edit .env and set LLM_API_KEY=<your ocp-apim-subscription-key>
```

> **GPU note:** faster-whisper needs the CUDA 12 + cuDNN 9 runtime libraries on the
> host. On an H200 these are normally present. If you hit a `libcudnn` / `cublas`
> load error, install them (e.g. `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`)
> or set `WHISPER_DEVICE=cpu` to fall back.

## Run

### Web UI (upload audio in the browser)

```bash
python app.py
# then open http://<server-ip>:7860
# for a temporary public share link instead:  UI_SHARE=1 python app.py
```

The Whisper model loads once at startup; each upload runs STT → LLM and shows the
transcript and the model's reply.

### CLI

```bash
# Confirm the LLM endpoint works first
python check_llm.py

# Full pipeline
python main.py --audio path/to/input.wav

# With an instruction for the model, JSON output saved to a file
python main.py --audio call.mp3 --instruction "Summarize this call:" --json --out result.json
```

### Useful flags

| Flag                  | Meaning                                              |
| --------------------- | ---------------------------------------------------- |
| `--audio, -a`         | Input audio file (required)                          |
| `--instruction, -i`   | Text prepended to the transcript before the LLM      |
| `--system, -s`        | Override the system prompt                           |
| `--model`             | Whisper model (`large-v3`, `medium`, `small`, ...)   |
| `--device`            | `auto` \| `cuda` \| `cpu`                            |
| `--json` / `--out`    | Emit full JSON result / write it to a file           |

## Configuration (`.env`)

| Variable               | Default                                                            |
| ---------------------- | ----------------------------------------------------------------- |
| `LLM_API_URL`          | `https://ltceip4prod.azure-api.net/qwen32b/chat/completions`      |
| `LLM_API_KEY`          | *(required)* — the `ocp-apim-subscription-key`                    |
| `LLM_SYSTEM_PROMPT`    | `You are a helpful assistant.`                                    |
| `LLM_MAX_TOKENS`       | `5000`                                                            |
| `WHISPER_MODEL`        | `large-v3`                                                        |
| `WHISPER_DEVICE`       | `auto` (→ `cuda` on the H200)                                     |
| `WHISPER_COMPUTE_TYPE` | `auto` (→ `float16` on GPU)                                       |

The `.env` file is git-ignored, so your subscription key is never committed.
