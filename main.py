"""CLI entrypoint: transcribe an audio file and send the transcript to the Qwen LLM.

Usage:
    python main.py --audio path/to/input.wav
    python main.py --audio input.mp3 --instruction "Summarize this call:" --json
"""

import argparse
import json
import sys

from src.config import Config
from src.pipeline import STTtoLLMPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="STT -> LLM: transcribe an audio file (faster-whisper) "
        "and send the text to the Qwen-32B API."
    )
    p.add_argument("--audio", "-a", required=True, help="Path to the input audio file.")
    p.add_argument(
        "--instruction",
        "-i",
        default=None,
        help="Optional instruction prepended to the transcript before sending to the LLM.",
    )
    p.add_argument("--system", "-s", default=None, help="Override the system prompt.")
    p.add_argument(
        "--model", default=None, help="Override the Whisper model (e.g. large-v3, medium, small)."
    )
    p.add_argument("--device", default=None, help="Override device: auto | cuda | cpu.")
    p.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    p.add_argument("--out", default=None, help="Write the full result (JSON) to this file.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    config = Config()
    if args.model:
        config.whisper_model = args.model
    if args.device:
        config.whisper_device = args.device

    if not config.llm_api_key:
        print(
            "ERROR: LLM_API_KEY is not set. Copy .env.example to .env and add your "
            "ocp-apim-subscription-key (or export LLM_API_KEY).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        print(f"[1/2] Loading Whisper '{config.whisper_model}' and transcribing...", file=sys.stderr)
        pipeline = STTtoLLMPipeline(config)
        print("[2/2] Sending transcript to the Qwen LLM API...", file=sys.stderr)
        result = pipeline.run(
            args.audio, system_prompt=args.system, instruction=args.instruction
        )
    except Exception as exc:  # surface a clean error on the server console
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved result to {args.out}", file=sys.stderr)

    if args.json:
        print(output)
    else:
        print("\n=== TRANSCRIPT ===")
        print(result["transcript"])
        print(
            f"\n(language={result['language']}, device={result['device']}, "
            f"stt={result['stt_seconds']}s)"
        )
        print("\n=== LLM RESPONSE ===")
        print(result["response"])
        print(f"\n(llm={result['llm_seconds']}s)")


if __name__ == "__main__":
    main()
