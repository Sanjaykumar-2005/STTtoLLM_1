"""Web UI: upload an audio file, see the transcript and the Qwen LLM response.

Built on Gradio. The Whisper model is loaded ONCE at startup (not per request),
so the GPU model stays warm.

Run on the server:
    python app.py
Then open http://<server-ip>:7860  (or use the share link if enabled).
"""

import os
import sys

import gradio as gr

from src.config import Config
from src.pipeline import STTtoLLMPipeline

# --- Load the pipeline once at startup (loads Whisper into GPU memory) ---
config = Config()
if not config.llm_api_key:
    print(
        "WARNING: LLM_API_KEY is not set. The UI will start, but LLM calls will "
        "fail until you set it in .env and restart.",
        file=sys.stderr,
    )

print("Loading Whisper model (one-time)...", file=sys.stderr)
pipeline = STTtoLLMPipeline(config)
print(f"Ready. STT device = {pipeline.transcriber.device}", file=sys.stderr)


def process(audio_path, instruction, system_prompt):
    """Run the full STT -> LLM pipeline for one uploaded file."""
    if not audio_path:
        return "", "", "Please upload or record an audio file first."

    try:
        result = pipeline.run(
            audio_path,
            system_prompt=system_prompt or None,
            instruction=instruction or None,
        )
    except Exception as exc:
        return "", "", f"Error: {exc}"

    info = (
        f"language={result['language']} | device={result['device']} | "
        f"STT {result['stt_seconds']}s | LLM {result['llm_seconds']}s"
    )
    return result["transcript"], result["response"], info


with gr.Blocks(title="STT → LLM") as demo:
    gr.Markdown(
        "# 🎙️ STT → LLM\n"
        "Upload (or record) an audio file. It is transcribed with **faster-whisper** "
        "on the GPU, then the transcript is sent to the **Qwen-32B** API and the "
        "response is shown below."
    )

    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio file",
            )
            instruction = gr.Textbox(
                label="Instruction (optional)",
                placeholder="e.g. Summarize this, or Answer the question above.",
            )
            system_prompt = gr.Textbox(
                label="System prompt (optional)",
                placeholder="You are a helpful assistant.",
            )
            run_btn = gr.Button("Transcribe & Ask LLM", variant="primary")

        with gr.Column():
            transcript_out = gr.Textbox(label="Transcript", lines=5)
            response_out = gr.Markdown(label="LLM response")
            info_out = gr.Markdown()

    run_btn.click(
        fn=process,
        inputs=[audio_in, instruction, system_prompt],
        outputs=[transcript_out, response_out, info_out],
    )

    gr.Examples(
        examples=[
            [os.path.join("samples", "sample2_question.wav")],
            [os.path.join("samples", "sample3_meeting.wav")],
        ],
        inputs=audio_in,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # reachable from outside the server
        server_port=int(os.getenv("UI_PORT", "7860")),
        share=os.getenv("UI_SHARE", "0") == "1",  # set UI_SHARE=1 for a public link
    )
