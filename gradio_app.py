"""Gradio UI adapter for the reusable AudioDiT service."""

from __future__ import annotations

import argparse
from typing import Optional

import gradio as gr

from services import AudioDiTService, DEFAULT_MODEL_DIR, resolve_device
from utils import TextSegment


def _segment_rows(segments: list[TextSegment]) -> list[list[object]]:
    return [
        [index + 1, segment.text, round(segment.estimated_seconds, 2), segment.boundary]
        for index, segment in enumerate(segments)
    ]


def _service_or_error(service: Optional[AudioDiTService]) -> AudioDiTService:
    if service is None:
        raise gr.Error("Model is not loaded. Start the app through gradio_app.py.")
    return service


def _as_gradio_error(operation):
    try:
        return operation()
    except gr.Error:
        raise
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise gr.Error(str(error)) from error


def create_demo(service: Optional[AudioDiTService] = None) -> gr.Blocks:
    """Construct the page without loading a model at import time."""

    def generate_tts_ui(text, language, steps, method, strength):
        return _as_gradio_error(
            lambda: _service_or_error(service).generate_tts(
                text,
                language=language,
                steps=steps,
                guidance_method=method,
                guidance_strength=strength,
            )
        )

    def generate_clone_ui(
        text, language, prompt_audio, prompt_text, steps, method, strength
    ):
        return _as_gradio_error(
            lambda: _service_or_error(service).generate_voice_clone(
                text,
                prompt_audio,
                prompt_text,
                language=language,
                steps=steps,
                guidance_method=method,
                guidance_strength=strength,
            )
        )

    def preview_long_ui(text, language, target_seconds, max_seconds):
        return _as_gradio_error(
            lambda: _segment_rows(
                _service_or_error(service).preview_long_text(
                    text, language, target_seconds, max_seconds
                )
            )
        )

    def generate_long_ui(
        text,
        language,
        prompt_audio,
        prompt_text,
        target_seconds,
        max_seconds,
        steps,
        method,
        strength,
    ):
        def generate():
            segments, output = _service_or_error(service).generate_long_audio(
                text,
                language,
                prompt_audio,
                prompt_text,
                target_seconds,
                max_seconds,
                steps,
                method,
                strength,
            )
            output_path = str(output)
            return _segment_rows(segments), output_path, output_path

        return _as_gradio_error(generate)

    with gr.Blocks(title="LongCat-AudioDiT") as demo:
        gr.Markdown(
            "# LongCat-AudioDiT\n"
            "24 kHz English/Spanish text-to-speech and voice cloning test page."
        )
        with gr.Row():
            language = gr.Radio(
                [("English", "en"), ("Español", "es")],
                value="en",
                label="Language",
            )
            steps = gr.Slider(2, 32, value=16, step=1, label="ODE steps")
            guidance_method = gr.Radio(
                ["cfg", "apg"], value="cfg", label="Guidance method"
            )
            guidance_strength = gr.Slider(
                0, 8, value=4.0, step=0.1, label="Guidance strength"
            )

        with gr.Tab("Text synthesis"):
            tts_text = gr.Textbox(label="Text to synthesize", lines=4)
            tts_button = gr.Button("Generate speech", variant="primary")
            tts_output = gr.Audio(label="Generated audio", type="numpy")
            tts_button.click(
                generate_tts_ui,
                inputs=[tts_text, language, steps, guidance_method, guidance_strength],
                outputs=tts_output,
            )

        with gr.Tab("Voice cloning"):
            gr.Markdown(
                "The sample transcript must exactly match the uploaded sample audio."
            )
            prompt_audio = gr.Audio(label="Sample audio", type="filepath")
            prompt_text = gr.Textbox(label="Sample transcript", lines=3)
            clone_text = gr.Textbox(label="Text to synthesize", lines=4)
            clone_button = gr.Button("Generate cloned speech", variant="primary")
            clone_output = gr.Audio(label="Generated audio", type="numpy")
            clone_button.click(
                generate_clone_ui,
                inputs=[
                    clone_text,
                    language,
                    prompt_audio,
                    prompt_text,
                    steps,
                    guidance_method,
                    guidance_strength,
                ],
                outputs=clone_output,
            )

        with gr.Tab("Long text generation"):
            gr.Markdown(
                "Every segment uses the same required sample audio and transcript. "
                "The final file is a 24 kHz mono 192 kbps MP3."
            )
            long_text = gr.Textbox(label="Long text", lines=10)
            with gr.Row():
                target_seconds = gr.Slider(
                    5, 18, value=15, step=1, label="Target segment seconds"
                )
                max_seconds = gr.Slider(
                    8, 20, value=20, step=1, label="Maximum segment seconds"
                )
            long_prompt_audio = gr.Audio(
                label="Sample audio (required)", type="filepath"
            )
            long_prompt_text = gr.Textbox(
                label="Accurate sample transcript (required)", lines=3
            )
            with gr.Row():
                preview_button = gr.Button("Preview segments")
                long_generate_button = gr.Button(
                    "Generate complete MP3", variant="primary"
                )
            segment_table = gr.Dataframe(
                headers=["Index", "Spoken text", "Estimated seconds", "Boundary"],
                datatype=["number", "str", "number", "str"],
                interactive=False,
                label="Segments",
            )
            long_audio_output = gr.Audio(label="Complete MP3", type="filepath")
            long_file_output = gr.File(label="Download MP3")
            preview_button.click(
                preview_long_ui,
                inputs=[long_text, language, target_seconds, max_seconds],
                outputs=segment_table,
            )
            long_generate_button.click(
                generate_long_ui,
                inputs=[
                    long_text,
                    language,
                    long_prompt_audio,
                    long_prompt_text,
                    target_seconds,
                    max_seconds,
                    steps,
                    guidance_method,
                    guidance_strength,
                ],
                outputs=[segment_table, long_audio_output, long_file_output],
            )
    return demo


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the LongCat-AudioDiT Gradio demo"
    )
    parser.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help="Hugging Face model ID or local model directory",
    )
    parser.add_argument(
        "--device", default=None, help="Torch device, defaults to CUDA when available"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", default=7860, type=int, help="TCP port to bind")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Loading model from {args.model_dir} on {device}...")
    service = AudioDiTService.from_pretrained(args.model_dir, device)
    print(f"Starting Gradio at http://{args.host}:{args.port}")
    create_demo(service).launch(
        server_name=args.host, server_port=args.port, share=False
    )


if __name__ == "__main__":
    main()
