"""Gradio demo for LongCat-AudioDiT text-to-speech and voice cloning.

Example:
    python gradio_app.py --model_dir meituan-longcat/LongCat-AudioDiT-3.5B
"""

import argparse
from typing import Optional

import gradio as gr
import numpy as np
import torch
from transformers import AutoTokenizer

import audiodit  # Registers AudioDiT with Transformers.
from audiodit import AudioDiTModel
from utils import approx_duration_from_text, load_audio, normalize_text


DEFAULT_MODEL_DIR = "meituan-longcat/LongCat-AudioDiT-3.5B"

_model: Optional[AudioDiTModel] = None
_tokenizer = None
_device: Optional[torch.device] = None


def resolve_device(device_name: Optional[str]) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: str, device: torch.device) -> None:
    """Load the model once, before Gradio starts accepting requests."""
    global _model, _tokenizer, _device

    model = AudioDiTModel.from_pretrained(model_dir).to(device)
    # The released checkpoint uses a half-precision VAE on CUDA. Keep CPU
    # execution in float32 because half-precision convolution is not portable.
    if device.type == "cuda":
        model.vae.to_half()
    model.eval()

    _model = model
    _tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
    _device = device


def _require_model() -> tuple[AudioDiTModel, object, torch.device]:
    if _model is None or _tokenizer is None or _device is None:
        raise RuntimeError("Model is not loaded. Start the app through gradio_app.py.")
    return _model, _tokenizer, _device


def _normalized_required(value: str, field_name: str) -> str:
    normalized = normalize_text(value or "").strip()
    if not normalized:
        raise gr.Error(f"{field_name}不能为空。")
    return normalized


@torch.inference_mode()
def generate_audio(
    text: str,
    prompt_audio_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    steps: int = 16,
    guidance_method: str = "cfg",
    guidance_strength: float = 4.0,
):
    """Generate one waveform and return Gradio's ``(sample_rate, samples)`` value."""
    model, tokenizer, device = _require_model()
    text = _normalized_required(text, "待合成文本")

    if steps < 2:
        raise gr.Error("ODE 步数至少为 2。")

    sampling_rate = model.config.sampling_rate
    latent_hop = model.config.latent_hop
    max_duration = model.config.max_wav_duration
    max_frames = int(max_duration * sampling_rate // latent_hop)

    prompt_wav = None
    prompt_frames = 0
    if prompt_audio_path:
        prompt_text = _normalized_required(prompt_text or "", "样本音频转写")
        prompt_wav = load_audio(prompt_audio_path, sampling_rate).unsqueeze(0)
        _, prompt_frames = model.encode_prompt_audio(prompt_wav)
        if prompt_frames >= max_frames:
            raise gr.Error(f"样本音频过长；其时长必须小于 {max_duration:.0f} 秒。")

        prompt_time = prompt_frames * latent_hop / sampling_rate
        generated_seconds = approx_duration_from_text(
            text, max_duration=max_duration - prompt_time
        )
        estimated_prompt_seconds = approx_duration_from_text(
            prompt_text, max_duration=max_duration
        )
        duration_ratio = float(
            np.clip(prompt_time / estimated_prompt_seconds, 1.0, 1.5)
        )
        generated_seconds *= duration_ratio
        full_text = f"{prompt_text} {text}"
    else:
        generated_seconds = approx_duration_from_text(text, max_duration=max_duration)
        full_text = text

    duration = int(generated_seconds * sampling_rate // latent_hop) + prompt_frames
    duration = min(duration, max_frames)
    inputs = tokenizer([full_text], padding="longest", return_tensors="pt")
    output = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_audio=prompt_wav,
        duration=duration,
        steps=int(steps),
        cfg_strength=float(guidance_strength),
        guidance_method=guidance_method,
    )
    waveform = output.waveform.squeeze().detach().float().cpu().numpy()
    return sampling_rate, waveform


def generate_tts(
    text: str, steps: int, guidance_method: str, guidance_strength: float
):
    return generate_audio(
        text,
        steps=steps,
        guidance_method=guidance_method,
        guidance_strength=guidance_strength,
    )


def generate_voice_clone(
    text: str,
    prompt_audio_path: str,
    prompt_text: str,
    steps: int,
    guidance_method: str,
    guidance_strength: float,
):
    return generate_audio(
        text,
        prompt_audio_path=prompt_audio_path,
        prompt_text=prompt_text,
        steps=steps,
        guidance_method=guidance_method,
        guidance_strength=guidance_strength,
    )


def create_demo() -> gr.Blocks:
    """Construct the UI without loading a model, enabling lightweight imports/tests."""
    with gr.Blocks(title="LongCat-AudioDiT") as demo:
        gr.Markdown("# LongCat-AudioDiT\n24 kHz 文本语音合成与零样本音色克隆。")
        with gr.Row():
            steps = gr.Slider(2, 32, value=16, step=1, label="ODE 步数")
            guidance_method = gr.Radio(["cfg", "apg"], value="cfg", label="引导方法")
            guidance_strength = gr.Slider(0, 8, value=4.0, step=0.1, label="引导强度")

        with gr.Tab("文本合成"):
            tts_text = gr.Textbox(label="待合成文本", lines=4)
            tts_button = gr.Button("生成语音", variant="primary")
            tts_output = gr.Audio(label="生成结果", type="numpy")
            tts_button.click(
                generate_tts,
                inputs=[tts_text, steps, guidance_method, guidance_strength],
                outputs=tts_output,
            )

        with gr.Tab("音色克隆"):
            gr.Markdown("样本音频转写应与上传的样本音频内容一致。")
            prompt_audio = gr.Audio(label="样本音频", type="filepath")
            prompt_text = gr.Textbox(label="样本音频转写", lines=3)
            clone_text = gr.Textbox(label="待合成文本", lines=4)
            clone_button = gr.Button("生成克隆语音", variant="primary")
            clone_output = gr.Audio(label="生成结果", type="numpy")
            clone_button.click(
                generate_voice_clone,
                inputs=[clone_text, prompt_audio, prompt_text, steps, guidance_method, guidance_strength],
                outputs=clone_output,
            )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the LongCat-AudioDiT Gradio demo")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Hugging Face model ID or local model directory")
    parser.add_argument("--device", default=None, help="Torch device, defaults to CUDA when available")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", default=7860, type=int, help="TCP port to bind")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Loading model from {args.model_dir} on {device}...")
    load_model(args.model_dir, device)
    print(f"Starting Gradio at http://{args.host}:{args.port}")
    create_demo().launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
