"""Gradio demo for LongCat-AudioDiT text-to-speech and voice cloning.

Example:
    python gradio_app.py --model_dir meituan-longcat/LongCat-AudioDiT-1B
"""

import argparse
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer

import audiodit  # Registers AudioDiT with Transformers.
from audiodit import AudioDiTModel
from long_audio import TextSegment, segment_text, stitch_audio_files
from utils import approx_duration_from_text, load_audio, normalize_text


DEFAULT_MODEL_DIR = "meituan-longcat/LongCat-AudioDiT-1B"

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


def _segment_rows(segments: list[TextSegment]):
    return [
        [index + 1, segment.text, round(segment.estimated_seconds, 2), segment.boundary]
        for index, segment in enumerate(segments)
    ]


def preview_long_text(
    text: str, language: str, target_seconds: float, max_seconds: float
):
    try:
        return _segment_rows(
            segment_text(text, language, target_seconds, max_seconds)
        )
    except (ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error


def generate_long_audio(
    text: str,
    language: str,
    prompt_audio_path: Optional[str],
    prompt_text: Optional[str],
    target_seconds: float,
    max_seconds: float,
    steps: int,
    guidance_method: str,
    guidance_strength: float,
):
    model, _, _ = _require_model()
    has_audio = bool(prompt_audio_path)
    has_prompt_text = bool((prompt_text or "").strip())
    if has_audio != has_prompt_text:
        raise gr.Error("样本音频和样本音频转写必须同时提供。")

    effective_max = min(float(max_seconds), float(model.config.max_wav_duration))
    if has_audio:
        normalized_prompt = _normalized_required(prompt_text or "", "样本音频转写")
        prompt_wav = load_audio(prompt_audio_path, model.config.sampling_rate).unsqueeze(0)
        with torch.inference_mode():
            _, prompt_frames = model.encode_prompt_audio(prompt_wav)
        prompt_seconds = (
            prompt_frames * model.config.latent_hop / model.config.sampling_rate
        )
        estimated_prompt = approx_duration_from_text(
            normalized_prompt, max_duration=model.config.max_wav_duration
        )
        speed_ratio = float(np.clip(prompt_seconds / estimated_prompt, 1.0, 1.5))
        effective_max = min(
            effective_max,
            (model.config.max_wav_duration - prompt_seconds) / speed_ratio,
        )
    if effective_max <= 0:
        raise gr.Error("样本音频已占满模型的最大时长，无法生成新内容。")
    effective_target = min(float(target_seconds), effective_max)

    try:
        segments = segment_text(text, language, effective_target, effective_max)
    except (ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error

    job_dir = Path(tempfile.mkdtemp(prefix="longcat_long_"))
    wav_files: list[Path] = []
    for index, segment in enumerate(segments):
        # Reusing the seed keeps prompt VAE sampling and initial noise consistent
        # across independently generated segments.
        torch.manual_seed(1024)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1024)
        sample_rate, waveform = generate_audio(
            segment.text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            steps=steps,
            guidance_method=guidance_method,
            guidance_strength=guidance_strength,
        )
        wav_path = job_dir / f"segment_{index:06d}.wav"
        sf.write(wav_path, waveform, sample_rate, subtype="PCM_16")
        wav_files.append(wav_path)

    output_mp3 = job_dir / "longcat_output.mp3"
    try:
        stitch_audio_files(
            wav_files,
            [segment.boundary for segment in segments],
            output_mp3,
            bitrate="192k",
            sample_rate=model.config.sampling_rate,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise gr.Error(str(error)) from error
    for wav_path in wav_files:
        wav_path.unlink()
    rows = _segment_rows(segments)
    return rows, str(output_mp3), str(output_mp3)


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

        with gr.Tab("长文本生成"):
            gr.Markdown(
                "面向英语/西语长文本。分段后由本地模型串行生成，最终输出 192 kbps MP3。"
            )
            long_text = gr.Textbox(label="长文本", lines=10)
            with gr.Row():
                language = gr.Radio(
                    [("English", "en"), ("Español", "es")],
                    value="en",
                    label="语言",
                )
                target_seconds = gr.Slider(
                    5, 18, value=15, step=1, label="目标分段时长（秒）"
                )
                max_seconds = gr.Slider(
                    8, 20, value=20, step=1, label="最大分段时长（秒）"
                )
            long_prompt_audio = gr.Audio(label="样本音频（可选）", type="filepath")
            long_prompt_text = gr.Textbox(label="样本音频转写（可选）", lines=3)
            with gr.Row():
                preview_button = gr.Button("预览分段")
                long_generate_button = gr.Button("生成完整 MP3", variant="primary")
            segment_table = gr.Dataframe(
                headers=["序号", "文本", "预计秒数", "边界"],
                datatype=["number", "str", "number", "str"],
                interactive=False,
                label="分段结果",
            )
            long_audio_output = gr.Audio(label="完整 MP3", type="filepath")
            long_file_output = gr.File(label="下载 MP3")
            preview_button.click(
                preview_long_text,
                inputs=[long_text, language, target_seconds, max_seconds],
                outputs=segment_table,
            )
            long_generate_button.click(
                generate_long_audio,
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
