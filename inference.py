"""Command-line inference through the reusable AudioDiT business service."""

from __future__ import annotations

import argparse

import soundfile as sf

from app_config import (
    DEFAULT_GUIDANCE_METHOD,
    DEFAULT_GUIDANCE_STRENGTH,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_DIR,
    DEFAULT_ODE_STEPS,
    DEFAULT_SPEECH_RATE,
)
from services import AudioDiTService, resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AudioDiT TTS inference")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--prompt_text", default=None, help="Sample audio transcript")
    parser.add_argument("--prompt_audio", default=None, help="Sample audio path")
    parser.add_argument("--output_audio", required=True, help="Output WAV path")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--language", choices=["en", "es"], default=DEFAULT_LANGUAGE)
    parser.add_argument(
        "--nfe", type=int, default=DEFAULT_ODE_STEPS, help="Number of ODE steps"
    )
    parser.add_argument(
        "--guidance_strength", type=float, default=DEFAULT_GUIDANCE_STRENGTH
    )
    parser.add_argument("--speech_rate", type=float, default=DEFAULT_SPEECH_RATE)
    parser.add_argument(
        "--guidance_method", default=DEFAULT_GUIDANCE_METHOD, choices=["cfg", "apg"]
    )
    parser.add_argument("--seed", type=int, default=1024)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    service = AudioDiTService.from_pretrained(args.model_dir, device)
    sample_rate, waveform = service.generate_audio(
        args.text,
        language=args.language,
        prompt_audio_path=args.prompt_audio,
        prompt_text=args.prompt_text,
        steps=args.nfe,
        guidance_method=args.guidance_method,
        guidance_strength=args.guidance_strength,
        speech_rate=args.speech_rate,
        seed=args.seed,
    )
    sf.write(args.output_audio, waveform, sample_rate)
    print(f"Saved: {args.output_audio} ({len(waveform) / sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
