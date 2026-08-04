"""Batch AudioDiT inference for SeedTTS-style evaluation lists."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Batch AudioDiT inference")
    parser.add_argument("--lst", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--language", choices=["en", "es"], default=DEFAULT_LANGUAGE)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--nfe", type=int, default=DEFAULT_ODE_STEPS)
    parser.add_argument(
        "--guidance_strength", type=float, default=DEFAULT_GUIDANCE_STRENGTH
    )
    parser.add_argument("--speech_rate", type=float, default=DEFAULT_SPEECH_RATE)
    parser.add_argument(
        "--guidance_method", default=DEFAULT_GUIDANCE_METHOD, choices=["cfg", "apg"]
    )
    return parser.parse_args(argv)


def read_items(list_path: Path) -> list[tuple[str, str, Path, str]]:
    items = []
    with list_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", maxsplit=3)
            if len(parts) != 4:
                raise ValueError(f"invalid list row {line_number}: expected four fields")
            uid, prompt_text, prompt_path, generated_text = parts
            items.append(
                (uid, prompt_text, list_path.parent / prompt_path, generated_text)
            )
    return items


def main() -> None:
    args = parse_args()
    list_path = Path(args.lst).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items = read_items(list_path)
    print(f"Loaded {len(items)} items from {list_path}")

    device = resolve_device(args.device)
    service = AudioDiTService.from_pretrained(args.model_dir, device)
    print(f"Model loaded on {device}, method={args.guidance_method}")

    started = time.time()
    for index, (uid, prompt_text, prompt_audio, generated_text) in enumerate(items):
        output_path = output_dir / f"{uid}.wav"
        if output_path.exists():
            continue
        try:
            sample_rate, waveform = service.generate_voice_clone(
                generated_text,
                str(prompt_audio),
                prompt_text,
                language=args.language,
                steps=args.nfe,
                guidance_method=args.guidance_method,
                guidance_strength=args.guidance_strength,
                speech_rate=args.speech_rate,
                seed=args.seed,
            )
            sf.write(output_path, waveform, sample_rate)
            elapsed = time.time() - started
            speed = (index + 1) / elapsed if elapsed else 0.0
            eta = (len(items) - index - 1) / speed if speed else 0.0
            print(
                f"[{index + 1}/{len(items)}] {uid}  "
                f"{len(waveform) / sample_rate:.1f}s  "
                f"({speed:.1f} it/s, ETA {eta / 60:.0f}min)"
            )
        except Exception as error:
            print(f"[{index + 1}/{len(items)}] ERROR {uid}: {error}")
    print(f"Done. {len(items)} items in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
