"""Reusable AudioDiT model lifecycle and speech-generation workflows."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
import torch
from app_config import (
    DEFAULT_GUIDANCE_METHOD,
    DEFAULT_GUIDANCE_STRENGTH,
    DEFAULT_MAX_SEGMENT_SECONDS,
    DEFAULT_MODEL_DIR,
    DEFAULT_MP3_BITRATE,
    DEFAULT_ODE_STEPS,
    DEFAULT_SPEECH_RATE,
    DEFAULT_SEED,
    DEFAULT_TARGET_SEGMENT_SECONDS,
    MAX_GENERATION_SECONDS,
    MAX_GENERATION_TEXT_CHARS,
    MAX_REFERENCE_TEXT_CHARS,
)
from utils import (
    TextSegment,
    approx_duration_from_text,
    load_audio,
    normalize_text,
    normalize_tts_text,
    segment_text,
    stitch_audio_files,
)


class GenerationCancelledError(RuntimeError):
    """Raised when a long-running generation is cancelled between segments."""


def resolve_device(device_name: Optional[str] = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioDiTService:
    """Own one loaded model and expose UI-independent generation operations."""

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device | str,
        max_generation_seconds: float = MAX_GENERATION_SECONDS,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_generation_seconds = float(max_generation_seconds)
        if self.max_generation_seconds <= 0:
            raise ValueError("max_generation_seconds must be greater than 0")
        # AudioDiT.forward applies this config value as its final hard cap.
        self.model.config.max_wav_duration = self.max_generation_seconds

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str = DEFAULT_MODEL_DIR,
        device: torch.device | str | None = None,
    ) -> "AudioDiTService":
        from transformers import AutoTokenizer

        import audiodit  # noqa: F401 - registers AudioDiT with Transformers.
        from audiodit import AudioDiTModel

        resolved_device = resolve_device(str(device) if device is not None else None)
        model = AudioDiTModel.from_pretrained(model_dir).to(resolved_device)
        if resolved_device.type == "cuda":
            model.vae.to_half()
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
        return cls(model, tokenizer, resolved_device)

    @staticmethod
    def _required(value: Optional[str], field_name: str) -> str:
        if not value or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")
        return value

    @staticmethod
    def _set_seed(seed: Optional[int]) -> None:
        if seed is None:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _spoken_text(self, text: str, language: str, field_name: str) -> str:
        source = self._required(text, field_name)
        source_limit = (
            MAX_REFERENCE_TEXT_CHARS
            if field_name == "prompt transcript"
            else MAX_GENERATION_TEXT_CHARS
        )
        if len(source) > source_limit:
            raise ValueError(f"{field_name} exceeds {source_limit} characters")
        spoken = normalize_tts_text(source, language).spoken_text
        normalized = normalize_text(spoken)
        if not normalized:
            raise ValueError(f"{field_name} cannot be empty")
        if len(normalized) > source_limit:
            raise ValueError(
                f"normalized {field_name} exceeds {source_limit} characters"
            )
        return normalized

    def normalize_spoken_text(
        self, text: str, language: str, field_name: str = "text"
    ) -> str:
        """Validate and normalize text before a task is admitted to the queue."""
        return self._spoken_text(text, language, field_name)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @property
    def model_name(self) -> str:
        return str(getattr(self.model.config, "name_or_path", DEFAULT_MODEL_DIR))

    @torch.inference_mode()
    def generate_audio(
        self,
        text: str,
        language: str = "en",
        prompt_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        steps: int = DEFAULT_ODE_STEPS,
        guidance_method: str = DEFAULT_GUIDANCE_METHOD,
        guidance_strength: float = DEFAULT_GUIDANCE_STRENGTH,
        speech_rate: float = DEFAULT_SPEECH_RATE,
        seed: Optional[int] = None,
    ) -> tuple[int, np.ndarray]:
        """Generate one waveform as ``(sample_rate, mono_float_samples)``."""
        if int(steps) < 2:
            raise ValueError("ODE steps must be at least 2")
        guidance_method = (guidance_method or "").lower()
        if guidance_method not in {"cfg", "apg"}:
            raise ValueError("guidance_method must be 'cfg' or 'apg'")
        speech_rate = float(speech_rate)
        if speech_rate <= 0:
            raise ValueError("speech_rate must be greater than 0")

        spoken_text = self._spoken_text(text, language, "text")
        sampling_rate = self.model.config.sampling_rate
        latent_hop = self.model.config.latent_hop
        max_duration = self.model.config.max_wav_duration
        max_frames = int(max_duration * sampling_rate // latent_hop)

        prompt_wav = None
        prompt_frames = 0
        if prompt_audio_path:
            spoken_prompt = self._spoken_text(
                prompt_text or "", language, "prompt transcript"
            )
            prompt_wav = load_audio(prompt_audio_path, sampling_rate).unsqueeze(0)
            _, prompt_frames = self.model.encode_prompt_audio(prompt_wav)
            if prompt_frames >= max_frames:
                raise ValueError(
                    f"prompt audio must be shorter than {max_duration:.0f} seconds"
                )
            prompt_time = prompt_frames * latent_hop / sampling_rate
            generated_seconds = approx_duration_from_text(
                spoken_text, max_duration=max_duration - prompt_time
            )
            estimated_prompt_seconds = approx_duration_from_text(
                spoken_prompt, max_duration=max_duration
            )
            if estimated_prompt_seconds <= 0:
                raise ValueError("prompt transcript cannot be empty")
            duration_ratio = float(
                np.clip(prompt_time / estimated_prompt_seconds, 1.0, 1.5)
            )
            generated_seconds *= duration_ratio
            full_text = f"{spoken_prompt} {spoken_text}"
        elif prompt_text and prompt_text.strip():
            raise ValueError(
                "prompt audio is required when a prompt transcript is provided"
            )
        else:
            generated_seconds = approx_duration_from_text(
                spoken_text, max_duration=max_duration
            )
            full_text = spoken_text

        # AudioDiT has no dedicated pace control. A shorter generated duration
        # produces faster speech while leaving the reference audio unchanged.
        generated_seconds /= speech_rate
        duration = int(generated_seconds * sampling_rate // latent_hop) + prompt_frames
        duration = min(duration, max_frames)
        inputs = self.tokenizer([full_text], padding="longest", return_tensors="pt")
        self._set_seed(seed)
        output = self.model(
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

    def generate_tts(self, text: str, language: str = "en", **kwargs):
        return self.generate_audio(text, language=language, **kwargs)

    def generate_voice_clone(
        self,
        text: str,
        prompt_audio_path: str,
        prompt_text: str,
        language: str = "en",
        **kwargs,
    ):
        self._required(prompt_audio_path, "prompt audio")
        self._required(prompt_text, "prompt transcript")
        return self.generate_audio(
            text,
            language=language,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            **kwargs,
        )

    def preview_long_text(
        self,
        text: str,
        language: str,
        target_seconds: float = DEFAULT_TARGET_SEGMENT_SECONDS,
        max_seconds: float = DEFAULT_MAX_SEGMENT_SECONDS,
    ) -> list[TextSegment]:
        spoken_text = self._spoken_text(text, language, "text")
        return segment_text(spoken_text, language, target_seconds, max_seconds)

    def _long_text_budget(
        self,
        prompt_audio_path: str,
        prompt_text: str,
        language: str,
        max_seconds: float,
    ) -> tuple[float, str]:
        spoken_prompt = self._spoken_text(prompt_text, language, "prompt transcript")
        prompt_wav = load_audio(
            prompt_audio_path, self.model.config.sampling_rate
        ).unsqueeze(0)
        with torch.inference_mode():
            _, prompt_frames = self.model.encode_prompt_audio(prompt_wav)
        prompt_seconds = (
            prompt_frames
            * self.model.config.latent_hop
            / self.model.config.sampling_rate
        )
        estimated_prompt = approx_duration_from_text(
            spoken_prompt, max_duration=self.model.config.max_wav_duration
        )
        if estimated_prompt <= 0:
            raise ValueError("prompt transcript cannot be empty")
        speed_ratio = float(np.clip(prompt_seconds / estimated_prompt, 1.0, 1.5))
        available = (self.model.config.max_wav_duration - prompt_seconds) / speed_ratio
        effective_max = min(
            float(max_seconds), float(self.model.config.max_wav_duration), available
        )
        if effective_max <= 0:
            raise ValueError("prompt audio leaves no model duration for generated speech")
        return effective_max, spoken_prompt

    def generate_long_audio(
        self,
        text: str,
        language: str,
        prompt_audio_path: str,
        prompt_text: str,
        target_seconds: float = DEFAULT_TARGET_SEGMENT_SECONDS,
        max_seconds: float = DEFAULT_MAX_SEGMENT_SECONDS,
        steps: int = DEFAULT_ODE_STEPS,
        guidance_method: str = DEFAULT_GUIDANCE_METHOD,
        guidance_strength: float = DEFAULT_GUIDANCE_STRENGTH,
        speech_rate: float = DEFAULT_SPEECH_RATE,
        seed: int = DEFAULT_SEED,
        bitrate: str = DEFAULT_MP3_BITRATE,
        should_cancel: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[TextSegment], Path]:
        self._required(prompt_audio_path, "prompt audio")
        self._required(prompt_text, "prompt transcript")
        effective_max, _ = self._long_text_budget(
            prompt_audio_path, prompt_text, language, max_seconds
        )
        effective_target = min(float(target_seconds), effective_max)
        segments = self.preview_long_text(
            text, language, effective_target, effective_max
        )
        total_segments = len(segments)
        if progress_callback:
            progress_callback(0, total_segments)

        job_dir = Path(tempfile.mkdtemp(prefix="longcat_long_"))
        wav_files: list[Path] = []
        success = False
        try:
            for index, segment in enumerate(segments):
                if should_cancel and should_cancel():
                    raise GenerationCancelledError("generation cancelled")
                sample_rate, waveform = self.generate_voice_clone(
                    segment.text,
                    prompt_audio_path,
                    prompt_text,
                    language=language,
                    steps=steps,
                    guidance_method=guidance_method,
                    guidance_strength=guidance_strength,
                    speech_rate=speech_rate,
                    seed=seed,
                )
                if should_cancel and should_cancel():
                    raise GenerationCancelledError("generation cancelled")
                wav_path = job_dir / f"segment_{index:06d}.wav"
                sf.write(wav_path, waveform, sample_rate, subtype="PCM_16")
                wav_files.append(wav_path)
                if progress_callback:
                    progress_callback(index + 1, total_segments)

            if should_cancel and should_cancel():
                raise GenerationCancelledError("generation cancelled")
            output_mp3 = job_dir / "longcat_output.mp3"
            stitch_audio_files(
                wav_files,
                [segment.boundary for segment in segments],
                output_mp3,
                bitrate=bitrate,
                sample_rate=self.model.config.sampling_rate,
            )
            success = True
            return segments, output_mp3
        finally:
            for wav_path in wav_files:
                wav_path.unlink(missing_ok=True)
            if not success:
                shutil.rmtree(job_dir, ignore_errors=True)


__all__ = [
    "AudioDiTService",
    "GenerationCancelledError",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_SPEECH_RATE",
    "resolve_device",
]
