"""Business services reusable by Gradio, CLI, and future HTTP APIs."""

from .tts_service import (
    AudioDiTService,
    DEFAULT_MODEL_DIR,
    DEFAULT_SPEECH_RATE,
    resolve_device,
)

__all__ = [
    "AudioDiTService",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_SPEECH_RATE",
    "resolve_device",
]
