"""Business services reusable by Gradio, CLI, and future HTTP APIs."""

from .tts_service import (
    AudioDiTService,
    DEFAULT_MODEL_DIR,
    DEFAULT_SPEECH_RATE,
    resolve_device,
)
from app_config import MAX_GENERATION_SECONDS

__all__ = [
    "AudioDiTService",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_SPEECH_RATE",
    "MAX_GENERATION_SECONDS",
    "resolve_device",
]
