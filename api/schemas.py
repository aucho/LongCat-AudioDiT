"""Request models for the asynchronous LongCat API."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app_config import (
    DEFAULT_GUIDANCE_METHOD,
    DEFAULT_GUIDANCE_STRENGTH,
    DEFAULT_MAX_SEGMENT_SECONDS,
    DEFAULT_ODE_STEPS,
    DEFAULT_SEED,
    DEFAULT_TARGET_SEGMENT_SECONDS,
    MAX_GENERATION_SECONDS,
    MAX_GENERATION_TEXT_CHARS,
)
from .validation import ResourceId


class GenerateAudioRequest(BaseModel):
    step_id: ResourceId
    text: str = Field(min_length=1, max_length=MAX_GENERATION_TEXT_CHARS)
    language: Literal["en", "es"]
    reference_id: ResourceId
    seed: int = DEFAULT_SEED
    speech_rate: float | None = Field(default=None, gt=0)
    target_seconds: float = Field(
        default=DEFAULT_TARGET_SEGMENT_SECONDS, gt=0, le=MAX_GENERATION_SECONDS
    )
    max_seconds: float = Field(
        default=DEFAULT_MAX_SEGMENT_SECONDS, gt=0, le=MAX_GENERATION_SECONDS
    )
    steps: int = Field(default=DEFAULT_ODE_STEPS, ge=2)
    guidance_method: Literal["cfg", "apg"] = DEFAULT_GUIDANCE_METHOD
    guidance_strength: float = DEFAULT_GUIDANCE_STRENGTH

    @model_validator(mode="after")
    def validate_segment_duration(self):
        if self.target_seconds > self.max_seconds:
            raise ValueError("target_seconds must not exceed max_seconds")
        return self


__all__ = ["GenerateAudioRequest"]
