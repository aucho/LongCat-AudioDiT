"""General text cleanup and duration-estimation helpers."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Apply the checkpoint's existing lowercase and whitespace cleanup."""
    text = (text or "").lower()
    text = re.sub(r'["“”‘’]', " ", text)
    return re.sub(r"\s+", " ", text).strip()


def approx_duration_from_text(text: str, max_duration: float = 30.0) -> float:
    """Estimate speech duration using the project's character-rate heuristic."""
    en_duration_per_char = 0.082
    zh_duration_per_char = 0.21
    text = re.sub(r"\s+", "", text or "")
    num_zh = num_en = num_other = 0
    for character in text:
        if "\u4e00" <= character <= "\u9fff":
            num_zh += 1
        elif character.isalpha():
            num_en += 1
        else:
            num_other += 1
    if num_zh > num_en:
        num_zh += num_other
    else:
        num_en += num_other
    return min(
        max_duration,
        num_zh * zh_duration_per_char + num_en * en_duration_per_char,
    )


__all__ = ["approx_duration_from_text", "normalize_text"]
