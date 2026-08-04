"""English/Spanish long-text segmentation and streamed MP3 stitching helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .text import approx_duration_from_text


SUPPORTED_LANGUAGES = {"en", "es"}
BOUNDARY_PAUSES = {
    "paragraph": 0.18,
    "sentence": 0.12,
    "clause": 0.09,
    "word": 0.06,
}

_ABBREVIATIONS = {
    "en": {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
        "vs.", "etc.", "e.g.", "i.e.", "a.m.", "p.m.", "u.s.",
        "u.k.", "no.", "fig.", "inc.", "ltd.",
    },
    "es": {
        "sr.", "sra.", "srta.", "dr.", "dra.", "prof.", "ud.",
        "uds.", "pág.", "págs.", "núm.", "aprox.", "etc.", "p. ej.",
    },
}


@dataclass(frozen=True)
class TextSegment:
    text: str
    boundary: str
    estimated_seconds: float


def _estimate(text: str) -> float:
    return approx_duration_from_text(text, max_duration=float("inf"))


def _protect_non_boundaries(text: str, language: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        key = f"\ue000{len(replacements)}\ue001"
        replacements[key] = match.group(0)
        return key

    protected = re.sub(
        r"https?://[^\s,;!?]+(?<!\.)|www\.[^\s,;!?]+(?<!\.)|[\w.+-]+@[\w.-]+\.\w+",
        protect,
        text,
    )
    protected = re.sub(r"(?<=\d)\.(?=\d)", protect, protected)
    for abbreviation in sorted(_ABBREVIATIONS[language], key=len, reverse=True):
        protected = re.sub(
            rf"(?<!\w){re.escape(abbreviation)}(?!\w)",
            protect,
            protected,
            flags=re.IGNORECASE,
        )
    return protected, replacements


def _restore(text: str, replacements: dict[str, str]) -> str:
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def _split_sentences(paragraph: str, language: str) -> list[str]:
    protected, replacements = _protect_non_boundaries(paragraph, language)
    sentences: list[str] = []
    start = 0
    index = 0
    closers = {'"', "'", "”", "’", "»", ")", "]"}
    while index < len(protected):
        if protected[index] not in ".!?":
            index += 1
            continue
        end = index + 1
        while end < len(protected) and protected[end] in ".!?":
            end += 1
        while end < len(protected) and protected[end] in closers:
            end += 1
        if end == len(protected) or protected[end].isspace():
            sentence = _restore(protected[start:end], replacements).strip()
            if sentence:
                sentences.append(sentence)
            start = end
            while start < len(protected) and protected[start].isspace():
                start += 1
            index = start
        else:
            index = end
    tail = _restore(protected[start:], replacements).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _pack_parts(parts: Sequence[str], target_seconds: float, boundary: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = " ".join([*current, part])
        if current and _estimate(candidate) > target_seconds:
            chunks.append((" ".join(current), boundary))
            current = [part]
        else:
            current.append(part)
    if current:
        chunks.append((" ".join(current), boundary))
    return chunks


def _split_long_sentence(
    sentence: str, language: str, target_seconds: float, max_seconds: float
) -> list[tuple[str, str]]:
    split_patterns = [
        r"(?<=[;:])\s+",
        r"(?<=,)\s+|\s+(?=[—–-]\s)",
        (
            r"\s+(?=(?:and|but|or|because|while|however|therefore|"
            r"y|pero|o|porque|aunque|sin embargo)\b)"
        ),
    ]
    for pattern in split_patterns:
        parts = re.split(pattern, sentence, flags=re.IGNORECASE)
        if len(parts) <= 1:
            continue
        chunks = _pack_parts(parts, target_seconds, "clause")
        if all(_estimate(text) <= max_seconds for text, _ in chunks):
            return chunks

    words = sentence.split()
    chunks = _pack_parts(words, target_seconds, "word")
    if any(_estimate(text) > max_seconds for text, _ in chunks):
        raise ValueError(
            "A single token exceeds the maximum segment duration; add a safe split point."
        )
    return chunks


def segment_text(
    text: str,
    language: str,
    target_seconds: float = 15.0,
    max_seconds: float = 20.0,
) -> list[TextSegment]:
    """Split English or Spanish text into ordered, duration-bounded segments."""
    language = (language or "").lower().strip()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("language must be 'en' or 'es'")
    if not text or not text.strip():
        raise ValueError("text cannot be empty")
    if target_seconds <= 0 or max_seconds <= 0 or target_seconds > max_seconds:
        raise ValueError("target_seconds must be positive and no greater than max_seconds")

    atoms: list[tuple[str, str]] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    for paragraph in paragraphs:
        sentences = _split_sentences(re.sub(r"\s+", " ", paragraph), language)
        for sentence_index, sentence in enumerate(sentences):
            boundary = "paragraph" if sentence_index == len(sentences) - 1 else "sentence"
            if _estimate(sentence) <= max_seconds:
                atoms.append((sentence, boundary))
            else:
                split = _split_long_sentence(
                    sentence, language, target_seconds, max_seconds
                )
                split[-1] = (split[-1][0], boundary)
                atoms.extend(split)

    packed: list[TextSegment] = []
    current_text = ""
    current_boundary = "sentence"
    for atom_text, atom_boundary in atoms:
        candidate = f"{current_text} {atom_text}".strip()
        crosses_paragraph = current_text and current_boundary == "paragraph"
        if current_text and (crosses_paragraph or _estimate(candidate) > target_seconds):
            packed.append(
                TextSegment(current_text, current_boundary, _estimate(current_text))
            )
            current_text = atom_text
        else:
            current_text = candidate
        current_boundary = atom_boundary
    if current_text:
        packed.append(TextSegment(current_text, current_boundary, _estimate(current_text)))

    if any(segment.estimated_seconds > max_seconds for segment in packed):
        raise RuntimeError("segmenter produced a segment over max_seconds")
    return packed


def _prepare_wav(
    source: Path,
    destination: Path,
    pause_seconds: float,
    sample_rate: int,
    fade_seconds: float,
) -> None:
    import librosa
    import numpy as np
    import soundfile as sf

    audio, source_rate = sf.read(source, dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"audio file is empty: {source}")
    mono = audio.mean(axis=1)
    if source_rate != sample_rate:
        mono = librosa.resample(mono, orig_sr=source_rate, target_sr=sample_rate)

    fade_samples = min(int(fade_seconds * sample_rate), len(mono) // 2)
    if fade_samples:
        ramp = np.linspace(0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32)
        mono[:fade_samples] *= ramp
        mono[-fade_samples:] *= ramp[::-1]
    pause = np.zeros(int(pause_seconds * sample_rate), dtype=np.float32)
    sf.write(destination, np.concatenate([mono, pause]), sample_rate, subtype="PCM_16")


def stitch_audio_files(
    files: Sequence[str | Path],
    boundaries: Sequence[str],
    output_mp3: str | Path,
    bitrate: str = "192k",
    sample_rate: int = 24000,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    """Stream ordered audio files through FFmpeg into one compressed MP3."""
    if not files:
        raise ValueError("at least one audio file is required")
    if len(boundaries) != len(files):
        raise ValueError("boundaries must contain one entry for every audio file")
    invalid_boundaries = set(boundaries) - set(BOUNDARY_PAUSES)
    if invalid_boundaries:
        raise ValueError(f"unsupported boundaries: {sorted(invalid_boundaries)}")
    if shutil.which(ffmpeg_bin) is None and not Path(ffmpeg_bin).is_file():
        raise FileNotFoundError(f"FFmpeg executable not found: {ffmpeg_bin}")

    sources = [Path(file).resolve() for file in files]
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"audio files not found: {missing}")

    destination = Path(output_mp3).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="longcat_stitch_") as temp_name:
        temp_dir = Path(temp_name)
        concat_lines: list[str] = []
        for index, (source, boundary) in enumerate(zip(sources, boundaries)):
            processed = temp_dir / f"{index:06d}.wav"
            pause = 0.0 if index == len(sources) - 1 else BOUNDARY_PAUSES[boundary]
            _prepare_wav(source, processed, pause, sample_rate, fade_seconds=0.01)
            concat_lines.append(f"file '{processed.name}'")

        concat_file = temp_dir / "concat.txt"
        concat_file.write_text("\n".join(concat_lines), encoding="utf-8")
        command = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-vn",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(destination),
        ]
        try:
            subprocess.run(
                command,
                cwd=temp_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            details = (error.stderr or error.stdout or str(error)).strip()
            raise RuntimeError(f"FFmpeg failed to create MP3: {details}") from error
    return destination


__all__ = ["TextSegment", "segment_text", "stitch_audio_files"]
