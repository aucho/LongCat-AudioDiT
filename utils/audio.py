"""Stateless audio loading helpers."""

from __future__ import annotations


def load_audio(wavpath, sr):
    import librosa
    import torch

    audio, _ = librosa.load(wavpath, sr=sr, mono=True)
    return torch.from_numpy(audio).unsqueeze(0)


__all__ = ["load_audio"]
