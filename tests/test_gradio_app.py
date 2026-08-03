import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import gradio_app


class _Tokenizer:
    def __call__(self, text, **kwargs):
        del text, kwargs
        return types.SimpleNamespace(
            input_ids=torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1, 1]])
        )


class _Model:
    config = types.SimpleNamespace(
        sampling_rate=24000, latent_hop=2048, max_wav_duration=30.0
    )

    def __init__(self):
        self.calls = []

    def encode_prompt_audio(self, prompt_audio):
        del prompt_audio
        return torch.zeros(1, 4, 64), 4

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(waveform=torch.zeros(1, 16))


class GradioAppTest(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        gradio_app._model = self.model
        gradio_app._tokenizer = _Tokenizer()
        gradio_app._device = torch.device("cpu")

    def tearDown(self):
        gradio_app._model = None
        gradio_app._tokenizer = None
        gradio_app._device = None

    def test_text_to_speech_passes_no_prompt(self):
        sample_rate, waveform = gradio_app.generate_audio("Hello world")

        self.assertEqual(sample_rate, 24000)
        self.assertEqual(waveform.shape, (16,))
        self.assertIsNone(self.model.calls[-1]["prompt_audio"])

    @patch("gradio_app.load_audio", return_value=torch.zeros(1, 2048))
    def test_voice_clone_passes_prompt_and_joined_text(self, load_audio):
        gradio_app.generate_audio("new text", "sample.wav", "sample transcript")

        self.assertTrue(load_audio.called)
        self.assertIsNotNone(self.model.calls[-1]["prompt_audio"])
        self.assertGreater(self.model.calls[-1]["duration"], 4)

    @patch("gradio_app.stitch_audio_files")
    @patch("gradio_app.generate_audio", return_value=(24000, torch.zeros(16).numpy()))
    def test_long_audio_generates_each_segment_and_returns_mp3(
        self, generate_audio, stitch_audio_files
    ):
        def create_mp3(files, boundaries, output, **kwargs):
            del files, boundaries, kwargs
            Path(output).write_bytes(b"mp3")
            return Path(output)

        stitch_audio_files.side_effect = create_mp3
        rows, audio_path, file_path = gradio_app.generate_long_audio(
            "First sentence. Second sentence.",
            "en",
            None,
            None,
            2,
            3,
            16,
            "cfg",
            4.0,
        )

        self.assertGreaterEqual(generate_audio.call_count, 2)
        self.assertEqual(audio_path, file_path)
        self.assertTrue(Path(file_path).is_file())
        self.assertEqual(len(rows), generate_audio.call_count)


if __name__ == "__main__":
    unittest.main()
