import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from services import AudioDiTService


class _Tokenizer:
    def __init__(self):
        self.texts = []

    def __call__(self, texts, **kwargs):
        del kwargs
        self.texts.append(texts[0])
        return types.SimpleNamespace(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.tensor([[1, 1]]),
        )


class _Model:
    config = types.SimpleNamespace(
        sampling_rate=24000,
        latent_hop=2048,
        max_wav_duration=30.0,
        text_encoder_model="mock-tokenizer",
    )

    def __init__(self):
        self.calls = []

    def encode_prompt_audio(self, prompt_audio):
        del prompt_audio
        return torch.zeros(1, 4, 64), 4

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(waveform=torch.zeros(1, 16))


class AudioDiTServiceTest(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        self.tokenizer = _Tokenizer()
        self.service = AudioDiTService(self.model, self.tokenizer, "cpu")

    def test_tts_tokenizes_spoken_number_text_and_uses_its_duration(self):
        sample_rate, waveform = self.service.generate_tts("I have 12.5 apples.")

        self.assertEqual(sample_rate, 24000)
        self.assertEqual(waveform.shape, (16,))
        self.assertEqual(
            self.tokenizer.texts[-1], "i have twelve point five apples."
        )
        self.assertIsNone(self.model.calls[-1]["prompt_audio"])
        self.assertGreater(self.model.calls[-1]["duration"], 10)

    def test_default_speech_rate_is_slightly_faster(self):
        self.service.generate_tts("This is a sentence.", speech_rate=1.0)
        original_duration = self.model.calls[-1]["duration"]
        self.service.generate_tts("This is a sentence.")
        faster_duration = self.model.calls[-1]["duration"]

        self.assertLess(faster_duration, original_duration)

    def test_speech_rate_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "speech_rate"):
            self.service.generate_tts("Hello.", speech_rate=0)

    @patch("services.tts_service.load_audio", return_value=torch.zeros(1, 2048))
    def test_voice_clone_normalizes_prompt_and_generated_text(self, load_audio):
        self.service.generate_voice_clone(
            "I have 12 apples.", "sample.wav", "Sample 2.", language="en"
        )

        self.assertTrue(load_audio.called)
        self.assertEqual(
            self.tokenizer.texts[-1], "sample two. i have twelve apples."
        )
        self.assertIsNotNone(self.model.calls[-1]["prompt_audio"])

    def test_long_preview_segments_spoken_text(self):
        segments = self.service.preview_long_text(
            "I have 12.5 apples. You have 1000 pears.", "en", 3, 5
        )
        combined = " ".join(segment.text for segment in segments)
        self.assertNotIn("12.5", combined)
        self.assertIn("twelve point five", combined)
        self.assertIn("one thousand", combined)

    @patch("services.tts_service.stitch_audio_files")
    @patch("services.tts_service.load_audio", return_value=torch.zeros(1, 2048))
    def test_long_audio_reuses_clone_inputs_and_cleans_wavs(self, load_audio, stitch):
        def create_mp3(files, boundaries, output, **kwargs):
            del boundaries, kwargs
            self.assertGreaterEqual(len(files), 2)
            Path(output).write_bytes(b"mp3")
            return Path(output)

        stitch.side_effect = create_mp3
        with patch.object(
            self.service,
            "generate_voice_clone",
            return_value=(24000, np.zeros(16, dtype=np.float32)),
        ) as generate:
            segments, output = self.service.generate_long_audio(
                "First sentence. Second sentence.",
                "en",
                "sample.wav",
                "Sample transcript.",
                target_seconds=2,
                max_seconds=3,
            )

        self.assertEqual(len(segments), generate.call_count)
        self.assertTrue(output.is_file())
        self.assertFalse(list(output.parent.glob("segment_*.wav")))
        self.assertTrue(load_audio.called)
        for call in generate.call_args_list:
            self.assertEqual(call.args[1], "sample.wav")
            self.assertEqual(call.args[2], "Sample transcript.")
            self.assertEqual(call.kwargs["speech_rate"], 1.3)

    def test_long_audio_requires_voice_clone_inputs(self):
        with self.assertRaisesRegex(ValueError, "prompt audio"):
            self.service.generate_long_audio(
                "First sentence.", "en", "", "", 15, 20
            )


if __name__ == "__main__":
    unittest.main()
