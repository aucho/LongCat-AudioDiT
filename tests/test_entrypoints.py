import unittest

import batch_inference
import api_app
import gradio_app
import inference
from app_config import (
    DEFAULT_MAX_PENDING_TASKS,
    DEFAULT_MAX_SEGMENT_SECONDS,
    DEFAULT_SPEECH_RATE,
    DEFAULT_TARGET_SEGMENT_SECONDS,
    MAX_GENERATION_SECONDS,
    SEGMENT_SLIDER_MAX_SECONDS,
)
from services import DEFAULT_MODEL_DIR


class EntrypointTest(unittest.TestCase):
    def test_gradio_defaults_without_loading_model(self):
        args = gradio_app.parse_args([])
        self.assertEqual(args.model_dir, DEFAULT_MODEL_DIR)
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 7860)
        self.assertEqual(MAX_GENERATION_SECONDS, 180)
        self.assertEqual(SEGMENT_SLIDER_MAX_SECONDS, 180)
        self.assertEqual(DEFAULT_TARGET_SEGMENT_SECONDS, 50)
        self.assertEqual(DEFAULT_MAX_SEGMENT_SECONDS, 60)

    def test_cli_language_defaults(self):
        inference_args = inference.parse_args(
            ["--text", "hello", "--output_audio", "out.wav"]
        )
        batch_args = batch_inference.parse_args(
            ["--lst", "items.lst", "--output_dir", "out"]
        )
        self.assertEqual(inference_args.language, "en")
        self.assertEqual(batch_args.language, "en")
        self.assertEqual(inference_args.model_dir, DEFAULT_MODEL_DIR)
        self.assertEqual(inference_args.speech_rate, DEFAULT_SPEECH_RATE)
        self.assertEqual(batch_args.speech_rate, DEFAULT_SPEECH_RATE)

    def test_api_cli_defaults_without_loading_model(self):
        args = api_app.build_parser().parse_args([])
        self.assertEqual(args.model_dir, DEFAULT_MODEL_DIR)
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 7861)
        self.assertEqual(args.result_ttl_hours, 24)
        self.assertEqual(args.max_pending_tasks, DEFAULT_MAX_PENDING_TASKS)


if __name__ == "__main__":
    unittest.main()
