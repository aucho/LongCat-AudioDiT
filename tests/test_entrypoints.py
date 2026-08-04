import unittest

import batch_inference
import gradio_app
import inference
from services import DEFAULT_MODEL_DIR


class EntrypointTest(unittest.TestCase):
    def test_gradio_defaults_without_loading_model(self):
        args = gradio_app.parse_args([])
        self.assertEqual(args.model_dir, DEFAULT_MODEL_DIR)
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 7860)

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
        self.assertEqual(inference_args.speech_rate, 1.3)
        self.assertEqual(batch_args.speech_rate, 1.3)


if __name__ == "__main__":
    unittest.main()
