import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.long_audio import segment_text, stitch_audio_files


class SegmentTextTest(unittest.TestCase):
    def test_english_abbreviations_url_and_decimal_are_not_split(self):
        text = (
            "Dr. Smith reported 3.14 points at https://example.com/test. "
            "The second sentence is complete."
        )
        segments = segment_text(text, "en", target_seconds=5, max_seconds=20)
        combined = " ".join(segment.text for segment in segments)

        self.assertIn("Dr. Smith", combined)
        self.assertIn("3.14", combined)
        self.assertIn("https://example.com/test", combined)
        self.assertEqual(combined, text)
        self.assertEqual(len(segments), 2)

    def test_spanish_abbreviation_and_question_remain_intact(self):
        text = "La Dra. Ruiz llegó temprano. ¿Cómo está usted? Todo está bien."
        segments = segment_text(text, "es", target_seconds=3, max_seconds=8)
        combined = " ".join(segment.text for segment in segments)

        self.assertEqual(combined, text)
        self.assertIn("La Dra. Ruiz llegó temprano.", [item.text for item in segments])

    def test_long_unpunctuated_text_falls_back_to_word_boundaries(self):
        text = " ".join(["continuous"] * 80)
        segments = segment_text(text, "en", target_seconds=3, max_seconds=4)

        self.assertGreater(len(segments), 1)
        self.assertTrue(all(item.estimated_seconds <= 4 for item in segments))
        self.assertEqual(" ".join(item.text for item in segments), text)

    def test_paragraph_boundary_is_preserved(self):
        segments = segment_text("First paragraph.\n\nSecond paragraph.", "en")

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].boundary, "paragraph")


class StitchAudioTest(unittest.TestCase):
    def test_stitch_uses_ordered_concat_and_mp3_settings(self):
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            first = temp / "first.wav"
            second = temp / "second.wav"
            output = temp / "output.mp3"
            sf.write(first, np.ones(240, dtype=np.float32) * 0.1, 24000)
            sf.write(second, np.ones(240, dtype=np.float32) * 0.2, 24000)

            def fake_run(command, cwd, **kwargs):
                del kwargs
                concat_path = Path(command[command.index("-i") + 1])
                lines = concat_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(lines, ["file '000000.wav'", "file '000001.wav'"])
                self.assertEqual(Path(cwd), concat_path.parent)
                first_processed, first_rate = sf.read(
                    Path(cwd) / "000000.wav", dtype="float32"
                )
                # 240 source samples plus the 0.12-second sentence pause.
                self.assertEqual(first_rate, 24000)
                self.assertEqual(len(first_processed), 240 + int(0.12 * 24000))
                Path(command[-1]).write_bytes(b"mp3")
                return subprocess.CompletedProcess(command, 0)

            with patch("utils.long_audio.shutil.which", return_value="ffmpeg"), patch(
                "utils.long_audio.subprocess.run", side_effect=fake_run
            ) as run:
                result = stitch_audio_files(
                    [first, second], ["sentence", "paragraph"], output
                )

            command = run.call_args.args[0]
            self.assertEqual(result, output.resolve())
            self.assertEqual(command[command.index("-ar") + 1], "24000")
            self.assertEqual(command[command.index("-ac") + 1], "1")
            self.assertEqual(command[command.index("-b:a") + 1], "192k")


if __name__ == "__main__":
    unittest.main()
