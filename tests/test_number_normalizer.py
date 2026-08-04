import unittest

from utils import normalize_tts_text


class NumberNormalizerTest(unittest.TestCase):
    def test_english_examples(self):
        cases = {
            "10.2-inch": "ten point two inch",
            "12000": "twelve thousand",
            "12,000": "twelve thousand",
            "12-1": "twelve dash one",
            "1000000": "one million",
            "12.1": "twelve point one",
            "Sample 2.": "Sample two.",
            "I have 12.5 apples.": "I have twelve point five apples.",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(normalize_tts_text(source, "en").spoken_text, expected)

    def test_spanish_numbers_and_separators(self):
        self.assertEqual(
            normalize_tts_text("Tengo 12,50 kg.", "es").spoken_text,
            "Tengo doce coma cinco cero kilogramos.",
        )
        self.assertEqual(
            normalize_tts_text("12-1", "es").spoken_text, "doce guion uno"
        )
        self.assertEqual(
            normalize_tts_text("1.000 metros", "es").spoken_text,
            "mil metros",
        )

    def test_sign_ordinal_percent_and_unit_plural(self):
        result = normalize_tts_text("-2 kg, 21st, 12.50%, 1 inch, 2 inch", "en")
        self.assertEqual(
            result.spoken_text,
            "minus two kilograms, twenty-first, twelve point five zero percent, "
            "one inch, two inches",
        )

    def test_ambiguous_identifiers_are_protected(self):
        text = (
            "https://example.com/12.5 a@b.com 192.168.1.1 12:30 "
            "2026-08-04 v1.2.3 +1 555 123 4567 AB-123"
        )
        self.assertEqual(normalize_tts_text(text, "en").spoken_text, text)

    def test_normalization_is_idempotent(self):
        first = normalize_tts_text("I have 12.50 apples.", "en")
        second = normalize_tts_text(first.spoken_text, "en")
        self.assertEqual(second.spoken_text, first.spoken_text)
        self.assertTrue(first.replacements)
        self.assertFalse(second.replacements)

    def test_rejects_unsupported_language(self):
        with self.assertRaisesRegex(ValueError, "language"):
            normalize_tts_text("12", "fr")


if __name__ == "__main__":
    unittest.main()
