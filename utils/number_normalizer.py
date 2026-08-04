"""English and Spanish number verbalization for TTS input."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from num2words import num2words


SUPPORTED_LANGUAGES = {"en", "es"}


@dataclass(frozen=True)
class NormalizedText:
    original_text: str
    spoken_text: str
    replacements: tuple[tuple[str, str], ...]


_UNIT_FORMS = {
    "en": {
        "in": ("inch", "inches"),
        "inch": ("inch", "inches"),
        "inches": ("inch", "inches"),
        "mm": ("millimeter", "millimeters"),
        "cm": ("centimeter", "centimeters"),
        "m": ("meter", "meters"),
        "km": ("kilometer", "kilometers"),
        "g": ("gram", "grams"),
        "kg": ("kilogram", "kilograms"),
        "lb": ("pound", "pounds"),
        "lbs": ("pound", "pounds"),
        "oz": ("ounce", "ounces"),
        "°c": ("degree celsius", "degrees celsius"),
        "°f": ("degree fahrenheit", "degrees fahrenheit"),
    },
    "es": {
        "in": ("pulgada", "pulgadas"),
        "inch": ("pulgada", "pulgadas"),
        "inches": ("pulgada", "pulgadas"),
        "mm": ("milímetro", "milímetros"),
        "cm": ("centímetro", "centímetros"),
        "m": ("metro", "metros"),
        "km": ("kilómetro", "kilómetros"),
        "g": ("gramo", "gramos"),
        "kg": ("kilogramo", "kilogramos"),
        "lb": ("libra", "libras"),
        "lbs": ("libra", "libras"),
        "oz": ("onza", "onzas"),
        "°c": ("grado Celsius", "grados Celsius"),
        "°f": ("grado Fahrenheit", "grados Fahrenheit"),
    },
}

_NUMBER_BODY = r"(?:\d{1,3}(?:[,.]\d{3})+(?:[,.]\d+)?|\d+(?:[,.]\d+)?)"
_UNIT_PATTERN = re.compile(
    rf"(?<![\w.])(?P<number>[+-]?{_NUMBER_BODY})"
    r"(?P<separator>\s*-\s*|\s*)"
    r"(?P<unit>inch(?:es)?|in|mm|cm|km|kg|lbs?|oz|°\s*[cf]|m|g)(?!\w)",
    re.IGNORECASE,
)
_PERCENT_PATTERN = re.compile(
    rf"(?<![\w.])(?P<number>[+-]?{_NUMBER_BODY})\s*%(?!\w)"
)
_EN_ORDINAL_PATTERN = re.compile(r"(?<!\w)(?P<number>\d+)(?:st|nd|rd|th)(?!\w)", re.I)
_ES_ORDINAL_PATTERN = re.compile(
    r"(?<!\w)(?P<number>\d+)\s*(?:\.\s*)?(?:º|ª|°)(?!\w)", re.I
)
_DASH_PATTERN = re.compile(r"(?<![\w/])(?P<left>\d+)\s*-\s*(?P<right>\d+)(?![\w/-])")
_PLAIN_NUMBER_PATTERN = re.compile(rf"(?<![\w.])(?P<number>[+-]?{_NUMBER_BODY})(?!\w)")

_PROTECTED_PATTERNS = (
    r"https?://[^\s]+|www\.[^\s]+",
    r"[\w.+-]+@[\w.-]+\.\w+",
    r"(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)",
    r"(?<!\w)\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?(?!\w)",
    r"(?<!\w)(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})(?!\w)",
    r"(?<!\w)v?\d+(?:\.\d+){2,}(?!\w)",
    r"(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)",
    r"(?<!\w)(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+(?!\w)",
)


def _parse_number(value: str, language: str) -> tuple[str, str, Decimal]:
    """Return integer digits, fractional digits, and numeric value."""
    sign = ""
    if value[:1] in {"+", "-"}:
        sign, value = value[0], value[1:]

    decimal_separator: str | None = None
    if "," in value and "." in value:
        decimal_separator = "," if value.rfind(",") > value.rfind(".") else "."
    elif "," in value:
        groups = value.split(",")
        if language == "es" or not (len(groups) > 1 and all(len(group) == 3 for group in groups[1:])):
            decimal_separator = ","
    elif "." in value:
        groups = value.split(".")
        if language == "en" or not (len(groups) > 1 and all(len(group) == 3 for group in groups[1:])):
            decimal_separator = "."

    if decimal_separator:
        integer_part, fractional_part = value.rsplit(decimal_separator, 1)
        grouping_separator = "," if decimal_separator == "." else "."
        integer_digits = integer_part.replace(grouping_separator, "")
    else:
        integer_digits = value.replace(",", "").replace(".", "")
        fractional_part = ""

    integer_digits = integer_digits or "0"
    canonical = f"{sign}{integer_digits}"
    if fractional_part:
        canonical += f".{fractional_part}"
    try:
        numeric = Decimal(canonical)
    except InvalidOperation as error:
        raise ValueError(f"invalid numeric value: {value}") from error
    return f"{sign}{integer_digits}", fractional_part, numeric


def _verbalize_number(value: str, language: str) -> str:
    integer_digits, fractional_digits, _ = _parse_number(value, language)
    integer_words = num2words(int(integer_digits), lang=language)
    if not fractional_digits:
        return integer_words
    decimal_word = "point" if language == "en" else "coma"
    digits = " ".join(num2words(int(digit), lang=language) for digit in fractional_digits)
    return f"{integer_words} {decimal_word} {digits}"


def _placeholder(index: int) -> str:
    # Private-use characters contain no digits, so later numeric regexes ignore them.
    return f"\ue000{chr(0xE100 + index)}\ue001"


def normalize_tts_text(text: str, language: str) -> NormalizedText:
    """Convert common numeric expressions to stable English/Spanish speech text."""
    language = (language or "").lower().strip()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("language must be 'en' or 'es'")
    original = text or ""
    working = original
    protected: dict[str, str] = {}
    replacements: list[tuple[str, str]] = []

    def protect(match: re.Match[str]) -> str:
        token = _placeholder(len(protected))
        protected[token] = match.group(0)
        return token

    for pattern in _PROTECTED_PATTERNS[:2]:
        working = re.sub(pattern, protect, working, flags=re.IGNORECASE)

    def replace_unit(match: re.Match[str]) -> str:
        source = match.group(0)
        number = match.group("number")
        unit_key = re.sub(r"\s+", "", match.group("unit").lower())
        words = _verbalize_number(number, language)
        _, _, numeric = _parse_number(number, language)
        singular, plural = _UNIT_FORMS[language][unit_key]
        # Hyphenated forms are attributive ("10.2-inch screen") and stay singular.
        unit_words = (
            singular
            if "-" in match.group("separator") or abs(numeric) == 1
            else plural
        )
        result = f"{words} {unit_words}"
        replacements.append((source, result))
        return result

    working = _UNIT_PATTERN.sub(replace_unit, working)

    for pattern in _PROTECTED_PATTERNS[2:]:
        working = re.sub(pattern, protect, working, flags=re.IGNORECASE)

    def replace_dash(match: re.Match[str]) -> str:
        source = match.group(0)
        dash = "dash" if language == "en" else "guion"
        result = (
            f"{_verbalize_number(match.group('left'), language)} {dash} "
            f"{_verbalize_number(match.group('right'), language)}"
        )
        replacements.append((source, result))
        return result

    working = _DASH_PATTERN.sub(replace_dash, working)

    def replace_percent(match: re.Match[str]) -> str:
        source = match.group(0)
        suffix = "percent" if language == "en" else "por ciento"
        result = f"{_verbalize_number(match.group('number'), language)} {suffix}"
        replacements.append((source, result))
        return result

    working = _PERCENT_PATTERN.sub(replace_percent, working)

    ordinal_pattern = _EN_ORDINAL_PATTERN if language == "en" else _ES_ORDINAL_PATTERN

    def replace_ordinal(match: re.Match[str]) -> str:
        source = match.group(0)
        result = num2words(int(match.group("number")), lang=language, to="ordinal")
        replacements.append((source, result))
        return result

    working = ordinal_pattern.sub(replace_ordinal, working)

    def replace_plain(match: re.Match[str]) -> str:
        source = match.group("number")
        result = _verbalize_number(source, language)
        replacements.append((source, result))
        return result

    working = _PLAIN_NUMBER_PATTERN.sub(replace_plain, working)
    for token, value in protected.items():
        working = working.replace(token, value)
    return NormalizedText(original, working, tuple(replacements))


__all__ = ["NormalizedText", "SUPPORTED_LANGUAGES", "normalize_tts_text"]
