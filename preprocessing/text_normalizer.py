"""Text normalization for Ethiopic and Latin-script low-resource languages."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class NormalizationResult:
    """Result of text normalization."""

    original: str
    normalized: str
    changes: list[str]


# Amharic/Ethiopic punctuation and common Unicode noise
_ETHIOPIC_PUNCT = re.compile(r"[\u1361-\u1368\u1360]")  # Ethiopic word separators
_ARABIC_NUMERALS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_ETHIOPIC_DIGITS = str.maketrans("፩፪፫፬፭፮፯፰፱፲", "1234567890")

# Ge'ez numeral words (common in Amharic text)
_GEEZ_NUMBERS: dict[str, str] = {
    "፩": "1", "፪": "2", "፫": "3", "፬": "4", "፭": "5",
    "፮": "6", "፯": "7", "፰": "8", "፱": "9", "፲": "10",
    "፳": "20", "፴": "30", "፵": "40", "፶": "50",
    "፷": "60", "፸": "70", "፹": "80", "፺": "90", "፻": "100",
}


class EthiopicNormalizer:
    """
    Normalize Amharic (Ethiopic) transcriptions for ASR training.

    Handles:
    - Unicode NFC normalization
    - Ethiopic punctuation removal
    - Ge'ez numeral transliteration
    - Duplicate whitespace removal
    - Smart quote normalization
    """

    def normalize(self, text: str) -> NormalizationResult:
        """
        Normalize Amharic text.

        Args:
            text: Raw Amharic transcription.

        Returns:
            NormalizationResult with normalized text and change log.
        """
        original = text
        changes: list[str] = []

        # NFC Unicode normalization
        nfc = unicodedata.normalize("NFC", text)
        if nfc != text:
            changes.append("unicode_nfc")
        text = nfc

        # Smart quotes
        q_replaced = text.replace("\u2019", "'").replace("\u2018", "'")
        if q_replaced != text:
            changes.append("smart_quotes")
        text = q_replaced

        # Ge'ez numerals
        for geez, arabic in _GEEZ_NUMBERS.items():
            if geez in text:
                text = text.replace(geez, arabic)
                changes.append("geez_numerals")
                break

        # Ethiopic sentence-end punctuation -> space
        punct_removed = _ETHIOPIC_PUNCT.sub(" ", text)
        if punct_removed != text:
            changes.append("ethiopic_punct")
        text = punct_removed

        # Remove ASCII punctuation (keep apostrophes)
        clean = re.sub(r'[!"#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~]', " ", text)
        if clean != text:
            changes.append("ascii_punct")
        text = clean

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return NormalizationResult(original=original, normalized=text, changes=changes)


class OromiaNormalizer:
    """
    Normalize Afaan Oromo (Latin-script) transcriptions.

    Handles:
    - Lowercase conversion
    - Oromia-specific diacritics (barred letters)
    - Duplicate whitespace, punctuation stripping
    """

    # Barred letters common in Oromia orthography
    _DIACRITIC_MAP: dict[str, str] = {
        "\u0253": "b",  # ɓ -> b
        "\u0257": "d",  # ɗ -> d
        "\u025f": "j",  # ɟ -> j
        "\u01b4": "y",  # ƴ -> y
        "\u0294": "'",  # ʔ (glottal) -> apostrophe
    }

    def normalize(self, text: str) -> NormalizationResult:
        """
        Normalize Afaan Oromo text.

        Args:
            text: Raw Oromo transcription.

        Returns:
            NormalizationResult.
        """
        original = text
        changes: list[str] = []

        # NFC
        nfc = unicodedata.normalize("NFC", text)
        if nfc != text:
            changes.append("unicode_nfc")
        text = nfc

        # Lowercase
        lower = text.lower()
        if lower != text:
            changes.append("lowercase")
        text = lower

        # Diacritic substitutions
        for char, repl in self._DIACRITIC_MAP.items():
            if char in text:
                text = text.replace(char, repl)
                changes.append(f"diacritic_{char}")

        # Arabic numeral transliteration
        text = text.translate(_ARABIC_NUMERALS)

        # Punctuation removal
        text = re.sub(r'[!"#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~]', " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return NormalizationResult(original=original, normalized=text, changes=changes)


class TextNormalizer:
    """
    Dispatcher for language-aware text normalization.

    Selects the appropriate normalizer based on language code.
    """

    _NORMALIZERS: dict[str, type] = {
        "am": EthiopicNormalizer,
        "ti": EthiopicNormalizer,
        "om": OromiaNormalizer,
    }

    def __init__(self, language: str = "am") -> None:
        """
        Initialize normalizer for the given language.

        Args:
            language: ISO 639-1 language code.
        """
        self.language = language
        normalizer_cls = self._NORMALIZERS.get(language, EthiopicNormalizer)
        self._normalizer = normalizer_cls()

    def normalize(self, text: str) -> str:
        """
        Normalize text using the language-appropriate normalizer.

        Args:
            text: Raw transcription.

        Returns:
            Normalized text string.
        """
        return self._normalizer.normalize(text).normalized

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """
        Normalize a list of transcriptions.

        Args:
            texts: Raw transcriptions.

        Returns:
            List of normalized strings.
        """
        return [self.normalize(t) for t in texts]

    @staticmethod
    def supported_languages() -> list[str]:
        """Return list of explicitly supported language codes."""
        return list(TextNormalizer._NORMALIZERS.keys())
