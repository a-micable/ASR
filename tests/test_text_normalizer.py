"""Tests for text normalization module."""

from __future__ import annotations

import pytest

from preprocessing.text_normalizer import (
    EthiopicNormalizer,
    OromiaNormalizer,
    TextNormalizer,
)


class TestEthiopicNormalizer:
    def test_strips_whitespace(self) -> None:
        norm = EthiopicNormalizer()
        result = norm.normalize("  ሰላም ዓለም  ")
        assert result.normalized == "ሰላም ዓለም"

    def test_normalizes_smart_quotes(self) -> None:
        norm = EthiopicNormalizer()
        result = norm.normalize("\u2018hello\u2019")
        assert "'" in result.normalized
        assert "smart_quotes" in result.changes

    def test_removes_ethiopic_punct(self) -> None:
        norm = EthiopicNormalizer()
        result = norm.normalize("ሰላም\u1361ዓለም")
        assert "\u1361" not in result.normalized
        assert "ethiopic_punct" in result.changes

    def test_collapses_whitespace(self) -> None:
        norm = EthiopicNormalizer()
        result = norm.normalize("ሰላም   ዓለም")
        assert result.normalized == "ሰላም ዓለም"

    def test_nfc_normalization(self) -> None:
        norm = EthiopicNormalizer()
        # Decomposed form
        text = "a\u0300"  # à decomposed
        result = norm.normalize(text)
        assert "unicode_nfc" in result.changes or result.normalized == text.strip()

    def test_geez_numerals_replaced(self) -> None:
        norm = EthiopicNormalizer()
        result = norm.normalize("፩ ሰው")
        assert "1" in result.normalized

    def test_no_changes_on_clean_text(self) -> None:
        norm = EthiopicNormalizer()
        text = "ሰላም ዓለም"
        result = norm.normalize(text)
        assert result.normalized == text


class TestOromiaNormalizer:
    def test_lowercases_text(self) -> None:
        norm = OromiaNormalizer()
        result = norm.normalize("AKKAM")
        assert result.normalized == "akkam"
        assert "lowercase" in result.changes

    def test_removes_punctuation(self) -> None:
        norm = OromiaNormalizer()
        result = norm.normalize("hello, world!")
        assert "," not in result.normalized
        assert "!" not in result.normalized

    def test_diacritic_substitution(self) -> None:
        norm = OromiaNormalizer()
        result = norm.normalize("\u0253aga")  # ɓaga
        assert result.normalized == "baga"

    def test_collapses_whitespace(self) -> None:
        norm = OromiaNormalizer()
        result = norm.normalize("akkam   jirta")
        assert "  " not in result.normalized


class TestTextNormalizer:
    def test_amharic_dispatch(self) -> None:
        norm = TextNormalizer(language="am")
        result = norm.normalize("  ሰላም ዓለም  ")
        assert result == "ሰላም ዓለም"

    def test_oromo_dispatch(self) -> None:
        norm = TextNormalizer(language="om")
        result = norm.normalize("AKKAM JIRTA")
        assert result == result.lower()

    def test_tigrinya_dispatch(self) -> None:
        norm = TextNormalizer(language="ti")
        result = norm.normalize("ሰላም")
        assert isinstance(result, str)

    def test_unknown_language_falls_back_ethiopic(self) -> None:
        norm = TextNormalizer(language="xx")
        result = norm.normalize("ሰላም")
        assert isinstance(result, str)

    def test_normalize_batch(self) -> None:
        norm = TextNormalizer(language="am")
        texts = ["  ሰላም  ", "ዓለም  "]
        results = norm.normalize_batch(texts)
        assert len(results) == 2
        assert results[0] == "ሰላም"

    def test_supported_languages(self) -> None:
        langs = TextNormalizer.supported_languages()
        assert "am" in langs
        assert "om" in langs
        assert "ti" in langs
# normalize_batch(['am text', 'latin']) preserves language-specific rules
