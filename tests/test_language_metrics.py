"""Tests for language-specific evaluation metrics."""

from __future__ import annotations

import pytest

from evaluation.language_metrics import (
    analyze_script,
    amharic_character_confusion,
    bootstrap_confidence_interval,
    compute_mer,
    syllable_error_rate,
    token_error_rate,
)


class TestAnalyzeScript:
    def test_ethiopic_text(self) -> None:
        stats = analyze_script("ሰላም ዓለም")
        assert stats.ethiopic_chars > 0
        assert stats.primary_script == "ethiopic"
        assert stats.ethiopic_ratio > 0.5

    def test_latin_text(self) -> None:
        stats = analyze_script("hello world")
        assert stats.latin_chars > 0
        assert stats.primary_script == "latin"

    def test_mixed_text(self) -> None:
        stats = analyze_script("ሰላም hello")
        assert stats.total_chars > 0

    def test_empty_text(self) -> None:
        stats = analyze_script("")
        assert stats.total_chars == 0
        assert stats.ethiopic_ratio == 0.0

    def test_digit_count(self) -> None:
        stats = analyze_script("test 123")
        assert stats.digit_chars == 3


class TestTokenErrorRate:
    def test_perfect_match(self) -> None:
        assert token_error_rate("ሰላም ዓለም", "ሰላም ዓለም") == 0.0

    def test_full_mismatch(self) -> None:
        rate = token_error_rate("ሰላም", "ዓለም")
        assert rate == 1.0

    def test_empty_reference(self) -> None:
        assert token_error_rate("", "anything") == 1.0

    def test_empty_both(self) -> None:
        assert token_error_rate("", "") == 0.0

    def test_partial_match(self) -> None:
        rate = token_error_rate("a b c", "a b d")
        assert 0.0 < rate < 1.0


class TestAmharicCharacterConfusion:
    def test_returns_dict(self) -> None:
        confusion = amharic_character_confusion(
            ["\u1230\u1208\u121d"],
            ["\u1230\u1208\u12d0"],
        )
        assert isinstance(confusion, dict)

    def test_counts_substitutions(self) -> None:
        confusion = amharic_character_confusion(
            ["\u1230\u1208\u121d"],
            ["\u1230\u1208\u12d0"],
        )
        # \u121d was substituted with \u12d0
        assert "\u121d" in confusion
        assert confusion["\u121d"].get("\u12d0", 0) == 1

    def test_no_confusion_on_perfect(self) -> None:
        confusion = amharic_character_confusion(["abc"], ["abc"])
        assert len(confusion) == 0


class TestComputeMER:
    def test_perfect_match(self) -> None:
        mer = compute_mer(["hello world"], ["hello world"])
        assert mer == 0.0

    def test_full_mismatch(self) -> None:
        mer = compute_mer(["hello"], ["world"])
        assert mer > 0.0

    def test_batch(self) -> None:
        mer = compute_mer(
            ["hello world", "test phrase"],
            ["hello earth", "test phrase"],
        )
        assert 0.0 < mer < 1.0


class TestSyllableErrorRate:
    def test_ethiopic_perfect(self) -> None:
        rate = syllable_error_rate("ሰላም", "ሰላም", language="am")
        assert rate == 0.0

    def test_ethiopic_mismatch(self) -> None:
        rate = syllable_error_rate("ሰላም", "ዓለም", language="am")
        assert rate > 0.0

    def test_latin(self) -> None:
        rate = syllable_error_rate("hello", "hello", language="om")
        assert rate >= 0.0

    def test_empty_reference(self) -> None:
        rate = syllable_error_rate("", "test", language="am")
        assert rate == 1.0


class TestBootstrapCI:
    def test_returns_tuple(self) -> None:
        scores = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.12, 0.28, 0.19, 0.21]
        lo, hi = bootstrap_confidence_interval(scores, n_bootstrap=100)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo <= hi

    def test_interval_contains_mean(self) -> None:
        import numpy as np
        scores = [float(x) for x in np.random.default_rng(42).uniform(0.1, 0.3, 50)]
        mean = sum(scores) / len(scores)
        lo, hi = bootstrap_confidence_interval(scores, n_bootstrap=500, seed=42)
        assert lo <= mean <= hi

    def test_single_value(self) -> None:
        lo, hi = bootstrap_confidence_interval([0.5] * 15, n_bootstrap=100)
        assert abs(lo - 0.5) < 0.05
        assert abs(hi - 0.5) < 0.05
