"""Language-specific evaluation metrics for Ethiopic ASR."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# Ethiopic Unicode block ranges
_ETHIOPIC_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB01-\uAB2F]+")

# Amharic labialized consonant clusters (fidel pairs)
_LABIALIZED = re.compile(r"[\u1200-\u12BF][\u12C0-\u12CF]")


@dataclass
class ScriptStats:
    """Unicode script composition of a text sample."""

    total_chars: int
    ethiopic_chars: int
    latin_chars: int
    digit_chars: int
    punctuation_chars: int
    other_chars: int
    primary_script: str

    @property
    def ethiopic_ratio(self) -> float:
        if self.total_chars == 0:
            return 0.0
        return self.ethiopic_chars / self.total_chars


def analyze_script(text: str) -> ScriptStats:
    """
    Analyze Unicode script composition of text.

    Args:
        text: Input string.

    Returns:
        ScriptStats with character-level breakdown.
    """
    total = len(text)
    ethiopic = sum(1 for c in text if "\u1200" <= c <= "\u137F" or "\u1380" <= c <= "\u139F")
    latin = sum(1 for c in text if c.isalpha() and ord(c) < 0x0250)
    digit = sum(1 for c in text if c.isdigit())
    punct = sum(1 for c in text if unicodedata.category(c).startswith("P"))
    other = total - ethiopic - latin - digit - punct

    primary = "ethiopic" if ethiopic >= latin else ("latin" if latin > 0 else "unknown")

    return ScriptStats(
        total_chars=total,
        ethiopic_chars=ethiopic,
        latin_chars=latin,
        digit_chars=digit,
        punctuation_chars=punct,
        other_chars=other,
        primary_script=primary,
    )


def token_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute token-level error rate for Ethiopic text.

    For Ethiopic script, tokens are separated by Ethiopic word separator (፡)
    and whitespace, rather than only whitespace.

    Args:
        reference: Ground truth transcription.
        hypothesis: Model prediction.

    Returns:
        Token error rate in [0, 1].
    """
    # Split on whitespace and Ethiopic separator
    ref_tokens = re.split(r"[\s\u1361]+", reference.strip())
    hyp_tokens = re.split(r"[\s\u1361]+", hypothesis.strip())

    ref_tokens = [t for t in ref_tokens if t]
    hyp_tokens = [t for t in hyp_tokens if t]

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    # Levenshtein distance on token sequences
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)

    return dp[m] / n


def amharic_character_confusion(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, dict[str, int]]:
    """
    Build a character confusion matrix for Amharic transcriptions.

    Tracks which Ethiopic characters are most often substituted for each other,
    providing insights for targeted model improvement.

    Args:
        references: Ground truth strings.
        hypotheses: Model prediction strings.

    Returns:
        Nested dict: confusion[ref_char][hyp_char] = count.
    """
    confusion: dict[str, dict[str, int]] = {}

    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        min_len = min(len(ref_chars), len(hyp_chars))
        for r_c, h_c in zip(ref_chars[:min_len], hyp_chars[:min_len]):
            if r_c != h_c and "\u1200" <= r_c <= "\u137F":
                confusion.setdefault(r_c, {})
                confusion[r_c][h_c] = confusion[r_c].get(h_c, 0) + 1

    return confusion


def compute_mer(references: list[str], hypotheses: list[str]) -> float:
    """
    Compute Match Error Rate (MER) for a batch.

    MER = (S + D + I) / (S + D + I + H) where H = hits.

    Args:
        references: Reference texts.
        hypotheses: Hypothesis texts.

    Returns:
        MER in [0, 1].
    """
    import jiwer

    output = jiwer.process_words(references, hypotheses)
    total = output.substitutions + output.deletions + output.insertions + output.hits
    if total == 0:
        return 0.0
    return (output.substitutions + output.deletions + output.insertions) / total


def syllable_error_rate(reference: str, hypothesis: str, language: str = "am") -> float:
    """
    Approximate syllable error rate for Ethiopic script.

    In Ge'ez/fidel, each character is a consonant-vowel unit (syllable),
    making character error rate equivalent to syllable error rate for Ethiopic.
    For Latin-script Oromo, approximate syllables using vowel-boundary splitting.

    Args:
        reference: Ground truth.
        hypothesis: Prediction.
        language: ISO 639-1 code.

    Returns:
        Syllable error rate in [0, 1].
    """
    if language in {"am", "ti"}:
        # Ethiopic: each fidel character is one syllable unit
        ref_units = [c for c in reference if "\u1200" <= c <= "\u137F"]
        hyp_units = [c for c in hypothesis if "\u1200" <= c <= "\u137F"]
    else:
        # Latin: split on vowel boundaries
        vowels = set("aeiouAEIOU")
        ref_units = re.findall(r"[^aeiouAEIOU\s]*[aeiouAEIOU]", reference)
        hyp_units = re.findall(r"[^aeiouAEIOU\s]*[aeiouAEIOU]", hypothesis)

    if not ref_units:
        return 0.0 if not hyp_units else 1.0

    n, m = len(ref_units), len(hyp_units)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if ref_units[i - 1] == hyp_units[j - 1] else 1
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)

    return dp[m] / n


def bootstrap_confidence_interval(
    scores: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a list of metric scores.

    Args:
        scores: Per-sample metric values.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95%).
        seed: Random seed.

    Returns:
        (lower_bound, upper_bound) of the confidence interval.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = np.array(scores)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)]
    alpha = (1 - confidence) / 2
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1 - alpha))
    return lo, hi
# bootstrap CI operates on normalized per-sample WER/CER scores
# np.quantile used for bootstrap CI percentile computation
# token_error_rate splits on whitespace and U+1361 Ethiopic word separator
# numeric-only Ethiopic strings handled via fallback to empty unit list
