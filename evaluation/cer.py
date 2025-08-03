"""Character Error Rate (CER) evaluation with language-specific support."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

import jiwer

logger = logging.getLogger(__name__)

# Ethiopic script range for Amharic, Afaan Oromo (when written in Ge'ez), Tigrinya, etc.
ETHIOPIC_PATTERN = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]")


@dataclass
class CERResult:
    """CER evaluation result."""

    cer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    num_reference_chars: int
    num_hypothesis_chars: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CharacterErrorRate:
    """
    Compute Character Error Rate with language-specific normalization.

    Optimized for Ethiopic-script languages (Amharic) and Latin-script
    Afaan Oromo transcriptions.
    """

    LANGUAGE_NORMALIZERS: dict[str, str] = {
        "am": "ethiopic",
        "om": "latin",
        "ti": "ethiopic",
        "en": "latin",
    }

    def __init__(
        self,
        language: str = "am",
        remove_spaces: bool = False,
        normalize_unicode: bool = True,
    ) -> None:
        """
        Initialize CER evaluator.

        Args:
            language: ISO 639-1 language code.
            remove_spaces: Exclude spaces from CER computation.
            normalize_unicode: Apply NFC normalization.
        """
        self.language = language
        self.remove_spaces = remove_spaces
        self.normalize_unicode = normalize_unicode
        self.script = self.LANGUAGE_NORMALIZERS.get(language, "latin")

    def normalize_text(self, text: str) -> str:
        """
        Language-aware text normalization for CER.

        Args:
            text: Raw text.

        Returns:
            Normalized character sequence.
        """
        text = text.strip()

        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        if self.script == "ethiopic":
            text = text.replace("\u2019", "'").replace("\u2018", "'")
            text = re.sub(r"[^\w\s\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF'-]", "", text)
        else:
            text = text.lower()
            text = re.sub(r"[^\w\s'-]", "", text)

        text = re.sub(r"\s+", " ", text)

        if self.remove_spaces:
            text = text.replace(" ", "")

        return text

    def compute(self, reference: str, hypothesis: str) -> CERResult:
        """
        Compute CER for a single pair.

        Args:
            reference: Ground truth.
            hypothesis: Prediction.

        Returns:
            CERResult with alignment statistics.
        """
        ref = self.normalize_text(reference)
        hyp = self.normalize_text(hypothesis)

        if not ref:
            cer = 0.0 if not hyp else 1.0
            return CERResult(
                cer=cer,
                substitutions=0,
                deletions=0,
                insertions=len(hyp),
                hits=0,
                num_reference_chars=0,
                num_hypothesis_chars=len(hyp),
            )

        output = jiwer.process_characters(ref, hyp)
        cer_value = getattr(output, "cer", getattr(output, "wer", 0.0))
        return CERResult(
            cer=cer_value,
            substitutions=output.substitutions,
            deletions=output.deletions,
            insertions=output.insertions,
            hits=output.hits,
            num_reference_chars=len(ref),
            num_hypothesis_chars=len(hyp),
        )

    def compute_batch(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> dict[str, float]:
        """
        Compute aggregate CER over a batch.

        Args:
            references: Ground truth list.
            hypotheses: Prediction list.

        Returns:
            Dict with cer and error counts.
        """
        if len(references) != len(hypotheses):
            raise ValueError("Reference and hypothesis lists must have equal length")

        refs = [self.normalize_text(r) for r in references]
        hyps = [self.normalize_text(h) for h in hypotheses]

        if not refs:
            return {"cer": 0.0}

        output = jiwer.process_characters(refs, hyps)
        cer_value = getattr(output, "cer", getattr(output, "wer", 0.0))
        return {
            "cer": cer_value,
            "substitutions": float(output.substitutions),
            "deletions": float(output.deletions),
            "insertions": float(output.insertions),
            "hits": float(output.hits),
        }

    def evaluate_dataset(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> dict[str, Any]:
        """
        Dataset-level CER report.

        Args:
            references: All references.
            hypotheses: All predictions.

        Returns:
            Report with aggregate and per-sample CER.
        """
        per_sample = []
        for ref, hyp in zip(references, hypotheses, strict=True):
            result = self.compute(ref, hyp)
            per_sample.append(
                {
                    "reference": ref,
                    "hypothesis": hyp,
                    "cer": result.cer,
                    **result.to_dict(),
                }
            )

        aggregate = self.compute_batch(references, hypotheses)
        report = {
            "language": self.language,
            "script": self.script,
            "aggregate": aggregate,
            "num_samples": len(references),
            "per_sample": per_sample,
        }
        logger.info(
            "Dataset CER [%s]: %.2f%% (%d samples)",
            self.language,
            aggregate["cer"] * 100,
            len(references),
        )
        return report

    @staticmethod
    def detect_script(text: str) -> str:
        """
        Detect whether text is primarily Ethiopic or Latin script.

        Args:
            text: Input text.

        Returns:
            'ethiopic' or 'latin'.
        """
        ethiopic_chars = len(ETHIOPIC_PATTERN.findall(text))
        latin_chars = len(re.findall(r"[a-zA-Z]", text))
        return "ethiopic" if ethiopic_chars >= latin_chars else "latin"
# empty hypothesis returns cer=1.0 when reference is non-empty
