"""Word Error Rate (WER) evaluation utilities."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import jiwer

logger = logging.getLogger(__name__)


@dataclass
class WERResult:
    """WER evaluation result for a single or batch comparison."""

    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    num_reference_words: int
    num_hypothesis_words: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class WordErrorRate:
    """
    Compute Word Error Rate using jiwer.

    Supports single-pair, batch, and dataset-level evaluation with reporting.
    """

    def __init__(self, normalize: bool = True) -> None:
        """
        Initialize WER evaluator.

        Args:
            normalize: Apply text normalization before comparison.
        """
        self.normalize = normalize
        self._transform = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
            ]
        )

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for fair WER comparison.

        Args:
            text: Raw transcription text.

        Returns:
            Normalized text.
        """
        if not self.normalize:
            return text.strip()
        return self._transform(text)

    def compute(self, reference: str, hypothesis: str) -> WERResult:
        """
        Compute WER for a single reference-hypothesis pair.

        Args:
            reference: Ground truth transcription.
            hypothesis: Model prediction.

        Returns:
            WERResult with detailed alignment stats.
        """
        ref = self.normalize_text(reference)
        hyp = self.normalize_text(hypothesis)

        if not ref.split():
            wer = 0.0 if not hyp.split() else 1.0
            return WERResult(
                wer=wer,
                substitutions=0,
                deletions=0,
                insertions=len(hyp.split()),
                hits=0,
                num_reference_words=0,
                num_hypothesis_words=len(hyp.split()),
            )

        output = jiwer.process_words(ref, hyp)
        return WERResult(
            wer=output.wer,
            substitutions=output.substitutions,
            deletions=output.deletions,
            insertions=output.insertions,
            hits=output.hits,
            num_reference_words=len(ref.split()),
            num_hypothesis_words=len(hyp.split()),
        )

    def compute_batch(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> dict[str, float]:
        """
        Compute aggregate WER over a batch.

        Args:
            references: List of ground truth strings.
            hypotheses: List of predictions.

        Returns:
            Dict with wer and component error rates.
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Length mismatch: {len(references)} refs vs {len(hypotheses)} hyps"
            )

        refs = [self.normalize_text(r) for r in references]
        hyps = [self.normalize_text(h) for h in hypotheses]

        if not refs:
            return {"wer": 0.0, "mer": 0.0, "wil": 0.0, "wip": 0.0}

        output = jiwer.process_words(refs, hyps)
        return {
            "wer": output.wer,
            "mer": output.mer,
            "wil": output.wil,
            "wip": output.wip,
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
        Full dataset-level WER report with per-sample breakdown.

        Args:
            references: All reference transcriptions.
            hypotheses: All model predictions.

        Returns:
            Report dict with aggregate and per-sample metrics.
        """
        per_sample = []
        for ref, hyp in zip(references, hypotheses, strict=True):
            result = self.compute(ref, hyp)
            per_sample.append(
                {
                    "reference": ref,
                    "hypothesis": hyp,
                    "wer": result.wer,
                    **result.to_dict(),
                }
            )

        aggregate = self.compute_batch(references, hypotheses)
        report = {
            "aggregate": aggregate,
            "num_samples": len(references),
            "per_sample": per_sample,
        }
        logger.info(
            "Dataset WER: %.2f%% (%d samples)",
            aggregate["wer"] * 100,
            len(references),
        )
        return report
# empty reference edge case returns 0.0 WER when hypothesis is also empty
# jiwer.process_words called once on full batch for O(n) instead of O(n^2)
