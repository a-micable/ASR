"""HTML and JSON report generation for ASR evaluation results."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Full evaluation report aggregating all metrics."""

    model_name: str
    language: str
    split: str
    timestamp: str
    num_samples: int
    wer: float
    cer: float
    mer: float
    wer_std: float
    cer_std: float
    wer_ci_lower: float
    wer_ci_upper: float
    cer_ci_lower: float
    cer_ci_upper: float
    worst_samples: list[dict[str, Any]] = field(default_factory=list)
    per_speaker: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReportGenerator:
    """
    Generate evaluation reports in JSON and HTML formats.

    Produces human-readable HTML reports with metric summaries,
    worst-performing examples, and per-speaker breakdowns.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """
        Initialize report generator.

        Args:
            output_dir: Directory to write reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_report(
        self,
        references: list[str],
        hypotheses: list[str],
        model_name: str,
        language: str,
        split: str,
        speakers: list[str] | None = None,
        per_sample_wer: list[float] | None = None,
        per_sample_cer: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """
        Build a full evaluation report from references and hypotheses.

        Args:
            references: Ground truth transcriptions.
            hypotheses: Model predictions.
            model_name: Identifier for the model.
            language: ISO 639-1 code.
            split: Dataset split name.
            speakers: Optional per-sample speaker labels.
            per_sample_wer: Pre-computed per-sample WER values.
            per_sample_cer: Pre-computed per-sample CER values.
            metadata: Arbitrary extra metadata.

        Returns:
            EvaluationReport dataclass.
        """
        from evaluation.cer import CharacterErrorRate
        from evaluation.language_metrics import bootstrap_confidence_interval, compute_mer
        from evaluation.wer import WordErrorRate

        wer_eval = WordErrorRate()
        cer_eval = CharacterErrorRate(language=language)

        agg_wer = wer_eval.compute_batch(references, hypotheses)
        agg_cer = cer_eval.compute_batch(references, hypotheses)
        mer = compute_mer(references, hypotheses)

        if per_sample_wer is None:
            per_sample_wer = [wer_eval.compute(r, h).wer for r, h in zip(references, hypotheses)]
        if per_sample_cer is None:
            per_sample_cer = [cer_eval.compute(r, h).cer for r, h in zip(references, hypotheses)]

        wer_arr = np.array(per_sample_wer)
        cer_arr = np.array(per_sample_cer)

        wer_ci = bootstrap_confidence_interval(per_sample_wer) if len(per_sample_wer) > 10 else (0.0, 1.0)
        cer_ci = bootstrap_confidence_interval(per_sample_cer) if len(per_sample_cer) > 10 else (0.0, 1.0)

        # Top-10 worst by WER
        worst_idx = sorted(range(len(per_sample_wer)), key=lambda i: per_sample_wer[i], reverse=True)[:10]
        worst = [
            {
                "index": i,
                "reference": references[i],
                "hypothesis": hypotheses[i],
                "wer": per_sample_wer[i],
                "cer": per_sample_cer[i],
                "speaker": (speakers[i] if speakers else "unknown"),
            }
            for i in worst_idx
        ]

        # Per-speaker breakdown
        per_speaker: dict[str, dict[str, float]] = {}
        if speakers:
            from collections import defaultdict
            spk_refs: dict[str, list[str]] = defaultdict(list)
            spk_hyps: dict[str, list[str]] = defaultdict(list)
            for spk, r, h in zip(speakers, references, hypotheses):
                spk_refs[spk].append(r)
                spk_hyps[spk].append(h)
            for spk in spk_refs:
                spk_wer = wer_eval.compute_batch(spk_refs[spk], spk_hyps[spk])
                spk_cer = cer_eval.compute_batch(spk_refs[spk], spk_hyps[spk])
                per_speaker[spk] = {
                    "wer": spk_wer["wer"],
                    "cer": spk_cer["cer"],
                    "num_samples": len(spk_refs[spk]),
                }

        return EvaluationReport(
            model_name=model_name,
            language=language,
            split=split,
            timestamp=datetime.now(timezone.utc).isoformat(),
            num_samples=len(references),
            wer=agg_wer["wer"],
            cer=agg_cer["cer"],
            mer=mer,
            wer_std=float(wer_arr.std()),
            cer_std=float(cer_arr.std()),
            wer_ci_lower=wer_ci[0],
            wer_ci_upper=wer_ci[1],
            cer_ci_lower=cer_ci[0],
            cer_ci_upper=cer_ci[1],
            worst_samples=worst,
            per_speaker=per_speaker,
            metadata=metadata or {},
        )

    def save_json(self, report: EvaluationReport, filename: str = "evaluation_report.json") -> Path:
        """Save report as JSON."""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Saved JSON report: %s", path)
        return path

    def save_html(self, report: EvaluationReport, filename: str = "evaluation_report.html") -> Path:
        """Save report as an HTML page with inline CSS."""
        path = self.output_dir / filename
        html = self._render_html(report)
        path.write_text(html, encoding="utf-8")
        logger.info("Saved HTML report: %s", path)
        return path

    def _render_html(self, r: EvaluationReport) -> str:
        worst_rows = "\n".join(
            f"<tr><td>{i+1}</td><td>{s['reference']}</td>"
            f"<td>{s['hypothesis']}</td>"
            f"<td>{s['wer']:.3f}</td><td>{s['cer']:.3f}</td>"
            f"<td>{s.get('speaker', '-')}</td></tr>"
            for i, s in enumerate(r.worst_samples)
        )
        spk_rows = "\n".join(
            f"<tr><td>{spk}</td><td>{v['wer']:.3f}</td>"
            f"<td>{v['cer']:.3f}</td><td>{int(v['num_samples'])}</td></tr>"
            for spk, v in r.per_speaker.items()
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ASR Evaluation Report — {r.model_name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
    h1 {{ color: #1a237e; }}
    .metric-card {{ display: inline-block; background: #e8eaf6; border-radius: 8px;
                    padding: 16px 24px; margin: 8px; min-width: 120px; text-align: center; }}
    .metric-value {{ font-size: 2em; font-weight: bold; color: #283593; }}
    .metric-label {{ font-size: 0.85em; color: #555; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th {{ background: #3949ab; color: white; padding: 8px; text-align: left; }}
    td {{ padding: 6px 8px; border-bottom: 1px solid #ddd; }}
    tr:hover {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>ASR Evaluation Report</h1>
  <p><strong>Model:</strong> {r.model_name} &nbsp;|&nbsp;
     <strong>Language:</strong> {r.language} &nbsp;|&nbsp;
     <strong>Split:</strong> {r.split} &nbsp;|&nbsp;
     <strong>Samples:</strong> {r.num_samples}</p>
  <p><em>Generated: {r.timestamp}</em></p>

  <h2>Summary Metrics</h2>
  <div>
    <div class="metric-card">
      <div class="metric-value">{r.wer*100:.1f}%</div>
      <div class="metric-label">WER</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{r.cer*100:.1f}%</div>
      <div class="metric-label">CER</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{r.mer*100:.1f}%</div>
      <div class="metric-label">MER</div>
    </div>
  </div>
  <p>WER 95% CI: [{r.wer_ci_lower*100:.1f}%, {r.wer_ci_upper*100:.1f}%] &nbsp;
     CER 95% CI: [{r.cer_ci_lower*100:.1f}%, {r.cer_ci_upper*100:.1f}%]</p>

  <h2>Worst Performing Samples</h2>
  <table>
    <tr><th>#</th><th>Reference</th><th>Hypothesis</th><th>WER</th><th>CER</th><th>Speaker</th></tr>
    {worst_rows}
  </table>

  <h2>Per-Speaker Breakdown</h2>
  <table>
    <tr><th>Speaker</th><th>WER</th><th>CER</th><th>Samples</th></tr>
    {spk_rows}
  </table>
</body>
</html>"""
# per_speaker dict maps speaker_id -> {wer, cer, num_samples}
# single-speaker dataset handled; per_speaker dict may have 1 entry
# metadata dict stores model_path, config, hardware info
