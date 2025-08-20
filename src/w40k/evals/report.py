"""Aggregate and format eval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Summary:
    """Aggregate view of an eval run.

    Attributes:
        count: Number of evaluated samples.
        json_ok_rate: Share of samples with parsable structured output.
        citation_valid_rate: Share with structurally valid citation markers.
        context_coverage_avg: Average fraction of expected sources found in context.
        grounded_rate: Share judged grounded by the LLM judge.
        relevance_rate: Share judged relevant to the question.
        refusal_rate: Share detected as refusals by a lightweight heuristic.
        ece: Expected Calibration Error across confidence/correctness pairs.
    """

    count: int
    json_ok_rate: float
    citation_valid_rate: float
    context_coverage_avg: float
    grounded_rate: float
    relevance_rate: float
    refusal_rate: float
    ece: float


def make_summary(rows: List[Dict[str, Any]]) -> Summary:
    """Summarize per‑sample metrics into a single `Summary` object.

    Args:
        rows: List of per‑sample result dicts containing metric keys.

    Returns:
        A populated `Summary` with averages and counts.
    """
    n = max(1, len(rows))
    def avg(key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in rows) / n

    return Summary(
        count=len(rows),
        json_ok_rate=avg("json_ok"),
        citation_valid_rate=avg("citations_valid"),
        context_coverage_avg=avg("context_coverage"),
        grounded_rate=avg("grounded"),
        relevance_rate=avg("relevant"),
        refusal_rate=avg("refusal"),
        ece=float(rows[0].get("ece", 0.0)) if rows else 0.0,
    )


def format_markdown(summary: Summary) -> str:
    """Render a concise Markdown report for human review.

    Args:
        summary: Aggregated metrics from `make_summary`.

    Returns:
        Markdown string with key rates and counts.
    """
    return (
        f"# Eval Summary\n\n"
        f"- Samples: {summary.count}\n"
        f"- JSON OK: {summary.json_ok_rate:.2%}\n"
        f"- Citations valid: {summary.citation_valid_rate:.2%}\n"
        f"- Context coverage (avg): {summary.context_coverage_avg:.2%}\n"
        f"- Grounded: {summary.grounded_rate:.2%}\n"
        f"- Relevant: {summary.relevance_rate:.2%}\n"
        f"- Refusal rate: {summary.refusal_rate:.2%}\n"
        f"- ECE: {summary.ece:.4f}\n"
    )
