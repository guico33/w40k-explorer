"""Core evaluation metrics (deterministic, no network)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Dict, Any, Tuple
import math


def json_ok(citations: Optional[List[Dict[str, Any]]]) -> bool:
    """Return True if citations is a JSON‑serializable list structure.

    Args:
        citations: Parsed citations field from the answer output.

    Returns:
        True if `citations` is a list (possibly empty); False otherwise.
    """
    return isinstance(citations, list)


def citations_valid(answer_text: str, citations: List[Dict[str, Any]]) -> bool:
    """Check structural validity of `[n]` citation markers.

    Ensures that every referenced marker in the answer (e.g., "[1]") has a
    corresponding entry in the `citations` list. Does not validate semantic
    correctness or URL mapping — see `context_coverage` for coverage.

    Args:
        answer_text: Rendered assistant answer containing citation markers.
        citations: Structured citations list aligned by 1‑based indices.

    Returns:
        True if all referenced indices exist within `citations`; False otherwise.
    """
    if not citations:
        # If there are no citations, answer shouldn't contain [n]
        return "[" not in answer_text

    used_ids = set()
    for i in range(1, len(citations) + 1):
        token = f"[{i}]"
        if token in answer_text:
            used_ids.add(i)
    # valid if every referenced id exists (we don't require that every citation be used)
    return all(1 <= i <= len(citations) for i in used_ids)


def context_coverage(expected_sources: Optional[List[str]], context_items: Optional[List[Dict[str, Any]]]) -> float:
    """Compute fraction of expected sources present in final packed context.

    Args:
        expected_sources: List of canonical source URLs (or IDs) to look for.
        context_items: Final context items provided to the LLM (after selection
            and truncation). Each may contain `canonical_url` or `url`.

    Returns:
        Value in [0, 1], the proportion of expected sources found among the
        context items. Returns 0.0 if either argument is missing.
    """
    if not expected_sources:
        return 0.0
    if not context_items:
        return 0.0
    urls = set()
    for c in context_items:
        url = c.get("canonical_url") or c.get("url")
        if isinstance(url, str) and url:
            urls.add(url)
    hits = sum(1 for s in expected_sources if s in urls)
    return hits / max(1, len(expected_sources))


def recall_at_k(expected: List[str], retrieved_urls: List[str], k: int = 20) -> float:
    """Compute Recall@k given expected URLs and retrieved URLs.

    Args:
        expected: Ground‑truth set of relevant URLs.
        retrieved_urls: Ranked list of retrieved URLs.
        k: Cutoff position to consider.

    Returns:
        Recall@k in [0, 1]. Returns 0.0 if `expected` is empty.
    """
    if not expected:
        return 0.0
    subset = set(retrieved_urls[:k])
    hit = sum(1 for s in expected if s in subset)
    return hit / len(expected)


def ndcg_at_k(expected: List[str], retrieved_urls: List[str], k: int = 20) -> float:
    """Compute nDCG@k with binary relevance based on URL matches.

    Args:
        expected: Ground‑truth relevant URLs.
        retrieved_urls: Ranked list of retrieved URLs.
        k: Cutoff position to consider.

    Returns:
        Normalized Discounted Cumulative Gain in [0, 1]. Returns 0.0 if `expected`
        is empty.
    """
    if not expected:
        return 0.0
    ideal = 0.0
    for i in range(min(k, len(expected))):
        ideal += 1.0 / math.log2(i + 2)
    if ideal == 0.0:
        return 0.0

    dcg = 0.0
    for i, url in enumerate(retrieved_urls[:k]):
        if url in expected:
            dcg += 1.0 / math.log2(i + 2)
    return dcg / ideal


def expected_calibration_error(confidences: List[float], correct: List[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over confidence/correctness pairs.

    Args:
        confidences: Model‑reported confidence scores in [0, 1].
        correct: Boolean flags indicating per‑sample correctness.
        n_bins: Number of bins to use for calibration histogram.

    Returns:
        ECE value in [0, 1], where lower is better calibrated.
    """
    assert len(confidences) == len(correct)
    if not confidences:
        return 0.0
    bins = [0] * n_bins
    bin_correct = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins

    for c, y in zip(confidences, correct):
        c = min(max(c, 0.0), 1.0)
        idx = min(n_bins - 1, int(c * n_bins))
        bins[idx] += 1
        bin_correct[idx] += 1 if y else 0
        bin_conf_sum[idx] += c

    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        if bins[i] == 0:
            continue
        acc = bin_correct[i] / bins[i]
        avg_conf = bin_conf_sum[i] / bins[i]
        ece += (bins[i] / n) * abs(acc - avg_conf)
    return ece
