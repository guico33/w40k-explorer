"""Dataset utilities for evaluation runs.

Loads a simple JSONL dataset with fields:
- id (str)
- question (str)
- optional: expected_answer (str)
- optional: expected_sources (list[str])
- optional: tags (list[str])
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class QAItem:
    """Single QA sample for evals.

    Attributes:
        id: Stable sample identifier used for joins and reporting.
        question: Natural‑language question to pass to the answer service.
        expected_answer: Optional short reference answer for similarity checks.
        expected_sources: Optional list of canonical URLs (or IDs) expected
            to appear in retrieved context; used for recall/coverage metrics.
        tags: Optional list of labels (topic, difficulty) for analysis.
    """

    id: str
    question: str
    expected_answer: Optional[str] = None
    expected_sources: Optional[List[str]] = None
    tags: Optional[List[str]] = None


def load_jsonl(path: str | Path) -> List[QAItem]:
    """Load a JSONL dataset of QAItem rows.

    Args:
        path: Filesystem path to a JSON Lines file. Each line must be a JSON
            object containing at least `id` and `question` keys.

    Returns:
        A list of validated QAItem instances in file order.

    Raises:
        ValueError: If a line is not valid JSON, or required fields are missing.
    """
    p = Path(path)
    items: List[QAItem] = []
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {ln}: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"Line {ln}: expected object, got {type(obj)!r}")

            id_ = str(obj.get("id", "")).strip()
            q = str(obj.get("question", "")).strip()
            if not id_ or not q:
                raise ValueError(
                    f"Line {ln}: missing required fields 'id' or 'question'"
                )

            exp_ans = obj.get("expected_answer")
            exp_src = obj.get("expected_sources")
            # Handle dataset that uses source_url instead of expected_sources
            if exp_src is None and "source_url" in obj:
                exp_src = [obj["source_url"]]
            tags = obj.get("tags")

            items.append(
                QAItem(
                    id=id_,
                    question=q,
                    expected_answer=str(exp_ans) if isinstance(exp_ans, str) else None,
                    expected_sources=(
                        [str(s) for s in (exp_src or [])]
                        if isinstance(exp_src, list)
                        else None
                    ),
                    tags=(
                        [str(t) for t in (tags or [])]
                        if isinstance(tags, list)
                        else None
                    ),
                )
            )
    return items


def take_subset(items: List[QAItem], subset: Optional[int] = None) -> List[QAItem]:
    """Return a prefix subset of items for quick, low‑cost runs.

    Args:
        items: Full dataset loaded via `load_jsonl`.
        subset: Number of leading items to keep. If None, returns `items`.

    Returns:
        A list containing at most `subset` items, preserving input order.
    """
    if subset is None:
        return items
    if subset <= 0:
        return []
    return items[:subset]
