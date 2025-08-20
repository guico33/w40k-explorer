#!/usr/bin/env python3
"""Run evaluation on a QA dataset.

Usage examples:
  uv run python scripts/run_evals.py --dataset tests/evals/datasets/qa_dataset.jsonl
  uv run python scripts/run_evals.py --dataset tests/evals/datasets/qa_dataset.jsonl --subset 10 --vector-provider pinecone
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# Make src importable
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from w40k.config.factory import create_answer_service, create_llm_client
from w40k.config.settings import Settings, get_settings
from w40k.evals.dataset import load_jsonl, take_subset
from w40k.evals.judges import judge_groundedness, judge_relevance
from w40k.evals.metrics import citations_valid as metric_citations_valid
from w40k.evals.metrics import context_coverage as metric_context_coverage
from w40k.evals.metrics import expected_calibration_error
from w40k.evals.metrics import json_ok as metric_json_ok
from w40k.evals.report import format_markdown, make_summary


def _ensure_outdir(outdir: Path) -> None:
    """Create the output directory (and parents) if missing.

    Args:
        outdir: Target directory where run artifacts will be written.
    """
    outdir.mkdir(parents=True, exist_ok=True)


def _is_refusal(answer_text: str) -> bool:
    """Heuristic to flag likely refusal/safety responses.

    Args:
        answer_text: Model's generated answer text.

    Returns:
        True if the text contains common refusal phrases; False otherwise.
    """
    t = (answer_text or "").lower()
    return any(
        s in t
        for s in [
            "cannot help with",
            "cannot assist",
            "refuse",
            "not able to",
            "i can't ",
        ]
    )


def _process_single_item(
    item, answer_service, judge_client, eval_model, start_time
) -> Dict[str, Any]:
    """Process a single evaluation item.

    Args:
        item: QA item to evaluate
        answer_service: Service for generating answers
        judge_client: LLM client for judges
        eval_model: Model name for evaluation
        start_time: Start time for logging

    Returns:
        Dict with evaluation results
    """

    def _trunc(text: str, max_len: int = 240) -> str:
        return text if len(text) <= max_len else text[: max_len - 1] + "…"

    try:
        start_dt = datetime.utcnow()
        logging.info(
            "START id=%s at=%s question=%s",
            item.id,
            start_dt.isoformat(timespec="seconds") + "Z",
            _trunc(item.question.replace("\n", " ")),
        )

        result = answer_service.answer_query(item.question)
    except Exception as e:
        logging.exception("Evaluation failed for id=%s: %s", item.id, e)
        return None

    # Core fields
    answer_text = result.answer or ""
    citations = result.citations or []
    json_ok = metric_json_ok(citations)
    cite_valid = metric_citations_valid(answer_text, citations)
    context_items = getattr(result, "context_items", None)
    coverage = metric_context_coverage(item.expected_sources, context_items)

    # Judges (cheap and simple)
    try:
        grounded = judge_groundedness(
            judge_client,
            eval_model,
            item.question,
            answer_text,
            context_items or [],
        )
    except Exception as e:
        logging.error("Groundedness judge failed: %s", e)
        grounded = {"grounded": False, "reasons": f"judge-error: {str(e)}"}

    try:
        relevance = judge_relevance(
            judge_client, eval_model, item.question, answer_text
        )
    except Exception as e:
        logging.error("Relevance judge failed: %s", e)
        relevance = {"relevant": False, "reasons": f"judge-error: {str(e)}"}

    grounded_bool = bool(grounded.get("grounded", False))
    relevant_bool = bool(relevance.get("relevant", False))
    refusal = _is_refusal(answer_text)

    end_dt = datetime.utcnow()
    elapsed_ms = int((end_dt - start_dt).total_seconds() * 1000)
    logging.info(
        "END   id=%s at=%s ms=%s json_ok=%s cite_valid=%s cov=%.2f grounded=%s relevant=%s refusal=%s conf=%.2f",
        item.id,
        end_dt.isoformat(timespec="seconds") + "Z",
        elapsed_ms,
        json_ok,
        cite_valid,
        coverage,
        grounded_bool,
        relevant_bool,
        refusal,
        float(getattr(result, "confidence", 0.0) or 0.0),
    )
    logging.info("ANSWER id=%s %s", item.id, _trunc(answer_text.replace("\n", " ")))

    return {
        "id": item.id,
        "question": item.question,
        "answer": answer_text,
        "citations": citations,
        "json_ok": 1.0 if json_ok else 0.0,
        "citations_valid": 1.0 if cite_valid else 0.0,
        "context_coverage": float(coverage),
        "grounded": 1.0 if grounded_bool else 0.0,
        "relevant": 1.0 if relevant_bool else 0.0,
        "refusal": 1.0 if refusal else 0.0,
        "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
        "query_time_ms": int(getattr(result, "query_time_ms", 0) or 0),
        "judged_grounded": grounded,
        "judged_relevant": relevance,
    }


def main() -> int:
    """Run evals for a given dataset and write a report to disk.

    Returns:
        Process exit code (0 on success).
    """
    ap = argparse.ArgumentParser(description="Run evals against AnswerService")
    ap.add_argument(
        "--dataset", help="Path to qa_dataset.jsonl (optional; defaults resolved)"
    )
    ap.add_argument("--subset", type=int, help="Limit to first N samples")
    ap.add_argument(
        "--vector-provider",
        choices=["qdrant", "pinecone"],
        help="Override VECTOR_PROVIDER",
    )
    ap.add_argument("--model", help="Override TEST_MODEL for inference")
    ap.add_argument("--eval-model", help="Override EVAL_MODEL for judges")
    ap.add_argument("--outdir", default="evals/runs", help="Output directory for runs")
    ap.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Max concurrent workers for parallel evaluation (default: 5)",
    )
    args = ap.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    settings = get_settings()
    # Vector provider resolution with typo correction
    vp = (args.vector_provider or settings.vector_provider or "").lower()

    settings.vector_provider = vp

    test_model = args.model or os.getenv("TEST_MODEL") or settings.get_llm_model()
    eval_model = args.eval_model or os.getenv("EVAL_MODEL") or settings.get_llm_model()

    logging.info("Using vector provider: %s", settings.vector_provider)
    logging.info("TEST_MODEL=%s, EVAL_MODEL=%s", test_model, eval_model)

    # Spin up inference stack
    answer_service, stats = create_answer_service(settings=settings)
    # Override model for eval run if requested
    if test_model and test_model != answer_service.model:
        answer_service.model = test_model

    # LLM judge client (reuse provider from settings)
    judge_client = create_llm_client(settings)

    # Resolve dataset path: prefer explicit arg; otherwise try common locations
    def _resolve_dataset_path(arg: Optional[str]) -> Path:
        """Resolve a dataset path from CLI arg or default locations.

        Order:
        1) Explicit --dataset path if provided
        2) src/w40k/evals/qa_dataset.jsonl
        3) tests/evals/datasets/qa_dataset.jsonl
        """
        candidates: List[Path] = []
        if arg:
            candidates.append(Path(arg))
        candidates.extend(
            [
                Path("src/w40k/evals/qa_dataset.jsonl"),
                Path("tests/evals/datasets/qa_dataset.jsonl"),
            ]
        )
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            "qa_dataset.jsonl not found. Provide --dataset or place it at "
            "src/w40k/evals/qa_dataset.jsonl or tests/evals/datasets/qa_dataset.jsonl"
        )

    dataset_path = _resolve_dataset_path(args.dataset)
    dataset = load_jsonl(dataset_path)
    logging.info("Loaded dataset: %s (%d samples)", dataset_path, len(dataset))
    dataset = take_subset(dataset, args.subset)
    if not dataset:
        print("No samples to evaluate.")
        return 0

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = (
        Path(args.outdir) / f"{ts}_{settings.llm_provider}_{settings.vector_provider}"
    )
    _ensure_outdir(outdir)
    samples_path = outdir / "samples.jsonl"
    summary_path = outdir / "summary.json"
    report_md_path = outdir / "report.md"
    meta_path = outdir / "meta.json"

    # File logging for this run
    try:
        fh = logging.FileHandler(outdir / "run.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logging.getLogger().addHandler(fh)
        logging.info("Run directory: %s", str(outdir))
    except Exception:
        pass

    rows: List[Dict[str, Any]] = []
    confidences: List[float] = []
    correct_flags: List[bool] = []

    with samples_path.open("w", encoding="utf-8") as f:
        # Progress indicator
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            tqdm = None

        # Process items in parallel using ThreadPoolExecutor
        start_time = datetime.utcnow()

        # Use ThreadPoolExecutor for I/O bound operations (API calls)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(
                    _process_single_item,
                    item,
                    answer_service,
                    judge_client,
                    eval_model,
                    start_time,
                ): item
                for item in dataset
            }

            # Progress tracking
            if tqdm:
                pbar = tqdm(total=len(dataset), desc="Evaluating", unit="sample")

            # Collect results as they complete
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    row = future.result()
                    if row is not None:  # Skip failed items
                        rows.append(row)

                        # Extract values for ECE calculation
                        grounded_bool = bool(row["grounded"])
                        relevant_bool = bool(row["relevant"])
                        refusal = bool(row["refusal"])

                        confidences.append(row["confidence"])
                        correct_flags.append(
                            grounded_bool and relevant_bool and not refusal
                        )

                        # Write to file immediately
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()  # Ensure data is written immediately

                except Exception as e:
                    logging.exception("Failed to process item %s: %s", item.id, e)

                if tqdm:
                    pbar.update(1)  # type: ignore

            if tqdm:
                pbar.close()  # type: ignore

    ece = expected_calibration_error(confidences, correct_flags, n_bins=10)
    # Attach ECE to all rows for simple reporting
    for r in rows:
        r["ece"] = ece

    summary = make_summary(rows)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary.__dict__, f, indent=2)

    with report_md_path.open("w", encoding="utf-8") as f:
        f.write(format_markdown(summary))

    meta = {
        "llm_provider": settings.llm_provider,
        "vector_provider": settings.vector_provider,
        "test_model": test_model,
        "eval_model": eval_model,
        "dataset": str(dataset_path.resolve()),
        "timestamp": ts,
        "chunks_count": stats.get("chunks_count"),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nEval run complete:")
    print(f" - Samples: {len(rows)}")
    print(f" - Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
