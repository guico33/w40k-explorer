# scripts/check_parser.py
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter

# Make sure we can import src/rag/parser.py when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import your parser entrypoint
try:
    from rag.parser import parse_article_html  # type: ignore
except Exception as e:
    print("Failed to import rag.parser.parse_article_html:", e, file=sys.stderr)
    sys.exit(2)


def run_one(path: Path, serialize: bool = True) -> tuple[bool, str]:
    """Return (ok, message)."""
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
        doc = parse_article_html(html, fetched_at=None)
        if serialize:
            # Ensure the result is JSON-serializable
            json.dumps(doc, ensure_ascii=False)
        return True, "ok"
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"{e.__class__.__name__}: {e}\n{tb}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run parser on a directory of HTML files.")
    ap.add_argument(
        "directory",
        nargs="?",
        default="data/sample_pages",
        help="Directory to scan (default: data/sample_pages)",
    )
    ap.add_argument(
        "--glob",
        default="*.html",
        help="Filename glob pattern (default: *.html)",
    )
    ap.add_argument(
        "--no-serialize",
        action="store_true",
        help="Skip JSON serialization check (slightly faster).",
    )
    ap.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only print failures and the final summary.",
    )
    args = ap.parse_args()

    root = Path(args.directory)
    if not root.exists():
        print(f"Directory not found: {root}", file=sys.stderr)
        return 2

    files = sorted(root.rglob(args.glob))
    if not files:
        print(f"No files matching {args.glob} under {root}")
        return 0

    start = perf_counter()
    ok_count = 0
    fail_count = 0
    failures: list[tuple[Path, str]] = []

    for i, path in enumerate(files, 1):
        ok, msg = run_one(path, serialize=not args.no_serialize)
        if ok:
            ok_count += 1
            if not args.quiet:
                print(f"[{i:>4}/{len(files)}] ✅ {path}")
        else:
            fail_count += 1
            print(f"[{i:>4}/{len(files)}] ❌ {path}")
            print(msg)
            failures.append((path, msg))

    dur = perf_counter() - start
    print("\n=== Summary ===")
    print(f"Scanned: {len(files)} files in {dur:.2f}s")
    print(f"Passed : {ok_count}")
    print(f"Failed : {fail_count}")

    # Non-zero exit if there were failures
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
