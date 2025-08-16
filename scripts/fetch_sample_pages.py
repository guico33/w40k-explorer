#!/usr/bin/env python3
"""
Fetch random pages from data/article_urls.json and save the HTML into data/sample_pages.
Usage: python3 scripts/fetch_sample_pages.py --count 5
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_COUNT = 5
ROOT = Path(__file__).resolve().parents[1]
URLS_PATH = ROOT / "data" / "article_urls.json"
OUT_DIR = ROOT / "data" / "sample_pages"


def safe_name(url: str) -> str:
    p = urllib.parse.urlparse(url)
    path = (p.netloc + p.path) or p.netloc
    if path.endswith("/"):
        path = path + "index"
    # replace non-filename-safe chars
    name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in path)
    # ensure it's not empty
    if not name:
        name = "page"
    # truncate
    return name[:200]


def fetch_url(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "w40k-explorer-fetcher/0.1"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of random pages to fetch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )
    parser.add_argument(
        "--urls", type=str, default=str(URLS_PATH), help="Path to JSON file with URLs"
    )
    parser.add_argument(
        "--out", type=str, default=str(OUT_DIR), help="Output directory for HTML files"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    urls_file = Path(args.urls)
    if not urls_file.exists():
        print(f"URLs file not found: {urls_file}")
        sys.exit(1)

    try:
        with urls_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read or parse {urls_file}: {e}")
        sys.exit(1)

    # Accept multiple known shapes: a top-level list, or an object with a 'urls' key.
    urls = None
    if isinstance(data, list):
        urls = data
    elif isinstance(data, dict):
        # common key used by the scraper
        for key in ("urls", "links", "items", "data"):
            if key in data and isinstance(data[key], list):
                urls = data[key]
                break
        # fallback: grab the first list found in the dict
        if urls is None:
            for v in data.values():
                if isinstance(v, list):
                    urls = v
                    break

    if not isinstance(urls, list) or not urls:
        print(f"No URLs found in {urls_file}")
        sys.exit(1)

    n = min(args.count, len(urls))
    selected = random.sample(urls, n)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    successes = []
    failures = []

    for i, url in enumerate(selected, start=1):
        try:
            print(f"[{i}/{n}] Fetching: {url}")
            data = fetch_url(url)
            filename = f"{i:02d}_{safe_name(url)}.html"
            out_path = out_dir / filename
            with out_path.open("wb") as fw:
                fw.write(data)
            successes.append(str(out_path))
            print(f"  Saved -> {out_path} ({len(data)} bytes)")
        except Exception as e:
            failures.append((url, str(e)))
            print(f"  Failed to fetch {url}: {e}")
        # polite delay between fetches to avoid hammering servers
        time.sleep(1)

    print("\nSummary:")
    print(f"  Requested: {args.count}")
    print(f"  Fetched: {len(successes)}")
    print(f"  Failed: {len(failures)}")
    if failures:
        print("Failures detail:")
        for url, err in failures:
            print(f" - {url} -> {err}")


if __name__ == "__main__":
    main()
