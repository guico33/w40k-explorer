#!/usr/bin/env python
"""
Add embedding tracking columns to the existing `chunk` table.

- Adds columns to `chunk` (if missing):
  has_embedding BOOLEAN DEFAULT 0
  embedding_generated_at TEXT
  embedding_failed_count INTEGER DEFAULT 0
  last_embedding_error TEXT

- Creates indexes (if missing):
  idx_chunk_has_embedding on chunk(has_embedding)
  idx_chunk_embedding_generated_at on chunk(embedding_generated_at)

Usage:
  uv run python migrations/2025_08_17_add_embedding_tracking.py path/to/your.db
  # add --dry-run to preview changes without applying
"""
from __future__ import annotations

import argparse
import sqlite3

CHUNK_COLUMNS = [
    ("has_embedding", "BOOLEAN DEFAULT 0"),
    ("embedding_generated_at", "TEXT"),
    ("embedding_failed_count", "INTEGER DEFAULT 0"),
    ("last_embedding_error", "TEXT"),
]

CHUNK_INDEXES = [
    ("idx_chunk_has_embedding", "chunk", "has_embedding"),
    ("idx_chunk_embedding_generated_at", "chunk", "embedding_generated_at"),
]


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,)
    )
    return cur.fetchone() is not None


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info('{table}');")
    return any(row[1] == column for row in cur.fetchall())


def existing_indexes(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA index_list('{table}')")
    return {row[1] for row in cur.fetchall()}  # row[1] = index name


def add_column(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {coltype};')


def create_index_if_missing(
    conn: sqlite3.Connection, index_name: str, table: str, column: str
) -> None:
    conn.execute(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table}"("{column}");')


def ensure_chunk_columns(conn: sqlite3.Connection, dry_run: bool) -> list[str]:
    planned: list[str] = []

    if not table_exists(conn, "chunk"):
        planned.append(
            "ERROR: chunk table does not exist. Run the parsing migration first."
        )
        return planned

    for col, coltype in CHUNK_COLUMNS:
        if not column_exists(conn, "chunk", col):
            planned.append(f"ALTER TABLE chunk ADD COLUMN {col} {coltype};")
            if not dry_run:
                add_column(conn, "chunk", col, coltype)
    return planned


def ensure_chunk_indexes(conn: sqlite3.Connection, dry_run: bool) -> list[str]:
    planned: list[str] = []

    if not table_exists(conn, "chunk"):
        return planned

    idx_names = existing_indexes(conn, "chunk")
    for idx_name, table, column in CHUNK_INDEXES:
        if idx_name not in idx_names:
            planned.append(f"CREATE INDEX {idx_name} ON {table}({column});")
            if not dry_run:
                create_index_if_missing(conn, idx_name, table, column)
    return planned


def summarize_schema(conn: sqlite3.Connection) -> None:
    print("\nChunk table schema:")
    if not table_exists(conn, "chunk"):
        print("  - chunk: (missing)")
        return

    cols = conn.execute("PRAGMA table_info('chunk')").fetchall()
    embedding_cols = [c for c in cols if c[1] in [col[0] for col, _ in CHUNK_COLUMNS]]

    if embedding_cols:
        print("  Embedding tracking columns:")
        for col in embedding_cols:
            print(f"    - {col[1]}: {col[2]}")
    else:
        print("  No embedding tracking columns found")

    idxs = conn.execute("PRAGMA index_list('chunk')").fetchall()
    embedding_idxs = [
        i for i in idxs if i[1] in [idx[0] for idx, _, _ in CHUNK_INDEXES]
    ]

    if embedding_idxs:
        print("  Embedding tracking indexes:")
        for idx in embedding_idxs:
            print(f"    - {idx[1]}")
    else:
        print("  No embedding tracking indexes found")


def show_chunk_stats(conn: sqlite3.Connection) -> None:
    if not table_exists(conn, "chunk"):
        return

    try:
        total_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunk WHERE active = 1;"
        ).fetchone()[0]
        print(f"  Total active chunks: {total_chunks:,}")

        if column_exists(conn, "chunk", "has_embedding"):
            with_embeddings = conn.execute(
                "SELECT COUNT(*) FROM chunk WHERE has_embedding = 1;"
            ).fetchone()[0]
            failed_chunks = conn.execute(
                "SELECT COUNT(*) FROM chunk WHERE embedding_failed_count > 0;"
            ).fetchone()[0]
            print(f"  Chunks with embeddings: {with_embeddings:,}")
            print(f"  Chunks with failures: {failed_chunks:,}")
            if total_chunks > 0:
                coverage = (with_embeddings / total_chunks) * 100
                print(f"  Embedding coverage: {coverage:.1f}%")
    except Exception as e:
        print(f"  Could not retrieve stats: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SQLite migration: add embedding tracking columns to chunk table."
    )
    ap.add_argument("db_path", help="Path to your SQLite database file")
    ap.add_argument(
        "--dry-run", action="store_true", help="Show planned changes without applying"
    )
    args = ap.parse_args()

    conn = sqlite3.connect(args.db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")

        summarize_schema(conn)
        show_chunk_stats(conn)

        print("\nPlanned changes:")
        planned: list[str] = []

        with conn:
            planned += ensure_chunk_columns(conn, args.dry_run)
            planned += ensure_chunk_indexes(conn, args.dry_run)

        if planned:
            for stmt in planned:
                print("  -", stmt)
            if args.dry_run:
                print("\n(DRY RUN) No changes applied.")
            else:
                print("\nApplied changes successfully.")
        else:
            print("  (none) — schema already up to date.")

        summarize_schema(conn)
        show_chunk_stats(conn)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
