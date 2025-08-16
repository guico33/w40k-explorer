#!/usr/bin/env python
"""
Add parsing-related columns to the existing `article` table and create the `chunk` table.

- Adds columns to `article` (if missing):
  parsed_json TEXT
  parser_version TEXT
  parsed_at TEXT
  last_parsed_html_hash TEXT
  canonical_url TEXT
  lead TEXT
  infobox_type TEXT

- Creates indexes (if missing):
  idx_article_canonical_url on article(canonical_url)
  idx_article_infobox_type on article(infobox_type)
  idx_chunk_article on chunk(article_id)

- Creates table `chunk` (if missing):
  id INTEGER PRIMARY KEY
  article_id INTEGER NOT NULL REFERENCES article(id) ON DELETE CASCADE
  section_path TEXT
  paragraph_idx INTEGER
  text TEXT NOT NULL
  token_count INTEGER

Usage:
  uv run python migrations/2025_08_16_add_parsing_columns.py path/to/your.db
  # add --dry-run to preview changes without applying
"""
from __future__ import annotations

import argparse
import sqlite3
from typing import Iterable, Tuple

ARTICLE_COLUMNS = [
    ("parsed_json", "TEXT"),
    ("parser_version", "TEXT"),
    ("parsed_at", "TEXT"),
    ("last_parsed_html_hash", "TEXT"),
    ("canonical_url", "TEXT"),
    ("lead", "TEXT"),
    ("infobox_type", "TEXT"),
]

ARTICLE_INDEXES = [
    ("idx_article_canonical_url", "article", "canonical_url"),
    ("idx_article_infobox_type", "article", "infobox_type"),
]

CHUNK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS "chunk" (
  id INTEGER PRIMARY KEY,
  article_id INTEGER NOT NULL REFERENCES article(id) ON DELETE CASCADE,
  section_path TEXT,
  paragraph_idx INTEGER,
  text TEXT NOT NULL,
  token_count INTEGER
);
"""

CHUNK_INDEXES = [
    ("idx_chunk_article", "chunk", "article_id"),
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


def ensure_article_columns(conn: sqlite3.Connection, dry_run: bool) -> list[str]:
    planned: list[str] = []
    for col, coltype in ARTICLE_COLUMNS:
        if not column_exists(conn, "article", col):
            planned.append(f"ALTER TABLE article ADD COLUMN {col} {coltype};")
            if not dry_run:
                add_column(conn, "article", col, coltype)
    return planned


def ensure_article_indexes(conn: sqlite3.Connection, dry_run: bool) -> list[str]:
    planned: list[str] = []
    idx_names = existing_indexes(conn, "article")
    for idx_name, table, column in ARTICLE_INDEXES:
        if idx_name not in idx_names:
            planned.append(f"CREATE INDEX {idx_name} ON {table}({column});")
            if not dry_run:
                create_index_if_missing(conn, idx_name, table, column)
    return planned


def ensure_chunk_table_and_indexes(
    conn: sqlite3.Connection, dry_run: bool
) -> list[str]:
    planned: list[str] = []
    if not table_exists(conn, "chunk"):
        planned.append("-- create table chunk")
        if not dry_run:
            conn.execute(CHUNK_TABLE_SQL)

    # indexes
    if table_exists(conn, "chunk"):
        idx_names = existing_indexes(conn, "chunk")
        for idx_name, table, column in CHUNK_INDEXES:
            if idx_name not in idx_names:
                planned.append(f"CREATE INDEX {idx_name} ON {table}({column});")
                if not dry_run:
                    create_index_if_missing(conn, idx_name, table, column)
    return planned


def summarize_schema(conn: sqlite3.Connection) -> None:
    print("\nCurrent schema summary:")
    for tbl in ("article", "chunk"):
        if not table_exists(conn, tbl):
            print(f"  - {tbl}: (missing)")
            continue
        cols = conn.execute(f"PRAGMA table_info('{tbl}')").fetchall()
        col_list = ", ".join(f"{c[1]}:{c[2]}" for c in cols)
        idxs = conn.execute(f"PRAGMA index_list('{tbl}')").fetchall()
        idx_list = ", ".join(i[1] for i in idxs) if idxs else "(none)"
        print(f"  - {tbl}: {col_list}")
        print(f"    indexes: {idx_list}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SQLite migration: add parsing columns & chunk table."
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

        print("\nPlanned changes:")
        planned: list[str] = []

        with conn:
            planned += ensure_article_columns(conn, args.dry_run)
            planned += ensure_article_indexes(conn, args.dry_run)
            planned += ensure_chunk_table_and_indexes(conn, args.dry_run)

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
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
