"""Structure-aware chunker for Warhammer 40k wiki articles.

Main entrypoint: `chunk_article(parsed: dict, params: ChunkParams | None = None) -> list[Chunk]`

- Consumes one parsed-JSON article (output of your parser)
- Produces a list of `Chunk` ORM instances (ready to persist)

This module implements the micro-block -> packing -> Chunk pipeline described in chat:
- Respect article structure (lead, preface, sections)
- Treat lists as item-level micro-blocks
- Flatten tables as groups of rows
- Sentence-split oversize blocks
- Pack greedily to a token budget with overlap, never crossing section boundaries
- Deterministic `chunk_uid`

You can adapt the few constants in `ChunkParams` for tuning.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, List, Optional, Sequence, Tuple

import tiktoken

from ..database.models import BlockType, Chunk


@dataclass
class ChunkParams:
    target_tokens: int = 350
    overlap_tokens: int = 70
    max_tokens_per_micro: int = 300
    table_rows_per_micro: int = 6
    kv_preview_keys: Tuple[str, ...] = (
        "Affiliation",
        "Allegiance",
        "Founding",
        "Homeworld",
        "Type",
    )
    # Embedding header layout
    include_header: bool = True


# -----------------
# Tokenization utils
# -----------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])")
_WS_RE = re.compile(r"\s+")


def _get_tokenizer(model: str | None = None):
    """Return a callable(text:str)->int counting tokens.
    Prefers tiktoken if available, else uses a simple heuristic (~whitespace split).
    """
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model or "text-embedding-3-small")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")

        def _count(s: str) -> int:
            return len(enc.encode(s))

        return _count

    # Fallback heuristic: ~1.3 words per token -> tokens ~ ceil(words / 1.3)
    def _count_fallback(s: str) -> int:
        words = len(_WS_RE.split(s.strip())) if s.strip() else 0
        return int(math.ceil(words / 1.3))

    return _count_fallback


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # First try regex split; fallback to naive if single piece
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    if not parts:
        return [text]
    return parts


# -----------------
# Micro-block model
# -----------------


@dataclass
class Micro:
    text: str
    section_path: Tuple[str, ...]  # immutable path for grouping
    block_type: BlockType


# -----------------
# Extraction helpers (from article JSON)
# -----------------


def _kv_preview_from_infobox(
    infobox: Optional[dict], keys: Sequence[str]
) -> Optional[str]:
    if not infobox:
        return None
    kv = infobox.get("kv_raw") or {}
    if not isinstance(kv, dict):
        return None
    pairs = []
    for k in keys:
        v = kv.get(k)
        if v is None:
            continue
        # v could be str or list; normalize to str
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v if x)
        v = str(v).strip()
        if v:
            pairs.append(f"{k}={v}")
    return "; ".join(pairs) if pairs else None


def _normalize_table_rows(table: dict) -> List[str]:
    """Flatten a table structure into list of row strings: 'Header: ... | Row: ...'"""
    if not isinstance(table, dict):
        return []
    headers = table.get("headers") or []
    rows = table.get("rows") or []
    out: List[str] = []
    header_line = " | ".join(str(h).strip() for h in headers if str(h).strip())
    for r in rows:
        row_line = " | ".join(str(c).strip() for c in (r or []) if str(c).strip())
        if header_line and row_line:
            out.append(f"{header_line}\n{row_line}")
        elif row_line:
            out.append(row_line)
    return out


# -----------------
# Micro-block generation
# -----------------


def _yield_micro_from_container(
    container: Optional[dict], path: Tuple[str, ...], params: ChunkParams, tk_count
) -> Iterator[Micro]:
    if not container:
        return

    # paragraphs
    for i, p in enumerate(container.get("paragraphs", []) or []):
        txt = (p or "").strip()
        if not txt:
            continue
        yield from _maybe_split_micro(txt, path, BlockType.PARAGRAPH, params, tk_count)

    # lists: items as individual micros
    for lst in container.get("lists", []) or []:
        items = (lst or {}).get("items") or []
        for i, it in enumerate(items):
            txt = (it.get("text", "") if isinstance(it, dict) else str(it)).strip()
            if not txt:
                continue
            yield from _maybe_split_micro(
                txt, path, BlockType.LIST_ITEM, params, tk_count
            )

    # quotes
    for i, q in enumerate(container.get("quotes", []) or []):
        txt = (q or "").strip()
        if not txt:
            continue
        yield from _maybe_split_micro(txt, path, BlockType.QUOTE, params, tk_count)

    # tables -> group rows
    for t in container.get("tables", []) or []:
        rows = _normalize_table_rows(t)
        if not rows:
            continue
        step = max(1, params.table_rows_per_micro)
        for i in range(0, len(rows), step):
            txt = "\n".join(rows[i : i + step]).strip()
            if not txt:
                continue
            # Use hard split for table rows only when extremely large
            yield from _maybe_split_micro(
                txt, path, BlockType.TABLE_ROWS, params, tk_count
            )

    # figures (optional)
    for i, fig in enumerate(container.get("figures", []) or []):
        cap = None
        if isinstance(fig, dict):
            cap = fig.get("caption")
        elif fig:
            cap = str(fig)
        cap = (cap or "").strip()
        if cap:
            yield Micro(
                text=cap,
                section_path=path,
                block_type=BlockType.FIGURE_CAPTION,
            )


def _maybe_split_micro(
    text: str,
    path: Tuple[str, ...],
    btype: BlockType,
    params: ChunkParams,
    tk_count,
) -> Iterator[Micro]:
    # if text too big, sentence split until each piece <= max_tokens_per_micro
    if tk_count(text) <= params.max_tokens_per_micro:
        yield Micro(text=text, section_path=path, block_type=btype)
        return
    # sentence split
    sentences = _split_sentences(text)
    if not sentences:
        yield Micro(text=text, section_path=path, block_type=btype)
        return
    acc: List[str] = []
    acc_tokens = 0
    for s in sentences:
        st = tk_count(s)
        if acc and acc_tokens + st > params.max_tokens_per_micro:
            yield Micro(text=" ".join(acc), section_path=path, block_type=btype)
            acc, acc_tokens = [s], st
        else:
            acc.append(s)
            acc_tokens += st
    if acc:
        yield Micro(text=" ".join(acc), section_path=path, block_type=btype)


def _walk_sections(
    sections: Sequence[dict] | None,
    path_prefix: Tuple[str, ...],
    params: ChunkParams,
    tk_count,
) -> Iterator[Micro]:
    if not sections:
        return
    for sec in sections:
        heading = (sec.get("heading") or "").strip()
        path = path_prefix + ((heading,) if heading else tuple())
        # body containers at this level
        yield from _yield_micro_from_container(sec, path, params, tk_count)
        # recurse
        yield from _walk_sections(sec.get("subsections") or [], path, params, tk_count)


# -----------------
# Packing
# -----------------


@dataclass
class Packed:
    section_path: Tuple[str, ...]
    block_types: List[BlockType]
    text: str  # body-only
    token_count: int


def _pack_micros(
    micros: Sequence[Micro], params: ChunkParams, tk_count
) -> List[Packed]:
    """Pack ordered micro-blocks into chunks without crossing section boundaries."""
    if not micros:
        return []

    chunks: List[Packed] = []
    i = 0
    while i < len(micros):
        # start new chunk
        start = i
        section = micros[i].section_path
        tok = 0
        texts: List[str] = []
        btypes: List[BlockType] = []
        while i < len(micros) and micros[i].section_path == section:
            mtok = tk_count(micros[i].text)
            if texts and tok + mtok > params.target_tokens:
                break
            texts.append(micros[i].text)
            btypes.append(micros[i].block_type)
            tok += mtok
            i += 1
        # emit chunk
        body = "\n\n".join(texts).strip()
        if body:
            chunks.append(
                Packed(
                    section_path=section,
                    block_types=btypes,
                    text=body,
                    token_count=tok,
                )
            )
        # overlap: step back to include trailing context, unless at end or section change next
        if i >= len(micros):
            break
        # Only create overlap within the same section
        if micros[i].section_path != section:
            continue
        # Walk backwards accumulating ~overlap_tokens
        back_tokens = 0
        j = i - 1
        while j >= start and back_tokens < params.overlap_tokens:
            back_tokens += tk_count(micros[j].text)
            j -= 1
        # restart at the max(start, j+1)
        i = max(start, j + 1)
        if i <= start:
            # Ensure forward progress
            i = start + 1

    return chunks


# -----------------
# Public API
# -----------------

_DEF_MODEL = None  # set to your embedding model name if you want tiktoken specificity


def chunk_article(parsed: dict, params: Optional[ChunkParams] = None) -> List[Chunk]:
    """Create structure-aware chunks for one article JSON.

    Args:
        parsed: dict output of your parser (see chat spec).
        params: optional overrides for chunking behavior.

    Returns:
        List[Chunk] instances (not committed). `chunk_uid` is filled and deterministic.
    """
    params = params or ChunkParams()
    tk_count = _get_tokenizer(_DEF_MODEL)

    # Handle article_id - parser outputs string URLs, we need an integer
    raw_id = parsed.get("id")
    if isinstance(raw_id, str):
        # Use hash of URL/ID as integer article_id
        article_id = abs(hash(raw_id)) % (2**31)  # Ensure positive 32-bit int
    elif raw_id is not None:
        try:
            article_id = int(raw_id)
        except (ValueError, TypeError):
            # Fallback for invalid numeric values
            article_id = abs(hash(str(raw_id))) % (2**31)
    else:
        article_id = 0
    title = (parsed.get("title") or "").strip() or "Untitled"
    canonical_url = (parsed.get("canonical_url") or parsed.get("url") or "").strip()

    # Build kv preview
    kv_preview = _kv_preview_from_infobox(parsed.get("infobox"), params.kv_preview_keys)

    # Parser version (for audit)
    parser_version = None
    prov = parsed.get("provenance") or {}
    if isinstance(prov, dict):
        parser_version = prov.get("parser_version")

    # Gather micro-blocks, preserving order within each container/section
    micros: List[Micro] = []

    # Lead
    lead_text = (parsed.get("lead") or "").strip()
    if lead_text:
        micros.append(
            Micro(
                text=lead_text,
                section_path=("lead",),
                block_type=BlockType.LEAD,
            )
        )

    # Preface (content before first heading)
    preface = parsed.get("preface") if isinstance(parsed.get("preface"), dict) else None
    micros.extend(
        _yield_micro_from_container(preface, ("preface",), params, tk_count) or []
    )

    # Sections (recursive)
    micros.extend(
        _walk_sections(parsed.get("sections") or [], tuple(), params, tk_count) or []
    )

    # Sort micros by section_path to ensure deterministic order
    micros.sort(key=lambda m: m.section_path)

    # Pack
    packed = _pack_micros(micros, params, tk_count)

    # Build final Chunks
    chunks: List[Chunk] = []

    def _encode_path(path: Tuple[str, ...]) -> str:
        return json.dumps(list(path), ensure_ascii=False)

    def _make_uid(
        url: str, path: Tuple[str, ...], chunk_index: int, embedding_input: str
    ) -> str:
        h = hashlib.sha1()
        h.update(url.encode("utf-8"))
        h.update(b"|")
        h.update("||".join(path).encode("utf-8"))
        h.update(b"|")
        h.update(str(chunk_index).encode("utf-8"))
        h.update(b"|")
        h.update(hashlib.sha1(embedding_input.encode("utf-8")).digest())
        return h.hexdigest()

    links_internal = parsed.get("links_internal") or []
    if isinstance(links_internal, list):
        links_payload = json.dumps(
            [str(x) for x in links_internal][:64], ensure_ascii=False
        )
    else:
        links_payload = None

    chunk_counter = 0
    for p in packed:
        # Decide predominant block type for the packed chunk
        if p.block_types:
            # Majority vote; ties fall back to PARAGRAPH
            most = Counter(p.block_types).most_common(1)[0][0]
            block_type = most or BlockType.PARAGRAPH
        else:
            block_type = BlockType.PARAGRAPH

        # Build embedding input (header + body)
        if params.include_header:
            header_lines = [f"Article: {title}"]
            if p.section_path:
                header_lines.append(
                    "Section: "
                    + " > ".join(
                        s
                        for s in p.section_path
                        if s and s != "lead" and s != "preface"
                    )
                )
            if kv_preview:
                header_lines.append(kv_preview)
            header_lines.append("---")
            embedding_input = "\n".join([*header_lines, p.text]).strip()
        else:
            embedding_input = p.text

        # Deterministic id
        chunk_uid = _make_uid(
            canonical_url or str(article_id),
            p.section_path,
            chunk_counter,
            embedding_input,
        )

        chunks.append(
            Chunk(
                chunk_uid=chunk_uid,
                article_id=article_id,
                article_title=title,
                canonical_url=canonical_url,
                section_path=_encode_path(p.section_path),
                block_type=block_type,
                chunk_index=chunk_counter,
                text=p.text,
                embedding_input=embedding_input,
                token_count=p.token_count,
                kv_preview=kv_preview,
                lead=(p.section_path == ("lead",)),
                parser_version=parser_version,
                links_out=links_payload,
                active=True,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        )
        chunk_counter += 1

    return chunks


# --------------
# Convenience CLI (optional)
# --------------
if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Chunk one parsed article from STDIN or file"
    )
    ap.add_argument(
        "--file", "-f", help="Path to parsed JSON file. If omitted, read STDIN."
    )
    ap.add_argument(
        "--pretty", action="store_true", help="Print a compact preview of chunks"
    )
    args = ap.parse_args()

    raw = None
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            raw = fh.read()
    else:
        raw = sys.stdin.read()

    article = json.loads(raw)
    out_chunks = chunk_article(article)

    # Print a minimal JSON preview per chunk (no vectors)
    preview = []
    for c in out_chunks:
        preview.append(
            {
                "chunk_uid": c.chunk_uid,
                "article_id": c.article_id,
                "title": c.article_title,
                "section_path": json.loads(c.section_path),
                "block_type": getattr(c.block_type, "value", str(c.block_type)),
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "kv_preview": c.kv_preview,
                "lead": c.lead,
                "text_preview": c.text[:200] + ("…" if len(c.text) > 200 else ""),
            }
        )

    if args.pretty:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(preview, ensure_ascii=False))
