"""RAG utilities for parsing KV previews and link lists.

These helpers are used when preparing payloads for vector storage and can be
reused by other infrastructure components.
"""

from __future__ import annotations

import ast
import html
import json
import re
from typing import Dict, List, Optional, Union


def parse_links_out(links_str: Optional[str]) -> List[Dict[str, str]]:
    """Parse links_out string into structured JSON array.

    Args:
        links_str: JSON string containing Python dict representations

    Returns:
        List of link dictionaries with text and href keys
    """
    if not links_str:
        return []

    try:
        # Parse the JSON array of string representations
        link_strings = json.loads(links_str)
        links: List[Dict[str, str]] = []
        for link_str in link_strings:
            if not link_str:
                continue
            try:
                # Parse each string representation of dict using ast.literal_eval
                link_dict = ast.literal_eval(link_str)
                if (
                    isinstance(link_dict, dict)
                    and "text" in link_dict
                    and "href" in link_dict
                ):
                    # Only include links with actual content
                    if link_dict["text"].strip() and link_dict["href"].strip():
                        links.append(
                            {
                                "text": link_dict["text"].strip(),
                                "href": link_dict["href"].strip(),
                            }
                        )
            except (ValueError, SyntaxError):
                # Skip malformed individual link entries
                continue
        return links
    except (json.JSONDecodeError, TypeError):
        # Return empty list for completely malformed data
        return []


KV = Dict[str, Union[str, List[str]]]

_PAIR_RE = re.compile(
    r"""
    \s*                          # leading space
    (?P<key>[^=;]+?)             # key = anything up to '=' or ';'
    \s*=\s*
    (?P<val>[^;]*)               # value = anything up to next ';' (greedy)
    \s*(?:;|$)                   # ends with ';' or EOS
""",
    re.VERBOSE,
)


def parse_kv_preview(kv_str: Optional[str]) -> KV:
    """
    Parse a 'key=value; key2=value2' string to a dict.
    - Trims whitespace
    - Unescapes HTML entities
    - Strips surrounding quotes in values
    - Handles trailing semicolons and duplicate keys (collates to list)
    """
    if not kv_str:
        return {}

    out: KV = {}

    for m in _PAIR_RE.finditer(kv_str):
        raw_key = m.group("key").strip()
        raw_val = m.group("val").strip()

        if not raw_key:
            continue

        # Unescape HTML entities and strip surrounding quotes
        key = html.unescape(raw_key).strip().strip('\"\'“”‘’')
        val = html.unescape(raw_val).strip().strip('\"\'“”‘’')

        if not val:
            continue

        # Collate duplicates into a list
        if key in out:
            existing_value = out[key]
            if isinstance(existing_value, list):
                existing_value.append(val)
            else:
                out[key] = [existing_value, val]
        else:
            out[key] = val

    return out


def normalize_section_path(section_path: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalize a section_path value to a list of strings.

    Accepts either a JSON-encoded string or a list. Returns an empty list when
    not parseable.
    """
    if section_path is None:
        return []
    if isinstance(section_path, list):
        return [str(x) for x in section_path]
    if isinstance(section_path, str):
        try:
            data = json.loads(section_path)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            return []
    return []


def create_qdrant_filters(
    article_ids: Optional[List[int]] = None,
    block_types: Optional[List[str]] = None,
    lead_only: Optional[bool] = None,
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    active_only: bool = True,
) -> Dict:
    """Build Qdrant filter conditions.

    Returns a dict compatible with `qdrant_client.http.models.Filter(**dict)`.
    """
    conditions: List[Dict] = []

    if article_ids:
        conditions.append({"key": "article_id", "match": {"any": article_ids}})

    if block_types:
        conditions.append({"key": "block_type", "match": {"any": block_types}})

    if lead_only is not None:
        conditions.append({"key": "lead", "match": {"value": lead_only}})

    if min_tokens is not None:
        conditions.append({"key": "token_count", "range": {"gte": min_tokens}})

    if max_tokens is not None:
        conditions.append({"key": "token_count", "range": {"lte": max_tokens}})

    if active_only:
        conditions.append({"key": "active", "match": {"value": True}})

    return {"must": conditions} if conditions else {}
