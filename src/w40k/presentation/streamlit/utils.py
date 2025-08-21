"""UI utility functions for citation formatting and display."""

import re
from typing import Dict, List, Tuple


def create_url_with_fragment(url: str, section: str) -> str:
    """Create URL with section anchor if section exists and isn't generic.

    Args:
        url: Base URL
        section: Section name

    Returns:
        URL with fragment anchor if applicable
    """
    if not section or section.lower() in ["main", "preface", ""]:
        return url

    # Convert section to URL fragment (spaces to underscores, handle special chars)
    section_fragment = section.replace(" ", "_").replace("›", "_").strip()
    # Remove any characters that aren't valid in URL fragments
    section_fragment = "".join(c for c in section_fragment if c.isalnum() or c in "_-")
    # Clean up multiple consecutive underscores
    while "__" in section_fragment:
        section_fragment = section_fragment.replace("__", "_")
    section_fragment = section_fragment.strip("_")

    if section_fragment:
        return f"{url}#{section_fragment}"
    return url


def extract_citation_order_from_text(answer: str) -> List[int]:
    """Extract citation numbers from answer text in order of appearance.

    Supports both `[id:NUMBER]` and `[NUMBER]` formats.

    Args:
        answer: Answer text containing citations like [5] or [id:5]

    Returns:
        List of unique citation IDs in order of first appearance
    """
    # Match either [id:5] or [5]
    citation_pattern = r"\[(?:id:)?(\d+)\]"
    citation_matches = re.findall(citation_pattern, answer)

    # Get unique citation IDs in order of first appearance
    seen_ids = set()
    ordered_citation_ids = []
    for match in citation_matches:
        citation_id = int(match)
        if citation_id not in seen_ids:
            seen_ids.add(citation_id)
            ordered_citation_ids.append(citation_id)

    return ordered_citation_ids


def remap_citations_in_text(answer: str, id_mapping: Dict[int, int]) -> str:
    """Remap citation markers in the answer text using the provided mapping.

    Accepts both `[id:NUMBER]` and `[NUMBER]` and rewrites to normalized
    sequential `[NUMBER]` based on the mapping of original context IDs to
    display order. Uses regex to remap per‑occurrence and avoid partial matches.

    Args:
        answer: The generated answer containing citation markers.
        id_mapping: Mapping from original context id -> sequential display id.

    Returns:
        Answer text with citation markers remapped to `[1]..[n]`.
    """
    pattern = re.compile(r"\[(?:id:)?(\d+)\]")

    def _repl(match: re.Match[str]) -> str:
        try:
            original = int(match.group(1))
        except Exception:
            return match.group(0)
        new = id_mapping.get(original)
        return f"[{new}]" if new is not None else match.group(0)

    return pattern.sub(_repl, answer)


def format_single_source(citation: Dict, sequential_id: int) -> str:
    """Format a single citation source with sequential numbering and anchor.

    Outputs HTML so we can set an in-page anchor target (id="ref-N") that
    superscript citation links can jump to, and a clickable external link.

    Args:
        citation: Citation dictionary with title, section, url, etc.
        sequential_id: Sequential number to display (1, 2, 3, etc.)

    Returns:
        HTML string with an in-page anchor and a clickable external link.
    """
    title = citation.get("title", "Unknown")
    section = citation.get("section", "")
    url = citation.get("url", "")

    # Create display text
    display_text = f"{title}"
    if section:
        display_text = f"{title} › {section}"

    # Add URL with section anchor if available
    link_html = display_text
    if url:
        final_url = create_url_with_fragment(url, section)
        link_html = f'<a href="{final_url}" target="_blank" rel="noopener noreferrer">{display_text}</a>'

    # Include an in-page anchor target so superscripts can link here
    return f'<span id="ref-{sequential_id}"></span><strong>[{sequential_id}]</strong> {link_html}'


def format_sources_with_sequential_numbering(
    answer: str, citations: List[Dict], separator: str = " • "
) -> Tuple[str, str]:
    """Format citation sources with sequential numbers and remap answer citations.

    This function:
    1. Extracts citation order from the answer text
    2. Remaps citations to sequential numbers (1, 2, 3, etc.)
    3. Formats sources with clickable links and separators
    4. Returns both the remapped answer and formatted sources

    Args:
        answer: Answer text containing citations like [9], [11], etc.
        citations: List of citation dictionaries
        separator: String to separate sources (default: " • ")

    Returns:
        Tuple of (remapped_answer, formatted_sources_string)
    """
    if not citations:
        return answer, "No sources available"

    # Determine ordering of citations by appearance in the answer.
    # This makes the remapped [1], [2], ... align with the first time each
    # citation id appears in the text (Wikipedia-like behavior).
    order_in_text = extract_citation_order_from_text(answer)

    # Map id -> citation (first matching entry wins); keep original list for fallbacks
    by_id: Dict[int, Dict] = {}
    for c in citations:
        cid = c.get("id")
        if isinstance(cid, int) and cid not in by_id:
            by_id[cid] = c

    # Build final ordered list: first those that appear in text, then any remaining
    ordered: List[Dict] = []
    seen_ids: set[int] = set()
    for cid in order_in_text:
        if cid in by_id and cid not in seen_ids:
            ordered.append(by_id[cid])
            seen_ids.add(cid)
    for c in citations:
        cid = c.get("id")
        if isinstance(cid, int) and cid not in seen_ids:
            ordered.append(c)
            seen_ids.add(cid)

    # Build mapping: original context id -> sequential display id (1-based)
    original_to_seq: Dict[int, int] = {}
    formatted_sources: List[str] = []
    for idx, citation in enumerate(ordered, start=1):
        cid = citation.get("id")
        if isinstance(cid, int):
            original_to_seq[cid] = idx
            formatted_sources.append(format_single_source(citation, idx))

    # Remap in-text citations to sequential [1], [2], ...
    remapped_answer = remap_citations_in_text(answer, original_to_seq)

    # Join sources with separator
    sources_text = separator.join(formatted_sources)

    return remapped_answer, sources_text


def style_citation_superscripts(text: str) -> str:
    """Render in-text numeric citations like "[1]" as HTML superscripts.

    Converts bare numeric markers to `<sup class="citation-sup">[n]</sup>` while
    avoiding Markdown link syntax like `[label](url)`. Intended for display only;
    does not change underlying citation mapping.

    Args:
        text: Rendered answer text with `[1]`, `[2]`, ... citation markers.

    Returns:
        HTML string with superscripted citation markers.
    """

    # Match [digits] not immediately followed by '(' to avoid Markdown links.
    pattern = re.compile(r'\[(\d+)\](?!\()')
    # Wrap the superscript in an anchor linking to the in-page source target
    return pattern.sub(r'<a class="citation-link" href="#ref-\1"><sup class="citation-sup">[\1]</sup></a>', text)


def build_seq_to_url_map(answer: str, citations: List[Dict]) -> Dict[int, str]:
    """Build a mapping from sequential citation number to external URL (with fragment).

    Orders citations by first appearance in the answer, then appends any remaining.

    Args:
        answer: Answer text containing citation markers.
        citations: List of citation dicts with keys: id, title, section, url.

    Returns:
        Mapping from sequential number (1-based) to final external URL string.
    """
    order_in_text = extract_citation_order_from_text(answer)
    by_id: Dict[int, Dict] = {}
    for c in citations:
        cid = c.get("id")
        if isinstance(cid, int) and cid not in by_id:
            by_id[cid] = c

    ordered: List[Dict] = []
    seen: set[int] = set()
    for cid in order_in_text:
        if cid in by_id and cid not in seen:
            ordered.append(by_id[cid])
            seen.add(cid)
    for c in citations:
        cid = c.get("id")
        if isinstance(cid, int) and cid not in seen:
            ordered.append(c)
            seen.add(cid)

    seq_to_url: Dict[int, str] = {}
    for idx, c in enumerate(ordered, start=1):
        url = c.get("url") or ""
        section = c.get("section") or ""
        if url:
            seq_to_url[idx] = create_url_with_fragment(url, section)
    return seq_to_url


def style_citation_superscripts_with_links(text: str, seq_to_url: Dict[int, str]) -> str:
    """Render in-text citations as superscripts linked to external sources.

    If a sequential number doesn't have a URL in the mapping, the marker is left
    unchanged.

    Args:
        text: The remapped answer text containing [1]..[n].
        seq_to_url: Mapping from sequential number to external URL.

    Returns:
        HTML string with superscripted, clickable citations.
    """
    pattern = re.compile(r'\[(\d+)\](?!\()')

    def _repl(m: re.Match[str]) -> str:
        try:
            n = int(m.group(1))
        except Exception:
            return m.group(0)
        href = seq_to_url.get(n)
        if not href:
            return m.group(0)
        return f'<a class="citation-link" href="{href}" target="_blank" rel="noopener noreferrer"><sup class="citation-sup">[{n}]</sup></a>'

    return pattern.sub(_repl, text)
