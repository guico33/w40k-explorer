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
    """Remap citation numbers in answer text using provided mapping.

    Accepts both `[id:NUMBER]` and `[NUMBER]` in the original text and outputs
    normalized `[NUMBER]` with sequential IDs.
    """
    remapped = answer
    for original_id, sequential_id in id_mapping.items():
        # Replace tagged form first to avoid double-touching
        remapped = remapped.replace(f"[id:{original_id}]", f"[{sequential_id}]")
        # Replace bare form
        remapped = remapped.replace(f"[{original_id}]", f"[{sequential_id}]")
    return remapped


def format_single_source(citation: Dict, sequential_id: int) -> str:
    """Format a single citation source with sequential numbering.

    Args:
        citation: Citation dictionary with title, section, url, etc.
        sequential_id: Sequential number to display (1, 2, 3, etc.)

    Returns:
        Formatted source string with number and clickable link
    """
    title = citation.get("title", "Unknown")
    section = citation.get("section", "")
    url = citation.get("url", "")

    # Create display text
    if section:
        source_text = f"**{title}** › {section}"
    else:
        source_text = f"**{title}**"

    # Add URL with section anchor if available
    if url:
        final_url = create_url_with_fragment(url, section)
        source_text = f"[{source_text}]({final_url})"

    # Add sequential number
    return f"**[{sequential_id}]** {source_text}"


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

    # Prefer the order provided by citations list (already reflects citations_used)
    # Build mapping: original context id -> sequential display id
    original_to_seq: Dict[int, int] = {}
    formatted_sources: List[str] = []

    for idx, citation in enumerate(citations, start=1):
        cid = citation.get("id")
        if cid is None:
            # Skip items without an id
            continue
        original_to_seq[int(cid)] = idx
        formatted_sources.append(format_single_source(citation, idx))

    # Remap in-text citations to sequential [1], [2], ...
    remapped_answer = remap_citations_in_text(answer, original_to_seq)

    # Join sources with separator
    sources_text = separator.join(formatted_sources)

    return remapped_answer, sources_text
