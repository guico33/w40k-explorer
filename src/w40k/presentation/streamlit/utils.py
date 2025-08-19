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

    Args:
        answer: Answer text containing citations like [1], [2], etc.

    Returns:
        List of unique citation IDs in order of first appearance
    """
    citation_pattern = r"\[(\d+)\]"
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

    Args:
        answer: Answer text with citations
        id_mapping: Mapping from original ID to new ID

    Returns:
        Answer text with remapped citation numbers
    """
    remapped_answer = answer
    for original_id, sequential_id in id_mapping.items():
        remapped_answer = remapped_answer.replace(
            f"[{original_id}]", f"[{sequential_id}]"
        )
    return remapped_answer


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

    # Create lookup from citation ID to citation object
    citations_by_id = {
        citation.get("id"): citation
        for citation in citations
        if citation.get("id") is not None
    }

    # Find citation order in text
    ordered_citation_ids = extract_citation_order_from_text(answer)

    # Filter to only citations that actually exist
    valid_citation_ids = [cid for cid in ordered_citation_ids if cid in citations_by_id]

    if not valid_citation_ids:
        return answer, "No valid citations found"

    # Create mapping from original IDs to sequential numbers (1, 2, 3, etc.)
    original_to_sequential = {
        original_id: i + 1 for i, original_id in enumerate(valid_citation_ids)
    }

    # Remap citation numbers in answer text
    remapped_answer = remap_citations_in_text(answer, original_to_sequential)

    # Format sources in order they appear in text
    formatted_sources = []
    for i, original_id in enumerate(valid_citation_ids):
        citation = citations_by_id[original_id]
        sequential_id = i + 1
        formatted_source = format_single_source(citation, sequential_id)
        formatted_sources.append(formatted_source)

    # Join sources with separator
    sources_text = separator.join(formatted_sources)

    return remapped_answer, sources_text