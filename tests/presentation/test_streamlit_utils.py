from w40k.presentation.streamlit.utils import (
    format_sources_with_sequential_numbering,
)


def test_sources_formatting_remaps_brackets_and_orders_sources():
    # Given an answer using bare [ID] citations
    answer = "Horus was named Warmaster. [5] Another sentence. [11]"
    citations = [
        {"id": 5, "title": "Horus", "section": "Biography", "url": "https://x/horus"},
        {"id": 11, "title": "Warmaster", "section": "Appointment", "url": "https://x/warmaster"},
    ]

    remapped, sources = format_sources_with_sequential_numbering(answer, citations)

    # Expect remapped citations to be [1] and [2]
    assert "[1]" in remapped and "[2]" in remapped
    # Sources should list [1] then [2] in order
    assert sources.startswith("**[1]**") and "**[2]**" in sources

