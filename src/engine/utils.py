import json
from typing import List, Union, Optional

def format_section_path(section_path: Union[str, List[str], None]) -> str:
    """Format section path for display."""
    if section_path is None:
        return "Main"
    
    if isinstance(section_path, str):
        try:
            section_path = json.loads(section_path)
        except (json.JSONDecodeError, TypeError):
            return "Main"
    
    if not section_path or not isinstance(section_path, list):
        return "Main"
    
    # Limit to 3 levels and join with breadcrumb separator
    return " › ".join(str(s) for s in section_path[:3])

def truncate_text(text: str, max_words: int = 130) -> str:
    """Truncate text to approximately max_words."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."