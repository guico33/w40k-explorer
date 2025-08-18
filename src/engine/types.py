"""Type definitions for the query engine."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QueryResult:
    """Structured result from query engine."""

    answer: str
    citations: List[Dict[str, str]]
    confidence: float
    sources_used: int
    citations_used: List[int] = field(default_factory=list)
    context_items: List[Dict] = field(default_factory=list)
    from_cache: bool = False
    query_time_ms: int = 0
    error: Optional[str] = None