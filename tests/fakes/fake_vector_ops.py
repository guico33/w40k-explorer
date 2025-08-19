from typing import Dict, List, Optional

from w40k.ports.vector_operations import VectorOperationsPort


class FakeVectorOperations(VectorOperationsPort):
    """Deterministic VectorOperations fake for tests (no DB, no network)."""

    def __init__(self, hits: Optional[List[Dict]] = None):
        # Provide a small default corpus if none supplied
        self._hits = hits or [
            {
                "chunk_uid": "uid-0",
                "article_title": "Horus",
                "canonical_url": "https://example.org/horus",
                "section_path": ["Biography"],
                "lead": True,
                "kv_data": {"Title": "Warmaster"},
                "text": "Horus was named Warmaster.",
                "score": 0.95,
            },
            {
                "chunk_uid": "uid-1",
                "article_title": "Emperor of Mankind",
                "canonical_url": "https://example.org/emperor",
                "section_path": ["Great Crusade"],
                "lead": False,
                "kv_data": {},
                "text": "The Emperor appointed Horus as Warmaster.",
                "score": 0.90,
            },
        ]
        self.queries: List[str] = []

    def search_similar_chunks(
        self,
        query_text: str,
        limit: int = 10,
        article_ids: Optional[List[int]] = None,
        block_types: Optional[List[str]] = None,
        lead_only: Optional[bool] = None,
        min_score: Optional[float] = None,
        active_only: bool = True,
    ) -> List[Dict]:
        self.queries.append(query_text)
        hits = list(self._hits)
        if min_score is not None:
            hits = [h for h in hits if h.get("score", 0) >= min_score]
        return hits[:limit]

    def get_embedding_stats(self) -> Dict:
        return {
            "embeddings_in_qdrant": 2,
            "coverage_percentage": 100.0,
        }
