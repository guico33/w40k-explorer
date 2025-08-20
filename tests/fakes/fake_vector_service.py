from typing import Dict, List, Optional

from w40k.ports.vector_service import VectorServicePort


class FakeVectorService(VectorServicePort):
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

    def get_collection_info(self) -> Dict:
        # Minimal collection info for stats
        return {
            "name": "test_collection",
            "points_count": len(self._hits),
            "vector_size": 1536,
            "distance": "cosine",
            "segments_count": 1,
            "status": "green",
            "indexed_vectors_count": len(self._hits),
        }

    # New no-op/indexing methods to satisfy the expanded port
    def create_collection(self, recreate: bool = False) -> bool:
        return True

    def upsert_chunks(self, chunks_with_embeddings, batch_size: int = 100, show_progress: bool = False, ensure_collection: bool = False):  # type: ignore[override]
        # Accept any input and pretend all were upserted successfully
        try:
            count = len(chunks_with_embeddings)
        except Exception:
            count = 0
        # Return (count, list_of_chunk_uids) per contract
        uids = []
        for pair in (chunks_with_embeddings or []):
            try:
                chunk, _ = pair
                uids.append(getattr(chunk, "chunk_uid", None) or "")
            except Exception:
                continue
        return count, uids

    def delete_points(self, point_ids: List[str]) -> bool:
        return True

    def delete_collection(self) -> bool:
        return True

    def health_check(self) -> bool:
        return True
