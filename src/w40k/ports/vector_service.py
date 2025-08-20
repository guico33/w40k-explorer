"""Vector service port: unified retrieval and indexing surface used by the app.

This is the single abstraction for vector operations across inference and
ingestion. Adapters (e.g., Qdrant) implement both retrieval and index
management where applicable.
"""

from typing import Dict, List, Optional, Protocol, Tuple
from abc import abstractmethod


class VectorServicePort(Protocol):
    """Interface for vector retrieval and index management."""

    @abstractmethod
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
        """Search for semantically similar chunks and return payload dictionaries."""
        ...

    @abstractmethod
    def get_collection_info(self) -> Dict:
        """Return vector collection info for stats (vendor-specific fields ok)."""
        ...

    # Indexing/management operations (used by ingestion scripts)
    @abstractmethod
    def create_collection(self, recreate: bool = False) -> bool:
        """Create the collection (optionally deleting existing first)."""
        ...

    @abstractmethod
    def upsert_chunks(
        self,
        chunks_with_embeddings: List[tuple],
        batch_size: int = 100,
        show_progress: bool = False,
        ensure_collection: bool = False,
    ) -> Tuple[int, List[str]]:
        """Upsert (chunk, embedding) pairs. Returns (count, successful_chunk_uids)."""
        ...

    @abstractmethod
    def delete_points(self, point_ids: List[str]) -> bool:
        """Delete points by their IDs (adapter decides id format)."""
        ...

    @abstractmethod
    def delete_collection(self) -> bool:
        """Delete the current collection if it exists."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Lightweight availability check for the backing vector DB."""
        ...
