"""Vector operations port interface - combines vector store and embedding operations."""

from typing import Dict, List, Optional, Protocol
from abc import abstractmethod


class VectorOperationsPort(Protocol):
    """Interface for combined vector storage and embedding operations.
    
    This interface combines the functionality needed for both generating
    embeddings and searching for similar content, which is what the
    QueryEngine actually needs.
    """

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
        """Search for semantically similar chunks.

        Args:
            query_text: Text to search for
            limit: Maximum number of results
            article_ids: Filter by specific article IDs
            block_types: Filter by block types
            lead_only: Filter for lead paragraphs only
            min_score: Minimum similarity score threshold
            active_only: If True, only search active chunks

        Returns:
            List of search results with chunk data and scores
        """
        ...

    @abstractmethod
    def get_embedding_stats(self) -> Dict:
        """Get comprehensive statistics about embeddings.

        Returns:
            Dictionary with embedding statistics
        """
        ...