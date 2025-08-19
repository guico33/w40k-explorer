"""Vector store port interface."""

from typing import Dict, List, Optional, Protocol
from abc import abstractmethod


class VectorStore(Protocol):
    """Interface for vector storage and similarity search operations."""

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filter_conditions: Optional filters for search
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with payload and scores
        """
        ...

    @abstractmethod
    def upsert_chunks(
        self, 
        chunks_with_embeddings: List[tuple],
        batch_size: int = 100
    ) -> tuple[int, List[str]]:
        """Store chunks with their embeddings.
        
        Args:
            chunks_with_embeddings: List of (chunk, embedding) tuples
            batch_size: Batch size for uploads
            
        Returns:
            Tuple of (uploaded_count, successful_point_ids)
        """
        ...

    @abstractmethod
    def get_collection_info(self) -> Optional[Dict]:
        """Get information about the vector collection.
        
        Returns:
            Dictionary with collection statistics or None if unavailable
        """
        ...

    @abstractmethod
    def create_collection(self, recreate: bool = False) -> bool:
        """Create or ensure vector collection exists.
        
        Args:
            recreate: If True, recreate the collection
            
        Returns:
            True if successful, False otherwise
        """
        ...

    @abstractmethod
    def delete_points(self, point_ids: List[str]) -> bool:
        """Delete specific points from collection.
        
        Args:
            point_ids: List of point IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        ...