"""Embedder port interface."""

from typing import Dict, List, Optional, Protocol, Union
from abc import abstractmethod


class Embedder(Protocol):
    """Interface for generating text embeddings."""

    @abstractmethod
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        ...

    @abstractmethod
    def process_chunks(self, chunks) -> List[tuple]:
        """Process multiple chunks and generate embeddings.
        
        Args:
            chunks: List of chunk objects to process
            
        Returns:
            List of (chunk, embedding) tuples
        """
        ...

    @abstractmethod
    def estimate_cost(self, num_chunks: int) -> Dict[str, Union[int, float]]:
        """Estimate cost for processing given number of chunks.
        
        Args:
            num_chunks: Number of chunks to estimate for
            
        Returns:
            Dictionary with cost estimation details
        """
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        ...