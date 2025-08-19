"""Qdrant adapter implementing the VectorStore port."""

from typing import Optional

from ...infrastructure.rag.qdrant_vector_store import QdrantVectorStore


class QdrantAdapter(QdrantVectorStore):
    """Qdrant implementation of the VectorStore port.

    This adapter wraps the existing QdrantVectorStore to implement
    the VectorStore protocol interface.
    """

    def __init__(
        self,
        collection_name: str = "w40k_chunks",
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: int = 1536,
    ):
        """Initialize Qdrant adapter.

        Args:
            collection_name: Name of the collection to store vectors
            host: Qdrant server host (for local deployment)
            port: Qdrant server port (for local deployment)
            url: Full URL for Qdrant cloud (overrides host/port)
            api_key: API key for Qdrant cloud
            vector_size: Dimension of embedding vectors
        """
        from qdrant_client.http.models import Distance

        super().__init__(
            collection_name=collection_name,
            host=host,
            port=port,
            url=url,
            api_key=api_key,
            vector_size=vector_size,
            distance=Distance.COSINE,
        )

    # The parent class already implements all required methods:
    # - search()
    # - upsert_chunks()
    # - get_collection_info()
    # - create_collection()
    # - delete_points()

    # No additional implementation needed - the adapter pattern
    # allows us to use the existing functionality while conforming
    # to the VectorStore protocol interface.
