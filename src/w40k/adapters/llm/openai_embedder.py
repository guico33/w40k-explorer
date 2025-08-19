"""OpenAI embedder adapter implementing the Embedder port."""

from typing import Optional

from ...infrastructure.rag.embeddings import EmbeddingGenerator
from ...ports.embedder import Embedder


class OpenAIEmbedder(EmbeddingGenerator):
    """OpenAI implementation of the Embedder port.

    This adapter wraps the existing EmbeddingGenerator to implement
    the Embedder protocol interface.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize OpenAI embedder adapter.

        Args:
            model: OpenAI embedding model to use
            api_key: OpenAI API key
            batch_size: Number of chunks to process in each batch
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Initial delay between retries (exponential backoff)
        """
        super().__init__(
            model=model,
            api_key=api_key,
            batch_size=batch_size,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    # The parent class already implements all required methods:
    # - generate_embedding()
    # - process_chunks()
    # - estimate_cost()
    # - get_stats()

    # No additional implementation needed - the adapter pattern
    # allows us to use the existing functionality while conforming
    # to the Embedder protocol interface.
