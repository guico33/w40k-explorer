"""Factory for creating and configuring application services."""

from typing import Dict, Optional, Tuple

from .settings import Settings, get_settings
from ..infrastructure.rag.embeddings import EmbeddingGenerator  
from ..adapters.llm.openai_client import OpenAIClient
from ..adapters.llm.anthropic_client import AnthropicClient
from ..adapters.vector_services.qdrant_service import QdrantVectorService
from ..usecases.answer import AnswerService
from ..ports.llm_client import LLMClient


def create_llm_client(settings: Settings) -> LLMClient:
    """Create and configure the appropriate LLM client based on settings.
    
    Args:
        settings: Settings instance with LLM provider configuration
        
    Returns:
        Configured LLM client instance
        
    Raises:
        ValueError: If provider configuration is invalid
    """
    # Validate provider configuration
    settings.validate_llm_provider()
    
    if settings.llm_provider == "openai":
        return OpenAIClient(
            api_key=settings.openai_api_key,
            timeout=settings.get_llm_timeout(),
        )
    elif settings.llm_provider == "anthropic":
        return AnthropicClient(
            api_key=settings.anthropic_api_key,
            timeout=settings.get_llm_timeout(),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def create_answer_service(
    initial_k: int = 60,
    min_score: Optional[float] = 0.2,
    max_context: int = 12,
    context_max_words: int = 200,
    expand_queries: int = 0,
    lower_threshold_on_empty: bool = True,
    active_only: bool = True,
    settings: Optional[Settings] = None,
) -> Tuple[AnswerService, Dict]:
    """Create and configure the answer service with all dependencies.
    
    Args:
        initial_k: Number of initial chunks to retrieve
        min_score: Minimum similarity score threshold
        max_context: Maximum number of context chunks
        context_max_words: Max words per context chunk
        expand_queries: Number of query expansions to generate
        lower_threshold_on_empty: Relax threshold when no hits found
        active_only: Only search active chunks
        settings: Settings instance (if None, creates new one)
        
    Returns:
        Tuple of (answer_service, stats_dict)
        
    Raises:
        Exception: If initialization fails
    """
    # Initialize settings
    if settings is None:
        settings = get_settings()
    # SQLite is not required for inference
    
    # Connection info
    if settings.is_qdrant_cloud():
        connection_info = f"Qdrant Cloud: {settings.qdrant_url}"
    else:
        connection_info = f"Local Qdrant: {settings.qdrant_host}:{settings.qdrant_port}"
    
    # Initialize embedding generator (always uses OpenAI for embeddings)
    # Note: Even when using Anthropic for LLM, we still use OpenAI for embeddings
    # as Anthropic doesn't provide embedding models
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key is required for embeddings, even when using Anthropic for LLM")
    
    embedding_gen = EmbeddingGenerator(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        max_retries=settings.max_retries,
        retry_delay=settings.retry_delay,
    )
    
    # Initialize vector service (creates its own Qdrant client)
    vector_ops = QdrantVectorService(
        embedding_gen,
        collection_name=settings.qdrant_collection_name,
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        vector_size=settings.vector_size,
    )
    
    # Initialize LLM client based on provider
    llm_client = create_llm_client(settings)
    
    # Create answer service
    answer_service = AnswerService(
        vector_operations=vector_ops,
        llm_client=llm_client,
        model=settings.get_llm_model(),
        initial_k=initial_k,
        max_context_chunks=max_context,
        min_similarity_score=min_score,
        context_max_words=context_max_words,
        query_expansion_n=expand_queries,
        lower_threshold_on_empty=lower_threshold_on_empty,
        active_only=active_only,
    )
    
    # Get vector store stats only (no SQLite-dependent coverage)
    try:
        col = vector_ops.get_collection_info() or {}
        stats = {
            "chunks_count": col.get("points_count", "unknown"),
            "provider": settings.llm_provider,
            "model": settings.get_llm_model(),
            "connection_info": connection_info,
        }
    except Exception as e:
        stats = {
            "chunks_count": "unknown",
            "provider": settings.llm_provider,
            "model": settings.get_llm_model(),
            "connection_info": connection_info,
            "error": str(e),
        }
    
    return answer_service, stats


def validate_environment() -> Tuple[bool, Optional[str]]:
    """Validate required environment variables.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        settings = get_settings()
        settings.validate_llm_provider()
        # Ensure embeddings configuration (OpenAI) is present since embeddings always use OpenAI
        # Use os.getenv here to avoid implicit .env loading during validation in tests
        import os as _os
        if not _os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is required for embeddings even if Anthropic is the LLM provider"
            )
        return True, None
    except Exception as e:
        return False, str(e)
