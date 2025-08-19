"""Factory for creating and configuring application services."""

from pathlib import Path
from typing import Dict, Optional, Tuple

from .settings import Settings, get_settings
from ..infrastructure.database.connection import DatabaseManager
from ..infrastructure.rag.embeddings import EmbeddingGenerator  
from ..infrastructure.rag.qdrant_vector_store import QdrantVectorStore
from ..adapters.llm.openai_client import OpenAIClient
from ..adapters.llm.anthropic_client import AnthropicClient
from ..adapters.persistence.vector_operations_adapter import VectorOperationsAdapter
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
    db_path: str = "data/articles.db",
    initial_k: int = 60,
    min_score: Optional[float] = 0.2,
    max_context: int = 12,
    context_max_words: int = 200,
    expand_queries: int = 0,
    lower_threshold_on_empty: bool = True,
    active_only: bool = True,
    use_sqlite: bool = True,
    settings: Optional[Settings] = None,
) -> Tuple[AnswerService, Dict]:
    """Create and configure the answer service with all dependencies.
    
    Args:
        db_path: Path to SQLite database
        initial_k: Number of initial chunks to retrieve
        min_score: Minimum similarity score threshold
        max_context: Maximum number of context chunks
        context_max_words: Max words per context chunk
        expand_queries: Number of query expansions to generate
        lower_threshold_on_empty: Relax threshold when no hits found
        active_only: Only search active chunks
        use_sqlite: Whether to use SQLite (if False, uses Qdrant-only mode)
        settings: Settings instance (if None, creates new one)
        
    Returns:
        Tuple of (answer_service, stats_dict)
        
    Raises:
        Exception: If initialization fails
    """
    # Initialize settings
    if settings is None:
        settings = get_settings()
    # Decide if we will use SQLite
    will_use_sqlite = use_sqlite and Path(db_path).exists()
    
    # Initialize database manager if available
    db_manager = None
    if will_use_sqlite:
        db_manager = DatabaseManager(db_path)
    
    # Initialize Qdrant vector store
    if settings.is_qdrant_cloud():
        vector_store = QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection_name,
            vector_size=settings.vector_size,
        )
        connection_info = f"Qdrant Cloud: {settings.qdrant_url}"
    else:
        vector_store = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            vector_size=settings.vector_size,
        )
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
    
    # Initialize vector operations using the adapter
    vector_ops = VectorOperationsAdapter(db_manager, vector_store, embedding_gen)
    
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
    
    # Get collection stats
    stats = {}
    try:
        embedding_stats = vector_ops.get_embedding_stats()
        stats = {
            "chunks_count": embedding_stats.get("embeddings_in_qdrant", "unknown"),
            "coverage_percentage": embedding_stats.get("coverage_percentage", 0),
            "provider": settings.llm_provider,
            "model": settings.get_llm_model(),
            "connection_info": connection_info,
            "db_path": db_path if will_use_sqlite else "N/A (Qdrant-only mode)",
        }
    except Exception as e:
        stats = {
            "chunks_count": "unknown",
            "coverage_percentage": 0,
            "provider": settings.llm_provider,
            "model": settings.get_llm_model(),
            "connection_info": connection_info,
            "db_path": db_path if will_use_sqlite else "N/A (Qdrant-only mode)",
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
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for embeddings even if Anthropic is the LLM provider"
            )
        return True, None
    except Exception as e:
        return False, str(e)
