"""Bootstrap utilities for query engine initialization."""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.query_engine import SimpleQueryEngine

from openai import OpenAI

# Ensure project root is in path for imports (same as streamlit app)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.database.connection import DatabaseManager
from src.database.vector_operations import VectorOperations  
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import QdrantVectorStore


def validate_environment() -> Tuple[bool, Optional[str]]:
    """Validate required environment variables.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_vars = ["OPENAI_API_KEY", "OPENAI_LLM_MODEL", "EMBEDDING_MODEL"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        return False, error_msg
    
    return True, None


def setup_query_engine(
    db_path: str = "data/articles.db",
    initial_k: int = 60,
    min_score: Optional[float] = 0.2,
    max_context: int = 12,
    context_max_words: int = 200,
    expand_queries: int = 0,
    lower_threshold_on_empty: bool = True,
    active_only: bool = True,
) -> Tuple["SimpleQueryEngine", Dict]:
    """Initialize and return configured query engine with stats.
    
    Args:
        db_path: Path to SQLite database
        initial_k: Number of initial chunks to retrieve
        min_score: Minimum similarity score threshold
        max_context: Maximum number of context chunks
        context_max_words: Max words per context chunk
        expand_queries: Number of query expansions to generate (0 disables)
        lower_threshold_on_empty: Relax threshold when no hits found
        active_only: Only search active chunks
        
    Returns:
        Tuple of (engine, stats_dict)
        
    Raises:
        FileNotFoundError: If database doesn't exist
        Exception: If initialization fails
    """
    # Import here to avoid circular import issues
    from src.engine.query_engine import SimpleQueryEngine
    
    # Check database exists
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    # Initialize database manager
    db_manager = DatabaseManager(db_path)
    
    # Initialize Qdrant vector store
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_url and qdrant_api_key:
        vector_store = QdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        connection_info = f"Qdrant Cloud: {qdrant_url}"
    else:
        vector_store = QdrantVectorStore(
            host="localhost",
            port=6333,
        )
        connection_info = "Local Qdrant: localhost:6333"
    
    # Initialize embedding generator
    embedding_gen = EmbeddingGenerator(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL")
    )
    
    # Initialize vector operations
    vec_ops = VectorOperations(db_manager, vector_store, embedding_gen)
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create query engine
    engine = SimpleQueryEngine(
        vec_ops,
        openai_client,
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
        embedding_stats = vec_ops.get_embedding_stats()
        stats = {
            "chunks_count": embedding_stats.get("embeddings_in_qdrant", "unknown"),
            "coverage_percentage": embedding_stats.get("coverage_percentage", 0),
            "model": engine.model,
            "connection_info": connection_info,
            "db_path": db_path,
        }
    except Exception:
        stats = {
            "chunks_count": "unknown",
            "coverage_percentage": 0,
            "model": engine.model,
            "connection_info": connection_info,
            "db_path": db_path,
        }
    
    return engine, stats