#!/usr/bin/env python3
"""Interactive CLI chat interface for testing W40K query engine."""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Configure logging for debug mode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openai import OpenAI
from src.engine.query_engine import SimpleQueryEngine, QueryResult
from src.database.vector_operations import VectorOperations
from src.database.connection import DatabaseManager
from src.rag.vector_store import QdrantVectorStore  
from src.rag.embeddings import EmbeddingGenerator


def validate_environment():
    """Validate required environment variables."""
    required_vars = ["OPENAI_API_KEY", "OPENAI_LLM_MODEL", "EMBEDDING_MODEL"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease check your .env file and ensure all variables are set.")
        sys.exit(1)
    
    print("✅ Environment variables validated")


def setup_query_engine(
    db_path: str = "data/articles.db",
    verbose: bool = False,
    initial_k: int = 60,
    min_score: float | None = 0.2,
    max_context: int = 12,
    context_max_words: int = 200,
    expand_queries: int = 0,
    lower_threshold_on_empty: bool = True,
    active_only: bool = True,
) -> SimpleQueryEngine:
    """Initialize and return configured query engine."""
    
    print(f"📚 Initializing query engine with database: {db_path}")
    
    # Check database exists
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        print("Please run the data pipeline scripts first to create the database.")
        sys.exit(1)
    
    try:
        # Initialize database manager
        if verbose:
            print("  🔌 Connecting to SQLite database...")
        db_manager = DatabaseManager(db_path)
        
        # Initialize Qdrant vector store
        if verbose:
            print("  🔌 Connecting to Qdrant vector store...")
        
        # Try cloud URL first, then fallback to local
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url and qdrant_api_key:
            print(f"  🌐 Using Qdrant Cloud: {qdrant_url}")
            vector_store = QdrantVectorStore(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
        else:
            print("  🏠 Using local Qdrant: localhost:6333")
            vector_store = QdrantVectorStore(
                host="localhost",
                port=6333,
            )
        
        # Initialize embedding generator
        if verbose:
            print("  🧠 Initializing embedding generator...")
        embedding_gen = EmbeddingGenerator(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("EMBEDDING_MODEL")
        )
        
        # Initialize vector operations
        if verbose:
            print("  🔍 Setting up vector operations...")
        vec_ops = VectorOperations(db_manager, vector_store, embedding_gen)
        
        # Initialize OpenAI client
        if verbose:
            print("  🤖 Connecting to OpenAI...")
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create query engine
        if verbose:
            print("  ⚙️  Creating query engine...")
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
        
        print(f"✅ Query engine initialized with model: {engine.model}")
        
        # Get collection stats if available
        try:
            stats = vec_ops.get_embedding_stats()
            chunks_count = stats.get("embeddings_in_qdrant", "unknown")
            coverage = stats.get("coverage_percentage", 0)
            print(f"📊 Vector database: {chunks_count} chunks ({coverage:.1f}% coverage)")
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not get database stats: {e}")
        
        return engine
        
    except Exception as e:
        print(f"❌ Failed to initialize query engine: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def display_result(result: QueryResult, verbose: bool = False):
    """Display query result in a formatted way."""
    
    print("\n" + "="*60)
    
    # Display answer
    print(f"📝 **Answer** (Confidence: {result.confidence:.2f}, {result.query_time_ms}ms):")
    print(f"{result.answer}")
    
    # Display citations
    if result.citations:
        print(f"\n📚 **Sources** ({len(result.citations)} used):")
        for cite in result.citations:
            if cite.get("url"):
                print(f"• {cite['title']} › {cite['section']} ({cite['url']})")
            else:
                print(f"• {cite['title']} › {cite['section']}")
    else:
        print("\n📚 **Sources**: None")
    
    # Display confidence indicator
    if result.confidence < 0.5:
        print("\n⚠️  Low confidence answer - limited sources available")
    elif result.confidence < 0.7:
        print("\nℹ️  Moderate confidence - additional sources might help")
    
    # Display error if any
    if result.error:
        print(f"\n❌ **Error**: {result.error}")
    
    # Display cache info
    if result.from_cache:
        print("\n✨ Served from cache")
    
    # Verbose debug info
    if verbose:
        print(f"\n🔍 **Debug Info**:")
        print(f"  - Sources processed: {result.sources_used}")
        print(f"  - Citations generated: {len(result.citations)}")
        print(f"  - Query time: {result.query_time_ms}ms")
        print(f"  - From cache: {result.from_cache}")
        if result.error:
            print(f"  - Error: {result.error}")
    
    print("="*60)


def chat_loop(engine: SimpleQueryEngine, verbose: bool = False):
    """Main interactive chat loop."""
    
    print("\n🤖 Warhammer 40K Knowledge Base - CLI Chat Interface")
    print("💬 Type your question (or 'quit' to exit):\n")
    
    while True:
        try:
            # Get user input
            question = input("> ").strip()
            
            # Handle exit commands
            if question.lower() in ['quit', 'exit', 'q', '']:
                if not question:
                    continue  # Empty input, just continue
                print("👋 Goodbye!")
                break
            
            # Handle help command
            if question.lower() in ['help', '?']:
                print("\n📋 Available commands:")
                print("  - Ask any question about Warhammer 40K lore")
                print("  - 'quit', 'exit', 'q' - Exit the chat")
                print("  - 'help', '?' - Show this help message")
                continue
            
            # Process the query
            print("🤔 Searching archives...")
            
            result = engine.query(question)
            
            # Display results
            display_result(result, verbose)
            
            print(f"\n💬 Type your next question (or 'quit' to exit):")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            print("\n💬 Try another question:")


def main():
    """Main entry point."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat interface for testing W40K query engine"
    )
    parser.add_argument(
        "--db", 
        default="data/articles.db",
        help="Path to SQLite database (default: data/articles.db)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose debug information"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="Initial number of chunks to retrieve (default: 60)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum similarity score threshold (omit to disable)",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=12,
        help="Maximum number of context chunks (default: 12)",
    )
    parser.add_argument(
        "--context-max-words",
        type=int,
        default=200,
        help="Max words per context chunk (default: 200)",
    )
    parser.add_argument(
        "--expand-queries",
        type=int,
        default=0,
        help="Number of query expansions to generate (0 disables)",
    )
    parser.add_argument(
        "--no-relax-threshold",
        action="store_true",
        help="Do not relax threshold when no hits are found",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include inactive chunks in vector search",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query and exit (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting W40K Query Engine CLI")
    print("=" * 40)
    
    # Validate environment
    validate_environment()
    
    # Setup query engine
    # Configure logging verbosity for engine
    if args.verbose:
        logging.getLogger("src.engine").setLevel(logging.DEBUG)
        logging.getLogger("engine").setLevel(logging.DEBUG)
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    engine = setup_query_engine(
        db_path=args.db,
        verbose=args.verbose,
        initial_k=args.k,
        min_score=args.min_score,
        max_context=args.max_context,
        context_max_words=args.context_max_words,
        expand_queries=args.expand_queries,
        lower_threshold_on_empty=not args.no_relax_threshold,
        active_only=not args.include_inactive,
    )
    
    # Handle direct query mode or interactive chat
    if args.query:
        # Non-interactive mode: run single query and exit
        print(f"\n🔍 Processing query: {args.query}")
        print("🤔 Searching archives...")
        
        try:
            result = engine.query(args.query)
            display_result(result, args.verbose)
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        # Interactive chat mode
        try:
            chat_loop(engine, args.verbose)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
