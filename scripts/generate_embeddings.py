#!/usr/bin/env python3
"""Script to generate embeddings and store them in Qdrant vector database."""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.database.connection import DatabaseManager
from src.database.vector_operations import VectorOperations
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import QdrantVectorStore

MODEL = os.getenv("EMBEDDING_MODEL")
if not MODEL:
    print("❌ EMBEDDING_MODEL environment variable is required.")
    sys.exit(1)

DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
vector_size: int = DIMS.get(MODEL, 1536)


def generate_embeddings(
    db_path: str = "data/articles.db",
    collection_name: str = "w40k_chunks",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    batch_size: int = 100,
    max_chunks: Optional[int] = None,
    force_recreate: bool = False,
    retry_failed: bool = False,
    dry_run: bool = False,
):
    """Generate embeddings for chunks and store in Qdrant.

    Args:
        db_path: Path to SQLite database
        collection_name: Qdrant collection name
        qdrant_host: Qdrant host (for local deployment)
        qdrant_port: Qdrant port (for local deployment)
        batch_size: Batch size for processing
        max_chunks: Maximum chunks to process (for testing)
        force_recreate: Recreate collection and all embeddings
        retry_failed: Retry chunks that previously failed
        dry_run: Preview operations without executing them

    Environment variables required:
        OPENAI_API_KEY: OpenAI API key
        EMBEDDING_MODEL: OpenAI embedding model to use
        QDRANT_URL: Qdrant cloud URL (optional, overrides host/port)
        QDRANT_API_KEY: Qdrant cloud API key (optional)
    """
    print("🚀 Warhammer 40k Embedding Generator")
    print("=" * 50)

    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable is required.")
        return 1

    if not os.getenv("EMBEDDING_MODEL"):
        print("❌ EMBEDDING_MODEL environment variable is required.")
        return 1

    # Initialize components
    print("🔧 Initializing components...")

    # Database
    db_manager = DatabaseManager(db_path)
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return 1

    # Qdrant Vector Store
    try:
        vector_store = QdrantVectorStore(
            collection_name=collection_name,
            host=qdrant_host,
            port=qdrant_port,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            vector_size=vector_size,
        )

        # Health check
        if not vector_store.health_check():
            print(
                "❌ Qdrant is not accessible. Please start Qdrant server or check connection."
            )
            return 1

        print("✅ Connected to Qdrant")

    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return 1

    # Embedding Generator
    try:
        embedding_generator = EmbeddingGenerator(batch_size=batch_size)
        print(
            f"✅ Initialized OpenAI embedding generator ({os.getenv('EMBEDDING_MODEL')})"
        )

    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        return 1

    # Vector Operations
    vector_ops = VectorOperations(db_manager, vector_store, embedding_generator)

    # Show initial statistics
    print("\n📊 Current Statistics:")
    stats = vector_ops.get_embedding_stats()
    print(f"   SQLite chunks (active): {stats['active_chunks_sqlite']:,}")
    print(f"   Qdrant embeddings: {stats['embeddings_in_qdrant']:,}")
    print(f"   Coverage: {stats['coverage_percentage']:.1f}%")

    if stats["coverage_percentage"] >= 99.9 and not force_recreate:
        print("\n✅ Embeddings are already up to date!")
        return 0

    # Estimate costs for remaining chunks
    chunks_needed = stats["active_chunks_sqlite"] - stats["embeddings_in_qdrant"]
    if max_chunks and chunks_needed > max_chunks:
        chunks_needed = max_chunks

    if chunks_needed > 0:
        cost_estimate = embedding_generator.estimate_cost(chunks_needed)
        print(f"\n💰 Cost Estimate:")
        print(f"   Chunks to process: {cost_estimate['num_chunks']:,}")
        print(f"   Estimated tokens: {cost_estimate['estimated_tokens']:,}")
        print(f"   Estimated cost: ${cost_estimate['estimated_cost_usd']:.4f}")

        # Show sample chunks in dry-run mode
        if dry_run:
            print("\n🔍 Sample chunks that would be processed:")
            chunks_to_preview = vector_ops.get_chunks_without_embeddings(
                limit=5, retry_failed=retry_failed
            )
            for i, chunk in enumerate(chunks_to_preview, 1):
                print(f"   {i}. {chunk.article_title} - {chunk.text[:100]}...")
                if chunk.embedding_failed_count > 0:
                    print(
                        f"      ⚠️  Previously failed {chunk.embedding_failed_count} times"
                    )

    # Handle dry-run mode
    if dry_run:
        print(f"\n🧪 DRY RUN MODE - No actual processing will occur")
        print(f"📊 Summary:")
        print(f"   Total active chunks: {stats['active_chunks_sqlite']:,}")
        print(f"   Already have embeddings: {stats['embeddings_in_qdrant']:,}")
        print(f"   Would process: {chunks_needed:,} chunks")
        if chunks_needed > 0:
            estimated_time = (chunks_needed / batch_size) * 2  # ~2 seconds per batch
            print(f"   Estimated processing time: {estimated_time/60:.1f} minutes")
        print(f"   Batch size: {batch_size}")
        print(f"   Embedding model: {os.getenv('EMBEDDING_MODEL')}")
        print(f"   Force recreate: {force_recreate}")
        print(f"   Retry failed: {retry_failed}")
        return 0

    # Confirm processing
    if not force_recreate and chunks_needed > 0:
        try:
            confirm = (
                input(f"\nGenerate embeddings for {chunks_needed:,} chunks? (y/n): ")
                .lower()
                .strip()
            )
            if confirm != "y":
                print("🛑 Embedding generation cancelled")
                return 0
        except EOFError:
            # Auto-confirm when running non-interactively
            print("🤖 Auto-confirming embedding generation (non-interactive mode)")

    # Process embeddings
    print("\n⏳ Starting embedding generation and storage...")

    try:
        results = vector_ops.sync_embeddings(
            force_recreate=force_recreate, retry_failed=retry_failed, max_chunks=max_chunks
        )

        if "error" in results:
            print(f"❌ Sync failed: {results['error']}")
            return 1

        # Show results
        print("\n" + "=" * 50)
        print("📊 Embedding Generation Complete!")
        print("=" * 50)

        sync_stats = results.get("sync_stats", {})
        print(f"Total processed: {sync_stats.get('total_processed', 0):,}")
        print(
            f"✅ Successful embeddings: {sync_stats.get('successful_embeddings', 0):,}"
        )
        print(f"📤 Successful uploads: {sync_stats.get('successful_uploads', 0):,}")
        print(f"🗑️  Removed inactive: {results.get('removed_inactive', 0):,}")

        final_stats = results.get("after_stats", {})
        print(f"\n🎯 Final Coverage: {final_stats.get('coverage_percentage', 0):.1f}%")
        print(
            f"📊 Total embeddings in Qdrant: {final_stats.get('embeddings_in_qdrant', 0):,}"
        )

        # Usage statistics
        if "api_calls" in sync_stats:
            print(f"\n📈 API Usage:")
            print(f"   API calls: {sync_stats.get('api_calls', 0):,}")
            print(f"   Total tokens: {sync_stats.get('total_tokens', 0):,}")
            print(f"   Success rate: {sync_stats.get('success_rate', 0):.1f}%")
            print(f"   Retries: {sync_stats.get('retry_count', 0):,}")

        # Collection info
        collection_info = sync_stats.get("collection_info", {})
        if collection_info:
            print(f"\n🗄️  Qdrant Collection Info:")
            print(f"   Status: {collection_info.get('status', 'unknown')}")
            print(f"   Vector size: {collection_info.get('vector_size', 0)}")
            print(f"   Distance metric: {collection_info.get('distance', 'unknown')}")
            print(f"   Segments: {collection_info.get('segments_count', 0)}")

        if final_stats.get("coverage_percentage", 0) >= 95:
            print("\n🎉 Embedding generation successful!")
            print("✅ Ready for Phase 4: RAG Pipeline Implementation!")
        else:
            print(f"\n⚠️  Coverage below 95%. Check for errors above.")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Embedding generation interrupted by user")
        print("Progress has been saved - you can resume by running this script again")
        return 1

    except Exception as e:
        print(f"\n❌ Error during embedding generation: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate embeddings and store in Qdrant"
    )

    # Database options
    parser.add_argument("--db", default="data/articles.db", help="SQLite database path")

    # Qdrant options
    parser.add_argument(
        "--collection", default="w40k_chunks", help="Qdrant collection name"
    )
    parser.add_argument(
        "--qdrant-host", default="localhost", help="Qdrant host (for local deployment)"
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant port (for local deployment)",
    )

    # Processing options
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for processing"
    )
    parser.add_argument(
        "--max-chunks", type=int, help="Maximum chunks to process (for testing)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force recreate all embeddings"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry chunks that previously failed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without executing them",
    )

    args = parser.parse_args()

    return generate_embeddings(
        db_path=args.db,
        collection_name=args.collection,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
        force_recreate=args.force,
        retry_failed=args.retry_failed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
