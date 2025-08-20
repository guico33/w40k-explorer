#!/usr/bin/env python3
"""Test script for embedding generation and Qdrant integration."""

import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path for new architecture
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from w40k.config.settings import get_settings
from w40k.infrastructure.database.connection import DatabaseManager
from w40k.infrastructure.rag.embeddings import (
    EmbeddingGenerator,
    validate_embedding_dimensions,
)
from w40k.adapters.vector_services.qdrant_service import QdrantVectorService


def test_embeddings():
    """Test embedding generation and Qdrant integration on sample chunks."""

    print("🧪 Testing Embedding Generation & Qdrant Integration")
    print("=" * 60)

    # Initialize database
    db_path = "data/articles.db"
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return 1

    db_manager = DatabaseManager(db_path)

    # Get sample chunks
    print("📊 Getting sample chunks...")
    with next(db_manager.get_session()) as session:
        from sqlmodel import select

        from w40k.infrastructure.database.models import Chunk

        sample_chunks = session.exec(
            select(Chunk).where(Chunk.active == True).limit(5)
        ).all()

    if not sample_chunks:
        print("❌ No chunks found in database")
        return 1

    print(f"✅ Found {len(sample_chunks)} sample chunks")
    for i, chunk in enumerate(sample_chunks):
        print(f"   {i+1}. {chunk.article_title} ({chunk.token_count} tokens)")

    # Test embedding generation
    print("\n🔧 Testing OpenAI embedding generation...")

    settings = get_settings()

    try:
        embedding_generator = EmbeddingGenerator(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            batch_size=5,
        )
        print("✅ Initialized OpenAI client")
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        return 1

    # Generate embeddings for sample
    print("🚀 Generating embeddings...")
    chunks_with_embeddings = embedding_generator.process_chunks(list(sample_chunks))

    # Validate results
    embeddings = [emb for _, emb in chunks_with_embeddings if emb is not None]
    successful_count = len(embeddings)

    print(f"✅ Generated {successful_count}/{len(sample_chunks)} embeddings")

    if successful_count == 0:
        print("❌ No embeddings were generated successfully")
        return 1

    # Validate dimensions
    print("🔍 Validating embedding dimensions...")
    if validate_embedding_dimensions(embeddings, model=settings.embedding_model):
        print("✅ Embedding dimensions match the selected model")
    else:
        print("❌ Embedding dimension validation failed")
        return 1

    # Show embedding stats
    stats = embedding_generator.get_stats()
    print(f"\n📈 Embedding Statistics:")
    print(f"   API calls: {stats['api_calls']}")
    print(f"   Total tokens: {stats['total_tokens']:,}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Avg tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")

    # Test Qdrant integration
    print("\n🗄️  Testing Qdrant integration...")

    try:
        if settings.is_qdrant_cloud():
            vector_service = QdrantVectorService(
                embedding_generator,
                collection_name="test_w40k_chunks",
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                vector_size=1536,
            )
        else:
            vector_service = QdrantVectorService(
                embedding_generator,
                collection_name="test_w40k_chunks",
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                vector_size=1536,
            )

        if not vector_service.health_check():
            print("⚠️  Qdrant not available - skipping vector store tests")
            return 0

        print("✅ Connected to Qdrant")

    except Exception as e:
        print(f"⚠️  Qdrant connection failed: {e}")
        return 0

    # Create test collection
    print("🏗️  Creating test collection...")
    if not vector_service.create_collection(recreate=True):
        print("❌ Failed to create collection")
        return 1

    print("✅ Created test collection")

    # Upload embeddings (filter out None embeddings)
    print("📤 Uploading embeddings to Qdrant...")
    valid_chunks_with_embeddings = [
        (chunk, embedding)
        for chunk, embedding in chunks_with_embeddings
        if embedding is not None
    ]
    uploaded_count, _ = vector_service.upsert_chunks(
        valid_chunks_with_embeddings, batch_size=5
    )

    if uploaded_count != successful_count:
        print(f"⚠️  Only {uploaded_count}/{successful_count} embeddings uploaded")
    else:
        print(f"✅ Uploaded {uploaded_count} embeddings")

    # Test search
    print("\n🔍 Testing semantic search...")

    if successful_count > 0:
        # Use first chunk's text as query
        query_text = sample_chunks[0].text[:100] + "..."
        print(f"   Query: {query_text}")

        # Use service high-level search by text
        search_results = vector_service.search_similar_chunks(
            query_text=sample_chunks[0].text, limit=3
        )

        print(f"✅ Found {len(search_results)} search results")
        for i, result in enumerate(search_results):
            title = result.get("article_title", "Unknown")
            score = result.get("score", 0.0)
            print(f"   {i+1}. Score: {score:.3f} - {title}")

    # Get collection info
    print("\n📊 Collection Statistics:")
    info = vector_service.get_collection_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Clean up test collection
    print("\n🧹 Cleaning up test collection...")
    try:
        if vector_service.delete_collection():
            print("✅ Cleaned up test collection")
        else:
            print("⚠️  Cleanup warning: delete_collection returned False")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

    print("\n🎉 All tests completed successfully!")
    print("✅ Ready to run full embedding generation")

    return 0


def main():
    """Main entry point."""
    try:
        return test_embeddings()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
