"""Vector operations for managing embeddings and vector service integration."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from sqlmodel import select

from .connection import DatabaseManager
from .models import Chunk

from ..rag.embeddings import EmbeddingGenerator
from ...ports.vector_service import VectorServicePort
from ..rag.utils import create_qdrant_filters


class VectorOperations:
    """High-level operations for managing embeddings and vector storage."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager],
        vector_service: VectorServicePort,
        embedding_generator: EmbeddingGenerator,
    ):
        """Initialize vector operations.

        Args:
            db_manager: Database manager for SQLite operations (None for Qdrant-only mode)
            vector_service: Unified vector service instance
            embedding_generator: Embedding generator instance
        """
        self.db_manager = db_manager
        self.vector_service = vector_service
        self.embedding_generator = embedding_generator

    def get_chunks_without_embeddings(
        self, limit: int = 1000, retry_failed: bool = False
    ) -> List[Chunk]:
        """Get chunks that don't have embeddings yet.

        Args:
            limit: Maximum number of chunks to return
            retry_failed: If True, include chunks that previously failed

        Returns:
            List of chunks that need embeddings
        """
        if self.db_manager is None:
            raise RuntimeError("SQLite is not configured. This method requires database access.")
        
        with next(self.db_manager.get_session()) as session:
            query = select(Chunk).where(Chunk.active == True)

            if retry_failed:
                # Include chunks that failed or never tried
                query = query.where(
                    (Chunk.has_embedding == False) | (Chunk.embedding_failed_count > 0)
                )
                print(f"🔄 Including previously failed chunks in processing")
            else:
                # Only chunks that haven't been processed yet
                query = query.where(Chunk.has_embedding == False)

            chunks = session.exec(query.limit(limit)).all()

            if retry_failed and chunks:
                # Sort by failure count (process chunks with fewer failures first)
                chunks = sorted(chunks, key=lambda c: c.embedding_failed_count)

            return list(chunks)

    def generate_and_store_embeddings(
        self,
        chunks: Optional[List[Chunk]] = None,
        batch_size: int = 100,
        max_chunks: Optional[int] = None,
        retry_failed: bool = False,
    ) -> Dict:
        """Generate embeddings for chunks and store them in Qdrant.

        Args:
            chunks: Specific chunks to process (if None, get chunks needing embeddings)
            batch_size: Batch size for processing embeddings and uploads
            max_chunks: Maximum number of chunks to process
            retry_failed: If True, include chunks that previously failed

        Returns:
            Dictionary with processing statistics
        """
        if self.db_manager is None:
            raise RuntimeError("SQLite is not configured. This method requires database access.")
        
        # Get chunks to process
        if chunks is None:
            print("🔍 Finding chunks that need embeddings...")
            chunks = self.get_chunks_without_embeddings(
                limit=max_chunks or 10000, retry_failed=retry_failed
            )

        if not chunks:
            print("✅ All chunks already have embeddings!")
            return {
                "total_processed": 0,
                "successful_embeddings": 0,
                "successful_uploads": 0,
            }

        if max_chunks and len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            print(f"🔢 Limited to {max_chunks:,} chunks for processing")

        print(f"📊 Processing {len(chunks):,} chunks in batches of {batch_size}")

        # Estimate cost
        cost_estimate = self.embedding_generator.estimate_cost(len(chunks))
        print(f"💰 Estimated cost: ${cost_estimate['estimated_cost_usd']:.4f}")

        # Initialize tracking variables
        total_successful_embeddings = []
        total_failed_chunks = []
        total_uploaded = 0

        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size

            print(
                f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)"
            )

            # Generate embeddings for this batch
            chunks_with_embeddings = self.embedding_generator.process_chunks(
                batch_chunks
            )

            # Process results and update database tracking for this batch
            batch_successful_embeddings = []
            batch_failed_chunks = []

            with next(self.db_manager.get_session()) as session:
                for chunk, embedding in chunks_with_embeddings:
                    if embedding is not None:
                        batch_successful_embeddings.append((chunk, embedding))
                        # Mark as successful in database
                        chunk.has_embedding = True
                        chunk.embedding_generated_at = datetime.now()
                        chunk.last_embedding_error = None
                        session.add(chunk)
                    else:
                        batch_failed_chunks.append(chunk)
                        # Mark as failed in database
                        chunk.embedding_failed_count += 1
                        chunk.last_embedding_error = (
                            f"Failed to generate embedding at {datetime.now()}"
                        )
                        session.add(chunk)

                # Before committing, detach chunks from session to avoid lazy loading issues
                for chunk, embedding in batch_successful_embeddings:
                    session.expunge(chunk)  # Detach from session but keep data

                session.commit()

            print(f"   ✅ Generated {len(batch_successful_embeddings)} embeddings")
            if batch_failed_chunks:
                print(f"   ❌ Failed to generate {len(batch_failed_chunks)} embeddings")

            # Store this batch in the vector service if we have successful embeddings
            if batch_successful_embeddings:
                print(
                    f"   📤 Uploading {len(batch_successful_embeddings)} embeddings to vector service..."
                )
                batch_uploaded_count, batch_successful_point_ids = self.vector_service.upsert_chunks(
                    batch_successful_embeddings, batch_size=batch_size
                )
                total_uploaded += batch_uploaded_count

                # Update database for failed uploads in this batch
                if batch_uploaded_count < len(batch_successful_embeddings):
                    print(
                        f"   ⚠️  Only {batch_uploaded_count}/{len(batch_successful_embeddings)} uploaded"
                    )
                    # Mark failed uploads by checking which chunks are NOT in successful_point_ids
                    successful_ids_set = set(batch_successful_point_ids)
                    with next(self.db_manager.get_session()) as session:
                        for chunk, _ in batch_successful_embeddings:
                            if chunk.chunk_uid not in successful_ids_set:
                                # This chunk failed to upload
                                chunk.has_embedding = False
                                chunk.embedding_failed_count += 1
                                chunk.last_embedding_error = (
                                    f"Failed to upload to vector service at {datetime.now()}"
                                )
                                session.add(chunk)
                        session.commit()

            # Accumulate totals
            total_successful_embeddings.extend(batch_successful_embeddings)
            total_failed_chunks.extend(batch_failed_chunks)

        # Final summary
        print(f"🎯 Total embeddings generated: {len(total_successful_embeddings):,}")
        if total_failed_chunks:
            print(f"❌ Total failed embeddings: {len(total_failed_chunks):,}")
        print(f"📤 Total uploaded to vector service: {total_uploaded:,}")

        if not total_successful_embeddings:
            return {
                "total_processed": len(chunks),
                "successful_embeddings": 0,
                "successful_uploads": 0,
                "failed_embeddings": len(total_failed_chunks),
            }

        # Get final statistics
        stats = self.embedding_generator.get_stats()
        additional_stats = {
            "total_processed": len(chunks),
            "successful_uploads": total_uploaded,
            "failed_embeddings": len(total_failed_chunks),
            "collection_info": self.vector_service.get_collection_info(),
        }
        stats.update(additional_stats)

        return stats

    def search_similar_chunks(
        self,
        query_text: str,
        limit: int = 10,
        article_ids: Optional[List[int]] = None,
        block_types: Optional[List[str]] = None,
        lead_only: Optional[bool] = None,
        min_score: Optional[float] = None,
        active_only: bool = True,
    ) -> List[Dict]:
        """Search for semantically similar chunks.

        Args:
            query_text: Text to search for
            limit: Maximum number of results
            article_ids: Filter by specific article IDs
            block_types: Filter by block types
            lead_only: Filter for lead paragraphs only
            min_score: Minimum similarity score threshold
            active_only: If True, only search active chunks

        Returns:
            List of search results with chunk data and scores
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []

        # Create filter conditions
        filter_conditions = create_qdrant_filters(
            article_ids=article_ids,
            block_types=block_types,
            lead_only=lead_only,
            active_only=active_only,
        )

        # Search using active vector service
        return self.vector_service.search_similar_chunks(
            query_text=query_text,
            limit=limit,
            article_ids=article_ids,
            block_types=block_types,
            lead_only=lead_only,
            min_score=min_score,
            active_only=active_only,
        )

    def get_embedding_stats(self) -> Dict:
        """Get comprehensive statistics about embeddings.

        Returns:
            Dictionary with embedding statistics
        """
        # Get vector service stats
        collection_info = self.vector_service.get_collection_info() or {}
        embeddings_count = int(collection_info.get("points_count", 0))

        # Get SQLite chunk stats if available
        total_chunks = active_chunks = None
        if self.db_manager is not None:
            with next(self.db_manager.get_session()) as session:
                total_chunks = len(session.exec(select(Chunk)).all())
                active_chunks = len(
                    session.exec(select(Chunk).where(Chunk.active == True)).all()
                )

        # Calculate coverage
        if active_chunks is not None and active_chunks > 0:
            # SQLite available: calculate actual coverage
            coverage = (embeddings_count / active_chunks) * 100
        elif embeddings_count > 0:
            # Vector-service-only mode: assume 100% coverage of available data
            coverage = 100.0
        else:
            # No data available
            coverage = 0.0

        return {
            "total_chunks_sqlite": total_chunks,
            "active_chunks_sqlite": active_chunks,
            "embeddings_in_qdrant": embeddings_count,
            "coverage_percentage": float(coverage),
            "collection_status": collection_info.get("status", "unknown"),
            "vector_size": collection_info.get("vector_size", 0),
            "distance_metric": collection_info.get("distance", "unknown"),
            "indexed_vectors": collection_info.get("indexed_vectors_count", 0),
            "segments_count": collection_info.get("segments_count", 0),
        }

    def cleanup_inactive_embeddings(self) -> int:
        """Remove embeddings for chunks that are marked as inactive.

        Returns:
            Number of embeddings removed
        """
        if self.db_manager is None:
            print("ℹ️  SQLite not configured - skipping inactive cleanup")
            return 0
        
        print("🧹 Finding inactive chunks to remove from vector service...")

        with next(self.db_manager.get_session()) as session:
            inactive_chunks = session.exec(
                select(Chunk).where(Chunk.active == False)
            ).all()

            if not inactive_chunks:
                print("✅ No inactive chunks found")
                return 0

            inactive_uids = [chunk.chunk_uid for chunk in inactive_chunks]
            print(f"🗑️  Removing {len(inactive_uids)} inactive chunk embeddings...")

            success = self.vector_service.delete_points(inactive_uids)
            if success:
                print(f"✅ Removed {len(inactive_uids)} embeddings")
                return len(inactive_uids)
            else:
                print("❌ Failed to remove inactive embeddings")
                return 0

    def sync_embeddings(
        self,
        force_recreate: bool = False,
        retry_failed: bool = False,
        max_chunks: Optional[int] = None,
    ) -> Dict:
        """Synchronize embeddings between SQLite and the vector service.

        Args:
            force_recreate: If True, recreate all embeddings
            retry_failed: If True, retry previously failed chunks
            max_chunks: Maximum number of chunks to process

        Returns:
            Dictionary with sync statistics
        """
        if self.db_manager is None:
            raise RuntimeError("SQLite is not configured. This method requires database access.")
        
        print("🔄 Synchronizing embeddings between SQLite and vector service...")

        if force_recreate:
            print("🗑️  Force recreate: clearing existing collection...")
            self.vector_service.create_collection(recreate=True)
            # Reset all embedding tracking when force recreating
            with next(self.db_manager.get_session()) as session:
                chunks = session.exec(select(Chunk)).all()
                for chunk in chunks:
                    chunk.has_embedding = False
                    chunk.embedding_generated_at = None
                    chunk.embedding_failed_count = 0
                    chunk.last_embedding_error = None
                    session.add(chunk)
                session.commit()
                print(f"🔄 Reset embedding tracking for {len(chunks):,} chunks")

        # Ensure collection exists
        if not self.vector_service.create_collection():
            return {"error": "Failed to create vector collection"}

        # Get statistics before sync
        before_stats = self.get_embedding_stats()
        print(f"📊 Before sync: {before_stats['embeddings_in_qdrant']:,} embeddings in vector service")

        # Generate embeddings for missing chunks
        sync_stats = self.generate_and_store_embeddings(
            retry_failed=retry_failed, max_chunks=max_chunks
        )

        # Clean up inactive embeddings
        removed_count = self.cleanup_inactive_embeddings()

        # Get statistics after sync
        after_stats = self.get_embedding_stats()
        print(f"📊 After sync: {after_stats['embeddings_in_qdrant']:,} embeddings in vector service")
        print(f"📈 Coverage: {after_stats['coverage_percentage']:.1f}%")

        return {
            "before_stats": before_stats,
            "after_stats": after_stats,
            "sync_stats": sync_stats,
            "removed_inactive": removed_count,
            "final_coverage": after_stats["coverage_percentage"],
        }
