#!/usr/bin/env python3
"""Script to generate chunks from parsed articles and populate the Chunk table."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tqdm import tqdm

from src.database.connection import DatabaseManager
from src.database.operations import ChunkDatabase
from src.database.models import Article, ScrapingStatus
from src.rag.chunker import chunk_article, ChunkParams
from sqlmodel import select


def generate_chunks(
    db_path: str = "data/articles.db",
    batch_size: int = 50,
    force_rechunk: bool = False,
    max_articles: int | None = None
):
    """Generate chunks for all parsed articles.
    
    Args:
        db_path: Path to database file
        batch_size: Number of articles to process per batch
        force_rechunk: If True, regenerate chunks for all articles
        max_articles: Maximum number of articles to process (for testing)
    """
    print("🧩 Warhammer 40k Chunk Generator")
    print("=" * 50)
    
    # Initialize database
    db_manager = DatabaseManager(db_path)
    db_manager.create_tables()  # Ensure Chunk table exists
    
    chunk_db = ChunkDatabase(db_manager)
    
    # Get articles that need chunking
    print("🔍 Finding articles to chunk...")
    
    if force_rechunk:
        # Get all parsed articles
        with next(db_manager.get_session()) as session:
            articles_to_chunk = session.exec(
                select(Article).where(
                    (Article.status == ScrapingStatus.SUCCESS) &
                    (Article.parsed_json != None)  # type: ignore[arg-type]
                )
            ).all()
        print(f"📊 Force rechunk: {len(articles_to_chunk):,} articles")
    else:
        # Get articles needing chunks
        article_ids_needing_chunks = chunk_db.get_articles_needing_chunks(
            limit=max_articles or 10000
        )
        
        if not article_ids_needing_chunks:
            print("✅ All articles already have chunks!")
            return
        
        # Get full Article objects
        with next(db_manager.get_session()) as session:
            from sqlmodel import col
            articles_to_chunk = session.exec(
                select(Article).where(col(Article.id).in_(article_ids_needing_chunks))
            ).all()
        
        print(f"📊 Articles needing chunks: {len(articles_to_chunk):,}")
    
    if not articles_to_chunk:
        print("✅ No articles to process!")
        return
    
    # Apply max_articles limit if specified
    if max_articles and len(articles_to_chunk) > max_articles:
        articles_to_chunk = articles_to_chunk[:max_articles]
        print(f"🔢 Limited to {max_articles:,} articles for testing")
    
    # Show chunking parameters
    params = ChunkParams()
    print(f"🔧 Chunk parameters:")
    print(f"   Target tokens: {params.target_tokens}")
    print(f"   Overlap tokens: {params.overlap_tokens}")
    print(f"   Max tokens per micro: {params.max_tokens_per_micro}")
    
    # Get current chunk statistics
    current_stats = chunk_db.get_chunking_stats()
    print(f"🗄️  Current chunks: {current_stats['total_chunks']:,}")
    
    # Confirm processing
    if not force_rechunk:
        try:
            confirm = input(f"\nGenerate chunks for {len(articles_to_chunk):,} articles? (y/n): ").lower().strip()
            if confirm != 'y':
                print("🛑 Chunk generation cancelled")
                return
        except EOFError:
            # Auto-confirm when running non-interactively (e.g., in automation)
            print("🤖 Auto-confirming chunk generation (non-interactive mode)")
            pass
    
    # Process articles
    print("\n⏳ Starting chunk generation...")
    
    stats = {
        "total": len(articles_to_chunk),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "total_chunks_generated": 0,
        "total_chunks_saved": 0,
    }
    
    with tqdm(total=len(articles_to_chunk), desc="Generating chunks", unit="articles") as pbar:
        for i, article in enumerate(articles_to_chunk):
            try:
                # Parse JSON content
                if not article.parsed_json:
                    stats["skipped"] += 1
                    pbar.set_description(f"Skipped {article.title[:30]}...")
                    continue
                
                parsed_doc = json.loads(article.parsed_json)
                
                # Validate parsed document
                if not parsed_doc or not parsed_doc.get("title"):
                    stats["failed"] += 1
                    print(f"⚠️  Invalid parsed document for: {article.url}")
                    continue
                
                # Clear existing chunks if force rechunking
                if force_rechunk and article.id is not None:
                    cleared_count = chunk_db.clear_chunks_for_article(article.id)
                    if cleared_count > 0:
                        print(f"🗑️  Cleared {cleared_count} existing chunks for {article.title}")
                
                # Generate chunks
                chunks = chunk_article(parsed_doc, params)
                
                if not chunks:
                    stats["failed"] += 1
                    print(f"⚠️  No chunks generated for: {article.title}")
                    continue
                
                # Update chunk article_id to match database article ID
                if article.id is not None:
                    for chunk in chunks:
                        chunk.article_id = article.id
                else:
                    stats["failed"] += 1
                    print(f"⚠️  Article has no ID: {article.title}")
                    continue
                
                # Save chunks to database
                saved_count = chunk_db.save_chunks(chunks)
                
                if saved_count > 0:
                    stats["success"] += 1
                    stats["total_chunks_generated"] += len(chunks)
                    stats["total_chunks_saved"] += saved_count
                    
                    # Log progress for larger articles
                    if len(chunks) > 10:
                        print(f"✅ '{article.title}' → {len(chunks)} chunks")
                else:
                    stats["failed"] += 1
                    print(f"⚠️  Failed to save chunks for: {article.title}")
                
            except json.JSONDecodeError as e:
                stats["failed"] += 1
                print(f"❌ JSON decode error for {article.url}: {e}")
            except Exception as e:
                stats["failed"] += 1
                error_type = type(e).__name__
                print(f"❌ {error_type} for {article.url}: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")
                
                # For debugging: log the first few problematic articles
                if stats["failed"] <= 5:
                    print(f"   🔍 Debug info - Article ID: {article.id}, Title: {article.title}")
            
            # Update progress
            pbar.update(1)
            pbar.set_postfix({
                "Success": stats["success"],
                "Failed": stats["failed"],
                "Chunks": stats["total_chunks_saved"]
            })
            
            # Batch progress reporting
            if (i + 1) % batch_size == 0:
                chunks_in_batch = stats["total_chunks_saved"]
                print(f"📈 Processed {i + 1:,}/{len(articles_to_chunk):,} articles → {chunks_in_batch:,} chunks")
    
    # Show final results
    print("\n" + "=" * 50)
    print("📊 Chunk Generation Complete!")
    print("=" * 50)
    print(f"Total processed: {stats['total']:,}")
    print(f"✅ Successful: {stats['success']:,}")
    print(f"❌ Failed: {stats['failed']:,}")
    print(f"⏭️  Skipped: {stats['skipped']:,}")
    print(f"🧩 Chunks generated: {stats['total_chunks_generated']:,}")
    print(f"💾 Chunks saved: {stats['total_chunks_saved']:,}")
    
    success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    if stats['total_chunks_generated'] > 0:
        avg_chunks_per_article = stats['total_chunks_generated'] / max(1, stats['success'])
        print(f"📊 Average chunks per article: {avg_chunks_per_article:.1f}")
    
    # Show updated database statistics
    print(f"\n🗄️  Final Database Statistics:")
    final_stats = chunk_db.get_chunking_stats()
    print(f"Total chunks: {final_stats['total_chunks']:,}")
    print(f"Unique articles: {final_stats['unique_articles']:,}")
    print(f"Average chunks per article: {final_stats['avg_chunks_per_article']:.1f}")
    print(f"Average token count: {final_stats['avg_token_count']:.1f}")
    
    # Token distribution
    if final_stats['token_distribution']:
        print(f"\n📏 Token Distribution:")
        for range_name, count in final_stats['token_distribution'].items():
            percentage = (count / final_stats['total_chunks']) * 100
            print(f"   {range_name}: {count:,} ({percentage:.1f}%)")
    
    # Block type distribution
    if final_stats['block_type_distribution']:
        print(f"\n📊 Block Type Distribution:")
        sorted_types = sorted(
            final_stats['block_type_distribution'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for block_type, count in sorted_types:
            percentage = (count / final_stats['total_chunks']) * 100
            print(f"   {block_type}: {count:,} ({percentage:.1f}%)")
    
    # Database size info
    db_info = db_manager.get_db_info()
    if db_info['exists']:
        print(f"\n💽 Database size: {db_info['size_mb']:.1f} MB")
    
    if stats['failed'] > 0:
        print(f"\n⚠️  {stats['failed']} articles failed to process")
        print("Check the logs above for specific errors")
    else:
        print("\n🎉 All articles processed successfully!")
    
    print("\n✅ Ready for Phase 5: Embedding Generation!")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate chunks from parsed articles")
    parser.add_argument("--db", default="data/articles.db", help="Database file path")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for progress reporting")
    parser.add_argument("--force", action="store_true", help="Force regenerate all chunks")
    parser.add_argument("--max", type=int, help="Maximum number of articles to process (for testing)")
    
    args = parser.parse_args()
    
    try:
        generate_chunks(
            db_path=args.db,
            batch_size=args.batch_size,
            force_rechunk=args.force,
            max_articles=args.max
        )
    except KeyboardInterrupt:
        print("\n⏹️  Chunk generation interrupted by user")
        print("Progress has been saved - you can resume by running this script again")
        return 1
    except Exception as e:
        print(f"\n❌ Error during chunk generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())