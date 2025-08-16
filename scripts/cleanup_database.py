#!/usr/bin/env python3
"""Script to clean up incomplete database entries after scraping."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database.connection import DatabaseManager
from src.database.operations import ArticleDatabase
from src.database.models import Article, ScrapingStatus
from sqlmodel import select


def cleanup_database(db_path: str = "data/articles.db", dry_run: bool = True):
    """Clean up incomplete database entries.
    
    Args:
        db_path: Path to database file
        dry_run: If True, only show what would be deleted
    """
    print("🧹 Database Cleanup Tool")
    print("=" * 40)
    
    # Initialize database
    db_manager = DatabaseManager(db_path)
    article_db = ArticleDatabase(db_manager)
    
    # Get database stats before cleanup
    print("📊 Database stats before cleanup:")
    stats = article_db.get_scraping_stats()
    for status, count in stats.items():
        if status != "success_rate":
            print(f"  {status.capitalize()}: {count:,}")
    
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    
    # Find all incomplete entries (avoiding double counting)
    with next(db_manager.get_session()) as session:
        # Get all articles that should be deleted (any non-SUCCESS or empty content)
        articles_to_delete = session.exec(
            select(Article).where(
                (Article.status != ScrapingStatus.SUCCESS) |
                (Article.raw_html_compressed == b"") |
                (Article.content_hash == "") |
                (Article.title == "")
            )
        ).all()
        
        # Count by category for reporting (but no double counting in total)
        failed_count = len([a for a in articles_to_delete if a.status == ScrapingStatus.FAILED])
        skipped_count = len([a for a in articles_to_delete if a.status == ScrapingStatus.SKIPPED])
        pending_count = len([a for a in articles_to_delete if a.status == ScrapingStatus.PENDING])
        empty_count = len([a for a in articles_to_delete if 
                          a.raw_html_compressed == b"" or a.content_hash == "" or a.title == ""])
        
        total_to_delete = len(articles_to_delete)
    
    if total_to_delete == 0:
        print("\n✅ Database is already clean! No incomplete entries found.")
        return
    
    print(f"\n🗑️  Found {total_to_delete:,} incomplete entries to clean up:")
    if failed_count > 0:
        print(f"  - Failed articles: {failed_count:,}")
    if skipped_count > 0:
        print(f"  - Skipped articles: {skipped_count:,}")
    if pending_count > 0:
        print(f"  - Pending articles: {pending_count:,}")
    if empty_count > 0:
        print(f"  - Articles with empty content: {empty_count:,}")
    
    print(f"  Note: Some articles may fall into multiple categories")
    
    if dry_run:
        print("\n👀 DRY RUN MODE - No changes will be made")
        print("Run with --execute to actually delete these entries")
        return
    
    # Confirm deletion
    confirm = input(f"\n⚠️  Delete {total_to_delete:,} incomplete entries? (y/n): ").lower().strip()
    if confirm != 'y':
        print("🛑 Cleanup cancelled")
        return
    
    # Perform cleanup
    print("\n🧹 Cleaning up database...")
    
    with next(db_manager.get_session()) as session:
        # Delete all incomplete articles in one go
        for article in articles_to_delete:
            session.delete(article)
        
        # Commit changes
        session.commit()
        deleted_count = len(articles_to_delete)
    
    print(f"✅ Deleted {deleted_count:,} incomplete entries")
    
    # Show stats after cleanup
    print("\n📊 Database stats after cleanup:")
    stats_after = article_db.get_scraping_stats()
    for status, count in stats_after.items():
        if status != "success_rate":
            print(f"  {status.capitalize()}: {count:,}")
    
    print(f"  Success rate: {stats_after['success_rate']:.1f}%")
    
    # Show database size
    db_info = db_manager.get_db_info()
    if db_info['exists']:
        print(f"\n💾 Database size: {db_info['size_mb']:.1f} MB")
    
    print(f"\n🎉 Cleanup complete! Database now contains only successful articles.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up incomplete database entries")
    parser.add_argument("--db", default="data/articles.db", help="Database file path")
    parser.add_argument("--execute", action="store_true", help="Actually perform cleanup (default is dry run)")
    
    args = parser.parse_args()
    
    try:
        cleanup_database(args.db, dry_run=not args.execute)
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())