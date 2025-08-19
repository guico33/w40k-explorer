#!/usr/bin/env python3
"""Script to parse HTML content of scraped articles and populate parsed fields."""

import sys
from pathlib import Path

# Add src to path for new architecture
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tqdm import tqdm

from w40k.infrastructure.database.connection import DatabaseManager
from w40k.infrastructure.database.models import ScrapingStatus
from w40k.infrastructure.database.operations import ArticleDatabase
from w40k.infrastructure.rag.parser import PARSER_VERSION, parse_article_html


def parse_articles(
    db_path: str = "data/articles.db",
    batch_size: int = 100,
    force_reparse: bool = False,
):
    """Parse all successfully scraped articles and populate parsed fields.

    Args:
        db_path: Path to database file
        batch_size: Number of articles to process per batch
        force_reparse: If True, reparse all articles even if already parsed
    """
    print("📝 Warhammer 40k Article Parser")
    print("=" * 50)

    # Initialize database
    db_manager = DatabaseManager(db_path)
    article_db = ArticleDatabase(db_manager)

    # Get articles that need parsing
    print("🔍 Finding articles to parse...")
    if force_reparse:
        # Get all successful articles
        articles_to_parse = article_db.get_articles_by_status(ScrapingStatus.SUCCESS)
        print(f"📊 Force reparse: {len(articles_to_parse):,} articles")
    else:
        # Get articles needing parse (new method)
        articles_to_parse = article_db.get_articles_needing_parse(
            limit=10000, parser_version=PARSER_VERSION  # Large limit to get all
        )
        print(f"📊 Articles needing parse: {len(articles_to_parse):,}")

    if not articles_to_parse:
        print("✅ All articles are already parsed with current parser version!")
        return

    # Show current parser version
    print(f"🔧 Parser version: {PARSER_VERSION}")

    # Confirm processing
    confirm = (
        input(f"Parse {len(articles_to_parse):,} articles? (y/n): ").lower().strip()
    )
    if confirm != "y":
        print("🛑 Parsing cancelled")
        return

    # Process articles
    print("⏳ Starting article parsing...")

    stats = {
        "total": len(articles_to_parse),
        "success": 0,
        "failed": 0,
        "skipped": 0,
    }

    with tqdm(
        total=len(articles_to_parse), desc="Parsing articles", unit="articles"
    ) as pbar:
        for i, article in enumerate(articles_to_parse):
            try:
                # Get HTML content
                html_content = article.get_html_content()

                # Validate HTML content
                if not html_content or len(html_content.strip()) < 100:
                    stats["failed"] += 1
                    print(f"⚠️  Article has insufficient HTML content: {article.url}")
                    continue

                # Parse the HTML
                parsed_doc = parse_article_html(html_content)

                # Validate parsed document
                if not parsed_doc or not parsed_doc.get("title"):
                    stats["failed"] += 1
                    print(f"⚠️  Parser returned invalid document for: {article.url}")
                    continue

                # Save parsed results
                updated_article = article_db.save_parsed_article(
                    article.url, parsed_doc, PARSER_VERSION
                )

                if updated_article:
                    stats["success"] += 1
                    # Log key parsed fields for verification
                    title = parsed_doc.get("title", "No title")
                    lead_length = len(parsed_doc.get("lead", ""))
                    sections_count = len(parsed_doc.get("sections", []))

                    if (i + 1) % (
                        batch_size // 4
                    ) == 0:  # Log every 25 articles in a batch of 100
                        print(
                            f"✅ Parsed '{title}' - Lead: {lead_length} chars, Sections: {sections_count}"
                        )
                else:
                    stats["failed"] += 1
                    print(f"⚠️  Failed to save parsed data for: {article.url}")

            except ValueError as e:
                stats["failed"] += 1
                print(f"❌ Data validation error for {article.url}: {e}")
            except Exception as e:
                stats["failed"] += 1
                error_type = type(e).__name__
                print(
                    f"❌ {error_type} for {article.url}: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}"
                )

                # For debugging: log the first few problematic URLs
                if stats["failed"] <= 5:
                    print(
                        f"   🔍 Debug info - Article ID: {article.id}, Content length: {len(article.get_html_content()) if hasattr(article, 'get_html_content') else 'unknown'}"
                    )

            # Update progress
            pbar.update(1)
            pbar.set_postfix(
                {
                    "Success": stats["success"],
                    "Failed": stats["failed"],
                    "Skipped": stats["skipped"],
                }
            )

            # Batch progress reporting
            if (i + 1) % batch_size == 0:
                print(f"📈 Processed {i + 1:,}/{len(articles_to_parse):,} articles")

    # Show final results
    print("\\n" + "=" * 50)
    print("📊 Parsing Complete!")
    print("=" * 50)
    print(f"Total processed: {stats['total']:,}")
    print(f"✅ Successful: {stats['success']:,}")
    print(f"❌ Failed: {stats['failed']:,}")
    print(f"⏭️  Skipped: {stats['skipped']:,}")

    success_rate = (
        (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    )
    print(f"📈 Success rate: {success_rate:.1f}%")

    # Show database summary
    db_stats = article_db.get_scraping_stats()
    print(f"\\n🗄️  Database Summary:")
    print(f"Total articles: {db_stats['total']:,}")
    print(f"Successfully scraped: {db_stats['success']:,}")

    # Count parsed articles
    with next(db_manager.get_session()) as session:
        from sqlmodel import select

        from src.database.models import Article

        parsed_count = len(
            session.exec(
                select(Article).where(
                    (Article.status == ScrapingStatus.SUCCESS)
                    & (Article.parsed_json != None)  # type: ignore[arg-type]
                )
            ).all()
        )

    print(f"Successfully parsed: {parsed_count:,}")

    db_info = db_manager.get_db_info()
    if db_info["exists"]:
        print(f"Database size: {db_info['size_mb']:.1f} MB")

    if stats["failed"] > 0:
        print(f"\\n⚠️  {stats['failed']} articles failed to parse")
        print("Check the logs above for specific errors")
    else:
        print("\\n🎉 All articles parsed successfully!")

    print("\\n✅ Ready for Phase 4: Chunking and Embeddings!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse scraped article HTML content")
    parser.add_argument("--db", default="data/articles.db", help="Database file path")
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for progress reporting"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force reparse all articles"
    )

    args = parser.parse_args()

    try:
        parse_articles(
            db_path=args.db, batch_size=args.batch_size, force_reparse=args.force
        )
    except KeyboardInterrupt:
        print("\\n⏹️  Parsing interrupted by user")
        print("Progress has been saved - you can resume by running this script again")
        return 1
    except Exception as e:
        print(f"\\n❌ Error during parsing: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
