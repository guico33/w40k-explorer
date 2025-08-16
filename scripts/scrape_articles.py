#!/usr/bin/env python3
"""Script to scrape all Warhammer 40k wiki articles and store raw HTML."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.scraper.article_scraper import ArticleScraper


async def main():
    """Run article scraping."""
    print("🕷️  Starting Warhammer 40k Article Scraping")
    print("=" * 50)

    # Initialize scraper with conservative rate limiting
    scraper = ArticleScraper(requests_per_second=3)

    try:
        # Load URLs from file
        print("📂 Loading URLs from file...")
        all_urls = scraper.load_urls_from_file("data/article_urls.json")

        if not all_urls:
            print("❌ No URLs found! Make sure data/article_urls.json exists.")
            return 1

        print(f"📊 Total URLs in file: {len(all_urls):,}")

        # Check what's already been scraped
        print("🔍 Checking for existing articles...")
        remaining_urls = scraper.get_remaining_urls(all_urls)

        if not remaining_urls:
            print("✅ All articles already scraped!")
            summary = scraper.get_scraping_summary()
            print(
                f"📈 Database contains {summary['database_stats']['success']:,} articles"
            )
            return 0

        print(f"📋 URLs remaining to scrape: {len(remaining_urls):,}")

        # Ask user to confirm
        confirm = (
            input(f"Start scraping {len(remaining_urls):,} articles? (y/n): ")
            .lower()
            .strip()
        )
        if confirm != "y":
            print("🛑 Scraping cancelled by user")
            return 0

        # Create scraping session
        session = scraper.session_db.create_session(total_urls=len(remaining_urls))

        session_id = session.id
        print(f"🚀 Created scraping session {session_id}")

        # Start scraping
        print("⏳ Starting article scraping...")
        stats = await scraper.scrape_articles(
            remaining_urls,
            session_id=session_id,
        )

        # Mark session as completed
        session.mark_completed()
        scraper.session_db.update_session_progress(
            session_id,  # type: ignore
            processed=len(remaining_urls),
            success=stats["success"],
            failed=stats["failed"],
            skipped=stats["skipped"],
        )

        # Show final results
        print("\\n" + "=" * 50)
        print("📊 Scraping Complete!")
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
        summary = scraper.get_scraping_summary()
        db_stats = summary["database_stats"]
        print(f"\\n🗄️  Database Summary:")
        print(f"Total articles: {db_stats['total']:,}")
        print(f"Success: {db_stats['success']:,}")
        print(f"Failed: {db_stats['failed']:,}")

        db_info = summary["database_info"]
        if db_info["exists"]:
            print(f"Database size: {db_info['size_mb']:.1f} MB")

        if stats["failed"] > 0:
            print(f"\\n⚠️  {stats['failed']} articles failed to scrape")
            print("You can re-run this script to retry failed articles")
        else:
            print("\\n🎉 All articles scraped successfully!")

        print("\\n✅ Ready for Phase 3: Data Processing!")

    except KeyboardInterrupt:
        print("\\n⏹️  Scraping interrupted by user")
        print("Progress has been saved - you can resume by running this script again")
        return 1
    except Exception as e:
        print(f"\\n❌ Error during scraping: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
