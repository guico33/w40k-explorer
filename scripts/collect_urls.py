#!/usr/bin/env python3
"""Simple script to collect all Warhammer 40k wiki URLs."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.scraper.url_collector import WikiUrlCollector


async def main():
    """Run URL collection."""
    print("🚀 Starting Warhammer 40k Wiki URL Collection")
    print("=" * 50)

    # Create collector with 1 request per second to be respectful
    collector = WikiUrlCollector(requests_per_second=1.0)

    try:
        # Set up file paths
        output_file = "data/article_urls.json"
        jsonl_file = "data/article_urls.jsonl"

        # Check if we have existing progress
        from pathlib import Path

        if Path(jsonl_file).exists():
            existing_count = len(collector._load_urls_from_jsonl(jsonl_file))
            if existing_count > 0:
                print(f"📁 Found existing progress: {existing_count:,} URLs")
                resume = input("Resume from where we left off? (y/n): ").lower().strip()
                if resume != "y":
                    print("🗑️ Starting fresh (existing file will be overwritten)")
                    Path(jsonl_file).unlink()

        # Collect all URLs (saves progressively)
        urls = await collector.collect_all_urls(jsonl_file)

        # Convert to final JSON format
        print(f"\n🔄 Converting to final JSON format...")
        collector.convert_jsonl_to_json(jsonl_file, output_file)

        # Show statistics
        stats = collector.get_url_statistics(urls)

        print("\n" + "=" * 50)
        print("📊 Collection Complete!")
        print("=" * 50)
        print(f"Total URLs collected: {stats['total']:,}")
        print(f"Saved to: {output_file}")

        print(f"\nSample URLs:")
        for i, url in enumerate(stats["sample_urls"][:5], 1):
            print(f"  {i}. {url}")

        print(f"\n✅ Ready for Phase 2: Content Scraping!")

    except KeyboardInterrupt:
        print("\n⏹️  Collection interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
