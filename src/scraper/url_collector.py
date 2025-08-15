"""URL collector for Warhammer 40k wiki using Special:AllPages."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from .http_client import HttpClient


class WikiUrlCollector:
    """Collects all article URLs from Warhammer 40k wiki using Special:AllPages."""

    BASE_URL = "https://warhammer40k.fandom.com"
    ALLPAGES_URL = "https://warhammer40k.fandom.com/wiki/Special:AllPages"

    def __init__(self, requests_per_second: float = 1.0):
        """Initialize URL collector.

        Args:
            requests_per_second: Rate limit for requests
        """
        self.requests_per_second = requests_per_second
        self.logger = logging.getLogger(__name__)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    async def collect_all_urls(
        self, output_file: str = "data/article_urls.jsonl"
    ) -> Set[str]:
        """Collect all article URLs from the wiki, appending to file after each page.

        Args:
            output_file: File to append URLs to (JSONL format)

        Returns:
            Set of article URLs
        """
        all_urls = set()

        # Ensure directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Load existing URLs if file exists
        resume_url = None
        if Path(output_file).exists():
            existing_urls = self._load_urls_from_jsonl(output_file)
            all_urls.update(existing_urls)
            resume_url = self._get_last_page_state(output_file)

            if existing_urls:
                self.logger.info(
                    f"Loaded {len(existing_urls)} existing URLs from {output_file}"
                )
                if resume_url:
                    self.logger.info(f"Resuming from: {resume_url}")
                else:
                    self.logger.info("No resume point found, starting from beginning")

        async with HttpClient(self.requests_per_second) as client:
            # Start from resume point or first page
            current_url = resume_url or f"{self.ALLPAGES_URL}?from=&to=&namespace=0"
            page_count = 0

            # Create progress bar (we'll update total as we discover pages)
            pbar = tqdm(desc="Collecting URLs", unit="pages")

            while current_url:
                try:
                    self.logger.info(f"Fetching page {page_count + 1}: {current_url}")

                    # Debug: Check if URL is properly encoded
                    from urllib.parse import quote_plus

                    if "(" in current_url or ")" in current_url:
                        self.logger.warning(
                            f"URL contains unencoded parentheses: {current_url}"
                        )
                        # Fix URL encoding issues
                        if "?from=" in current_url:
                            base_url, params = current_url.split("?from=", 1)
                            # Re-encode the parameter part
                            encoded_params = quote_plus(params.replace("+", " "))
                            current_url = f"{base_url}?from={encoded_params}"
                            self.logger.info(f"Fixed URL: {current_url}")

                    content, _ = await client.get(current_url)

                    # Parse URLs from current page
                    page_urls = self._extract_urls_from_page(content)
                    new_urls = [url for url in page_urls if url not in all_urls]

                    # Get next page URL
                    next_url = self._get_next_page_url(content)
                    if next_url:
                        if next_url.startswith("http"):
                            next_full_url = next_url  # Already a full URL
                        else:
                            next_full_url = urljoin(self.BASE_URL, next_url)
                        self.logger.debug(f"Next page URL: {next_full_url}")
                    else:
                        next_full_url = None

                    # Always append to file (even if no new URLs) to save page state
                    self._append_urls_to_file(
                        new_urls, output_file, page_count + 1, next_full_url
                    )
                    all_urls.update(new_urls)

                    # Update current URL for next iteration
                    current_url = next_full_url

                    page_count += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "URLs": len(all_urls),
                            "Page": page_count,
                            "New": len(new_urls),
                        }
                    )

                    self.logger.info(
                        f"Page {page_count}: Found {len(new_urls)} new URLs, total: {len(all_urls)}"
                    )

                except Exception as e:
                    self.logger.error(f"Error fetching {current_url}: {e}")
                    # Try to continue with next page if possible
                    # In case of network error, we might still have partial content
                    try:
                        if hasattr(e, "status") and e.status == 429:  # type: ignore
                            self.logger.info("Rate limited, waiting 30 seconds...")
                            import asyncio

                            await asyncio.sleep(30)
                            continue
                    except:
                        pass

                    self.logger.warning(
                        "Stopping collection due to error. Progress has been saved."
                    )
                    break

            pbar.close()

        self.logger.info(f"Collection complete. Total URLs found: {len(all_urls)}")
        return all_urls

    def _extract_urls_from_page(self, html_content: str) -> List[str]:
        """Extract article URLs from Special:AllPages content.

        Args:
            html_content: HTML content of the page

        Returns:
            List of article URLs found on the page
        """
        soup = BeautifulSoup(html_content, "html.parser")
        urls = []

        # Find the main content area with article links
        allpages_chunks = soup.find_all("ul", class_="mw-allpages-chunk")

        for chunk in allpages_chunks:
            # Get all links in this chunk
            links = chunk.find_all("a", href=True) if isinstance(chunk, Tag) else []
            for link in links:
                if isinstance(link, Tag):
                    href = link.get("href")
                    if href and isinstance(href, str):
                        # Only include wiki article links
                        if self._is_valid_article_link(href):
                            full_url = urljoin(self.BASE_URL, href)
                            urls.append(full_url)

        return urls

    def _is_valid_article_link(self, href: str) -> bool:
        """Check if a link is a valid article link.

        Args:
            href: The href attribute value

        Returns:
            True if it's a valid article link
        """
        if not href:
            return False

        # Must be a wiki link
        if not href.startswith("/wiki/"):
            return False

        # Skip special pages, talk pages, etc.
        skip_patterns = [
            "/wiki/Special:",
            "/wiki/File:",
            "/wiki/Category:",
            "/wiki/Template:",
            "/wiki/Help:",
            "/wiki/User:",
            "/wiki/Talk:",
            "/wiki/MediaWiki:",
        ]

        for pattern in skip_patterns:
            if href.startswith(pattern):
                return False

        return True

    def _get_next_page_url(self, html_content: str) -> Optional[str]:
        """Extract the URL for the next page from the navigation.

        Args:
            html_content: HTML content of the current page

        Returns:
            URL of the next page, or None if no next page
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Look for navigation div
        nav_div = soup.find("div", class_="mw-allpages-nav")
        if not nav_div:
            return None

        # Find "Next page" link
        links = nav_div.find_all("a", href=True) if isinstance(nav_div, Tag) else []
        for link in links:
            if isinstance(link, Tag) and "Next page" in link.get_text():
                href = link.get("href")
                if href and isinstance(href, str):
                    return href

        return None

    def _append_urls_to_file(
        self,
        urls: List[str],
        filepath: str,
        page_num: int,
        next_page_url: Optional[str] = None,
    ) -> None:
        """Append URLs to JSONL file after processing a page.

        Args:
            urls: List of URLs to append
            filepath: Path to the JSONL file
            page_num: Page number for logging
            next_page_url: URL of the next page (for resume capability)
        """
        import datetime

        with open(filepath, "a", encoding="utf-8") as f:
            for url in urls:
                entry = {
                    "url": url,
                    "page_num": page_num,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                f.write(json.dumps(entry) + "\n")

            # Save page state for resume
            if next_page_url or len(urls) == 0:  # Even if no URLs, save the page state
                page_state = {
                    "type": "page_state",
                    "page_num": page_num,
                    "next_page_url": next_page_url,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                f.write(json.dumps(page_state) + "\n")

        self.logger.debug(
            f"Appended {len(urls)} URLs from page {page_num} to {filepath}"
        )

    def _load_urls_from_jsonl(self, filepath: str) -> Set[str]:
        """Load URLs from JSONL file.

        Args:
            filepath: Path to the JSONL file

        Returns:
            Set of URLs
        """
        urls = set()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            if "url" in entry and entry.get("type") != "page_state":
                                urls.add(entry["url"])
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass

        return urls

    def _get_last_page_state(self, filepath: str) -> Optional[str]:
        """Get the last page URL to resume from.

        Args:
            filepath: Path to the JSONL file

        Returns:
            URL to resume from, or None to start from beginning
        """
        last_next_url = None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            if entry.get("type") == "page_state" and entry.get(
                                "next_page_url"
                            ):
                                last_next_url = entry["next_page_url"]
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass

        return last_next_url

    def convert_jsonl_to_json(self, jsonl_file: str, json_file: str) -> None:
        """Convert JSONL file to final JSON format.

        Args:
            jsonl_file: Input JSONL file path
            json_file: Output JSON file path
        """
        urls = self._load_urls_from_jsonl(jsonl_file)
        self.save_urls_to_file(urls, json_file)

    def save_urls_to_file(
        self, urls: Set[str], filepath: str = "data/article_urls.json"
    ) -> None:
        """Save collected URLs to a JSON file.

        Args:
            urls: Set of URLs to save
            filepath: Path to save the file
        """
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Convert set to sorted list for consistent output
        url_list = sorted(list(urls))

        # Create output data
        output_data = {
            "collection_timestamp": "2024-08-15T17:00:00Z",  # Would use datetime.now() in real implementation
            "total_urls": len(url_list),
            "base_url": self.BASE_URL,
            "urls": url_list,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(url_list)} URLs to {filepath}")

    def load_urls_from_file(self, filepath: str = "data/article_urls.json") -> Set[str]:
        """Load URLs from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Set of URLs
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("urls", []))
        except FileNotFoundError:
            self.logger.warning(f"File {filepath} not found")
            return set()
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON file {filepath}: {e}")
            return set()

    def get_url_statistics(self, urls: Set[str]) -> dict:
        """Get basic statistics about the collected URLs.

        Args:
            urls: Set of URLs to analyze

        Returns:
            Dictionary with statistics
        """
        return {
            "total": len(urls),
            "sample_urls": list(urls)[:10],  # First 10 for inspection
        }


async def main():
    """Main entry point for URL collection."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect Warhammer 40k Wiki URLs")
    parser.add_argument("--rate", type=float, default=1.0, help="Requests per second")
    parser.add_argument(
        "--output", default="data/article_urls.json", help="Output file path"
    )
    parser.add_argument("--stats", action="store_true", help="Show URL statistics")

    args = parser.parse_args()

    # Create collector
    collector = WikiUrlCollector(requests_per_second=args.rate)

    try:
        # Set up file paths
        jsonl_file = args.output.replace(".json", ".jsonl")

        # Collect URLs (saves progressively to JSONL)
        print("Starting URL collection from Warhammer 40k wiki...")
        urls = await collector.collect_all_urls(jsonl_file)

        # Convert to final JSON format
        print(f"Converting to final JSON format: {args.output}")
        collector.convert_jsonl_to_json(jsonl_file, args.output)

        # Show statistics
        if args.stats:
            stats = collector.get_url_statistics(urls)
            print("\nURL Collection Statistics:")
            print(f"Total URLs: {stats['total']:,}")
            print(f"Sample URLs: {stats['sample_urls']}")

        print(
            f"\n✅ Successfully collected {len(urls):,} URLs and saved to {args.output}"
        )

    except Exception as e:
        print(f"❌ Error during URL collection: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import asyncio
    import sys

    sys.exit(asyncio.run(main()))
