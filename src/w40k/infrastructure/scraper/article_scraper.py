"""Article scraper for individual Warhammer 40k wiki pages."""

import json
import logging
from typing import List, Optional

from bs4 import BeautifulSoup
from tqdm import tqdm

from ..database.connection import DatabaseManager
from ..database.operations import ArticleDatabase, SessionDatabase
from ..database.models import ScrapingStatus
from .http_client import HttpClient


class ArticleScraper:
    """Scrapes individual wiki articles and stores raw HTML."""
    
    def __init__(self, requests_per_second: float = 0.3, db_path: str = "data/articles.db"):
        """Initialize article scraper.
        
        Args:
            requests_per_second: Rate limit for requests
            db_path: Path to SQLite database
        """
        self.requests_per_second = requests_per_second
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        
        # Initialize database
        self.db_manager = DatabaseManager(db_path)
        self.article_db = ArticleDatabase(self.db_manager)
        self.session_db = SessionDatabase(self.db_manager)
    
    async def scrape_article(self, url: str, client: HttpClient) -> bool:
        """Scrape a single article and store in database.
        
        Args:
            url: Article URL to scrape
            client: HTTP client instance
            
        Returns:
            True if successful, False if failed
        """
        try:
            # Check if already scraped
            if self.article_db.article_exists(url):
                self.logger.debug(f"Article already exists, skipping: {url}")
                return True
            
            # Fetch HTML content
            html_content, status_code = await client.get(url)
            
            # Extract title from HTML
            title = self._extract_title(html_content, url)
            
            # Save to database
            article = self.article_db.save_article(url, title, html_content)
            self.logger.debug(
                f"Saved article: {title} ({len(html_content)} bytes, "
                f"compressed to {len(article.raw_html_compressed)} bytes)"
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to scrape {url}: {str(e)}"
            self.logger.error(error_msg)
            
            # Mark as failed in database
            self.article_db.mark_article_failed(url, error_msg)
            return False
    
    def _extract_title(self, html_content: str, url: str) -> str:
        """Extract title from HTML content.
        
        Args:
            html_content: Raw HTML content
            url: Article URL (for fallback)
            
        Returns:
            Extracted title or fallback
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try to get page title
            title_tag = soup.find('title')
            if title_tag and title_tag.get_text():
                title = title_tag.get_text().strip()
                # Remove common wiki suffixes
                title = title.replace(' | Warhammer 40k | Fandom', '')
                title = title.replace(' - Warhammer 40k - Fandom', '')
                return title
            
            # Fallback: try h1 tag
            h1_tag = soup.find('h1')
            if h1_tag and h1_tag.get_text():
                return h1_tag.get_text().strip()
            
            # Fallback: extract from URL
            path_parts = url.split('/')
            if path_parts:
                return path_parts[-1].replace('_', ' ')
            
            return "Unknown Title"
            
        except Exception as e:
            self.logger.warning(f"Failed to extract title from {url}: {e}")
            # Extract from URL as final fallback
            try:
                path_parts = url.split('/')
                if path_parts:
                    return path_parts[-1].replace('_', ' ')
            except:
                pass
            
            return "Unknown Title"
    
    async def scrape_articles(
        self, 
        urls: List[str], 
        session_id: Optional[int] = None
    ) -> dict:
        """Scrape articles sequentially.
        
        Args:
            urls: List of URLs to scrape
            session_id: Optional session ID for tracking
            
        Returns:
            Dictionary with scraping statistics
        """
        stats = {
            "total": len(urls),
            "success": 0,
            "failed": 0,
            "skipped": 0,
        }
        
        async with HttpClient(self.requests_per_second) as client:
            # Process URLs with progress bar
            with tqdm(total=len(urls), desc="Scraping articles", unit="articles") as pbar:
                
                for i, url in enumerate(urls):
                    try:
                        # Scrape single article
                        await self.scrape_article(url, client)
                        
                        # Check result and update stats
                        article = self.article_db.get_article_by_url(url)
                        if article:
                            if article.status == ScrapingStatus.SUCCESS:
                                stats["success"] += 1
                            elif article.status == ScrapingStatus.FAILED:
                                stats["failed"] += 1
                            elif article.status == ScrapingStatus.SKIPPED:
                                stats["skipped"] += 1
                        else:
                            stats["failed"] += 1
                    except Exception as e:
                        # If scraping fails completely, count as failed and continue
                        self.logger.error(f"Scraping failed for {url}: {e}")
                        stats["failed"] += 1
                    
                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix({
                        "Success": stats["success"],
                        "Failed": stats["failed"],
                        "Skipped": stats["skipped"]
                    })
                    
                    # Update session every 10 articles
                    if session_id and (i + 1) % 10 == 0:
                        self.session_db.update_session_progress(
                            session_id,
                            processed=i + 1,
                            success=stats["success"],
                            failed=stats["failed"],
                            skipped=stats["skipped"]
                        )
        
        return stats
    
    def load_urls_from_file(self, filepath: str = "data/article_urls.json") -> List[str]:
        """Load URLs from JSON file.
        
        Args:
            filepath: Path to JSON file with URLs
            
        Returns:
            List of URLs to scrape
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('urls', [])
        except FileNotFoundError:
            self.logger.error(f"URL file not found: {filepath}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON file {filepath}: {e}")
            return []
    
    def get_remaining_urls(self, all_urls: List[str]) -> List[str]:
        """Get URLs that haven't been successfully scraped yet.
        
        Args:
            all_urls: Complete list of URLs
            
        Returns:
            List of URLs not yet scraped or without SUCCESS status
        """
        remaining = []
        for url in all_urls:
            article = self.article_db.get_article_by_url(url)
            if not article or article.status != ScrapingStatus.SUCCESS:
                remaining.append(url)
        
        self.logger.info(f"Found {len(remaining)} URLs remaining out of {len(all_urls)} total")
        return remaining
    
    def get_scraping_summary(self) -> dict:
        """Get comprehensive scraping statistics.
        
        Returns:
            Dictionary with detailed statistics
        """
        stats = self.article_db.get_scraping_stats()
        latest_session = self.session_db.get_latest_session()
        
        summary = {
            "database_stats": stats,
            "latest_session": {
                "id": latest_session.id if latest_session else None,
                "started_at": latest_session.started_at if latest_session else None,
                "progress_percentage": latest_session.progress_percentage if latest_session else 0.0,
                "status": latest_session.session_status if latest_session else None,
            } if latest_session else None,
            "database_info": self.db_manager.get_db_info()
        }
        
        return summary