"""Database operations for article storage and retrieval."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable, List, Optional

from sqlmodel import select

from .connection import DatabaseManager
from .models import Article, Chunk, ScrapingSession, ScrapingStatus, SessionStatus


class ArticleDatabase:
    """Database operations for wiki articles."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    # ---------------- existing methods unchanged ----------------

    def save_article(self, url: str, title: str, html_content: str) -> Article:
        with next(self.db_manager.get_session()) as session:
            existing = self.get_article_by_url(url)
            if existing and existing.status == ScrapingStatus.SUCCESS:
                return existing

            if existing:
                session.delete(existing)
                session.commit()

            try:
                article = Article.create_from_html(url, title, html_content)
                session.add(article)
                session.commit()
                session.refresh(article)
                return article
            except Exception as e:
                session.rollback()
                if "UNIQUE constraint failed: article.content_hash" in str(e):
                    import hashlib

                    content_hash = hashlib.sha256(
                        html_content.encode("utf-8")
                    ).hexdigest()
                    existing_with_hash = session.exec(
                        select(Article).where(Article.content_hash == content_hash)
                    ).first()
                    if existing_with_hash:
                        skipped_article = Article(
                            url=url,
                            title=title,
                            raw_html_compressed=b"",
                            content_hash=f"duplicate_of_{content_hash}",
                            file_size_bytes=0,
                            compression_ratio=0.0,
                            status=ScrapingStatus.SKIPPED,
                            error_message=f"Duplicate content of {existing_with_hash.url}",
                        )
                        session.add(skipped_article)
                        session.commit()
                        session.refresh(skipped_article)
                        return skipped_article
                raise e

    def get_article_by_url(self, url: str) -> Optional[Article]:
        with next(self.db_manager.get_session()) as session:
            statement = select(Article).where(Article.url == url)
            return session.exec(statement).first()

    def article_exists(self, url: str) -> bool:
        return self.get_article_by_url(url) is not None

    def mark_article_failed(self, url: str, error_message: str) -> bool:
        with next(self.db_manager.get_session()) as session:
            statement = select(Article).where(Article.url == url)
            article = session.exec(statement).first()
            if article:
                article.status = ScrapingStatus.FAILED
                article.error_message = error_message
                session.add(article)
                session.commit()
                return True
            else:
                article = Article(
                    url=url,
                    title="",
                    raw_html_compressed=b"",
                    content_hash="",
                    file_size_bytes=0,
                    compression_ratio=0.0,
                    status=ScrapingStatus.FAILED,
                    error_message=error_message,
                )
                session.add(article)
                session.commit()
                return True

    def get_articles_by_status(
        self, status: ScrapingStatus, limit: Optional[int] = None
    ) -> List[Article]:
        with next(self.db_manager.get_session()) as session:
            statement = select(Article).where(Article.status == status)
            if limit:
                statement = statement.limit(limit)
            return list(session.exec(statement))

    def get_scraping_stats(self) -> dict:
        with next(self.db_manager.get_session()) as session:
            total_articles = session.exec(select(Article)).all()
            stats = {
                "total": len(total_articles),
                "success": len(
                    [a for a in total_articles if a.status == ScrapingStatus.SUCCESS]
                ),
                "failed": len(
                    [a for a in total_articles if a.status == ScrapingStatus.FAILED]
                ),
                "pending": len(
                    [a for a in total_articles if a.status == ScrapingStatus.PENDING]
                ),
                "skipped": len(
                    [a for a in total_articles if a.status == ScrapingStatus.SKIPPED]
                ),
                "success_rate": 0.0,
            }
            stats["success_rate"] = (
                stats["success"] / stats["total"] if stats["total"] else 0.0
            )
            return stats

    def get_duplicate_content_count(self) -> int:
        with next(self.db_manager.get_session()) as session:
            articles = session.exec(select(Article)).all()
            content_hashes = [a.content_hash for a in articles if a.content_hash]
            unique_hashes = set(content_hashes)
            return len(content_hashes) - len(unique_hashes)

    # ---------------- new: parsing-related ops ----------------

    def get_articles_needing_parse(
        self, limit: int = 100, parser_version: Optional[str] = None
    ) -> List[Article]:
        """Return articles whose parsed_json is missing or stale."""
        with next(self.db_manager.get_session()) as session:
            stmt = select(Article).where(Article.status == ScrapingStatus.SUCCESS)

            # Articles need parsing if ANY of these conditions are true:
            parse_conditions = [
                # Never parsed
                Article.parsed_json == None,  # type: ignore[arg-type]
                # Hash tracking missing
                Article.last_parsed_html_hash == None,  # type: ignore[arg-type]
                # Content changed since last parse
                Article.last_parsed_html_hash != Article.content_hash,
            ]

            # Parser version changed (if provided)
            if parser_version:
                parse_conditions.extend(
                    [
                        Article.parser_version == None,  # type: ignore[arg-type]
                        Article.parser_version != parser_version,
                    ]
                )

            # Combine all conditions with OR
            from sqlmodel import or_

            stmt = stmt.where(or_(*parse_conditions))
            stmt = stmt.limit(limit)
            return list(session.exec(stmt))

    def save_parsed_article(
        self, url: str, parsed_doc: dict, parser_version: str
    ) -> Optional[Article]:
        """Persist parsed JSON + cache a few denormalized fields for indexing."""
        with next(self.db_manager.get_session()) as session:
            article = session.exec(select(Article).where(Article.url == url)).first()
            if not article:
                return None

            # Serialize JSON
            parsed_json_str = json.dumps(parsed_doc, ensure_ascii=False)

            # Denormalized helpers
            canonical_url = parsed_doc.get("canonical_url")
            lead = parsed_doc.get("lead")
            infobox_type = (parsed_doc.get("infobox") or {}).get("type")
            raw_html_hash = article.content_hash

            article.parsed_json = parsed_json_str
            article.parser_version = parser_version
            article.parsed_at = datetime.now()
            article.last_parsed_html_hash = raw_html_hash
            article.canonical_url = canonical_url
            article.lead = lead
            article.infobox_type = infobox_type

            session.add(article)
            session.commit()
            session.refresh(article)
            return article


class SessionDatabase:
    """Database operations for scraping sessions."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize with database manager."""
        self.db_manager = db_manager

    def create_session(self, total_urls: int) -> ScrapingSession:
        """Create a new scraping session.

        Args:
            total_urls: Total number of URLs to process

        Returns:
            Created session
        """
        with next(self.db_manager.get_session()) as session:
            scraping_session = ScrapingSession(total_urls=total_urls)
            session.add(scraping_session)
            session.commit()
            session.refresh(scraping_session)
            return scraping_session

    def update_session_progress(
        self,
        session_id: int,
        processed: int,
        success: int,
        failed: int,
        skipped: int = 0,
    ) -> bool:
        """Update session progress.

        Args:
            session_id: Session ID
            processed: Total processed count
            success: Success count
            failed: Failed count
            skipped: Skipped count

        Returns:
            True if updated successfully
        """
        with next(self.db_manager.get_session()) as session:
            statement = select(ScrapingSession).where(ScrapingSession.id == session_id)
            scraping_session = session.exec(statement).first()

            if scraping_session:
                scraping_session.processed_count = processed
                scraping_session.success_count = success
                scraping_session.failed_count = failed
                scraping_session.skipped_count = skipped
                session.add(scraping_session)
                session.commit()
                return True

            return False

    def get_latest_session(self) -> Optional[ScrapingSession]:
        """Get the most recent scraping session.

        Returns:
            Latest session or None
        """
        with next(self.db_manager.get_session()) as session:
            latest = session.exec(
                select(ScrapingSession)
                .order_by(ScrapingSession.started_at.desc())  # type: ignore
                .limit(1)
            ).first()
            return latest

    def get_active_sessions(self) -> List[ScrapingSession]:
        """Get all active (running) sessions.

        Returns:
            List of active sessions
        """
        with next(self.db_manager.get_session()) as session:
            statement = select(ScrapingSession).where(
                ScrapingSession.session_status == SessionStatus.RUNNING
            )
            return list(session.exec(statement))


class ChunkDatabase:
    """Database operations for article chunks."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def save_chunks(self, chunks: List[Chunk]) -> int:
        """Save chunks to database with conflict resolution.
        
        Args:
            chunks: List of Chunk instances to save
            
        Returns:
            Number of chunks successfully saved
        """
        if not chunks:
            return 0
        
        saved_count = 0
        with next(self.db_manager.get_session()) as session:
            for chunk in chunks:
                try:
                    # Check if chunk already exists
                    existing = session.exec(
                        select(Chunk).where(Chunk.chunk_uid == chunk.chunk_uid)
                    ).first()
                    
                    if existing:
                        # Update existing chunk
                        existing.text = chunk.text
                        existing.embedding_input = chunk.embedding_input
                        existing.token_count = chunk.token_count
                        existing.updated_at = datetime.now()
                        session.add(existing)
                    else:
                        # Add new chunk
                        session.add(chunk)
                    
                    saved_count += 1
                    
                except Exception as e:
                    print(f"⚠️  Failed to save chunk {chunk.chunk_uid[:16]}...: {e}")
                    continue
            
            session.commit()
        
        return saved_count

    def get_chunks_by_article_id(self, article_id: int) -> List[Chunk]:
        """Get all chunks for a specific article.
        
        Args:
            article_id: Article ID to get chunks for
            
        Returns:
            List of chunks for the article
        """
        with next(self.db_manager.get_session()) as session:
            statement = select(Chunk).where(
                Chunk.article_id == article_id
            ).order_by(Chunk.chunk_index.asc()) # type: ignore
            return list(session.exec(statement))

    def clear_chunks_for_article(self, article_id: int) -> int:
        """Remove all chunks for a specific article.
        
        Args:
            article_id: Article ID to clear chunks for
            
        Returns:
            Number of chunks removed
        """
        with next(self.db_manager.get_session()) as session:
            chunks_to_delete = session.exec(
                select(Chunk).where(Chunk.article_id == article_id)
            ).all()
            
            count = len(chunks_to_delete)
            for chunk in chunks_to_delete:
                session.delete(chunk)
            
            session.commit()
            return count

    def get_chunking_stats(self) -> dict:
        """Get statistics about chunks in the database.
        
        Returns:
            Dictionary with chunking statistics
        """
        with next(self.db_manager.get_session()) as session:
            all_chunks = session.exec(select(Chunk)).all()
            
            if not all_chunks:
                return {
                    "total_chunks": 0,
                    "unique_articles": 0,
                    "avg_chunks_per_article": 0.0,
                    "avg_token_count": 0.0,
                    "token_distribution": {},
                    "block_type_distribution": {},
                }
            
            # Calculate statistics
            unique_articles = len(set(chunk.article_id for chunk in all_chunks))
            avg_chunks_per_article = len(all_chunks) / max(1, unique_articles)
            
            # Token statistics
            token_counts = [chunk.token_count for chunk in all_chunks if chunk.token_count]
            avg_token_count = sum(token_counts) / max(1, len(token_counts))
            
            # Token distribution
            token_ranges = {
                "0-100": 0,
                "101-250": 0, 
                "251-350": 0,
                "351-500": 0,
                "500+": 0
            }
            
            for token_count in token_counts:
                if token_count <= 100:
                    token_ranges["0-100"] += 1
                elif token_count <= 250:
                    token_ranges["101-250"] += 1
                elif token_count <= 350:
                    token_ranges["251-350"] += 1
                elif token_count <= 500:
                    token_ranges["351-500"] += 1
                else:
                    token_ranges["500+"] += 1
            
            # Block type distribution
            block_types = {}
            for chunk in all_chunks:
                block_type = str(chunk.block_type)
                block_types[block_type] = block_types.get(block_type, 0) + 1
            
            return {
                "total_chunks": len(all_chunks),
                "unique_articles": unique_articles,
                "avg_chunks_per_article": avg_chunks_per_article,
                "avg_token_count": avg_token_count,
                "token_distribution": token_ranges,
                "block_type_distribution": block_types,
            }

    def get_articles_needing_chunks(self, limit: int = 100) -> List[int]:
        """Get article IDs that need chunks generated.
        
        Args:
            limit: Maximum number of article IDs to return
            
        Returns:
            List of article IDs that need chunking
        """
        with next(self.db_manager.get_session()) as session:
            # Get all articles with parsed JSON
            parsed_articles = session.exec(
                select(Article.id).where(
                    (Article.status == ScrapingStatus.SUCCESS) &
                    (Article.parsed_json != None)  # type: ignore[arg-type]
                )
            ).all()
            
            # Get articles that already have chunks
            chunked_articles = session.exec(
                select(Chunk.article_id).distinct()
            ).all()
            
            chunked_set = set(chunked_articles)
            
            # Find articles needing chunks - filter out None values
            needing_chunks = [
                article_id for article_id in parsed_articles
                if article_id is not None and article_id not in chunked_set
            ]
            
            return needing_chunks[:limit]
