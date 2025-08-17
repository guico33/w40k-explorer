"""SQLModel database models for Warhammer 40k wiki data storage."""

from __future__ import annotations

import gzip
import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class ScrapingStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class SessionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class BlockType(str, Enum):
    LEAD = "lead"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_ROWS = "table_rows"
    QUOTE = "quote"
    FIGURE_CAPTION = "figure_caption"
    MIXED = "mixed"


class Article(SQLModel, table=True):
    """Model for storing scraped wiki articles and parsed results."""

    id: Optional[int] = Field(default=None, primary_key=True)

    # Scrape-time
    url: str = Field(unique=True, index=True)
    title: str = Field(index=True)
    raw_html_compressed: bytes = Field()
    content_hash: str = Field(unique=True, index=True)  # sha256 of raw HTML bytes
    scraped_at: datetime = Field(default_factory=datetime.now, index=True)
    status: ScrapingStatus = Field(default=ScrapingStatus.PENDING, index=True)
    error_message: Optional[str] = Field(default=None)
    file_size_bytes: int = Field()
    compression_ratio: float = Field()

    # Parse-time (new)
    parsed_json: Optional[str] = Field(default=None)  # full JSON (TEXT)
    parser_version: Optional[str] = Field(default=None)  # e.g., "w40k-parser/1.0.1"
    parsed_at: Optional[datetime] = Field(default=None, index=True)
    last_parsed_html_hash: Optional[str] = Field(default=None, index=True)

    # Denormalized/search helpers (from parsed_json)
    canonical_url: Optional[str] = Field(default=None, index=True)
    lead: Optional[str] = Field(default=None)
    infobox_type: Optional[str] = Field(default=None, index=True)

    # Relationships (for chunks later) - temporarily commented out to fix relationship issue
    # chunks: List["Chunk"] = Relationship(back_populates="article")

    @classmethod
    def create_from_html(cls, url: str, title: str, html_content: str) -> "Article":
        """Create Article instance from HTML content with compression."""
        html_bytes = html_content.encode("utf-8")
        compressed_html = gzip.compress(html_bytes)
        content_hash = hashlib.sha256(html_bytes).hexdigest()
        return cls(
            url=url,
            title=title,
            raw_html_compressed=compressed_html,
            content_hash=content_hash,
            file_size_bytes=len(html_bytes),
            compression_ratio=len(compressed_html) / max(1, len(html_bytes)),
            status=ScrapingStatus.SUCCESS,
        )

    def get_html_content(self) -> str:
        """Decompress and return HTML content."""
        decompressed_bytes = gzip.decompress(self.raw_html_compressed)
        return decompressed_bytes.decode("utf-8")


class ScrapingSession(SQLModel, table=True):
    """Model for tracking scraping sessions."""

    id: Optional[int] = Field(default=None, primary_key=True)
    started_at: datetime = Field(default_factory=datetime.now, index=True)
    completed_at: Optional[datetime] = Field(default=None)
    total_urls: int = Field()
    processed_count: int = Field(default=0)
    success_count: int = Field(default=0)
    failed_count: int = Field(default=0)
    skipped_count: int = Field(default=0)
    session_status: SessionStatus = Field(default=SessionStatus.RUNNING, index=True)
    error_message: Optional[str] = Field(default=None)

    @property
    def progress_percentage(self) -> float:
        if self.total_urls == 0:
            return 0.0
        return (self.processed_count / self.total_urls) * 100

    def mark_completed(self) -> None:
        self.completed_at = datetime.now()
        self.session_status = SessionStatus.COMPLETED

    def mark_failed(self, error: str) -> None:
        self.completed_at = datetime.now()
        self.session_status = SessionStatus.FAILED
        self.error_message = error


class Chunk(SQLModel, table=True):
    """
    Structure-aware chunk suitable for embedding + Qdrant payload.

    Store the minimal text needed for rendering/debug (`text`) and the
    exact string you embed (`embedding_input`) if you want perfect reproducibility.
    Use `chunk_uid` as the deterministic ID (also use as Qdrant point ID).
    """

    id: Optional[int] = Field(default=None, primary_key=True)

    # Deterministic ID for idempotent upserts (also Qdrant point ID)
    chunk_uid: str = Field(index=True, unique=True)

    # Foreign key & denorm for fast filtering
    article_id: int = Field(foreign_key="article.id", index=True)
    article_title: str = Field(index=True)
    canonical_url: str = Field(index=True)

    # Section/breadcrumb
    section_path: str = Field(
        index=True
    )  # JSON-encoded list[str], e.g. '["History","13th Black Crusade"]'
    block_type: BlockType = Field(default=BlockType.PARAGRAPH, index=True)

    # Ordering within the article
    chunk_index: int = Field(index=True)  # order within the article

    # Texts
    text: str = Field()  # body-only (what you display)
    embedding_input: Optional[str] = Field(default=None)  # header+body (what you embed)

    # Hints & metrics
    token_count: Optional[int] = Field(default=None, index=True)
    kv_preview: Optional[str] = Field(default=None)  # "Affiliation=..., Founding=..."
    lead: bool = Field(default=False, index=True)
    parser_version: Optional[str] = Field(default=None, index=True)

    # Optional: link graph/lightweight flags
    links_out: Optional[str] = Field(default=None)  # JSON list[str], optional
    active: bool = Field(default=True, index=True)  # for soft deletes

    # Embedding tracking
    has_embedding: bool = Field(default=False, index=True)
    embedding_generated_at: Optional[datetime] = Field(default=None, index=True)
    
    # Error tracking
    embedding_failed_count: int = Field(default=0)
    last_embedding_error: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.now, index=True)
    updated_at: datetime = Field(default_factory=datetime.now, index=True)
