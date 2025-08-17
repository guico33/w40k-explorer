"""Qdrant vector database integration for Warhammer 40k wiki chunks."""

from __future__ import annotations

import ast
import html
import json
import os
import re
import uuid
from typing import Dict, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm

try:
    from ..database.models import Chunk
except ImportError:
    # Handle case when running as script
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database.models import Chunk


def point_id_from_chunk_uid(chunk_uid: str) -> str:
    """Convert chunk UID to deterministic UUID for Qdrant compatibility.

    Args:
        chunk_uid: String chunk UID (hash)

    Returns:
        UUID string that Qdrant can accept as point ID
    """
    # deterministic UUIDv5 (namespaced); any fixed namespace works
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_uid))


def parse_links_out(links_str: Optional[str]) -> List[Dict[str, str]]:
    """Parse links_out string into structured JSON array.

    Args:
        links_str: JSON string containing Python dict representations

    Returns:
        List of link dictionaries with text and href keys
    """
    if not links_str:
        return []

    try:
        # Parse the JSON array of string representations
        link_strings = json.loads(links_str)
        links = []
        for link_str in link_strings:
            if not link_str:
                continue
            try:
                # Parse each string representation of dict using ast.literal_eval
                link_dict = ast.literal_eval(link_str)
                if (
                    isinstance(link_dict, dict)
                    and "text" in link_dict
                    and "href" in link_dict
                ):
                    # Only include links with actual content
                    if link_dict["text"].strip() and link_dict["href"].strip():
                        links.append(
                            {
                                "text": link_dict["text"].strip(),
                                "href": link_dict["href"].strip(),
                            }
                        )
            except (ValueError, SyntaxError):
                # Skip malformed individual link entries
                continue
        return links
    except (json.JSONDecodeError, TypeError):
        # Return empty list for completely malformed data
        return []


KV = Dict[str, Union[str, List[str]]]

_PAIR_RE = re.compile(
    r"""
    \s*                          # leading space
    (?P<key>[^=;]+?)             # key = anything up to '=' or ';'
    \s*=\s*
    (?P<val>[^;]*)               # value = anything up to next ';' (greedy)
    \s*(?:;|$)                   # ends with ';' or EOS
""",
    re.VERBOSE,
)


def parse_kv_preview(kv_str: Optional[str]) -> KV:
    """
    Parse a 'key=value; key2=value2' string to a dict.
    - Trims whitespace
    - Unescapes HTML entities
    - Strips surrounding quotes in values
    - Handles trailing semicolons and duplicate keys (collates to list)
    """
    if not kv_str:
        return {}

    out: KV = {}

    for m in _PAIR_RE.finditer(kv_str):
        raw_key = m.group("key").strip()
        raw_val = m.group("val").strip()

        if not raw_key:
            continue

        # Unescape HTML entities and strip surrounding quotes
        key = html.unescape(raw_key).strip().strip("\"'“”‘’")
        val = html.unescape(raw_val).strip().strip("\"'“”‘’")

        if not val:
            continue

        # Collate duplicates into a list
        if key in out:
            existing_value = out[key]
            if isinstance(existing_value, list):
                existing_value.append(val)
            else:
                out[key] = [existing_value, val]
        else:
            out[key] = val

    return out


class QdrantVectorStore:
    """Qdrant vector database wrapper for semantic search."""

    def __init__(
        self,
        collection_name: str = "w40k_chunks",
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: int = 1536,  # text-embedding-3-small
        distance: Distance = Distance.COSINE,
    ):
        """Initialize Qdrant client and configuration.

        Args:
            collection_name: Name of the collection to store vectors
            host: Qdrant server host (for local deployment)
            port: Qdrant server port (for local deployment)
            url: Full URL for Qdrant cloud (overrides host/port)
            api_key: API key for Qdrant cloud
            vector_size: Dimension of embedding vectors
            distance: Distance metric for similarity search
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        # Initialize client based on deployment type
        if url or api_key:
            # Qdrant Cloud
            self.client = QdrantClient(
                url=url or os.getenv("QDRANT_URL"),
                api_key=api_key or os.getenv("QDRANT_API_KEY"),
            )
        else:
            # Local Qdrant
            self.client = QdrantClient(host=host, port=port)

    def create_collection(self, recreate: bool = False) -> bool:
        """Create the collection with proper schema.

        Args:
            recreate: If True, delete existing collection and recreate

        Returns:
            True if collection was created or already exists
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if collection_exists and recreate:
                print(f"🗑️  Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                collection_exists = False

            if not collection_exists:
                print(f"🏗️  Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=self.distance
                    ),
                )

                # Create indexes for efficient filtering
                self._create_indexes()
                print(
                    f"✅ Collection created with {self.vector_size}D vectors using {self.distance} distance"
                )
            else:
                print(f"📁 Collection already exists: {self.collection_name}")

            return True

        except Exception as e:
            print(f"❌ Failed to create collection: {e}")
            return False

    def _create_indexes(self) -> None:
        """Create payload indexes for efficient filtering."""
        indexes_to_create = [
            ("article_id", models.PayloadSchemaType.INTEGER),
            ("block_type", models.PayloadSchemaType.KEYWORD),
            ("lead", models.PayloadSchemaType.BOOL),
            ("token_count", models.PayloadSchemaType.INTEGER),
            ("article_title", models.PayloadSchemaType.TEXT),
            ("canonical_url", models.PayloadSchemaType.KEYWORD),
            ("parser_version", models.PayloadSchemaType.KEYWORD),
            ("active", models.PayloadSchemaType.BOOL),
            ("section_path", models.PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_type in indexes_to_create:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                # Index might already exist, continue
                print(f"⚠️  Index creation warning for {field_name}: {e}")

    def upsert_chunks(
        self,
        chunks_with_embeddings: List[tuple[Chunk, List[float]]],
        batch_size: int = 100,
    ) -> tuple[int, List[str]]:
        """Upsert chunks with their embeddings to Qdrant.

        Args:
            chunks_with_embeddings: List of (chunk, embedding) tuples
            batch_size: Number of points to upsert per batch

        Returns:
            Tuple of (number of points successfully upserted, list of successfully upserted point IDs)
        """
        if not chunks_with_embeddings:
            return 0, []

        total_upserted = 0
        successful_point_ids = []
        batch_count = (len(chunks_with_embeddings) + batch_size - 1) // batch_size

        # Process in batches with progress bar
        with tqdm(
            total=len(chunks_with_embeddings),
            desc="Uploading to Qdrant",
            unit="chunks",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for i in range(0, len(chunks_with_embeddings), batch_size):
                batch = chunks_with_embeddings[i : i + batch_size]
                points = []

                batch_num = i // batch_size + 1
                pbar.set_description(f"Batch {batch_num}/{batch_count}")

                # Validate dimension once per batch before processing
                expected = self.vector_size
                for chunk, emb in batch:
                    if emb and len(emb) != expected:
                        raise ValueError(
                            f"Embedding dim mismatch: got {len(emb)} but collection is {expected}D"
                        )

                for chunk, embedding in batch:
                    if not embedding:
                        pbar.update(1)  # Count skipped chunks
                        continue  # Skip chunks without embeddings

                    # Build payload from chunk metadata with structured data
                    payload = {
                        "chunk_uid": chunk.chunk_uid,  # Keep original chunk UID for debugging
                        "article_id": chunk.article_id,
                        "article_title": chunk.article_title,
                        "canonical_url": chunk.canonical_url,
                        "section_path": (
                            json.loads(chunk.section_path)
                            if isinstance(chunk.section_path, str)
                            else chunk.section_path
                        ),
                        "block_type": getattr(
                            chunk.block_type,
                            "value",
                            str(chunk.block_type).split(".")[-1].lower(),
                        ),
                        "chunk_index": chunk.chunk_index,
                        "micro_start": chunk.micro_start,
                        "micro_end": chunk.micro_end,
                        "text": chunk.text,
                        "embedding_input": chunk.embedding_input,
                        "token_count": chunk.token_count,
                        "kv_preview": chunk.kv_preview,  # Keep original for backwards compatibility
                        "kv_data": parse_kv_preview(
                            chunk.kv_preview
                        ),  # Add structured version
                        "lead": chunk.lead,
                        "parser_version": chunk.parser_version,
                        "links_out": parse_links_out(
                            chunk.links_out
                        ),  # Parse to structured array
                        "active": chunk.active,
                        "created_at": (
                            chunk.created_at.isoformat() if chunk.created_at else None
                        ),
                        "updated_at": (
                            chunk.updated_at.isoformat() if chunk.updated_at else None
                        ),
                    }

                    # Remove None values to reduce payload size
                    payload = {k: v for k, v in payload.items() if v is not None}

                    # Convert chunk UID to UUID for Qdrant compatibility
                    point_id = point_id_from_chunk_uid(chunk.chunk_uid)

                    points.append(
                        models.PointStruct(
                            id=point_id,  # Use UUID derived from chunk UID
                            vector=embedding,
                            payload=payload,
                        )
                    )

                if points:
                    try:
                        self.client.upsert(
                            collection_name=self.collection_name, points=points
                        )
                        # Collect successful point IDs (convert back to chunk UIDs for tracking)
                        batch_chunk_uids = [
                            point.payload["chunk_uid"] for point in points
                        ]
                        successful_point_ids.extend(batch_chunk_uids)
                        total_upserted += len(points)
                        pbar.update(
                            len(points)
                        )  # Update for successfully uploaded points
                    except Exception as e:
                        print(f"❌ Failed to upsert batch {batch_num}: {e}")
                        pbar.update(
                            len(batch)
                        )  # Update progress even for failed batches

        return total_upserted, successful_point_ids

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
    ) -> List[models.ScoredPoint]:
        """Search for similar chunks.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            filter_conditions: Qdrant filter conditions
            score_threshold: Minimum similarity score

        Returns:
            List of scored points with chunk data
        """
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=(
                    models.Filter(**filter_conditions) if filter_conditions else None
                ),
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,  # Don't return vectors to save bandwidth
            )
            return search_result
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def get_collection_info(self) -> Dict:
        """Get collection statistics and info.

        Returns:
            Dictionary with collection information
        """
        try:
            info = self.client.get_collection(self.collection_name)
            cfg = getattr(info.config.params, "vectors", None)

            size = 0
            dist = "N/A"
            if isinstance(cfg, dict) and cfg:
                first = next(iter(cfg.values()))
                size = getattr(first, "size", 0)
                dist = str(getattr(first, "distance", "N/A"))
            elif hasattr(cfg, "size"):  # single vector
                size = getattr(cfg, "size", 0)
                dist = str(getattr(cfg, "distance", "N/A"))

            return {
                "name": self.collection_name,
                "vector_size": size,
                "distance": dist,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status,
                "optimizer_status": (
                    str(info.optimizer_status) if info.optimizer_status else "N/A"
                ),
                "indexed_vectors_count": info.indexed_vectors_count or 0,
            }
        except Exception as e:
            print(f"❌ Failed to get collection info: {e}")
            return {}

    def delete_points(self, point_ids: List[str]) -> bool:
        """Delete points by their IDs.

        Args:
            point_ids: List of point IDs to delete

        Returns:
            True if deletion was successful
        """
        try:
            # Create list with Union type for Qdrant compatibility
            extended_point_ids: List[Union[str, int]] = list(point_ids)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=extended_point_ids),
            )
            print(f"🗑️  Deleted {len(point_ids)} points")
            return True
        except Exception as e:
            print(f"❌ Failed to delete points: {e}")
            return False

    def count_points(self, filter_conditions: Optional[Dict] = None) -> int:
        """Count points in collection.

        Args:
            filter_conditions: Optional filter conditions

        Returns:
            Number of points matching the filter
        """
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=(
                    models.Filter(**filter_conditions) if filter_conditions else None
                ),
            )
            return result.count
        except Exception as e:
            print(f"❌ Failed to count points: {e}")
            return 0

    def health_check(self) -> bool:
        """Check if Qdrant is healthy and accessible.

        Returns:
            True if Qdrant is healthy
        """
        try:
            # Try to get collections list
            self.client.get_collections()
            return True
        except Exception as e:
            print(f"❌ Qdrant health check failed: {e}")
            return False


def create_qdrant_filters(
    article_ids: Optional[List[int]] = None,
    block_types: Optional[List[str]] = None,
    lead_only: Optional[bool] = None,
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    active_only: bool = True,
) -> Dict:
    """Helper function to create Qdrant filter conditions.

    Args:
        article_ids: Filter by specific article IDs
        block_types: Filter by block types
        lead_only: Filter for lead paragraphs only
        min_tokens: Minimum token count
        max_tokens: Maximum token count
        active_only: Filter for active chunks only

    Returns:
        Qdrant filter dictionary
    """
    conditions = []

    if article_ids:
        conditions.append({"key": "article_id", "match": {"any": article_ids}})

    if block_types:
        conditions.append({"key": "block_type", "match": {"any": block_types}})

    if lead_only is not None:
        conditions.append({"key": "lead", "match": {"value": lead_only}})

    if min_tokens is not None:
        conditions.append({"key": "token_count", "range": {"gte": min_tokens}})

    if max_tokens is not None:
        conditions.append({"key": "token_count", "range": {"lte": max_tokens}})

    if active_only:
        conditions.append({"key": "active", "match": {"value": True}})

    if conditions:
        return {"must": conditions}
    return {}
