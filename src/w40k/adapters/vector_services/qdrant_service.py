"""Qdrant-backed implementation of the VectorServicePort.

Consolidated adapter: initializes the Qdrant client and exposes both
retrieval and indexing/management APIs; replaces the old vector client.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import uuid

from ...ports.vector_service import VectorServicePort
from ...infrastructure.rag.embeddings import EmbeddingGenerator
from ...infrastructure.rag.utils import (
    create_qdrant_filters,
    parse_kv_preview,
    parse_links_out,
    normalize_section_path,
)
from ...infrastructure.database.models import Chunk
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger(__name__)


def point_id_from_chunk_uid(chunk_uid: str) -> str:
    """Deterministically map a chunk UID to a UUIDv5 for Qdrant IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_uid))


class QdrantVectorService(VectorServicePort):
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        *,
        collection_name: str = "w40k_chunks",
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE,
    ) -> None:
        self.embedding_generator = embedding_generator
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        # Initialize Qdrant client (cloud or local)
        if url or api_key:
            if not (url and api_key):
                raise ValueError("Both url and api_key must be provided for Qdrant Cloud configuration.")
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port)

    def search_similar_chunks(
        self,
        query_text: str,
        limit: int = 10,
        article_ids: Optional[List[int]] = None,
        block_types: Optional[List[str]] = None,
        lead_only: Optional[bool] = None,
        min_score: Optional[float] = None,
        active_only: bool = True,
    ) -> List[Dict]:
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        if not query_embedding:
            return []

        filter_conditions = create_qdrant_filters(
            article_ids=article_ids,
            block_types=block_types,
            lead_only=lead_only,
            active_only=active_only,
        )

        # Query Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=(models.Filter(**filter_conditions) if filter_conditions else None),
            score_threshold=min_score,
            with_payload=True,
            with_vectors=False,
        )

        formatted: List[Dict] = []
        for result in results:
            payload = result.payload or {}
            formatted.append(
                {
                    "point_id": result.id,
                    "chunk_uid": payload.get("chunk_uid"),
                    "score": result.score,
                    "article_id": payload.get("article_id"),
                    "article_title": payload.get("article_title"),
                    "text": payload.get("text"),
                    "block_type": payload.get("block_type"),
                    "lead": payload.get("lead"),
                    "section_path": payload.get("section_path"),
                    "canonical_url": payload.get("canonical_url"),
                    "token_count": payload.get("token_count"),
                    "kv_preview": payload.get("kv_preview"),
                    "kv_data": payload.get("kv_data"),
                    "links_out": payload.get("links_out"),
                }
            )

        return formatted

    def get_collection_info(self) -> Dict:
        try:
            info = self.client.get_collection(self.collection_name)
            cfg = getattr(info.config.params, "vectors", None)
            size = 0
            dist = "N/A"
            if isinstance(cfg, dict) and cfg:
                first = next(iter(cfg.values()))
                size = getattr(first, "size", 0)
                dist = str(getattr(first, "distance", "N/A"))
            elif hasattr(cfg, "size"):
                size = getattr(cfg, "size", 0)
                dist = str(getattr(cfg, "distance", "N/A"))
            return {
                "name": self.collection_name,
                "vector_size": size,
                "distance": dist,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status,
                "indexed_vectors_count": info.indexed_vectors_count or 0,
            }
        except Exception as e:
            return {"error": str(e)}

    # Index management APIs
    def create_collection(self, recreate: bool = False) -> bool:
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if exists and recreate:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                exists = False
            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
                )
                self._create_indexes()
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def _create_indexes(self) -> None:
        indexes = [
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
        for field_name, field_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                logger.warning(f"Index creation warning for {field_name}: {e}")

    def upsert_chunks(
        self,
        chunks_with_embeddings: List[Tuple[Chunk, List[float]]],
        batch_size: int = 100,
        show_progress: bool = False,
        ensure_collection: bool = False,
    ) -> Tuple[int, List[str]]:
        if not chunks_with_embeddings:
            return 0, []
        if ensure_collection:
            self.create_collection(recreate=False)

        total_upserted = 0
        successful_chunk_uids: List[str] = []

        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore

                pbar = tqdm(
                    total=len(chunks_with_embeddings),
                    desc="Uploading to Qdrant",
                    unit="chunks",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )
            except Exception:
                pbar = None

        try:
            batch_count = (len(chunks_with_embeddings) + batch_size - 1) // batch_size
            for i in range(0, len(chunks_with_embeddings), batch_size):
                batch = chunks_with_embeddings[i : i + batch_size]
                points: List[models.PointStruct] = []
                batch_num = i // batch_size + 1
                if pbar:
                    pbar.set_description(f"Batch {batch_num}/{batch_count}")

                expected = self.vector_size
                for chunk, emb in batch:
                    if emb and len(emb) != expected:
                        raise ValueError(
                            f"Embedding dim mismatch: got {len(emb)} but collection is {expected}D"
                        )

                for chunk, embedding in batch:
                    if not embedding:
                        if pbar:
                            pbar.update(1)
                        continue

                    payload = {
                        "chunk_uid": chunk.chunk_uid,
                        "article_id": chunk.article_id,
                        "article_title": chunk.article_title,
                        "canonical_url": chunk.canonical_url,
                        "section_path": normalize_section_path(chunk.section_path),
                        "block_type": getattr(
                            chunk.block_type,
                            "value",
                            str(chunk.block_type).split(".")[-1].lower(),
                        ),
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "embedding_input": chunk.embedding_input,
                        "token_count": chunk.token_count,
                        "kv_preview": chunk.kv_preview,
                        "kv_data": parse_kv_preview(chunk.kv_preview),
                        "lead": chunk.lead,
                        "parser_version": chunk.parser_version,
                        "links_out": parse_links_out(chunk.links_out),
                        "active": chunk.active,
                        "created_at": (chunk.created_at.isoformat() if chunk.created_at else None),
                        "updated_at": (chunk.updated_at.isoformat() if chunk.updated_at else None),
                    }
                    payload = {k: v for k, v in payload.items() if v is not None}

                    point_id = point_id_from_chunk_uid(chunk.chunk_uid)
                    points.append(
                        models.PointStruct(id=point_id, vector=embedding, payload=payload)
                    )

                if points:
                    try:
                        self.client.upsert(collection_name=self.collection_name, points=points)
                        batch_chunk_uids = [p.payload["chunk_uid"] for p in points if p.payload]
                        successful_chunk_uids.extend(batch_chunk_uids)
                        total_upserted += len(points)
                        if pbar:
                            pbar.update(len(points))
                    except Exception as e:
                        logger.error(f"Failed to upsert batch {batch_num}: {e}")
                        if pbar:
                            pbar.update(len(batch))
        finally:
            if pbar:
                pbar.close()

        return total_upserted, successful_chunk_uids

    def delete_points(self, point_ids: List[str]) -> bool:
        try:
            extended: List[Union[str, int]] = list(point_ids)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=extended),
            )
            logger.info(f"Deleted {len(point_ids)} points from {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete points: {e}")
            return False

    def delete_collection(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
