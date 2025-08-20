"""Pinecone-backed implementation of the VectorServicePort.

This adapter mirrors the QdrantVectorService surface so the app and scripts
can switch providers via settings without code changes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import uuid

from ...ports.vector_service import VectorServicePort
from ...infrastructure.rag.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


def point_id_from_chunk_uid(chunk_uid: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_uid))


def _build_pinecone_filter(
    *,
    article_ids: Optional[List[int]] = None,
    block_types: Optional[List[str]] = None,
    lead_only: Optional[bool] = None,
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    active_only: bool = True,
) -> Dict:
    f: Dict[str, Dict] = {}
    if article_ids:
        f["article_id"] = {"$in": article_ids}
    if block_types:
        f["block_type"] = {"$in": block_types}
    if lead_only is not None:
        f["lead"] = {"$eq": bool(lead_only)}
    if min_tokens is not None and max_tokens is not None:
        f["token_count"] = {"$gte": min_tokens, "$lte": max_tokens}
    elif min_tokens is not None:
        f["token_count"] = {"$gte": min_tokens}
    elif max_tokens is not None:
        f["token_count"] = {"$lte": max_tokens}
    if active_only:
        f["active"] = {"$eq": True}
    return f


class PineconeVectorService(VectorServicePort):
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        *,
        api_key: str,
        index_name: str,
        environment: Optional[str] = None,
        vector_size: int = 1536,
        metric: str = "cosine",
    ) -> None:
        self.embedding_generator = embedding_generator
        self.index_name = index_name
        self.vector_size = vector_size
        self.metric = metric
        # Internal client/index handles; typed Any to keep optional dep light and satisfy Pylance
        self._pc: Any = None
        self._index: Any = None
        self._environment: Optional[str] = environment

        try:
            # Lazy import so repo doesn't require pinecone unless used
            from pinecone import Pinecone  # type: ignore

            self._pc = Pinecone(api_key=api_key)
            # Environment is used on index creation via ServerlessSpec; not needed to connect to existing index
        except Exception as e:
            # Defer error until used
            logger.error("Failed to initialize Pinecone client: %s", e)
            raise

        try:
            self._index = self._pc.Index(self.index_name)
        except Exception:
            # Index may not exist yet; operations like create_collection will create it lazily
            self._index = None

    def _ensure_index(self) -> None:
        """Ensure the Pinecone index handle is available."""
        if self._index is None:
            self._index = self._pc.Index(self.index_name)

    # Retrieval
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
        if not self._index:
            # Attempt to open index now if created after construction
            self._ensure_index()

        query_embedding = self.embedding_generator.generate_embedding(query_text)
        if not query_embedding:
            return []

        flt = _build_pinecone_filter(
            article_ids=article_ids,
            block_types=block_types,
            lead_only=lead_only,
            active_only=active_only,
        )

        try:
            res = self._index.query(
                vector=query_embedding,
                top_k=limit,
                include_values=False,
                include_metadata=True,
                filter=flt or None,
            )
        except Exception as e:
            logger.error("Pinecone query failed: %s", e)
            return []

        out: List[Dict] = []
        for match in getattr(res, "matches", []) or []:
            md = getattr(match, "metadata", {}) or {}
            score = getattr(match, "score", 0.0) or 0.0
            if min_score is not None and score < float(min_score):
                continue
            out.append(
                {
                    "point_id": getattr(match, "id", None),
                    "chunk_uid": md.get("chunk_uid"),
                    "score": score,
                    "article_id": md.get("article_id"),
                    "article_title": md.get("article_title"),
                    "text": md.get("text"),
                    "block_type": md.get("block_type"),
                    "lead": md.get("lead"),
                    "section_path": md.get("section_path"),
                    "canonical_url": md.get("canonical_url"),
                    "token_count": md.get("token_count"),
                    "kv_preview": md.get("kv_preview"),
                    "kv_data": md.get("kv_data"),
                    "links_out": md.get("links_out"),
                }
            )

        return out

    def get_collection_info(self) -> Dict:
        try:
            if not self._index:
                self._ensure_index()
            stats = self._index.describe_index_stats()
            total: int = 0
            if isinstance(stats, dict):
                total = int(
                    stats.get("total_vector_count")
                    or stats.get("totalVectorCount")
                    or 0
                )
                if not total:
                    ns = stats.get("namespaces") or {}
                    if isinstance(ns, dict):
                        total = sum(
                            int(v.get("vectorCount") or v.get("vector_count") or 0)
                            for v in ns.values()
                            if isinstance(v, dict)
                        )
            else:
                total = int(
                    getattr(stats, "total_vector_count", 0)
                    or getattr(stats, "totalVectorCount", 0)
                    or 0
                )

            return {
                "name": self.index_name,
                "vector_size": self.vector_size,
                "distance": self.metric,
                "points_count": total,
                "segments_count": 0,
                "status": "ready",
                "indexed_vectors_count": total,
            }
        except Exception as e:
            return {"error": str(e)}

    # Indexing/management
    def create_collection(self, recreate: bool = False) -> bool:
        try:
            # Pinecone v3 serverless index creation
            from pinecone import ServerlessSpec  # type: ignore

            pc = self._pc  # type: ignore[assignment]
            existing = {i["name"] for i in pc.list_indexes()}
            if self.index_name in existing and recreate:
                pc.delete_index(self.index_name)
                existing.remove(self.index_name)

            if self.index_name not in existing:
                spec = None
                if self._environment:
                    # Best-effort: assume AWS in provided region
                    spec = ServerlessSpec(cloud="aws", region=self._environment)
                pc.create_index(
                    name=self.index_name,
                    dimension=self.vector_size,
                    metric=self.metric,
                    spec=spec,
                )
            # Open handle
            self._index = pc.Index(self.index_name)
            return True
        except Exception as e:
            logger.error("Failed to create Pinecone index: %s", e)
            return False

    def upsert_chunks(
        self,
        chunks_with_embeddings: List[Tuple[object, List[float]]],
        batch_size: int = 100,
        show_progress: bool = False,
        ensure_collection: bool = False,
    ) -> Tuple[int, List[str]]:
        if not chunks_with_embeddings:
            return 0, []
        if ensure_collection:
            self.create_collection(recreate=False)
        if not self._index:
            self._ensure_index()

        total = 0
        uids: List[str] = []
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore

                pbar = tqdm(total=len(chunks_with_embeddings), desc="Uploading to Pinecone", unit="chunks")
            except Exception:
                pbar = None

        try:
            for i in range(0, len(chunks_with_embeddings), batch_size):
                batch = chunks_with_embeddings[i : i + batch_size]
                vectors = []
                for chunk, emb in batch:
                    if not emb:
                        if pbar:
                            pbar.update(1)
                        continue
                    # Build metadata mirroring Qdrant payload
                    md = {
                        "chunk_uid": getattr(chunk, "chunk_uid", None),
                        "article_id": getattr(chunk, "article_id", None),
                        "article_title": getattr(chunk, "article_title", None),
                        "canonical_url": getattr(chunk, "canonical_url", None),
                        "section_path": getattr(chunk, "section_path", None),
                        "block_type": getattr(chunk, "block_type", None),
                        "chunk_index": getattr(chunk, "chunk_index", None),
                        "text": getattr(chunk, "text", None),
                        "embedding_input": getattr(chunk, "embedding_input", None),
                        "token_count": getattr(chunk, "token_count", None),
                        "kv_preview": getattr(chunk, "kv_preview", None),
                        "kv_data": getattr(chunk, "kv_data", None),
                        "lead": getattr(chunk, "lead", None),
                        "parser_version": getattr(chunk, "parser_version", None),
                        "links_out": getattr(chunk, "links_out", None),
                        "active": getattr(chunk, "active", True),
                    }
                    md = {k: v for k, v in md.items() if v is not None}
                    cid = getattr(chunk, "chunk_uid", None) or str(uuid.uuid4())
                    vid = point_id_from_chunk_uid(cid)
                    vectors.append({"id": vid, "values": emb, "metadata": md})
                    uids.append(cid)

                if vectors:
                    self._index.upsert(vectors=vectors)
                    total += len(vectors)
                    if pbar:
                        pbar.update(len(vectors))
        finally:
            if pbar:
                pbar.close()

        return total, uids

    def delete_points(self, point_ids: List[str]) -> bool:
        try:
            if not self._index:
                self._ensure_index()
            self._index.delete(ids=list(point_ids))
            return True
        except Exception as e:
            logger.error("Failed to delete Pinecone points: %s", e)
            return False

    def delete_collection(self) -> bool:
        try:
            pc = self._pc
            pc.delete_index(self.index_name)
            return True
        except Exception as e:
            logger.error("Failed to delete Pinecone index: %s", e)
            return False

    def health_check(self) -> bool:
        try:
            pc = self._pc
            pc.list_indexes()
            return True
        except Exception as e:
            logger.error("Pinecone health check failed: %s", e)
            return False
