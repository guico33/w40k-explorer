"""Answer generation use case - core business logic for answering queries."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from ..core.models import QueryResult
from ..ports.vector_operations import VectorOperationsPort
from ..ports.llm_client import LLMClient
from .utils import format_section_path, truncate_text

logger = logging.getLogger(__name__)


class AnswerService:
    """Service for generating answers to user queries using RAG."""

    def __init__(
        self,
        vector_operations: VectorOperationsPort,
        llm_client: LLMClient,
        model: Optional[str] = None,
        initial_k: int = 60,
        max_context_chunks: int = 12,
        min_similarity_score: Optional[float] = 0.2,
        max_tokens: int = 900,
        context_max_words: int = 200,
        query_expansion_n: int = 0,
        lower_threshold_on_empty: bool = True,
        active_only: bool = True,
    ):
        """Initialize answer service.

        Args:
            vector_operations: Vector operations for similarity search
            llm_client: LLM client instance
            model: LLM model to use (if None, reads from OPENAI_LLM_MODEL env var)
            initial_k: Number of initial chunks to retrieve
            max_context_chunks: Maximum chunks to include in context
            min_similarity_score: Minimum similarity score threshold
            max_tokens: Maximum tokens for LLM response
            context_max_words: Max words per context chunk
            query_expansion_n: Number of query expansions to generate
            lower_threshold_on_empty: Relax threshold when no hits found
            active_only: Only search active chunks

        Raises:
            ValueError: If model is None and OPENAI_LLM_MODEL env var is not set
        """
        self.vec_ops = vector_operations
        self.llm_client = llm_client

        # Require explicit model (should be provided by Settings via factory)
        if not model:
            raise ValueError("Model is required. Pass settings.openai_llm_model to AnswerService.")
        self.model = model

        self.initial_k = initial_k
        self.max_context_chunks = max_context_chunks
        self.min_similarity_score = min_similarity_score
        self.max_tokens = max_tokens
        self.context_max_words = context_max_words
        self.query_expansion_n = max(0, int(query_expansion_n or 0))
        self.lower_threshold_on_empty = lower_threshold_on_empty
        self.active_only = active_only

    def answer_query(self, question: str) -> QueryResult:
        """Generate an answer to a user query.
        
        Args:
            question: User's question
            
        Returns:
            QueryResult with answer, citations, and metadata
        """
        start_time = datetime.now()

        try:
            # Step 1: Retrieve
            hits = self._retrieve(question)

            if not hits:
                return QueryResult(
                    answer="I couldn't find relevant information about this topic in the archives.",
                    citations=[],
                    confidence=0.0,
                    sources_used=0,
                    query_time_ms=int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    ),
                )

            # Step 2: Diversify with MMR
            diverse_chunks = self._diversify_mmr(hits, self.max_context_chunks)

            # Step 3: Pack context
            context_items = self._pack_context(diverse_chunks)

            # Step 4: Generate answer
            result = self._generate_answer(question, context_items)

            # Add timing
            result.query_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return result

        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            return QueryResult(
                answer="An error occurred while processing your query.",
                citations=[],
                confidence=0.0,
                sources_used=0,
                error=str(e),
                query_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            )

    def _retrieve(self, question: str) -> List[Dict]:
        """Retrieve semantically similar chunks with optional query expansion."""
        queries = [question]
        if self.query_expansion_n and self.query_expansion_n > 0:
            try:
                expansions = self._expand_queries(question, self.query_expansion_n)
                seen = set()
                for q in expansions:
                    if q and q not in seen:
                        queries.append(q)
                        seen.add(q)
                logger.info(
                    f"Query expansion generated {len(expansions)} variants; using {len(queries)} total queries"
                )
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")

        combined: Dict[str, Dict] = {}
        per_query_limit = max(1, self.initial_k // len(queries))

        for q in queries:
            q_hits = self.vec_ops.search_similar_chunks(
                query_text=q,
                limit=per_query_limit,
                min_score=self.min_similarity_score,
                active_only=self.active_only,
            )
            for h in q_hits:
                uid = h.get("chunk_uid") or h.get("point_id")
                if uid and (
                    uid not in combined
                    or (h.get("score", 0) > combined[uid].get("score", 0))
                ):
                    combined[uid] = h

        hits = sorted(combined.values(), key=lambda x: -x.get("score", 0.0))

        if not hits and self.lower_threshold_on_empty and self.min_similarity_score:
            logger.info(
                "No hits with current threshold; retrying with relaxed threshold"
            )
            for q in queries:
                q_hits = self.vec_ops.search_similar_chunks(
                    query_text=q,
                    limit=per_query_limit,
                    min_score=None,
                    active_only=self.active_only,
                )
                for h in q_hits:
                    uid = h.get("chunk_uid") or h.get("point_id")
                    if uid and (
                        uid not in combined
                        or (h.get("score", 0) > combined[uid].get("score", 0))
                    ):
                        combined[uid] = h
            hits = sorted(combined.values(), key=lambda x: -x.get("score", 0.0))

        logger.info(
            f"Retrieved {len(hits)} combined chunks from {len(queries)} queries for: '{question[:50]}...'"
        )
        if hits:
            top = hits[0]
            logger.info(
                f"Top hit: {top.get('article_title', 'Unknown')} (score: {top.get('score', 0):.3f})"
            )
            # Log a brief preview of top 5 results
            for i, h in enumerate(hits[:5]):
                logger.info(
                    f"#{i+1}: {h.get('article_title', 'Unknown')} › {format_section_path(h.get('section_path'))} (score={h.get('score', 0):.3f})"
                )
        else:
            logger.warning("No chunks retrieved from vector search")
        return hits

    def _expand_queries(self, question: str, n: int = 5) -> List[str]:
        """Generate alternative phrasings to improve recall using the LLM.

        Inspired by LangChain's MultiQueryRetriever concept: produce multiple
        semantically diverse reformulations, including British/American spelling
        variants and common aliases/titles.
        """
        prompt = f"""
Propose alternative search queries that capture synonyms, aliases, and spelling variants.
Domain: Warhammer 40K.
Original question: {question}
Include British/American spellings (favored/favoured), titles (Warmaster), and entity names.
Return ONLY a JSON array of up to {n} short query strings.
"""
        response = self.llm_client.generate_structured_response(
            input_messages=[{"role": "user", "content": prompt}],
            model=self.model,
            text_format={
                "format": {
                    "type": "json_schema",
                    "name": "queries",
                    "strict": True,
                    "schema": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": max(1, n),
                        "items": {"type": "string"},
                    },
                }
            },
            max_output_tokens=200,
        )

        # Extract the message content text
        content_text = None
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                cont = (
                    item.content[0] if isinstance(item.content, list) else item.content
                )
                if getattr(cont, "type", None) == "output_text":
                    content_text = getattr(cont, "text", None)
                    break

        import json as _json

        if not content_text:
            return []
        try:
            arr = _json.loads(content_text)
            if isinstance(arr, list):
                return [str(x) for x in arr if isinstance(x, str) and x.strip()][:n]
        except Exception:
            return []
        return []

    def _diversify_mmr(self, hits: List[Dict], max_chunks: int) -> List[Dict]:
        """Apply MMR-style diversity: avoid duplicate article/sections."""
        selected = []
        seen_keys = set()
        article_counts = {}

        # Sort by score and lead preference (avoid penalizing longer chunks)
        sorted_hits = sorted(
            hits,
            key=lambda x: (
                -x.get("score", 0.0),
                -int(bool(x.get("lead", False))),  # Prefer lead paragraphs
            ),
        )

        for hit in sorted_hits:
            # Create deduplication key
            section_path = hit.get("section_path", [])
            if isinstance(section_path, str):
                try:
                    section_path = json.loads(section_path)
                except:
                    section_path = []

            article = hit.get("article_title", "")
            section_key = tuple(section_path[:2]) if section_path else ()
            dedupe_key = (hit.get("canonical_url", ""), section_key)

            # Skip if we've seen this exact article/section
            if dedupe_key in seen_keys:
                continue

            # Limit chunks per article (max 4)
            article_count = article_counts.get(article, 0)
            if article_count >= 4:
                continue

            # Accept this chunk
            selected.append(hit)
            seen_keys.add(dedupe_key)
            article_counts[article] = article_count + 1

            if len(selected) >= max_chunks:
                break

        return selected

    def _pack_context(self, chunks: List[Dict]) -> List[Dict]:
        """Format chunks for LLM context."""
        context_items = []

        for i, chunk in enumerate(chunks):
            # Extract KV facts if this is an infobox/lead chunk
            kv_facts = {}
            if chunk.get("lead") and chunk.get("kv_data"):
                kv_data = chunk.get("kv_data", {})
                if isinstance(kv_data, dict):
                    # Take top 5 most relevant KV pairs
                    kv_facts = dict(list(kv_data.items())[:5])

            context_items.append(
                {
                    "id": i,
                    "article": chunk.get("article_title", "Unknown"),
                    "section": format_section_path(chunk.get("section_path")),
                    "url": chunk.get("canonical_url", ""),
                    "is_lead": bool(chunk.get("lead")),
                    "text": truncate_text(
                        chunk.get("text", ""), max_words=self.context_max_words
                    ),
                    "kv_facts": kv_facts,  # Include structured facts
                    "score": round(chunk.get("score", 0), 3),  # For debugging
                }
            )

        return context_items

    def _generate_answer(self, question: str, context_items: List[Dict]) -> QueryResult:
        """Generate answer using GPT with structured output."""

        # System prompt - concise and strict
        system_prompt = """You are a Warhammer 40K lore expert answering from a curated archive.

STRICT RULES:
1. Answer ONLY from the provided context passages
2. EVERY factual claim must cite its source using [id] notation
3. If information is not in context, explicitly say so
4. Be precise with names, dates, and factions
5. Use in-universe tone when appropriate
6. If context has kv_facts, prioritize those for entity attributes
7. Keep the answer to AT MOST 10 sentences total.
8. Each sentence must end with a [id] citation and the whole answer must fit under ~3000 characters."""

        # Format context for prompt
        context_str = json.dumps(context_items, ensure_ascii=False, indent=2)

        user_prompt = f"""Question: {question}

Context passages:
{context_str}

Provide your response following the required JSON structure."""

        try:
            # Use structured LLM client with proper JSON schema
            response = self.llm_client.generate_structured_response(
                input_messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                text_format={
                    "format": {
                        "type": "json_schema",
                        "name": "query_result",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "description": "≤10 sentences, each with [id] citation; concise.",
                                    "maxLength": 3000,
                                },
                                "citations_used": {
                                    "type": "array",
                                    "maxItems": 8,
                                    "items": {"type": "integer"},
                                    "description": "Array of context IDs actually cited",
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Float 0-1 based on context quality",
                                },
                            },
                            "required": ["answer", "citations_used", "confidence"],
                            "additionalProperties": False,
                        },
                    }
                },
                max_output_tokens=self.max_tokens,
            )

            # Handle different response types
            logger.info(f"Response status: {response.status}")
            if hasattr(response, "incomplete_details") and response.incomplete_details:
                logger.info(
                    f"Incomplete details reason: {response.incomplete_details.reason}"
                )

            if response.status == "incomplete" and getattr(
                response, "incomplete_details", None
            ):
                if (
                    response.incomplete_details
                    and response.incomplete_details.reason == "max_output_tokens"
                ):
                    # Try to parse whatever text we received anyway
                    partial = self._extract_output_text(response)
                    if partial:
                        try:
                            partial_json = json.loads(partial)
                            parsed = QueryResult(
                                answer=partial_json.get("answer", ""),
                                citations=[],
                                confidence=float(partial_json.get("confidence", 0.5)),
                                sources_used=len(context_items),
                                citations_used=partial_json.get("citations_used", []),
                                context_items=context_items,
                                error="truncated",
                            )

                            # Rebuild citations from the parsed data
                            citations = []
                            for cid in partial_json.get("citations_used", []):
                                if isinstance(cid, int) and 0 <= cid < len(
                                    context_items
                                ):
                                    ctx = context_items[cid]
                                    citations.append(
                                        {
                                            "id": cid,
                                            "title": ctx["article"],
                                            "section": ctx["section"],
                                            "url": ctx["url"],
                                        }
                                    )

                            parsed.citations = citations
                            return parsed
                        except json.JSONDecodeError:
                            pass
                    # Last resort: compression retry (≤3 sentences, tighter cap)
                    return self._retry_compressed(question, context_items)
                elif (
                    response.incomplete_details
                    and response.incomplete_details.reason == "content_filter"
                ):
                    return QueryResult(
                        answer="Response was filtered for content policy.",
                        citations=[],
                        confidence=0.0,
                        sources_used=len(context_items),
                        error="content_filter",
                    )

            # Response processed successfully
            logger.debug(
                f"LLM response received with {len(response.output)} output items"
            )

            # Handle response based on actual structure
            if not response.output or len(response.output) == 0:
                return QueryResult(
                    answer="No response generated by the model.",
                    citations=[],
                    confidence=0.0,
                    sources_used=len(context_items),
                    error="empty_output",
                )

            # Find the message output (skip reasoning items)
            message_output = None
            for item in response.output:
                if hasattr(item, "type") and item.type == "message":
                    message_output = item
                    break

            if not message_output:
                return QueryResult(
                    answer="No message output found in response.",
                    citations=[],
                    confidence=0.0,
                    sources_used=len(context_items),
                    error="no_message_output",
                )

            # Extract content from message
            if hasattr(message_output, "content") and message_output.content:
                content = (
                    message_output.content[0]
                    if isinstance(message_output.content, list)
                    else message_output.content
                )

                if hasattr(content, "type") and content.type == "refusal":
                    refusal_text = getattr(content, "refusal", "No reason provided")
                    return QueryResult(
                        answer="I cannot provide an answer to this query.",
                        citations=[],
                        confidence=0.0,
                        sources_used=len(context_items),
                        error=f"refusal: {refusal_text}",
                    )

                # Extract text for parsing - only ResponseOutputText has .text attribute
                if (
                    hasattr(content, "type")
                    and content.type == "output_text"
                    and hasattr(content, "text")
                ):
                    response_text = content.text
                else:
                    response_text = str(content)
            else:
                return QueryResult(
                    answer="Could not extract content from message output.",
                    citations=[],
                    confidence=0.0,
                    sources_used=len(context_items),
                    error="no_content_found",
                )

            # Parse JSON response
            try:
                result_json = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback - maybe it's not JSON
                return QueryResult(
                    answer=response_text,
                    citations=[],
                    confidence=0.5,
                    sources_used=len(context_items),
                    error="non_json_response",
                )

            # Build proper citations
            citations = []
            for cid in result_json.get("citations_used", []):
                if isinstance(cid, int) and 0 <= cid < len(context_items):
                    ctx = context_items[cid]
                    citations.append(
                        {
                            "id": cid,
                            "title": ctx["article"],
                            "section": ctx["section"],
                            "url": ctx["url"],
                        }
                    )

            return QueryResult(
                answer=result_json.get("answer", "No answer generated."),
                citations=citations,
                confidence=min(
                    max(float(result_json.get("confidence", 0.5)), 0.0), 1.0
                ),
                sources_used=len(context_items),
                citations_used=result_json.get("citations_used", []),
                context_items=context_items,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return QueryResult(
                answer="Failed to generate properly formatted answer.",
                citations=[],
                confidence=0.0,
                sources_used=len(context_items),
                error=f"JSON parse error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return QueryResult(
                answer="An error occurred while generating the answer.",
                citations=[],
                confidence=0.0,
                sources_used=len(context_items),
                error=str(e),
            )

    def _extract_output_text(self, response) -> Optional[str]:
        """Extract text content from OpenAI response, handling truncated responses."""

        if getattr(response, "output_text", None):
            return response.output_text

        logger.info(
            f"Extracting output text from response with {len(getattr(response, 'output', []))} items"
        )

        # First, try to find a message item (normal case)
        for i, item in enumerate(getattr(response, "output", []) or []):
            item_type = getattr(item, "type", None)
            logger.info(f"Item {i}: type={item_type}")
            if item_type == "message":
                content = getattr(item, "content", None)
                if isinstance(content, list) and len(content) > 0:
                    cont = content[0]
                else:
                    cont = content
                if cont:
                    cont_type = getattr(cont, "type", None)
                    logger.info(f"Content type: {cont_type}")
                    if cont_type == "output_text":
                        text = getattr(cont, "text", None)
                        logger.info(
                            f"Extracted text length: {len(text) if text else 0}"
                        )
                        return text

        # If no message found, look for any text content in reasoning items (truncated case)
        logger.info(
            "No message item found, checking reasoning items for any text content..."
        )
        for i, item in enumerate(getattr(response, "output", []) or []):
            item_type = getattr(item, "type", None)
            if item_type == "reasoning":
                # Check if reasoning has any content we can use
                content = getattr(item, "content", None)
                if content:
                    logger.info(f"Found reasoning content: {type(content)}")
                    if isinstance(content, str) and content.strip():
                        logger.info(f"Using reasoning text: {len(content)} chars")
                        return content

        logger.warning("No extractable text found in response")
        return None

    def _retry_compressed(
        self, question: str, context_items: List[Dict]
    ) -> QueryResult:
        """Last resort: retry with compressed constraints for extremely verbose queries."""
        logger.info(f"Attempting compression retry for query: {question[:50]}...")
        # Use stricter limits for compression retry
        compressed_prompt = """You are a Warhammer 40K lore expert answering from a curated archive.

STRICT RULES:
1. Answer ONLY from the provided context passages
2. EVERY factual claim must cite its source using [id] notation
3. Keep the answer to EXACTLY 3 sentences or fewer
4. Be extremely concise and direct"""

        context_str = json.dumps(context_items, ensure_ascii=False, indent=2)
        user_prompt = f"""Question: {question}

Context passages:
{context_str}

Provide a compressed response following the required JSON structure."""

        try:
            response = self.llm_client.generate_structured_response(
                input_messages=[
                    {"role": "system", "content": compressed_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                text_format={
                    "format": {
                        "type": "json_schema",
                        "name": "query_result",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "description": "≤3 sentences with [id] citations",
                                    "maxLength": 600,
                                },
                                "citations_used": {
                                    "type": "array",
                                    "maxItems": 5,
                                    "items": {"type": "integer"},
                                    "description": "Array of context IDs actually cited",
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Float 0-1 based on context quality",
                                },
                            },
                            "required": ["answer", "citations_used", "confidence"],
                            "additionalProperties": False,
                        },
                    }
                },
                max_output_tokens=250,  # Very strict limit
            )

            # Parse the compressed response
            logger.info(f"Compression retry response status: {response.status}")
            if hasattr(response, "incomplete_details") and response.incomplete_details:
                logger.info(
                    f"Compression retry incomplete reason: {response.incomplete_details.reason}"
                )

            content_text = self._extract_output_text(response)
            if content_text:
                logger.info(f"Compression retry got content: {len(content_text)} chars")
                try:
                    result_json = json.loads(content_text)

                    # Build citations
                    citations = []
                    for cid in result_json.get("citations_used", []):
                        if isinstance(cid, int) and 0 <= cid < len(context_items):
                            ctx = context_items[cid]
                            citations.append(
                                {
                                    "id": cid,
                                    "title": ctx["article"],
                                    "section": ctx["section"],
                                    "url": ctx["url"],
                                }
                            )

                    return QueryResult(
                        answer=result_json.get(
                            "answer", "Unable to provide compressed answer."
                        ),
                        citations=citations,
                        confidence=min(
                            max(float(result_json.get("confidence", 0.3)), 0.0), 1.0
                        ),
                        sources_used=len(context_items),
                        citations_used=result_json.get("citations_used", []),
                        context_items=context_items,
                        error="compressed_retry",
                    )
                except json.JSONDecodeError as e:
                    logger.info(f"JSON parse failed for compression retry: {e}")
                    # Try to extract partial content even if truncated
                    if "Expecting" in str(e) and content_text:
                        # Attempt to salvage partial response
                        try:
                            # Try to find partial answer in the text
                            if '"answer":' in content_text:
                                answer_start = content_text.find('"answer":"') + 10
                                # Find the end of the answer (either next quote followed by comma, or end of string)
                                answer_end = content_text.find('","', answer_start)
                                if answer_end == -1:
                                    answer_end = content_text.find(
                                        '"', answer_start + 1
                                    )
                                if answer_end > answer_start:
                                    partial_answer = content_text[
                                        answer_start:answer_end
                                    ]
                                    return QueryResult(
                                        answer=f"{partial_answer} [Answer truncated]",
                                        citations=[],
                                        confidence=0.2,
                                        sources_used=len(context_items),
                                        error="partial_recovery",
                                    )
                        except Exception:
                            pass
            else:
                logger.warning("No content extracted from compression retry")

        except Exception as e:
            logger.warning(f"Compression retry failed: {e}")

        # Ultimate fallback
        return QueryResult(
            answer="Unable to generate answer due to response length constraints.",
            citations=[],
            confidence=0.0,
            sources_used=len(context_items),
            error="compression_failed",
        )
