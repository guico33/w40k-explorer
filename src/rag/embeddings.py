"""Embedding generation for Warhammer 40k wiki chunks."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import openai
from openai import OpenAI
from tqdm import tqdm

try:
    from ..database.models import Chunk
except ImportError:
    # Handle case when running as script
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from database.models import Chunk


class EmbeddingGenerator:
    """Generate embeddings for article chunks using OpenAI's text-embedding models."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize embedding generator.

        Args:
            model: OpenAI embedding model to use (uses EMBEDDING_MODEL env var if not provided)
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            batch_size: Number of chunks to process in each batch
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Initial delay between retries (exponential backoff)
        """
        # Get model from env var or parameter, but require it
        model_name = model or os.getenv("EMBEDDING_MODEL")
        if not model_name:
            raise ValueError(
                "Embedding model is required. Set EMBEDDING_MODEL environment variable or pass model parameter."
            )
        self.model = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=api_key, timeout=30.0)

        # Track usage statistics
        self.stats: Dict[str, Union[int, float]] = {
            "total_chunks": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "total_tokens": 0,
            "api_calls": 0,
            "retry_count": 0,
        }

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of embedding values or None if failed
        """
        if not text.strip():
            return None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embeddings.create(model=self.model, input=text)

                self.stats["api_calls"] += 1
                tokens = getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
                self.stats["total_tokens"] += tokens

                return response.data[0].embedding

            except openai.APITimeoutError as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    print(
                        f"⏰ Request timeout, retrying in {delay:.1f}s... (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    time.sleep(delay)
                    self.stats["retry_count"] += 1
                else:
                    print(
                        f"❌ Request timeout after {self.max_retries} retries: {e}"
                    )
                    return None
                    
            except openai.RateLimitError as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    print(
                        f"⏳ Rate limit hit, retrying in {delay:.1f}s... (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    time.sleep(delay)
                    self.stats["retry_count"] += 1
                else:
                    print(
                        f"❌ Rate limit exceeded after {self.max_retries} retries: {e}"
                    )
                    return None

            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    print(
                        f"⚠️  API error, retrying in {delay:.1f}s... (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                    )
                    time.sleep(delay)
                    self.stats["retry_count"] += 1
                else:
                    print(
                        f"❌ Failed to generate embedding after {self.max_retries} retries: {e}"
                    )
                    return None

        return None

    def generate_batch_embeddings(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (same order as input, None for failures)
        """
        if not texts:
            return []

        # Filter out empty texts but preserve indices
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)

        if not valid_texts:
            return [None] * len(texts)

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=self.model, input=valid_texts
                )

                self.stats["api_calls"] += 1
                tokens = getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
                self.stats["total_tokens"] += tokens

                # Build result list with None for invalid indices
                results: List[Optional[List[float]]] = [None] * len(texts)
                for i, embedding_data in enumerate(response.data):
                    original_index = valid_indices[i]
                    results[original_index] = embedding_data.embedding

                return results

            except openai.APITimeoutError as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    print(
                        f"⏰ Batch timeout, retrying in {delay:.1f}s... (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    time.sleep(delay)
                    self.stats["retry_count"] += 1
                else:
                    print(
                        f"❌ Batch timeout after {self.max_retries} retries: {e}"
                    )
                    return [None] * len(texts)

            except openai.RateLimitError as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    print(
                        f"⏳ Rate limit hit, retrying batch in {delay:.1f}s... (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    time.sleep(delay)
                    self.stats["retry_count"] += 1
                else:
                    print(
                        f"❌ Rate limit exceeded for batch after {self.max_retries} retries: {e}"
                    )
                    return [None] * len(texts)

            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    print(
                        f"⚠️  API error for batch, retrying in {delay:.1f}s... (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                    )
                    time.sleep(delay)
                    self.stats["retry_count"] += 1
                else:
                    print(
                        f"❌ Failed to generate batch embeddings after {self.max_retries} retries: {e}"
                    )
                    return [None] * len(texts)

        return [None] * len(texts)

    def process_chunks(
        self, chunks: List[Chunk]
    ) -> List[Tuple[Chunk, Optional[List[float]]]]:
        """Process a list of chunks and generate embeddings.

        Args:
            chunks: List of Chunk objects to process

        Returns:
            List of (chunk, embedding) tuples
        """
        if not chunks:
            return []

        print(f"🔄 Processing {len(chunks)} chunks for embedding generation...")

        results = []

        # Process in batches with progress bar
        batch_count = (len(chunks) + self.batch_size - 1) // self.batch_size
        
        with tqdm(
            total=len(chunks), 
            desc="Generating embeddings",
            unit="chunks",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]
                batch_texts = [chunk.embedding_input or chunk.text for chunk in batch]

                # Update progress bar description with current batch
                batch_num = i // self.batch_size + 1
                pbar.set_description(f"Batch {batch_num}/{batch_count}")

                embeddings = self.generate_batch_embeddings(batch_texts)

                # Pair chunks with their embeddings
                for chunk, embedding in zip(batch, embeddings):
                    results.append((chunk, embedding))
                    self.stats["total_chunks"] += 1

                    if embedding is not None:
                        self.stats["successful_embeddings"] += 1
                    else:
                        self.stats["failed_embeddings"] += 1
                    
                    # Update progress bar
                    pbar.update(1)

        success_rate = (
            self.stats["successful_embeddings"] / self.stats["total_chunks"]
        ) * 100
        print(
            f"✅ Embedding generation complete: {self.stats['successful_embeddings']}/{self.stats['total_chunks']} successful ({success_rate:.1f}%)"
        )

        return results

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get embedding generation statistics.

        Returns:
            Dictionary with usage statistics
        """
        stats = self.stats.copy()
        if stats["total_chunks"] > 0:
            stats["success_rate"] = (
                stats["successful_embeddings"] / stats["total_chunks"]
            ) * 100
        else:
            stats["success_rate"] = 0.0

        if stats["successful_embeddings"] > 0:
            stats["avg_tokens_per_chunk"] = (
                stats["total_tokens"] / stats["successful_embeddings"]
            )
        else:
            stats["avg_tokens_per_chunk"] = 0.0

        return stats

    def estimate_cost(self, num_chunks: int, avg_tokens_per_chunk: int = 200) -> dict:
        """Estimate the cost of embedding generation.

        Args:
            num_chunks: Number of chunks to process
            avg_tokens_per_chunk: Average tokens per chunk

        Returns:
            Dictionary with cost estimates
        """
        # OpenAI embedding model pricing per 1K tokens
        prices = {
            "text-embedding-3-small": 0.00002,
            "text-embedding-3-large": 0.00013,
            "text-embedding-ada-002": 0.00010,
        }
        cost_per_1k_tokens = prices.get(self.model, 0.00002)  # Default to small pricing

        total_tokens = num_chunks * avg_tokens_per_chunk
        total_cost = (total_tokens / 1000) * cost_per_1k_tokens

        return {
            "num_chunks": num_chunks,
            "estimated_tokens": total_tokens,
            "estimated_cost_usd": total_cost,
            "cost_per_chunk": total_cost / num_chunks if num_chunks > 0 else 0,
            "model": self.model,
            "cost_per_1k_tokens": cost_per_1k_tokens,
        }


def validate_embedding_dimensions(
    embeddings: List[List[float]], model: str = "text-embedding-3-small"
) -> bool:
    """Validate that embeddings have the expected dimensions.

    Args:
        embeddings: List of embedding vectors
        model: Model name to check expected dimensions

    Returns:
        True if all embeddings have correct dimensions
    """
    if not embeddings:
        return True

    # Expected dimensions for different models
    expected_dims = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    expected_dim = expected_dims.get(model, 1536)

    for i, embedding in enumerate(embeddings):
        if embedding and len(embedding) != expected_dim:
            print(
                f"❌ Embedding {i} has {len(embedding)} dimensions, expected {expected_dim}"
            )
            return False

    return True
