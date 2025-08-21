"""Centralized configuration settings for W40K Explorer."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables."""

    # LLM Provider Configuration
    llm_provider: str = Field(
        default="openai",
        description="LLM provider to use: 'openai' or 'anthropic'",
        validation_alias="LLM_PROVIDER",
    )

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM and embeddings",
        validation_alias="OPENAI_API_KEY",
    )
    openai_llm_model: str = Field(
        default="gpt-4o-mini-2024-07-18",
        description="OpenAI model for text generation",
        validation_alias="OPENAI_LLM_MODEL",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
        validation_alias="EMBEDDING_MODEL",
    )

    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for LLM",
        validation_alias="ANTHROPIC_API_KEY",
    )
    anthropic_llm_model: str = Field(
        default="claude-3-7-sonnet-latest",
        description="Anthropic model for text generation",
        validation_alias="ANTHROPIC_LLM_MODEL",
    )

    # Qdrant Configuration
    qdrant_url: Optional[str] = Field(
        default=None,
        description="Qdrant cloud URL (if using cloud)",
        validation_alias="QDRANT_URL",
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key (if using cloud)",
        validation_alias="QDRANT_API_KEY",
    )
    qdrant_host: str = Field(
        default="localhost", description="Qdrant host (if using local)"
    )
    qdrant_port: int = Field(default=6333, description="Qdrant port (if using local)")
    qdrant_collection_name: str = Field(
        default="w40k_chunks", description="Qdrant collection name"
    )

    # Vector Service Provider
    vector_provider: str = Field(
        default="qdrant",
        description="Vector service provider: 'qdrant' or 'pinecone'",
        validation_alias="VECTOR_PROVIDER",
    )

    # Pinecone Configuration (optional; used when vector_provider='pinecone')
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API key",
        validation_alias="PINECONE_API_KEY",
    )
    pinecone_index: Optional[str] = Field(
        default=None,
        description="Pinecone index name",
        validation_alias="PINECONE_INDEX",
    )

    # Database Configuration
    db_path: str = Field(
        default="data/articles.db",
        description="Path to SQLite database",
        validation_alias="DB_PATH",
    )

    # Vector Configuration
    vector_size: int = Field(default=1536, description="Embedding vector dimensions")

    # Query Engine Configuration
    initial_k: int = Field(
        default=60, description="Number of initial chunks to retrieve"
    )
    max_context_chunks: int = Field(
        default=12, description="Maximum chunks to include in context"
    )
    min_similarity_score: Optional[float] = Field(
        default=0.2, description="Minimum similarity score threshold"
    )
    max_tokens: int = Field(default=900, description="Maximum tokens for LLM response")
    context_max_words: int = Field(
        default=200, description="Max words per context chunk"
    )
    query_expansion_n: int = Field(
        default=0, description="Number of query expansions to generate"
    )
    lower_threshold_on_empty: bool = Field(
        default=True, description="Relax threshold when no hits found"
    )
    active_only: bool = Field(default=True, description="Only search active chunks")

    # Performance Configuration
    embedding_batch_size: int = Field(
        default=100, description="Batch size for embedding generation"
    )
    max_retries: int = Field(
        default=3, description="Maximum retry attempts for API calls"
    )
    retry_delay: float = Field(default=1.0, description="Initial delay between retries")
    openai_timeout: float = Field(
        default=30.0, description="OpenAI API request timeout in seconds"
    )
    anthropic_timeout: float = Field(
        default=30.0, description="Anthropic API request timeout in seconds"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def is_qdrant_cloud(self) -> bool:
        """Check if using Qdrant cloud configuration."""
        return bool(self.qdrant_url and self.qdrant_api_key)

    def get_db_path(self) -> Path:
        """Get database path as Path object."""
        return Path(self.db_path)

    def db_exists(self) -> bool:
        """Check if the SQLite database file exists."""
        return self.get_db_path().exists()

    def is_anthropic_enabled(self) -> bool:
        """Check if Anthropic provider is enabled and configured."""
        return self.llm_provider == "anthropic" and bool(self.anthropic_api_key)

    def is_openai_enabled(self) -> bool:
        """Check if OpenAI provider is enabled and configured."""
        return self.llm_provider == "openai" and bool(self.openai_api_key)

    def validate_llm_provider(self) -> None:
        """Validate that the selected LLM provider has required configuration."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key is required when llm_provider is 'openai'")
        elif self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "Anthropic API key is required when llm_provider is 'anthropic'"
            )
        elif self.llm_provider not in ["openai", "anthropic"]:
            raise ValueError(
                f"Unsupported llm_provider: {self.llm_provider}. Must be 'openai' or 'anthropic'"
            )

    def get_llm_model(self) -> str:
        """Get the appropriate LLM model for the current provider."""
        if self.llm_provider == "openai":
            return self.openai_llm_model
        elif self.llm_provider == "anthropic":
            return self.anthropic_llm_model
        else:
            raise ValueError(f"Unsupported llm_provider: {self.llm_provider}")

    def get_llm_timeout(self) -> float:
        """Get the appropriate timeout for the current provider."""
        if self.llm_provider == "openai":
            return self.openai_timeout
        elif self.llm_provider == "anthropic":
            return self.anthropic_timeout
        else:
            raise ValueError(f"Unsupported llm_provider: {self.llm_provider}")

    def validate_vector_provider(self) -> None:
        """Validate vector provider configuration when used."""
        if self.vector_provider not in ["qdrant", "pinecone"]:
            raise ValueError(
                f"Unsupported vector_provider: {self.vector_provider}. Must be 'qdrant' or 'pinecone'"
            )
        if self.vector_provider == "pinecone":
            if not (self.pinecone_api_key and self.pinecone_index):
                raise ValueError(
                    "Pinecone configuration requires PINECONE_API_KEY and PINECONE_INDEX"
                )


def get_settings() -> Settings:
    """Get application settings instance.

    Pylance may flag missing constructor args because required fields
    are populated from environment variables at runtime. We ignore the
    type checker here since BaseSettings handles this injection.
    """
    return Settings()  # type: ignore[call-arg]
