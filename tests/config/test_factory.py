import os
import pytest
from w40k.config.settings import Settings
from w40k.config.factory import create_llm_client, validate_environment


def test_create_llm_client_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    s = Settings()

    client = create_llm_client(s)
    assert client.__class__.__name__ == "OpenAIClient"
    assert hasattr(client, "generate_response")


def test_create_llm_client_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    # Also set OpenAI key for embeddings requirement later in validate_environment
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    s = Settings()

    client = create_llm_client(s)
    assert client.__class__.__name__ == "AnthropicClient"
    assert hasattr(client, "generate_response")


def test_validate_environment_requires_openai_for_embeddings_with_anthropic(monkeypatch):
    # Using Anthropic for LLM, but embeddings still need OpenAI key
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    ok, err = validate_environment()
    assert ok is False
    assert "OPENAI_API_KEY is required" in (err or "")


def test_validate_environment_openai_ok(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")

    ok, err = validate_environment()
    assert ok is True
    assert err is None

