from typing import Any, Dict, List, Optional

import json
import pytest

from w40k.adapters.llm.anthropic_client import AnthropicClient


class _FakeBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, text: str, status: str = "completed"):
        self.status = status
        self.model = "claude-test"
        self.content = [_FakeBlock(text)]
        self.incomplete_details = None
        # Provide optional attributes used by the adapter
        class _Usage:
            def __init__(self):
                self.input_tokens = 10
                self.output_tokens = 5

        self.usage = _Usage()
        self.stop_reason = "end_turn"


class _RecorderMessages:
    def __init__(self, response: _FakeResponse):
        self.last_kwargs: Optional[Dict[str, Any]] = None
        self._response = response

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeAnthropic:
    def __init__(self, api_key: str, timeout: float):
        # Return a fixed response; tests can inspect last_kwargs
        self._recorder = _RecorderMessages(_FakeResponse(text=json.dumps({"ok": True})))
        self.messages = self._recorder


def test_generate_response_maps_system_and_returns_text(monkeypatch):
    # Patch the Anthropic SDK used by the adapter
    monkeypatch.setattr(
        "w40k.adapters.llm.anthropic_client.anthropic.Anthropic",
        _FakeAnthropic,
    )

    client = AnthropicClient(api_key="test", timeout=5.0)

    messages = [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "hello"},
    ]
    out = client.generate_response(messages=messages, model="claude", max_tokens=64)

    # Expect a dict with content field and model present
    assert isinstance(out, dict)
    assert "content" in out
    assert out["model"] == "claude-test"


def test_generate_structured_response_returns_responses_like_object(monkeypatch):
    # Patch to return known text content
    fake_resp = _FakeResponse(text="{\"answer\": \"OK\", \"citations_used\": [], \"confidence\": 0.9}")

    class _AnthropicWithKnown(_FakeAnthropic):
        def __init__(self, api_key: str, timeout: float):
            self._recorder = _RecorderMessages(fake_resp)
            self.messages = self._recorder

    monkeypatch.setattr(
        "w40k.adapters.llm.anthropic_client.anthropic.Anthropic",
        _AnthropicWithKnown,
    )

    client = AnthropicClient(api_key="test", timeout=5.0)

    resp = client.generate_structured_response(
        input_messages=[
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "question"},
        ],
        model="claude",
        text_format={"format": {"type": "json_schema", "schema": {"type": "object"}}},
        max_output_tokens=256,
    )

    # Should look like the OpenAI Responses shape expected by AnswerService
    assert getattr(resp, "status", None) == "completed"
    assert hasattr(resp, "output") and isinstance(resp.output, list) and resp.output
    message = resp.output[0]
    assert getattr(message, "type", None) == "message"
    assert hasattr(message, "content") and isinstance(message.content, list) and message.content
    content0 = message.content[0]
    assert getattr(content0, "type", None) == "output_text"
    assert isinstance(getattr(content0, "text", None), str)
