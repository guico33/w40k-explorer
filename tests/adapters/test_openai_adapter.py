from typing import Any

import json
import pytest

from w40k.adapters.llm.openai_client import OpenAIClient


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop"):
        class _Msg:
            def __init__(self, content: str):
                self.content = content

        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt: int, completion: int):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _FakeChatCompletions:
    def __init__(self, content: str):
        class _Resp:
            def __init__(self, content: str):
                self.model = "gpt-test"
                self.choices = [_FakeChoice(content)]
                self.usage = _FakeUsage(10, 5)

        self._resp = _Resp(content)

    def create(self, **kwargs):  # noqa: D401
        # kwargs contains model, messages, etc. We can assert shape if needed
        return self._resp


class _FakeOutputText:
    def __init__(self, text: str):
        self.type = "output_text"
        self.text = text


class _FakeMessage:
    def __init__(self, text: str):
        self.type = "message"
        self.content = [_FakeOutputText(text)]


class _FakeResponses:
    def __init__(self, json_text: str, status: str = "completed"):
        class _Resp:
            def __init__(self, text: str, status: str):
                self.status = status
                self.model = "gpt-test"
                self.output = [_FakeMessage(text)]
                self.incomplete_details = None

        self._resp = _Resp(json_text, status)

    def create(self, **kwargs):
        return self._resp


class _FakeOpenAI:
    def __init__(self, api_key: str, timeout: float):
        # default behavior; tests will monkeypatch attributes if needed
        self.chat = type("_Chat", (), {"completions": _FakeChatCompletions("ok")})()
        self.responses = _FakeResponses(json.dumps({"answer": "ok", "citations_used": [], "confidence": 0.9}))


def test_generate_response_returns_expected_shape(monkeypatch):
    # Patch OpenAI SDK constructor used in adapter to our fake
    monkeypatch.setattr(
        "w40k.adapters.llm.openai_client.OpenAI",
        _FakeOpenAI,
    )

    client = OpenAIClient(api_key="test", timeout=5.0)
    out = client.generate_response(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt",
        max_tokens=32,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    # Adapter should map to a dict with content, model, usage, finish_reason
    assert isinstance(out, dict)
    assert out["content"] == "ok"
    assert out["model"] == "gpt-test"
    assert out["usage"]["total_tokens"] == 15
    assert out["finish_reason"] == "stop"


def test_generate_structured_response_is_responses_like(monkeypatch):
    monkeypatch.setattr(
        "w40k.adapters.llm.openai_client.OpenAI",
        _FakeOpenAI,
    )

    client = OpenAIClient(api_key="test", timeout=5.0)
    resp = client.generate_structured_response(
        input_messages=[{"role": "user", "content": "q"}],
        model="gpt",
        text_format={"format": {"type": "json_schema", "schema": {"type": "object"}}},
        max_output_tokens=64,
    )

    assert getattr(resp, "status", None) == "completed"
    assert hasattr(resp, "output") and resp.output
    msg = resp.output[0]
    assert getattr(msg, "type", None) == "message"
    content0 = msg.content[0]
    assert getattr(content0, "type", None) == "output_text"
    assert isinstance(getattr(content0, "text", None), str)

