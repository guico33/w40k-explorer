from typing import Any, Dict, List, Optional

from w40k.ports.llm_client import LLMClient


class FakeLLMClient(LLMClient):
    """LLM fake that returns a minimal Responses-like object with JSON payload."""

    def __init__(self, mode: str = "ok"):
        # mode: "ok" | "incomplete" | "non_json" | "refusal" | "bad_conf" | "expand"
        self.mode = mode

    # Minimal shapes to satisfy AnswerService parser
    class _OutputText:
        def __init__(self, text: str):
            self.type = "output_text"
            self.text = text

    class _Message:
        def __init__(self, text: str):
            self.type = "message"
            self.content = [FakeLLMClient._OutputText(text)]

    class _IncompleteDetails:
        def __init__(self, reason: str):
            self.reason = reason

    class _Response:
        def __init__(
            self,
            text: str,
            status: str = "completed",
            incomplete_reason: Optional[str] = None,
        ):
            self.status = status
            self.model = "fake-model"
            # Use a broad type to simplify test doubles
            self.output: List[Any] = [FakeLLMClient._Message(text)] if text else []
            self.incomplete_details = (
                FakeLLMClient._IncompleteDetails(incomplete_reason)
                if incomplete_reason
                else None
            )

    def generate_structured_response(
        self,
        input_messages: List[Dict[str, str]],
        model: str,
        text_format: Dict[str, Any],
        max_output_tokens: Optional[int] = None,
    ) -> Any:
        # Handle query expansion schema (array) if requested
        fmt = text_format.get("format") if isinstance(text_format, dict) else None
        if (
            self.mode == "expand"
            and isinstance(fmt, dict)
            and fmt.get("type") == "json_schema"
        ):
            schema = fmt.get("schema", {})
            if isinstance(schema, dict) and schema.get("type") == "array":
                return self._Response(
                    text=_to_json(["alternative phrasing one", "alternate form two"])
                )

        if self.mode == "ok":
            # Valid JSON matching schema; cite first two context items (ids 0 and 1)
            payload = {
                "answer": "Horus was named Warmaster. [0]",
                "citations_used": [0],
                "confidence": 0.85,
            }
            return self._Response(text=_to_json(payload))
        if self.mode == "incomplete":
            # First call returns incomplete; compression retry uses a lower max_output_tokens
            if max_output_tokens is not None and max_output_tokens <= 300:
                payload = {
                    "answer": "Compressed: Horus was named Warmaster. [0]",
                    "citations_used": [0],
                    "confidence": 0.7,
                }
                return self._Response(text=_to_json(payload))
            # Trigger truncation path on initial call
            return self._Response(
                text="", status="incomplete", incomplete_reason="max_output_tokens"
            )
        if self.mode == "non_json":
            return self._Response(text="This is plain text, not JSON.")
        if self.mode == "refusal":
            # Return a content block of type 'refusal'
            class _Refusal:
                def __init__(self, reason: str):
                    self.type = "refusal"
                    self.refusal = reason

            class _MessageRefusal:
                def __init__(self, reason: str):
                    self.type = "message"
                    self.content = [_Refusal(reason)]

            r = self._Response(text="")
            r.output = [_MessageRefusal("policy_violation")]
            return r
        if self.mode == "bad_conf":
            payload = {
                "answer": "Text with bad confidence. [0]",
                "citations_used": [0],
                "confidence": 1.5,  # beyond bounds
            }
            return self._Response(text=_to_json(payload))
        return self._Response(
            text=_to_json({"answer": "", "citations_used": [], "confidence": 0.0})
        )

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict] = None,
    ) -> Dict:
        return {"content": "stub", "model": model}


def _to_json(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
