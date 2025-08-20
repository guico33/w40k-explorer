"""LLM-based judges used in evals.

Judges use the existing LLM client; EVAL_MODEL controls the model used.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

from ..ports.llm_client import LLMClient


def _extract_output_text(response: Any) -> Optional[str]:
    """Extract plain `output_text` from a Responses‑like object.

    Args:
        response: Adapter response object with `.output` message list, where
            each message content may be an `output_text` payload.

    Returns:
        The extracted string if present; otherwise None.
    """
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            cont = item.content[0] if isinstance(item.content, list) else item.content
            if getattr(cont, "type", None) == "output_text":
                return getattr(cont, "text", None)
    return None


def judge_groundedness(
    llm: LLMClient,
    model: str,
    question: str,
    answer: str,
    context_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Judge whether the answer is grounded in the provided context.

    Uses the LLM in `model` to evaluate grounding with a strict JSON output.

    Args:
        llm: LLM client implementing the generation interface.
        model: Model name to use for judging (from EVAL_MODEL or override).
        question: The original user question.
        answer: The generated answer text to evaluate.
        context_items: Context passages (title, url, text) used to support the answer.

    Returns:
        Dict with keys: `grounded` (bool) and `reasons` (string rationale).
    """
    ctx_text = []
    for i, c in enumerate(context_items[:8]):  # limit to 8 for brevity
        title = c.get("article_title", "")
        url = c.get("canonical_url", "")
        text = c.get("text", "")
        ctx_text.append(f"[{i+1}] {title} ({url})\n{text}")
    ctx_blob = "\n\n".join(ctx_text) or "(no context)"

    system = "You are a strict evaluator of factual grounding. Only use the provided context."
    user = (
        "Evaluate whether the answer is grounded in the provided context.\n"
        "Return a strict JSON object with keys: grounded (boolean), reasons (string).\n\n"
        f"Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{ctx_blob}"
    )

    try:
        resp = llm.generate_structured_response(
            input_messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model,
            text_format={
                "format": {
                    "type": "json_schema", 
                    "name": "groundedness",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "grounded": {"type": "boolean"},
                            "reasons": {"type": "string"},
                        },
                        "required": ["grounded", "reasons"],
                        "additionalProperties": False,
                    },
                }
            },
            max_output_tokens=200,
        )

        txt = _extract_output_text(resp)
        import json

        out = {"grounded": False, "reasons": "no-output"}
        if txt:
            try:
                out = json.loads(txt)
            except Exception:
                out = {"grounded": False, "reasons": "invalid-json"}
        return out
    except Exception as e:
        return {"grounded": False, "reasons": f"api-error: {str(e)}"}


def judge_relevance(llm: LLMClient, model: str, question: str, answer: str) -> Dict[str, Any]:
    """Judge whether the answer addresses the question.

    Args:
        llm: LLM client implementing the generation interface.
        model: Model name to use for judging (from EVAL_MODEL or override).
        question: The user question.
        answer: The generated answer text.

    Returns:
        Dict with keys: `relevant` (bool) and `reasons` (string rationale).
    """
    system = "You evaluate whether an answer addresses the question."
    user = (
        "Does the answer address the question directly and helpfully?\n"
        "Return JSON: { relevant: boolean, reasons: string }\n\n"
        f"Question:\n{question}\n\nAnswer:\n{answer}"
    )

    try:
        resp = llm.generate_structured_response(
            input_messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model,
            text_format={
                "format": {
                    "type": "json_schema",
                    "name": "relevance", 
                    "schema": {
                        "type": "object",
                        "properties": {
                            "relevant": {"type": "boolean"},
                            "reasons": {"type": "string"},
                        },
                        "required": ["relevant", "reasons"],
                        "additionalProperties": False,
                    },
                }
            },
            max_output_tokens=200,
        )

        txt = _extract_output_text(resp)
        import json

        out = {"relevant": False, "reasons": "no-output"}
        if txt:
            try:
                out = json.loads(txt)
            except Exception:
                out = {"relevant": False, "reasons": "invalid-json"}
        return out
    except Exception as e:
        return {"relevant": False, "reasons": f"api-error: {str(e)}"}
