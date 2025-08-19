"""Anthropic client adapter implementing the LLMClient port."""

import json
from typing import Any, Dict, List, Optional

import anthropic

from ...ports.llm_client import LLMClient


class AnthropicClient(LLMClient):
    """Anthropic implementation of the LLMClient port."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize Anthropic client adapter.

        Args:
            api_key: Anthropic API key (required)
            timeout: Request timeout in seconds
        """
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Pass settings.anthropic_api_key to AnthropicClient."
            )

        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict] = None,
    ) -> Dict:
        """Generate a response from the language model.

        Args:
            messages: List of message dictionaries with role and content
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            response_format: Response format specification (ignored for Anthropic)

        Returns:
            Dictionary with response and metadata
        """
        try:
            # Prepare arguments for the Anthropic API
            # Map system role to top-level 'system' as expected by Anthropic
            system_parts = [
                m.get("content", "") for m in messages if m.get("role") == "system"
            ]
            system_text = "\n\n".join([p for p in system_parts if p]) or None
            non_system_messages = [
                m for m in messages if m.get("role") in ("user", "assistant")
            ]

            kwargs = {
                "model": model,
                "messages": non_system_messages,
            }
            if system_text:
                kwargs["system"] = system_text

            # max_tokens is required for Anthropic
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            else:
                # Set a reasonable default if not provided
                kwargs["max_tokens"] = 1024

            if temperature is not None:
                kwargs["temperature"] = temperature

            # Note: response_format is not supported by Anthropic Messages API directly
            # We ignore it here; structured needs should use generate_structured_response

            # Make the API call
            response = self.client.messages.create(**kwargs)

            # Extract response content
            content = ""
            if response.content and len(response.content) > 0:
                # Anthropic response.content is a list of content blocks
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

            # Return structured response matching OpenAI format
            return {
                "content": content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": (
                        response.usage.input_tokens if response.usage else 0
                    ),
                    "completion_tokens": (
                        response.usage.output_tokens if response.usage else 0
                    ),
                    "total_tokens": (
                        (response.usage.input_tokens + response.usage.output_tokens)
                        if response.usage
                        else 0
                    ),
                },
                "finish_reason": response.stop_reason,
            }

        except Exception as e:
            return {
                "content": "",
                "error": str(e),
                "model": model,
            }

    def generate_structured_response(
        self,
        input_messages: List[Dict[str, str]],
        model: str,
        text_format: Dict[str, Any],
        max_output_tokens: Optional[int] = 900,
    ) -> Any:
        """Generate a structured response for compatibility.

        Note: Anthropic doesn't have a direct equivalent to OpenAI's responses.create API,
        so we fall back to regular message generation with structured prompts.

        Args:
            input_messages: List of input message dictionaries
            model: Model name to use
            text_format: Text format specification (used to enhance prompt)
            max_output_tokens: Maximum tokens to generate

        Returns:
            Mock response object for compatibility with existing code
        """
        try:
            # Convert text_format to a prompt instruction, embed schema when present
            format_instruction = self._format_to_instruction(text_format)

            # Enhance the last user message with format instructions
            enhanced_messages = list(input_messages)
            if enhanced_messages:
                last = enhanced_messages[-1]
                enhanced_messages[-1] = {
                    "role": last.get("role", "user"),
                    "content": f"{last.get('content','')}\n\n{format_instruction}",
                }

            # Call Anthropic with mapped roles and system prompt handling
            sys_parts = [
                m.get("content", "")
                for m in enhanced_messages
                if m.get("role") == "system"
            ]
            system_text = "\n\n".join([p for p in sys_parts if p]) or None
            non_system_messages = [
                m for m in enhanced_messages if m.get("role") in ("user", "assistant")
            ]

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": non_system_messages,
                "max_tokens": max_output_tokens,
            }
            if system_text:
                kwargs["system"] = system_text

            resp = self.client.messages.create(**kwargs)

            # Extract concatenated text
            out_text = ""
            if resp.content:
                for block in resp.content:
                    if hasattr(block, "text"):
                        out_text += block.text

            # Build a Responses-like mock that AnswerService can parse
            class OutputText:
                def __init__(self, text: str):
                    self.type = "output_text"
                    self.text = text

            class Message:
                def __init__(self, content_text: str):
                    self.type = "message"
                    self.content = [OutputText(content_text)]

            class MockStructuredResponse:
                def __init__(self, text: str, model_name: str):
                    self.status = "completed"
                    self.output = [Message(text)]
                    self.model = model_name
                    self.incomplete_details = None

            return MockStructuredResponse(out_text, model)

        except Exception as e:
            # Return a mock response object with error information
            class ErrorResponse:
                def __init__(self, error_msg: str):
                    self.status = "error"
                    self.error = error_msg
                    self.output = []

            return ErrorResponse(str(e))

    def _format_to_instruction(self, text_format: Dict[str, Any]) -> str:
        """Convert text format specification to instruction text."""
        if not text_format:
            return "Please provide a clear and structured response."
        # OpenAI Responses-style format
        fmt = text_format.get("format") if isinstance(text_format, dict) else None
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            schema = fmt.get("schema", {})
            schema_str = json.dumps(schema, ensure_ascii=False)
            return (
                "Return ONLY a single JSON object that strictly conforms to this schema:\n"
                f"{schema_str}\n"
                "Do not include any extra text, code fences, or explanations."
            )
        # Fallbacks
        if text_format.get("type") == "json":
            return "Return ONLY valid JSON with no extra text."
        if "schema" in text_format:
            return "Return ONLY a JSON object matching the provided schema."
        return "Please provide a well-structured, concise response."
