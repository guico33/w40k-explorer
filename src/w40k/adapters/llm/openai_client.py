"""OpenAI client adapter implementing the LLMClient port."""

from typing import Any, Dict, List, Optional
from openai import OpenAI
from ...ports.llm_client import LLMClient


class OpenAIClient(LLMClient):
    """OpenAI implementation of the LLMClient port."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize OpenAI client adapter.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            timeout: Request timeout in seconds
        """
        if not api_key:
            raise ValueError("OpenAI API key is required. Pass settings.openai_api_key to OpenAIClient.")
        
        self.client = OpenAI(api_key=api_key, timeout=timeout)
    
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
            response_format: Response format specification
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Prepare arguments for the OpenAI API
            kwargs = {
                "model": model,
                "messages": messages,
            }
            
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
                
            if temperature is not None:
                kwargs["temperature"] = temperature
                
            if response_format is not None:
                kwargs["response_format"] = response_format
            
            # Make the API call
            response = self.client.chat.completions.create(**kwargs)
            
            # Extract response content
            content = ""
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    content = message.content
            
            # Return structured response
            return {
                "content": content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
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
        max_output_tokens: Optional[int] = None,
    ) -> Any:
        """Generate a structured response using OpenAI's responses.create API.
        
        Args:
            input_messages: List of input message dictionaries
            model: Model name to use
            text_format: Text format specification for structured output
            max_output_tokens: Maximum tokens to generate
            
        Returns:
            Raw OpenAI response object
        """
        try:
            # Prepare arguments for the OpenAI responses API
            kwargs = {
                "model": model,
                "input": input_messages,
                "text": text_format,
            }
            
            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens
            
            # Make the structured API call
            response = self.client.responses.create(**kwargs)
            
            return response
            
        except Exception as e:
            # Return a mock response object with error information
            class ErrorResponse:
                def __init__(self, error_msg: str):
                    self.status = "error"
                    self.error = error_msg
                    self.output = []
            
            return ErrorResponse(str(e))
