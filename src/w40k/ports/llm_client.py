"""LLM client port interface."""

from typing import Any, Dict, List, Optional, Protocol
from abc import abstractmethod


class LLMClient(Protocol):
    """Interface for language model interactions."""

    @abstractmethod
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
        ...

    @abstractmethod
    def generate_structured_response(
        self,
        input_messages: List[Dict[str, str]],
        model: str,
        text_format: Dict[str, Any],
        max_output_tokens: Optional[int] = None,
    ) -> Any:
        """Generate a structured response using advanced API features.
        
        Args:
            input_messages: List of input message dictionaries
            model: Model name to use
            text_format: Text format specification (e.g., JSON schema)
            max_output_tokens: Maximum tokens to generate
            
        Returns:
            Raw response object from the LLM API
        """
        ...