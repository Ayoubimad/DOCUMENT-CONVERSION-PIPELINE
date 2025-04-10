"""
Language model integration module providing OpenAI and other LLM implementations.
"""

from typing import Optional
import os
from openai import OpenAI


class OpenAIModel:
    """OpenAI language model implementation."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 1500,
    ):
        """Initialize the OpenAI model.

        Args:
            model: The OpenAI model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or "https://api.openai.com/v1",
        )

    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: The prompt to send to the model

        Returns:
            The model's response text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
