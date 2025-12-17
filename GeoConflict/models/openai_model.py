from typing import List, Dict, Any, Optional
from openai import OpenAI


class OpenAIModel:
    """
    Thin wrapper around OpenAI ChatCompletion API.

    Design principles:
    - Does NOT construct prompts
    - Does NOT decide modality (text / image)
    - Does NOT store max_tokens
    - Only executes given messages
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config:
                {
                    "model": "gpt-4.1-mini",
                    "api_key": "..."
                }
        """
        self.model_name = config["model"]
        self.client = OpenAI(api_key=config["api_key"])

    def run(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Args:
            messages:
                OpenAI-style messages.
                Modality (image/text) must already be encoded here.
            max_tokens:
                Max tokens for this call.
            temperature:
                Sampling temperature.
            **kwargs:
                Forwarded to OpenAI API (e.g., top_p, response_format)

        Returns:
            Model response content as string.
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        return response.choices[0].message.content
