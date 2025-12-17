from typing import List, Dict, Any
import dashscope
from dashscope import MultiModalConversation


class DashScopeModel:
    """
    Thin wrapper for DashScope multimodal models (e.g. image editing).

    Design principles:
    - Does NOT construct prompts
    - Does NOT infer modality
    - Does NOT hardcode generation parameters
    - Only executes given messages
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config:
                {
                    "model": "qwen-image-edit-plus",
                    "api_key": "...",
                    "base_url": "https://dashscope-intl.aliyuncs.com/api/v1"
                }
        """
        self.model_name = config["model"]
        self.api_key = config["api_key"]

        if "base_url" in config:
            dashscope.base_http_api_url = config["base_url"]

    def run(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Any:
        """
        Args:
            messages:
                DashScope-style multimodal messages.
                Example:
                [
                    {
                        "role": "user",
                        "content": [
                            {"image": "xxx.png"},
                            {"text": "Edit the image ..."}
                        ]
                    }
                ]

            **kwargs:
                Forwarded to DashScope API, e.g.:
                - size
                - watermark
                - prompt_extend
                - negative_prompt
                - n

        Returns:
            Raw DashScope response object.
            (Upper layers decide how to parse / save images.)
        """

        response = MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model_name,
            messages=messages,
            stream=False,
            **kwargs,
        )

        return response
