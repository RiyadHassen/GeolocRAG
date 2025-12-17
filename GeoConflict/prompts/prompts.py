# prompts/prompt.py
from typing import List, Dict, Any, Optional
from .utils import encode_image_to_base64, get_image_mime_type, is_image_file


class OpenAIPrompt:
    @staticmethod
    def build(
        text: str,
        image_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        content = []

        if image_path is not None:
            if not is_image_file(image_path):
                raise ValueError(f"Not a supported image file: {image_path}")

            mime = get_image_mime_type(image_path)
            image_b64 = encode_image_to_base64(image_path)

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{image_b64}"
                }
            })

        content.append({"type": "text", "text": text})

        return [{"role": "user", "content": content}]


class DashScopePrompt:
    @staticmethod
    def build(
        text: str,
        image_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        content = []

        if image_path is not None:
            if not is_image_file(image_path):
                raise ValueError(f"Not a supported image file: {image_path}")
            content.append({"image": image_path})

        content.append({"text": text})

        return [{"role": "user", "content": content}]