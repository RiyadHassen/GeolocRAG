# prompts/utils.py
import base64
from pathlib import Path
from typing import Union


IMAGE_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def is_image_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in IMAGE_MIME_MAP


def get_image_mime_type(path: Union[str, Path]) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in IMAGE_MIME_MAP:
        raise ValueError(f"Unsupported image type: {suffix}")
    return IMAGE_MIME_MAP[suffix]


def encode_image_to_base64(path: Union[str, Path]) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
