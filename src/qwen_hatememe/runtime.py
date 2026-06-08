"""Runtime helpers kept tiny so demo scripts stay readable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def join_cli_text(value: str | list[str]) -> str:
    """Normalize argparse text fields that may arrive as one or more tokens."""
    if isinstance(value, list):
        return " ".join(value)
    return value


def resolve_image_reference(image: str | None) -> str | None:
    """Return an absolute local path for files while preserving URLs."""
    if not image:
        return None
    if image.startswith(("http://", "https://", "file://", "oss://")):
        return image
    return str(Path(image).expanduser().resolve())


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, eps)


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def save_json(data: dict[str, Any], path: str) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
