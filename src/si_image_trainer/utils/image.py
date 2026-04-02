from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


def open_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def resize_rgb(image: Image.Image, size: int) -> np.ndarray:
    return np.asarray(image.resize((size, size), RESAMPLE_BILINEAR), dtype=np.float32) / 255.0


def resize_gray(image: Image.Image, size: int) -> np.ndarray:
    gray = ImageOps.grayscale(image)
    return np.asarray(gray.resize((size, size), RESAMPLE_BILINEAR), dtype=np.float32) / 255.0


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
