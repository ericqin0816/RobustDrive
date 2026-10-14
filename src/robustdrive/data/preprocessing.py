from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def read_rgb(path: str | Path) -> np.ndarray:
    """Pillow handles Unicode paths; every caller receives uint8 RGB."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


@dataclass(frozen=True)
class Preprocessor:
    """The sole deterministic preprocessing path for training, tests, and driving."""

    target_height: int = 66
    target_width: int = 200
    crop_top: int = 50
    crop_bottom: int = 20

    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected H x W x 3 uint8 RGB image.")
        if self.crop_top < 0 or self.crop_bottom < 0:
            raise ValueError("Crop cannot be negative.")
        end = rgb.shape[0] - self.crop_bottom
        if end <= self.crop_top:
            raise ValueError(f"Crop removes entire {rgb.shape[0]}-pixel image.")
        cropped = rgb[self.crop_top:end]
        resized = cv2.resize(cropped, (self.target_width, self.target_height),
                             interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1)))
        return tensor.float().div(127.5).sub(1.0)
