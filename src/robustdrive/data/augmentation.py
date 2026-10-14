"""Small, independently switchable perturbations applied only to training RGB."""

import cv2
import numpy as np


def camera_steering(steering: float, camera: str, correction: float) -> float:
    """Udacity: positive steers right. Left-camera recovery therefore adds correction."""
    if camera not in {"center", "left", "right"}:
        raise ValueError(f"Unknown camera: {camera}")
    offset = {"center": 0, "left": correction, "right": -correction}[camera]
    return float(np.clip(steering + offset, -1, 1))


def horizontal_flip(image: np.ndarray, steering: float) -> tuple[np.ndarray, float]:
    return np.ascontiguousarray(image[:, ::-1]), -steering


class Augmenter:
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, image: np.ndarray, steering: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, float]:
        c = self.config
        if not c["enabled"]:
            return image, steering
        image = image.copy()
        h, w = image.shape[:2]
        if c["flip"] and rng.random() < c["flip_probability"]:
            image, steering = horizontal_flip(image, steering)
        if c["translation"] and rng.random() < 0.5:
            dx = rng.uniform(-c["translation_pixels"], c["translation_pixels"])
            dy = rng.uniform(-2, 2)
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            # Road shifted right in the image approximates a camera displaced left.
            steering += dx * c["translation_correction"]
        if c["rotation"] and rng.random() < 0.3:
            # Tiny camera roll only, not vehicle yaw; disabled in provided configs.
            angle = rng.uniform(-c["rotation_degrees"], c["rotation_degrees"])
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        work = image.astype(np.float32)
        if c["brightness"] and rng.random() < 0.5:
            work *= rng.uniform(0.75, 1.25)
        if c["contrast"] and rng.random() < 0.5:
            work = (work - 127.5) * rng.uniform(0.8, 1.2) + 127.5
        if c["shadows"] and rng.random() < 0.5:
            yy, xx = np.mgrid[:h, :w]
            boundary = rng.uniform(0.2, 0.8) * w + (yy - h / 2) * rng.uniform(-0.5, 0.5)
            work[xx < boundary] *= rng.uniform(0.5, 0.85)
        if c["noise"] and rng.random() < 0.3:
            work += rng.normal(0, rng.uniform(1, 5), work.shape)
        image = np.clip(work, 0, 255).astype(np.uint8)
        if c["blur"] and rng.random() < 0.2:
            image = cv2.GaussianBlur(image, (3, 3), 0.7)
        return image, float(np.clip(steering, -1, 1))
