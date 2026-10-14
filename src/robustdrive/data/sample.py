"""Procedural road-like pictures for pipeline tests, never driving evidence."""

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from robustdrive.utils.logging import write_json


def create_sample_data(output: str | Path, count: int = 120, seed: int = 42) -> Path:
    if count < 30:
        raise ValueError("Use at least 30 sample frames.")
    output = Path(output)
    if (output / "driving_log.csv").exists():
        raise FileExistsError(f"Refusing to replace {output}/driving_log.csv; use a new directory.")
    (output / "IMG").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        steering = float(0.45 * np.sin(i / 7) + 0.12 * np.sin(i / 2.7))
        paths = []
        for camera, offset in (("center", 0), ("left", 15), ("right", -15)):
            image = np.zeros((160, 320, 3), dtype=np.uint8)
            image[:65] = (107, 155, 190)
            image[65:] = (74, 118, 62)
            horizon = int(160 + steering * 80 + offset)
            road = np.array([[horizon - 20, 65], [horizon + 20, 65],
                             [310 + offset, 159], [10 + offset, 159]], np.int32)
            cv2.fillPoly(image, [road], (75, 76, 78))
            cv2.line(image, (horizon, 66), (160 + offset, 159), (230, 217, 124), 2)
            image = np.clip(image.astype(float) + rng.normal(0, 2, image.shape), 0, 255).astype(np.uint8)
            cv2.putText(image, "SYNTHETIC", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            name = f"IMG/{camera}_{i:06d}.jpg"
            Image.fromarray(image).save(output / name)
            paths.append(name)
        rows.append([*paths, steering, 0.2, 0.0, 15.0])
    path = output / "driving_log.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["center", "left", "right", "steering", "throttle", "brake", "speed"])
        writer.writerows(rows)
    write_json(output / "dataset_metadata.json", {
        "kind": "synthetic", "purpose": "SMOKE TEST ONLY; PENDING REAL DATA",
        "frames": count, "seed": seed, "generator": "robustdrive.data.sample"})
    return path
