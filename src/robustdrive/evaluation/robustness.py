"""Deterministic corruptions of raw RGB; labels stay fixed as an offline stress test."""

import hashlib

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from robustdrive.evaluation.metrics import regression_metrics
from robustdrive.evaluation.plots import robustness_plots
from robustdrive.utils.logging import write_json

CORRUPTIONS = ("brightness_down", "brightness_up", "contrast", "gaussian_blur",
               "motion_blur", "gaussian_noise", "occlusion", "horizontal_shift",
               "vertical_shift", "shadow", "color_shift")


def corruption_seed(seed: int, row_id: int, corruption: str, severity: int) -> int:
    value = f"{seed}:{row_id}:{corruption}:{severity}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def corrupt(image: np.ndarray, name: str, severity: int, seed: int = 42) -> np.ndarray:
    if severity not in range(6):
        raise ValueError("Severity must be an integer in [0, 5].")
    if name not in (*CORRUPTIONS, "clean"):
        raise ValueError(f"Unknown corruption: {name}")
    if name == "clean" and severity != 0:
        raise ValueError("Clean condition must have severity zero.")
    if severity == 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    x = image.astype(np.float32)
    h, w = image.shape[:2]
    s = severity
    if name == "brightness_down":
        x *= 1 - 0.13 * s
    elif name == "brightness_up":
        x *= 1 + 0.18 * s
    elif name == "contrast":
        x = (x - 127.5) * (1 - 0.15 * s) + 127.5
    elif name == "gaussian_blur":
        x = cv2.GaussianBlur(x, (2 * s + 1, 2 * s + 1), 0.5 * s)
    elif name == "motion_blur":
        size = 2 * s + 3
        kernel = np.zeros((size, size), np.float32)
        kernel[size // 2, :] = 1 / size
        x = cv2.filter2D(x, -1, kernel)
    elif name == "gaussian_noise":
        x += rng.normal(0, 5 * s, x.shape)
    elif name == "occlusion":
        # Fixed center-lower location isolates severity from random placement.
        bh, bw = max(1, int(h * (0.04 + 0.025 * s))), max(1, int(w * 0.07 * s))
        y, left = int(0.67 * h), (w - bw) // 2
        x[y:y + bh, left:left + bw] = 0
    elif name in {"horizontal_shift", "vertical_shift"}:
        dx = round(w * 0.015 * s) if name == "horizontal_shift" else 0
        dy = round(h * 0.012 * s) if name == "vertical_shift" else 0
        x = cv2.warpAffine(x, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                           borderMode=cv2.BORDER_REFLECT_101)
    elif name == "shadow":
        yy, xx = np.mgrid[:h, :w]
        x[xx < (w * 0.55 + (yy - h / 2) * 0.35)] *= 1 - 0.13 * s
    elif name == "color_shift":
        x[..., 0] += 8 * s
        x[..., 2] -= 8 * s
    return np.clip(x, 0, 255).astype(np.uint8)


def benchmark(run, names=CORRUPTIONS, severities=None) -> dict:
    levels = severities if severities is not None else run.config["robustness"]["severity_levels"]
    levels = sorted(set([0, *levels]))
    clean = run.predict()
    clean_metrics = regression_metrics(clean.target, clean.prediction)
    rows, predictions = [], []
    for name in tqdm(names, desc="Robustness conditions"):
        for level in levels:
            frame = clean.copy() if level == 0 else run.predict(corruption=name, severity=level)
            frame["corruption"] = name
            metrics = regression_metrics(frame.target, frame.prediction)
            rows.append({"corruption": name, "severity": level, **metrics,
                         "delta_mae": metrics["mae"] - clean_metrics["mae"],
                         "delta_rmse": metrics["rmse"] - clean_metrics["rmse"],
                         "status": run.status})
            if level:
                predictions.append(frame)
    summary = pd.DataFrame(rows)
    summary.to_csv(run.output / "robustness.csv", index=False)
    pd.concat([clean, *predictions], ignore_index=True).to_csv(run.output / "robustness_predictions.csv", index=False)
    nonclean = summary[summary.severity > 0]
    result = {**run.provenance, "split": "test", "clean": clean_metrics,
              "corruptions": list(names), "severities": levels,
              "mean_corrupted_mae": float(nonclean.mae.mean()) if len(nonclean) else None,
              "mean_delta_mae": float(nonclean.delta_mae.mean()) if len(nonclean) else None,
              "aggregation": "Unweighted mean across selected corruption/severity pairs with severity > 0; lower is better."}
    write_json(run.output / "robustness.json", result)
    robustness_plots(summary, run.figures, run.status)
    return result
