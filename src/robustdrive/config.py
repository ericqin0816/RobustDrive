"""Small, validated YAML configuration with optional relative-file inheritance."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "experiment": {"name": "baseline", "seed": 42, "output_root": "experiments/results"},
    "data": {
        "csv_path": "data/raw/driving_log.csv",
        "image_root": None,
        "kind": "real",
        "train_split": 0.70,
        "validation_split": 0.15,
        "test_split": 0.15,
        "gap_frames": 30,
        "invalid_policy": "skip",
    },
    "preprocessing": {"target_height": 66, "target_width": 200, "crop_top": 50, "crop_bottom": 20},
    "augmentation": {
        "enabled": False,
        "flip": False,
        "flip_probability": 0.5,
        "side_cameras": False,
        "steering_correction": 0.2,
        "brightness": False,
        "contrast": False,
        "shadows": False,
        "translation": False,
        "translation_pixels": 12,
        "translation_correction": 0.002,
        "rotation": False,
        "rotation_degrees": 2.0,
        "noise": False,
        "blur": False,
    },
    "model": {"architecture": "nvidia_cnn", "dropout": 0.0},
    "training": {
        "epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "patience": 6,
        "num_workers": 0,
        "device": "auto",
        "cpu_threads": 4,
        "scheduler_patience": 2,
    },
    "uncertainty": {"mc_samples": 20, "error_threshold": 0.15, "validation_quantile": 0.90},
    "robustness": {"severity_levels": [0, 1, 2, 3, 4, 5]},
}


def merge(base: dict, updates: dict) -> dict:
    result = deepcopy(base)
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(value, dict) and isinstance(base[key], dict):
            result[key] = merge(base[key], value)
        else:
            result[key] = value
    return result


def validate(config: dict) -> dict:
    d, p, t, a = (config[k] for k in ("data", "preprocessing", "training", "augmentation"))
    fractions = [d[k] for k in ("train_split", "validation_split", "test_split")]
    if any(x <= 0 for x in fractions) or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("Split fractions must be positive and sum to one.")
    if d["kind"] not in {"real", "synthetic"}:
        raise ValueError("data.kind must be real or synthetic.")
    if d["invalid_policy"] not in {"skip", "error"} or d["gap_frames"] < 0:
        raise ValueError("Use skip/error and a nonnegative gap_frames.")
    if min(p["crop_top"], p["crop_bottom"]) < 0:
        raise ValueError("Crop sizes cannot be negative.")
    if (p["target_height"], p["target_width"]) != (66, 200):
        raise ValueError("Version 1 NVIDIA CNN expects target_height=66, target_width=200.")
    if config["model"]["architecture"] != "nvidia_cnn":
        raise ValueError("Version 1 supports nvidia_cnn only.")
    if not 0 <= config["model"]["dropout"] < 1:
        raise ValueError("Dropout must be in [0, 1).")
    if min(t[k] for k in ("epochs", "batch_size", "patience", "cpu_threads")) < 1:
        raise ValueError("Epochs, batch size, patience and CPU threads must be positive.")
    if t["learning_rate"] <= 0 or t["weight_decay"] < 0 or t["num_workers"] < 0:
        raise ValueError("Invalid optimizer or worker configuration.")
    if not 0 <= a["flip_probability"] <= 1 or a["steering_correction"] < 0:
        raise ValueError("Invalid flip probability or steering correction.")
    if config["uncertainty"]["mc_samples"] < 2:
        raise ValueError("MC dropout needs at least two samples.")
    if not 0 < config["uncertainty"]["validation_quantile"] < 1:
        raise ValueError("Validation quantile must be in (0, 1).")
    if config["uncertainty"]["error_threshold"] <= 0:
        raise ValueError("Error threshold must be positive.")
    if any(s not in range(6) for s in config["robustness"]["severity_levels"]):
        raise ValueError("Corruption severities must be integers from 0 through 5.")
    return config


def load_config(path: str | Path) -> dict:
    def read(file: Path, seen: set[Path]) -> dict:
        file = file.resolve()
        if file in seen:
            raise ValueError("Cyclic config inheritance.")
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        parent = raw.pop("extends", None)
        base = read(file.parent / parent, seen | {file}) if parent else DEFAULTS
        return merge(base, raw)

    return validate(read(Path(path), set()))


def save_config(config: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
