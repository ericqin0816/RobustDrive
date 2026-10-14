import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def result_status(kind: str) -> str:
    return ("SYNTHETIC SMOKE ONLY — PENDING REAL DATA" if kind == "synthetic"
            else "REAL DATA — OFFLINE EVALUATION; CLOSED-LOOP CLAIMS REQUIRE DRIVE LOGS")
