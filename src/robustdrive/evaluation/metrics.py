import numpy as np
from sklearn.metrics import r2_score


def regression_metrics(target, prediction) -> dict:
    y, p = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("Expected equally sized, nonempty one-dimensional arrays.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require finite predictions and targets.")
    residual = p - y
    mse = float(np.mean(residual ** 2))
    return {"n": len(y), "mse": mse, "mae": float(np.mean(abs(residual))),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 1e-12 else None}
