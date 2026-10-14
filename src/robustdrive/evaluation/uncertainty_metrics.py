import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from robustdrive.evaluation.metrics import regression_metrics


def detector_metrics(
    target, prediction, uncertainty, error_threshold: float, uncertainty_threshold: float
) -> dict:
    y, p, u = (np.asarray(x, dtype=float) for x in (target, prediction, uncertainty))
    regression_metrics(y, p)
    if u.shape != y.shape or not np.isfinite(u).all() or np.any(u < 0):
        raise ValueError("Uncertainty must be finite, nonnegative and match targets.")
    if error_threshold <= 0 or uncertainty_threshold < 0:
        raise ValueError("Detector thresholds must be positive/nonnegative.")
    errors = abs(y - p)
    dangerous, flagged = errors > error_threshold, u > uncertainty_threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        dangerous, flagged, average="binary", zero_division=0
    )
    two_classes = len(np.unique(dangerous)) == 2
    defined_correlation = len(y) > 1 and np.ptp(errors) > 1e-12 and np.ptp(u) > 1e-12
    if two_classes:
        pr_precision, pr_recall, _ = precision_recall_curve(dangerous, u)
        pr_auc = float(auc(pr_recall, pr_precision))
    else:
        pr_auc = None
    return {
        "n": len(y),
        "dangerous_count": int(dangerous.sum()),
        "flagged_count": int(flagged.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(dangerous, u)) if two_classes else None,
        "pr_auc": pr_auc,
        "average_precision": float(average_precision_score(dangerous, u)) if two_classes else None,
        "pearson": float(pearsonr(u, errors).statistic) if defined_correlation else None,
        "spearman": float(spearmanr(u, errors).statistic) if defined_correlation else None,
        "error_threshold": error_threshold,
        "uncertainty_threshold": uncertainty_threshold,
        "auc_note": None if two_classes else "Undefined: held-out labels contain only one class.",
        "zero_division_policy": "precision/recall/F1 use 0 for a zero denominator; inspect counts.",
    }


def selective_prediction(
    target, prediction, uncertainty, coverages=(1.0, 0.95, 0.90, 0.80, 0.70, 0.50)
) -> list[dict]:
    y, p, u = (np.asarray(x, dtype=float) for x in (target, prediction, uncertainty))
    regression_metrics(y, p)
    if u.shape != y.shape or not np.isfinite(u).all():
        raise ValueError("Invalid uncertainty array.")
    # Stable ties: row order is fixed and does not inspect the held-out labels.
    order = np.argsort(u, kind="stable")
    rows = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("Coverage must be in (0, 1].")
        count = max(1, int(np.floor(len(y) * coverage)))
        selected = order[:count]
        rows.append(
            {
                "requested_coverage": coverage,
                "actual_coverage": count / len(y),
                "accepted": count,
                **regression_metrics(y[selected], p[selected]),
            }
        )
    return rows
