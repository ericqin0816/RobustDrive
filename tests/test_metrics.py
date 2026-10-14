import numpy as np
import pytest

from robustdrive.evaluation.metrics import regression_metrics
from robustdrive.evaluation.uncertainty_metrics import detector_metrics, selective_prediction


def test_known_regression_values():
    result = regression_metrics([0, 1, 2], [1, 1, 0])
    assert result["mae"] == 1
    assert result["mse"] == pytest.approx(5 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert regression_metrics([1, 1], [1, 2])["r2"] is None
    with pytest.raises(ValueError):
        regression_metrics([1], [float("nan")])


def test_perfect_detector_and_undefined_auc():
    metrics = detector_metrics([0, 0, 0, 0], [0, 0.1, 0.4, 0.6], [0.01, 0.02, 0.3, 0.4], 0.2, 0.1)
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "average_precision"):
        assert metrics[key] == pytest.approx(1)
    degenerate = detector_metrics([0, 0], [0, 0], [0, 0], 0.2, 0.1)
    assert degenerate["roc_auc"] is None
    assert degenerate["pr_auc"] is None
    assert degenerate["pearson"] is None


def test_selective_coverage_and_ties():
    rows = selective_prediction([0] * 4, [0.1, 0.2, 0.4, 0.8], [1, 2, 3, 4], [1, 0.5])
    assert rows[1]["mae"] == pytest.approx(0.15)
    assert rows[1]["actual_coverage"] == 0.5
    ties = selective_prediction([0] * 4, [0.1, 0.2, 0.4, 0.8], [1] * 4, [0.5])
    assert ties[0]["mae"] == pytest.approx(0.15)
