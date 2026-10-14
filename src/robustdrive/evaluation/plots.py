"""Compact, labeled figures generated only from recorded measurements."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robustdrive.data.preprocessing import read_rgb

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
    }
)


def save_figure(fig, path: Path, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.5, 0.012, status, ha="center", fontsize=7, color="#9b332c")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def history_plot(history: pd.DataFrame, folder: Path, status: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for column, label in (("train_mse", "Train"), ("validation_mse", "Validation")):
        ax.plot(history.epoch, history[column], marker="o", label=label)
    ax.set(xlabel="Epoch", ylabel="Steering MSE", title="Behavioral-cloning learning curve")
    ax.legend()
    save_figure(fig, folder / "training_loss.png", status)


def steering_plot(values, folder: Path, status: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=41, color="#2d6978")
    ax.set(
        xlabel="Steering (negative = left)", ylabel="Frames", title="Recorded steering distribution"
    )
    save_figure(fig, folder / "steering_distribution.png", status)


def prediction_plots(frame: pd.DataFrame, folder: Path, status: str) -> None:
    y, p = frame.target.to_numpy(), frame.prediction.to_numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, p, s=12, alpha=0.5, color="#2d6978")
    bounds = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(bounds, bounds, "--", color="gray")
    ax.set(
        xlabel="Recorded steering", ylabel="Predicted steering", title="Test steering predictions"
    )
    save_figure(fig, folder / "predicted_vs_actual.png", status)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p - y, bins=min(40, max(5, len(y) // 3)), color="#2d6978")
    ax.set(xlabel="Prediction − target", ylabel="Frames", title="Test residual distribution")
    save_figure(fig, folder / "residuals.png", status)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(abs(y), abs(p - y), s=12, alpha=0.5, color="#2d6978")
    ax.set(
        xlabel="Absolute recorded steering",
        ylabel="Absolute steering error",
        title="Error versus steering magnitude",
    )
    save_figure(fig, folder / "error_vs_steering.png", status)
    example_grid(frame.head(9), folder / "sample_predictions.png", status)


def example_grid(frame: pd.DataFrame, path: Path, status: str) -> None:
    if frame.empty:
        return
    from robustdrive.evaluation.robustness import corrupt

    frame = frame.head(9)
    fig, axes = plt.subplots(
        int(np.ceil(len(frame) / 3)), 3, figsize=(11, 3 * np.ceil(len(frame) / 3)), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, frame.iterrows()):
        image = read_rgb(row.path)
        condition, severity = row.get("corruption", "clean"), int(row.get("severity", 0))
        if severity:
            image = corrupt(image, condition, severity, int(row.corruption_seed))
        ax.imshow(image)
        uncertainty = row.get("uncertainty", np.nan)
        ax.set_title(
            f"target {row.target:+.3f} | pred {row.prediction:+.3f}\n"
            f"error {abs(row.target - row.prediction):.3f} | std {uncertainty:.3f}\n"
            f"{condition}, severity {severity}",
            fontsize=8,
        )
    save_figure(fig, path, status)


def robustness_plots(frame: pd.DataFrame, folder: Path, status: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, group in frame.groupby("corruption", sort=True):
        ax.plot(group.severity, group.mae, marker=".", label=name)
        single, axis = plt.subplots(figsize=(5, 4))
        axis.plot(group.severity, group.mae, marker="o", color="#2d6978")
        axis.set(xlabel="Severity", ylabel="Steering MAE", title=name.replace("_", " "))
        save_figure(single, folder / f"robustness_{name}.png", status)
    ax.set(
        xlabel="Severity (0 = clean)", ylabel="Steering MAE", title="Controlled visual robustness"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save_figure(fig, folder / "robustness_summary.png", status)


def uncertainty_plots(
    frame: pd.DataFrame, selective: pd.DataFrame, folder: Path, status: str, error_threshold: float
) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    error = abs(frame.target - frame.prediction)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(frame.uncertainty, error, s=12, alpha=0.4, color="#2d6978")
    ax.set(
        xlabel="MC steering standard deviation",
        ylabel="Absolute steering error",
        title="Does uncertainty track error?",
    )
    save_figure(fig, folder / "uncertainty_vs_error.png", status)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, part in (
        ("Clean", frame[frame.severity == 0]),
        ("Corrupted", frame[frame.severity > 0]),
    ):
        if len(part):
            ax.hist(part.uncertainty, bins=25, alpha=0.5, label=label, density=True)
    ax.set(xlabel="MC steering standard deviation", ylabel="Density", title="Prediction dispersion")
    ax.legend()
    save_figure(fig, folder / "uncertainty_distributions.png", status)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, part in frame[frame.severity > 0].groupby("corruption"):
        group = part.groupby("severity").uncertainty.mean()
        ax.plot(group.index, group.values, marker="o", label=name)
    ax.set(
        xlabel="Corruption severity",
        ylabel="Mean MC standard deviation",
        title="Uncertainty under shift",
    )
    if ax.lines:
        ax.legend(fontsize=8)
    save_figure(fig, folder / "uncertainty_vs_severity.png", status)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, group in selective.groupby("condition"):
        ax.plot(100 * group.actual_coverage, group.mae, marker="o", label=name)
    ax.set(
        xlabel="Accepted coverage (%)", ylabel="Accepted steering MAE", title="Selective prediction"
    )
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, folder / "coverage_vs_error.png", status)
    # Only plot valid discrimination curves. A single class has no ROC-AUC.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, part in (("clean", frame[frame.severity == 0]), ("pooled", frame)):
        dangerous = (abs(part.target - part.prediction) > error_threshold).to_numpy()
        if len(np.unique(dangerous)) < 2:
            continue
        fpr, tpr, _ = roc_curve(dangerous, part.uncertainty)
        precision, recall, _ = precision_recall_curve(dangerous, part.uncertainty)
        axes[0].plot(fpr, tpr, label=label)
        axes[1].plot(recall, precision, label=label)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="Recall", title="High-error detection: ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="High-error detection: precision–recall")
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        else:
            ax.text(0.1, 0.5, "Undefined: only one label class", transform=ax.transAxes)
    save_figure(fig, folder / "failure_detection_curves.png", status)
