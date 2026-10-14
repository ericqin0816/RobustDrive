import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from robustdrive.evaluation.metrics import regression_metrics
from robustdrive.evaluation.plots import example_grid, uncertainty_plots
from robustdrive.evaluation.uncertainty_metrics import detector_metrics, selective_prediction
from robustdrive.utils.logging import write_json


def evaluate_uncertainty(
    run,
    names=("brightness_down", "gaussian_blur", "gaussian_noise", "shadow"),
    passes: int | None = None,
    error_threshold: float | None = None,
) -> dict:
    c = run.config["uncertainty"]
    passes = c["mc_samples"] if passes is None else passes
    if passes < 2:
        raise ValueError("MC uncertainty needs at least two stochastic passes.")
    error_threshold = error_threshold if error_threshold is not None else c["error_threshold"]
    validation = run.predict(split="validation", passes=passes)
    threshold = float(np.quantile(validation.uncertainty, c["validation_quantile"]))
    validation.to_csv(run.output / "uncertainty_validation.csv", index=False)
    clean = run.predict(passes=passes)
    frames = [clean]
    for name in tqdm(names, desc="MC dropout under shift"):
        for severity in run.config["robustness"]["severity_levels"]:
            if severity:
                frames.append(run.predict(corruption=name, severity=severity, passes=passes))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(run.output / "uncertainty_predictions.csv", index=False)
    groups = [("clean", clean), ("pooled", frame)]
    groups += [
        (f"{name}:{int(severity)}", part)
        for (name, severity), part in frame[frame.severity > 0].groupby(["corruption", "severity"])
    ]
    metrics, selective = [], []
    for name, group in groups:
        detector = detector_metrics(
            group.target, group.prediction, group.uncertainty, error_threshold, threshold
        )
        metrics.append(
            {"condition": name, **regression_metrics(group.target, group.prediction), **detector}
        )
        selective.extend(
            {"condition": name, **row}
            for row in selective_prediction(group.target, group.prediction, group.uncertainty)
        )
    pd.DataFrame(metrics).to_csv(run.output / "uncertainty_by_condition.csv", index=False)
    selection = pd.DataFrame(selective)
    selection.to_csv(run.output / "selective_prediction.csv", index=False)
    result = {
        **run.provenance,
        "split": "test",
        "mc_samples": passes,
        "threshold_source": "Clean validation only; never tuned on test error labels.",
        "validation_quantile": c["validation_quantile"],
        "uncertainty_threshold": threshold,
        "error_threshold": error_threshold,
        "corruptions": list(names),
        "severities": run.config["robustness"]["severity_levels"],
        "clean": metrics[0],
        "pooled": metrics[1],
        "interpretation": "High steering error is an offline proxy, not observed road departure. Pooled rows reuse test images and are not independent.",
    }
    write_json(run.output / "uncertainty.json", result)
    uncertainty_plots(
        frame,
        selection[selection.condition.isin(["clean", "pooled"])],
        run.figures,
        run.status,
        error_threshold,
    )
    high_error = frame.absolute_error > error_threshold
    high_u = frame.uncertainty > threshold

    def examples(part, key):
        # Prefer different source frames to nine corruptions of one timestamp.
        return part.sort_values(key, ascending=False).drop_duplicates("row_id").head(9)

    categories = {
        "highest_error": examples(frame, "absolute_error"),
        "highest_uncertainty": examples(frame, "uncertainty"),
        "high_error_high_uncertainty": examples(frame[high_error & high_u], "absolute_error"),
        "confidently_wrong": examples(frame[high_error & ~high_u], "absolute_error"),
        "low_error_high_uncertainty": examples(frame[~high_error & high_u], "uncertainty"),
    }
    for name, subset in categories.items():
        subset.to_csv(run.output / f"failure_cases_{name}.csv", index=False)
        example_grid(subset, run.figures / f"failure_cases_{name}.png", run.status)
    return result
