import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from robustdrive.config import load_config
from robustdrive.data.dataset import load_log
from robustdrive.evaluation.plots import save_figure
from robustdrive.evaluation.runner import EvaluationRun, evaluate
from robustdrive.training.trainer import checkpoint_digest, train


def ablation_configs(smoke: bool = False) -> list[dict]:
    base = load_config("configs/baseline.yaml")
    if smoke:
        base["data"].update(csv_path="data/sample/driving_log.csv", kind="synthetic", gap_frames=3)
        base["training"].update(epochs=2, batch_size=16, device="cpu")
    variants = []
    for name in ("a_baseline", "b_flip", "c_recovery", "d_visual", "e_dropout"):
        c = deepcopy(base)
        c["experiment"]["name"] = ("smoke_ablation_" if smoke else "ablation_") + name
        variants.append(c)
        if name == "a_baseline":
            base["augmentation"].update(enabled=True, flip=True)
        elif name == "b_flip":
            base["augmentation"]["side_cameras"] = True
        elif name == "c_recovery":
            for key in ("brightness", "contrast", "shadows", "translation", "noise", "blur"):
                base["augmentation"][key] = True
        elif name == "d_visual":
            base["model"]["dropout"] = 0.25
    return variants


def run_ablations(smoke: bool = False) -> None:
    for config in ablation_configs(smoke):
        output = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
        checkpoint = output / "best.pt"
        if checkpoint.exists() and (output / "metrics.json").exists():
            previous = load_config(output / "config.yaml")
            if (
                previous != config
                or not json.loads((output / "metrics.json").read_text())["complete"]
            ):
                raise ValueError(
                    f"Existing incomplete/different run at {output}; resume it explicitly."
                )
            print(f"Reusing completed {output.name}")
        else:
            train(config)
        run = EvaluationRun(checkpoint)
        # Loading EvaluationRun verifies data fingerprint even when reusing training.
        if (
            not (output / "evaluation.json").exists()
            or json.loads((output / "evaluation.json").read_text())["checkpoint_sha256"]
            != run.provenance["checkpoint_sha256"]
        ):
            evaluate(run)


def aggregate(
    root: str | Path = "experiments/results",
    output: str | Path = "reports/tables",
    kind: str = "real",
) -> pd.DataFrame:
    rows = []
    for folder in sorted(Path(root).glob("*")):
        if not (folder / "metadata.json").is_file() or not (folder / "best.pt").is_file():
            continue
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if meta["data_kind"] != kind:
            continue
        training_metrics = folder / "metrics.json"
        if not training_metrics.exists() or not json.loads(training_metrics.read_text()).get(
            "complete"
        ):
            continue
        c = load_config(folder / "config.yaml")
        d = c["data"]
        log = load_log(d["csv_path"], d["image_root"], d["kind"], d["invalid_policy"])
        if log.fingerprint != meta["data_fingerprint"]:
            raise ValueError(f"Dataset changed for {folder.name}; aggregation would be stale.")
        digest = checkpoint_digest(folder / "best.pt")
        row = {
            "Experiment": folder.name,
            "Status": meta["status"],
            "Training samples": meta["training_samples"],
            "Parameters": meta["parameters"],
            "Recovery Data": c["augmentation"]["side_cameras"],
            "Flip": c["augmentation"]["flip"],
            "Visual Augmentation": c["augmentation"]["brightness"],
            "Dropout": c["model"]["dropout"],
        }
        for file, mapping in (
            (
                "metrics.json",
                {
                    "training_seconds": "Training seconds",
                    "best_validation_mse": "Best validation MSE",
                },
            ),
            ("evaluation.json", {"mae": "MAE", "rmse": "RMSE", "mse": "MSE"}),
            ("robustness.json", {"mean_corrupted_mae": "Mean corrupted MAE"}),
        ):
            if (folder / file).is_file():
                value = json.loads((folder / file).read_text(encoding="utf-8"))
                if "checkpoint_sha256" in value and value["checkpoint_sha256"] != digest:
                    continue
                row.update({title: value.get(key) for key, title in mapping.items()})
        if (folder / "uncertainty.json").is_file():
            value = json.loads((folder / "uncertainty.json").read_text(encoding="utf-8"))
            if value["checkpoint_sha256"] == digest:
                row.update(
                    {
                        "Uncertainty/Error Pearson": value["clean"]["pearson"],
                        "Uncertainty/Error Spearman": value["clean"]["spearman"],
                    }
                )
        rows.append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = "ablation" if kind == "real" else "synthetic_ablation"
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["Experiment", "Status", "MAE", "RMSE", "MSE"])
    frame.to_csv(output / f"{prefix}.csv", index=False)
    lines = [
        "# Ablation measurements",
        "",
        "PENDING REAL DATA"
        if kind == "real" and not rows
        else (
            "SYNTHETIC SMOKE ONLY — NOT RESEARCH RESULTS"
            if kind == "synthetic"
            else "Recorded offline measurements"
        ),
        "",
    ]
    lines += [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for _, row in frame.fillna("").iterrows():
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    (output / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if "MAE" in frame and frame.MAE.notna().any():
        valid = frame[frame.MAE.notna()]
        fig, ax = plt.subplots(figsize=(8, max(3, len(valid) * 0.5)))
        ax.barh(valid.Experiment, valid.MAE, color="#2d6978")
        ax.set(xlabel="Held-out steering MAE", title="Recorded experiment comparison")
        save_figure(fig, Path("reports/figures") / f"{prefix}.png", lines[2])
    return frame
