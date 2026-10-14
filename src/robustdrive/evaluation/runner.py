from pathlib import Path

import numpy as np
import pandas as pd
import torch

from robustdrive.data.dataset import DrivingDataset, load_log
from robustdrive.evaluation.metrics import regression_metrics
from robustdrive.evaluation.plots import prediction_plots
from robustdrive.models.uncertainty import mc_predict
from robustdrive.training.trainer import checkpoint_digest, load_model
from robustdrive.utils.logging import result_status, write_json
from robustdrive.utils.seed import choose_device, seed_everything


class EvaluationRun:
    """Load saved preprocessing/splits; refuse silently changed datasets."""

    def __init__(
        self, checkpoint: str | Path, data: str | None = None, image_root: str | None = None
    ):
        self.path = Path(checkpoint)
        self.model, self.saved = load_model(checkpoint)
        self.config = self.saved["config"]
        c = self.config
        seed_everything(c["experiment"]["seed"], c["training"]["cpu_threads"])
        self.device = choose_device(c["training"]["device"])
        self.model.to(self.device)
        d = c["data"]
        log = load_log(
            data or d["csv_path"], image_root or d["image_root"], d["kind"], d["invalid_policy"]
        )
        if log.fingerprint != self.saved["data_fingerprint"]:
            raise ValueError(
                "Dataset differs from checkpoint. Original held-out membership cannot be verified."
            )
        self.datasets = {
            s: DrivingDataset(
                log.records, self.saved["splits"][s], c["preprocessing"], c["augmentation"], False
            )
            for s in ("validation", "test")
        }
        self.output = self.path.parent
        self.figures = Path("reports/figures") / self.output.name
        self.status = result_status(log.kind)
        self.provenance = {
            "status": self.status,
            "data_kind": log.kind,
            "data_fingerprint": log.fingerprint,
            "checkpoint_sha256": checkpoint_digest(checkpoint),
        }

    @torch.inference_mode()
    def predict(
        self, split: str = "test", corruption: str = "clean", severity: int = 0, passes: int = 0
    ) -> pd.DataFrame:
        from robustdrive.evaluation.robustness import corrupt, corruption_seed

        dataset = self.datasets[split]
        batch_size = self.config["training"]["batch_size"]
        result = []
        # Common dropout random numbers across conditions reduce Monte Carlo comparison noise.
        seed_everything(
            self.config["experiment"]["seed"] + 1000, self.config["training"]["cpu_threads"]
        )
        self.model.eval()
        for start in range(0, len(dataset), batch_size):
            images, rows = [], []
            for i in range(start, min(start + batch_size, len(dataset))):
                image, target, info = dataset.raw(i)
                noise_seed = corruption_seed(
                    self.config["experiment"]["seed"], info["row_id"], corruption, severity
                )
                if severity:
                    image = corrupt(image, corruption, severity, noise_seed)
                images.append(dataset.preprocess(image))
                rows.append(
                    {
                        **info,
                        "target": target,
                        "corruption": corruption,
                        "severity": severity,
                        "corruption_seed": noise_seed,
                        "split": split,
                        "status": self.status,
                    }
                )
            tensor = torch.stack(images).to(self.device)
            if passes:
                predicted = mc_predict(self.model, tensor, passes)
                means = predicted["mean"].cpu().numpy()
                stds = predicted["std"].cpu().numpy()
                variances = predicted["variance"].cpu().numpy()
            else:
                means = self.model(tensor).cpu().numpy()
                stds = variances = np.full(len(rows), np.nan)
            for row, mean, std, variance in zip(rows, means, stds, variances):
                result.append(
                    {
                        **row,
                        "prediction": float(mean),
                        "uncertainty": float(std),
                        "variance": float(variance),
                        "absolute_error": abs(float(mean) - row["target"]),
                    }
                )
        return pd.DataFrame(result)


def evaluate(run: EvaluationRun) -> dict:
    frame = run.predict()
    metrics = {
        **run.provenance,
        "split": "test",
        **regression_metrics(frame.target, frame.prediction),
    }
    frame.to_csv(run.output / "predictions.csv", index=False)
    write_json(run.output / "evaluation.json", metrics)
    prediction_plots(frame, run.figures, run.status)
    return metrics
