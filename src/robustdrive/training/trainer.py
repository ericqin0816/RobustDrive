import hashlib
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from robustdrive.config import save_config, validate
from robustdrive.data.dataset import DrivingDataset, load_log
from robustdrive.data.splits import temporal_split
from robustdrive.evaluation.metrics import regression_metrics
from robustdrive.evaluation.plots import history_plot
from robustdrive.models.nvidia_cnn import NvidiaCNN
from robustdrive.training.losses import steering_loss
from robustdrive.utils.logging import result_status, utc_now, write_json
from robustdrive.utils.seed import choose_device, seed_everything


def read_checkpoint(path: str | Path, device="cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def load_model(path: str | Path, device="cpu") -> tuple[NvidiaCNN, dict]:
    checkpoint = read_checkpoint(path, device)
    model = NvidiaCNN(checkpoint["config"]["model"]["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def checkpoint_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(config: dict, resume: str | Path | None = None) -> Path:
    config = validate(deepcopy(config))
    t, d = config["training"], config["data"]
    seed = config["experiment"]["seed"]
    seed_everything(seed, t["cpu_threads"])
    device = choose_device(t["device"])
    log = load_log(d["csv_path"], d["image_root"], d["kind"], d["invalid_policy"])
    d["kind"] = log.kind
    splits = temporal_split(
        len(log.records), d["train_split"], d["validation_split"], d["test_split"], d["gap_frames"]
    )
    output = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"{output} already contains a run. Use --resume or --name NEW_NAME.")
    output.mkdir(parents=True, exist_ok=True)
    datasets = {
        key: DrivingDataset(
            log.records,
            splits[key],
            config["preprocessing"],
            config["augmentation"],
            key == "train",
            seed,
        )
        for key in ("train", "validation")
    }
    model = NvidiaCNN(config["model"]["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=t["scheduler_patience"], factor=0.5
    )
    history, start, best, best_epoch, stale, elapsed = [], 0, float("inf"), 0, 0, 0.0
    if resume:
        if Path(resume).resolve() != (output / "latest.pt").resolve():
            raise ValueError("Resume the run's latest.pt in its original output directory.")
        saved = read_checkpoint(resume, device)
        previous = deepcopy(saved["config"])
        previous["training"]["epochs"] = t["epochs"]
        if previous != config or saved["data_fingerprint"] != log.fingerprint:
            raise ValueError("Resume requires identical data/config; only epochs may increase.")
        if not (output / "best.pt").exists():
            raise ValueError("Resume requires the original best.pt alongside latest.pt.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, start = saved["history"], saved["epoch"]
        best, best_epoch, stale, elapsed = (
            saved[k] for k in ("best_validation_mse", "best_epoch", "stale", "training_seconds")
        )
        if start >= t["epochs"]:
            raise ValueError("Increase --epochs above the saved epoch to resume.")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    metadata = {
        "timestamp": utc_now(),
        "seed": seed,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "parameters": model.parameter_count,
        "git_commit": commit,
        "git_dirty": dirty,
        "data_kind": log.kind,
        "status": result_status(log.kind),
        "data_fingerprint": log.fingerprint,
        "training_samples": len(datasets["train"]),
        "validation_samples": len(datasets["validation"]),
        "resume_from": str(resume) if resume else None,
    }
    save_config(config, output / "config.yaml")
    write_json(output / "metrics.json", {"status": metadata["status"], "complete": False})
    write_json(output / "metadata.json", metadata)
    write_json(output / "inspection.json", log.inspection)
    write_json(
        output / "splits.json",
        {
            "indices": splits,
            "data_fingerprint": log.fingerprint,
            "source_rows": [r.row_id for r in log.records],
        },
    )
    loss_fn = steering_loss()
    for epoch in range(start, t["epochs"]):
        began = time.perf_counter()
        seed_everything(seed + epoch, t["cpu_threads"])
        datasets["train"].epoch = epoch
        row = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"]}
        for phase in ("train", "validation"):
            training = phase == "train"
            model.train(training)
            loader = DataLoader(
                datasets[phase],
                batch_size=t["batch_size"],
                shuffle=training,
                num_workers=t["num_workers"],
                generator=torch.Generator().manual_seed(seed + epoch),
            )
            predictions, targets = [], []
            with torch.set_grad_enabled(training):
                for images, labels in tqdm(loader, desc=f"Epoch {epoch + 1} {phase}", leave=False):
                    images, labels = images.to(device), labels.to(device)
                    prediction = model(images)
                    loss = loss_fn(prediction, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            "Nonfinite training loss; check data and learning rate."
                        )
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    predictions.extend(prediction.detach().cpu().tolist())
                    targets.extend(labels.cpu().tolist())
            row.update(
                {f"{phase}_{k}": v for k, v in regression_metrics(targets, predictions).items()}
            )
        validation_loss = row["validation_mse"]
        scheduler.step(validation_loss)
        improved = validation_loss < best
        if improved:
            best, best_epoch, stale = validation_loss, epoch + 1, 0
        else:
            stale += 1
        elapsed += time.perf_counter() - began
        row["training_seconds"] = elapsed
        history.append(row)
        saved = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "config": config,
            "data_fingerprint": log.fingerprint,
            "splits": splits,
            "history": history,
            "best_validation_mse": best,
            "best_epoch": best_epoch,
            "stale": stale,
            "training_seconds": elapsed,
            "metadata": metadata,
        }
        # Write then replace, so interruption cannot leave half a checkpoint.
        for filename in ["best.pt", "latest.pt"] if improved else ["latest.pt"]:
            temp = output / (filename + ".tmp")
            torch.save(saved, temp)
            temp.replace(output / filename)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(
            f"Epoch {epoch + 1}: train MSE={row['train_mse']:.6f}, validation MSE={validation_loss:.6f}"
        )
        if stale >= t["patience"]:
            break
    write_json(
        output / "metrics.json",
        {
            "status": metadata["status"],
            "complete": True,
            "best_validation_mse": best,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "training_seconds": elapsed,
            "training_samples": len(datasets["train"]),
            "parameters": model.parameter_count,
        },
    )
    history_plot(pd.DataFrame(history), Path("reports/figures") / output.name, metadata["status"])
    print(f"{metadata['status']}\nSaved {output / 'best.pt'}")
    return output
