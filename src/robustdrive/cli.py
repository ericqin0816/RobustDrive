"""Shared command implementations; scripts remain thin, readable entry points."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robustdrive.utils.logging import json_safe, result_status, write_json


def data_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path / "driving_log.csv" if path.is_dir() or path.suffix != ".csv" else path)


def print_result(value) -> None:
    print(json.dumps(json_safe(value), indent=2, allow_nan=False))


def sample_main() -> None:
    from robustdrive.data.sample import create_sample_data

    parser = argparse.ArgumentParser(description="Generate SYNTHETIC SMOKE ONLY camera records.")
    parser.add_argument("--output", default="data/sample")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(create_sample_data(args.output, args.frames, args.seed))
    print("SYNTHETIC SMOKE ONLY — PENDING REAL DATA")


def inspect_main() -> None:
    from robustdrive.data.dataset import load_log
    from robustdrive.evaluation.plots import steering_plot

    parser = argparse.ArgumentParser(
        description="Validate driving logs and plot steering distribution."
    )
    parser.add_argument(
        "--data", default="data/raw", help="Directory containing driving_log.csv, or CSV path"
    )
    parser.add_argument("--image-root")
    parser.add_argument("--kind", choices=["real", "synthetic"], default="real")
    parser.add_argument("--output", default="reports/tables/dataset_inspection.json")
    parser.add_argument("--figure-dir", default="reports/figures/dataset")
    args = parser.parse_args()
    log = load_log(data_path(args.data), args.image_root, args.kind)
    steering = np.array([r.steering for r in log.records])
    speed = np.array([r.speed for r in log.records if r.speed is not None])

    def distribution(values):
        return (
            {
                "count": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "quantiles_5_50_95": np.quantile(values, [0.05, 0.5, 0.95]).tolist(),
            }
            if len(values)
            else None
        )

    report = {
        **log.inspection,
        "status": result_status(log.kind),
        "fingerprint": log.fingerprint,
        "steering_distribution": distribution(steering),
        "speed_distribution": distribution(speed),
        "nearly_straight_definition": "abs(steering) < 0.05",
        "nearly_straight_percent": float(100 * np.mean(abs(steering) < 0.05)),
    }
    write_json(args.output, report)
    steering_plot(steering, Path(args.figure_dir), report["status"])
    print_result(report)


def train_main() -> None:
    from robustdrive.config import load_config
    from robustdrive.training.trainer import train

    parser = argparse.ArgumentParser(
        description="Train a steering CNN from scratch or resume latest.pt."
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--data", help="Override CSV location; original data order must be retained"
    )
    parser.add_argument("--image-root")
    parser.add_argument("--name", help="New experiment directory name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", help="Path to original run's latest.pt")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data:
        config["data"]["csv_path"] = data_path(args.data)
    if args.image_root:
        config["data"]["image_root"] = args.image_root
    if args.name:
        config["experiment"]["name"] = args.name
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.device:
        config["training"]["device"] = args.device
    train(config, args.resume)


def evaluation_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", required=True, help="best.pt from a completed experiment")
    parser.add_argument("--data", help="Relocated original CSV; must match saved fingerprint")
    parser.add_argument("--image-root", help="Relocated image directory")
    return parser


def evaluation_run(args):
    from robustdrive.evaluation.runner import EvaluationRun

    return EvaluationRun(args.checkpoint, data_path(args.data), args.image_root)


def evaluate_main() -> None:
    from robustdrive.evaluation.runner import evaluate

    args = evaluation_args("Evaluate the checkpoint on its original clean TEST split.").parse_args()
    print_result(evaluate(evaluation_run(args)))


def robustness_main() -> None:
    from robustdrive.evaluation.robustness import CORRUPTIONS, benchmark

    parser = evaluation_args("Benchmark controlled corruptions of held-out camera images.")
    parser.add_argument("--corruptions", nargs="+", choices=CORRUPTIONS, default=list(CORRUPTIONS))
    parser.add_argument("--severities", nargs="+", type=int, choices=range(6))
    args = parser.parse_args()
    print_result(benchmark(evaluation_run(args), args.corruptions, args.severities))


def uncertainty_main() -> None:
    from robustdrive.evaluation.robustness import CORRUPTIONS
    from robustdrive.evaluation.uncertainty_analysis import evaluate_uncertainty

    parser = evaluation_args("MC dropout; validation-only cutoff; TEST failure/rejection metrics.")
    parser.add_argument("--mc-samples", type=int)
    parser.add_argument(
        "--error-threshold",
        type=float,
        help="Predeclare a high-error definition; do not tune on test",
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=CORRUPTIONS,
        default=["brightness_down", "gaussian_blur", "gaussian_noise", "shadow"],
    )
    args = parser.parse_args()
    print_result(
        evaluate_uncertainty(
            evaluation_run(args), args.corruptions, args.mc_samples, args.error_threshold
        )
    )


def ablation_main() -> None:
    from robustdrive.evaluation.ablation import aggregate, run_ablations

    parser = argparse.ArgumentParser(
        description="Aggregate existing runs; optionally train five controlled ablations."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train missing ablations; reuse verified completed runs",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Use synthetic sample data and two epochs"
    )
    parser.add_argument("--kind", choices=["real", "synthetic"], default="real")
    parser.add_argument("--root", default="experiments/results")
    parser.add_argument("--output", default="reports/tables")
    args = parser.parse_args()
    if args.train:
        run_ablations(args.smoke)
    frame = aggregate(args.root, args.output, "synthetic" if args.smoke else args.kind)
    print(frame.to_string(index=False) if len(frame) else "PENDING REAL DATA")


def figures_main() -> None:
    from robustdrive.evaluation.ablation import aggregate
    from robustdrive.evaluation.plots import (
        history_plot,
        prediction_plots,
        robustness_plots,
        uncertainty_plots,
    )

    parser = argparse.ArgumentParser(
        description="Regenerate figures from saved measurements; never invent results."
    )
    parser.add_argument("--run", help="Experiment directory; omitted means all completed runs")
    args = parser.parse_args()
    folders = [Path(args.run)] if args.run else sorted(Path("experiments/results").glob("*"))
    for folder in folders:
        if not (folder / "metadata.json").is_file():
            continue
        # A resumed run may have a new best checkpoint but old evaluation files.
        from robustdrive.evaluation.runner import EvaluationRun

        verified = EvaluationRun(folder / "best.pt")
        for filename in ("evaluation.json", "robustness.json", "uncertainty.json"):
            if (folder / filename).exists():
                saved = json.loads((folder / filename).read_text(encoding="utf-8"))
                if saved["checkpoint_sha256"] != verified.provenance["checkpoint_sha256"]:
                    raise ValueError(f"Stale {folder / filename}; rerun that evaluation first.")
        status = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["status"]
        figures = Path("reports/figures") / folder.name
        if (folder / "history.csv").exists():
            history_plot(pd.read_csv(folder / "history.csv"), figures, status)
        if (folder / "predictions.csv").exists():
            prediction_plots(pd.read_csv(folder / "predictions.csv"), figures, status)
        if (folder / "robustness.csv").exists():
            robustness_plots(pd.read_csv(folder / "robustness.csv"), figures, status)
        if (folder / "uncertainty.json").exists():
            info = json.loads((folder / "uncertainty.json").read_text(encoding="utf-8"))
            selection = pd.read_csv(folder / "selective_prediction.csv")
            uncertainty_plots(
                pd.read_csv(folder / "uncertainty_predictions.csv"),
                selection[selection.condition.isin(["clean", "pooled"])],
                figures,
                status,
                info["error_threshold"],
            )
        print(f"Figures: {figures} ({status})")
    aggregate(kind="real")
    aggregate(kind="synthetic")


def drive_main() -> None:
    import asyncio

    from aiohttp import web

    from robustdrive.simulator.controller import SpeedController
    from robustdrive.simulator.interface import DriveSession, NeuralSteering
    from robustdrive.simulator.udacity import create_app, run_until_stopped

    parser = argparse.ArgumentParser(
        description="Connect trained steering to Udacity on localhost:4567."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New directory for this drive run")
    parser.add_argument("--target-speed", type=float, default=12.0, help="mph; begin slowly")
    parser.add_argument("--kp", type=float, default=0.08)
    parser.add_argument("--max-throttle", type=float, default=0.3)
    parser.add_argument("--port", type=int, default=4567)
    parser.add_argument(
        "--mc-samples", type=int, default=0, help="0 deterministic, >=2 for dropout checkpoint"
    )
    parser.add_argument(
        "--uncertainty-report",
        help="uncertainty.json fitted using this checkpoint's validation data",
    )
    parser.add_argument("--record-frames", action="store_true")
    parser.add_argument("--record-fps", type=float, default=5.0, help="Maximum saved camera fps")
    parser.add_argument("--max-recorded-frames", type=int, default=10000, help="Saved JPG limit")
    parser.add_argument("--stop-file", type=Path, help="Exit cleanly when this local file appears")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Protocol tests ONLY; cannot substantiate driving claims",
    )
    args = parser.parse_args()
    policy = NeuralSteering(args.checkpoint, args.mc_samples)
    if policy.provenance["training_data_kind"] == "synthetic" and not args.allow_synthetic:
        parser.error(
            "Synthetic checkpoint is for software tests; collect real data before driving."
        )
    threshold = None
    if args.uncertainty_report:
        report = json.loads(Path(args.uncertainty_report).read_text(encoding="utf-8"))
        if (
            report["checkpoint_sha256"] != policy.provenance["checkpoint_sha256"]
            or report["mc_samples"] != args.mc_samples
        ):
            parser.error("Uncertainty report must match this checkpoint and MC pass count.")
        threshold = report["uncertainty_threshold"]
    session = DriveSession(
        policy,
        args.output,
        SpeedController(args.target_speed, args.kp, args.max_throttle),
        {**policy.provenance, "source": "udacity-live"},
        threshold,
        args.record_frames,
        args.record_fps,
        args.max_recorded_frames,
    )
    print("Annotate in a second terminal with scripts/mark_event.py. Risk flags are observational.")
    print("Mark lap_start, takeovers/resumes, off_track/collision and lap_complete or lap_abort.")
    try:
        if args.stop_file:
            asyncio.run(run_until_stopped(session, args.port, args.stop_file))
        else:
            web.run_app(create_app(session), host="127.0.0.1", port=args.port)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()


def mark_main() -> None:
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from robustdrive.simulator.interface import EVENTS

    parser = argparse.ArgumentParser(
        description="Timestamp a human-observed event in a running drive session."
    )
    parser.add_argument("event", choices=EVENTS)
    parser.add_argument("--note", default="")
    parser.add_argument("--port", type=int, default=4567)
    args = parser.parse_args()
    payload = json.dumps({"event": args.event, "note": args.note}).encode()
    request = Request(
        f"http://127.0.0.1:{args.port}/event",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            print(response.read().decode())
    except HTTPError as error:
        parser.error(error.read().decode())


def autonomy_main() -> None:
    from robustdrive.evaluation.autonomy import autonomy_metrics

    parser = argparse.ArgumentParser(
        description="Calculate autonomous-run metrics from real logs and manual annotations."
    )
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--takeover-penalty", type=float, default=6.0)
    args = parser.parse_args()
    print_result(autonomy_metrics(args.log_dir, args.takeover_penalty))
