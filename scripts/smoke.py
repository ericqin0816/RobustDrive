"""Run synthetic software verification from the repository root."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run all offline stages on SYNTHETIC SMOKE ONLY data."
    )
    parser.add_argument(
        "--ablation", action="store_true", help="Also train/reuse all five tiny ablations"
    )
    args = parser.parse_args()

    def run(script, *arguments):
        subprocess.run([sys.executable, f"scripts/{script}.py", *arguments], check=True)

    sample = Path("data/sample/driving_log.csv")
    if not sample.exists():
        run("create_sample_data")
    marker = sample.parent / "dataset_metadata.json"
    if not marker.exists() or json.loads(marker.read_text())["kind"] != "synthetic":
        raise ValueError("data/sample must be generated synthetic data.")
    checkpoint = "experiments/results/smoke/best.pt"
    if not Path(checkpoint).exists():
        run("train", "--config", "configs/smoke.yaml")
    run("inspect_dataset", "--data", "data/sample", "--figure-dir", "reports/figures/smoke_dataset")
    run("evaluate", "--checkpoint", checkpoint)
    run("benchmark_robustness", "--checkpoint", checkpoint)
    run("evaluate_uncertainty", "--checkpoint", checkpoint)
    if args.ablation:
        run("run_ablation", "--train", "--smoke")
    run("run_ablation", "--kind", "synthetic")
    run("run_ablation")
    print("Synthetic software verification finished. All research experiments: PENDING REAL DATA.")


if __name__ == "__main__":
    main()
