"""Windows play-and-learn launcher; no terminal is needed with a desktop shortcut."""

import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open the Windows play-and-learn launcher.")
    parser.add_argument("--attach-pid", type=int, help="Adopt an already opened simulator process")
    parser.add_argument(
        "--check", action="store_true", help="Check local setup without opening windows"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.check:
        from robustdrive.desktop import simulator_path

        for path in (root / ".venv/Scripts/python.exe", root / "data/raw", simulator_path(root)):
            if not path.exists():
                raise SystemExit(f"Missing: {path}")
        print("Python, recording folder, and simulator are ready.")
    else:
        from robustdrive.desktop import main

        main(root, args.attach_pid)
