"""Verify help output for every entry point without training or simulator downloads."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    for script in sorted(Path("scripts").glob("*.py")):
        if script.name == "check_cli.py":
            continue
        subprocess.run(
            [sys.executable, str(script), "--help"], check=True, stdout=subprocess.DEVNULL
        )
        print(f"OK {script.name} --help")
