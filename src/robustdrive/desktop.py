"""A small local launcher: play, stop recording, close the game, then train."""

import ctypes
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from robustdrive.utils.logging import utc_now, write_json


def simulator_path(root: Path) -> Path:
    return root / "simulator/beta_simulator_windows/beta_simulator.exe"


def finished_process(pid: int):
    """Wait for an already-launched Windows game without manipulating its UI."""
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
    if not handle:
        return
    try:
        kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
    finally:
        kernel.CloseHandle(handle)


def train_recording(root: Path, notify) -> dict:
    """Run existing scientific CLIs in order, with outputs tied to one frozen input."""
    from robustdrive.config import load_config, save_config
    from robustdrive.recording import prepare_recording

    notify("Checking your recording and preserving a training snapshot...")
    csv_path = prepare_recording(root / "data/raw/driving_log.csv", root / "data/processed")
    tag = csv_path.parent.name
    log_dir = root / "local/play_runs" / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    python = root / ".venv/Scripts/python.exe"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    output_log = log_dir / "training.log"
    result = {
        "recording": str(csv_path.relative_to(root)),
        "status": "training",
        "started": utc_now(),
    }
    write_json(root / "local/play_status.json", result)

    def run(script, *arguments):
        with output_log.open("a", encoding="utf-8") as log:
            command = [str(python), str(root / "scripts" / f"{script}.py"), *map(str, arguments)]
            completed = subprocess.run(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                env={**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "Agg"},
            )
        if completed.returncode:
            raise RuntimeError(f"{script} stopped. Details are in {output_log}")

    run(
        "inspect_dataset",
        "--data",
        csv_path,
        "--output",
        log_dir / "inspection.json",
        "--figure-dir",
        root / "reports/figures" / f"{tag}_dataset",
    )
    for index, variant in enumerate(("baseline", "augmented", "uncertainty"), 1):
        name = f"{tag}_{variant}"
        config = load_config(root / "configs" / f"{variant}.yaml")
        config["experiment"]["name"] = name
        config["data"]["csv_path"] = csv_path.relative_to(root).as_posix()
        saved_config = log_dir / f"{variant}.yaml"
        save_config(config, saved_config)
        output = root / "experiments/results" / name
        complete = output / "metrics.json"
        if not complete.exists() or not json.loads(complete.read_text(encoding="utf-8")).get(
            "complete"
        ):
            notify(f"Training model {index}/3: {variant}. You can leave this window minimized.")
            arguments = ["--config", saved_config]
            latest = output / "latest.pt"
            if latest.exists():
                arguments += ["--resume", latest]
            run("train", *arguments)
        notify(f"Evaluating {variant} on held-out frames...")
        checkpoint = output / "best.pt"
        run("evaluate", "--checkpoint", checkpoint)
        notify(f"Checking how {variant} handles visual changes...")
        run("benchmark_robustness", "--checkpoint", checkpoint)
        if variant == "uncertainty":
            notify("Measuring uncertainty and prediction rejection...")
            run("evaluate_uncertainty", "--checkpoint", checkpoint)
    run("run_ablation")
    result.update(
        status="ready",
        completed=utc_now(),
        checkpoint=f"experiments/results/{tag}_baseline/best.pt",
        uncertainty_checkpoint=f"experiments/results/{tag}_uncertainty/best.pt",
        figures=f"reports/figures/{tag}_baseline",
    )
    write_json(root / "local/play_status.json", result)
    notify("Your models and reports are ready. Watch my model is now available.")
    return result


def launch(root: Path, attach_pid: int | None = None) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    os.chdir(root)
    root.joinpath("local").mkdir(exist_ok=True)
    window = tk.Tk()
    window.title("RobustDrive — Play & Learn")
    window.geometry("630x440")
    window.resizable(False, False)
    frame = ttk.Frame(window, padding=24)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="RobustDrive", font=("Segoe UI", 23, "bold")).pack(anchor="w")
    ttk.Label(frame, text="Your driving teaches the car.", font=("Segoe UI", 12)).pack(
        anchor="w", pady=(0, 14)
    )
    instructions = (
        "1. Choose Training Mode in the game.\n"
        "2. Click Record, select Documents > RobustDrive Recordings,\n"
        "   then click Record again so it says RECORDING.\n"
        "3. Drive with the arrow keys (or W/A/S/D).\n\n"
        "When finished: press R, wait for saving to finish, then close\n"
        "the game. Training and analysis will start automatically."
    )
    ttk.Label(frame, text=instructions, font=("Segoe UI", 10), justify="left").pack(anchor="w")
    status = tk.StringVar(
        value="Ready to play. Aim for several smooth laps with left and right turns."
    )
    ttk.Separator(frame).pack(fill="x", pady=16)
    ttk.Label(frame, textvariable=status, wraplength=580, font=("Segoe UI", 10)).pack(anchor="w")
    events = queue.Queue()
    busy = False
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def background(action):
        nonlocal busy
        if busy:
            return
        busy = True
        for button in (play_button, train_button, watch_button):
            button.state(["disabled"])

        def work():
            try:
                action()
            except Exception as error:
                events.put(("status", str(error)))
                write_json(
                    root / "local/play_error.json", {"timestamp": utc_now(), "error": str(error)}
                )
            finally:
                events.put(("idle", None))

        threading.Thread(target=work, daemon=False).start()

    def notify(text):
        events.put(("status", text))

    def train_saved():
        train_recording(root, notify)

    def play(existing_pid=None):
        def work():
            if existing_pid is not None:
                notify(
                    "The game is open. Record your drive; stop recording and close it when done."
                )
                finished_process(existing_pid)
            else:
                executable = simulator_path(root)
                if not executable.is_file():
                    raise FileNotFoundError(
                        "Simulator is missing. See docs/play.md for installation."
                    )
                process = subprocess.Popen(
                    [
                        str(executable),
                        "-screen-fullscreen",
                        "0",
                        "-screen-width",
                        "1280",
                        "-screen-height",
                        "720",
                    ],
                    cwd=executable.parent,
                )
                notify(
                    "The game is open. Record your drive; stop recording and close it when done."
                )
                process.wait()
            train_saved()

        background(work)

    def watch():
        def work():
            saved = json.loads((root / "local/play_status.json").read_text(encoding="utf-8"))
            if saved.get("status") != "ready":
                raise ValueError("Record a drive and finish training before watching the model.")
            checkpoint = root / saved["checkpoint"]
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    "The trained checkpoint was moved; train your recording again."
                )
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", 4567))
                except OSError as error:
                    raise RuntimeError(
                        "Another driving server is using port 4567. Close it before watching this model."
                    ) from error
            run_name = "drive_play_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            output = root / "experiments/results" / run_name
            log = (root / "local/watch_model.log").open("a", encoding="utf-8")
            server = subprocess.Popen(
                [
                    str(root / ".venv/Scripts/python.exe"),
                    str(root / "scripts/drive.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(output),
                    "--record-frames",
                ],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            try:
                from urllib.request import urlopen

                deadline = time.monotonic() + 30
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(
                            "Driving server could not start. See local/watch_model.log."
                        )
                    try:
                        with urlopen("http://127.0.0.1:4567/", timeout=1) as response:
                            if json.load(response).get("adapter") == "udacity-websocket":
                                break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            "Driving server did not become ready. See local/watch_model.log."
                        )
                    time.sleep(0.25)
                game = subprocess.Popen(
                    [
                        str(simulator_path(root)),
                        "-screen-fullscreen",
                        "0",
                        "-screen-width",
                        "1280",
                        "-screen-height",
                        "720",
                    ],
                    cwd=simulator_path(root).parent,
                )
                notify(
                    "Choose Autonomous Mode in the game. Supervise it; this is a first learned model."
                )
                game.wait()
            finally:
                if server.poll() is None:
                    server.terminate()
                    server.wait(timeout=15)
                log.close()
            notify(
                "Drive log saved. Completion and intervention claims still need real event annotations."
            )

        background(work)

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=(16, 8))
    play_button = ttk.Button(buttons, text="Play & record", command=play)
    train_button = ttk.Button(
        buttons, text="Train saved drive", command=lambda: background(train_saved)
    )
    watch_button = ttk.Button(buttons, text="Watch my model", command=watch)
    for button in (play_button, train_button, watch_button):
        button.pack(side="left", padx=(0, 8))
    ttk.Button(
        frame,
        text="Copy recording folder path",
        command=lambda: (window.clipboard_clear(), window.clipboard_append(str(root / "data/raw"))),
    ).pack(anchor="w")

    def enable_idle_buttons():
        for button in (play_button, train_button):
            button.state(["!disabled"])
        ready = False
        saved = root / "local/play_status.json"
        if saved.exists():
            try:
                info = json.loads(saved.read_text(encoding="utf-8"))
                ready = info.get("status") == "ready" and (root / info["checkpoint"]).is_file()
            except (ValueError, KeyError):
                pass
        watch_button.state(["!disabled"] if ready else ["disabled"])

    enable_idle_buttons()

    def refresh():
        nonlocal busy
        while not events.empty():
            kind, value = events.get_nowait()
            if kind == "status":
                status.set(value)
            elif kind == "idle":
                busy = False
                enable_idle_buttons()
        window.after(250, refresh)

    def close():
        if busy:
            window.iconify()
            status.set("Still running. Restore this window from the taskbar to see progress.")
        else:
            window.destroy()

    # Hold a per-workspace lock for the entire launcher lifetime.
    lock = (root / "local/play.lock").open("a+b")
    try:
        import msvcrt

        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except (ImportError, OSError):
        messagebox.showinfo(
            "RobustDrive",
            "The launcher is already open, or this is not Windows. Use its taskbar window.",
        )
        lock.close()
        window.destroy()
        return
    window.protocol("WM_DELETE_WINDOW", close)
    refresh()
    if attach_pid:
        window.after(100, lambda: play(attach_pid))
    try:
        window.mainloop()
    finally:
        lock.close()


def main(root: Path, attach_pid: int | None = None):
    if sys.platform != "win32":
        raise SystemExit(
            "The desktop launcher is for Windows; use the cross-platform Python CLIs elsewhere."
        )
    launch(root, attach_pid)
