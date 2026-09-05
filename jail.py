import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_lock = threading.Lock()
_proc = None
_workdir = None


def start(script):
    global _proc, _workdir
    with _lock:
        _stop()
        workdir = Path(tempfile.mkdtemp(prefix="netgui-run-"))
        script_path = workdir / "train.py"
        script_path.write_text(script, encoding="utf-8")
        data_dir = ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        (workdir / "data").mkdir(exist_ok=True)
        cmd = [
            "nsjail",
            "-Mo",
            "-Q",
            "--chroot",
            "/",
            "--disable_rlimits",
            "--time_limit",
            "0",
            "--user",
            str(os.getuid()),
            "--group",
            str(os.getgid()),
            "--cwd",
            str(workdir),
            "--bindmount",
            str(workdir),
            "--bindmount",
            f"{data_dir}:{workdir / 'data'}",
            "--bindmount",
            "/tmp",
            "--keep_env",
            "--",
            sys.executable,
            str(script_path),
        ]
        try:
            _proc = subprocess.Popen(cmd)
            _workdir = workdir
        except OSError as e:
            print(f"failed to start nsjail: {e}", flush=True)
            shutil.rmtree(workdir, ignore_errors=True)
            _proc = None
            _workdir = None


def _stop():
    global _proc, _workdir
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _proc.kill()
            _proc.wait()
    _proc = None
    if _workdir is not None:
        shutil.rmtree(_workdir, ignore_errors=True)
        _workdir = None
