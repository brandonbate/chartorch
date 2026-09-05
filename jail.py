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
_reader = None


def start(script):
    global _proc, _workdir, _reader
    with _lock:
        _stop()
        run_root = ROOT / "run"
        run_root.mkdir(exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="netgui-run-", dir=run_root))
        script_path = workdir / "train.py"
        script_path.write_text(script, encoding="utf-8")
        data_dir = ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        (workdir / "data").mkdir(exist_ok=True)
        cmd = [
            "nsjail",
            "-Mo",
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
            "--tmpfsmount",
            "/tmp",
            "--keep_env",
            "--env",
            "HOME=/tmp",
            "--",
            sys.executable,
            "-u",
            str(script_path),
        ]
        try:
            _proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            print(f"failed to start nsjail: {e}", flush=True)
            shutil.rmtree(workdir, ignore_errors=True)
            _proc = None
            _workdir = None
            _reader = None
            return
        _workdir = workdir
        _reader = threading.Thread(target=_pump, args=(_proc,), daemon=True)
        _reader.start()
        print("nsjail started", flush=True)


def _pump(proc):
    if proc.stdout is not None:
        for line in proc.stdout:
            print(line, end="", flush=True)
    code = proc.wait()
    print(f"nsjail exited {code}", flush=True)


def _stop():
    global _proc, _workdir, _reader
    if _proc is not None:
        if _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _proc.kill()
                _proc.wait()
        if _reader is not None:
            _reader.join(timeout=2)
        if _proc.stdout is not None:
            _proc.stdout.close()
    _proc = None
    _reader = None
    if _workdir is not None:
        shutil.rmtree(_workdir, ignore_errors=True)
        _workdir = None
