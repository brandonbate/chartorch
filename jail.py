import json
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
_script = None
_epochs = None
_done = False
_exit_code = None
_params_printed = False
_traceback = None
_payload = None


def remember(script, epochs, payload=None):
    global _script, _epochs, _payload
    with _lock:
        _stop()
        _script = script
        _epochs = epochs
        _payload = payload


def has_script():
    return _script is not None


def is_running():
    return _proc is not None and _proc.poll() is None


def stored_epochs():
    return _epochs


def stored_payload():
    return _payload


def has_model():
    return _workdir is not None and (_workdir / "model.pt").is_file()


def start():
    global _proc, _workdir, _reader, _done, _exit_code, _params_printed, _traceback
    with _lock:
        if _script is None:
            return "no_script"
        if _proc is not None and _proc.poll() is None:
            return "running"
        _stop()
        run_root = ROOT / "run"
        run_root.mkdir(exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="netgui-run-", dir=run_root))
        script_path = workdir / "train.py"
        script_path.write_text(_script, encoding="utf-8")
        _done = False
        _exit_code = None
        _params_printed = False
        _traceback = None
        if not _spawn(workdir, script_path):
            shutil.rmtree(workdir, ignore_errors=True)
            _workdir = None
            return "failed"
        _workdir = workdir
        print("nsjail started", flush=True)
        return "ok"


def start_test(script):
    global _done, _exit_code, _traceback
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return "running"
        if _workdir is None or not (_workdir / "model.pt").is_file():
            return "no_model"
        if not isinstance(script, str) or not script:
            return "no_script"
        _kill_proc()
        results = _workdir / "test_results.json"
        results.unlink(missing_ok=True)
        script_path = _workdir / "test.py"
        script_path.write_text(script, encoding="utf-8")
        _done = False
        _exit_code = None
        _traceback = None
        if not _spawn(_workdir, script_path):
            return "failed"
        print("nsjail test started", flush=True)
        return "ok"


def status():
    with _lock:
        running = _proc is not None and _proc.poll() is None
        return {
            "running": running,
            "points": _read_metrics(),
            "done": _done and not running,
            "exit_code": _exit_code,
            "epochs": _epochs,
            "traceback": _traceback if _done and _exit_code not in (None, 0) else None,
        }


def test_status():
    with _lock:
        running = _proc is not None and _proc.poll() is None
        done = _done and not running
        result = _read_test_results() if done and _exit_code == 0 else None
        return {
            "running": running,
            "done": done,
            "exit_code": _exit_code,
            "traceback": _traceback if done and _exit_code not in (None, 0) else None,
            "result": result,
        }


def _spawn(workdir, script_path):
    global _proc, _reader, _done, _exit_code, _traceback
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
        _proc = None
        _reader = None
        _done = True
        _exit_code = -1
        _traceback = f"failed to start nsjail: {e}"
        return False
    _reader = threading.Thread(target=_pump, args=(_proc,), daemon=True)
    _reader.start()
    return True


def _read_test_results():
    if _workdir is None:
        return None
    path = _workdir / "test_results.json"
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    loss = obj.get("loss")
    if not isinstance(loss, (int, float)):
        return None
    acc = obj.get("accuracy")
    if acc is not None and not isinstance(acc, (int, float)):
        return None
    n_fail = obj.get("n_failures")
    if n_fail is not None and not isinstance(n_fail, int):
        return None
    n = obj.get("n")
    if n is not None and not isinstance(n, int):
        n = None
    raw = obj.get("failures")
    failures = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if not isinstance(idx, int):
                continue
            failures.append({
                "index": idx,
                "label": str(item.get("label", "")),
                "pred": str(item.get("pred", "")),
            })
    return {
        "loss": loss,
        "n": n,
        "accuracy": acc,
        "n_failures": n_fail,
        "failures": failures,
    }


def _read_metrics():
    if _workdir is None:
        return []
    path = _workdir / "metrics.jsonl"
    if not path.is_file():
        return []
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        epoch = obj.get("epoch")
        loss = obj.get("loss")
        if isinstance(epoch, int) and isinstance(loss, (int, float)):
            points.append({"epoch": epoch, "loss": loss})
    return points


def _print_params():
    global _params_printed
    if _params_printed or _workdir is None:
        return
    path = _workdir / "params.txt"
    if path.is_file():
        print(path.read_text(encoding="utf-8"), end="", flush=True)
    _params_printed = True


def _extract_traceback(text):
    marker = "Traceback (most recent call last):"
    idx = text.find(marker)
    if idx >= 0:
        return text[idx:]
    return text


def _pump(proc):
    global _done, _exit_code, _traceback
    chunks = []
    if proc.stdout is not None:
        for line in proc.stdout:
            print(line, end="", flush=True)
            chunks.append(line)
    code = proc.wait()
    _print_params()
    _exit_code = code
    if code != 0:
        _traceback = _extract_traceback("".join(chunks))
    else:
        _traceback = None
    _done = True
    print(f"nsjail exited {code}", flush=True)


def _kill_proc():
    global _proc, _reader
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


def _stop():
    global _workdir
    _kill_proc()
    if _workdir is not None:
        shutil.rmtree(_workdir, ignore_errors=True)
        _workdir = None
