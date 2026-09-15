import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

ROOT = Path(__file__).resolve().parent
SESSIONS_ROOT = ROOT / "run" / "sessions"
STALE_SECONDS = 24 * 3600
ARTIFACTS = (
    "model.pt",
    "metrics.jsonl",
    "traceback.txt",
    "params.txt",
    "test_results.json",
    "test.py",
)

_local_lock = threading.Lock()
_local = {}


def valid_sid(sid):
    if not isinstance(sid, str):
        return False
    try:
        parsed = uuid.UUID(sid)
    except ValueError:
        return False
    return str(parsed) == sid


def remember(sid, script, epochs, payload=None):
    workdir = _session_dir(sid)
    workdir.mkdir(parents=True, exist_ok=True)
    with _lock_session(workdir):
        _stop_job(sid, workdir)
        _clear_artifacts(workdir)
        (workdir / "train.py").write_text(script, encoding="utf-8")
        if payload is not None:
            (workdir / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
        else:
            (workdir / "payload.json").unlink(missing_ok=True)
        _write_meta(
            workdir,
            {
                "epochs": epochs,
                "pid": None,
                "phase": "idle",
                "done": True,
                "exit_code": None,
                "cancelled": False,
            },
        )
    cleanup_stale()


def has_script(sid):
    path = _session_dir(sid) / "train.py"
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def is_running(sid):
    return _session_running(sid, _session_dir(sid))


def stored_epochs(sid):
    return _read_meta(_session_dir(sid)).get("epochs")


def stored_payload(sid):
    path = _session_dir(sid) / "payload.json"
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def has_model(sid):
    return (_session_dir(sid) / "model.pt").is_file()


def start(sid):
    cleanup_stale()
    workdir = _session_dir(sid)
    if not (workdir / "train.py").is_file():
        return "no_script"
    with _lock_session(workdir):
        if _session_running(sid, workdir):
            return "running"
        _stop_job(sid, workdir)
        _clear_artifacts(workdir)
        meta = _read_meta(workdir)
        meta.update(
            {
                "pid": None,
                "phase": "train",
                "done": False,
                "exit_code": None,
                "cancelled": False,
            }
        )
        _write_meta(workdir, meta)
        if not _spawn(sid, workdir, workdir / "train.py", "train"):
            meta = _read_meta(workdir)
            meta.update({"phase": "idle", "done": True, "exit_code": -1, "pid": None})
            _write_meta(workdir, meta)
            return "failed"
        return "ok"


def stop_training(sid):
    workdir = _session_dir(sid)
    if not workdir.is_dir():
        return "idle"
    with _lock_session(workdir):
        if not _session_running(sid, workdir):
            return "idle"
        meta = _read_meta(workdir)
        meta["cancelled"] = True
        _write_meta(workdir, meta)
        _stop_job(sid, workdir)
        return "ok"


def start_test(sid, script):
    cleanup_stale()
    workdir = _session_dir(sid)
    if not workdir.is_dir() or not (workdir / "model.pt").is_file():
        return "no_model"
    if not isinstance(script, str) or not script:
        return "no_script"
    with _lock_session(workdir):
        if _session_running(sid, workdir):
            return "running"
        _stop_job(sid, workdir)
        (workdir / "test_results.json").unlink(missing_ok=True)
        (workdir / "traceback.txt").unlink(missing_ok=True)
        script_path = workdir / "test.py"
        script_path.write_text(script, encoding="utf-8")
        meta = _read_meta(workdir)
        meta.update(
            {
                "pid": None,
                "phase": "test",
                "done": False,
                "exit_code": None,
                "cancelled": False,
            }
        )
        _write_meta(workdir, meta)
        if not _spawn(sid, workdir, script_path, "test"):
            meta = _read_meta(workdir)
            meta.update({"phase": "idle", "done": True, "exit_code": -1, "pid": None})
            _write_meta(workdir, meta)
            return "failed"
        print("nsjail test started", flush=True)
        return "ok"


def status(sid):
    workdir = _session_dir(sid)
    meta = _read_meta(workdir)
    running = _session_running(sid, workdir)
    done = _job_done(running, meta)
    exit_code = meta.get("exit_code")
    cancelled = bool(meta.get("cancelled")) if done else False
    return {
        "running": running,
        "points": _read_metrics(workdir),
        "done": done,
        "exit_code": exit_code,
        "epochs": meta.get("epochs"),
        "cancelled": cancelled,
        "traceback": _read_traceback(workdir) if done and exit_code not in (None, 0) and not cancelled else None,
    }


def test_status(sid):
    workdir = _session_dir(sid)
    meta = _read_meta(workdir)
    running = _session_running(sid, workdir)
    done = _job_done(running, meta)
    exit_code = meta.get("exit_code")
    result = _read_test_results(workdir) if done and exit_code == 0 else None
    return {
        "running": running,
        "done": done,
        "exit_code": exit_code,
        "traceback": _read_traceback(workdir) if done and exit_code not in (None, 0) else None,
        "result": result,
    }


def cleanup_stale(max_age=STALE_SECONDS):
    if not SESSIONS_ROOT.is_dir():
        return
    now = time.time()
    for path in SESSIONS_ROOT.iterdir():
        if not path.is_dir() or not valid_sid(path.name):
            continue
        try:
            if now - _newest_mtime(path) < max_age:
                continue
        except OSError:
            continue
        if _session_running(path.name, path):
            continue
        shutil.rmtree(path, ignore_errors=True)


def _session_dir(sid):
    if not valid_sid(sid):
        raise ValueError("invalid session id")
    return SESSIONS_ROOT / sid


def _empty_meta():
    return {
        "epochs": None,
        "pid": None,
        "phase": "idle",
        "done": False,
        "exit_code": None,
        "cancelled": False,
    }


def _read_meta(workdir):
    path = workdir / "meta.json"
    if not path.is_file():
        return _empty_meta()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_meta()
    if not isinstance(obj, dict):
        return _empty_meta()
    meta = _empty_meta()
    meta.update(obj)
    return meta


def _write_meta(workdir, meta):
    path = workdir / "meta.json"
    tmp = workdir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.replace(path)


@contextmanager
def _lock_session(workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    fd = os.open(workdir / ".lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _local_rec(sid):
    with _local_lock:
        rec = _local.get(sid)
        if rec is None:
            return None, None
        return rec.get("proc"), rec.get("reader")


def _set_local(sid, proc, reader):
    with _local_lock:
        _local[sid] = {"proc": proc, "reader": reader}


def _pop_local(sid, proc=None):
    with _local_lock:
        rec = _local.get(sid)
        if rec is None:
            return
        if proc is not None and rec.get("proc") is not proc:
            return
        _local.pop(sid, None)


def _pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        text = cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return "nsjail" in text


def _session_running(sid, workdir):
    proc, _ = _local_rec(sid)
    if proc is not None and proc.poll() is None:
        return True
    return _pid_alive(_read_meta(workdir).get("pid"))


def _job_done(running, meta):
    if running:
        return False
    if meta.get("done"):
        return True
    return meta.get("phase") in ("train", "test")


def _clear_artifacts(workdir):
    for name in ARTIFACTS:
        (workdir / name).unlink(missing_ok=True)


def _newest_mtime(path):
    latest = path.stat().st_mtime
    for child in path.iterdir():
        try:
            latest = max(latest, child.stat().st_mtime)
        except OSError:
            continue
    return latest


def _spawn(sid, workdir, script_path, phase):
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
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except OSError as e:
        print(f"failed to start nsjail: {e}", flush=True)
        (workdir / "traceback.txt").write_text(f"failed to start nsjail: {e}", encoding="utf-8")
        return False
    reader = threading.Thread(target=_pump, args=(sid, workdir, proc), daemon=True)
    reader.start()
    _set_local(sid, proc, reader)
    meta = _read_meta(workdir)
    meta["pid"] = proc.pid
    meta["phase"] = phase
    meta["done"] = False
    meta["exit_code"] = None
    _write_meta(workdir, meta)
    if phase == "train":
        print("nsjail started", flush=True)
    return True


def _read_test_results(workdir):
    path = workdir / "test_results.json"
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
    plot = _parse_plot(obj.get("plot"))
    return {
        "loss": loss,
        "n": n,
        "accuracy": acc,
        "n_failures": n_fail,
        "failures": failures,
        "plot": plot,
    }


def _parse_xy_list(raw):
    points = []
    if not isinstance(raw, list):
        return points
    for item in raw:
        if not isinstance(item, dict):
            continue
        x = item.get("x")
        y = item.get("y")
        if isinstance(x, bool) or isinstance(y, bool):
            continue
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            pt = {"x": x, "y": y}
            color = _parse_color(item.get("color"))
            if color is not None:
                pt["color"] = color
            points.append(pt)
    return points


def _parse_color(value):
    if isinstance(value, str) and HEX_COLOR.fullmatch(value):
        return value
    return None


def _parse_plot(raw):
    if not isinstance(raw, dict):
        return None
    points = _parse_xy_list(raw.get("points"))
    curve = _parse_xy_list(raw.get("curve"))
    if raw.get("kind") == "regions2d":
        return _parse_regions2d(raw, points)
    if not points and not curve:
        return None
    return {"points": points, "curve": curve}


def _parse_regions2d(raw, points):
    nx = raw.get("nx")
    ny = raw.get("ny")
    bounds = {}
    for key in ("x_min", "x_max", "y_min", "y_max"):
        val = raw.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            if not points:
                return None
            return {"kind": "regions2d", "points": points, "curve": []}
        bounds[key] = val
    pred = []
    raw_pred = raw.get("pred")
    if isinstance(nx, int) and isinstance(ny, int) and nx >= 1 and ny >= 1 and isinstance(raw_pred, list):
        for val in raw_pred[: nx * ny]:
            if isinstance(val, bool) or not isinstance(val, int):
                pred = []
                break
            pred.append(val)
        if len(pred) != nx * ny:
            pred = []
    class_colors = {}
    raw_cc = raw.get("class_colors")
    if isinstance(raw_cc, dict):
        for key, val in raw_cc.items():
            color = _parse_color(val)
            if color is not None:
                class_colors[str(key)] = color
    if not pred:
        if not points:
            return None
        return {"kind": "regions2d", "points": points, "curve": [], **bounds}
    return {
        "kind": "regions2d",
        **bounds,
        "nx": nx,
        "ny": ny,
        "pred": pred,
        "class_colors": class_colors,
        "points": points,
        "curve": [],
    }


def _read_metrics(workdir):
    path = workdir / "metrics.jsonl"
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


def _print_params(workdir):
    path = workdir / "params.txt"
    if path.is_file():
        print(path.read_text(encoding="utf-8"), end="", flush=True)


def _read_traceback(workdir):
    path = workdir / "traceback.txt"
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return text or None


def _extract_traceback(text):
    marker = "Traceback (most recent call last):"
    idx = text.find(marker)
    if idx >= 0:
        return text[idx:]
    return text


def _pump(sid, workdir, proc):
    chunks = []
    if proc.stdout is not None:
        for line in proc.stdout:
            print(line, end="", flush=True)
            chunks.append(line)
    code = proc.wait()
    _print_params(workdir)
    meta = _read_meta(workdir)
    if meta.get("pid") != proc.pid:
        _pop_local(sid, proc)
        print(f"nsjail exited {code}", flush=True)
        return
    cancelled = bool(meta.get("cancelled"))
    if code != 0 and not cancelled:
        (workdir / "traceback.txt").write_text(_extract_traceback("".join(chunks)), encoding="utf-8")
    else:
        (workdir / "traceback.txt").unlink(missing_ok=True)
    meta["done"] = True
    meta["exit_code"] = code
    meta["pid"] = None
    meta["phase"] = "idle"
    _write_meta(workdir, meta)
    _pop_local(sid, proc)
    print(f"nsjail exited {code}", flush=True)


def _kill_pid(pid):
    if not _pid_alive(pid):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
    deadline = time.time() + 2
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _stop_job(sid, workdir):
    proc, reader = _local_rec(sid)
    pid = proc.pid if proc is not None else _read_meta(workdir).get("pid")
    if proc is not None:
        if proc.poll() is None:
            _kill_pid(proc.pid)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _kill_pid(proc.pid)
                proc.wait()
        if reader is not None:
            reader.join(timeout=2)
        if proc.stdout is not None:
            proc.stdout.close()
        _pop_local(sid, proc)
    elif pid:
        _kill_pid(pid)
