import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
SAFE_ID = re.compile(r"^[\w.-]+$")
HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

_preview_modules = {}


def _dataset_dir(dataset_id):
    if not isinstance(dataset_id, str) or not SAFE_ID.fullmatch(dataset_id):
        return None
    path = DATASETS_DIR / dataset_id
    if not path.is_dir():
        return None
    return path


def list_datasets():
    if not DATASETS_DIR.is_dir():
        return []
    names = []
    for path in sorted(DATASETS_DIR.iterdir()):
        if path.is_dir() and (path / "dataloader.py").is_file() and SAFE_ID.fullmatch(path.name):
            names.append(path.name)
    return names


def dataloader_snippet(dataset_id):
    folder = _dataset_dir(dataset_id)
    if folder is None:
        return None
    file = folder / "dataloader.py"
    if not file.is_file():
        return None
    return file.read_text(encoding="utf-8").strip("\n")


def test_dataloader_snippet(dataset_id):
    folder = _dataset_dir(dataset_id)
    if folder is None:
        return None
    file = folder / "test_dataloader.py"
    if not file.is_file():
        return None
    return file.read_text(encoding="utf-8").strip("\n")


def _preview_mod(dataset_id):
    if dataset_id in _preview_modules:
        return _preview_modules[dataset_id]
    folder = _dataset_dir(dataset_id)
    if folder is None:
        return None
    file = folder / "preview.py"
    if not file.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"netgui_dataset_{dataset_id}_preview", file)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _preview_modules[dataset_id] = mod
    return mod


def preview_payload(dataset_id, index=0, split="train"):
    mod = _preview_mod(dataset_id)
    if mod is None:
        return None
    fn = getattr(mod, "preview", None)
    if not callable(fn):
        return None
    result = fn(index, split=split)
    if not isinstance(result, dict):
        return None
    return result


def view_payload(dataset_id):
    mod = _preview_mod(dataset_id)
    if mod is None:
        return None
    view = getattr(mod, "VIEW", None)
    if not isinstance(view, dict):
        view = {}
    sample = view.get("sample", "image")
    if sample not in ("image", "text"):
        sample = "image"
    train_chart = bool(view.get("train_chart"))
    test = view.get("test")
    if test not in ("failures", "fit1d", "regions2d"):
        test = None
    points = None
    charts = None
    chart_id = None
    fn = getattr(mod, "chart", None)
    if (train_chart or test == "regions2d") and callable(fn):
        try:
            raw = fn("train")
        except Exception:
            raw = None
        parsed = _charts(raw)
        if parsed:
            charts = parsed
            default = getattr(mod, "DEFAULT_CHART", None)
            if not any(item["id"] == default for item in charts):
                default = charts[0]["id"]
            chart_id = default
            points = next((item["points"] for item in charts if item["id"] == default), charts[0]["points"])
        elif isinstance(raw, list):
            points = raw
    if not train_chart:
        points = None
    return {
        "sample": sample,
        "train_chart": train_chart,
        "test": test,
        "points": points,
        "charts": charts,
        "chart_id": chart_id,
        "class_colors": _class_colors(mod, view),
    }


def _charts(raw):
    if not isinstance(raw, list) or not raw:
        return None
    first = raw[0]
    if not isinstance(first, dict) or "points" not in first:
        return None
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cid = item.get("id")
        pts = item.get("points")
        xi = item.get("xi")
        yi = item.get("yi")
        if not isinstance(cid, str) or not cid:
            continue
        if not isinstance(pts, list) or not isinstance(xi, int) or not isinstance(yi, int):
            continue
        x_label = item.get("x_label")
        y_label = item.get("y_label")
        label = item.get("label")
        out.append({
            "id": cid,
            "label": label if isinstance(label, str) and label else cid,
            "xi": xi,
            "yi": yi,
            "x_label": x_label if isinstance(x_label, str) else "",
            "y_label": y_label if isinstance(y_label, str) else "",
            "points": pts,
        })
    return out or None


def _class_colors(mod, view):
    raw = getattr(mod, "CLASS_COLORS", None)
    if raw is None and isinstance(view, dict):
        raw = view.get("colors")
    if not isinstance(raw, dict):
        return None
    out = {}
    for key, val in raw.items():
        try:
            cls = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(val, str) and HEX_COLOR.fullmatch(val):
            out[str(cls)] = val
    return out or None
