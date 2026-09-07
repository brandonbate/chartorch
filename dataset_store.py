import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
SAFE_ID = re.compile(r"^[\w.-]+$")

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
    if test not in ("failures", "fit1d"):
        test = None
    points = None
    if train_chart:
        fn = getattr(mod, "chart", None)
        if callable(fn):
            try:
                raw = fn("train")
            except Exception:
                raw = None
            if isinstance(raw, list):
                points = raw
    return {
        "sample": sample,
        "train_chart": train_chart,
        "test": test,
        "points": points,
    }
