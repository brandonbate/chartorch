from pathlib import Path

from flask import Flask, abort, request, send_from_directory

from codegen import to_python, to_test_python
from dataset_store import list_datasets, preview_payload, view_payload
from jail import (
    has_model,
    has_script,
    is_running,
    remember,
    start,
    start_test,
    status,
    stored_epochs,
    stored_payload,
    test_status,
)

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/css/<path:filename>")
def css(filename):
    return send_from_directory(ROOT / "css", filename)


@app.get("/js/<path:filename>")
def js(filename):
    return send_from_directory(ROOT / "js", filename)


@app.get("/api/datasets")
def datasets():
    return {"datasets": list_datasets()}


@app.get("/api/dataset/<dataset_id>/view")
def dataset_view(dataset_id):
    payload = view_payload(dataset_id)
    if payload is None:
        abort(404)
    return payload


@app.get("/api/preview/<dataset_id>")
def preview(dataset_id):
    raw = request.args.get("index", "0")
    try:
        index = int(raw)
    except (TypeError, ValueError):
        index = 0
    split = request.args.get("split", "train")
    if split not in ("train", "test"):
        split = "train"
    payload = preview_payload(dataset_id, index, split=split)
    if payload is None:
        abort(404)
    return payload


@app.post("/api/generate")
def generate():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400)
    script = to_python(payload)
    if script is None:
        abort(400)
    epochs = (payload.get("training") or {}).get("epochs")
    remember(script, epochs, payload)
    print(script, flush=True)
    return {"code": script}


@app.post("/api/train")
def train():
    if not has_script():
        abort(400)
    if is_running():
        abort(409)
    result = start()
    if result == "running":
        abort(409)
    if result != "ok":
        abort(500)
    return {"epochs": stored_epochs()}


@app.get("/api/train/status")
def train_status():
    return status()


@app.post("/api/test")
def test():
    if is_running():
        abort(409)
    if not has_model():
        abort(400)
    payload = stored_payload()
    script = to_test_python(payload) if payload is not None else None
    if script is None:
        abort(400)
    result = start_test(script)
    if result == "running":
        abort(409)
    if result != "ok":
        abort(500)
    return {"ok": True}


@app.get("/api/test/status")
def test_status_route():
    return test_status()


if __name__ == "__main__":
    app.run(debug=True)
