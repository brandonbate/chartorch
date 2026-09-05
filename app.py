from pathlib import Path

from flask import Flask, abort, request, send_from_directory

from codegen import to_python
from dataset_store import list_datasets, preview_payload
from jail import start as start_jail

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


@app.get("/api/preview/<dataset_id>")
def preview(dataset_id):
    raw = request.args.get("index", "0")
    try:
        index = int(raw)
    except (TypeError, ValueError):
        index = 0
    payload = preview_payload(dataset_id, index)
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
    print(script, flush=True)
    start_jail(script)
    return {"code": script}


if __name__ == "__main__":
    app.run(debug=True)
