from pathlib import Path

from flask import Flask, abort, request, send_from_directory

from codegen import to_python

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


@app.post("/api/generate")
def generate():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400)
    script = to_python(payload)
    if script is None:
        abort(400)
    print(script, flush=True)
    return ("", 204)


if __name__ == "__main__":
    app.run(debug=True)
