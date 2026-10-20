"""A local server for the labelling UI. No dependencies, no build step.

Labels are written to disk on every click rather than accumulated in the browser
and exported at the end. Labelling a few hundred frames is half an hour of
someone's attention, and losing it to a closed tab or a refresh is the kind of
avoidable loss that stops the job being done a second time.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .labelling import FRAME_DIR, LABEL_FILE, Label, LabelSet, load_manifest

UI_DIR = Path(__file__).resolve().parent / "label_ui"

TYPES = {".html": "text/html; charset=utf-8", ".jpg": "image/jpeg",
         ".png": "image/png", ".json": "application/json"}


def serve(port: int = 4176, frame_dir: Path = FRAME_DIR,
          label_file: Path = LABEL_FILE, log=print) -> None:
    shots = load_manifest(frame_dir)
    labels = LabelSet.load(label_file)
    frame_dir, label_file = Path(frame_dir), Path(label_file)

    if not shots:
        raise SystemExit(
            f"no frames in {frame_dir}. Run `python -m gridiron_tracking label "
            f"--extract` first."
        )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: D102 - one line per click is noise
            pass

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                return self._send(200, (UI_DIR / "index.html").read_bytes(), TYPES[".html"])
            if path == "/manifest.json":
                return self._send(200, json.dumps([s.to_json() for s in shots]).encode(),
                                  TYPES[".json"])
            if path == "/labels":
                return self._send(200, json.dumps(
                    {k: v.to_json() for k, v in labels.labels.items()}).encode(),
                    TYPES[".json"])
            if path.startswith("/frames/"):
                # Resolved against the frame directory and checked to still be
                # inside it, so a crafted path cannot read the rest of the disk.
                name = Path(path[len("/frames/"):]).name
                f = (frame_dir / name).resolve()
                if not str(f).startswith(str(frame_dir.resolve())) or not f.exists():
                    return self._send(404, b"no such frame", "text/plain")
                return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if unquote(urlparse(self.path).path) != "/labels":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            label = Label.from_json(payload)
            labels.labels[label.key] = label
            labels.save(label_file)
            return self._send(200, json.dumps(labels.counts()).encode(), TYPES[".json"])

        def do_DELETE(self):  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if not path.startswith("/labels/"):
                return self._send(404, b"not found", "text/plain")
            labels.labels.pop(path[len("/labels/"):], None)
            labels.save(label_file)
            return self._send(200, json.dumps(labels.counts()).encode(), TYPES[".json"])

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    log(f"  {len(shots)} frames, {labels.summary()}")
    log(f"  labelling at http://localhost:{port}  — every click is saved to "
        f"{label_file}")
    log("  ctrl-c when you have had enough; it picks up where you left off")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log(f"\n  stopped. {LabelSet.load(label_file).summary()}")
    finally:
        server.server_close()
