"""The labelling server: what it serves, what it saves, and what it refuses.

It serves files from a directory by a name taken from the URL, which is the one
thing in this repository that reads a path from outside it. That guard gets a test
whatever else does.

It was driven by hand in a browser once — the magnifier tracked, a click landed on
the right source pixel, N recorded a frame without a ball. None of that was
repeatable, so the parts that can be are here.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from gridiron_tracking import labelling


@pytest.fixture
def served(tmp_path):
    """A running server over two frames, torn down at the end."""
    frames = tmp_path / "frames"
    frames.mkdir()
    shots = []
    for i in range(2):
        name = f"clip_{i:05d}.jpg"
        (frames / name).write_bytes(b"\xff\xd8\xff" + bytes(50))      # jpeg-ish
        shots.append(labelling.Shot(key=f"clip/{i}", clip="clip", frame=i,
                                    path=str(frames / name), width=1280, height=720))
    (frames / "manifest.json").write_text(json.dumps([s.to_json() for s in shots]))
    (tmp_path / "secret.txt").write_text("not for the browser")

    from gridiron_tracking import label_server

    labels = tmp_path / "labels.json"
    holder = {}

    def run():
        # Rebuild the handler the way serve() does, but on an ephemeral port and
        # without the blocking serve_forever/log noise.
        import gridiron_tracking.label_server as ls
        original = ThreadingHTTPServer

        def capture(addr, handler):
            srv = original(("127.0.0.1", 0), handler)
            holder["srv"] = srv
            holder["port"] = srv.server_port
            started.set()
            srv.serve_forever()
            return srv

        ls.ThreadingHTTPServer = capture
        try:
            ls.serve(port=0, frame_dir=frames, label_file=labels, log=lambda *a: None)
        finally:
            ls.ThreadingHTTPServer = original

    started = threading.Event()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert started.wait(5), "server did not start"
    yield f"http://127.0.0.1:{holder['port']}", frames, labels, tmp_path
    holder["srv"].shutdown()


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=5) as r:
        return r.status, r.read()


def post(url, path, payload):
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_it_serves_the_page_the_manifest_and_a_frame(served):
    url, frames, _labels, _root = served

    status, body = get(url, "/")
    assert status == 200 and b"Label the football" in body

    status, body = get(url, "/manifest.json")
    assert status == 200
    assert [s["key"] for s in json.loads(body)] == ["clip/0", "clip/1"]

    status, body = get(url, "/frames/clip_00000.jpg")
    assert status == 200 and body.startswith(b"\xff\xd8\xff")


def test_a_path_outside_the_frame_directory_is_refused(served):
    """The only place this repository takes a path from outside itself."""
    url, _frames, _labels, root = served
    assert (root / "secret.txt").exists()

    for attempt in ("/frames/../secret.txt",
                    "/frames/..%2fsecret.txt",
                    "/frames/%2e%2e/secret.txt",
                    "/frames/subdir/../../secret.txt"):
        try:
            status, body = get(url, attempt)
        except urllib.error.HTTPError as e:
            assert e.code == 404, attempt
            continue
        assert b"not for the browser" not in body, f"{attempt} escaped the directory"


def test_a_label_is_on_disk_before_the_response_comes_back(served):
    """Half an hour of someone's attention must not live in a browser tab."""
    url, _frames, labels, _root = served

    status, counts = post(url, "/labels",
                          {"key": "clip/0", "x": 640.0, "y": 360.0, "size": 22})
    assert status == 200
    assert counts == {"labelled": 1, "with_a_ball": 1, "without": 0}

    on_disk = labelling.LabelSet.load(labels)
    assert on_disk.labels["clip/0"].x == 640.0


def test_a_frame_with_no_ball_is_recorded_as_such(served):
    url, _frames, labels, _root = served
    post(url, "/labels", {"key": "clip/0", "x": None, "y": None, "size": 22})

    stored = labelling.LabelSet.load(labels).labels["clip/0"]
    assert not stored.visible
    assert labelling.LabelSet.load(labels).counts()["without"] == 1


def test_a_label_can_be_taken_back(served):
    url, _frames, labels, _root = served
    post(url, "/labels", {"key": "clip/1", "x": 10.0, "y": 20.0, "size": 22})
    assert "clip/1" in labelling.LabelSet.load(labels).labels

    req = urllib.request.Request(url + "/labels/clip%2F1", method="DELETE")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200
    assert "clip/1" not in labelling.LabelSet.load(labels).labels


def test_labels_survive_a_restart_of_the_page(served):
    """Reloading must come back to the same work, not an empty store."""
    url, _frames, _labels, _root = served
    post(url, "/labels", {"key": "clip/0", "x": 1.0, "y": 2.0, "size": 22})

    status, body = get(url, "/labels")
    assert status == 200
    assert json.loads(body)["clip/0"]["x"] == 1.0


def test_an_unknown_route_is_a_plain_404(served):
    url, _frames, _labels, _root = served
    with pytest.raises(urllib.error.HTTPError) as e:
        get(url, "/wat")
    assert e.value.code == 404


def test_traversal_is_refused_even_unnormalised_on_the_wire(served):
    """urllib collapses ``..`` before it sends, so the tests above may never put a
    traversal in front of the server at all. This writes the request by hand.

    The guard is structural rather than a check: the name is taken with
    ``Path(...).name``, which has no directory component left to escape with.
    """
    import socket

    url, _frames, _labels, root = served
    port = int(url.rsplit(":", 1)[1])
    assert (root / "secret.txt").read_text() == "not for the browser"

    def raw(path):
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall(f"GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".encode())
        chunks = []
        while True:
            c = s.recv(4096)
            if not c:
                break
            chunks.append(c)
        s.close()
        return b"".join(chunks)

    for attempt in ("/frames/../secret.txt",
                    "/frames/../../secret.txt",
                    "/frames/%2e%2e%2fsecret.txt",
                    "/frames/....//secret.txt",
                    "/frames/..%5csecret.txt"):
        body = raw(attempt)
        assert b"404" in body.split(b"\r\n")[0], attempt
        assert b"not for the browser" not in body, f"{attempt} escaped the directory"
