"""Phase W1 tests: the gaze tracker's assets are served where it expects them.

The tracker itself cannot be tested here — it needs a camera pointed at a real
face. These tests cover the packaging, which is where the first two real bugs
were: a missing worker bundle and un-namespaced globals.
"""

import importlib
import json
import re
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"


@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("SCANPATH_DATA_DIR", tmp)
    monkeypatch.setenv("SCANPATH_CONTROL_TOKEN", "test-token")
    root = Path(tmp)
    (root / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), "red").save(root / "images" / "a.jpg")
    from app import config, db, urls, main
    for mod in (config, db, urls, main):
        importlib.reload(mod)
    with TestClient(main.app) as c:
        yield c


def test_gazetest_page_is_served(client):
    r = client.get("/gazetest")
    assert r.status_code == 200
    assert "Gaze tracking test" in r.text


def test_model_weights_are_served_at_the_hardcoded_path(client):
    """The vendored bundle hard-codes '/web/model.json'. Serving it anywhere
    else silently breaks gaze estimation."""
    r = client.get("/web/model.json")
    assert r.status_code == 200
    manifest = r.json()
    assert "weightsManifest" in manifest or "modelTopology" in manifest

    shard = client.get("/web/group1-shard1of1.bin")
    assert shard.status_code == 200
    assert len(shard.content) > 500_000, "weights shard looks truncated"


def test_the_bundle_really_does_hardcode_that_path():
    """If upstream ever makes the path configurable, this test should fail so
    the mount can be reconsidered rather than left as a mystery."""
    bundle = (STATIC / "vendor" / "webeyetrack" / "webeyetrack.umd.js").read_text(
        errors="ignore")
    assert '"/web/model.json"' in bundle


def test_tracker_module_is_served_and_exports_the_interface(client):
    r = client.get("/static/gaze/tracker.js")
    assert r.status_code == 200
    for name in ("GazeTrackerBase", "WebEyeTrackBackend", "MockBackend",
                 "createTracker", "preflight", "describeError", "STATUS",
                 "DEPENDENCIES"):
        assert re.search(rf"^export (?:async )?(?:class|function|const) {name}\b",
                         r.text, re.M), f"{name} is not exported"


def test_vendored_bundle_is_served_whole(client):
    r = client.get("/static/vendor/webeyetrack/webeyetrack.umd.js")
    assert r.status_code == 200
    assert len(r.content) > 2_000_000


def test_licence_is_vendored_alongside_the_code():
    """MIT requires the notice travels with the code."""
    licence = (STATIC / "vendor" / "webeyetrack" / "LICENSE").read_text()
    assert "MIT" in licence or "Permission is hereby granted" in licence


def test_adapter_uses_the_main_thread_class_not_the_proxy():
    """WebEyeTrackProxy needs index.worker.js, which the published package
    does not contain. Using it would fail at runtime."""
    src = (STATIC / "gaze" / "tracker.js").read_text()
    assert re.search(r"new\s+lib\.WebEyeTrack\s*\(", src)
    # The proxy must not actually be constructed anywhere.
    assert not re.search(r"new\s+\w*\.?WebEyeTrackProxy\s*\(", src)


def test_tracker_is_constructed_with_an_explicit_max_points():
    """WebEyeTrack's maxPoints defaults to 5 and pruneCalibData() keeps only
    the most recent that many. A 9-point calibration on the default would
    silently discard the first four and report a confident, wrong model."""
    src = (STATIC / "gaze" / "tracker.js").read_text()
    m = re.search(r"new\s+lib\.WebEyeTrack\s*\(\s*([^)]+?)\s*\)", src)
    assert m and m.group(1).strip(), "must pass maxPoints explicitly"

    default = re.search(r"maxPoints\s*=\s*(\d+)", src)
    assert default and int(default.group(1)) >= 9, \
        "maxPoints must hold at least the 9 calibration points"


def test_adapter_reads_globals_off_window_not_a_namespace():
    """The UMD wrapper copies exports straight onto window; expecting a
    window.webeyetrack namespace fails, which is how this was first found."""
    src = (STATIC / "gaze" / "tracker.js").read_text()
    assert "window.WebEyeTrack" in src and "window.WebcamClient" in src


def test_adapter_stops_camera_tracks_explicitly():
    """A camera left live after a demo is the one bug nobody forgives."""
    src = (STATIC / "gaze" / "tracker.js").read_text()
    assert "getTracks" in src and "srcObject = null" in src


def test_test_page_releases_camera_when_hidden():
    src = (STATIC / "gazetest.html").read_text()
    assert "visibilitychange" in src and "pagehide" in src


def test_dependency_list_names_the_remote_ones():
    """These two are the venue-wifi risk; the preflight has to cover them."""
    src = (STATIC / "gaze" / "tracker.js").read_text()
    assert "cdn.jsdelivr.net" in src
    assert "storage.googleapis.com" in src


def test_gazetest_page_states_the_privacy_position(client):
    text = client.get("/gazetest").text
    assert "never uploaded" in text
