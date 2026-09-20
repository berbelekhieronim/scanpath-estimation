"""Phase W3 tests: the viewing stage — sampling, off-image handling, upload."""

import importlib
import re
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
AUTH = {"X-Control-Token": "test-token"}


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


def calibrate(client, grade="good", uuid="participant-0001"):
    return client.post("/api/gaze/session", json={
        "participant_uuid": uuid, "grade": grade, "tracker": "mock",
        "mean_error": 0.22, "points_accepted": 9, "points_total": 9,
    }).json()["session_id"]


def samples(n=20, on_image=True):
    return [{"t": i * 250, "x": 0.4 if on_image else None,
             "y": 0.5 if on_image else None, "on_image": on_image}
            for i in range(n)]


# --- upload ----------------------------------------------------------------

def test_samples_upload_and_aggregate(client):
    sid = calibrate(client)
    r = client.post("/api/gaze/samples", json={"session_id": sid, "samples": samples(20)})
    assert r.status_code == 200 and r.json()["saved"] == 20

    agg = client.get("/api/gaze/aggregate").json()
    assert agg["contributors"] == 1 and len(agg["points"]) == 20


def test_upload_needs_a_real_session(client):
    assert client.post("/api/gaze/samples",
                       json={"session_id": 9999, "samples": samples(3)}).status_code == 404


def test_out_of_range_coordinates_are_rejected(client):
    sid = calibrate(client)
    bad = [{"t": 0, "x": 1.4, "y": 0.5, "on_image": True}]
    assert client.post("/api/gaze/samples",
                       json={"session_id": sid, "samples": bad}).status_code == 422


def test_off_image_samples_are_stored_but_left_out_of_the_aggregate(client):
    """Clamping them to the border would invent a pile of fixations along the
    edges that nobody made."""
    sid = calibrate(client)
    mixed = samples(10, on_image=True) + samples(6, on_image=False)
    client.post("/api/gaze/samples", json={"session_id": sid, "samples": mixed})

    from app import db
    with db.connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM gaze_samples").fetchone()[0]
    assert total == 16, "everything is stored"
    assert len(client.get("/api/gaze/aggregate").json()["points"]) == 10


def test_reuploading_replaces_rather_than_doubling(client):
    sid = calibrate(client)
    client.post("/api/gaze/samples", json={"session_id": sid, "samples": samples(20)})
    client.post("/api/gaze/samples", json={"session_id": sid, "samples": samples(5)})
    assert len(client.get("/api/gaze/aggregate").json()["points"]) == 5


def test_excluded_sessions_are_left_out_but_still_counted(client):
    """The exclusion rate has to stay reportable (spec 5.1)."""
    good = calibrate(client, "good", "participant-0001")
    poor = calibrate(client, "poor", "participant-0002")
    client.post("/api/gaze/samples", json={"session_id": good, "samples": samples(10)})
    client.post("/api/gaze/samples", json={"session_id": poor, "samples": samples(10)})

    agg = client.get("/api/gaze/aggregate").json()
    assert agg["contributors"] == 1, "only the usable session contributes"
    assert agg["sessions"] == 2 and agg["excluded"] == 1


def test_aggregate_groups_by_participant(client):
    for i in range(3):
        sid = calibrate(client, uuid=f"participant-{i:04d}")
        client.post("/api/gaze/samples", json={"session_id": sid, "samples": samples(8)})
    agg = client.get("/api/gaze/aggregate").json()
    assert agg["contributors"] == 3
    assert all(len(path) == 8 for path in agg["paths"])


def test_aggregate_is_public(client):
    assert client.get("/api/gaze/aggregate").status_code == 200


def test_empty_round_returns_a_shape_not_an_error(client):
    agg = client.get("/api/gaze/aggregate").json()
    assert agg["points"] == [] and agg["contributors"] == 0


# --- page ------------------------------------------------------------------

def test_view_page_loads(client):
    assert client.get("/view").status_code == 200


def test_view_does_not_clamp_off_image_gaze():
    src = (STATIC / "view.html").read_text()
    fn = src[src.index("function toImageCoords"):src.index("async function loadState")]
    assert "on_image: on" in fn
    assert "Math.min" not in fn and "Math.max" not in fn, "must not clamp"


def test_view_window_is_long_enough_for_the_cpu_frame_rate():
    """At ~4Hz on the CPU backend a 3-second window yields ~11 samples."""
    src = (STATIC / "view.html").read_text()
    ms = int(re.search(r"VIEW_MS = Number\(params\.get\('ms'\)\) \|\| (\d+)", src).group(1))
    assert ms >= 5000


def test_view_gives_the_eye_nothing_to_look_at_but_the_picture():
    """A counting digit is itself a fixation target."""
    src = (STATIC / "view.html").read_text()
    assert "ring" in src
    assert "Don't tap anything" in src


def test_calibration_hands_off_to_the_viewing_stage(client):
    """Previously it dumped the participant on the tap page, which knew
    nothing about the calibration that had just happened."""
    src = (STATIC / "calibrate.html").read_text()
    assert "location.href = '/view'" in src
    assert "scanpath_gaze_session_id" in src


def test_view_releases_the_camera():
    src = (STATIC / "view.html").read_text()
    assert "pagehide" in src and "visibilitychange" in src
    assert "tracker.destroy()" in src
