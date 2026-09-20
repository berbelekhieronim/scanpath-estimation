"""Phase W2 tests: consent, calibration geometry, quality gating, storage."""

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


def session(grade="good", uuid="participant-0001", **over):
    p = {"participant_uuid": uuid, "grade": grade, "tracker": "mock",
         "tracker_version": "1", "mean_error": 0.12, "worst_error": 0.2,
         "points_accepted": 9, "points_total": 9,
         "validation": [{"target": [0.3, 0.3], "error": 0.11}]}
    p.update(over)
    return p


# --- geometry --------------------------------------------------------------

def _points(name):
    src = (STATIC / "gaze" / "calibration.js").read_text()
    body = re.search(rf"export const {name} = \[(.*?)\];", src, re.S).group(1)
    return [[float(v) for v in m] for m in re.findall(r"\[([\d.]+),\s*([\d.]+)\]", body)]


def test_nine_calibration_points_and_four_validation_points():
    assert len(_points("CALIB_POINTS")) == 9
    assert len(_points("VALIDATION_POINTS")) == 4


def test_validation_points_are_never_calibration_points():
    """Scoring on a trained point measures memorisation, not accuracy."""
    for v in _points("VALIDATION_POINTS"):
        for c in _points("CALIB_POINTS"):
            assert ((v[0] - c[0]) ** 2 + (v[1] - c[1]) ** 2) ** 0.5 > 0.1


def test_points_are_inset_from_the_edges():
    """Gaze at the extreme edge is the worst-estimated part of the range."""
    for x, y in _points("CALIB_POINTS") + _points("VALIDATION_POINTS"):
        assert 0.1 <= x <= 0.9 and 0.1 <= y <= 0.9


def test_calibration_points_clear_the_trackers_proximity_filter():
    """Upstream drops a point within 0.05 units of the previous one."""
    pts = _points("CALIB_POINTS")
    for i, a in enumerate(pts):
        for bpt in pts[i + 1:]:
            assert abs(a[0] - bpt[0]) >= 0.06 or abs(a[1] - bpt[1]) >= 0.06


def test_validation_samples_the_window_before_the_tap():
    """After the tap nothing holds the participant's gaze on the target, so
    sampling afterwards measures drift and scores it as error."""
    src = (STATIC / "gaze" / "calibration.js").read_text()
    block = src[src.index("PHASE.VALIDATING"):src.index("this._setPhase(PHASE.DONE)")]
    assert "await this._awaitTap();\n        const mean = this._recentMean" in block


# --- quality grading -------------------------------------------------------

@pytest.mark.parametrize("err,expected", [
    (0.05, "good"), (0.18, "good"),
    (0.19, "usable"), (0.30, "usable"),
    (0.31, "poor"), (0.9, "poor"),
    (None, "failed"),
])
def test_grade_boundaries(err, expected):
    src = (STATIC / "gaze" / "calibration.js").read_text()
    good = float(re.search(r"GOOD:\s*([\d.]+)", src).group(1))
    usable = float(re.search(r"USABLE:\s*([\d.]+)", src).group(1))
    if err is None:
        grade = "failed"
    elif err <= good:
        grade = "good"
    elif err <= usable:
        grade = "usable"
    else:
        grade = "poor"
    assert grade == expected


# --- storage ---------------------------------------------------------------

def test_a_good_session_is_stored_as_usable(client):
    r = client.post("/api/gaze/session", json=session("good"))
    assert r.status_code == 200
    d = r.json()
    assert d["usable"] is True
    assert d["stats"]["usable"] == 1 and d["stats"]["excluded"] == 0


@pytest.mark.parametrize("grade", ["poor", "failed"])
def test_a_failed_session_is_stored_and_counted_as_excluded(client, grade):
    """Discarding failures would make every other number look better than it
    is. The exclusion rate has to be reportable (spec 5.1)."""
    d = client.post("/api/gaze/session", json=session(grade)).json()
    assert d["usable"] is False
    assert d["stats"]["total"] == 1
    assert d["stats"]["excluded"] == 1
    assert d["stats"]["usable"] == 0


def test_stats_accumulate_a_mixed_room(client):
    for i, g in enumerate(["good", "good", "usable", "poor", "failed"]):
        client.post("/api/gaze/session", json=session(g, uuid=f"participant-{i:04d}"))
    s = client.get("/api/gaze/stats").json()
    assert s["total"] == 5 and s["usable"] == 3 and s["excluded"] == 2
    assert s["grades"] == {"good": 2, "usable": 1, "poor": 1, "failed": 1}


def test_an_unknown_grade_is_rejected(client):
    assert client.post("/api/gaze/session",
                       json=session("excellent")).status_code == 422


def test_session_needs_no_token(client):
    """Participants must never be asked to authenticate."""
    assert client.post("/api/gaze/session", json=session()).status_code == 200


def test_recalibrating_records_both_attempts(client):
    """Attempts are data: two poor tries then a good one is worth knowing."""
    client.post("/api/gaze/session", json=session("poor", uuid="participant-0001"))
    client.post("/api/gaze/session", json=session("good", uuid="participant-0001"))
    assert client.get("/api/gaze/stats").json()["total"] == 2


def test_session_is_scoped_to_the_current_round(client):
    client.post("/api/gaze/session", json=session())
    assert client.get("/api/gaze/stats").json()["total"] == 1
    client.post("/api/control/reset-round", headers={"X-Control-Token": "test-token"})
    assert client.get("/api/gaze/stats").json()["total"] == 0, "new round starts clean"


# --- pages -----------------------------------------------------------------

def test_consent_and_calibrate_pages_load(client):
    assert client.get("/consent").status_code == 200
    assert client.get("/calibrate").status_code == 200
    assert client.get("/static/gaze/calibration.js").status_code == 200


def test_consent_states_the_promises_plainly(client):
    text = client.get("/consent").text
    assert "never sent anywhere" in text
    assert "No video or photos are uploaded" in text
    assert "Declining" in text, "declining must be presented as an equal option"


def test_calibrate_releases_the_camera_on_hide(client):
    text = client.get("/calibrate").text
    assert "visibilitychange" in text and "pagehide" in text
