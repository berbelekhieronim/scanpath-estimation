"""Phase 2 tests: marker submission, validation, round safety, join URL."""

import importlib
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("SCANPATH_DATA_DIR", tmp)
    monkeypatch.setenv("SCANPATH_CONTROL_TOKEN", "test-token")
    monkeypatch.delenv("SCANPATH_PUBLIC_URL", raising=False)
    monkeypatch.delenv("CODESPACE_NAME", raising=False)

    from app import config, db, main, urls
    for mod in (config, db, urls, main):
        importlib.reload(mod)

    images = Path(tmp) / "images"
    images.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), "red").save(images / "a.jpg")
    Image.new("RGB", (640, 480), "blue").save(images / "b.png")

    with TestClient(main.app) as c:
        c.data_dir = Path(tmp)
        yield c


AUTH = {"X-Control-Token": "test-token"}


def submit(client, points, uuid="participant-0001", round_id=None):
    if round_id is None:
        round_id = client.get("/api/state").json()["round_id"]
    return client.post("/api/markers", json={
        "round_id": round_id, "participant_uuid": uuid, "points": points,
    })


PTS = [[0.1, 0.2], [0.5, 0.5], [0.9, 0.8], [0.3, 0.7], [0.6, 0.1]]


def test_submission_is_stored_and_counted(client):
    r = submit(client, PTS)
    assert r.status_code == 200
    assert r.json()["saved"] == 5
    assert client.get("/api/state").json()["responses"] == {"participants": 1, "markers": 5}


def test_tap_order_is_preserved(client):
    submit(client, PTS)
    from app import db
    rows = db.get_round_markers(client.get("/api/state").json()["round_id"])
    assert [r["seq"] for r in rows] == [0, 1, 2, 3, 4]
    assert [(round(r["x"], 3), round(r["y"], 3)) for r in rows] == [tuple(p) for p in PTS]


def test_many_participants_accumulate(client):
    for i in range(7):
        assert submit(client, PTS, uuid=f"participant-{i:04d}").status_code == 200
    assert client.get("/api/state").json()["responses"] == {"participants": 7, "markers": 35}


def test_resubmitting_replaces_rather_than_duplicating(client):
    submit(client, PTS)
    submit(client, [[0.4, 0.4], [0.5, 0.5]])
    state = client.get("/api/state").json()
    assert state["responses"] == {"participants": 1, "markers": 2}


@pytest.mark.parametrize("bad", [
    [[1.5, 0.5]], [[-0.1, 0.5]], [[0.5, 2.0]], [[0.5, -3.0]],
])
def test_out_of_range_coordinates_are_rejected(client, bad):
    assert submit(client, bad).status_code == 422


def test_empty_and_oversized_submissions_are_rejected(client):
    assert submit(client, []).status_code == 422
    assert submit(client, [[0.5, 0.5]] * 51).status_code == 422


def test_submitting_to_a_stale_round_is_refused(client):
    stale = client.get("/api/state").json()["round_id"]
    images = client.get("/api/images").json()["images"]
    client.post(f"/api/admin/active?image_id={images[1]['id']}", headers=AUTH)

    r = submit(client, PTS, round_id=stale)
    assert r.status_code == 409, "taps belong to the image the participant saw"
    assert "changed" in r.json()["detail"].lower()


def test_a_new_round_starts_empty(client):
    submit(client, PTS)
    images = client.get("/api/images").json()["images"]
    client.post(f"/api/admin/active?image_id={images[1]['id']}", headers=AUTH)
    assert client.get("/api/state").json()["responses"] == {"participants": 0, "markers": 0}


def test_the_same_device_is_one_participant_across_rounds(client):
    submit(client, PTS, uuid="same-device")
    images = client.get("/api/images").json()["images"]
    client.post(f"/api/admin/active?image_id={images[1]['id']}", headers=AUTH)
    submit(client, PTS, uuid="same-device")

    from app import db
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM participants").fetchone()[0] == 1


def test_submission_needs_no_token(client):
    """Participants must never be asked to authenticate."""
    assert submit(client, PTS).status_code == 200


def test_join_url_flags_an_unreachable_address(client):
    d = client.get("/api/join-url").json()
    assert d["source"] == "request"
    assert d["reachable"] is False, "a bare hostname must be flagged"


@pytest.mark.parametrize("url,reachable", [
    ("http://localhost:8000", False),
    ("http://127.0.0.1:8000", False),
    ("http://0.0.0.0:8000", False),
    ("http://testserver", False),          # bare hostname resolves nowhere useful
    ("http://mylaptop:8000", False),
    ("http://192.168.1.5:8000", True),     # venue wifi: genuinely reachable
    ("https://x-8000.app.github.dev", True),
])
def test_reachability_heuristic(url, reachable):
    from app import urls
    assert urls.is_reachable_by_others(url) is reachable


def test_explicit_public_url_wins(client, monkeypatch):
    monkeypatch.setenv("SCANPATH_PUBLIC_URL", "https://demo-8000.app.github.dev/")
    from app import urls
    importlib.reload(urls)
    info = urls.public_base_url("http://testserver/")
    assert info == {"url": "https://demo-8000.app.github.dev", "source": "SCANPATH_PUBLIC_URL"}
    assert urls.is_reachable_by_others(info["url"]) is True


def test_codespace_url_is_reconstructed(client, monkeypatch):
    monkeypatch.setenv("CODESPACE_NAME", "fuzzy-space-guide-xyz")
    monkeypatch.setenv("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN", "app.github.dev")
    from app import urls
    importlib.reload(urls)
    assert urls.codespace_url(8000) == "https://fuzzy-space-guide-xyz-8000.app.github.dev"


def test_qr_svg_is_scalable(client):
    r = client.get("/api/qr.svg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    body = r.text
    assert "viewBox" in body
    assert 'width="' not in body.split("<path")[0], "fixed width would not scale"


def test_capture_page_loads(client):
    assert client.get("/").status_code == 200
    assert client.get("/qr").status_code == 200


# --- the submitted state (Undo stayed visible on a real device) -----------

def test_hidden_attribute_is_enforced_globally():
    """The browser's own [hidden] rule is display:none at the lowest
    specificity, so any class with an explicit display beats it and .hidden
    silently does nothing. That is how the Undo button survived submission."""
    css = (Path(__file__).resolve().parent.parent
           / "app" / "static" / "app.css").read_text()
    assert "[hidden] { display: none !important; }" in css


def test_actions_bar_would_otherwise_have_beaten_hidden():
    """Guards the specific collision, so reordering the CSS cannot quietly
    reintroduce it."""
    capture = (Path(__file__).resolve().parent.parent
               / "app" / "static" / "capture.css").read_text()
    assert "display: flex" in capture.split(".actions")[1].split("}")[0]


def test_submitting_never_sends_a_tapper_into_calibration():
    """The tap page used to end with "now the other half" and a button into
    the camera stages. Tapping and eye tracking are separate conditions —
    one person in both is the contamination the whole split exists to avoid,
    and it was reachable with one tap."""
    src = (Path(__file__).resolve().parent.parent
           / "app" / "static" / "index.html").read_text()
    assert "to-camera" not in src
    assert "Measure where I really look" not in src
    # Routing an already-gaze-assigned arrival to /calibrate is the correct
    # use and stays; what is gone is reaching it from the thank-you screen.
    assert src.count("'/calibrate'") == 1
    assert "routeByCondition" in src


@pytest.mark.parametrize("flag,phrase", [
    ("viewed", "All done"),
    ("nocamera", "No camera measurement"),
    ("declined", "Camera not used"),
])
def test_returning_from_the_camera_stages_is_explained(flag, phrase):
    """Landing back on the tap screen with no explanation looks like the app
    forgot what just happened."""
    src = (Path(__file__).resolve().parent.parent
           / "app" / "static" / "index.html").read_text()
    assert flag in src and phrase in src
