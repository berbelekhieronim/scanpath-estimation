"""Phase 6 tests: pushing runs from another machine, and session export."""

import importlib
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

AUTH = {"X-Control-Token": "test-token"}


@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("SCANPATH_DATA_DIR", tmp)
    monkeypatch.setenv("SCANPATH_CONTROL_TOKEN", "test-token")
    root = Path(tmp)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "model").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), "red").save(root / "images" / "a.jpg")

    from app import config, db, urls, main
    for mod in (config, db, urls, main):
        importlib.reload(mod)
    with TestClient(main.app) as c:
        c.root = root
        yield c


def run_payload(**over):
    p = {
        "image": "a.jpg", "mode": "freeview", "n_fixations": 3,
        "scanpath_norm": [[0.1, 0.2], [0.5, 0.5], [0.8, 0.3]],
        "samples_norm": [[[0.1, 0.2], [0.5, 0.5], [0.8, 0.3]]],
        "source": "pushed", "model": "test", "device": "mps",
        "elapsed_seconds": 92.4,
    }
    p.update(over)
    return p


# --- push ------------------------------------------------------------------

def test_a_pushed_run_appears_immediately(client):
    r = client.post("/api/model/push", json=run_payload(), headers=AUTH)
    assert r.status_code == 200
    assert r.json()["stored_as"] == "pushed"

    run = client.get("/api/model").json()["run"]
    assert run["source"] == "pushed"
    assert run["device"] == "mps"
    assert run["elapsed_seconds"] == 92.4


def test_push_requires_the_token(client):
    assert client.post("/api/model/push", json=run_payload()).status_code == 403
    assert client.post("/api/model/push", json=run_payload(),
                       headers={"X-Control-Token": "wrong"}).status_code == 403


def test_push_rejects_an_unknown_image_and_says_which_are_known(client):
    r = client.post("/api/model/push", json=run_payload(image="ghost.jpg"),
                    headers=AUTH)
    assert r.status_code == 404
    assert "a.jpg" in r.json()["detail"], "the error must name the valid options"


@pytest.mark.parametrize("bad", [
    {"scanpath_norm": [[1.5, 0.5]]},
    {"scanpath_norm": [[-0.2, 0.5]]},
    {"scanpath_norm": []},
    {"mode": "nonsense"},
])
def test_push_validates_the_payload(client, bad):
    assert client.post("/api/model/push", json=run_payload(**bad),
                       headers=AUTH).status_code == 422


def test_pushing_twice_replaces_rather_than_duplicating(client):
    client.post("/api/model/push", json=run_payload(), headers=AUTH)
    client.post("/api/model/push",
                json=run_payload(scanpath_norm=[[0.9, 0.9], [0.1, 0.1], [0.5, 0.5]]),
                headers=AUTH)
    assert len(client.get("/api/model/runs").json()["runs"]) == 1
    assert client.get("/api/model").json()["run"]["scanpath_norm"][0] == [0.9, 0.9]


def test_a_pushed_run_overrides_a_precomputed_one(client):
    """The live run is the one the presenter just made; it should win."""
    (client.root / "model" / "a.json").write_text(json.dumps(
        run_payload(source="precomputed")))
    client.post("/api/control/rescan-model", headers=AUTH)
    assert client.get("/api/model").json()["run"]["source"] == "precomputed"

    client.post("/api/model/push", json=run_payload(source="pushed"), headers=AUTH)
    assert client.get("/api/model").json()["run"]["source"] == "pushed"


def test_a_pushed_run_feeds_the_analysis(client):
    rid = client.get("/api/state").json()["round_id"]
    for i in range(14):
        client.post("/api/markers", json={
            "round_id": rid, "participant_uuid": f"participant-{i:04d}",
            "points": [[0.1, 0.2], [0.5, 0.5], [0.8, 0.3]]})
    client.post("/api/model/push", json=run_payload(), headers=AUTH)

    a = client.get("/api/analysis").json()
    assert a["ok"] is True
    assert a["model_source"] == "pushed"


# --- export ----------------------------------------------------------------

def test_export_contains_everything_needed_to_rebuild_a_session(client):
    rid = client.get("/api/state").json()["round_id"]
    client.post("/api/markers", json={
        "round_id": rid, "participant_uuid": "participant-0001",
        "points": [[0.1, 0.2], [0.3, 0.4]]})
    client.post("/api/model/push", json=run_payload(), headers=AUTH)

    d = client.get("/api/export", headers=AUTH).json()
    assert d["counts"] == {"images": 1, "rounds": 1, "participants": 1,
                           "markers": 2, "model_runs": 1}
    for key in ("images", "rounds", "participants", "markers", "model_runs"):
        assert key in d
    assert d["model_runs"][0]["payload"]["source"] == "pushed"
    assert d["markers"][0]["x"] == 0.1


def test_export_omits_user_agent(client):
    """Anonymity is the promise made on the participant screen."""
    rid = client.get("/api/state").json()["round_id"]
    client.post("/api/markers",
                json={"round_id": rid, "participant_uuid": "participant-0001",
                      "points": [[0.1, 0.2]]},
                headers={"User-Agent": "Mozilla/5.0 (identifying string)"})

    d = client.get("/api/export", headers=AUTH).json()
    assert "user_agent" not in d["participants"][0]
    assert "identifying string" not in json.dumps(d)


def test_export_includes_closed_rounds(client):
    """Earlier rounds must survive a reset, or the export loses a group."""
    rid = client.get("/api/state").json()["round_id"]
    client.post("/api/markers", json={
        "round_id": rid, "participant_uuid": "participant-0001",
        "points": [[0.1, 0.2]]})
    client.post("/api/control/reset-round", headers=AUTH)

    d = client.get("/api/export", headers=AUTH).json()
    assert d["counts"]["rounds"] == 2
    assert d["counts"]["markers"] == 1, "the first round's response is still there"


def test_export_requires_the_token(client):
    assert client.get("/api/export").status_code == 403
