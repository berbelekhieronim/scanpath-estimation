"""Phase 3 tests: marker aggregation, layer control, round reset."""

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
    from app import config, db, urls, main
    for mod in (config, db, urls, main):
        importlib.reload(mod)
    images = Path(tmp) / "images"
    images.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), "red").save(images / "a.jpg")
    Image.new("RGB", (640, 480), "blue").save(images / "b.png")
    with TestClient(main.app) as c:
        yield c


AUTH = {"X-Control-Token": "test-token"}
P1 = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
P2 = [[0.8, 0.8], [0.7, 0.7]]


def submit(client, points, uuid):
    rid = client.get("/api/state").json()["round_id"]
    return client.post("/api/markers", json={
        "round_id": rid, "participant_uuid": uuid, "points": points})


def test_markers_are_grouped_into_per_participant_paths(client):
    submit(client, P1, "participant-a")
    submit(client, P2, "participant-b")
    d = client.get("/api/markers").json()
    assert d["count"] == 2
    assert sorted(d["paths"], key=len) == sorted([P1, P2], key=len)


def test_each_path_keeps_its_tap_order(client):
    submit(client, P1, "participant-a")
    assert client.get("/api/markers").json()["paths"][0] == P1


def test_points_is_the_flattened_pool(client):
    submit(client, P1, "participant-a")
    submit(client, P2, "participant-b")
    d = client.get("/api/markers").json()
    assert len(d["points"]) == len(P1) + len(P2)
    assert all(p in d["points"] for p in P1 + P2)


def test_markers_endpoint_is_public(client):
    """The display must render without a token."""
    assert client.get("/api/markers").status_code == 200


def test_empty_round_returns_an_empty_shape_not_an_error(client):
    d = client.get("/api/markers").json()
    assert d["count"] == 0 and d["paths"] == [] and d["points"] == []


def test_default_layers(client):
    """Only the human heatmap starts on; every reveal is a deliberate act."""
    layers = client.get("/api/state").json()["layers"]
    assert layers == {"heatmap": True, "paths": False, "model": False,
                      "analysis": False, "prompt": False, "json": False}


def test_layers_toggle_and_persist(client):
    out = client.post("/api/control/layers", json={"paths": True, "heatmap": False},
                      headers=AUTH).json()
    assert out["layers"]["paths"] is True and out["layers"]["heatmap"] is False
    assert client.get("/api/state").json()["layers"]["paths"] is True


def test_layer_control_is_token_gated(client):
    assert client.post("/api/control/layers", json={"paths": True}).status_code == 403


def test_unknown_layer_is_rejected(client):
    assert client.post("/api/control/layers", json={"nope": True},
                       headers=AUTH).status_code == 400


def test_hiding_a_layer_does_not_delete_data(client):
    submit(client, P1, "participant-a")
    client.post("/api/control/layers", json={"heatmap": False}, headers=AUTH)
    assert client.get("/api/markers").json()["count"] == 1, "data survives a 'clear'"
    client.post("/api/control/layers", json={"heatmap": True}, headers=AUTH)
    assert client.get("/api/markers").json()["count"] == 1


def test_reset_round_empties_the_display_but_keeps_the_responses(client):
    submit(client, P1, "participant-a")
    old = client.get("/api/state").json()["round_id"]

    out = client.post("/api/control/reset-round", headers=AUTH).json()
    assert out["previous_round_id"] == old and out["round_id"] != old

    assert client.get("/api/markers").json()["count"] == 0, "display starts empty"

    from app import db
    assert len(db.get_round_markers(old)) == 3, "old responses are kept"


def test_reset_round_stays_on_the_same_image(client):
    before = client.get("/api/state").json()["image"]["id"]
    client.post("/api/control/reset-round", headers=AUTH)
    assert client.get("/api/state").json()["image"]["id"] == before


def test_reset_round_is_token_gated(client):
    assert client.post("/api/control/reset-round").status_code == 403


def test_display_and_control_pages_load(client):
    for path in ("/display", "/control", "/static/heatmap.js", "/static/display.css"):
        assert client.get(path).status_code == 200, path
