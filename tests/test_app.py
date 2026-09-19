"""Phase 1 tests: schema, image sync, round lifecycle, auth, routing.

Runs against a temporary data directory so it never touches real session data.
"""

import importlib
import os
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

    from app import config, db, main
    for mod in (config, db, main):
        importlib.reload(mod)

    images = Path(tmp) / "images"
    images.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), "red").save(images / "a.jpg")
    Image.new("RGB", (640, 480), "blue").save(images / "b.png")

    with TestClient(main.app) as c:
        c.data_dir = Path(tmp)
        yield c


AUTH = {"X-Control-Token": "test-token"}


def test_images_discovered_with_real_dimensions(client):
    imgs = client.get("/api/images").json()["images"]
    assert {i["filename"] for i in imgs} == {"a.jpg", "b.png"}
    by_name = {i["filename"]: i for i in imgs}
    assert (by_name["a.jpg"]["width"], by_name["a.jpg"]["height"]) == (800, 600)
    assert (by_name["b.png"]["width"], by_name["b.png"]["height"]) == (640, 480)


def test_a_round_opens_automatically_on_first_boot(client):
    state = client.get("/api/state").json()
    assert state["round_id"] is not None
    assert state["image"] is not None
    assert state["tap_count"] == 5
    assert state["responses"] == {"participants": 0, "markers": 0}


@pytest.mark.parametrize("headers,expected", [
    ({}, 403),
    ({"X-Control-Token": "wrong"}, 403),
    (AUTH, 200),
])
def test_presenter_routes_are_token_gated(client, headers, expected):
    assert client.post("/api/admin/rescan", headers=headers).status_code == expected


def test_token_accepted_via_query_param_too(client):
    assert client.post("/api/admin/rescan?k=test-token").status_code == 200
    assert client.post("/api/admin/rescan?k=nope").status_code == 403


def test_setting_active_image_opens_a_round_and_closes_the_old_one(client):
    imgs = client.get("/api/images").json()["images"]
    first, second = imgs[0]["id"], imgs[1]["id"]

    r1 = client.post(f"/api/admin/active?image_id={first}", headers=AUTH).json()
    r2 = client.post(f"/api/admin/active?image_id={second}", headers=AUTH).json()
    assert r1["round_id"] != r2["round_id"]

    state = client.get("/api/state").json()
    assert state["round_id"] == r2["round_id"]
    assert state["image"]["id"] == second

    from app import db
    with db.connect() as conn:
        still_open = conn.execute(
            "SELECT COUNT(*) FROM rounds WHERE closed_at IS NULL"
        ).fetchone()[0]
    assert still_open == 1, "exactly one round may be open at a time"


def test_unknown_image_id_is_rejected(client):
    assert client.post("/api/admin/active?image_id=9999", headers=AUTH).status_code == 404


def test_missing_file_is_hidden_but_its_row_survives(client):
    (client.data_dir / "images" / "a.jpg").unlink()
    result = client.post("/api/admin/rescan", headers=AUTH).json()["result"]
    assert result["missing"] == ["a.jpg"]
    assert {i["filename"] for i in client.get("/api/images").json()["images"]} == {"b.png"}

    # The row is kept, not deleted — a round may still reference it.
    from app import db
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 2


def test_restoring_a_file_brings_it_back(client):
    path = client.data_dir / "images" / "a.jpg"
    data = path.read_bytes()
    path.unlink()
    client.post("/api/admin/rescan", headers=AUTH)
    path.write_bytes(data)
    result = client.post("/api/admin/rescan", headers=AUTH).json()["result"]
    assert result["restored"] == ["a.jpg"]
    assert len(client.get("/api/images").json()["images"]) == 2


def test_unreadable_and_non_image_files_are_skipped(client):
    (client.data_dir / "images" / "notes.txt").write_text("not an image")
    (client.data_dir / "images" / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\nGARBAGE")
    client.post("/api/admin/rescan", headers=AUTH)
    assert len(client.get("/api/images").json()["images"]) == 2


def test_rescan_is_idempotent(client):
    client.post("/api/admin/rescan", headers=AUTH)
    second = client.post("/api/admin/rescan", headers=AUTH).json()["result"]
    assert second == {"added": [], "restored": [], "missing": []}


@pytest.mark.parametrize("path", ["/", "/admin", "/display", "/control", "/healthz", "/favicon.ico"])
def test_pages_are_served(client, path):
    assert client.get(path).status_code == 200


def test_images_are_served_but_the_database_is_not(client):
    assert client.get("/img/a.jpg").status_code == 200
    assert client.get("/img/app.sqlite").status_code == 404
    assert client.get("/app.sqlite").status_code == 404
