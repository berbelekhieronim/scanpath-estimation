"""Tests for the start page and its status endpoint."""

import importlib
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

AUTH = {"X-Control-Token": "test-token"}
ROUTES = ["/start", "/", "/qr", "/display", "/control", "/admin"]


@pytest.fixture()
def env(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("SCANPATH_DATA_DIR", tmp)
    monkeypatch.setenv("SCANPATH_CONTROL_TOKEN", "test-token")
    root = Path(tmp)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "model").mkdir(parents=True, exist_ok=True)
    return root


def start(env):
    from app import config, db, urls, main
    for mod in (config, db, urls, main):
        importlib.reload(mod)
    return TestClient(main.app)


def with_image(env):
    Image.new("RGB", (800, 600), "red").save(env / "images" / "a.jpg")
    return env


def test_every_route_serves_a_distinct_page(env):
    """The bug this page exists to prevent: routes collapsing onto one view."""
    with start(with_image(env)) as c:
        bodies = {}
        for route in ROUTES:
            r = c.get(route)
            assert r.status_code == 200, route
            bodies[route] = r.text
        titles = {route: body.split("<title>")[1].split("</title>")[0]
                  for route, body in bodies.items()}
        assert len(set(titles.values())) == len(ROUTES), \
            f"routes share a title: {titles}"
        assert len(set(bodies.values())) == len(ROUTES), "routes share a body"


def test_status_reports_readiness(env):
    with start(with_image(env)) as c:
        s = c.get("/api/status").json()
        assert s["images"]["count"] == 1
        assert s["images"]["active"]["filename"] == "a.jpg"
        assert s["model_runs"] == {"count": 0, "synthetic": 0}
        assert s["responses"] == {"participants": 0, "markers": 0}


def test_status_never_leaks_the_control_token(env):
    """The start page is reachable by anyone who can reach the app."""
    with start(with_image(env)) as c:
        assert "test-token" not in json.dumps(c.get("/api/status").json())


def test_status_is_public(env):
    with start(with_image(env)) as c:
        assert c.get("/api/status").status_code == 200


def test_no_images_is_an_error_warning(env):
    with start(env) as c:
        levels = {w["level"] for w in c.get("/api/status").json()["warnings"]}
        texts = " ".join(w["text"] for w in c.get("/api/status").json()["warnings"])
        assert "error" in levels and "No images loaded" in texts


def test_unreachable_join_url_is_an_error_warning(env):
    with start(with_image(env)) as c:
        s = c.get("/api/status").json()
        assert s["join_url_reachable"] is False
        assert any("Ports panel" in w["text"] for w in s["warnings"])


def test_synthetic_runs_raise_a_warning(env):
    with_image(env)
    (env / "model" / "a.json").write_text(json.dumps({
        "image": "a.jpg", "mode": "freeview", "n_fixations": 5,
        "scanpath_norm": [[0.5, 0.5]], "source": "synthetic",
    }))
    with start(env) as c:
        s = c.get("/api/status").json()
        assert s["model_runs"]["synthetic"] == 1
        assert any("synthetic" in w["text"] for w in s["warnings"])


def test_missing_model_runs_raise_a_warning(env):
    with start(with_image(env)) as c:
        assert any("No model runs yet" in w["text"]
                   for w in c.get("/api/status").json()["warnings"])


def test_a_ready_server_has_no_error_warnings(env, monkeypatch):
    monkeypatch.setenv("SCANPATH_PUBLIC_URL", "https://demo-8000.app.github.dev")
    with_image(env)
    (env / "model" / "a.json").write_text(json.dumps({
        "image": "a.jpg", "mode": "freeview", "n_fixations": 5,
        "scanpath_norm": [[0.5, 0.5]], "source": "precomputed",
    }))
    with start(env) as c:
        s = c.get("/api/status").json()
        assert s["join_url_reachable"] is True
        assert [w for w in s["warnings"] if w["level"] == "error"] == []
