"""Tests for the start page and its status endpoint."""

import importlib
import json
import tempfile
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"

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


# --- stale process detection ----------------------------------------------

def test_status_reports_whether_the_process_is_older_than_the_code(env):
    """Static files are read from disk per request, so a pull updates the UI
    while Python keeps running what it imported at startup. That presents as
    a dozen unrelated bugs — 'Unknown layer: gaze' being one."""
    with start(with_image(env)) as c:
        srv = c.get("/api/status").json()["server"]
        assert {"started_at", "uptime_seconds", "code_mtime", "stale"} <= set(srv)
        assert srv["stale"] is False


def test_stale_server_is_flagged(env, monkeypatch):
    with start(with_image(env)) as c:
        from app import main
        monkeypatch.setattr(main, "_STARTED_AT", 0.0)   # process "started" in 1970
        assert c.get("/api/status").json()["server"]["stale"] is True


def test_start_page_warns_about_a_stale_server():
    src = (Path(__file__).resolve().parent.parent
           / "app" / "static" / "start.html").read_text()
    assert "running older code" in src
    assert "devserver.sh restart" in src


def test_the_start_page_stopped_duplicating_the_nav():
    """Every tile pointed at a page the top nav already reaches. What is left
    is what a nav bar cannot do."""
    page = (STATIC / "start.html").read_text()
    for gone in ("'main-links'", "'participant-links'", "'data-links'"):
        assert gone not in page, gone
    assert "'diag-links'" in page
    assert 'id="nav"' in page


def test_join_codes_cover_both_arms_and_always_reset_the_device():
    """One handset has to be able to rehearse either arm, and it cannot do
    that while it still holds the identity from the last run."""
    page = (STATIC / "start.html").read_text()
    for path in ("/?newid=1", "/?newid=1&as=tap", "/?newid=1&as=gaze"):
        assert path in page, path
    assert page.count("newid=1") >= 3


def test_a_join_code_can_only_point_back_at_this_server(env):
    """Otherwise it is a service for generating QR codes to anywhere."""
    with start(with_image(env)) as client:
        assert client.get("/api/qr.svg",
                          params={"path": "/?newid=1"}).status_code == 200
        for bad in ("https://evil.example", "//evil.example",
                    "javascript:alert(1)"):
            assert client.get("/api/qr.svg",
                              params={"path": bad}).status_code == 400, bad


def test_a_forced_assignment_is_recorded_as_forced(env):
    """A code that names its condition skips the balancer. Those are
    rehearsals, and the balance has to be readable afterwards without them
    being mistaken for participants the balancer chose."""
    # The lifespan builds the schema, so the client has to be entered.
    with start(with_image(env)) as client:
        r = client.post("/api/assign",
                        json={"participant_uuid": "rehearse-0001",
                              "force": "gaze"}).json()
        assert r["condition"] == "gaze"
        assert r["forced"] is True

        counts = client.get("/api/conditions").json()["counts"]
        assert counts["gaze"]["forced"] == 1
        assert counts["tap"]["forced"] == 0

        # A normal join is not marked.
        n = client.post("/api/assign",
                        json={"participant_uuid": "ordinary-0001"}).json()
        assert n["forced"] is False
        # An unknown condition is refused rather than silently ignored.
        assert client.post(
            "/api/assign", json={"participant_uuid": "bad-00001",
                                 "force": "elsewhere"}).status_code == 422
