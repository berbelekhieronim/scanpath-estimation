"""Phase 4 tests: model run loading, resolution, config, synthetic labelling."""

import importlib
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def make_run(image="a.jpg", mode="freeview", target=None, n=5,
             source="precomputed", samples=3):
    paths = [[[0.1 * (i + 1), 0.2 * (j + 1)] for j in range(n)] for i in range(samples)]
    return {
        "image": image, "mode": mode, "target": target, "n_fixations": n,
        "seed": 42, "temperature": 0.7, "prompt_text": "…",
        "scanpath_grid": [[int(x * 100), int(y * 100)] for x, y in paths[0]],
        "scanpath_norm": paths[0],
        "samples_norm": paths,
        "source": source, "model": "test-model",
        "created_at": "2026-09-19T00:00:00Z",
    }


@pytest.fixture()
def env(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("SCANPATH_DATA_DIR", tmp)
    monkeypatch.setenv("SCANPATH_CONTROL_TOKEN", "test-token")
    root = Path(tmp)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "model").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), "red").save(root / "images" / "a.jpg")
    Image.new("RGB", (640, 480), "blue").save(root / "images" / "b.png")
    return root


def start(env):
    from app import config, db, urls, main
    for mod in (config, db, urls, main):
        importlib.reload(mod)
    return TestClient(main.app)


AUTH = {"X-Control-Token": "test-token"}


def test_runs_on_disk_are_loaded_at_boot(env):
    (env / "model" / "a__freeview__n5.json").write_text(json.dumps(make_run()))
    with start(env) as c:
        d = c.get("/api/model").json()
        assert d["run"] is not None
        assert d["run"]["n_fixations"] == 5
        assert len(d["run"]["samples_norm"]) == 3


def test_no_run_is_a_normal_state_not_an_error(env):
    with start(env) as c:
        d = c.get("/api/model").json()
        assert d["run"] is None
        assert "reason" in d
        assert c.get("/api/model").status_code == 200


def test_run_for_a_missing_image_is_reported_orphaned(env):
    (env / "model" / "ghost.json").write_text(json.dumps(make_run(image="ghost.jpg")))
    with start(env) as c:
        r = c.post("/api/control/rescan-model", headers=AUTH).json()["result"]
        assert r["orphaned"] == ["ghost.json"]


def test_malformed_run_files_are_reported_not_fatal(env):
    (env / "model" / "bad.json").write_text("{ not json")
    (env / "model" / "incomplete.json").write_text(json.dumps({"image": "a.jpg"}))
    (env / "model" / "good.json").write_text(json.dumps(make_run()))
    with start(env) as c:
        r = c.post("/api/control/rescan-model", headers=AUTH).json()["result"]
        assert sorted(r["invalid"]) == ["bad.json", "incomplete.json"]
        assert r["loaded"] == ["good.json"], "a bad file must not block a good one"


def test_config_selects_which_run_is_shown(env):
    (env / "model" / "fv.json").write_text(json.dumps(make_run(mode="freeview")))
    (env / "model" / "se.json").write_text(
        json.dumps(make_run(mode="search", target="car")))
    with start(env) as c:
        assert c.get("/api/model").json()["run"]["mode"] == "freeview"
        c.post("/api/control/model-config",
               json={"mode": "search", "target": "car"}, headers=AUTH)
        run = c.get("/api/model").json()["run"]
        assert run["mode"] == "search" and run["target"] == "car"


def test_search_run_is_not_returned_for_the_wrong_target(env):
    (env / "model" / "se.json").write_text(
        json.dumps(make_run(mode="search", target="car")))
    with start(env) as c:
        c.post("/api/control/model-config",
               json={"mode": "search", "target": "laptop"}, headers=AUTH)
        assert c.get("/api/model").json()["run"] is None


def test_a_run_at_another_fixation_count_still_resolves(env):
    """Better to show an 8-fixation run than nothing when 5 was requested."""
    (env / "model" / "n8.json").write_text(json.dumps(make_run(n=8)))
    with start(env) as c:
        c.post("/api/control/model-config", json={"n_fixations": 5}, headers=AUTH)
        run = c.get("/api/model").json()["run"]
        assert run is not None and run["n_fixations"] == 8


def test_rerunning_a_config_replaces_rather_than_duplicates(env):
    (env / "model" / "a.json").write_text(json.dumps(make_run(samples=3)))
    with start(env) as c:
        (env / "model" / "a.json").write_text(json.dumps(make_run(samples=7)))
        c.post("/api/control/rescan-model", headers=AUTH)
        assert len(c.get("/api/model/runs").json()["runs"]) == 1
        assert len(c.get("/api/model").json()["run"]["samples_norm"]) == 7


def test_synthetic_source_is_preserved_all_the_way_through(env):
    """The UI badge depends on this surviving the round trip."""
    (env / "model" / "s.json").write_text(json.dumps(make_run(source="synthetic")))
    with start(env) as c:
        assert c.get("/api/model").json()["run"]["source"] == "synthetic"


def test_invalid_mode_is_rejected(env):
    with start(env) as c:
        assert c.post("/api/control/model-config", json={"mode": "nonsense"},
                      headers=AUTH).status_code == 400


def test_model_config_is_token_gated(env):
    with start(env) as c:
        assert c.post("/api/control/model-config", json={"mode": "search"}).status_code == 403


def test_model_endpoint_is_public(env):
    with start(env) as c:
        assert c.get("/api/model").status_code == 200


def test_model_layer_defaults_off(env):
    with start(env) as c:
        assert c.get("/api/state").json()["layers"]["model"] is False


# --- prompt fidelity -------------------------------------------------------

def test_prompt_templates_and_parser():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import gaze_prompts as gp

    p = gp.build_freeview_prompt(5)
    assert "exactly 5 fixation points" in p
    assert "free viewing for 3 seconds" in p
    assert "<image>" not in p, "the chat template adds the image, not the prompt"

    s = gp.build_search_prompt("car", 3)
    assert "searching for a car" in s and "exactly 3 fixation points" in s

    assert gp.parse_scanpath("[(51,46),(38,28)]") == [(51, 46), (38, 28)]
    assert gp.parse_scanpath("[(1,2,300),(4,5,600)]") == [(1, 2), (4, 5)]
    assert gp.parse_scanpath("[(150,200)]") == [(99, 99)], "clipped to the grid"
    assert gp.parse_scanpath("no coordinates here") == []
    assert gp.grid_to_norm([(50, 25)]) == [[0.5, 0.25]]
    assert len(gp.COCO_SEARCH18_TARGETS) == 18
