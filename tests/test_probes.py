"""Tests for the probe catalogue, prompt display, and the raw-data endpoint."""

import importlib
import json
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import gaze_prompts as gp  # noqa: E402

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


# --- catalogue -------------------------------------------------------------

def test_every_probe_has_the_fields_the_ui_needs():
    for p in gp.PROBES:
        assert {"id", "label", "kind", "mode", "target", "note"} <= set(p)
        assert p["kind"] in ("trained", "experimental")
        assert p["mode"] in ("freeview", "search", "probe")
        if p["mode"] == "probe":
            assert p["phrase"], f"{p['id']} needs a task phrase"


def test_probe_ids_are_unique():
    ids = [p["id"] for p in gp.PROBES]
    assert len(ids) == len(set(ids))


def test_only_genuinely_trained_probes_claim_to_be():
    """Mislabelling an untrained probe as trained would mislead the audience."""
    for p in gp.PROBES:
        if p["kind"] != "trained":
            continue
        assert p["mode"] in ("freeview", "search")
        if p["mode"] == "search":
            assert p["target"] in gp.COCO_SEARCH18_TARGETS, \
                f"{p['id']} claims trained but {p['target']!r} is not a COCO-Search18 target"


def test_experimental_probes_are_not_disguised_search_targets():
    for p in gp.PROBES:
        if p["kind"] == "experimental":
            assert p["target"] not in gp.COCO_SEARCH18_TARGETS


def test_probe_prompts_keep_the_output_format_scaffolding():
    """A bare instruction would not yield coordinates at all."""
    for p in gp.PROBES:
        prompt = gp.build_prompt(p["mode"], 5, p["target"])
        assert "exactly 5 fixation points" in prompt
        assert "Output ONLY a Python list of tuples" in prompt
        assert "(x, y)" in prompt
        assert "<image>" not in prompt


def test_probe_prompt_contains_its_task_phrase():
    for p in gp.PROBES:
        if p["mode"] == "probe":
            assert p["phrase"] in gp.build_prompt("probe", 5, p["target"])


def test_probe_prompts_differ_from_each_other():
    prompts = {gp.build_prompt(p["mode"], 5, p["target"]) for p in gp.PROBES}
    assert len(prompts) == len(gp.PROBES)


def test_unknown_probe_raises():
    with pytest.raises(ValueError):
        gp.build_prompt("probe", 5, "nonexistent")


def test_the_requested_probes_all_exist():
    for wanted in ["unexpected", "people", "cars", "roads", "count_buildings",
                   "living", "danger", "music", "robots"]:
        assert wanted in gp.PROBES_BY_ID


# --- API -------------------------------------------------------------------

def test_probes_endpoint_is_public(client):
    r = client.get("/api/probes")
    assert r.status_code == 200
    assert len(r.json()["probes"]) == len(gp.PROBES)


def test_selecting_a_probe_sets_mode_and_target(client):
    cfg = client.post("/api/control/model-config", json={"probe": "danger"},
                      headers=AUTH).json()["config"]
    assert cfg["mode"] == "probe" and cfg["target"] == "danger"
    assert cfg["probe"]["kind"] == "experimental"


def test_config_exposes_the_prompt_text(client):
    cfg = client.post("/api/control/model-config", json={"probe": "cars"},
                      headers=AUTH).json()["config"]
    assert "searching for a car" in cfg["prompt_text"]

    cfg = client.post("/api/control/model-config", json={"probe": "freeview"},
                      headers=AUTH).json()["config"]
    assert "free viewing for 3 seconds" in cfg["prompt_text"]


def test_unknown_probe_is_rejected(client):
    assert client.post("/api/control/model-config", json={"probe": "nope"},
                       headers=AUTH).status_code == 400


def test_probe_runs_resolve_to_their_own_run(client):
    for pid, coords in (("danger", [[0.1, 0.1]]), ("music", [[0.9, 0.9]])):
        p = gp.PROBES_BY_ID[pid]
        client.post("/api/model/push", json={
            "image": "a.jpg", "mode": p["mode"], "target": p["target"],
            "n_fixations": 5, "scanpath_norm": coords,
        }, headers=AUTH)

    client.post("/api/control/model-config", json={"probe": "danger"}, headers=AUTH)
    assert client.get("/api/model").json()["run"]["scanpath_norm"] == [[0.1, 0.1]]
    client.post("/api/control/model-config", json={"probe": "music"}, headers=AUTH)
    assert client.get("/api/model").json()["run"]["scanpath_norm"] == [[0.9, 0.9]]


def test_probe_mode_is_accepted_on_push(client):
    assert client.post("/api/model/push", json={
        "image": "a.jpg", "mode": "probe", "target": "robots",
        "n_fixations": 5, "scanpath_norm": [[0.5, 0.5]],
    }, headers=AUTH).status_code == 200


# --- layers and raw data ---------------------------------------------------

def test_prompt_and_json_layers_exist_and_default_off(client):
    layers = client.get("/api/state").json()["layers"]
    assert layers["prompt"] is False and layers["json"] is False


def test_prompt_and_json_layers_toggle(client):
    out = client.post("/api/control/layers", json={"prompt": True, "json": True},
                      headers=AUTH).json()["layers"]
    assert out["prompt"] is True and out["json"] is True


def test_raw_endpoint_returns_every_section(client):
    r = client.get("/api/raw")
    assert r.status_code == 200
    d = r.json()
    assert {"state", "model", "markers", "analysis"} <= set(d)


def test_raw_endpoint_is_public_and_leaks_no_token(client):
    body = json.dumps(client.get("/api/raw").json())
    assert "test-token" not in body
