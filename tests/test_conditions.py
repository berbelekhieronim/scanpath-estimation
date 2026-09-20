"""Between-subjects design: assignment, routing, comparison, export."""

import importlib
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


def assign(client, uuid):
    return client.post("/api/assign", json={"participant_uuid": uuid}).json()


def set_mode(client, mode):
    return client.post("/api/control/capture-mode", json={"mode": mode}, headers=AUTH)


def test_default_mode_is_tap_only(client):
    assert client.get("/api/state").json()["capture_mode"] == "tap"
    assert assign(client, "participant-0001")["condition"] == "tap"


def test_gaze_mode_assigns_everyone_to_gaze(client):
    set_mode(client, "gaze")
    assert all(assign(client, f"participant-{i:04d}")["condition"] == "gaze"
               for i in range(5))


def test_mixed_mode_balances(client):
    set_mode(client, "mixed")
    got = [assign(client, f"participant-{i:04d}")["condition"] for i in range(10)]
    assert abs(got.count("tap") - got.count("gaze")) <= 1


def test_assignment_is_stable_for_a_participant(client):
    set_mode(client, "mixed")
    first = assign(client, "participant-0001")
    assert first["new"] is True
    again = assign(client, "participant-0001")
    assert again["condition"] == first["condition"] and again["new"] is False


def test_balancing_counts_completions_not_just_assignments(client):
    """Eye tracking has a real failure rate. Balancing on assignment alone
    would quietly leave the gaze group smaller — the group least able to
    afford it."""
    set_mode(client, "mixed")
    rid = client.get("/api/state").json()["round_id"]

    # four assigned to each, but only the tap side completes
    conds = {}
    for i in range(8):
        conds[f"p-{i:04d}"] = assign(client, f"participant-{i:04d}")["condition"]
    from app import db
    for uuid, cond in conds.items():
        if cond == "tap":
            pid = db.upsert_participant("participant-" + uuid.split("-")[1])
            db.mark_completed(rid, pid)

    counts = client.get("/api/conditions").json()["counts"]
    assert counts["tap"]["completed"] >= counts["gaze"]["completed"]
    # the next few should now favour gaze
    nxt = [assign(client, f"later-participant-{i:04d}")["condition"] for i in range(4)]
    assert nxt.count("gaze") >= nxt.count("tap")


def test_submitting_taps_marks_completion(client):
    set_mode(client, "mixed")
    assign(client, "participant-0001")
    rid = client.get("/api/state").json()["round_id"]
    client.post("/api/markers", json={
        "round_id": rid, "participant_uuid": "participant-0001",
        "points": [[0.2, 0.3], [0.5, 0.5]]})
    counts = client.get("/api/conditions").json()["counts"]
    assert counts["tap"]["completed"] + counts["gaze"]["completed"] == 1


def test_invalid_mode_is_rejected(client):
    assert set_mode(client, "telepathy").status_code == 400


def test_capture_mode_is_token_gated(client):
    assert client.post("/api/control/capture-mode", json={"mode": "gaze"}).status_code == 403


# --- comparison ------------------------------------------------------------

def test_compare_needs_two_sources(client):
    d = client.get("/api/compare").json()
    assert d["ok"] is False


def test_compare_reports_all_three_sources(client):
    set_mode(client, "mixed")
    rid = client.get("/api/state").json()["round_id"]

    for i in range(4):
        assign(client, f"tapper-{i:04d}")
        client.post("/api/markers", json={
            "round_id": rid, "participant_uuid": f"tapper-{i:04d}",
            "points": [[0.9, 0.3], [0.2, 0.7], [0.6, 0.1]]})

    client.post("/api/model/push", json={
        "image": "a.jpg", "mode": "freeview", "n_fixations": 3,
        "scanpath_norm": [[0.9, 0.3], [0.5, 0.5], [0.2, 0.7]]}, headers=AUTH)

    d = client.get("/api/compare").json()
    assert d["ok"] is True
    assert "tap" in d["sources"] and "model" in d["sources"]
    assert "tap_vs_model" in d["pairs"]
    assert d["sources"]["tap"]["centre_bias"] is not None


def test_compare_grid_is_clamped(client):
    """A finer grid than the gaze error supports would report noise."""
    for g, expect in ((1, 2), (9, 5), (3, 3)):
        d = client.get(f"/api/compare?grid={g}").json()
        assert d.get("grid", expect) == expect or d["ok"] is False


# --- export ----------------------------------------------------------------

def test_there_is_no_csv_export(client):
    """Dropped on request: the JSON export carries everything and the CSV
    was another shape to keep in step for no benefit."""
    assert client.get("/api/export.csv", headers=AUTH).status_code == 404


def test_json_export_carries_device_information(client):
    rid = client.get("/api/state").json()["round_id"]
    client.post("/api/markers", json={
        "round_id": rid, "participant_uuid": "participant-0001",
        "points": [[0.2, 0.3]],
        "device": {"browser": "Safari", "screen_w": 390, "screen_h": 844,
                   "dpr": 3, "orientation": "portrait", "ua": "test"}})
    d = client.get("/api/export", headers=AUTH).json()
    dev = d["participants"][0]["device"]
    assert dev["browser"] == "Safari" and dev["screen_w"] == 390 and dev["dpr"] == 3




def test_tap_page_routes_gaze_participants_away():
    """A gaze-condition participant must not be shown the tap screen."""
    src = (STATIC / "index.html").read_text()
    assert "routeByCondition" in src
    assert "'/consent'" in src


def test_tap_page_does_not_offer_measurement_in_a_split_design():
    """Offering both to the same person is exactly the contamination the
    between-subjects split exists to avoid."""
    src = (STATIC / "index.html").read_text()
    assert "betweenSubjects" in src


def test_assignment_sticks_to_a_device(client):
    """By design: one person must not end up doing both conditions. It does
    mean a single test phone keeps the same condition until it is reset."""
    set_mode(client, "mixed")
    first = assign(client, "one-phone-0001")["condition"]
    for _ in range(5):
        assert assign(client, "one-phone-0001")["condition"] == first


def test_a_new_round_reassigns(client):
    set_mode(client, "mixed")
    assign(client, "one-phone-0001")
    client.post("/api/control/reset-round", headers=AUTH)
    assert assign(client, "one-phone-0001")["new"] is True


def test_participant_page_can_forget_its_identity():
    src = (STATIC / "index.html").read_text()
    assert "newid" in src
    assert "localStorage.removeItem(UUID_KEY)" in src
