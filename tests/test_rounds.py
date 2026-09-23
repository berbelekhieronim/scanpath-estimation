"""Rounds as portable records: export, import, and putting one back on screen.

The thing being defended here is a restore that quietly loses or clobbers
data. A demo snapshot is insurance, and insurance that silently pays out the
wrong amount is worse than none.
"""

import copy
import importlib
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
AUTH = {"X-Control-Token": "test-token"}


@pytest.fixture()
def env(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("SCANPATH_DATA_DIR", tmp)
    monkeypatch.setenv("SCANPATH_CONTROL_TOKEN", "test-token")
    root = Path(tmp)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "model").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), "red").save(root / "images" / "a.jpg")
    return root


def start(env):
    from app import config, db, urls, rounds, main
    for mod in (config, db, urls, rounds, main):
        importlib.reload(mod)
    return TestClient(main.app)


def collect(client, tapper="tapper-0001", gazer="gazer-0002"):
    """One tapper and one eye-tracked participant, as a round would have."""
    round_id = client.get("/api/state").json()["round_id"]
    client.post("/api/assign", json={"participant_uuid": tapper, "force": "tap"})
    client.post("/api/markers", json={
        "round_id": round_id, "participant_uuid": tapper,
        "points": [[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]]})

    client.post("/api/assign", json={"participant_uuid": gazer, "force": "gaze"})
    sid = client.post("/api/gaze/session", json={
        "participant_uuid": gazer, "tracker": "webeyetrack", "grade": "good",
        "mean_error": 0.05, "points_accepted": 4, "points_total": 4,
    }).json()["session_id"]
    client.post("/api/gaze/samples", json={"session_id": sid, "samples": [
        {"t_ms": 0, "x": 0.4, "y": 0.4, "on_image": True},
        {"t_ms": 300, "x": 0.6, "y": 0.5, "on_image": True},
        {"t_ms": 600, "x": None, "y": None, "on_image": False},
    ]})
    return round_id


# --- the round trip --------------------------------------------------------

def test_a_round_survives_a_round_trip(env):
    """Taps, gaze, calibration grades and group assignments all come back."""
    with start(env) as c:
        collect(c)
        snap = c.get("/api/rounds/1/export", headers=AUTH).json()
        assert snap["counts"] == {
            "participants": 2, "markers": 3, "assignments": 2,
            "gaze_sessions": 1, "gaze_samples": 3, "model_runs": 0}

        out = c.post("/api/rounds/import", json=snap, headers=AUTH).json()
        new_id = out["results"][0]["round_id"]
        assert out["results"][0]["imported"] is True
        assert new_id != 1

        c.post(f"/api/rounds/{new_id}/activate", headers=AUTH)
        assert c.get("/api/markers").json()["paths"] == [
            [[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]]]

        cmp = c.get("/api/compare").json()
        assert cmp["sources"]["tap"]["n_participants"] == 1
        assert cmp["sources"]["gaze"]["n_participants"] == 1
        assert cmp["conditions"]["gaze"]["assigned"] == 1

        listed = {r["id"]: r for r in c.get("/api/rounds").json()["rounds"]}
        assert listed[new_id]["gaze_usable"] == 1
        assert listed[new_id]["markers"] == 3


def test_off_image_gaze_samples_are_not_quietly_dropped(env):
    """A sample that landed off the picture is evidence about the tracking,
    not noise to tidy away — the usable fraction is computed from it."""
    with start(env) as c:
        collect(c)
        snap = c.get("/api/rounds/1/export", headers=AUTH).json()
        kept = snap["gaze_sessions"][0]["samples"]
        assert len(kept) == 3
        assert [s["on_image"] for s in kept] == [1, 1, 0]


def test_importing_never_overwrites_the_live_round(env):
    """The failure this guards is losing tonight's data restoring last week's."""
    with start(env) as c:
        collect(c)
        snap = c.get("/api/rounds/1/export", headers=AUTH).json()

        c.post("/api/control/reset-round", headers=AUTH)
        live = c.get("/api/state").json()["round_id"]
        collect(c, tapper="tonight-0001", gazer="tonight-0002")

        c.post("/api/rounds/import", json=snap, headers=AUTH)

        assert c.get("/api/state").json()["round_id"] == live
        assert c.get("/api/markers").json()["count"] == 1
        rounds = c.get("/api/rounds").json()["rounds"]
        assert [r["id"] for r in rounds if r["active"]] == [live]
        assert len(rounds) == 3


def test_importing_the_same_file_twice_does_nothing(env):
    """Two clicks on a file picker should not produce two of the same sitting."""
    with start(env) as c:
        collect(c)
        snap = c.get("/api/rounds/1/export", headers=AUTH).json()
        first = c.post("/api/rounds/import", json=snap, headers=AUTH).json()
        second = c.post("/api/rounds/import", json=snap, headers=AUTH).json()
        assert second["results"][0]["imported"] is False
        assert second["results"][0]["round_id"] == first["results"][0]["round_id"]
        assert len(c.get("/api/rounds").json()["rounds"]) == 2

        # Unless you say you meant it.
        again = c.post("/api/rounds/import?duplicate=true", json=snap,
                       headers=AUTH).json()
        assert again["results"][0]["imported"] is True
        assert len(c.get("/api/rounds").json()["rounds"]) == 3


def test_a_participant_seen_before_is_not_cloned(env):
    """The same phone across two sittings is one person, or the balance
    counts and the per-participant maps both start double-counting them."""
    with start(env) as c:
        collect(c)
        snap = c.get("/api/rounds/1/export", headers=AUTH).json()
        c.post("/api/rounds/import", json=snap, headers=AUTH)
        with __import__("app.db", fromlist=["db"]).connect() as conn:
            n = conn.execute("SELECT COUNT(*) AS n FROM participants").fetchone()["n"]
        assert n == 2


# --- refusals --------------------------------------------------------------

def test_a_newer_snapshot_is_refused_rather_than_half_read(env):
    """Importing a shape this server does not understand would land partial
    data that looks complete."""
    with start(env) as c:
        collect(c)
        snap = c.get("/api/rounds/1/export", headers=AUTH).json()
        future = copy.deepcopy(snap)
        future["schema"] = snap["schema"] + 1
        r = c.post("/api/rounds/import", json=future, headers=AUTH)
        assert r.status_code == 422
        assert "newer" in r.json()["detail"]
        assert len(c.get("/api/rounds").json()["rounds"]) == 1


def test_arbitrary_json_is_not_a_round(env):
    with start(env) as c:
        r = c.post("/api/rounds/import", json={"hello": "world"}, headers=AUTH)
        assert r.status_code == 422


def test_export_and_import_need_the_token(env):
    """Listing rounds is public like every other read. Moving them is not."""
    with start(env) as c:
        collect(c)
        assert c.get("/api/rounds").status_code == 200
        assert c.get("/api/rounds/1/export").status_code == 403
        assert c.get("/api/rounds/export-all").status_code == 403
        assert c.post("/api/rounds/import", json={}).status_code == 403
        assert c.post("/api/rounds/1/activate").status_code == 403


def test_exporting_a_round_that_is_not_there(env):
    with start(env) as c:
        assert c.get("/api/rounds/999/export", headers=AUTH).status_code == 404
        assert c.post("/api/rounds/999/activate",
                      headers=AUTH).status_code == 404


# --- the missing picture ---------------------------------------------------

def test_a_snapshot_whose_image_is_absent_still_imports(env):
    """The responses are the irreplaceable half. Refusing them because the
    JPEG is on the other laptop would throw away the part that matters."""
    with start(env) as c:
        collect(c)
        snap = c.get("/api/rounds/1/export", headers=AUTH).json()
        snap["image"]["filename"] = "elsewhere.jpg"
        snap["exported_at"] = "2026-01-01T00:00:00+00:00"

        out = c.post("/api/rounds/import", json=snap, headers=AUTH).json()
        res = out["results"][0]
        assert res["imported"] is True
        assert res["counts"]["markers"] == 3
        assert any("not in data/images" in w for w in res["warnings"])

        # And the placeholder does not turn up as something to present.
        assert [i["filename"] for i in c.get("/api/images").json()["images"]] \
            == ["a.jpg"]


def test_a_resized_image_is_flagged_not_silently_accepted(env):
    """Normalised coordinates mean something only against the frame they were
    recorded in. A different crop is a different experiment."""
    with start(env) as c:
        collect(c)
        snap = c.get("/api/rounds/1/export", headers=AUTH).json()
        snap["image"]["width"] = 1600
        snap["exported_at"] = "2026-01-01T00:00:00+00:00"
        res = c.post("/api/rounds/import", json=snap,
                     headers=AUTH).json()["results"][0]
        assert any("1600" in w for w in res["warnings"])


# --- settings --------------------------------------------------------------

def test_a_round_remembers_the_settings_it_ran_under(env):
    """Restoring the data but not the state it was collected in gives you a
    round that reads wrong: five-second views scored as if they were taps."""
    with start(env) as c:
        c.post("/api/control/view-ms", json={"ms": 9000}, headers=AUTH)
        c.post("/api/control/capture-mode", json={"mode": "gaze"}, headers=AUTH)
        collect(c)
        snap = c.get("/api/rounds/1/export", headers=AUTH).json()
        assert snap["settings"]["view_ms"] == "9000"
        assert snap["settings"]["capture_mode"] == "gaze"

        # The presenter moves on, then restores the round.
        c.post("/api/control/view-ms", json={"ms": 3000}, headers=AUTH)
        c.post("/api/control/capture-mode", json={"mode": "tap"}, headers=AUTH)
        new_id = c.post("/api/rounds/import", json=snap,
                        headers=AUTH).json()["results"][0]["round_id"]
        c.post(f"/api/rounds/{new_id}/activate", headers=AUTH)

        state = c.get("/api/state").json()
        assert state["view_ms"] == 9000
        assert state["capture_mode"] == "gaze"


def test_settings_are_frozen_when_a_round_closes(env):
    """After a round closes the live settings describe the next one, so its
    own have to be written down at the moment it stops being current."""
    with start(env) as c:
        c.post("/api/control/view-ms", json={"ms": 8000}, headers=AUTH)
        collect(c)
        c.post("/api/control/reset-round", headers=AUTH)
        c.post("/api/control/view-ms", json={"ms": 2000}, headers=AUTH)

        old = c.get("/api/rounds/1/export", headers=AUTH).json()
        assert old["settings"]["view_ms"] == "8000"


def test_an_unknown_layer_in_a_snapshot_is_ignored(env):
    """A snapshot is a file off somebody's disk; it does not get to invent
    app_state keys."""
    with start(env) as c:
        collect(c)
        snap = c.get("/api/rounds/1/export", headers=AUTH).json()
        snap["settings"]["layers"]["nonsense"] = "1"
        snap["exported_at"] = "2026-01-01T00:00:00+00:00"
        rid = c.post("/api/rounds/import", json=snap,
                     headers=AUTH).json()["results"][0]["round_id"]
        c.post(f"/api/rounds/{rid}/activate", headers=AUTH)
        assert "nonsense" not in c.get("/api/state").json()["layers"]


# --- the whole machine -----------------------------------------------------

def test_everything_can_be_carried_in_one_file(env):
    """A container is deleted after its retention period. One file has to be
    enough to stand the demo back up somewhere else."""
    with start(env) as c:
        collect(c)
        c.post("/api/control/reset-round", headers=AUTH)
        collect(c, tapper="second-0001", gazer="second-0002")

        bundle = c.get("/api/rounds/export-all", headers=AUTH).json()
        assert bundle["counts"]["rounds"] == 2
        assert bundle["counts"]["markers"] == 6

    # A different machine, empty database, same file.
    with start(env) as fresh:
        with __import__("app.db", fromlist=["db"]).connect() as conn:
            conn.execute("DELETE FROM gaze_samples")
            conn.execute("DELETE FROM gaze_sessions")
            conn.execute("DELETE FROM markers")
            conn.execute("DELETE FROM assignments")
            conn.execute("DELETE FROM rounds")
        out = fresh.post("/api/rounds/import", json=bundle, headers=AUTH).json()
        assert out["bundle"] is True and out["imported"] == 2
        restored = fresh.get("/api/rounds").json()["rounds"]
        assert sum(r["markers"] for r in restored) == 6
        assert sum(r["gaze_sessions"] for r in restored) == 2


def test_the_filename_carries_the_date(env):
    """A folder of these has to sort and be identifiable without opening one."""
    with start(env) as c:
        collect(c)
        cd = c.get("/api/rounds/1/export",
                   headers=AUTH).headers["content-disposition"]
        assert cd.startswith("attachment; ")
        name = cd.split('filename="')[1].rstrip('"')
        assert name.startswith("round-1-a-") and name.endswith(".json")
        # YYYYMMDDTHHMMSS
        stamp = name[len("round-1-a-"):-len(".json")]
        assert len(stamp) == 15 and stamp[8] == "T" and stamp[:8].isdigit()


# --- the page --------------------------------------------------------------

def test_the_rounds_page_is_its_own_page_and_is_reachable(env):
    with start(env) as c:
        body = c.get("/rounds").text
        assert c.get("/rounds").status_code == 200
        assert "<title>Rounds" in body
    nav = (STATIC / "nav.js").read_text()
    assert "'/rounds'" in nav


def test_the_page_downloads_through_the_token_header(env):
    """A plain link cannot carry a header, so a naive <a href> would just
    render a 403 page where a file was expected."""
    page = (STATIC / "rounds.html").read_text()
    assert "createObjectURL" in page
    assert "X-Control-Token" in page
