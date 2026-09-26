"""The four queued items from docs/SCOPE.md section 2, once built.

Each test names the failure it prevents rather than the function it calls —
these are all features whose wrong version still renders something
plausible, which is the hard kind to notice.
"""

import importlib
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


def tapper(c, uuid, pts):
    rid = c.get("/api/state").json()["round_id"]
    c.post("/api/assign", json={"participant_uuid": uuid, "force": "tap"})
    c.post("/api/markers", json={"round_id": rid, "participant_uuid": uuid,
                                 "points": pts})


def gazer(c, uuid, samples, grade="usable"):
    c.post("/api/assign", json={"participant_uuid": uuid, "force": "gaze"})
    sid = c.post("/api/gaze/session", json={
        "participant_uuid": uuid, "tracker": "webeyetrack", "grade": grade,
        "mean_error": 0.18, "points_accepted": 9, "points_total": 9,
    }).json()["session_id"]
    c.post("/api/gaze/samples", json={"session_id": sid, "samples": samples})
    return sid


def on(t, x, y):
    return {"t_ms": t, "x": x, "y": y, "on_image": True}


def off(t):
    return {"t_ms": t, "x": None, "y": None, "on_image": False}


# --- 2.2 off-image gaze is reported, not discarded -------------------------

def test_gaze_that_missed_the_picture_is_counted_not_deleted(env):
    """It used to be filtered out in SQL, which made it invisible to
    everything downstream. A sample with a null coordinate is a real reading
    of someone looking at the letterboxing or the room — in development some
    sessions lost 35-73% of their data with no number anywhere saying so."""
    with start(env) as c:
        gazer(c, "gazer-0001", [on(0, .4, .4), off(300), off(600),
                                on(900, .6, .5)])
        d = c.get("/api/compare/maps").json()
        o = d["gaze_off_image"]
        assert o["samples"] == 4
        assert o["off"] == 2
        assert o["fraction"] == 0.5
        assert o["points_per_participant"] == [2]
        assert o["per_participant"] == [2]


def test_a_session_that_never_hit_the_picture_is_reported(env):
    """It has no map to contribute, and that is worth seeing: it usually
    means the calibration drifted off the screen entirely. Silently absent
    is the one thing it must not be."""
    with start(env) as c:
        gazer(c, "blind-0001", [off(0), off(300), off(600)])
        gazer(c, "seeing-001", [on(0, .5, .5), on(300, .5, .5)])
        o = c.get("/api/compare/maps").json()["gaze_off_image"]
        assert o["no_on_image_sessions"] == 1
        # And the one that did see the picture still contributes normally.
        assert o["points_per_participant"] == [2]


def test_the_off_picture_count_covers_every_sample_not_the_survivors(env):
    with start(env) as c:
        gazer(c, "gazer-0001", [on(0, .5, .5)] + [off(t) for t in (1, 2, 3)])
        o = c.get("/api/compare/maps").json()["gaze_off_image"]
        assert (o["off"], o["samples"]) == (3, 4)
        assert round(o["fraction"], 3) == 0.75


# --- 2.4 a round other than the live one -----------------------------------

def test_a_past_round_can_be_asked_for_by_name(env):
    """Every comparison endpoint was hard-wired to the open round, so a past
    sitting was unreachable even though the data was right there."""
    with start(env) as c:
        tapper(c, "first-0001", [[0.1, 0.1], [0.2, 0.2]])
        first = c.get("/api/state").json()["round_id"]
        c.post("/api/control/reset-round", headers=AUTH)
        tapper(c, "second-001", [[0.9, 0.9], [0.8, 0.8]])

        live = c.get("/api/compare/maps").json()
        past = c.get(f"/api/compare/maps?round={first}").json()
        assert live["round"]["id"] != first
        assert past["round"]["id"] == first
        # Different people looked at different corners, so the maps differ.
        assert past["maps"]["tapped"]["cells"] != live["maps"]["tapped"]["cells"]


def test_asking_for_a_round_that_is_not_there_says_so(env):
    with start(env) as c:
        d = c.get("/api/compare/maps?round=999").json()
        assert d["ok"] is False and "999" in d["reason"]


def test_a_round_is_listed_with_what_makes_it_comparable(env):
    """Two sittings of the same image differing only in the model run behind
    them are not the same experiment, and a list showing only a timestamp
    makes them look identical — which is the pair most likely to be compared
    by mistake."""
    with start(env) as c:
        tapper(c, "someone-01", [[0.5, 0.5]])
        r = c.get("/api/rounds").json()["rounds"][0]
        assert set(r["params"]) == {"task", "target", "n_fixations",
                                    "view_ms", "tap_count"}
        assert "freeview" in r["signature"]
        assert "n" in r["signature"]          # the fixation count


def test_a_past_round_is_compared_against_the_model_it_ran_with(env):
    """A round carries the settings it ran under. Comparing it against
    whatever the control page is set to now would silently put last week's
    people beside this week's model and report it as a result."""
    with start(env) as c:
        tapper(c, "past-00001", [[0.5, 0.5]])
        first = c.get("/api/state").json()["round_id"]
        c.post("/api/control/reset-round", headers=AUTH)
        # The presenter moves on to a different fixation count.
        c.post("/api/control/model-config", json={"n_fixations": 10},
               headers=AUTH)
        past = c.get(f"/api/compare/maps?round={first}").json()
        assert past["model_used"]["n_fixations"] == 5
        assert c.get("/api/compare/maps").json()["model_used"]["n_fixations"] == 10


# --- 2.5 time ---------------------------------------------------------------

def test_a_path_splits_into_bins_by_position_not_by_clock():
    """The model has no clock and the people have no fixation count. What
    both have is a proportion through their own looking."""
    from app import analysis
    p = [[i / 10, i / 10] for i in range(9)]
    bins = analysis.split_by_rank(p, 3)
    assert [len(b) for b in bins] == [3, 3, 3]
    assert bins[0][0] == [0.0, 0.0]
    assert bins[2][-1] == [0.8, 0.8]


def test_the_last_point_of_a_path_is_not_lost_off_the_end():
    """i*bins/n lands the final point in bin `bins`, one past the end."""
    from app import analysis
    for n in range(1, 12):
        p = [[i / n, 0.5] for i in range(n)]
        bins = analysis.split_by_rank(p, 3)
        assert sum(len(b) for b in bins) == n, n


def test_a_path_shorter_than_the_bins_leaves_bins_empty(env):
    """Padding it would put attention where nobody looked."""
    from app import analysis
    bins = analysis.split_by_rank([[0.1, 0.1], [0.9, 0.9]], 3)
    assert [len(b) for b in bins] == [1, 1, 0]


def test_temporal_tracks_agreement_changing_over_a_path():
    """The claim this exists to support: they agree at one end and part
    company at the other. A pooled map cannot say it."""
    from app import analysis
    walk = [[0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [0.9, 0.9]]
    fixed_end = [[0.9, 0.9]] * 5
    r = analysis.temporal_maps({"tapped": [walk, walk],
                                "model": [fixed_end]}, n=3)
    ccs = [f["pairs"]["tapped|model"] for f in r["frames"]]
    assert len(ccs) == 3
    assert ccs[0] < ccs[-1], f"agreement should rise along the path: {ccs}"


def test_the_temporal_endpoint_answers_for_a_named_round(env):
    with start(env) as c:
        tapper(c, "walker-001", [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
        rid = c.get("/api/state").json()["round_id"]
        d = c.get(f"/api/compare/temporal?round={rid}").json()
        assert d["ok"] is True
        assert d["bins"] == 3
        assert [f["label"] for f in d["frames"]] == [
            "First third", "Middle third", "Last third"]
        assert d["round"]["id"] == rid


def test_temporal_bins_are_clamped(env):
    with start(env) as c:
        tapper(c, "walker-001", [[0.1, 0.1], [0.9, 0.9]])
        assert c.get("/api/compare/temporal?bins=99").json()["bins"] == 4
        assert c.get("/api/compare/temporal?bins=0").json()["bins"] == 2


# --- 2.3 the controls live on the screen they affect -----------------------

def test_the_display_carries_its_own_layer_bar():
    page = (STATIC / "display.html").read_text()
    assert 'id="deck"' in page
    assert "/api/control/layers" in page
    # Every layer reachable, including the two that take over the screen.
    for key in ("grid", "heatmap", "paths", "gaze", "gaze_paths", "model",
                "prompt", "charts", "json"):
        assert f"key: '{key}'" in page, key


def test_the_layer_bar_sits_above_the_full_screen_layers():
    """It was at z-index 40 under overlays at 55 and 60, so its buttons were
    unclickable the moment charts or raw JSON went up — precisely the
    situation it exists to get out of. Caught by a click that could not land."""
    css = (STATIC / "display.css").read_text()
    deck = css[css.index(".deck {"):]
    deck = deck[:deck.index("}")]
    z = int(deck.split("z-index:")[1].split(";")[0].strip())
    others = [int(b.split(";")[0].strip())
              for b in css.split("z-index:")[1:]]
    assert z >= max(others), f"deck z-index {z} is not above {max(others)}"


def test_the_layer_bar_hides_itself():
    """This is the audience's screen. Controls parked on it for the whole
    talk are in the photograph of the talk."""
    page = (STATIC / "display.html").read_text()
    assert "wakeDeck" in page
    for ev in ("pointermove", "keydown"):
        assert ev in page, ev
    css = (STATIC / "display.css").read_text()
    assert ".deck.show" in css
    deck = css[css.index(".deck {"):]
    assert "opacity: 0" in deck[:deck.index("}")]


def test_the_control_page_no_longer_owns_the_layers():
    """Run setup only. Two places to change the same thing is two places to
    look when it is wrong."""
    page = (STATIC / "control.html").read_text()
    assert "LAYER_GROUPS" not in page
    assert 'id="toggles"' not in page
    assert "/api/control/layers" not in page
    # And it still points at where they went.
    assert 'id="open-display"' in page


def test_the_display_link_carries_the_token():
    """The bar posts layer changes. Without a token a browser that has never
    opened /control gets a row of buttons that all answer 'needs token'."""
    assert "'/display?k='" in (STATIC / "control.html").read_text()
    assert "token: true" in (STATIC / "nav.js").read_text()


def test_the_projected_legend_names_things_rather_than_explaining_them():
    """It used to describe the encoding — 'fewer to more', 'first tap ringed
    pale'. That is a key, and a key is something you study; at projector
    distance nobody studies anything."""
    page = (STATIC / "display.html").read_text()
    block = page[page.index("function renderLegend"):]
    block = block[:block.index("\n}")]
    assert "ringed pale" not in block
    assert "Tapped &mdash; predicted" in block
    assert "Measured &mdash; eye tracked" in block
