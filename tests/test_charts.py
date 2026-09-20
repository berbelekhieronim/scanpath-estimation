"""The comparison charts: per-participant maps, the endpoint, and the page.

The assertions that matter here are the ones with a known right answer. A
group that all looked at the same cell must put its mass there; two groups
that looked at opposite corners must not correlate; and one loud participant
must not be able to outvote the rest, because the unit of observation is the
person (SPEC-METRICS.md section 1).
"""

import importlib
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import analysis as A

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
MODULE = STATIC / "charts.js"
AUTH = {"X-Control-Token": "test-token"}

TOP_LEFT = [[0.12, 0.12], [0.16, 0.10], [0.10, 0.17]]
BOTTOM_RIGHT = [[0.88, 0.88], [0.84, 0.90], [0.90, 0.83]]
CENTRE = [[0.50, 0.50], [0.47, 0.52], [0.53, 0.49]]


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


def tap(client, uuid, points):
    rid = client.get("/api/state").json()["round_id"]
    return client.post("/api/markers", json={
        "round_id": rid, "participant_uuid": uuid, "points": points})


def gaze(client, uuid, points):
    client.post("/api/gaze/session",
                json={"participant_uuid": uuid, "grade": "usable"})
    return client.post("/api/gaze/samples", json={
        "participant_uuid": uuid,
        "samples": [{"x": x, "y": y, "t": i * 100}
                    for i, (x, y) in enumerate(points)]})


# --- the maths ------------------------------------------------------------

def test_each_participant_map_sums_to_one():
    maps = A.participant_maps([TOP_LEFT, BOTTOM_RIGHT, CENTRE])
    assert len(maps) == 3
    assert np.allclose(maps.sum(axis=1), 1.0)


def test_mass_lands_in_the_cell_that_was_looked_at():
    # Cell 0 is top-left, cell 8 bottom-right, on a row-major 3x3 grid.
    top = A.participant_maps([TOP_LEFT])[0]
    assert top.argmax() == 0
    assert A.participant_maps([BOTTOM_RIGHT])[0].argmax() == 8
    assert A.participant_maps([CENTRE])[0].argmax() == 4


def test_one_prolific_participant_cannot_outvote_the_group():
    """The whole reason maps are built per person, not from pooled points."""
    loud = TOP_LEFT * 12          # 36 points from one person
    quiet = [BOTTOM_RIGHT, BOTTOM_RIGHT, BOTTOM_RIGHT]   # 3 people, 9 points
    group = A.group_map([loud] + quiet)
    assert group["n"] == 4
    # Three people beat one, however much data the one produced.
    assert group["cells"][8] > group["cells"][0]


def test_opposite_corners_do_not_correlate():
    a = A.group_map([TOP_LEFT] * 5)["cells"]
    b = A.group_map([BOTTOM_RIGHT] * 5)["cells"]
    assert A.correlate(a, b) < 0


def test_identical_groups_correlate_perfectly():
    a = A.group_map([TOP_LEFT, CENTRE] * 3)["cells"]
    assert A.correlate(a, a) == pytest.approx(1.0)


def test_intervals_bracket_the_mean_and_widen_with_disagreement():
    agree = A.group_map([CENTRE] * 6)
    differ = A.group_map([TOP_LEFT, BOTTOM_RIGHT, CENTRE,
                          TOP_LEFT, BOTTOM_RIGHT, CENTRE])
    for m in (agree, differ):
        for v, (lo, hi) in zip(m["cells"], m["ci"]):
            assert lo <= v <= hi
    width = lambda m, i: m["ci"][i][1] - m["ci"][i][0]   # noqa: E731
    assert width(differ, 4) > width(agree, 4)


def test_ceiling_needs_four_people():
    assert A.split_half([CENTRE] * 3) is None
    assert A.split_half([CENTRE] * 8) is not None


def test_comparison_reports_nothing_rather_than_guessing():
    out = A.comparison_maps({"tapped": [], "measured": [], "model": []})
    assert out["ok"] is False


def test_difference_map_is_measured_minus_tapped():
    out = A.comparison_maps({"tapped": [TOP_LEFT] * 5,
                             "measured": [BOTTOM_RIGHT] * 5})
    d = out["difference"]
    assert d[8] > 0 and d[0] < 0
    # Both maps sum to 1, so the differences must cancel.
    assert sum(d) == pytest.approx(0.0, abs=1e-9)


def test_every_source_gets_the_same_blur():
    """A sharp map compared against a blurry one would fake disagreement."""
    out = A.comparison_maps({"tapped": [CENTRE] * 4, "measured": [CENTRE] * 4},
                            sigma=0.3)
    assert out["sigma"] == 0.3
    assert out["maps"]["tapped"]["cells"] == pytest.approx(
        out["maps"]["measured"]["cells"])


# --- the endpoint ---------------------------------------------------------

def test_maps_endpoint_separates_the_two_human_groups(client):
    for i in range(4):
        tap(client, f"tapper-{i:04d}", TOP_LEFT)
    for i in range(4):
        gaze(client, f"gazer-{i:04d}", BOTTOM_RIGHT)

    body = client.get("/api/compare/maps").json()
    assert body["ok"] is True
    assert body["maps"]["tapped"]["n"] == 4
    assert body["maps"]["measured"]["n"] == 4
    assert body["maps"]["tapped"]["cells"][0] > body["maps"]["measured"]["cells"][0]
    assert body["difference"][8] > 0


def test_taps_still_count_when_nobody_was_ever_assigned(client):
    """Tap-only rounds predate the between-subjects split and have no
    assignment rows at all. Those responses must not vanish from the charts."""
    tap(client, "tapper-0001", CENTRE)
    tap(client, "tapper-0002", CENTRE)
    body = client.get("/api/compare/maps").json()
    assert body["maps"]["tapped"]["n"] == 2


def test_maps_endpoint_omits_sources_with_no_data(client):
    tap(client, "tapper-0001", CENTRE)
    body = client.get("/api/compare/maps").json()
    assert "tapped" in body["maps"]
    assert "measured" not in body["maps"]
    assert body["difference"] is None


def test_maps_endpoint_carries_the_context_the_page_labels_with(client):
    # The participant page assigns before it submits, and the counts come
    # from the assignment table, so the test follows the same order.
    client.post("/api/assign", json={"participant_uuid": "tapper-0001"})
    tap(client, "tapper-0001", CENTRE)
    body = client.get("/api/compare/maps").json()
    assert body["image"]["filename"] == "a.jpg"
    assert body["config"]["probe"]["id"] == "freeview"
    assert body["conditions"]["tap"]["completed"] == 1
    assert body["grid"] == 3


def test_maps_endpoint_honours_the_grid_size(client):
    tap(client, "tapper-0001", CENTRE)
    body = client.get("/api/compare/maps", params={"grid": 4}).json()
    assert body["grid"] == 4
    assert len(body["maps"]["tapped"]["cells"]) == 16
    # Out-of-range values are clamped, not rejected: a stray query string
    # should never take the page down mid-session.
    assert client.get("/api/compare/maps", params={"grid": 99}).json()["grid"] == 5


def test_excluded_calibrations_are_reported_not_hidden(client):
    client.post("/api/gaze/session",
                json={"participant_uuid": "gazer-bad1", "grade": "poor"})
    tap(client, "tapper-0001", CENTRE)
    assert client.get("/api/compare/maps").json()["gaze_excluded"] == 1


def test_charts_page_is_served(client):
    r = client.get("/charts")
    assert r.status_code == 200
    assert "Tapped vs measured vs model" in r.text


# --- the page's own guarantees -------------------------------------------

def test_charts_use_only_validated_series_colours():
    """The hues are checked by running the validator; this pins the result.

    Dark slots pass all-pairs CVD 9.4 / normal-vision 20.9 on #1c1e28; the
    light set swaps only the blue. The display's older #5b8cff/#b06bff pair
    failed both gates, so it must not come back here by copy-paste.
    """
    module = MODULE.read_text()
    for hexcode in ("#3987e5", "#d95926", "#199e70", "#2a78d6"):
        assert hexcode in module
    for failed in ("#b06bff", "#5b8cff", "#ff7a45"):
        assert failed not in module


def test_page_offers_a_table_view_and_a_legend():
    page = (STATIC / "charts.html").read_text()
    assert 'id="table"' in page and "numbersTable" in page
    assert 'id="legend"' in page and "legendRow" in page


def test_diverging_scale_has_a_neutral_midpoint():
    """A hue in the middle would make "no difference" look like a finding."""
    mids = re.findall(r'mid:\s*"(#[0-9a-f]{6})"', MODULE.read_text())
    assert mids, "no diverging midpoint declared"
    for m in mids:
        r, g, b = (int(m[i:i + 2], 16) for i in (1, 3, 5))
        assert max(r, g, b) - min(r, g, b) <= 12, f"{m} is not neutral"


def test_cell_names_follow_the_grid_size():
    """?grid=4 once produced a 4x4 whose second cell was called "top-centre"."""
    module = MODULE.read_text()
    assert "export function cellNames(n)" in module
    assert "cellNames(data.grid)" in module
    # The grid size in the prose is set from the response, not typed in.
    assert 'id="grid-name"' in (STATIC / "charts.html").read_text()


def test_page_forwards_grid_and_sigma_to_the_endpoint():
    page = (STATIC / "charts.html").read_text()
    assert "['grid', 'sigma']" in page


def test_the_two_full_screen_layers_cannot_both_be_on(client):
    """Either would hide the other, so turning one on turns the other off."""
    client.post("/api/control/layers", json={"json": True}, headers=AUTH)
    assert client.get("/api/state").json()["layers"]["json"] is True

    out = client.post("/api/control/layers", json={"charts": True},
                      headers=AUTH).json()
    assert out["layers"]["charts"] is True
    assert out["layers"]["json"] is False

    out = client.post("/api/control/layers", json={"json": True},
                      headers=AUTH).json()
    assert out["layers"]["json"] is True
    assert out["layers"]["charts"] is False


def test_turning_a_full_screen_layer_off_leaves_the_other_alone(client):
    client.post("/api/control/layers", json={"charts": True}, headers=AUTH)
    out = client.post("/api/control/layers", json={"charts": False},
                      headers=AUTH).json()
    assert out["layers"]["charts"] is False
    assert out["layers"]["json"] is False


def test_image_overlays_are_untouched_by_the_full_screen_rule(client):
    client.post("/api/control/layers", json={"gaze": True, "paths": True},
                headers=AUTH)
    out = client.post("/api/control/layers", json={"charts": True},
                      headers=AUTH).json()
    assert out["layers"]["gaze"] is True and out["layers"]["paths"] is True


def test_display_draws_the_charts_from_the_shared_module(client):
    """Two screens, one copy of the drawing code — or they drift apart."""
    page = (STATIC / "display.html").read_text()
    assert "/static/charts.js" in page
    assert "renderCharts(state.layers.charts)" in page
    # The projected view deliberately omits the table and the dot plot.
    assert "numbersTable" not in page and "dotPlot" not in page


def test_a_stale_server_explains_its_own_404(client, monkeypatch):
    from app import main
    monkeypatch.setattr(main, "_server_freshness",
                        lambda: {"stale": True, "uptime_seconds": 3600,
                                 "started_at": 0, "code_mtime": 1})
    body = client.get("/charts-that-do-not-exist").json()
    assert body["stale_server"] is True
    assert "Restart it" in body["detail"]


def test_a_deliberate_404_keeps_its_own_message(client):
    """The stale-server hint must not overwrite a real explanation."""
    r = client.post("/api/model/push", headers=AUTH, json={
        "image": "nope.jpg", "mode": "freeview", "n_fixations": 5,
        "scanpath_norm": [[0.5, 0.5]]})
    assert r.status_code == 404
    assert "nope.jpg" in r.json()["detail"]
