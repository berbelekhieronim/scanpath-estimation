"""The comparison charts: per-participant maps, the endpoint, and the page.

The assertions that matter here are the ones with a known right answer. A
group that all looked at the same cell must put its mass there; two groups
that looked at opposite corners must not correlate; and one loud participant
must not be able to outvote the rest, because the unit of observation is the
person (SPEC-METRICS.md section 1).
"""

import importlib
import json
import re
import shutil
import subprocess
import tempfile
import sys
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


# --- the display's overlays must say what they are ------------------------

DISPLAY = STATIC / "display.html"
HEATMAP = STATIC / "heatmap.js"
DISPLAY_CSS = STATIC / "display.css"


def test_the_two_density_layers_do_not_share_a_ramp():
    """They did, and the legend called one of them purple while it wasn't.

    Both heatmaps were drawn by one call with one rainbow ramp, so the
    tapped and measured clouds were the same picture twice.
    """
    hm = HEATMAP.read_text()
    assert "export const RAMPS" in hm
    assert "blue:" in hm and "orange:" in hm
    page = DISPLAY.read_text()
    assert "'blue'" in page and "'orange'" in page


def test_measured_gaze_is_drawn_as_contours_not_a_second_cloud():
    """Different form, not just a different hue: a filled cloud painted over
    another filled cloud hides it completely."""
    assert "'contour'" in DISPLAY.read_text()
    assert "function contour(" in HEATMAP.read_text()


def test_the_legend_names_colours_that_are_actually_drawn():
    page = DISPLAY.read_text()
    for gone in ("Warm areas", "Purple cloud", "Purple traces"):
        assert gone not in page, f"{gone} describes a mark that is not drawn"
    assert "Blue cloud" in page
    assert "Orange rings" in page
    assert "Green numbered path" in page


def _without_comments(text: str) -> str:
    """Drop /* ... */ and // ... so a comment explaining why a colour was
    removed does not read as the colour coming back."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"^\s*//.*$", "", text, flags=re.M)


def test_display_overlays_use_the_validated_slots():
    """The old trio failed CVD validation: blue against purple separated by
    deltaE 1.5 under deutan simulation, 13.2 with normal vision."""
    for path in (DISPLAY, DISPLAY_CSS):
        text = _without_comments(path.read_text())
        for failed in ("#b06bff", "#d3a5ff", "#ff7a45", "#e8452e"):
            assert failed not in text, f"{failed} is back in {path.name}"


def test_the_caption_says_the_two_groups_are_different_people():
    """Two clouds on one picture read as before-and-after for one person
    unless it is said otherwise, and that is the one thing they are not."""
    assert "different people" in DISPLAY.read_text()


def test_analysis_declares_which_two_things_it_compares(client):
    """It reads markers, so it is the tap group against the model, and the
    measured group is not in it. A panel headed "agreement" beside a screen
    showing two human layers has to say which one it means."""
    client.post("/api/assign", json={"participant_uuid": "tapper-0001"})
    tap(client, "tapper-0001", CENTRE)
    body = client.get("/api/analysis").json()
    assert body["compares"] == {"a": "tapped", "b": "model", "n_a": 1,
                                "n_b": 0, "excludes": "measured"}


def test_analysis_ignores_gaze_entirely(client):
    """Pinning the scope: gaze sessions must not move these numbers."""
    for i in range(4):
        tap(client, f"tapper-{i:04d}", TOP_LEFT)
    before = client.get("/api/analysis").json()["compares"]["n_a"]
    for i in range(4):
        gaze(client, f"gazer-{i:04d}", BOTTOM_RIGHT)
    after = client.get("/api/analysis").json()["compares"]
    assert after["n_a"] == before == 4
    assert after["excludes"] == "measured"


def test_the_projected_charts_carry_no_session_specific_interpretation():
    """The headline sentence read the data for the audience. What it said
    changes every session, so it is the presenter's line, not the app's."""
    page = DISPLAY.read_text()
    assert "headlineText" not in page
    assert "headlineText" not in (STATIC / "charts.js").read_text()


def test_the_difference_map_can_be_drawn_over_the_scene():
    """So the audience remembers what the cells are cells of."""
    module = (STATIC / "charts.js").read_text()
    assert "o.image" in module and 'svg("image"' in module
    assert "image: opts.image" in module
    assert "sceneUrl(d)" in DISPLAY.read_text()


def test_grids_take_the_pictures_shape():
    """A square grid over a 4:3 photograph does not line up with the thing
    it describes."""
    module = (STATIC / "charts.js").read_text()
    assert "o.aspect" in module
    assert "H = Math.round(300 / (o.aspect || 1))" in module


# --- device breakdown -----------------------------------------------------

def test_device_classes_split_phones_from_laptops():
    """The split that matters is a screen at arm's length against one on a
    desk, because tracking error is angular and the phone's picture fills far
    less of the eye's field."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import device_report

    iphone = {"ua": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X)",
              "touch_points": 5, "screen_w": 375}
    mac = {"ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
           "touch_points": 0, "screen_w": 1470}
    ipad = {"ua": "Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X)",
            "touch_points": 5, "screen_w": 1024}

    assert device_report.device_class(iphone, 375, 700) == "phone"
    assert device_report.device_class(mac, 1440, 820) == "desktop/laptop"
    assert device_report.device_class(ipad, 1024, 768) == "tablet"
    # A session recorded before device capture existed must not be silently
    # counted as a laptop.
    assert device_report.device_class({}, None, None) == "unknown"


# --- landscape on a phone -------------------------------------------------

CALIBRATE = STATIC / "calibrate.html"
VIEWPORT_JS = STATIC / "gaze" / "viewport.js"


def test_layout_is_sized_from_the_measured_visible_viewport():
    """iOS sizes 100vh and `inset: 0` against the viewport WITHOUT toolbars.
    In landscape the toolbars stay, so the page was about a quarter taller
    than anything visible — and `overflow: hidden` meant no scrolling to it."""
    assert VIEWPORT_JS.exists()
    js = VIEWPORT_JS.read_text()
    assert "visualViewport" in js
    assert "--app-h" in js and "--app-w" in js

    css = CALIBRATE.read_text()
    assert "var(--app-h" in css
    # The old unconditional full-bleed rule must not come back.
    assert "position: fixed; inset: 0; touch-action" not in css


def test_gaze_is_mapped_against_the_same_box_as_the_layout():
    """Mapping with innerHeight while the page is sized to visualViewport
    stretches every sample against a box the participant cannot see."""
    page = CALIBRATE.read_text()
    assert "vx * vp.width" in page and "vy * vp.height" in page
    assert "window.innerWidth" not in page
    assert "window.innerHeight" not in page


def test_centred_overlays_stay_reachable_when_taller_than_the_screen():
    """`place-items: center` clips the top of anything too tall, and the page
    cannot scroll, so in landscape the Start button was unreachable."""
    css = CALIBRATE.read_text()
    assert "overflow-y: auto" in css
    assert ".overlay > div { margin: auto; }" in css
    assert "@media (max-height: 430px)" in css


def test_rotating_pauses_instead_of_recording_nonsense():
    """A calibration is fitted to one screen shape. Rotate and it is wrong,
    silently, in a way that looks like ordinary inaccuracy."""
    page = CALIBRATE.read_text()
    assert "rotate-guard" in page
    assert "lockOrientation()" in page
    assert "if (paused) { rotatedAway = true; return; }" in page


def test_handsets_must_be_upright_but_laptops_are_exempt():
    """Portrait is required because a phone held sideways gives the picture
    about a third of the eye's field. None of that applies to a laptop, which
    is landscape by definition and the most accurate device here — blocking
    it would turn a fix into an outage."""
    js = VIEWPORT_JS.read_text()
    assert "export function isHandset()" in js
    # Three independent signals, all required, because a false positive locks
    # someone out of the study on a machine where landscape is correct.
    assert "maxTouchPoints" in js
    assert "(pointer: coarse)" in js
    # Measured on the short edge, which does not change when rotated.
    assert "Math.min(screen.width || 0, screen.height || 0)" in js
    # Anything unexpected about the browser means no constraint at all.
    assert "} catch {" in js and "return false;" in js

    page = CALIBRATE.read_text()
    assert "isHandset()" in page and "'portrait'" in page
    assert "Turn your phone upright" in page


def test_the_guard_is_checked_on_load_and_before_starting():
    """Arriving sideways showed a 290px-tall intro with Start pushed off it,
    and starting sideways would fit the calibration to the orientation we are
    about to refuse."""
    page = CALIBRATE.read_text()
    assert "if (applyOrientationGuard()) return;" in page
    # Called at load, not only from begin().
    assert page.count("applyOrientationGuard()") >= 3


def test_the_orientation_guard_runs_after_its_dependencies_exist():
    """It referenced $ from the module preamble, before the const that
    defines it — a temporal-dead-zone throw that killed the whole module and
    left every page dead on arrival."""
    page = CALIBRATE.read_text()
    dollar = page.index("const $ = (id) => document.getElementById(id);")
    assert page.index("function applyOrientationGuard()") > dollar
    assert page.index("const REQUIRED = ") > dollar


def test_orientation_is_recorded_with_the_calibration(client):
    """So a session reads back knowing which way up it was taken. Pydantic
    drops unknown fields silently, so this asserts it round-trips rather
    than that the request was accepted."""
    import json as _json
    from app import db

    r = client.post("/api/gaze/session", json={
        "participant_uuid": "rotate-0001", "grade": "usable",
        "viewport_w": 812, "viewport_h": 294, "orientation": "landscape"})
    assert r.status_code == 200
    sess = db.get_gaze_session(r.json()["session_id"])
    assert _json.loads(sess["diagnostics_json"])["orientation"] == "landscape"


def test_a_stray_assignment_does_not_discard_unassigned_tappers(client):
    """The old rule kept only assigned tappers and fell back to "everyone"
    when the table was completely empty. One phone merely opening the
    participant page flipped that fallback off and silently dropped every
    tapper who had already submitted."""
    for i in range(3):
        tap(client, f"tapper-{i:04d}", CENTRE)
    # Somebody else is assigned, but never submits anything.
    client.post("/api/assign", json={"participant_uuid": "bystander-0001"})

    body = client.get("/api/compare/maps").json()
    assert body["maps"]["tapped"]["n"] == 3


def test_a_gaze_participant_is_never_counted_as_a_tapper(client):
    """The one reason to drop someone holding markers."""
    client.post("/api/control/capture-mode", json={"mode": "gaze"}, headers=AUTH)
    client.post("/api/assign", json={"participant_uuid": "gazer-0001"})
    tap(client, "gazer-0001", CENTRE)          # should not count
    client.post("/api/control/capture-mode", json={"mode": "tap"}, headers=AUTH)
    client.post("/api/assign", json={"participant_uuid": "tapper-0001"})
    tap(client, "tapper-0001", TOP_LEFT)

    body = client.get("/api/compare/maps").json()
    assert body["maps"]["tapped"]["n"] == 1


# --- the grid drawn onto the picture --------------------------------------

def test_the_grid_is_translucent_over_the_picture():
    """Opaque cells hide the thing they describe, which defeats the point of
    drawing them on the picture rather than beside it."""
    module = (STATIC / "charts.js").read_text()
    assert "o.fillOpacity ??" in module
    page = (STATIC / "display.html").read_text()
    assert "fillOpacity: 0.46" in page


def test_every_chart_grid_sits_over_the_scene():
    """Asked for explicitly: a cell should read as a part of the picture, not
    as an abstract square."""
    page = (STATIC / "charts.html").read_text()
    assert "densityPanels(d, {aspect, image: scene})" in page
    assert "image: scene, aspect, pair: key" in page


def test_the_grid_source_is_switchable_and_validated(client):
    for src in ("tapped", "measured", "model"):
        r = client.post("/api/control/grid-source", json={"source": src},
                        headers=AUTH)
        assert r.status_code == 200
        assert client.get("/api/state").json()["grid_source"] == src
    bad = client.post("/api/control/grid-source", json={"source": "guesses"},
                      headers=AUTH)
    assert bad.status_code == 422


def test_start_page_labels_do_not_run_together():
    """Both were inline spans, so every link read "ControlsLayers, capture
    mode, image, model" with no break between name and description."""
    page = (STATIC / "start.html").read_text()
    assert ".link .txt { display: flex; flex-direction: column;" in page
    assert '<span class="txt">' in page


def test_nobody_can_be_permanently_locked_out_by_the_orientation_check():
    """A live room has no time to debug a device the check reads wrongly.
    One sideways recording is a far smaller loss than one participant who
    could not take part at all."""
    page = CALIBRATE.read_text()
    assert "params.get('portrait') !== 'off'" in page       # manual override
    assert "rg-escape" in page and "skipOrientationCheck" in page
    # The way out appears on its own, without anyone knowing the URL trick.
    assert "setTimeout(() => { $('rg-escape').hidden = false; }, 7000)" in page
    # Only for the "must be upright" case — a rotation mid-run is the
    # participant's own doing and turning back is the actual fix.
    assert "if (problem === 'required')" in page


# --- the gaze correction --------------------------------------------------

NODE = shutil.which("node")
CALIB_JS = STATIC / "gaze" / "calibration.js"


def run_fit(measured_of):
    """Exercise the real correction arithmetic, not a description of it.

    String-matching the source would pass just as happily on a fit that
    inverted the gain, and this is the one piece of maths standing between a
    squashed tracker and the numbers the whole study reports.
    """
    script = """
    import('file://%s').then(m => {
      const C = Object.create(m.Calibration.prototype);
      C.validation = m.VALIDATION_POINTS.map(t => ({target: t, measured: (%s)(t)}));
      const fit = C.fit();
      const bias = C.bias();
      const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
      const errWith = (f) => mean(C.validation.map((v) => {
        const c = f ? m.applyFit(v.measured, f) : v.measured;
        return Math.hypot(c[0] - v.target[0], c[1] - v.target[1]);
      }));
      const offsetOnly = mean(C.validation.map((v) =>
        Math.hypot(v.measured[0] - bias[0] - v.target[0],
                   v.measured[1] - bias[1] - v.target[1])));
      console.log(JSON.stringify({gain: fit.gain, gainUsed: fit.gainUsed,
        raw: errWith(null), corrected: errWith(fit), offsetOnly}));
    });
    """ % (CALIB_JS, measured_of)
    out = subprocess.run([NODE, "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="node is needed to run the module")
def test_the_correction_undoes_a_tracker_that_squashes_toward_the_centre():
    """The classic webcam failure: the eye sweeps the screen, the tracker
    reports a huddle near the middle. An offset cannot touch it — here it
    makes things very slightly worse — and a gain fixes it outright."""
    r = run_fit("(t) => [0.5 + (t[0]-0.5)*0.6 + 0.04, 0.5 + (t[1]-0.5)*0.6 - 0.03]")
    assert r["gainUsed"] is True
    assert r["gain"][0] == pytest.approx(0.6, abs=0.01)
    assert r["corrected"] < 0.001
    assert r["offsetOnly"] > r["raw"] * 0.95      # offset alone achieves nothing


@pytest.mark.skipif(NODE is None, reason="node is needed to run the module")
def test_a_pure_offset_is_still_corrected():
    """The case that already worked must not regress."""
    r = run_fit("(t) => [t[0] + 0.08, t[1] - 0.05]")
    assert r["corrected"] < 0.001
    assert r["gain"][0] == pytest.approx(1.0, abs=0.05)


@pytest.mark.skipif(NODE is None, reason="node is needed to run the module")
def test_one_wild_sample_never_rescales_the_real_data():
    """Multiplying a whole recording by a slope fitted to one bad tap is far
    worse than leaving it alone, so an implausible gain is refused on both
    axes — not just the one that looks wrong."""
    r = run_fit("(t) => (t[0] < 0.4 && t[1] < 0.4) ? [0.95, 0.02] : [t[0], t[1]]")
    assert r["gainUsed"] is False
    assert r["gain"] == [1, 1]


# --- navigation -----------------------------------------------------------

def test_operator_pages_share_one_navigation_bar():
    """Every page used to carry its own handful of links — Controls four,
    Charts two, Images none — so moving between them meant knowing the URLs.
    Presenting is the wrong moment to be remembering paths."""
    for name in ("control.html", "charts.html", "admin.html", "start.html"):
        page = (STATIC / name).read_text()
        assert 'id="nav"' in page, name
        assert "renderNav(" in page, name


def test_projected_and_participant_pages_have_no_navigation():
    """A nav bar on the projected screen is a distraction; on a participant's
    phone it is an invitation to wander off mid-study."""
    for name in ("display.html", "qr.html", "index.html", "consent.html",
                 "calibrate.html", "view.html"):
        page = (STATIC / name).read_text()
        assert "renderNav" not in page, name


def test_navigation_carries_the_control_token():
    """Otherwise every hop to a locked page asks for it again."""
    js = (STATIC / "nav.js").read_text()
    assert "scanpath_control_token" in js
    assert "token: true" in js


@pytest.mark.skipif(NODE is None, reason="node is needed to run the module")
def test_one_bad_frame_does_not_move_the_validation_reading():
    """A 700ms window holds two or three predictions at this frame rate. A
    mean over three moves most of the way toward a single half-blink, and
    these readings are what both corrections are fitted to."""
    script = """
    import('file://%s').then(m => {
      const C = Object.create(m.Calibration.prototype);
      C.samples = [{x:0.50,y:0.50,wall:1e12},{x:0.52,y:0.48,wall:1e12},
                   {x:0.95,y:0.05,wall:1e12}];
      performance.now = () => 1e12;
      const r = C._recentPoint(700);
      console.log(JSON.stringify({point: r.point, spread: r.spread, n: r.n}));
    });
    """ % (CALIB_JS,)
    out = subprocess.run([NODE, "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    assert r["n"] == 3
    # The median ignores the outlier; a mean would have said 0.657.
    assert r["point"][0] == pytest.approx(0.52, abs=0.001)
    assert r["spread"] > 0.5           # and the disagreement is reported


@pytest.mark.skipif(NODE is None, reason="node is needed to run the module")
def test_a_validation_point_that_disagrees_with_itself_is_not_fitted():
    """It is measuring the moment, not the eye."""
    script = """
    import('file://%s').then(m => {
      const C = Object.create(m.Calibration.prototype);
      C.validation = m.VALIDATION_POINTS.map((t, i) => ({
        target: t, measured: [t[0] + 0.05, t[1] - 0.02],
        spread: i === 0 ? 0.6 : 0.05 }));
      const f = C.fit();
      console.log(JSON.stringify({usable: C._usable().length,
        gainUsed: f.gainUsed, offset: f.offset}));
    });
    """ % (CALIB_JS,)
    out = subprocess.run([NODE, "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    assert r["usable"] == 2
    # Two points can still carry an offset, which beats no correction.
    assert r["gainUsed"] is False
    assert r["offset"][0] == pytest.approx(0.05, abs=0.001)


# --- what was applied after the mobile-accuracy review --------------------

def test_calibration_uses_more_of_the_screen_than_it_used_to():
    """The 15% inset was a desktop heuristic: there a screen corner really is
    an extreme eye rotation. On a phone the whole screen spans about eleven
    degrees, so the inset was discarding a fifth of the baseline that the
    offset and gain corrections are fitted over."""
    src = (STATIC / "gaze" / "calibration.js").read_text()
    xs = [float(m) for m in re.findall(r"\[(0\.\d+), 0\.\d+\]", src)]
    assert min(xs) <= 0.08, "outermost calibration point is still inset"
    # And the dot shrank, or it would hang off the edge of a narrow phone.
    page = (STATIC / "calibrate.html").read_text()
    assert ".target { position: absolute; width: 44px" in page


def test_the_shared_kernel_actually_reaches_one_resolution():
    """Adding the same sigma to everyone preserves the differences it was
    meant to remove: a laptop at 0.08 and a phone at 0.22 both gain 0.21 and
    land at 0.22 and 0.30. Errors add in quadrature, so a common resolution
    means adding sqrt(target^2 - own^2)."""
    from app import analysis as A
    import math

    for own in (0.05, 0.08, 0.15, 0.20):
        add = A.blur_to_reach(0.21, own)
        assert math.hypot(own, add) == pytest.approx(0.21, abs=1e-6)
    # Someone already coarser than the target cannot be sharpened, and is
    # left alone rather than blurred further.
    assert A.blur_to_reach(0.21, 0.30) == 0.0
    # Unknown error keeps the old behaviour exactly.
    assert A.blur_to_reach(0.21, None) == 0.21


def test_participants_end_at_one_resolution_instead_of_merely_similar_ones():
    """The claim the shared kernel is supposed to make, tested against data
    as it actually arrives — scattered by each participant's own measurement
    error, rather than as tidy points.

    Adding a flat 0.21 to everybody leaves a precise observer visibly sharper
    than a coarse one. Adding only what each needs to reach the target makes
    them agree.
    """
    import numpy as np
    from app import analysis as A

    def centre_mass(own, seed, corrected, k=400):
        rng = np.random.default_rng(seed)
        pts = [[min(.99, max(.01, rng.normal(0.5, own))),
                min(.99, max(.01, rng.normal(0.5, own)))] for _ in range(k)]
        errs = [own] if corrected else None
        return A.participant_maps([pts], n=3, sigma=0.21, errors=errs)[0][4]

    owns = (0.04, 0.09, 0.14, 0.19)      # all at or below the target
    flat = [np.mean([centre_mass(o, s, False) for s in range(8)]) for o in owns]
    corr = [np.mean([centre_mass(o, s, True) for s in range(8)]) for o in owns]

    spread = lambda a: max(a) - min(a)   # noqa: E731
    assert spread(flat) > 0.06           # they clearly disagree today
    assert spread(corr) < 0.02           # and agree once corrected
    assert spread(corr) < spread(flat) / 3


def test_someone_coarser_than_the_target_is_left_alone_not_blurred_further():
    """A measurement cannot be sharpened after the fact, so the honest thing
    is to add nothing and let the map be as coarse as the data really is."""
    from app import analysis as A
    assert A.blur_to_reach(0.21, 0.30) == 0.0


def test_viewing_time_is_a_presenter_setting(client):
    """Tap count always was. Viewing time was a URL parameter only, so the
    two groups' effort was controlled in different places and one could not
    be changed without editing a link."""
    assert client.get("/api/state").json()["view_ms"] == 5000
    r = client.post("/api/control/view-ms", json={"ms": 10000}, headers=AUTH)
    assert r.status_code == 200
    assert client.get("/api/state").json()["view_ms"] == 10000
    # Bounded: it is the participant's time, and too short cannot be binned.
    assert client.post("/api/control/view-ms", json={"ms": 500},
                       headers=AUTH).status_code == 422
    assert client.post("/api/control/view-ms", json={"ms": 120000},
                       headers=AUTH).status_code == 422


def test_the_capture_page_takes_its_duration_from_the_server():
    page = (STATIC / "calibrate.html").read_text()
    assert "state.view_ms" in page
    # ?ms= still wins, for rehearsing without changing the session.
    assert "VIEW_MS_FORCED" in page


def test_every_pair_gets_a_difference_map_not_just_the_human_one():
    """A correlation says how much two sources disagree and never where.
    Measured against model is the comparison the project exists for, and it
    had only a number."""
    from app import analysis as A
    TL, BR, C = [[0.12, 0.12]] * 4, [[0.88, 0.88]] * 4, [[0.5, 0.5]] * 4
    out = A.comparison_maps({"tapped": [TL] * 5, "measured": [BR] * 5,
                             "model": [C] * 5})
    assert set(out["differences"]) == {"tapped|measured", "tapped|model",
                                       "measured|model"}
    # The legacy single map still resolves, so nothing reading it breaks.
    assert out["difference"] == out["differences"]["tapped|measured"]
    # Sign convention: the second named source minus the first.
    assert out["differences"]["measured|model"][4] > 0      # model is centred


def test_the_dot_plot_gave_way_to_a_correlation_matrix():
    """Nine rows times three series of overlapping intervals was correct and
    unreadable. A matrix answers "do these agree" in one glance."""
    page = (STATIC / "charts.html").read_text()
    assert "correlationMatrix" in page
    assert "dotPlot" not in page
    module = (STATIC / "charts.js").read_text()
    assert "export function correlationMatrix" in module


def test_the_matrix_diagonal_is_not_coloured_like_a_pair_score():
    """It is a ceiling, a different quantity. Dropping it into the same
    diverging scale would invite reading it as the best correlation."""
    module = (STATIC / "charts.js").read_text()
    assert "if (!self) td.style.background = divColour(v, 1);" in module
    css = (STATIC / "charts.css").read_text()
    assert ".chart-table.matrix td.cc.self" in css and "background: none" in css


def test_the_interpretation_says_when_it_cannot_be_trusted():
    """It is the one part of the page that states a conclusion, so it is the
    part most able to overstate one."""
    module = (STATIC / "charts.js").read_text()
    assert "export function interpretation" in module
    for caveat in ("synthetic", "fewer than six people",
                   "too few people for a ceiling"):
        assert caveat in module, caveat
    # And it names the comparison that decides whether any of it means
    # anything: the model against a plain centre blob.
    assert "centre bias" in module


def test_the_agreement_panel_left_the_projected_screen(client):
    """It crowded the picture it was meant to explain, and /charts can lay
    the same numbers out properly. The layer is gone, not hidden."""
    assert "analysis" not in client.get("/api/state").json()["layers"]
    r = client.post("/api/control/layers", json={"analysis": True},
                    headers=AUTH)
    assert r.status_code == 400

    page = (STATIC / "display.html").read_text()
    assert "renderAnalysis" not in page
    assert 'id="analysis"' not in page
    # And the control page no longer offers a switch for it.
    assert "Agreement metrics" not in (STATIC / "control.html").read_text()


def test_the_analysis_endpoint_survives_for_the_raw_view(client):
    """/api/raw still publishes it, and it is the one place the tapped-vs-
    model figures are computed. Removing the layer is a display decision."""
    assert client.get("/api/analysis").status_code == 200
    assert "analysis" in client.get("/api/raw").json()
