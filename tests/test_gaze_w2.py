"""Phase W2 tests: consent, calibration geometry, quality gating, storage."""

import importlib
import json
import re
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"


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


def session(grade="good", uuid="participant-0001", **over):
    p = {"participant_uuid": uuid, "grade": grade, "tracker": "mock",
         "tracker_version": "1", "mean_error": 0.12, "worst_error": 0.2,
         "points_accepted": 9, "points_total": 9,
         "validation": [{"target": [0.3, 0.3], "error": 0.11}]}
    p.update(over)
    return p


# --- geometry --------------------------------------------------------------

def _points(name):
    src = (STATIC / "gaze" / "calibration.js").read_text()
    body = re.search(rf"export const {name} = \[(.*?)\];", src, re.S).group(1)
    return [[float(v) for v in m] for m in re.findall(r"\[([\d.]+),\s*([\d.]+)\]", body)]


def test_calibration_and_validation_point_counts():
    """Nine calibration points because the tracker's few-shot adaptation
    expects that many. Three validation points because twelve taps total was
    already reported as feeling long on a phone — and because three is
    enough: they give two distinct target levels on each axis, which is the
    minimum that can measure a gain rather than only an offset."""
    assert len(_points("CALIB_POINTS")) == 9
    assert len(_points("VALIDATION_POINTS")) == 3
    total = len(_points("CALIB_POINTS")) + len(_points("VALIDATION_POINTS"))
    assert total <= 12, "more taps than this loses the room"


def test_validation_points_are_never_calibration_points():
    """Scoring on a trained point measures memorisation, not accuracy."""
    for v in _points("VALIDATION_POINTS"):
        for c in _points("CALIB_POINTS"):
            assert ((v[0] - c[0]) ** 2 + (v[1] - c[1]) ** 2) ** 0.5 > 0.1


# The narrowest phone still in common use, and the target's size on it.
NARROW_PX, TARGET_PX = 320, 44


def test_points_sit_as_close_to_the_edge_as_the_target_allows():
    """The old 15% inset came from desktop work, where a screen corner really
    is an extreme eye rotation. On a phone at arm's length the whole screen
    spans about eleven degrees, so its corners are four degrees off centre —
    and the inset was discarding a fifth of the baseline that the offset and
    gain corrections are fitted over.

    What constrains it now is geometry, not the eye: the dot has to stay on
    screen and stay tappable.
    """
    pts = _points("CALIB_POINTS") + _points("VALIDATION_POINTS")
    margin = (TARGET_PX / 2) / NARROW_PX          # ~0.069
    for x, y in pts:
        assert margin <= x <= 1 - margin, (x, "dot would clip off screen")
        assert margin <= y <= 1 - margin, (y, "dot would clip off screen")

    # And the baseline is actually wider than the old one, or none of this
    # was worth doing.
    xs = [x for x, _ in _points("CALIB_POINTS")]
    assert max(xs) - min(xs) > 0.70


def test_calibration_points_clear_the_trackers_proximity_filter():
    """Upstream drops a point within 0.05 units of the previous one."""
    pts = _points("CALIB_POINTS")
    for i, a in enumerate(pts):
        for bpt in pts[i + 1:]:
            assert abs(a[0] - bpt[0]) >= 0.06 or abs(a[1] - bpt[1]) >= 0.06


def test_validation_samples_the_window_before_the_tap():
    """After the tap nothing holds the participant's gaze on the target, so
    sampling afterwards measures drift and scores it as error."""
    src = (STATIC / "gaze" / "calibration.js").read_text()
    block = src[src.index("PHASE.VALIDATING"):src.index("this._setPhase(PHASE.DONE)")]
    # The reading is taken immediately after the tap resolves, from the
    # window that ended at the tap — not from anything sampled later.
    assert "await this._awaitTap();\n        const m = this._recentPoint" in block


# --- quality grading -------------------------------------------------------

@pytest.mark.parametrize("err,expected", [
    (0.05, "good"), (0.18, "good"),
    (0.19, "usable"), (0.30, "usable"),
    (0.31, "poor"), (0.9, "poor"),
    (None, "failed"),
])
def test_grade_boundaries(err, expected):
    src = (STATIC / "gaze" / "calibration.js").read_text()
    good = float(re.search(r"GOOD:\s*([\d.]+)", src).group(1))
    usable = float(re.search(r"USABLE:\s*([\d.]+)", src).group(1))
    if err is None:
        grade = "failed"
    elif err <= good:
        grade = "good"
    elif err <= usable:
        grade = "usable"
    else:
        grade = "poor"
    assert grade == expected


# --- storage ---------------------------------------------------------------

def test_a_good_session_is_stored_as_usable(client):
    r = client.post("/api/gaze/session", json=session("good"))
    assert r.status_code == 200
    d = r.json()
    assert d["usable"] is True
    assert d["stats"]["usable"] == 1 and d["stats"]["excluded"] == 0


@pytest.mark.parametrize("grade", ["poor", "failed"])
def test_a_failed_session_is_stored_and_counted_as_excluded(client, grade):
    """Discarding failures would make every other number look better than it
    is. The exclusion rate has to be reportable (spec 5.1)."""
    d = client.post("/api/gaze/session", json=session(grade)).json()
    assert d["usable"] is False
    assert d["stats"]["total"] == 1
    assert d["stats"]["excluded"] == 1
    assert d["stats"]["usable"] == 0


def test_stats_accumulate_a_mixed_room(client):
    for i, g in enumerate(["good", "good", "usable", "poor", "failed"]):
        client.post("/api/gaze/session", json=session(g, uuid=f"participant-{i:04d}"))
    s = client.get("/api/gaze/stats").json()
    assert s["total"] == 5 and s["usable"] == 3 and s["excluded"] == 2
    assert s["grades"] == {"good": 2, "usable": 1, "poor": 1, "failed": 1}


def test_an_unknown_grade_is_rejected(client):
    assert client.post("/api/gaze/session",
                       json=session("excellent")).status_code == 422


def test_session_needs_no_token(client):
    """Participants must never be asked to authenticate."""
    assert client.post("/api/gaze/session", json=session()).status_code == 200


def test_recalibrating_records_both_attempts(client):
    """Attempts are data: two poor tries then a good one is worth knowing."""
    client.post("/api/gaze/session", json=session("poor", uuid="participant-0001"))
    client.post("/api/gaze/session", json=session("good", uuid="participant-0001"))
    assert client.get("/api/gaze/stats").json()["total"] == 2


def test_session_is_scoped_to_the_current_round(client):
    client.post("/api/gaze/session", json=session())
    assert client.get("/api/gaze/stats").json()["total"] == 1
    client.post("/api/control/reset-round", headers={"X-Control-Token": "test-token"})
    assert client.get("/api/gaze/stats").json()["total"] == 0, "new round starts clean"


# --- pages -----------------------------------------------------------------

def test_consent_and_calibrate_pages_load(client):
    assert client.get("/consent").status_code == 200
    assert client.get("/calibrate").status_code == 200
    assert client.get("/static/gaze/calibration.js").status_code == 200


def test_consent_states_the_promises_plainly(client):
    text = client.get("/consent").text
    assert "never sent anywhere" in text
    assert "No video or photos are uploaded" in text
    assert "Declining" in text, "declining must be presented as an equal option"


def test_calibrate_releases_the_camera_on_hide(client):
    text = client.get("/calibrate").text
    assert "visibilitychange" in text and "pagehide" in text



# --- stall handling (added after a real phone hung with no result) ---------

def test_calibration_has_a_stall_watchdog():
    """Upstream's frame loop exits permanently when the video element pauses,
    which mobile browsers do to off-screen video. Without a watchdog that
    presents as an indefinite hang with no result, which is what happened."""
    src = (STATIC / "gaze" / "calibration.js").read_text()
    assert "StallError" in src
    assert "stallMs" in src
    assert re.search(r"this\.stallMs\s*=\s*opts\.stallMs\s*\?\?\s*\d+", src)


def test_run_always_returns_a_result():
    """Every path out of run() must produce something reportable."""
    src = (STATIC / "gaze" / "calibration.js").read_text()
    run = src[src.index("  async run()"):src.index("  async _run()")]
    assert "catch" in run and "return this.result(" in run


def test_calibration_exposes_live_diagnostics():
    src = (STATIC / "gaze" / "calibration.js").read_text()
    assert "diagnostics()" in src
    for field in ("framesSeen", "facesSeen", "faceRate", "msSinceFace"):
        assert field in src, field


def test_page_keeps_the_video_visible():
    """An invisible video element gets paused by mobile browsers, and
    upstream's frame loop never restarts after a pause."""
    src = (STATIC / "calibrate.html").read_text()
    style = src[src.index("<style>"):src.index("</style>")]
    video_rule = style[style.index("video {"):]
    assert "opacity: 0" not in video_rule.split("}")[0]
    assert "display: none" not in video_rule.split("}")[0]


def test_taps_are_acknowledged_immediately():
    """Upstream debounces calibration points at 1000ms, so an unacknowledged
    tap invites a second one that is then silently dropped."""
    src = (STATIC / "calibrate.html").read_text()
    assert "acknowledgeTap" in src
    assert "ripple" in src
    assert "navigator.vibrate" in src


def test_instructions_lead_with_eyes_visible_then_holding_still():
    """Revised twice. First after device testing: seeing your own eyes in the
    preview matters more than the posture, so either is allowed. Then again
    because the posture sentence described how to sit rather than the thing
    that actually costs accuracy — that the phone and the head must not move
    once calibration has measured them."""
    src = (STATIC / "calibrate.html").read_text()
    # Collapsed, because the source wraps these sentences across lines.
    flat = " ".join(src.split())
    assert "eyes are visible" in flat
    assert "stay still" in flat
    assert "head stay where they are" in flat
    # And what to do with the dots, without reference to the other condition.
    assert "Look straight at each" in flat
    assert "Don't tap" not in flat


def test_all_overlay_copy_lives_inside_ov_body():
    """overlay() replaces #ov-body only. A sibling paragraph would survive
    every later screen — which is how instructions about a moving dot ended
    up on the thank-you screen."""
    src = (STATIC / "calibrate.html").read_text()
    panel = src[src.index('<div class="overlay" id="overlay">'):src.index('id="ov-btn"')]
    after_body = panel[panel.index('id="ov-body"'):]
    assert "<p>" not in after_body.split("</div>")[1] if "</div>" in after_body else True
    assert '<div id="ov-body">' in src


def test_viewing_stage_shows_nothing_but_the_picture():
    """A countdown ring is itself a fixation target, and this is the one
    stage where the eye must have no target other than the image."""
    src = (STATIC / "calibrate.html").read_text()
    assert 'id="vring"' not in src and "vring-fg" not in src
    assert "$('cam').style.display = 'none'" in src
    assert "$('face-chip').style.display = 'none'" in src


def test_the_thank_you_screen_has_no_button_and_does_not_navigate():
    """Navigating away landed the participant on the tap screen — the wrong
    condition for them, and where the stray Submit and Undo came from."""
    src = (STATIC / "calibrate.html").read_text()
    assert "That's everything" in src
    assert "location.href = '/?viewed=1'" not in src


def test_no_screen_tells_a_participant_to_look_up_at_the_screen():
    """Written for a room where the projected display is the next thing to
    watch. Half the time it is not: the phone is being handed back, or the
    round is still filling, and an instruction that does not match the room
    reads as the app not knowing what is going on."""
    for page in ("calibrate.html", "view.html", "index.html"):
        flat = " ".join((STATIC / page).read_text().split()).lower()
        assert "look up at the screen" not in flat, page


def test_diagnostics_are_persisted(client):
    r = client.post("/api/gaze/session", json=session(
        "failed", failure="stalled",
        diagnostics={"framesSeen": 120, "facesSeen": 4, "faceRate": 0.03}))
    assert r.status_code == 200
    from app import db
    with db.connect() as conn:
        row = conn.execute("SELECT diagnostics_json FROM gaze_sessions").fetchone()
    assert json.loads(row["diagnostics_json"])["framesSeen"] == 120


# --- retry must not rebuild the model (crash on the 2nd/3rd attempt) -------

def test_page_reuses_the_tracker_across_attempts():
    """Constructing WebEyeTrack again loads another BlazeGaze model and
    another MediaPipe FaceLandmarker, neither of which upstream can dispose.
    A retry therefore used to double the memory held, and a third attempt
    tripled it — which is exactly when the tab died."""
    src = (STATIC / "calibrate.html").read_text()
    assert "if (!tracker) {" in src and "createTracker" in src
    assert "tracker.restart()" in src


def _method_body(src, signature):
    """Text of a class method: from its definition to the next one at the
    same indent. Plain str.index is unreliable here — several classes in the
    file share method names."""
    start = src.index(signature)
    rest = src[start + len(signature):]
    end = re.search(r"\n  [A-Za-z_$]", rest)
    return rest[:end.start()] if end else rest


def test_tracker_offers_restart_without_rebuilding():
    src = (STATIC / "gaze" / "tracker.js").read_text()
    body = _method_body(src, "  async restart() {")
    assert "new lib.WebEyeTrack" not in body, "restart must not build a new model"
    assert "new lib.WebcamClient" in body, "but it does need a fresh camera client"
    assert "this.start()" in body, "and must fall back when there is no model yet"


def test_destroy_releases_the_model_and_the_face_landmarker():
    """Upstream exposes no dispose, but tf.LayersModel has .dispose() and
    MediaPipe's FaceLandmarker has .close(); both are reachable."""
    body = _method_body((STATIC / "gaze" / "tracker.js").read_text(), "  destroy() {")
    assert "model.dispose()" in body
    assert "faceLandmarker.close()" in body


def test_tap_listener_is_bound_once():
    """A listener added per attempt fires N times on the Nth attempt."""
    src = (STATIC / "calibrate.html").read_text()
    assert "_tapBound" in src


def test_previous_run_is_shown_on_screen():
    """A crashed tab cannot report itself, and a server-side-only record is
    no use to someone holding the phone."""
    src = (STATIC / "calibrate.html").read_text()
    assert "showLastRun" in src
    assert "Previous run did not finish" in src



# --- backend selection (iOS tab kill at only 26MB of tensors) --------------

def test_ios_gets_the_cpu_backend_by_default():
    """A real iPhone died mid-calibration holding 26MB of tensors — far too
    little for a heap exhaustion. On WebGL every tensor is a GPU texture and
    iOS Safari's texture budget is much tighter than its heap, so the CPU
    backend removes that failure mode entirely."""
    src = (STATIC / "gaze" / "tracker.js").read_text()
    body = _method_body(src, "  async _selectBackend() {")
    assert "isIOS()" in body and "'cpu'" in body
    assert "setBackend" in body


def test_backend_choice_is_overridable():
    assert "params.get('backend')" in (STATIC / "calibrate.html").read_text()


def test_backend_selection_precedes_model_load():
    """Switching backends after the weights load would not move them."""
    src = (STATIC / "gaze" / "tracker.js").read_text()
    start = _method_body(src, "  async start() {\n    try {")
    assert start.index("_selectBackend") < start.index("new lib.WebEyeTrack")


def test_memory_is_recorded_per_point_not_just_once():
    """One snapshot cannot tell a steady state from a climb."""
    src = (STATIC / "gaze" / "calibration.js").read_text()
    assert "memorySeries" in src
    assert "this.memorySeries.push" in src


def test_backend_is_reported_in_diagnostics():
    src = (STATIC / "gaze" / "calibration.js").read_text()
    assert "backend: this.tracker.backend" in src


# --- systematic offset correction -----------------------------------------

def test_validation_measures_and_removes_a_shared_offset():
    """A real phone read consistently above where the person was looking. The
    adapter is fine-tuned from a pretrained prior in a few steps, so a
    systematic offset can survive calibration — and the validation points
    already measure exactly it."""
    src = (STATIC / "gaze" / "calibration.js").read_text()
    assert "bias()" in src
    assert "residualError" in src
    body = _method_body(src, "  bias() {")
    assert "measured[0] - v.target[0]" in body and "measured[1] - v.target[1]" in body


def test_grade_uses_the_corrected_error():
    """Grading on the uncorrected figure would reject calibrations whose data
    is fine once the shared offset is removed."""
    src = (STATIC / "gaze" / "calibration.js").read_text()
    assert "gradeError(residualError != null ? residualError : meanError)" in src


def test_viewing_samples_have_the_offset_removed():
    src = (STATIC / "calibrate.html").read_text()
    body = src[src.index("function toImageCoords"):src.index("async function runViewingStage")]
    assert "s.x - gazeBias[0]" in body and "s.y - gazeBias[1]" in body


def test_only_the_shared_component_is_removed():
    """Scatter around the offset is genuine measurement error and must stay in
    the numbers; only the mean is subtracted."""
    src = (STATIC / "gaze" / "calibration.js").read_text()
    body = _method_body(src, "  bias() {")
    assert "/ pts.length" in body
