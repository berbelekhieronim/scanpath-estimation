"""FastAPI application.

Phase 1: boot, schema, image sync, /admin. The participant capture view and
the presenter display arrive in Phases 2 and 3 — their routes exist here as
placeholders so the URL structure is settled from the start.
"""

import io
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "tools"))
import gaze_prompts  # noqa: E402  (tools/ is this repo's own code)

from . import analysis, config, db, rounds, urls


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    db.init_db()
    rounds.init_rounds()
    result = db.sync_images_from_disk()
    model_sync = db.sync_model_runs_from_disk()

    token = db.get_or_create_control_token()
    images = db.list_images()

    # If nothing is active yet, open a round on the first image so the app is
    # immediately usable rather than requiring a trip to /admin first.
    if images and not db.get_active_round():
        db.open_round(images[0]["id"])

    print("\n" + "=" * 62)
    print("  scanpath-estimation")
    print("=" * 62)
    print(f"  images found:   {len(images)}")
    if result["added"]:
        print(f"  newly added:    {', '.join(result['added'])}")
    if result["missing"]:
        print(f"  MISSING FILES:  {', '.join(result['missing'])}")
    print(f"  model runs:     {len(model_sync['loaded'])}")
    if model_sync["orphaned"]:
        print(f"  orphaned runs:  {len(model_sync['orphaned'])} "
              f"(no matching image)")
    if model_sync["invalid"]:
        print(f"  INVALID RUNS:   {', '.join(model_sync['invalid'])}")
    synthetic = [r for r in db.list_model_runs() if r["source"] == "synthetic"]
    if synthetic:
        print(f"\n  *** {len(synthetic)} SYNTHETIC run(s) loaded — placeholder")
        print(f"      data, NOT model output. Badged in the UI. ***")
    if not images:
        print("  none yet — drop image files into data/images/ and restart,")
        print("  or use the Rescan button in /admin")
    print(f"\n  START HERE:     /start        <- links to every page")
    print(f"\n  participant:    /            (what phones scan into)")
    print(f"  join screen:    /qr          (project while people join)")
    print(f"  display:        /display     (project during the demo)")
    print(f"  controls:       /control?k={token}")
    print(f"  admin:          /admin?k={token}")
    print(f"\n  control token:  {token}")
    print("=" * 62 + "\n")

    yield


app = FastAPI(title="scanpath-estimation", lifespan=lifespan)


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def require_token(
    k: Optional[str] = Query(None, description="Control token"),
    x_control_token: Optional[str] = Header(None),
) -> str:
    """Gate presenter-only routes.

    Not a security boundary against a determined attacker — it exists so a
    participant who guesses /control cannot clear the display mid-session.
    """
    supplied = k or x_control_token
    expected = db.get_or_create_control_token()
    if not supplied or supplied != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing control token")
    return supplied


class LAYERS:
    """Layer visibility. 'Clear' hides a layer; it never deletes data."""

    # "analysis" is gone: the agreement figures moved to /charts, where they
    # can be laid out and read, instead of crowding the projected picture.
    KEYS = {"heatmap": "1", "paths": "0", "model": "0",
            "prompt": "0", "json": "0", "gaze": "0", "gaze_paths": "0",
            "charts": "0", "grid": "0"}

    # Each of these takes over the whole screen, so two of them on at once
    # means one is silently hidden behind the other. Enforced here rather than
    # in the control page, because the display is what has the constraint.
    EXCLUSIVE = ("json", "charts")

    @classmethod
    def current(cls) -> dict:
        return {
            name: db.get_state(f"layer_{name}", default) == "1"
            for name, default in cls.KEYS.items()
        }

    @classmethod
    def set(cls, name: str, on: bool) -> None:
        if name not in cls.KEYS:
            raise HTTPException(status_code=400, detail=f"Unknown layer: {name}")
        db.set_state(f"layer_{name}", "1" if on else "0")
        if on and name in cls.EXCLUSIVE:
            for other in cls.EXCLUSIVE:
                if other != name:
                    db.set_state(f"layer_{other}", "0")


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

def _page(name: str) -> FileResponse:
    path = config.STATIC_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not built yet")
    return FileResponse(path)


@app.get("/", include_in_schema=False)
def page_capture():
    return _page("index.html")


@app.get("/display", include_in_schema=False)
def page_display():
    return _page("display.html")


@app.get("/control", include_in_schema=False)
def page_control():
    # The page loads for anyone; its API calls are what require the token.
    return _page("control.html")


@app.get("/admin", include_in_schema=False)
def page_admin():
    return _page("admin.html")


@app.get("/qr", include_in_schema=False)
def page_qr():
    return _page("qr.html")


@app.get("/start", include_in_schema=False)
def page_start():
    return _page("start.html")


@app.get("/consent", include_in_schema=False)
def page_consent():
    return _page("consent.html")


@app.get("/calibrate", include_in_schema=False)
def page_calibrate():
    return _page("calibrate.html")


@app.get("/view", include_in_schema=False)
def page_view():
    return _page("view.html")


@app.get("/gazetest", include_in_schema=False)
def page_gazetest():
    """Phase W1 diagnostic: does webcam gaze tracking work on this device?"""
    return _page("gazetest.html")


@app.get("/charts", include_in_schema=False)
def page_charts():
    """The comparison charts: tapped vs measured vs model, side by side."""
    return _page("charts.html")


@app.get("/rounds", include_in_schema=False)
def page_rounds():
    return _page("rounds.html")


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}


# An eye with a fixation cross, inline so there is no binary asset to ship.
FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#5b8cff"/>'
    '<path d="M5 16s4.6-7 11-7 11 7 11 7-4.6 7-11 7-11-7-11-7z" '
    'fill="none" stroke="#fff" stroke-width="2.2"/>'
    '<circle cx="16" cy="16" r="3.3" fill="#fff"/></svg>'
)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(content=FAVICON, media_type="image/svg+xml")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

_STARTED_AT = time.time()


def _server_freshness() -> dict:
    """Is the running process older than the code on disk?

    Static files are read from disk on every request, so a pull updates the
    pages immediately while the Python process keeps running whatever it
    imported at startup. The result is a UI offering features the backend has
    never heard of, which is confusing in a way that wastes a whole test
    cycle. Cheap to detect, so it is detected.
    """
    newest = 0.0
    for path in (config.BASE_DIR / "app").rglob("*.py"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return {
        "started_at": _STARTED_AT,
        "uptime_seconds": int(time.time() - _STARTED_AT),
        "code_mtime": newest,
        "stale": newest > _STARTED_AT + 1,
    }


@app.get("/api/status")
def api_status(request: Request):
    """Everything the start page needs to say whether the app is ready.

    Composed server-side so the start page is one request, and so the
    readiness warnings are decided in one place rather than reimplemented in
    the template.

    Deliberately does NOT include the control token. The start page is
    reachable by anyone who can reach the app, and the token is what stops a
    participant clearing the display mid-session.
    """
    round_ = db.get_active_round()
    image = db.get_image(round_["image_id"]) if round_ else None
    images = db.list_images()
    runs = db.list_model_runs()
    synthetic = [r for r in runs if r["source"] == "synthetic"]
    counts = db.count_responses(round_["id"]) if round_ else {"participants": 0,
                                                              "markers": 0}
    join = urls.public_base_url(request.base_url)
    reachable = urls.is_reachable_by_others(join["url"])

    warnings = []
    if not images:
        warnings.append({
            "level": "error",
            "text": "No images loaded. Put image files in data/images/, then "
                    "press Rescan in Admin.",
        })
    if not reachable:
        port_hint = urls.app_port()
        warnings.append({
            "level": "error",
            "text": f"The join address ({join['url']}) only works on this "
                    f"machine. In a Codespace, set port {port_hint} to Public "
                    f"in the Ports panel.",
        })
    if synthetic:
        warnings.append({
            "level": "warn",
            "text": f"{len(synthetic)} synthetic placeholder run(s) loaded — "
                    f"not model output. Delete data/model/*.json and run "
                    f"precompute.py for real before a session.",
        })
    if images and not runs:
        warnings.append({
            "level": "warn",
            "text": "No model runs yet. The model layer and the agreement "
                    "metrics stay empty until you run precompute.py.",
        })

    gaze = db.gaze_session_stats(round_["id"]) if round_ else {
        "total": 0, "usable": 0, "excluded": 0, "grades": {}}
    gaze_pts = db.round_gaze_points(round_["id"]) if round_ else {
        "points": [], "contributors": 0}
    conditions = db.assignment_counts(round_["id"]) if round_ else None

    return {
        "server": _server_freshness(),
        "capture_mode": db.get_state("capture_mode", "tap"),
        "grid_source": db.get_state("grid_source", "tapped"),
        "view_ms": int(db.get_state("view_ms", str(config.DEFAULT_VIEW_MS))),
        "conditions": conditions,
        "gaze": {
            "sessions": gaze["total"],
            "usable": gaze["usable"],
            "excluded": gaze["excluded"],
            "contributors": gaze_pts["contributors"],
            "samples": len(gaze_pts["points"]),
        },
        "join_url": join["url"],
        "join_url_source": join["source"],
        "join_url_reachable": reachable,
        "images": {"count": len(images), "active": image},
        "model_runs": {"count": len(runs), "synthetic": len(synthetic)},
        "responses": counts,
        "round_id": round_["id"] if round_ else None,
        "layers": LAYERS.current(),
        "warnings": warnings,
    }


@app.get("/api/images")
def api_images():
    return {"images": db.list_images()}


@app.get("/api/state")
def api_state():
    """Everything a client needs to render, in one poll."""
    round_ = db.get_active_round()
    image = db.get_image(round_["image_id"]) if round_ else None
    counts = db.count_responses(round_["id"]) if round_ else {"participants": 0, "markers": 0}
    return {
        "round_id": round_["id"] if round_ else None,
        "image": image,
        "tap_count": int(db.get_state("tap_count", config.DEFAULT_TAP_COUNT)),
        "responses": counts,
        "layers": LAYERS.current(),
        "capture_mode": db.get_state("capture_mode", "tap"),
        "grid_source": db.get_state("grid_source", "tapped"),
        "view_ms": int(db.get_state("view_ms", str(config.DEFAULT_VIEW_MS))),
    }


def model_config() -> dict:
    mode = db.get_state("model_mode", "freeview")
    target = db.get_state("model_target") or None
    probe = gaze_prompts.probe_for(mode, target)
    n = int(db.get_state("model_n_fixations", "5"))
    return {
        "mode": mode,
        "target": target,
        "n_fixations": n,
        "probe": probe,
        # The prompt the model would be given for this configuration. Shown on
        # the display so the audience sees what was actually asked, not just
        # the resulting dots.
        "prompt_text": _safe_prompt(mode, n, target),
    }


def _safe_prompt(mode: str, n: int, target):
    try:
        return gaze_prompts.build_prompt(mode, n, target)
    except ValueError:
        return None


class Assignment(BaseModel):
    participant_uuid: str = Field(min_length=8, max_length=64)
    # Set by a direct join link, which skips the balancer on purpose. Stored
    # as forced so a rehearsal is never mistaken for a balanced participant.
    force: Optional[str] = None

    @field_validator("force")
    @classmethod
    def known_condition(cls, v):
        if v is not None and v not in ("tap", "gaze"):
            raise ValueError("force must be tap or gaze")
        return v


@app.post("/api/assign")
def api_assign(payload: Assignment, request: Request):
    """Tell a participant which condition they are in.

    Between-subjects by design: asking someone to predict where they would
    look and then measuring where they do contaminates the measurement, so a
    participant does one or the other, never both.
    """
    round_ = db.get_active_round()
    if not round_:
        raise HTTPException(status_code=409, detail="No round is open")
    participant_id = db.upsert_participant(
        payload.participant_uuid, request.headers.get("user-agent"))
    mode = db.get_state("capture_mode", "tap")
    result = db.assign_condition(round_["id"], participant_id, mode,
                                 force=payload.force)
    return {**result, "mode": mode,
            "counts": db.assignment_counts(round_["id"])}


@app.get("/api/conditions")
def api_conditions():
    round_ = db.get_active_round()
    if not round_:
        return {"mode": db.get_state("capture_mode", "tap"),
                "counts": {c: {"assigned": 0, "completed": 0} for c in db.CONDITIONS}}
    return {"mode": db.get_state("capture_mode", "tap"),
            "counts": db.assignment_counts(round_["id"])}


@app.post("/api/control/capture-mode")
def api_set_capture_mode(payload: dict, _: str = Depends(require_token)):
    mode = payload.get("mode")
    if mode not in ("tap", "gaze", "mixed"):
        raise HTTPException(status_code=400,
                            detail="mode must be tap, gaze or mixed")
    db.set_state("capture_mode", mode)
    return {"mode": mode}


class ViewMs(BaseModel):
    # Bounded because it is the participant's time, and because a window too
    # short cannot be split into time bins at the rate gaze actually arrives.
    ms: int = Field(ge=3000, le=20000)


@app.post("/api/control/view-ms")
def api_set_view_ms(payload: ViewMs, _: str = Depends(require_token)):
    """How long the picture stays on screen for the eye-tracked group.

    Tap count has always been a presenter setting; this was a URL parameter
    only, so the two groups' effort was controlled in different places and
    one of them could not be changed at all without editing a link. It also
    decides whether any temporal analysis is possible: at roughly 3Hz, five
    seconds is about seven samples per half-window, which is too few to
    split (SPEC-METRICS section 10).
    """
    db.set_state("view_ms", str(payload.ms))
    return {"view_ms": payload.ms}


class GridSource(BaseModel):
    source: str

    @field_validator("source")
    @classmethod
    def known(cls, v):
        if v not in ("tapped", "measured", "model"):
            raise ValueError("source must be tapped, measured or model")
        return v


@app.post("/api/control/grid-source")
def api_set_grid_source(payload: GridSource, _: str = Depends(require_token)):
    """Which group the on-image attention grid draws.

    A separate setting from the layer toggle: the presenter switches between
    the two human groups on the same picture, which is the comparison the
    whole demo is for, and that should not mean turning a layer off and a
    different one on.
    """
    db.set_state("grid_source", payload.source)
    return {"grid_source": payload.source}


@app.get("/api/probes")
def api_probes():
    """The probe catalogue, for the control page's dropdown."""
    return {"probes": gaze_prompts.PROBES}


@app.get("/api/model")
def api_model():
    """The model run matching the active image and current model config.

    Returns a shape with run=None rather than 404 when nothing is precomputed,
    because 'no run yet' is an ordinary state the display renders as an empty
    layer, not an error.
    """
    round_ = db.get_active_round()
    cfg = model_config()
    if not round_:
        return {"run": None, "config": cfg, "reason": "no round open"}

    run = db.get_model_run(round_["image_id"], cfg["mode"], cfg["target"],
                           cfg["n_fixations"])
    if not run:
        # Fall back to any fixation count for this mode/target, so a run made
        # at a different length still shows rather than silently missing.
        run = db.get_model_run(round_["image_id"], cfg["mode"], cfg["target"])
    if not run:
        return {"run": None, "config": cfg,
                "reason": "no precomputed run for this image and configuration"}
    return {"run": run, "config": cfg}


@app.get("/api/model/runs")
def api_model_runs():
    return {"runs": db.list_model_runs()}


@app.get("/api/analysis")
def api_analysis():
    """Agreement between the **tap group's** responses and the model.

    Measured gaze is not in this comparison and never has been — it reads
    markers, and markers only ever come from tapping. That was harmless while
    tapping was the only condition; with two groups on screen, a panel headed
    "agreement" that silently means one of them is a trap. So the scope is
    declared in the payload and printed on the panel. For all three sources,
    /api/compare/maps is the endpoint, and the charts layer is the view.

    Computed on demand rather than cached: it takes milliseconds at this
    scale, and a stale panel during a live reveal would be worse than a
    recomputation.
    """
    round_ = db.get_active_round()
    if not round_:
        return {"ok": False, "reason": "no round open"}

    rows = db.get_round_markers(round_["id"])
    by_participant: dict[int, list] = {}
    for r in rows:
        by_participant.setdefault(r["participant_id"], []).append([r["x"], r["y"]])

    cfg = model_config()
    run = (db.get_model_run(round_["image_id"], cfg["mode"], cfg["target"],
                            cfg["n_fixations"])
           or db.get_model_run(round_["image_id"], cfg["mode"], cfg["target"]))
    model_paths = (run.get("samples_norm") or [run.get("scanpath_norm")]) if run else []

    result = analysis.analyse(list(by_participant.values()),
                              [p for p in model_paths if p])
    if run:
        result["model_source"] = run.get("source")
    result["compares"] = {
        "a": "tapped",
        "b": "model",
        "n_a": len(by_participant),
        "n_b": len([p for p in model_paths if p]),
        "excludes": "measured",
    }
    return result


@app.get("/api/raw")
def api_raw():
    """Everything the current screen is built from, as one JSON document.

    For showing an audience that the picture is computed from data, not drawn.
    Public, like every other read endpoint, and carries no token.
    """
    return {
        "note": "This is exactly what the display is rendering, nothing hidden.",
        "state": api_state(),
        "model": api_model(),
        "markers": api_markers(),
        "analysis": api_analysis(),
    }


def _condition_paths(round_id: int) -> dict:
    """Tap paths and gaze paths, each restricted to their assigned group."""
    tap_ids = db.participants_in_condition(round_id, "tap")
    gaze_ids = db.participants_in_condition(round_id, "gaze")

    by_participant: dict = {}
    for r in db.get_round_markers(round_id):
        by_participant.setdefault(r["participant_id"], []).append([r["x"], r["y"]])

    # Markers only ever come from tapping, so anyone holding markers tapped.
    # The only reason to drop one is an explicit assignment to the other
    # group. The previous rule kept only assigned tappers, falling back to
    # "everyone" when the assignment table was completely empty — so a single
    # stray assignment (one phone opening the participant page) flipped it and
    # silently discarded every tapper who had submitted without one.
    tap_paths = [v for k, v in by_participant.items() if k not in gaze_ids]

    gaze = db.round_gaze_points(round_id)
    return {"tap": tap_paths, "gaze": gaze["paths"],
            # Taps carry no measurement error of their own — a tap is where
            # the finger landed — so they take the full shared kernel.
            "tap_errors": [None] * len(tap_paths),
            "gaze_errors": gaze.get("errors") or [],
            "gaze_times": gaze.get("sample_times") or [],
            "gaze_excluded": gaze["excluded"]}


@app.get("/api/compare")
def api_compare(grid: int = 3):
    """Between-subjects comparison: tap group vs gaze group vs model."""
    round_ = db.get_active_round()
    if not round_:
        return {"ok": False, "reason": "no round open"}

    paths = _condition_paths(round_["id"])
    cfg = model_config()
    run = (db.get_model_run(round_["image_id"], cfg["mode"], cfg["target"],
                            cfg["n_fixations"])
           or db.get_model_run(round_["image_id"], cfg["mode"], cfg["target"]))
    model_paths = (run.get("samples_norm") or [run.get("scanpath_norm")]) if run else []

    result = analysis.compare_sources({
        "tap": paths["tap"],
        "gaze": paths["gaze"],
        "model": [p for p in model_paths if p],
    }, grid=max(2, min(5, grid)))
    result["conditions"] = db.assignment_counts(round_["id"])
    result["gaze_excluded"] = paths["gaze_excluded"]
    result["model_source"] = run.get("source") if run else None
    return result


@app.get("/api/compare/maps")
def api_compare_maps(grid: int = 3, sigma: float = analysis.SIGMA_DEFAULT):
    """Per-participant density maps for the three sources, on one scale.

    Separate from /api/compare because it answers a different question. That
    one scores agreement on pooled points; this one returns the maps the
    charts draw, each built per participant and smoothed with one shared
    kernel so the panels are actually comparable (SPEC-METRICS.md section 2).
    """
    round_ = db.get_active_round()
    if not round_:
        return {"ok": False, "reason": "no round open"}

    paths = _condition_paths(round_["id"])
    cfg = model_config()
    run = (db.get_model_run(round_["image_id"], cfg["mode"], cfg["target"],
                            cfg["n_fixations"])
           or db.get_model_run(round_["image_id"], cfg["mode"], cfg["target"]))
    model_paths = (run.get("samples_norm") or [run.get("scanpath_norm")]) if run else []

    result = analysis.comparison_maps(
        {
            "tapped": paths["tap"],
            "measured": paths["gaze"],
            "model": [p for p in model_paths if p],
        },
        n=max(2, min(5, grid)),
        sigma=max(0.05, min(0.4, sigma)),
        errors={"tapped": paths["tap_errors"],
                "measured": paths["gaze_errors"]},
    )
    result["image"] = db.get_image(round_["image_id"])
    result["config"] = cfg
    result["conditions"] = db.assignment_counts(round_["id"])
    result["gaze_excluded"] = paths["gaze_excluded"]
    result["model_source"] = run.get("source") if run else None
    return result


@app.get("/api/markers")
def api_markers():
    """Every response for the open round, grouped into per-participant paths.

    Returned as ordered coordinate lists rather than flat rows so the display
    can draw scanpaths without regrouping, and so the payload stays small
    enough to poll every two seconds without thinking about it.
    """
    round_ = db.get_active_round()
    if not round_:
        return {"round_id": None, "paths": [], "points": [], "count": 0}

    rows = db.get_round_markers(round_["id"])
    paths: dict[int, list] = {}
    for r in rows:
        paths.setdefault(r["participant_id"], []).append([r["x"], r["y"]])

    ordered = list(paths.values())
    return {
        "round_id": round_["id"],
        "paths": ordered,
        "points": [pt for path in ordered for pt in path],
        "count": len(ordered),
    }


class Submission(BaseModel):
    round_id: int
    participant_uuid: str = Field(min_length=8, max_length=64)
    device: Optional[dict] = None
    points: list[tuple[float, float]] = Field(min_length=1, max_length=50)

    @field_validator("points")
    @classmethod
    def in_unit_square(cls, v):
        for x, y in v:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("coordinates must be normalised to 0.0-1.0")
        return v


@app.post("/api/markers")
def api_submit_markers(sub: Submission, request: Request):
    """Accept one participant's taps for the round they were shown.

    The round_id check matters: if the presenter moved to the next image while
    someone was mid-tap, their taps belong to the image they actually saw, not
    the one now on screen. Rejecting with 409 lets the client reload cleanly
    rather than silently filing data against the wrong stimulus.
    """
    active = db.get_active_round()
    if not active:
        raise HTTPException(status_code=409, detail="No round is open")
    if sub.round_id != active["id"]:
        raise HTTPException(
            status_code=409,
            detail="The image changed while you were tapping. Reloading.",
        )

    participant_id = db.upsert_participant(
        sub.participant_uuid, request.headers.get("user-agent")
    )
    if sub.device:
        db.set_participant_device(participant_id, sub.device)
    n = db.save_markers(active["id"], participant_id, sub.points)
    db.mark_completed(active["id"], participant_id)
    return {"saved": n, "round_id": active["id"], "responses": db.count_responses(active["id"])}


class GazeSession(BaseModel):
    participant_uuid: str = Field(min_length=8, max_length=64)
    grade: str
    tracker: Optional[str] = None
    tracker_version: Optional[str] = None
    mean_error: Optional[float] = None
    worst_error: Optional[float] = None
    points_accepted: Optional[int] = None
    points_total: Optional[int] = None
    validation: Optional[list] = None
    viewport_w: Optional[int] = None
    viewport_h: Optional[int] = None
    # Which way up the device was. A calibration is a mapping fitted to one
    # screen shape, so a session taken in landscape is not comparable to one
    # taken in portrait and has to be readable back as such.
    orientation: Optional[str] = None
    device_label: Optional[str] = None
    device: Optional[dict] = None
    failure: Optional[str] = None
    diagnostics: Optional[dict] = None
    residual_error: Optional[float] = None
    bias: Optional[list[float]] = None
    # Per-axis gain and offset measured at the validation points. Recorded so
    # a session can be re-derived later knowing exactly what was corrected.
    fit: Optional[dict] = None

    @field_validator("grade")
    @classmethod
    def known_grade(cls, v):
        if v not in ("good", "usable", "poor", "failed"):
            raise ValueError("grade must be good, usable, poor or failed")
        return v


@app.post("/api/gaze/session")
def api_gaze_session(session: GazeSession, request: Request):
    """Record a calibration result, passed or failed.

    Failures are stored deliberately. The proportion of participants whose
    tracking did not work is a number the presenter has to be able to report;
    discarding failures would make every other figure look better than it is.
    """
    round_ = db.get_active_round()
    participant_id = db.upsert_participant(
        session.participant_uuid, request.headers.get("user-agent"))
    if session.device:
        db.set_participant_device(participant_id, session.device)
    payload = session.model_dump()
    session_id = db.create_gaze_session(
        participant_id, round_["id"] if round_ else None, payload)
    return {
        "session_id": session_id,
        "grade": session.grade,
        "usable": session.grade in ("good", "usable"),
        "stats": db.gaze_session_stats(round_["id"] if round_ else None),
    }


class GazeSamples(BaseModel):
    # Either identifier works. The session id is preferred, but it used to be
    # carried between stages in localStorage, and if that write was lost the
    # recording had nowhere to go — the participant saw "recorded, but not
    # saved" with their data discarded. A uuid fallback means a completed
    # viewing is never thrown away.
    session_id: Optional[int] = None
    participant_uuid: Optional[str] = None
    samples: list[dict] = Field(max_length=2000)
    duration_ms: Optional[int] = None
    blinks: Optional[int] = None
    off_image: Optional[int] = None

    @field_validator("samples")
    @classmethod
    def coordinates_in_range(cls, v):
        for s in v:
            x, y = s.get("x"), s.get("y")
            if x is None or y is None:
                continue          # off-image samples carry no coordinates
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("coordinates must be normalised to 0.0-1.0")
        return v


@app.post("/api/gaze/samples")
def api_gaze_samples(payload: GazeSamples):
    """Store one viewing window, uploaded as a batch at the end.

    Batched rather than streamed on purpose: a dropped connection mid-stream
    would lose a partial recording and leave no way to tell a short window
    from a truncated one.
    """
    sess = db.get_gaze_session(payload.session_id) if payload.session_id else None
    if not sess and payload.participant_uuid:
        sess = db.latest_gaze_session_for(payload.participant_uuid)
    if not sess:
        raise HTTPException(
            status_code=404,
            detail="No calibration on record for this participant")

    n = db.save_gaze_samples(sess["id"], payload.samples)
    round_ = db.get_active_round()
    if round_ and not sess["excluded"]:
        db.mark_completed(round_["id"], sess["participant_id"])
    return {
        "saved": n,
        "session_id": sess["id"],
        "excluded": bool(sess["excluded"]),
        "stats": db.gaze_session_stats(round_["id"] if round_ else None),
    }


@app.get("/api/gaze/aggregate")
def api_gaze_aggregate():
    """Pooled measured gaze for the current round, for the display."""
    round_ = db.get_active_round()
    if not round_:
        return {"paths": [], "points": [], "contributors": 0,
                "sessions": 0, "excluded": 0}
    return db.round_gaze_points(round_["id"])


@app.get("/api/gaze/stats")
def api_gaze_stats():
    round_ = db.get_active_round()
    return db.gaze_session_stats(round_["id"] if round_ else None)


@app.get("/api/join-url")
def api_join_url(request: Request):
    info = urls.public_base_url(request.base_url)
    return {**info, "reachable": urls.is_reachable_by_others(info["url"])}


@app.get("/api/qr.svg", include_in_schema=False)
def api_qr(request: Request, path: str = ""):
    """The join QR, optionally for a specific entry point.

    `path` lets the start page publish a tap-only and a track-only code
    alongside the balanced one, for rehearsing either arm without having to
    wait for the balancer to hand you the one you wanted. Restricted to a
    relative path so this cannot be turned into a generator of QR codes
    pointing anywhere.
    """
    import qrcode
    import qrcode.image.svg

    url = urls.public_base_url(request.base_url)["url"]
    if path:
        if not path.startswith("/") or path.startswith("//") or ":" in path:
            raise HTTPException(status_code=400,
                                detail="path must be a relative path")
        url = url.rstrip("/") + path
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, border=2)
    buf = io.BytesIO()
    img.save(buf)
    svg = buf.getvalue().decode()

    # The library hard-codes a millimetre width/height. Strip them so the
    # viewBox alone governs and CSS can scale it to fill a projector.
    svg = re.sub(r'\s(width|height)="[^"]*"', "", svg, count=2)

    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-store"},
    )


# --------------------------------------------------------------------------
# Presenter API (token-gated)
# --------------------------------------------------------------------------

@app.post("/api/admin/rescan")
def api_rescan(_: str = Depends(require_token)):
    result = db.sync_images_from_disk()
    return {"result": result, "images": db.list_images()}


@app.post("/api/admin/active")
def api_set_active(image_id: int = Query(...), _: str = Depends(require_token)):
    image = db.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="No such image")
    previous = db.get_active_round()
    if previous:
        rounds.freeze_settings(previous["id"])
    round_id = db.open_round(image_id)
    return {"round_id": round_id, "image": image}


@app.post("/api/control/layers")
def api_set_layers(payload: dict, _: str = Depends(require_token)):
    for name, on in payload.items():
        LAYERS.set(name, bool(on))
    return {"layers": LAYERS.current()}


@app.post("/api/control/model-config")
def api_set_model_config(payload: dict, _: str = Depends(require_token)):
    # A probe id sets mode and target together, which is what the UI uses.
    if "probe" in payload:
        probe = gaze_prompts.PROBES_BY_ID.get(payload["probe"])
        if not probe:
            raise HTTPException(status_code=400,
                                detail=f"unknown probe: {payload['probe']!r}")
        db.set_state("model_mode", probe["mode"])
        db.set_state("model_target", probe["target"] or "")
    else:
        if "mode" in payload:
            if payload["mode"] not in ("freeview", "search", "probe"):
                raise HTTPException(
                    status_code=400,
                    detail="mode must be freeview, search or probe")
            db.set_state("model_mode", payload["mode"])
        if "target" in payload:
            db.set_state("model_target", payload["target"] or "")
    if "n_fixations" in payload:
        db.set_state("model_n_fixations", int(payload["n_fixations"]))
    return {"config": model_config()}


class PushedRun(BaseModel):
    image: str
    mode: str
    n_fixations: int
    scanpath_norm: list[list[float]]
    target: Optional[str] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None
    prompt_text: Optional[str] = None
    prompt_kind: str = "trained"     # "trained" | "custom" (off-distribution)
    scanpath_grid: Optional[list[list[int]]] = None
    samples_norm: Optional[list[list[list[float]]]] = None
    samples_grid: Optional[list[list[list[int]]]] = None
    source: str = "pushed"
    model: Optional[str] = None
    device: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    created_at: Optional[str] = None

    @field_validator("scanpath_norm")
    @classmethod
    def in_unit_square(cls, v):
        if not v:
            raise ValueError("scanpath_norm is empty")
        for x, y in v:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("coordinates must be normalised to 0.0-1.0")
        return v

    @field_validator("mode")
    @classmethod
    def known_mode(cls, v):
        if v not in ("freeview", "search", "probe"):
            raise ValueError("mode must be freeview, search or probe")
        return v


@app.post("/api/model/push")
def api_push_model_run(run: PushedRun, _: str = Depends(require_token)):
    """Accept a scanpath run computed elsewhere — Tier C in the spec.

    This is what makes live inference from a laptop possible without the
    Codespace ever seeing a GPU: run the model on the machine that has one,
    POST the result here, and the display picks it up on its next poll.
    """
    images = {i["filename"]: i for i in db.list_images()}
    image = images.get(run.image)
    if not image:
        raise HTTPException(
            status_code=404,
            detail=f"No image named {run.image!r}. Known: {sorted(images)}",
        )

    payload = run.model_dump()
    payload.setdefault("created_at", db.utcnow())
    run_id = db.upsert_model_run(image["id"], payload)
    return {"run_id": run_id, "image_id": image["id"],
            "stored_as": payload["source"]}


@app.get("/api/export")
def api_export(_: str = Depends(require_token)):
    """Everything needed to reconstruct a session offline.

    A Codespace is deleted after a retention period, taking its SQLite file
    with it. Participant responses are the only thing in this project that
    cannot be regenerated, so exporting after each session is not optional.
    """
    import json as _json

    with db.connect() as conn:
        images = [dict(r) for r in conn.execute("SELECT * FROM images")]
        rounds = [dict(r) for r in conn.execute("SELECT * FROM rounds ORDER BY id")]
        participants = [dict(r) for r in conn.execute(
            "SELECT id, uuid, first_seen, device_json FROM participants ORDER BY id")]
        markers = [dict(r) for r in conn.execute(
            "SELECT * FROM markers ORDER BY round_id, participant_id, seq")]
        runs = [dict(r) for r in conn.execute("SELECT * FROM model_runs ORDER BY id")]

    for r in runs:
        try:
            r["payload"] = _json.loads(r.pop("coords_json"))
        except Exception:
            r["payload"] = None

    # Browser and screen details, useful for explaining why one participant's
    # tracking was worse than another's.
    for pt in participants:
        raw = pt.pop("device_json", None)
        try:
            pt["device"] = _json.loads(raw) if raw else None
        except Exception:
            pt["device"] = None

    return {
        "exported_at": db.utcnow(),
        "schema": 1,
        "counts": {
            "images": len(images), "rounds": len(rounds),
            "participants": len(participants), "markers": len(markers),
            "model_runs": len(runs),
        },
        "images": images, "rounds": rounds, "participants": participants,
        "markers": markers, "model_runs": runs,
    }


@app.post("/api/control/rescan-model")
def api_rescan_model(_: str = Depends(require_token)):
    return {"result": db.sync_model_runs_from_disk(), "runs": db.list_model_runs()}


@app.post("/api/control/reset-round")
def api_reset_round(_: str = Depends(require_token)):
    """Open a fresh round on the same image.

    The previous round is closed, not deleted — its responses stay in the
    database and can still be exported. This is for running the same stimulus
    with a second group, not for discarding the first.
    """
    round_ = db.get_active_round()
    if not round_:
        raise HTTPException(status_code=409, detail="No round is open")
    rounds.freeze_settings(round_["id"])
    new_id = db.open_round(round_["image_id"])
    return {"round_id": new_id, "previous_round_id": round_["id"]}


# --------------------------------------------------------------------------
# Rounds
# --------------------------------------------------------------------------

@app.get("/api/rounds")
def api_rounds():
    """Every sitting this database holds.

    Public, like the other read endpoints: it carries counts and timestamps,
    nothing a participant could not see on the screen anyway.
    """
    return {"rounds": rounds.list_rounds()}


@app.get("/api/rounds/{round_id}/export")
def api_round_export(round_id: int, _: str = Depends(require_token)):
    """One round as a file, named by date so a folder of them sorts itself."""
    try:
        snapshot = rounds.export_round(round_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return JSONResponse(
        content=snapshot,
        headers={"Content-Disposition":
                 f'attachment; filename="{rounds.export_filename(snapshot)}"'})


@app.get("/api/rounds/export-all")
def api_rounds_export_all(_: str = Depends(require_token)):
    bundle = rounds.export_all()
    return JSONResponse(
        content=bundle,
        headers={"Content-Disposition":
                 f'attachment; filename="{rounds.bundle_filename(bundle)}"'})


@app.post("/api/rounds/import")
def api_round_import(payload: dict, duplicate: bool = False,
                     _: str = Depends(require_token)):
    """Read a snapshot back in as a new round.

    Never overwrites: a restore that clobbers is how you lose live data
    twenty minutes before a talk. Re-importing the same file does nothing
    unless duplicate=1 says you meant it.
    """
    try:
        return rounds.import_any(payload, duplicate=duplicate)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/rounds/{round_id}/activate")
def api_round_activate(round_id: int, _: str = Depends(require_token)):
    """Put a past or imported round back on the screen, settings and all."""
    current = db.get_active_round()
    if current and current["id"] != round_id:
        rounds.freeze_settings(current["id"])
    try:
        return rounds.activate(round_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/admin/token-check")
def api_token_check(_: str = Depends(require_token)):
    return {"ok": True}


@app.exception_handler(404)
def not_found(request: Request, exc):
    """Say *why* a page is missing when the reason is a stale process.

    Routes are registered when Python imports this module; static files are
    read from disk per request. So after a `git pull` the pages and the links
    to them update at once while the running server still serves the route
    table it started with, and a brand-new page answers a bare "Not Found"
    that looks like a bug in the page. It has cost two debugging sessions
    already, so that one case now explains itself.

    Only that case. A 404 an endpoint raised on purpose already says
    something more useful than anything guessable from here, and is passed
    through untouched.
    """
    detail = getattr(exc, "detail", None)
    if detail and detail != "Not Found":
        return JSONResponse(status_code=404, content={"detail": detail})

    fresh = _server_freshness()
    if fresh["stale"]:
        return JSONResponse(status_code=404, content={
            "detail": f"No route {request.url.path} — but this server has "
                      f"been running {fresh['uptime_seconds'] // 60} minutes "
                      f"and the code on disk is newer than the process. "
                      f"Restart it (Ctrl-C, then ./tools/devserver.sh) and "
                      f"try again.",
            "stale_server": True,
        })
    return JSONResponse(status_code=404,
                        content={"detail": f"No route {request.url.path}"})


# --------------------------------------------------------------------------
# Static mounts
# --------------------------------------------------------------------------

config.ensure_dirs()
app.mount("/img", StaticFiles(directory=config.IMAGES_DIR), name="images")
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
# The WebEyeTrack bundle hard-codes the path "/web/model.json", so the
# BlazeGaze weights have to be served from the site root, not under /static.
app.mount("/web", StaticFiles(directory=config.STATIC_DIR / "web"), name="gazemodel")
