"""FastAPI application.

Phase 1: boot, schema, image sync, /admin. The participant capture view and
the presenter display arrive in Phases 2 and 3 — their routes exist here as
placeholders so the URL structure is settled from the start.
"""

import io
import re
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "tools"))
import gaze_prompts  # noqa: E402  (tools/ is this repo's own code)

from . import analysis, config, db, urls


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    db.init_db()
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

    KEYS = {"heatmap": "1", "paths": "0", "model": "0", "analysis": "0",
            "prompt": "0", "json": "0"}

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


@app.get("/gazetest", include_in_schema=False)
def page_gazetest():
    """Phase W1 diagnostic: does webcam gaze tracking work on this device?"""
    return _page("gazetest.html")


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
        warnings.append({
            "level": "error",
            "text": f"The join address ({join['url']}) only works on this "
                    f"machine. In a Codespace, set port 8000 to Public in the "
                    f"Ports panel.",
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

    return {
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
    """Agreement between the room's taps and the model's prediction.

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
    n = db.save_markers(active["id"], participant_id, sub.points)
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
    device_label: Optional[str] = None
    failure: Optional[str] = None

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
    payload = session.model_dump()
    session_id = db.create_gaze_session(
        participant_id, round_["id"] if round_ else None, payload)
    return {
        "session_id": session_id,
        "grade": session.grade,
        "usable": session.grade in ("good", "usable"),
        "stats": db.gaze_session_stats(round_["id"] if round_ else None),
    }


@app.get("/api/gaze/stats")
def api_gaze_stats():
    round_ = db.get_active_round()
    return db.gaze_session_stats(round_["id"] if round_ else None)


@app.get("/api/join-url")
def api_join_url(request: Request):
    info = urls.public_base_url(request.base_url)
    return {**info, "reachable": urls.is_reachable_by_others(info["url"])}


@app.get("/api/qr.svg", include_in_schema=False)
def api_qr(request: Request):
    import qrcode
    import qrcode.image.svg

    url = urls.public_base_url(request.base_url)["url"]
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
            "SELECT id, uuid, first_seen FROM participants ORDER BY id")]
        markers = [dict(r) for r in conn.execute(
            "SELECT * FROM markers ORDER BY round_id, participant_id, seq")]
        runs = [dict(r) for r in conn.execute("SELECT * FROM model_runs ORDER BY id")]

    for r in runs:
        try:
            r["payload"] = _json.loads(r.pop("coords_json"))
        except Exception:
            r["payload"] = None

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
    new_id = db.open_round(round_["image_id"])
    return {"round_id": new_id, "previous_round_id": round_["id"]}


@app.get("/api/admin/token-check")
def api_token_check(_: str = Depends(require_token)):
    return {"ok": True}


# --------------------------------------------------------------------------
# Static mounts
# --------------------------------------------------------------------------

config.ensure_dirs()
app.mount("/img", StaticFiles(directory=config.IMAGES_DIR), name="images")
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
# The WebEyeTrack bundle hard-codes the path "/web/model.json", so the
# BlazeGaze weights have to be served from the site root, not under /static.
app.mount("/web", StaticFiles(directory=config.STATIC_DIR / "web"), name="gazemodel")
