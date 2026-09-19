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

from . import config, db, urls


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    db.init_db()
    result = db.sync_images_from_disk()

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
    if not images:
        print("  none yet — drop image files into data/images/ and restart,")
        print("  or use the Rescan button in /admin")
    print(f"\n  participant:    /")
    print(f"  presenter:      /display")
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
        "layers": {
            "human": db.get_state("layer_human", "1") == "1",
            "model": db.get_state("layer_model", "0") == "1",
            "analysis": db.get_state("layer_analysis", "0") == "1",
        },
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


@app.get("/api/admin/token-check")
def api_token_check(_: str = Depends(require_token)):
    return {"ok": True}


# --------------------------------------------------------------------------
# Static mounts
# --------------------------------------------------------------------------

config.ensure_dirs()
app.mount("/img", StaticFiles(directory=config.IMAGES_DIR), name="images")
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
