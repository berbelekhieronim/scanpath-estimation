"""Paths and settings. Everything filesystem-related resolves from here."""

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("SCANPATH_DATA_DIR", BASE_DIR / "data"))
IMAGES_DIR = DATA_DIR / "images"
MODEL_DIR = DATA_DIR / "model"
EXPORTS_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "app.sqlite"

STATIC_DIR = BASE_DIR / "app" / "static"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Taps requested per participant. See spec section 10, decision 3.
DEFAULT_TAP_COUNT = 5

# Presenter routes (/control, /admin) are gated by this token so a participant
# who guesses the URL cannot wipe data mid-session. Set SCANPATH_CONTROL_TOKEN
# to pin it; otherwise one is generated on first run and persisted in the
# database, so it stays stable across restarts.
CONTROL_TOKEN_ENV = os.environ.get("SCANPATH_CONTROL_TOKEN") or None


def generate_token() -> str:
    return secrets.token_urlsafe(9)


def ensure_dirs() -> None:
    for d in (DATA_DIR, IMAGES_DIR, MODEL_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
