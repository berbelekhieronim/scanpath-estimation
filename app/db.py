"""SQLite schema and data access.

A connection per operation, which SQLite handles well at this scale and which
sidesteps the thread-affinity problem entirely. WAL mode so the presenter
display polling never blocks a participant submitting.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL UNIQUE,
    label       TEXT    NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    sort_order  INTEGER NOT NULL DEFAULT 0,
    present     INTEGER NOT NULL DEFAULT 1,  -- 0 if the file has gone missing
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS rounds (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id   INTEGER NOT NULL REFERENCES images(id),
    label      TEXT,
    opened_at  TEXT    NOT NULL,
    closed_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_rounds_image ON rounds(image_id);

CREATE TABLE IF NOT EXISTS participants (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid       TEXT    NOT NULL UNIQUE,
    first_seen TEXT    NOT NULL,
    user_agent TEXT
);

-- x and y are normalised to 0.0-1.0 against the image box, so a phone at
-- 390px and a projector at 1920px produce comparable data. seq is the tap
-- order, which is the part that makes this a scanpath rather than a heatmap.
CREATE TABLE IF NOT EXISTS markers (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id       INTEGER NOT NULL REFERENCES rounds(id),
    participant_id INTEGER NOT NULL REFERENCES participants(id),
    seq            INTEGER NOT NULL,
    x              REAL    NOT NULL,
    y              REAL    NOT NULL,
    created_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_markers_round ON markers(round_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_markers_unique
    ON markers(round_id, participant_id, seq);

CREATE TABLE IF NOT EXISTS model_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id     INTEGER NOT NULL REFERENCES images(id),
    mode         TEXT    NOT NULL,             -- 'freeview' | 'search'
    target       TEXT,                         -- search target, else NULL
    n_fixations  INTEGER NOT NULL,
    seed         INTEGER,
    temperature  REAL,
    prompt_text  TEXT,
    coords_json  TEXT    NOT NULL,             -- see spec section 6.3
    source       TEXT    NOT NULL,             -- 'precomputed'|'live'|'pushed'
    created_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_runs_image ON model_runs(image_id);

CREATE TABLE IF NOT EXISTS app_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
    init_gaze()


# --------------------------------------------------------------------------
# app_state
# --------------------------------------------------------------------------

def get_state(key: str, default: Optional[str] = None) -> Optional[str]:
    with connect() as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_state(key: str, value: Any) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, None if value is None else str(value)),
        )


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------

def list_images(include_missing: bool = False) -> list[dict]:
    sql = "SELECT * FROM images"
    if not include_missing:
        sql += " WHERE present = 1"
    sql += " ORDER BY sort_order, filename"
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql)]


def get_image(image_id: int) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    return dict(row) if row else None


def sync_images_from_disk() -> dict:
    """Reconcile the images table with what is actually in data/images/.

    Adds new files, reads their real pixel dimensions, and flags rows whose
    file has disappeared as present=0 rather than deleting them — a round may
    still reference the image, and losing that history to a stray file move
    would be worse than carrying a dead row.
    """
    from PIL import Image  # local import: keeps startup cheap

    config.ensure_dirs()
    on_disk = {
        p.name
        for p in config.IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    }

    added, restored, missing = [], [], []
    with connect() as conn:
        known = {r["filename"]: dict(r) for r in conn.execute("SELECT * FROM images")}

        for name in sorted(on_disk):
            path = config.IMAGES_DIR / name
            if name in known:
                if not known[name]["present"]:
                    conn.execute("UPDATE images SET present = 1 WHERE id = ?", (known[name]["id"],))
                    restored.append(name)
                continue
            try:
                with Image.open(path) as im:
                    width, height = im.size
            except Exception:
                continue  # not a readable image; ignore rather than fail the scan
            conn.execute(
                "INSERT INTO images (filename, label, width, height, sort_order, present, created_at) "
                "VALUES (?, ?, ?, ?, ?, 1, ?)",
                (name, Path(name).stem, width, height, len(known) + len(added), utcnow()),
            )
            added.append(name)

        for name, row in known.items():
            if name not in on_disk and row["present"]:
                conn.execute("UPDATE images SET present = 0 WHERE id = ?", (row["id"],))
                missing.append(name)

    return {"added": added, "restored": restored, "missing": missing}


# --------------------------------------------------------------------------
# rounds
# --------------------------------------------------------------------------

def open_round(image_id: int, label: Optional[str] = None) -> int:
    """Close the current round and open a fresh one for image_id."""
    with connect() as conn:
        current = conn.execute(
            "SELECT id FROM rounds WHERE closed_at IS NULL ORDER BY id DESC"
        ).fetchall()
        for row in current:
            conn.execute("UPDATE rounds SET closed_at = ? WHERE id = ?", (utcnow(), row["id"]))
        cur = conn.execute(
            "INSERT INTO rounds (image_id, label, opened_at) VALUES (?, ?, ?)",
            (image_id, label, utcnow()),
        )
        round_id = cur.lastrowid
    set_state("active_round_id", round_id)
    set_state("active_image_id", image_id)
    return round_id


def get_active_round() -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM rounds WHERE closed_at IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def count_responses(round_id: int) -> dict:
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT participant_id) AS participants, COUNT(*) AS markers "
            "FROM markers WHERE round_id = ?",
            (round_id,),
        ).fetchone()
    return {"participants": row["participants"], "markers": row["markers"]}


# --------------------------------------------------------------------------
# control token
# --------------------------------------------------------------------------

def get_or_create_control_token() -> str:
    if config.CONTROL_TOKEN_ENV:
        return config.CONTROL_TOKEN_ENV
    token = get_state("control_token")
    if not token:
        token = config.generate_token()
        set_state("control_token", token)
    return token


# --------------------------------------------------------------------------
# participants and markers
# --------------------------------------------------------------------------

def upsert_participant(uuid: str, user_agent: Optional[str] = None) -> int:
    with connect() as conn:
        row = conn.execute("SELECT id FROM participants WHERE uuid = ?", (uuid,)).fetchone()
        if row:
            return row["id"]
        cur = conn.execute(
            "INSERT INTO participants (uuid, first_seen, user_agent) VALUES (?, ?, ?)",
            (uuid, utcnow(), (user_agent or "")[:300]),
        )
        return cur.lastrowid


def save_markers(round_id: int, participant_id: int, points: list[tuple[float, float]]) -> int:
    """Replace this participant's markers for this round.

    Replacing rather than rejecting means a resubmission is idempotent and a
    participant who taps 'back' and submits again gets a sensible result,
    instead of a 409 they cannot act on.
    """
    now = utcnow()
    with connect() as conn:
        conn.execute(
            "DELETE FROM markers WHERE round_id = ? AND participant_id = ?",
            (round_id, participant_id),
        )
        conn.executemany(
            "INSERT INTO markers (round_id, participant_id, seq, x, y, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(round_id, participant_id, i, x, y, now) for i, (x, y) in enumerate(points)],
        )
    return len(points)


def get_round_markers(round_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT participant_id, seq, x, y FROM markers "
            "WHERE round_id = ? ORDER BY participant_id, seq",
            (round_id,),
        )
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# model runs
# --------------------------------------------------------------------------

def upsert_model_run(image_id: int, payload: dict) -> int:
    """Store a scanpath run. One row per (image, mode, target, n_fixations).

    Re-running a config replaces the previous row rather than accumulating
    duplicates the display would then have to choose between.
    """
    import json

    with connect() as conn:
        conn.execute(
            "DELETE FROM model_runs WHERE image_id = ? AND mode = ? "
            "AND IFNULL(target,'') = ? AND n_fixations = ?",
            (image_id, payload["mode"], payload.get("target") or "",
             payload["n_fixations"]),
        )
        cur = conn.execute(
            "INSERT INTO model_runs (image_id, mode, target, n_fixations, seed, "
            "temperature, prompt_text, coords_json, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id, payload["mode"], payload.get("target"),
                payload["n_fixations"], payload.get("seed"),
                payload.get("temperature"), payload.get("prompt_text"),
                json.dumps(payload), payload.get("source", "precomputed"),
                payload.get("created_at") or utcnow(),
            ),
        )
        return cur.lastrowid


def get_model_run(image_id: int, mode: str, target: Optional[str],
                  n_fixations: Optional[int] = None) -> Optional[dict]:
    import json

    sql = ("SELECT * FROM model_runs WHERE image_id = ? AND mode = ? "
           "AND IFNULL(target,'') = ?")
    params: list = [image_id, mode, target or ""]
    if n_fixations is not None:
        sql += " AND n_fixations = ?"
        params.append(n_fixations)
    sql += " ORDER BY created_at DESC, id DESC LIMIT 1"

    with connect() as conn:
        row = conn.execute(sql, params).fetchone()
    if not row:
        return None
    payload = json.loads(row["coords_json"])
    payload["run_id"] = row["id"]
    # Run files written before prompt_kind existed carry a trained prompt.
    payload.setdefault("prompt_kind", "trained")
    return payload


def list_model_runs() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT m.id, m.image_id, m.mode, m.target, m.n_fixations, "
            "m.source, m.created_at, i.filename, i.label "
            "FROM model_runs m JOIN images i ON i.id = m.image_id "
            "ORDER BY i.sort_order, m.mode, m.target"
        )
        return [dict(r) for r in rows]


def sync_model_runs_from_disk() -> dict:
    """Load data/model/*.json into the model_runs table.

    Files are the source of truth — they are what precompute.py writes and
    what gets committed — and the table is a queryable index over them.
    """
    import json

    config.ensure_dirs()
    by_filename = {i["filename"]: i for i in list_images(include_missing=True)}

    loaded, orphaned, invalid = [], [], []
    for path in sorted(config.MODEL_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            invalid.append(path.name)
            continue

        required = {"image", "mode", "n_fixations", "scanpath_norm"}
        if not required.issubset(payload):
            invalid.append(path.name)
            continue

        image = by_filename.get(payload["image"])
        if not image:
            orphaned.append(path.name)  # run for an image no longer present
            continue

        upsert_model_run(image["id"], payload)
        loaded.append(path.name)

    return {"loaded": loaded, "orphaned": orphaned, "invalid": invalid}


# --------------------------------------------------------------------------
# gaze sessions (webcam capture)
# --------------------------------------------------------------------------

GAZE_SCHEMA = """
CREATE TABLE IF NOT EXISTS gaze_sessions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id          INTEGER REFERENCES rounds(id),
    participant_id    INTEGER NOT NULL REFERENCES participants(id),
    started_at        TEXT    NOT NULL,
    tracker           TEXT,
    tracker_version   TEXT,
    grade             TEXT,     -- good | usable | poor | failed
    mean_error        REAL,     -- fraction of viewport width
    worst_error       REAL,
    points_accepted   INTEGER,
    points_total      INTEGER,
    validation_json   TEXT,
    diagnostics_json  TEXT,
    viewport_w        INTEGER,
    viewport_h        INTEGER,
    device_label      TEXT,
    excluded          INTEGER NOT NULL DEFAULT 0,
    exclusion_reason  TEXT
);
CREATE INDEX IF NOT EXISTS idx_gaze_sessions_round ON gaze_sessions(round_id);
"""


def init_gaze() -> None:
    with connect() as conn:
        conn.executescript(GAZE_SCHEMA)


def create_gaze_session(participant_id: int, round_id: Optional[int],
                        payload: dict) -> int:
    """Record a calibration outcome.

    A failed calibration is stored, not discarded: the exclusion rate is a
    reportable number (spec section 5.1), and it cannot be reported if the
    failures were never written down.
    """
    import json

    grade = payload.get("grade")
    excluded = 0 if grade in ("good", "usable") else 1
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO gaze_sessions (round_id, participant_id, started_at, "
            "tracker, tracker_version, grade, mean_error, worst_error, "
            "points_accepted, points_total, validation_json, diagnostics_json, "
            "viewport_w, viewport_h, device_label, excluded, exclusion_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                round_id, participant_id, utcnow(),
                payload.get("tracker"), payload.get("tracker_version"), grade,
                payload.get("mean_error"), payload.get("worst_error"),
                payload.get("points_accepted"), payload.get("points_total"),
                json.dumps(payload.get("validation") or []),
                json.dumps(payload.get("diagnostics") or {}),
                payload.get("viewport_w"), payload.get("viewport_h"),
                (payload.get("device_label") or "")[:120],
                excluded,
                None if not excluded else (payload.get("failure") or grade),
            ),
        )
        return cur.lastrowid


def gaze_session_stats(round_id: Optional[int]) -> dict:
    """Counts for the presenter: how many tracked, how many were excluded."""
    with connect() as conn:
        if round_id is None:
            return {"total": 0, "usable": 0, "excluded": 0, "grades": {}}
        rows = conn.execute(
            "SELECT grade, excluded, COUNT(*) AS n FROM gaze_sessions "
            "WHERE round_id = ? GROUP BY grade, excluded", (round_id,)
        ).fetchall()
    grades, total, excluded = {}, 0, 0
    for r in rows:
        grades[r["grade"] or "unknown"] = grades.get(r["grade"] or "unknown", 0) + r["n"]
        total += r["n"]
        if r["excluded"]:
            excluded += r["n"]
    return {"total": total, "usable": total - excluded, "excluded": excluded,
            "grades": grades}
