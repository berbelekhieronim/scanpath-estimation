"""Rounds as portable records: export one, import it back, list them all.

A round is already the unit of collection — markers, gaze sessions and
assignments all hang off a round_id — but until now it existed only inside
one SQLite file on one machine. That file lives in an ephemeral container.
Losing it loses the only thing in this project that cannot be regenerated:
what people actually did.

So a round can be written out as one self-contained JSON document and read
back in. Two things follow from that, and both are the point:

  * A demo can be rehearsed. Collect once with real people, export, and
    replay that exact state on stage without needing a room full of phones.
  * Rounds become comparable. Two snapshots side by side are two sittings,
    which is the shape a session-to-session comparison needs.

Import is always additive. It creates new rounds and never overwrites an
existing one, because the failure mode of a restore that clobbers is losing
live data twenty minutes before a talk. Re-importing the same file is a
no-op unless you ask for a duplicate: each imported round remembers where it
came from.
"""

from __future__ import annotations

import json
from typing import Optional

from . import config, db

# Bumped when the document shape changes in a way an older reader would get
# wrong. Readers refuse a newer schema rather than guessing at it.
SCHEMA_VERSION = 2

FORMAT = "scanpath-round-snapshot"

# The presenter-facing settings a round was collected under. Restoring these
# alongside the data is what makes an imported round look like the evening it
# was recorded, rather than like today's leftovers.
# Defaults spelled out rather than left absent: a setting that has never
# been touched is missing from app_state, and a snapshot recording it as
# missing restores as "whatever today happens to be set to". A round that
# comes back has to come back the way it ran.
SETTING_DEFAULTS = {
    "capture_mode": "tap",
    "grid_source": "tapped",
    "view_ms": str(config.DEFAULT_VIEW_MS),
    "tap_count": str(config.DEFAULT_TAP_COUNT),
    "model_mode": "freeview",
    "model_target": "",
    "model_n_fixations": "5",
}
SETTING_KEYS = tuple(SETTING_DEFAULTS)

LAYER_DEFAULTS = {"heatmap": "1", "paths": "0", "model": "0", "prompt": "0",
                  "json": "0", "gaze": "0", "gaze_paths": "0", "charts": "0",
                  "grid": "0"}


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def init_rounds() -> None:
    """Older databases predate the two columns a portable round needs."""
    with db.connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(rounds)")}
        if not cols:
            return
        if "imported_from" not in cols:
            conn.execute("ALTER TABLE rounds ADD COLUMN imported_from TEXT")
        if "settings_json" not in cols:
            conn.execute("ALTER TABLE rounds ADD COLUMN settings_json TEXT")


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------

def live_settings() -> dict:
    out = {k: db.get_state(k, d) for k, d in SETTING_DEFAULTS.items()}
    out["layers"] = {k: db.get_state(f"layer_{k}", d)
                     for k, d in LAYER_DEFAULTS.items()}
    return out


def freeze_settings(round_id: int) -> None:
    """Record the settings a round was collected under.

    Called when a round closes, because that is the last moment the live
    settings still describe it. After that the presenter moves on and the
    app_state keys start describing the next round instead.
    """
    with db.connect() as conn:
        conn.execute("UPDATE rounds SET settings_json = ? WHERE id = ?",
                     (json.dumps(live_settings()), round_id))


def settings_for(round_id: int) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT settings_json FROM rounds WHERE id = ?",
                           (round_id,)).fetchone()
    if row and row["settings_json"]:
        try:
            return json.loads(row["settings_json"])
        except Exception:
            pass
    return live_settings()


def apply_settings(settings: dict) -> list[str]:
    """Put a round's settings back into effect. Returns what changed."""
    applied = []
    for key in SETTING_KEYS:
        value = (settings or {}).get(key)
        if value is None:
            continue
        db.set_state(key, value)
        applied.append(key)
    for name, on in ((settings or {}).get("layers") or {}).items():
        if on is None or name not in LAYER_DEFAULTS:
            continue
        db.set_state(f"layer_{name}", on)
    return applied


# --------------------------------------------------------------------------
# listing
# --------------------------------------------------------------------------

def list_rounds() -> list[dict]:
    """Every round, newest first, with enough counts to choose between them."""
    active = db.get_active_round()
    active_id = active["id"] if active else None

    with db.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT r.*, i.filename, i.label AS image_label "
            "FROM rounds r JOIN images i ON i.id = r.image_id "
            "ORDER BY r.id DESC")]
        taps = {r["round_id"]: (r["people"], r["markers"]) for r in conn.execute(
            "SELECT round_id, COUNT(DISTINCT participant_id) AS people, "
            "COUNT(*) AS markers FROM markers GROUP BY round_id")}
        gaze = {r["round_id"]: (r["sessions"], r["usable"]) for r in conn.execute(
            "SELECT round_id, COUNT(*) AS sessions, "
            "SUM(CASE WHEN excluded = 0 THEN 1 ELSE 0 END) AS usable "
            "FROM gaze_sessions WHERE round_id IS NOT NULL GROUP BY round_id")}

    out = []
    for r in rows:
        t = taps.get(r["id"], (0, 0))
        g = gaze.get(r["id"], (0, 0))
        cfg = settings_for(r["id"])
        out.append({
            "id": r["id"],
            "label": r["label"],
            "image": r["image_label"],
            "filename": r["filename"],
            "opened_at": r["opened_at"],
            "closed_at": r["closed_at"],
            "active": r["id"] == active_id,
            "imported_from": r.get("imported_from"),
            "tappers": t[0], "markers": t[1],
            "gaze_sessions": g[0], "gaze_usable": g[1] or 0,
            # The parameters that decide whether two rounds can be compared
            # at all. Two sittings of the same image differing only in the
            # model run behind them are not the same experiment, and a list
            # that shows only a timestamp makes them look identical.
            "params": {
                "task": cfg.get("model_mode"),
                "target": cfg.get("model_target") or None,
                "n_fixations": cfg.get("model_n_fixations"),
                "view_ms": cfg.get("view_ms"),
                "tap_count": cfg.get("tap_count"),
            },
            "signature": describe_params(cfg, r["filename"]),
        })
    return out


def describe_params(cfg: dict, filename: str = "") -> str:
    """A short, stable name for what a round was run under.

    Generated rather than typed. A label somebody enters by hand is a label
    somebody forgets to change, and then two rounds differ only by a
    timestamp — which is exactly the pair most likely to be compared by
    mistake.
    """
    bits = []
    if filename:
        bits.append(filename.rsplit(".", 1)[0][:24])
    task = cfg.get("model_mode") or "freeview"
    target = cfg.get("model_target") or ""
    bits.append(f"{task}:{target}" if target else task)
    if cfg.get("model_n_fixations"):
        bits.append(f"n{cfg['model_n_fixations']}")
    try:
        bits.append(f"{int(cfg.get('view_ms') or 0) / 1000:g}s")
    except (TypeError, ValueError):
        pass
    return " · ".join(b for b in bits if b)


def activate(round_id: int) -> dict:
    """Make a round the live one, settings and all.

    Closing rounds rather than deleting them means any past round can be
    reopened: re-examining a sitting is a legitimate thing to want, and it
    is also how an imported snapshot gets onto the screen.
    """
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM rounds WHERE id = ?",
                           (round_id,)).fetchone()
        if not row:
            raise LookupError(f"no round {round_id}")
        current = conn.execute(
            "SELECT id FROM rounds WHERE closed_at IS NULL AND id != ?",
            (round_id,)).fetchall()
        image_id = row["image_id"]

    for other in current:
        freeze_settings(other["id"])
        with db.connect() as conn:
            conn.execute("UPDATE rounds SET closed_at = ? WHERE id = ?",
                         (db.utcnow(), other["id"]))

    with db.connect() as conn:
        conn.execute("UPDATE rounds SET closed_at = NULL WHERE id = ?", (round_id,))
    db.set_state("active_round_id", round_id)
    db.set_state("active_image_id", image_id)
    applied = apply_settings(settings_for(round_id))
    return {"round_id": round_id, "image_id": image_id, "settings": applied}


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def _loads(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def export_round(round_id: int) -> dict:
    """One round, self-contained: everything needed to replay it elsewhere.

    The image file itself is not embedded — it is in the repository, and a
    base64 JPEG would multiply the size of a document whose whole value is
    being small enough to mail to yourself. What travels is the filename and
    the dimensions, which is what the normalised coordinates need to mean
    anything.
    """
    with db.connect() as conn:
        round_ = conn.execute("SELECT * FROM rounds WHERE id = ?",
                              (round_id,)).fetchone()
        if not round_:
            raise LookupError(f"no round {round_id}")
        round_ = dict(round_)
        image = dict(conn.execute("SELECT * FROM images WHERE id = ?",
                                  (round_["image_id"],)).fetchone())

        markers = [dict(r) for r in conn.execute(
            "SELECT p.uuid, m.seq, m.x, m.y, m.created_at FROM markers m "
            "JOIN participants p ON p.id = m.participant_id "
            "WHERE m.round_id = ? ORDER BY m.participant_id, m.seq",
            (round_id,))]

        assignments = [dict(r) for r in conn.execute(
            "SELECT p.uuid, a.condition, a.assigned_at, a.completed, a.forced "
            "FROM assignments a JOIN participants p ON p.id = a.participant_id "
            "WHERE a.round_id = ? ORDER BY a.id", (round_id,))]

        sessions = [dict(r) for r in conn.execute(
            "SELECT g.*, p.uuid FROM gaze_sessions g "
            "JOIN participants p ON p.id = g.participant_id "
            "WHERE g.round_id = ? ORDER BY g.id", (round_id,))]
        for s in sessions:
            s["validation"] = _loads(s.pop("validation_json", None))
            s["diagnostics"] = _loads(s.pop("diagnostics_json", None))
            s["samples"] = [dict(r) for r in conn.execute(
                "SELECT t_ms, x, y, on_image FROM gaze_samples "
                "WHERE session_id = ? ORDER BY t_ms", (s["id"],))]
            s.pop("participant_id", None)
            s.pop("round_id", None)
            s.pop("id", None)

        uuids = ({m["uuid"] for m in markers} | {a["uuid"] for a in assignments}
                 | {s["uuid"] for s in sessions})
        people = []
        if uuids:
            q = ",".join("?" * len(uuids))
            people = [dict(r) for r in conn.execute(
                f"SELECT uuid, first_seen, user_agent, device_json "
                f"FROM participants WHERE uuid IN ({q})", tuple(uuids))]
        for p in people:
            p["device"] = _loads(p.pop("device_json", None))

        runs = [dict(r) for r in conn.execute(
            "SELECT * FROM model_runs WHERE image_id = ? ORDER BY id",
            (round_["image_id"],))]
        for r in runs:
            r["payload"] = _loads(r.pop("coords_json"))
            r.pop("id", None)
            r.pop("image_id", None)

    settings = settings_for(round_id)
    exported_at = db.utcnow()
    return {
        "format": FORMAT,
        "schema": SCHEMA_VERSION,
        "exported_at": exported_at,
        "exported_on": exported_at[:10],
        "round": {
            "source_id": round_id,
            "label": round_["label"],
            "opened_at": round_["opened_at"],
            "closed_at": round_["closed_at"],
        },
        "image": {"filename": image["filename"], "label": image["label"],
                  "width": image["width"], "height": image["height"]},
        "settings": settings,
        "participants": people,
        "assignments": assignments,
        "markers": markers,
        "gaze_sessions": sessions,
        "model_runs": runs,
        "counts": {
            "participants": len(people), "markers": len(markers),
            "assignments": len(assignments),
            "gaze_sessions": len(sessions),
            "gaze_samples": sum(len(s["samples"]) for s in sessions),
            "model_runs": len(runs),
        },
    }


def export_filename(snapshot: dict) -> str:
    """A name that sorts by date and says which sitting it was."""
    stamp = (snapshot.get("exported_at") or "")[:19].replace(":", "").replace("-", "")
    image = (snapshot.get("image") or {}).get("filename", "round")
    stem = image.rsplit(".", 1)[0][:40] or "round"
    rid = (snapshot.get("round") or {}).get("source_id", "x")
    return f"round-{rid}-{stem}-{stamp}.json"


# --------------------------------------------------------------------------
# import
# --------------------------------------------------------------------------

def _origin_key(snapshot: dict) -> str:
    return (f"{snapshot.get('exported_at')}#"
            f"{(snapshot.get('round') or {}).get('source_id')}")


def import_snapshot(payload: dict, duplicate: bool = False) -> dict:
    """Restore a round from a snapshot, as a new round.

    Nothing existing is modified. The image is matched by filename; if the
    file is not on this machine the image is recorded as missing rather than
    refused, because the responses are still worth having and the picture can
    be put back afterwards.
    """
    if not isinstance(payload, dict):
        raise ValueError("snapshot must be a JSON object")
    if payload.get("format") != FORMAT:
        raise ValueError("not a round snapshot")
    schema = payload.get("schema")
    if not isinstance(schema, int) or schema > SCHEMA_VERSION:
        raise ValueError(
            f"snapshot schema {schema} is newer than this server understands "
            f"({SCHEMA_VERSION}) — update the app rather than importing it")

    origin = _origin_key(payload)
    if not duplicate:
        with db.connect() as conn:
            seen = conn.execute("SELECT id FROM rounds WHERE imported_from = ?",
                                (origin,)).fetchone()
        if seen:
            return {"round_id": seen["id"], "imported": False,
                    "reason": "already imported"}

    img = payload.get("image") or {}
    filename = img.get("filename")
    if not filename:
        raise ValueError("snapshot names no image")

    warnings = []
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM images WHERE filename = ?",
                           (filename,)).fetchone()
        if row:
            image_id = row["id"]
            if (row["width"], row["height"]) != (img.get("width"), img.get("height")):
                warnings.append(
                    f"{filename} is {row['width']}x{row['height']} here but was "
                    f"{img.get('width')}x{img.get('height')} when recorded")
        else:
            # present=0: the row exists so the round has something to hang
            # off, and /admin's rescan will light it up if the file turns up.
            image_id = conn.execute(
                "INSERT INTO images (filename, label, width, height, "
                "sort_order, present, created_at) VALUES (?, ?, ?, ?, 0, 0, ?)",
                (filename, img.get("label") or filename,
                 int(img.get("width") or 0), int(img.get("height") or 0),
                 db.utcnow())).lastrowid
            warnings.append(f"{filename} is not in data/images — "
                            "the responses imported, the picture did not")

        # Participants are matched by uuid so a re-import, or two snapshots
        # from the same audience, do not clone everybody.
        ids: dict[str, int] = {}
        for p in payload.get("participants") or []:
            uuid = p.get("uuid")
            if not uuid:
                continue
            found = conn.execute("SELECT id FROM participants WHERE uuid = ?",
                                 (uuid,)).fetchone()
            if found:
                ids[uuid] = found["id"]
                continue
            ids[uuid] = conn.execute(
                "INSERT INTO participants (uuid, first_seen, user_agent, "
                "device_json) VALUES (?, ?, ?, ?)",
                (uuid, p.get("first_seen") or db.utcnow(), p.get("user_agent"),
                 json.dumps(p["device"]) if p.get("device") else None)).lastrowid

        def pid(uuid):
            if uuid in ids:
                return ids[uuid]
            found = conn.execute("SELECT id FROM participants WHERE uuid = ?",
                                 (uuid,)).fetchone()
            if found:
                ids[uuid] = found["id"]
            else:
                ids[uuid] = conn.execute(
                    "INSERT INTO participants (uuid, first_seen) VALUES (?, ?)",
                    (uuid, db.utcnow())).lastrowid
            return ids[uuid]

        src = payload.get("round") or {}
        label = src.get("label") or f"imported {payload.get('exported_on', '')}".strip()
        # Imported rounds arrive closed. Restoring a snapshot in the middle of
        # a live sitting must not silently take the screen.
        round_id = conn.execute(
            "INSERT INTO rounds (image_id, label, opened_at, closed_at, "
            "imported_from, settings_json) VALUES (?, ?, ?, ?, ?, ?)",
            (image_id, label, src.get("opened_at") or db.utcnow(),
             src.get("closed_at") or db.utcnow(), origin,
             json.dumps(payload.get("settings") or {}))).lastrowid

        n_markers = 0
        for m in payload.get("markers") or []:
            conn.execute(
                "INSERT OR IGNORE INTO markers (round_id, participant_id, seq, "
                "x, y, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (round_id, pid(m["uuid"]), int(m["seq"]), float(m["x"]),
                 float(m["y"]), m.get("created_at") or db.utcnow()))
            n_markers += 1

        n_assign = 0
        for a in payload.get("assignments") or []:
            conn.execute(
                "INSERT OR IGNORE INTO assignments (round_id, participant_id, "
                "condition, assigned_at, completed, forced) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (round_id, pid(a["uuid"]), a.get("condition") or "tap",
                 a.get("assigned_at") or db.utcnow(),
                 int(a.get("completed") or 0), int(a.get("forced") or 0)))
            n_assign += 1

        n_sessions = n_samples = 0
        for s in payload.get("gaze_sessions") or []:
            sid = conn.execute(
                "INSERT INTO gaze_sessions (round_id, participant_id, "
                "started_at, tracker, tracker_version, grade, mean_error, "
                "worst_error, points_accepted, points_total, validation_json, "
                "diagnostics_json, viewport_w, viewport_h, device_label, "
                "excluded, exclusion_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (round_id, pid(s["uuid"]), s.get("started_at") or db.utcnow(),
                 s.get("tracker"), s.get("tracker_version"), s.get("grade"),
                 s.get("mean_error"), s.get("worst_error"),
                 s.get("points_accepted"), s.get("points_total"),
                 json.dumps(s["validation"]) if s.get("validation") else None,
                 json.dumps(s["diagnostics"]) if s.get("diagnostics") else None,
                 s.get("viewport_w"), s.get("viewport_h"), s.get("device_label"),
                 int(s.get("excluded") or 0), s.get("exclusion_reason"))).lastrowid
            n_sessions += 1
            rows = [(sid, int(p.get("t_ms") or 0), p.get("x"), p.get("y"),
                     int(p.get("on_image") if p.get("on_image") is not None else 1))
                    for p in s.get("samples") or []]
            if rows:
                conn.executemany(
                    "INSERT INTO gaze_samples (session_id, t_ms, x, y, on_image) "
                    "VALUES (?, ?, ?, ?, ?)", rows)
                n_samples += len(rows)

    # Model runs go through the normal upsert so a run already on disk is not
    # duplicated by a snapshot that happens to carry it too.
    n_runs = 0
    for r in payload.get("model_runs") or []:
        run = dict(r.get("payload") or {})
        if not run:
            continue
        run.setdefault("mode", r.get("mode"))
        run.setdefault("target", r.get("target"))
        run.setdefault("n_fixations", r.get("n_fixations"))
        run.setdefault("source", r.get("source") or "precomputed")
        run.setdefault("created_at", r.get("created_at") or db.utcnow())
        for k in ("seed", "temperature", "prompt_text"):
            if r.get(k) is not None:
                run.setdefault(k, r[k])
        try:
            db.upsert_model_run(image_id, run)
            n_runs += 1
        except Exception as exc:                      # pragma: no cover
            warnings.append(f"model run not imported: {exc}")

    return {
        "round_id": round_id, "imported": True, "image_id": image_id,
        "warnings": warnings,
        "counts": {"participants": len(ids), "markers": n_markers,
                   "assignments": n_assign, "gaze_sessions": n_sessions,
                   "gaze_samples": n_samples, "model_runs": n_runs},
    }


# --------------------------------------------------------------------------
# bundles
# --------------------------------------------------------------------------

BUNDLE_FORMAT = "scanpath-round-bundle"


def export_all() -> dict:
    """Every round in one document — the whole machine, ready to move."""
    snaps = [export_round(r["id"]) for r in reversed(list_rounds())]
    at = db.utcnow()
    return {
        "format": BUNDLE_FORMAT,
        "schema": SCHEMA_VERSION,
        "exported_at": at,
        "exported_on": at[:10],
        "rounds": snaps,
        "counts": {"rounds": len(snaps),
                   "markers": sum(s["counts"]["markers"] for s in snaps),
                   "gaze_sessions": sum(s["counts"]["gaze_sessions"]
                                        for s in snaps)},
    }


def bundle_filename(bundle: dict) -> str:
    stamp = (bundle.get("exported_at") or "")[:19].replace(":", "").replace("-", "")
    return f"scanpath-rounds-{stamp}.json"


def import_any(payload: dict, duplicate: bool = False) -> dict:
    """Accept either a single round or a whole bundle."""
    if isinstance(payload, dict) and payload.get("format") == BUNDLE_FORMAT:
        results = [import_snapshot(s, duplicate=duplicate)
                   for s in payload.get("rounds") or []]
        return {"bundle": True, "results": results,
                "imported": sum(1 for r in results if r.get("imported")),
                "skipped": sum(1 for r in results if not r.get("imported"))}
    return {"bundle": False, "results": [import_snapshot(payload, duplicate=duplicate)]}
