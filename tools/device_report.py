#!/usr/bin/env python3
"""What the calibration error actually was, broken down by device.

Answers "is the phone worse because of the camera, or because of the
geometry?" from recorded sessions rather than from reasoning. Every
calibration already stores its own measured error as a fraction of viewport
width, so the comparison needs no new measurement — only reading what is
there.

    python tools/device_report.py

The physical-geometry section is the part that explains the answer. Tracking
error is fundamentally *angular*: the model is guessing a direction, and the
library's output is normalised to the screen with no physical size anywhere in
it. So the same angular error covers a share of the picture that depends
entirely on how much of your visual field the picture fills.
"""

import argparse
import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402


def device_class(dev: dict, viewport_w, viewport_h) -> str:
    """Coarse bucket. Deliberately coarse: the interesting split is the one
    between a screen held at arm's length and one on a desk."""
    ua = (dev.get("ua") or "").lower()
    touch = dev.get("touch_points") or 0
    w = dev.get("screen_w") or viewport_w or 0
    if "ipad" in ua or (touch and w >= 700):
        return "tablet"
    if "iphone" in ua or "android" in ua or (touch and w < 700):
        return "phone"
    if "mac" in ua or "windows" in ua or "linux" in ua:
        return "desktop/laptop"
    return "unknown"


def sampling_rate(rows):
    """How fast gaze actually arrived, per device class.

    Everything temporal rests on this. A fixation lasts 200-300ms, so a rate
    below about 4Hz means consecutive samples are not consecutive fixations
    and no amount of analysis recovers them. The number has been quoted from
    one session; this reads it from every session there is.
    """
    if not rows:
        return
    by_session: dict = {}
    for r in rows:
        by_session.setdefault(r["session_id"], {"t": [], "dev": r["device_json"],
                                                "vw": r["viewport_w"]})
        by_session[r["session_id"]]["t"].append(r["t_ms"])

    buckets: dict = {}
    for sid, d in by_session.items():
        ts = sorted(d["t"])
        if len(ts) < 3:
            continue
        gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        if not gaps:
            continue
        try:
            dev = json.loads(d["dev"] or "{}")
        except Exception:
            dev = {}
        cls = device_class(dev, d["vw"], None)
        b = buckets.setdefault(cls, {"gaps": [], "n": 0, "spans": []})
        b["gaps"].append(statistics.median(gaps))
        b["spans"].append((ts[-1] - ts[0]) / 1000.0)
        b["n"] += 1

    if not buckets:
        print("Sampling rate: not enough samples recorded yet.\n")
        return

    print("Sampling rate — how fast gaze actually arrived")
    print("(a fixation lasts 200-300ms; below ~4Hz consecutive samples are")
    print(" not consecutive fixations, and nothing recovers them)\n")
    for cls, b in sorted(buckets.items()):
        gap = statistics.median(b["gaps"])
        hz = 1000.0 / gap if gap else 0
        span = statistics.median(b["spans"])
        print(f"  {cls:18s} n={b['n']:3d}   {hz:5.2f} Hz   "
              f"one every {gap:4.0f} ms   {span:.1f}s recorded")
        per_bin = (hz * span) / 2
        verdict = ("enough for early-vs-late binning"
                   if per_bin >= 10 else
                   f"only {per_bin:.0f} samples per time bin — too few to "
                   f"split the window")
        print(f"  {'':18s} {verdict}")
    print()


def summarise(rows, label):
    errs = [r for r in rows if r is not None]
    if not errs:
        return f"  {label:18s}  —"
    med = statistics.median(errs)
    return (f"  {label:18s}  n={len(errs):3d}   median {med * 100:5.1f}%"
            f"   range {min(errs) * 100:4.1f}–{max(errs) * 100:4.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="defaults to the app database")
    args = ap.parse_args()

    path = Path(args.db) if args.db else config.DB_PATH
    if not path.exists():
        sys.exit(f"No database at {path}")

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rate_rows = conn.execute(
        "SELECT s.session_id, s.t_ms, p.device_json, g.viewport_w "
        "FROM gaze_samples s "
        "JOIN gaze_sessions g ON g.id = s.session_id "
        "JOIN participants p ON p.id = g.participant_id "
        "ORDER BY s.session_id, s.t_ms").fetchall()
    rows = conn.execute(
        "SELECT g.*, p.device_json FROM gaze_sessions g "
        "JOIN participants p ON p.id = g.participant_id "
        "ORDER BY g.started_at"
    ).fetchall()

    if not rows:
        sys.exit("No calibrations recorded yet.")

    buckets: dict = {}
    for r in rows:
        try:
            dev = json.loads(r["device_json"] or "{}")
        except Exception:
            dev = {}
        cls = device_class(dev, r["viewport_w"], r["viewport_h"])
        b = buckets.setdefault(cls, {"mean": [], "resid": [], "grades": {},
                                     "n": 0, "viewports": set()})
        b["n"] += 1
        b["mean"].append(r["mean_error"])
        # residual_error lands in the diagnostics blob, not its own column.
        try:
            diag = json.loads(r["diagnostics_json"] or "{}")
        except Exception:
            diag = {}
        b["resid"].append(diag.get("residual_error"))
        b["grades"][r["grade"]] = b["grades"].get(r["grade"], 0) + 1
        o = diag.get("orientation")
        if o:
            b.setdefault("orientation", {})
            b["orientation"][o] = b["orientation"].get(o, 0) + 1
        if r["viewport_w"]:
            b["viewports"].add(f"{r['viewport_w']}x{r['viewport_h']}")

    print(f"\n{len(rows)} calibration(s) in {path}\n")
    sampling_rate(rate_rows)
    print("Calibration error, as a fraction of viewport width")
    print("(lower is better; this is what the tracker measured about itself)\n")
    for cls, b in sorted(buckets.items()):
        print(f"{cls}   —   {b['n']} session(s)")
        print(summarise(b["mean"], "before correction"))
        print(summarise(b["resid"], "after correction"))
        grades = ", ".join(f"{k}:{v}" for k, v in sorted(b["grades"].items()))
        print(f"  {'grades':18s}  {grades}")
        if b.get("orientation"):
            held = ", ".join(f"{k}:{v}" for k, v in sorted(b["orientation"].items()))
            print(f"  {'held':18s}  {held}")
        if b["viewports"]:
            print(f"  {'viewports':18s}  {', '.join(sorted(b['viewports']))}")
        print()

    geometry_note()


def geometry_note():
    """The physics, so the numbers above have an explanation attached."""
    def span(size_cm, dist_cm):
        return 2 * math.degrees(math.atan((size_cm / 2) / dist_cm))

    # A 4:3 picture, letterboxed into each screen, at a typical viewing
    # distance for that device.
    devices = [
        ("MacBook Air 13.6\"", 29.4, 18.4, 55),
        ("iPhone 12 mini, portrait", 5.9, 13.1, 30),
        ("iPhone 12 mini, landscape", 13.1, 5.9, 30),
    ]
    print("Why, in geometry — how much of the visual field the PICTURE fills")
    print("(a 4:3 image fitted to each screen, at a typical viewing distance)\n")
    print(f"  {'device':28s} {'image on screen':>16s} {'subtends':>10s}"
          f" {'2.5° error =':>14s}")
    for name, sw, sh, dist in devices:
        # Fit 4:3 inside the screen.
        iw, ih = (sw, sw * 3 / 4) if sw * 3 / 4 <= sh else (sh * 4 / 3, sh)
        deg = span(iw, dist)
        print(f"  {name:28s} {iw:5.1f}x{ih:4.1f}cm {deg:8.1f}°"
              f" {2.5 / deg * 100:12.1f}% of image width")

    print("""
  The tracker's error is angular — it is guessing a direction, and the
  library hands back a point normalised to the screen with no physical size
  anywhere in it. Every bit of device geometry is absorbed by the nine-point
  calibration. So the same angular accuracy covers two and a half times more
  of a phone picture than of a laptop picture, purely because the phone
  picture fills less of the eye's field.

  The phone's camera is not the limit. At 30cm the face fills MORE of the
  frame than it does at 55cm on a laptop, so the eyes are sampled by more
  pixels, not fewer.

  The second effect is calibration conditioning: on a phone the nine points
  span roughly 11° horizontally, against 30° on a laptop. Fitting a mapping
  over a third of the baseline makes it a third as well determined, and that
  error is then extrapolated across the picture.""")


if __name__ == "__main__":
    main()
