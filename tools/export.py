#!/usr/bin/env python3
"""Export a session's data so it outlives the container.

A Codespace is deleted after a retention period, taking its SQLite file with
it. Participant responses are the only thing in this project that cannot be
regenerated, so run this after every session and commit the result.

    python tools/export.py --token YOUR_CONTROL_TOKEN
    python tools/export.py --url https://NAME-8000.app.github.dev --token ...

Writes data/exports/session-YYYYmmdd-HHMMSS.json, plus a flat CSV of markers
alongside it for anything that would rather have a table.
"""

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "exports"


def fetch(url: str, token: str) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/api/export", headers={"X-Control-Token": token})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise SystemExit("Rejected: bad control token.")
        raise SystemExit(f"Export failed ({exc.code}): {exc.read().decode()[:400]}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach {url}: {exc.reason}")


def write_csv(data: dict, path: Path) -> None:
    """Flat marker table: one row per tap, joined to round and image."""
    rounds = {r["id"]: r for r in data["rounds"]}
    images = {i["id"]: i for i in data["images"]}
    participants = {p["id"]: p for p in data["participants"]}

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round_id", "image", "participant_uuid", "seq", "x", "y",
                    "created_at"])
        for m in data["markers"]:
            rnd = rounds.get(m["round_id"], {})
            img = images.get(rnd.get("image_id"), {})
            par = participants.get(m["participant_id"], {})
            w.writerow([m["round_id"], img.get("filename", ""),
                        par.get("uuid", ""), m["seq"],
                        f"{m['x']:.6f}", f"{m['y']:.6f}", m["created_at"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("SCANPATH_URL",
                                                     "http://127.0.0.1:8000"))
    ap.add_argument("--token", default=os.environ.get("SCANPATH_TOKEN"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if not args.token:
        ap.error("--token is required (or set SCANPATH_TOKEN)")

    data = fetch(args.url, args.token)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"session-{stamp}.json"
    csv_path = out_dir / f"session-{stamp}-markers.csv"

    json_path.write_text(json.dumps(data, indent=2))
    write_csv(data, csv_path)

    c = data["counts"]
    print(f"Exported {c['participants']} participant(s), {c['markers']} marker(s), "
          f"{c['rounds']} round(s), {c['model_runs']} model run(s)")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    if c["markers"] == 0:
        print("\nWARNING: no markers in this export — is this the right server?")
    else:
        print("\nCommit these files. They cannot be regenerated.")


if __name__ == "__main__":
    main()
