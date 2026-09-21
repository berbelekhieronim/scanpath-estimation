#!/usr/bin/env python3
"""Push a scanpath run from this machine to the live web app (spec Tier C).

The model runs on your MacBook; the app runs in a Codespace with no GPU. This
bridges them: compute here, POST there, and the display picks it up on its
next poll, within about two seconds.

Run the model and push in one step:

    python tools/push_result.py --url https://NAME-8000.app.github.dev \\
        --token YOUR_CONTROL_TOKEN --repo ../DeepGaze3.5-VL \\
        --image data/images/street.jpg --mode freeview --num-fixations 5

Or push a file you already generated:

    python tools/push_result.py --url ... --token ... --json run.json

Inference takes 1-3 minutes on MPS, which is a long silence in a talk. Narrate
over it, and keep the precomputed run loaded as the fallback — see the spec's
tier recommendation.

URL and token can come from SCANPATH_URL and SCANPATH_TOKEN instead of flags.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def push(url: str, token: str, payload: dict, timeout: int = 30) -> dict:
    endpoint = url.rstrip("/") + "/api/model/push"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Control-Token": token},
        method="POST",
    )
    # Never route a request for the app through a proxy meant for outbound
    # traffic; the Codespace URL is reached directly.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        detail = body
        try:
            detail = json.loads(body).get("detail", body)
        except Exception:
            pass
        if exc.code == 403:
            raise SystemExit("Rejected: bad control token. It is printed in the "
                             "app's server console at startup.")
        if exc.code == 404:
            raise SystemExit(f"Rejected: {detail}\nThe image filename must match "
                             f"one already loaded in the app.")
        raise SystemExit(f"Push failed ({exc.code}): {detail}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach {endpoint}: {exc.reason}\n"
                         f"Check the URL, and that port 8000 is set to Public.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("SCANPATH_URL"),
                    help="App base URL (or set SCANPATH_URL)")
    ap.add_argument("--token", default=os.environ.get("SCANPATH_TOKEN"),
                    help="Control token (or set SCANPATH_TOKEN)")
    ap.add_argument("--json", help="Push an existing run JSON instead of computing one")

    ap.add_argument("--repo", help="DeepGaze3.5-VL checkout (when computing)")
    ap.add_argument("--image", help="Image to run (when computing)")
    ap.add_argument("--mode", choices=["freeview", "search"], default="freeview")
    ap.add_argument("--target")
    ap.add_argument("--num-fixations", type=int, default=5)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bfloat16", "float16", "float32"],
                    help="auto picks what the hardware supports")
    ap.add_argument("--prompt", help=(
        "EXPERIMENTAL. Your own prompt text instead of the trained template. "
        "Off-distribution; output quality is unvalidated and the run is "
        "badged in the UI."))
    ap.add_argument("--save", help="Also write the run JSON here")
    args = ap.parse_args()

    if not args.url:
        ap.error("--url is required (or set SCANPATH_URL)")
    if not args.token:
        ap.error("--token is required (or set SCANPATH_TOKEN)")

    if args.json:
        payload = json.loads(Path(args.json).read_text())
        print(f"Pushing {args.json} ({payload.get('source', '?')})…")
    else:
        if not (args.repo and args.image):
            ap.error("--repo and --image are required unless --json is given")

        import time
        import gaze_prompts as gp
        import predict
        from PIL import Image

        n = args.num_fixations
        if args.prompt:
            prompt_text, prompt_kind = args.prompt, "custom"
            print("*** EXPERIMENTAL PROMPT — off the trained distribution, "
                  "quality unvalidated. ***")
        else:
            prompt_text, prompt_kind = gp.build_prompt(args.mode, n, args.target), "trained"
        adapter = ("combined_adapter" if args.mode == "freeview"
                   else "visual_search_adapter")
        import backends
        device = predict.pick_device(args.device)
        prof = backends.detect(device)
        if args.dtype != "auto":
            prof.dtype = args.dtype
        dtype_name = prof.dtype
        print(backends.describe(prof), file=sys.stderr)

        print(f"Running on {device} — this takes a few minutes. "
              f"Keep the precomputed run on screen meanwhile.", flush=True)
        model, processor = predict.load_model(
            args.repo, adapter, device, dtype_name, attn=prof.attn,
            quant=prof.quant, on_device=prof.load_on_device)
        image = Image.open(args.image).convert("RGB")

        t0 = time.time()
        texts = predict.predict(model, processor, image, prompt_text,
                                    args.samples, args.temperature, args.seed,
                                    device, max(64, 16 * n + 16))
        elapsed = time.time() - t0
        samples = [s for s in (gp.parse_scanpath(t) for t in texts) if s]
        if not samples:
            raise SystemExit(f"Model returned no parseable coordinates: {texts[:1]}")

        payload = {
            "image": Path(args.image).name,
            "mode": args.mode, "target": args.target, "n_fixations": n,
            "seed": args.seed, "temperature": args.temperature,
            "prompt_text": prompt_text,
            "prompt_kind": prompt_kind,
            "scanpath_grid": samples[0],
            "scanpath_norm": gp.grid_to_norm(samples[0]),
            "samples_grid": samples,
            "samples_norm": [gp.grid_to_norm(s) for s in samples],
            "source": "pushed",
            "model": f"{gp.BASE_MODEL} + {adapter}",
            "device": device,
            "elapsed_seconds": round(elapsed, 1),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        print(f"Done in {elapsed:.1f}s: {samples[0]}")
        if args.save:
            Path(args.save).write_text(json.dumps(payload, indent=2))
            print(f"Saved {args.save}")

    out = push(args.url, args.token, payload)
    print(f"Pushed. run_id={out['run_id']}, stored as '{out['stored_as']}'.")
    print("Turn on the Model layer in /control if it is not already on.")


if __name__ == "__main__":
    main()
