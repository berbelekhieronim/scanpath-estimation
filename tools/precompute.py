#!/usr/bin/env python3
"""Batch-run the scanpath model over the stimulus images, ahead of a session.

This is the primary path (spec section 4.2, Tier A). Running the model before
the talk rather than during it costs nothing in fidelity — same weights, same
prompt, same output — and removes every way inference can fail in front of an
audience.

    python tools/precompute.py --repo ../DeepGaze3.5-VL \
        --modes freeview --samples 10 --temperature 0.7

The model is loaded once and reused across every image and config, and the
samples for one config share a single image prefill, so the cost is roughly
one prefill per image-config rather than per scanpath.

--synthetic generates placeholder scanpaths WITHOUT the model, for developing
the display when no GPU is to hand. Its output is stamped source="synthetic"
and the web app shows a loud warning badge whenever it renders one. Never
present synthetic output as a model prediction.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaze_prompts as gp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = REPO_ROOT / "data" / "images"
DEFAULT_OUT = REPO_ROOT / "data" / "model"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def run_name(image_stem, mode, target, n):
    bits = [image_stem, mode] + ([target.replace(" ", "-")] if target else []) + [f"n{n}"]
    return "__".join(bits) + ".json"


def synthetic_scanpaths(image_path, n_fix, n_samples, seed):
    """Plausible-looking placeholder paths. NOT model output.

    Deliberately simple: a centre-biased first fixation, then a walk with
    modest step sizes. It exists only so the rendering pipeline can be built
    and tested without a GPU.
    """
    rng = random.Random(f"{image_path}-{seed}")
    out = []
    for _ in range(n_samples):
        path = []
        x, y = rng.gauss(50, 9), rng.gauss(50, 9)
        for _ in range(n_fix):
            x = min(97, max(2, x + rng.gauss(0, 19)))
            y = min(97, max(2, y + rng.gauss(0, 15)))
            path.append((int(x), int(y)))
        out.append(path)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="DeepGaze3.5-VL checkout (required unless --synthetic)")
    ap.add_argument("--images-dir", default=str(DEFAULT_IMAGES))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--only", nargs="*", help="Limit to these image filenames")
    ap.add_argument("--modes", nargs="+", default=["freeview"],
                    choices=["freeview", "search"])
    ap.add_argument("--targets", nargs="*", default=["car"],
                    help="Search targets, used when 'search' is in --modes")
    ap.add_argument("--num-fixations", type=int, default=5,
                    help="Match the participant tap count for clean comparison")
    ap.add_argument("--samples", type=int, default=10, help="Virtual observers")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--synthetic", action="store_true",
                    help="Placeholder output, no model. Clearly marked as such.")
    ap.add_argument("--force", action="store_true", help="Re-run existing outputs")
    args = ap.parse_args()

    if not args.synthetic and not args.repo:
        ap.error("--repo is required (or use --synthetic for placeholder output)")

    images_dir, out_dir = Path(args.images_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if args.only:
        images = [p for p in images if p.name in set(args.only)]
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    jobs = []
    for img in images:
        for mode in args.modes:
            targets = args.targets if mode == "search" else [None]
            for target in targets:
                jobs.append((img, mode, target))

    print(f"{len(images)} image(s) x {len(jobs) // len(images)} config(s) "
          f"= {len(jobs)} run(s), {args.samples} sample(s) each")
    if args.synthetic:
        print("\n*** SYNTHETIC MODE — placeholder data, not model output ***\n")

    model = processor = None
    if not args.synthetic:
        import predict_mps
        device = predict_mps.pick_device(args.device)
        print(f"Device: {device}")
        # Freeview and search use different adapters, so a mixed run reloads.
        adapters_needed = {("combined_adapter" if m == "freeview"
                            else "visual_search_adapter") for _, m, _ in jobs}
        if len(adapters_needed) > 1:
            print("NOTE: freeview and search use different adapters; the model "
                  "will be reloaded when switching between them.")

    loaded_adapter = None
    written, skipped, failed = 0, 0, 0
    started = time.time()

    for i, (img, mode, target) in enumerate(jobs, 1):
        name = run_name(img.stem, mode, target, args.num_fixations)
        dest = out_dir / name
        if dest.exists() and not args.force:
            print(f"[{i}/{len(jobs)}] {name} — exists, skipping")
            skipped += 1
            continue

        label = f"{img.name} / {mode}" + (f" / {target}" if target else "")
        print(f"[{i}/{len(jobs)}] {label}…", flush=True)
        prompt_text = gp.build_prompt(mode, args.num_fixations, target)
        t0 = time.time()

        try:
            if args.synthetic:
                samples = synthetic_scanpaths(img.name, args.num_fixations,
                                              args.samples, args.seed)
                source, model_name, device_used = "synthetic", "SYNTHETIC (no model)", "none"
            else:
                import predict_mps
                from PIL import Image

                adapter = ("combined_adapter" if mode == "freeview"
                           else "visual_search_adapter")
                if adapter != loaded_adapter:
                    model, processor = predict_mps.load_model(
                        args.repo, adapter, device, args.dtype)
                    loaded_adapter = adapter

                image = Image.open(img).convert("RGB")
                texts = predict_mps.predict(
                    model, processor, image, prompt_text, args.samples,
                    args.temperature, args.seed, device,
                    max(64, 16 * args.num_fixations + 16))
                samples = [s for s in (gp.parse_scanpath(t) for t in texts) if s]
                if not samples:
                    raise RuntimeError(f"no parseable coordinates: {texts[:1]}")
                source = "precomputed"
                model_name = f"{gp.BASE_MODEL} + {adapter}"
                device_used = device
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failed += 1
            continue

        elapsed = time.time() - t0
        dest.write_text(json.dumps({
            "image": img.name,
            "mode": mode,
            "target": target,
            "n_fixations": args.num_fixations,
            "seed": args.seed,
            "temperature": args.temperature,
            "prompt_text": prompt_text,
            "scanpath_grid": samples[0],
            "scanpath_norm": gp.grid_to_norm(samples[0]),
            "samples_grid": samples,
            "samples_norm": [gp.grid_to_norm(s) for s in samples],
            "source": source,
            "model": model_name,
            "device": device_used,
            "elapsed_seconds": round(elapsed, 1),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
        print(f"    {len(samples)} sample(s) in {elapsed:.1f}s -> {dest.name}")
        written += 1

    total = time.time() - started
    print(f"\nWrote {written}, skipped {skipped}, failed {failed} "
          f"in {total / 60:.1f} min")
    if written and not args.synthetic:
        print("Restart the app (or press Rescan in /admin) to pick these up.")
    if args.synthetic:
        print("\nReminder: this output is SYNTHETIC and is badged as such in the UI.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
