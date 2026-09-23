#!/usr/bin/env python3
"""Batch-run the scanpath model over the stimulus images, ahead of a session.

This is the primary path (spec section 4.2, Tier A). Running the model before
the talk rather than during it costs nothing in fidelity — same weights, same
prompt, same output — and removes every way inference can fail in front of an
audience.

    python tools/precompute.py --repo ../DeepGaze3.5-VL \
        --modes freeview --samples 10 --temperature 0.7

Jobs are ordered so each LoRA adapter is loaded once, not once per image —
switching adapters re-reads sixteen gigabytes and re-merges the LoRA, and
built image-major a mixed run did that twenty times for ten images.

The samples for one config go through a single generate() call. That shares
the weights and the processor work; it does NOT obviously share the image
prefill, because transformers expands the batch for num_return_sequences
before prefill runs. `tools/bench_model.py` measures which it is on your
hardware, and --max-tiles is the lever if it turns out prefill dominates.

There used to be a --synthetic mode: placeholder scanpaths generated without
the model, for building the display with no GPU to hand. It is gone. It did
its job, and then it became the main risk in the room — a plausible centre-
biased blob, stamped source="synthetic" and badged in the UI, which is exactly
the amount of protection that fails when somebody is presenting and reading
the picture rather than the badge. Real runs exist now; there is nothing left
for a stand-in to do.
"""

import argparse
import json
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


def adapter_for(mode: str) -> str:
    """Free viewing has its own adapter; every task-directed mode shares the
    search one. Defined once because the job ordering and the run loop must
    agree — if they disagree the sort stops preventing reloads."""
    return "combined_adapter" if mode == "freeview" else "visual_search_adapter"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True,
                    help="DeepGaze3.5-VL checkout")
    ap.add_argument("--images-dir", default=str(DEFAULT_IMAGES))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--only", nargs="*", help="Limit to these image filenames")
    ap.add_argument("--modes", nargs="+", default=["freeview"],
                    choices=["freeview", "search", "probe"])
    ap.add_argument("--targets", nargs="*", default=["car"],
                    help="Search targets, used when 'search' is in --modes")
    ap.add_argument("--probes", nargs="*",
                    help="Probe ids to run (see --list-probes). Implies 'probe' mode.")
    ap.add_argument("--all-probes", action="store_true",
                    help="Run every probe in the catalogue")
    ap.add_argument("--list-probes", action="store_true",
                    help="Print the probe catalogue and exit")
    ap.add_argument("--num-fixations", type=int, default=5,
                    help="Match the participant tap count for clean comparison")
    ap.add_argument("--samples", type=int, default=10, help="Virtual observers")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bfloat16", "float16", "float32"],
                    help="auto picks what the hardware supports")
    ap.add_argument("--attn", default=None,
                    help="attn_implementation to request (sdpa, flash_attention_2)")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="Cap InternVL's dynamic image tiling")
    ap.add_argument("--load-on-device", action="store_true",
                    help="Merge the LoRA on the accelerator, not on CPU")
    ap.add_argument("--quant", choices=["none", "8bit", "4bit"], default=None,
                    help="Quantise the weights; the profile chooses by default")
    ap.add_argument("--chunk", type=int, default=None,
                    help="Samples per generate() call; the profile chooses")
    ap.add_argument("--force", action="store_true", help="Re-run existing outputs")
    args = ap.parse_args()

    if args.list_probes:
        print(f"{'id':18s} {'kind':14s} label")
        for pr in gp.PROBES:
            print(f"{pr['id']:18s} {pr['kind']:14s} {pr['label']}")
            print(f"{'':18s} {'':14s} {pr['note']}")
        return 0

    # Probe ids expand into (mode, target) pairs.
    probe_ids = list(args.probes or [])
    if args.all_probes:
        probe_ids = [pr["id"] for pr in gp.PROBES]
    probe_jobs = []
    for pid in probe_ids:
        pr = gp.PROBES_BY_ID.get(pid)
        if not pr:
            ap.error(f"unknown probe {pid!r}; see --list-probes")
        probe_jobs.append((pr["mode"], pr["target"]))

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
        if probe_jobs:
            for mode, target in probe_jobs:
                jobs.append((img, mode, target))
        else:
            for mode in args.modes:
                targets = args.targets if mode == "search" else [None]
                for target in targets:
                    jobs.append((img, mode, target))

    # Group by adapter, because switching adapters reloads sixteen gigabytes
    # of weights and re-merges the LoRA. Built image-major, a mixed run
    # alternated between the two adapters once per image: ten images asking
    # for free viewing and any probe meant twenty loads where two would do.
    # Sorting is stable, so within an adapter the original order survives.
    jobs.sort(key=lambda j: adapter_for(j[1]))

    print(f"{len(images)} image(s) x {len(jobs) // len(images)} config(s) "
          f"= {len(jobs)} run(s), {args.samples} sample(s) each")
    model = processor = None
    import predict
    import backends
    device = predict.pick_device(args.device)
    prof = backends.detect(device, args.quant, args.chunk)
    if args.dtype != "auto":
        prof.dtype = args.dtype
    if args.attn:
        prof.attn = args.attn
    if args.max_tiles:
        prof.max_tiles = args.max_tiles
    dtype_name = prof.dtype
    print(backends.describe(prof))
    # Freeview and search use different adapters, so a mixed run reloads.
    adapters_needed = {adapter_for(m) for _, m, _ in jobs}
    if len(adapters_needed) > 1:
        print(f"NOTE: this run needs {len(adapters_needed)} adapters; jobs "
              f"are ordered so each is loaded once.")

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
            import predict
            from PIL import Image

            # Probes are task-directed, so they use the search adapter.
            adapter = adapter_for(mode)
            if adapter != loaded_adapter:
                model, processor = predict.load_model(
                    args.repo, adapter, device, dtype_name,
                    attn=prof.attn, quant=prof.quant,
                    on_device=args.load_on_device or prof.load_on_device)
                loaded_adapter = adapter

            image = Image.open(img).convert("RGB")
            texts = predict.predict(
                model, processor, image, prompt_text, args.samples,
                args.temperature, args.seed, device,
                max(64, 16 * args.num_fixations + 16),
                max_tiles=prof.max_tiles, chunk=prof.sample_chunk)
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
            "prompt_kind": ("trained" if mode in ("freeview", "search")
                            else "experimental_probe"),
            "probe_id": (gp.probe_for(mode, target) or {}).get("id"),
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
    if written:
        print("Restart the app (or press Rescan in /admin) to pick these up.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
