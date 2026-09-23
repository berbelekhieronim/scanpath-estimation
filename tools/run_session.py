#!/usr/bin/env python3
"""Run the session's model predictions, on a machine you have to trust for hours.

This exists because `predict.py` is the right tool for one prediction and the
wrong one for a seven-hour job. Three things go wrong at that length:

  1. **It silently runs on the CPU.** On a Mac, a torch build without MPS, or
     a stray PYTORCH_ENABLE_MPS_FALLBACK, turns "the GPU is busy" into "the
     GPU is idle and this will take two days". There is no error — only
     slowness, which is indistinguishable from the model being big. So this
     script *measures* where the work is happening before committing to it.

  2. **You find out how long it takes by waiting.** One sample is timed
     first and the total extrapolated from it, so the decision to spend the
     evening is made in the first two minutes.

  3. **A crash at hour five loses hours one through four.** Samples are
     written to disk as they complete, and --resume picks up the rest.

    python tools/run_session.py --repo ../DeepGaze3.5-VL \
        --image data/images/street_capybara_sign.jpg \
        --samples 25 --num-fixations 10

Add --probe-only to get the timing estimate and stop.

A note on the Neural Engine, because it is the obvious question: PyTorch
cannot reach it. The ANE is addressable only through CoreML, which would mean
converting an 8B InternVL and its LoRA to a CoreML package — and the ANE's
working limits are nowhere near an 8B model in any case. On Apple Silicon
"using the hardware" means the GPU, through MPS, and that is what this checks.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaze_prompts as gp  # noqa: E402
import backends            # noqa: E402
import predict as P        # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Is the accelerator actually doing the work?
# --------------------------------------------------------------------------

def preflight(device: str) -> list:
    """Report what the machine will really use, and say so in plain terms.

    Returns a list of warnings. The caller decides whether they are fatal;
    the point is that they are never invisible.
    """
    import torch
    warnings = []
    print(f"torch {torch.__version__}")

    if device == "mps":
        built = torch.backends.mps.is_built()
        avail = torch.backends.mps.is_available()
        print(f"MPS built into this torch: {built}")
        print(f"MPS available right now:   {avail}")
        if not (built and avail):
            warnings.append(
                "This torch build cannot use the Mac's GPU. Everything will "
                "run on the CPU, which is roughly twenty times slower. "
                "Install a torch built with MPS: "
                "pip install --upgrade torch torchvision")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
            warnings.append(
                "PYTORCH_ENABLE_MPS_FALLBACK is set. Any operator MPS does "
                "not implement will run on the CPU *silently*, moving "
                "tensors back and forth each time. That is the usual reason "
                "a run 'works, but took all night'. Unset it and see whether "
                "the model actually needs it.")
        # A real measurement beats a capability flag: a build can report MPS
        # and still fall back for most of what a transformer does.
        try:
            print(_throughput_line(torch, "mps"))
            print(_throughput_line(torch, "cpu"))
        except Exception as exc:
            warnings.append(f"Could not benchmark the devices: {exc}")

    elif device == "cuda":
        if not torch.cuda.is_available():
            warnings.append("No CUDA device visible; this will run on the CPU.")
        else:
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {props.name}, {props.total_memory / 1e9:.0f}GB, "
                  f"sm_{props.major}{props.minor}")
            if props.major >= 12 and torch.__version__ < "2.7":
                warnings.append(
                    f"This card is sm_{props.major}{props.minor} (Blackwell) "
                    f"and torch {torch.__version__} has no kernels for it. "
                    f"Install torch 2.7+ built against CUDA 12.8.")
    else:
        warnings.append(
            "Running on the CPU. An 8B model generates a few tokens a second "
            "there; twenty-five samples is a day, not an evening.")
    return warnings


def _throughput_line(torch, dev: str) -> str:
    """Matrix multiply throughput, which is what the model is mostly doing."""
    import time as _t
    n = 2048
    try:
        a = torch.randn(n, n, device=dev, dtype=torch.float32)
        b = torch.randn(n, n, device=dev, dtype=torch.float32)
        for _ in range(2):                       # warm up kernels / caches
            a @ b
        _sync(torch, dev)
        t0 = _t.perf_counter()
        for _ in range(10):
            a @ b
        _sync(torch, dev)
        secs = _t.perf_counter() - t0
        gflops = 10 * 2 * n ** 3 / secs / 1e9
        return f"  {dev:4s} {gflops:8.0f} GFLOP/s"
    except Exception as exc:
        return f"  {dev:4s} unavailable ({exc})"


def _sync(torch, dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

def load_checkpoint(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("texts") or []
    except Exception:
        return []


def save_checkpoint(path: Path, texts: list, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"texts": texts, **meta}, indent=1))
    tmp.replace(path)          # atomic, so a crash mid-write cannot corrupt it


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--mode", choices=["freeview", "search"], default="freeview")
    ap.add_argument("--target", help="Search target (required for --mode search)")
    ap.add_argument("--samples", type=int, default=25,
                    help="Virtual observers (default 25)")
    ap.add_argument("--num-fixations", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--quant", choices=["none", "8bit", "4bit"], default=None)
    ap.add_argument("--output", help="Defaults to data/model/<derived name>.json")
    ap.add_argument("--probe-only", action="store_true",
                    help="Time one sample, print the estimate, stop")
    ap.add_argument("--resume", action="store_true",
                    help="Continue a run that stopped part way")
    ap.add_argument("--yes", action="store_true",
                    help="Do not pause for confirmation after the estimate")
    ap.add_argument("--force-cpu-ok", action="store_true",
                    help="Proceed even if the accelerator is not being used")
    args = ap.parse_args()

    if args.mode == "search" and not args.target:
        ap.error("--target is required when --mode is 'search'")
    if args.samples > 1 and args.temperature <= 0:
        ap.error("--samples > 1 with temperature 0 returns the same path N times")

    device = P.pick_device(args.device)
    print(f"\n=== Machine ===\ndevice: {device}")
    warnings = preflight(device)
    for w in warnings:
        print(f"\n!! {w}")
    if warnings and device != "cuda" and not args.force_cpu_ok:
        print("\nFix the above, or pass --force-cpu-ok to run anyway.")
        if not args.yes:
            if input("Continue regardless? [y/N] ").strip().lower() != "y":
                return 1

    prof = backends.detect(device, args.quant, args.chunk)
    if args.max_tiles:
        prof.max_tiles = args.max_tiles
    print(f"\n=== Profile ===\n{backends.describe(prof)}")

    n = args.num_fixations
    prompt_text = gp.build_prompt(args.mode, n, args.target)
    max_new = max(64, 16 * n + 16)
    adapter = ("combined_adapter" if args.mode == "freeview"
               else "visual_search_adapter")

    stem = Path(args.image).stem
    bits = [stem, args.mode] + ([args.target.replace(" ", "-")] if args.target else [])
    name = "__".join(bits + [f"n{n}"]) + ".json"
    out_path = Path(args.output) if args.output else REPO_ROOT / "data" / "model" / name
    ckpt = out_path.with_name(out_path.stem + ".partial.json")

    done = load_checkpoint(ckpt) if args.resume else []
    if done:
        print(f"\nResuming: {len(done)} of {args.samples} samples already done.")
    elif ckpt.exists() and not args.resume:
        print(f"\nNote: {ckpt.name} exists. Pass --resume to continue it, "
              f"or delete it to start over.")

    from PIL import Image
    image = Image.open(args.image).convert("RGB")

    print(f"\n=== Loading {adapter} ===")
    t_load = time.time()
    model, processor = P.load_model(
        args.repo, adapter, device, prof.dtype, attn=prof.attn,
        quant=prof.quant, on_device=prof.load_on_device)
    print(f"loaded in {time.time() - t_load:.0f}s")

    def run(count, seed_offset):
        return P.predict(model, processor, image, prompt_text, count,
                         args.temperature, args.seed + seed_offset, device,
                         max_new, max_tiles=prof.max_tiles, chunk=prof.sample_chunk)

    # --- the probe -------------------------------------------------------
    # One sample, timed. Everything about whether this evening is worth
    # spending follows from this number, and it costs two minutes to get.
    print("\n=== Timing one sample ===")
    t0 = time.time()
    first = run(1, 1000)
    per_sample = time.time() - t0
    print(f"one sample: {per_sample:.0f}s")

    remaining = max(0, args.samples - len(done))
    est = per_sample * remaining
    print(f"\n{remaining} sample(s) to go — estimated "
          f"{est / 60:.0f} min ({est / 3600:.1f} h)")
    print("The estimate is optimistic by however much the machine throttles "
          "or swaps over that span; treat it as a floor.")
    if args.probe_only:
        print("\n--probe-only: stopping here.")
        print(f"sample text: {first[0][:200] if first else '(none)'}")
        return 0
    if not args.yes:
        if input("\nRun it? [y/N] ").strip().lower() != "y":
            return 1

    # --- the run ---------------------------------------------------------
    meta = {"image": Path(args.image).name, "mode": args.mode,
            "target": args.target, "n_fixations": n, "samples": args.samples,
            "temperature": args.temperature, "seed": args.seed}
    started = time.time()
    # One chunk at a time, so the checkpoint is written between each. The
    # profile's chunk size is respected inside predict(); what changes here
    # is that control comes back often enough to save.
    step = max(1, prof.sample_chunk)
    while len(done) < args.samples:
        k = min(step, args.samples - len(done))
        # Seeded by how many are already done, so a resumed run does not
        # repeat the scanpaths it already has.
        done.extend(run(k, len(done)))
        save_checkpoint(ckpt, done, meta)
        elapsed = time.time() - started
        rate = elapsed / max(1, len(done))
        left = rate * (args.samples - len(done))
        print(f"  {len(done)}/{args.samples} done · {elapsed / 60:.0f} min in "
              f"· ~{left / 60:.0f} min left", flush=True)

    # --- the result ------------------------------------------------------
    grids = [g for g in (gp.parse_scanpath(t) for t in done) if g]
    if not grids:
        print("\nThe model returned nothing parseable. Raw output kept in "
              f"{ckpt}", file=sys.stderr)
        return 1
    short = [len(g) for g in grids if len(g) != n]
    if short:
        print(f"\nNOTE: {len(short)} sample(s) came back with a different "
              f"number of fixations than the {n} asked for: {sorted(set(short))}")

    result = {
        "image": Path(args.image).name, "mode": args.mode, "target": args.target,
        "n_fixations": n, "seed": args.seed, "temperature": args.temperature,
        "prompt_text": prompt_text, "prompt_kind": "trained",
        "scanpath_grid": grids[0], "scanpath_norm": gp.grid_to_norm(grids[0]),
        "samples_grid": grids, "samples_norm": [gp.grid_to_norm(g) for g in grids],
        "source": "precomputed",
        "model": f"{gp.BASE_MODEL} + {adapter}",
        "device": device, "dtype": prof.dtype, "profile": prof.name,
        "quant": prof.quant, "sample_chunk": prof.sample_chunk,
        "elapsed_seconds": round(time.time() - started, 1),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    ckpt.unlink(missing_ok=True)

    mins = (time.time() - started) / 60
    print(f"\nWrote {out_path}")
    print(f"{len(grids)} scanpaths of {n} fixations in {mins:.0f} min on {device}")
    print("\nCopy it to the app machine's data/model/ and press Rescan on "
          "/control, or restart the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
