#!/usr/bin/env python3
"""Scanpath inference for DeepGaze3.5-VL on NVIDIA, Apple Silicon, or CPU.

Upstream runs through vLLM, whose standard install requires CUDA — which
Apple Silicon does not have. vLLM is doing exactly one job there, fast batched
serving, and this workload is one image producing about sixty tokens. So this
script drops vLLM and drives the same weights through plain transformers.

The prompt construction is upstream's, reproduced verbatim in gaze_prompts.py
and verified against their source with `gaze_prompts.py --verify`. That is the
part that determines output quality, because the LoRA was fine-tuned on those
exact strings.

The device and the numeric format are both chosen for the hardware: bfloat16
on Apple Silicon and on Ampere-or-newer NVIDIA, float16 on older NVIDIA parts
that have no native bfloat16, float32 on CPU. A GTX card or an RTX 20-series
falls in that second group.

Setup (on whichever machine has the accelerator, NOT in the Codespace):

    git clone https://github.com/Susmit-A/DeepGaze3.5-VL   # needs git-lfs
    python3 -m venv .venv-model && source .venv-model/bin/activate
    pip install -r requirements-model.txt

Then:

    python tools/predict.py --repo ../DeepGaze3.5-VL \
        --image data/images/street.jpg --mode freeview --num-fixations 5

First run downloads ~16GB of base model from HuggingFace. The weights alone
then need about 16GB of VRAM in 16-bit, which is comfortable on a 24GB card
and will not fit a 10-12GB one; --max-tiles is the first lever if memory is
tight. Run tools/bench_model.py before trusting any timing estimate.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaze_prompts as gp  # noqa: E402


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    # CUDA first: where both exist it is the faster path by a wide margin,
    # and the ordering used to hand an RTX machine to Apple's backend.
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def pick_dtype(requested: str, device: str) -> str:
    """Choose a numeric format the hardware can actually run.

    bfloat16 is the right default on Apple Silicon and on Ampere or newer
    NVIDIA parts, and is NOT natively supported below compute capability 8.0.
    On a GTX card, or an RTX 20-series, asking for bfloat16 gets either an
    error or a slow emulated path — and 'bfloat16' was the hard-coded
    default, so every pre-Ampere GPU would have hit it.

    float32 on CPU, because bfloat16 there is slower than the format it was
    meant to speed up.

    torch is imported only where it is needed — inside the CUDA branch — so
    the answer for every other device can be had without it.
    """
    if requested != "auto":
        return requested
    if device == "cuda":
        import torch
        try:
            if torch.cuda.is_bf16_supported():
                return "bfloat16"
        except Exception:
            pass
        cap = torch.cuda.get_device_capability()
        print(f"NOTE: this GPU (compute capability {cap[0]}.{cap[1]}) has no "
              f"native bfloat16; using float16.", file=sys.stderr)
        return "float16"
    if device == "mps":
        return "bfloat16"
    return "float32"


def check_vram(device: str, dtype_name: str) -> None:
    """Say so before the download, not after the out-of-memory.

    Eight billion parameters at two bytes each is about 16GB of weights
    before any activations or KV cache. That fits a 24GB card and does not
    fit a 10 or 11GB one, which is the difference between most of the RTX
    xx90 line and most of everything older.
    """
    import torch

    if device != "cuda":
        return
    try:
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        name = torch.cuda.get_device_name(0)
    except Exception:
        return

    bytes_per = 4 if dtype_name == "float32" else 2
    weights = 8.0 * bytes_per          # ~8B parameters
    print(f"GPU: {name}, {total:.0f}GB — weights alone are about "
          f"{weights:.0f}GB in {dtype_name}.", file=sys.stderr)
    if total < weights + 4:
        print(f"WARNING: that leaves little or nothing for activations and "
              f"the KV cache. Expect an out-of-memory error. Options, in "
              f"order of how much they cost you: --max-tiles 4 (far smaller "
              f"prefill), fewer --samples, load in 8-bit or 4-bit "
              f"(bitsandbytes), or run on the Mac.", file=sys.stderr)


def load_model(repo: str, adapter: str, device: str, dtype_name: str,
               attn: str = None, on_device: bool = False):
    """Base model + LoRA adapter, merged, on the chosen device.

    `on_device` loads the weights straight onto the accelerator so the LoRA
    merge runs there too. The default path reads sixteen gigabytes to CPU,
    merges 8B parameters as CPU matmuls, and only then copies everything
    across. Opt-in rather than default because it changes how the weights are
    materialised, and a run that works slowly beats one that does not run.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    adapter_path = Path(repo) / "model" / adapter
    if not adapter_path.exists():
        raise SystemExit(
            f"Adapter not found: {adapter_path}\n"
            f"Pass --repo pointing at a DeepGaze3.5-VL checkout. If the folder "
            f"exists but the .safetensors is a small text stub, the clone did "
            f"not fetch Git LFS objects — run 'git lfs pull' in that checkout."
        )

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_name]

    print(f"Loading {gp.BASE_MODEL} ({dtype_name}) — first run downloads ~16GB…",
          file=sys.stderr)
    processor = AutoProcessor.from_pretrained(gp.BASE_MODEL, trust_remote_code=True)

    kw = {"dtype": dtype, "trust_remote_code": True, "low_cpu_mem_usage": True}
    if attn:
        kw["attn_implementation"] = attn
    if on_device and device != "cpu":
        kw["device_map"] = {"": device}

    try:
        model = AutoModelForImageTextToText.from_pretrained(gp.BASE_MODEL, **kw)
    except (ValueError, TypeError, ImportError) as exc:
        # An unsupported attention backend or a missing accelerate should cost
        # speed, never the run.
        dropped = [k for k in ("attn_implementation", "device_map") if k in kw]
        if not dropped:
            raise
        print(f"NOTE: {', '.join(dropped)} not usable here ({exc}); loading "
              f"without.", file=sys.stderr)
        for k in dropped:
            kw.pop(k)
        model = AutoModelForImageTextToText.from_pretrained(gp.BASE_MODEL, **kw)

    print(f"Merging LoRA adapter: {adapter_path}", file=sys.stderr)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()

    # Already there when device_map placed it; .to() is then a no-op.
    model.to(device).eval()
    return model, processor


def _tiling_kwargs(processor, image, chat, max_tiles):
    """Cap InternVL's dynamic tiling, if this processor version allows it.

    A photograph becomes a variable number of 448px tiles plus a thumbnail,
    each worth a few hundred vision tokens, and nothing here constrained it —
    so prefill cost was whatever the processor defaulted to. The kwarg name
    has moved between versions, so the accepted spelling is discovered rather
    than assumed, and an unsupported cap is reported instead of raising.
    """
    if not max_tiles:
        return {}
    for name in ("max_num_tiles", "max_num", "max_patches"):
        try:
            processor(images=image, text=chat, return_tensors="pt",
                      **{name: max_tiles})
            return {name: max_tiles}
        except TypeError:
            continue
    print(f"NOTE: this processor accepts no known tile-cap argument; "
          f"--max-tiles {max_tiles} ignored.", file=sys.stderr)
    return {}


def predict(model, processor, image, prompt_text, n_samples, temperature,
            seed, device, max_new_tokens, max_tiles=None):
    """All samples in one generate() call.

    This used to claim it paid the image prefill once for every sample. That
    is very likely false: transformers expands the batch for
    num_return_sequences *before* prefill, repeat-interleaving pixel_values
    with everything else, so the vision tower probably encodes n identical
    copies of the photograph. `tools/bench_model.py` settles it by timing one
    sample against n — if the ratio is near n, the prefill is being repeated.

    It is still one call rather than n calls, which keeps the weights and the
    processor work shared. But do not read it as free.
    """
    import torch

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt_text}]}]
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    proc_kw = _tiling_kwargs(processor, image, chat, max_tiles)
    inputs = processor(images=image, text=chat, return_tensors="pt",
                       **proc_kw).to(device)

    torch.manual_seed(seed)
    greedy = temperature <= 0.0
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            num_return_sequences=1 if greedy else n_samples,
            pad_token_id=processor.tokenizer.pad_token_id
                         or processor.tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    texts = processor.tokenizer.batch_decode(
        out[:, prompt_len:], skip_special_tokens=True)
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True,
                    help="Path to a DeepGaze3.5-VL checkout (for the adapters)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mode", choices=["freeview", "search", "probe"], default="freeview")
    ap.add_argument("--probe", help="Probe id; sets --mode and --target together")
    ap.add_argument("--target", help="Search target (required for --mode search)")
    ap.add_argument("--num-fixations", type=int, default=None,
                    help="Default 8 for freeview, 3 for search (upstream defaults)")
    ap.add_argument("--samples", type=int, default=1,
                    help="Virtual observers. Needs --temperature > 0")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy, one deterministic path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bfloat16", "float16", "float32"],
                    help="auto picks what the hardware supports: bfloat16 on "
                         "Apple Silicon and Ampere-or-newer NVIDIA, float16 "
                         "on older cards, float32 on CPU")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--attn", default=None,
                    help="attn_implementation to request (sdpa, "
                         "flash_attention_2). Measure with bench_model.py "
                         "before relying on it")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="Cap InternVL's dynamic image tiling. Fewer tiles "
                         "means a much cheaper prefill and a smaller cache, "
                         "at some loss of detail")
    ap.add_argument("--load-on-device", action="store_true",
                    help="Materialise weights on the accelerator so the LoRA "
                         "merge runs there instead of as CPU matmuls")
    ap.add_argument("--prompt", help=(
        "EXPERIMENTAL. Replace the trained prompt with your own text. The "
        "adapter was fine-tuned on two exact templates; anything else is "
        "off-distribution. The model will still return coordinates — it "
        "always does — but their quality is unvalidated and degrades "
        "silently. Runs made this way are badged in the UI."))
    ap.add_argument("--output", help="Write result JSON here")
    args = ap.parse_args()

    if args.probe:
        pr = gp.PROBES_BY_ID.get(args.probe)
        if not pr:
            ap.error(f"unknown probe {args.probe!r}; choices: "
                     f"{', '.join(p['id'] for p in gp.PROBES)}")
        args.mode, args.target = pr["mode"], pr["target"]
        if pr["kind"] == "experimental":
            print(f"NOTE: probe '{pr['id']}' is experimental — the adapter was "
                  f"never trained on this task. Output quality is unvalidated.",
                  file=sys.stderr)
    if args.mode == "search" and not args.target:
        ap.error("--target is required when --mode is 'search'")
    if args.samples > 1 and args.temperature <= 0.0:
        ap.error("--samples > 1 needs --temperature > 0, or every sample is identical")
    if args.mode == "search" and args.target not in gp.COCO_SEARCH18_TARGETS:
        print(f"WARNING: '{args.target}' is not one of the 18 trained targets "
              f"({', '.join(gp.COCO_SEARCH18_TARGETS)}). Proceeding, but output "
              f"quality is unvalidated.", file=sys.stderr)

    n = args.num_fixations or (8 if args.mode == "freeview" else 3)
    max_new = args.max_new_tokens or max(64, 16 * n + 16)

    if args.prompt:
        prompt_text = args.prompt
        prompt_kind = "custom"
        print("\n*** EXPERIMENTAL PROMPT — off the trained distribution. ***\n"
              "*** Output quality is unvalidated. Do not present this as   ***\n"
              "*** the model's validated prediction.                       ***\n",
              file=sys.stderr)
    else:
        prompt_text = gp.build_prompt(args.mode, n, args.target)
        prompt_kind = "trained"

    from PIL import Image
    image = Image.open(args.image).convert("RGB")

    device = pick_device(args.device)
    dtype_name = pick_dtype(args.dtype, device)
    print(f"Device: {device}   dtype: {dtype_name}", file=sys.stderr)
    check_vram(device, dtype_name)
    if device == "cpu":
        print("CPU inference works but is slow (10-20 min/image is normal).",
              file=sys.stderr)

    # Probes are task-directed, so they use the search adapter.
    adapter = "combined_adapter" if args.mode == "freeview" else "visual_search_adapter"
    model, processor = load_model(args.repo, adapter, device, dtype_name,
                                  attn=args.attn,
                                  on_device=args.load_on_device)

    t0 = time.time()
    texts = predict(model, processor, image, prompt_text, args.samples,
                    args.temperature, args.seed, device, max_new,
                    max_tiles=args.max_tiles)
    elapsed = time.time() - t0

    samples_grid = [gp.parse_scanpath(t) for t in texts]
    samples_grid = [s for s in samples_grid if s]
    if not samples_grid:
        print("Model returned no parseable coordinates. Raw output:", file=sys.stderr)
        for t in texts:
            print(f"  {t!r}", file=sys.stderr)
        raise SystemExit(1)

    short = [len(s) for s in samples_grid if len(s) != n]
    if short:
        print(f"NOTE: {len(short)} sample(s) returned a different number of "
              f"fixations than the {n} requested: {short}", file=sys.stderr)

    result = {
        "image": Path(args.image).name,
        "mode": args.mode,
        "target": args.target,
        "n_fixations": n,
        "seed": args.seed,
        "temperature": args.temperature,
        "prompt_text": prompt_text,
        "prompt_kind": prompt_kind,
        "scanpath_grid": samples_grid[0],
        "scanpath_norm": gp.grid_to_norm(samples_grid[0]),
        "samples_grid": samples_grid,
        "samples_norm": [gp.grid_to_norm(s) for s in samples_grid],
        "source": "live",
        "model": f"{gp.BASE_MODEL} + {adapter}",
        "device": device,
        "dtype": dtype_name,
        "elapsed_seconds": round(elapsed, 1),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print(f"\n{len(samples_grid)} scanpath(s) in {elapsed:.1f}s on {device}")
    for i, s in enumerate(samples_grid):
        print(f"  sample {i}: {s}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.output}")
    else:
        print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
