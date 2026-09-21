#!/usr/bin/env python3
"""Find out where the 85 minutes actually went.

One image and ten samples should not take an hour and a half. This measures
the pipeline instead of guessing at it, because every candidate explanation
below is plausible and only one or two of them will turn out to matter.

    python tools/bench_model.py --repo ../DeepGaze3.5-VL \
        --image data/images/street_capybara_sign.jpg --device cuda

What it measures, and why each is suspected:

1. **Load, split three ways** — weights, LoRA merge, device transfer. The
   model is read on CPU and only then moved, so a bf16 merge runs as CPU
   matmuls on 8B parameters before anything reaches the GPU.

2. **How many tokens the image becomes.** InternVL tiles images dynamically:
   a photo can arrive as a dozen 448x448 tiles plus a thumbnail, which is
   thousands of vision tokens. Nothing in this pipeline caps that, so it is
   whatever the processor felt like. Prefill cost and KV cache both scale
   with it.

3. **Whether ten samples cost ten prefills.** This is the one that matters
   most. predict() asks generate() for num_return_sequences=n, and the
   docstring claims that pays the image prefill once. Transformers expands
   the batch *before* prefill, repeat-interleaving pixel_values along with
   everything else — so the vision tower may be running on ten identical
   copies of the same photograph. If samples=10 costs about ten times
   samples=1, that is what is happening and the docstring is wrong.

Times are wall clock and the first generate() includes warm-up, so a warm-up
pass runs first and is discarded.
"""

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaze_prompts as gp  # noqa: E402

marks = []


@contextmanager
def step(label):
    t0 = time.time()
    yield
    dt = time.time() - t0
    marks.append((label, dt))
    print(f"  {label:34s} {dt:8.1f}s", file=sys.stderr)


def sync(device):
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--samples", type=int, default=8,
                    help="The n to compare against a single sample")
    ap.add_argument("--adapter", default="combined_adapter")
    ap.add_argument("--attn", default=None,
                    help="attn_implementation to request, e.g. sdpa or "
                         "flash_attention_2. Default: whatever transformers picks")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="Cap InternVL's dynamic tiling. Try 4 or 6 against "
                         "the default to see what the tiles are costing")
    ap.add_argument("--output", help="Write the measurements here as JSON")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    import predict_mps
    device = predict_mps.pick_device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"\nDevice: {device}   dtype: {args.dtype}", file=sys.stderr)
    print("\nLoad", file=sys.stderr)

    kw = {"dtype": dtype, "trust_remote_code": True, "low_cpu_mem_usage": True}
    if args.attn:
        kw["attn_implementation"] = args.attn

    with step("processor"):
        processor = AutoProcessor.from_pretrained(gp.BASE_MODEL,
                                                  trust_remote_code=True)
    with step("base weights (to CPU)"):
        model = AutoModelForImageTextToText.from_pretrained(gp.BASE_MODEL, **kw)
    with step("LoRA merge (on CPU)"):
        model = PeftModel.from_pretrained(
            model, str(Path(args.repo) / "model" / args.adapter))
        model = model.merge_and_unload()
    with step(f"transfer to {device}"):
        model.to(device).eval()
        sync(device)

    # --- what the image actually becomes -------------------------------
    image = Image.open(args.image).convert("RGB")
    prompt_text = gp.build_prompt("freeview", 5, None)
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt_text}]}]
    chat = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)

    proc_kw = {}
    if args.max_tiles is not None:
        # The kwarg name moves between processor versions, so try the known
        # spellings and report which one the installed one accepted.
        for name in ("max_num_tiles", "max_num", "max_patches"):
            try:
                processor(images=image, text=chat, return_tensors="pt",
                          **{name: args.max_tiles})
                proc_kw = {name: args.max_tiles}
                print(f"\n  tiling capped via {name}={args.max_tiles}",
                      file=sys.stderr)
                break
            except TypeError:
                continue
        if not proc_kw:
            print("\n  WARNING: this processor accepted none of the known "
                  "tile-cap kwargs; running uncapped", file=sys.stderr)

    inputs = processor(images=image, text=chat, return_tensors="pt",
                       **proc_kw).to(device)
    n_text = inputs["input_ids"].shape[-1]
    pv = inputs.get("pixel_values")
    tiles = tuple(pv.shape) if pv is not None else None

    print(f"\nInput\n  image                  {image.size[0]}x{image.size[1]}"
          f"\n  pixel_values           {tiles}"
          f"\n  prompt tokens          {n_text}", file=sys.stderr)
    if pv is not None and pv.dim() >= 3:
        print(f"  -> {pv.shape[0]} tile(s) through the vision tower",
              file=sys.stderr)

    # --- does n samples cost n prefills? -------------------------------
    def run(n):
        torch.manual_seed(42)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=96, do_sample=True, temperature=0.7,
                num_return_sequences=n,
                pad_token_id=processor.tokenizer.pad_token_id
                             or processor.tokenizer.eos_token_id)
        sync(device)
        return time.time() - t0, out.shape

    print("\nGeneration", file=sys.stderr)
    with step("warm-up (discarded)"):
        run(1)
    t1, shape1 = run(1)
    print(f"  {'samples=1':34s} {t1:8.1f}s   out {tuple(shape1)}", file=sys.stderr)
    tn, shapen = run(args.samples)
    print(f"  {f'samples={args.samples}':34s} {tn:8.1f}s   out {tuple(shapen)}",
          file=sys.stderr)

    ratio = tn / t1 if t1 else 0
    print(f"\n  ratio {ratio:.1f}x for {args.samples}x the samples",
          file=sys.stderr)
    if ratio > args.samples * 0.7:
        verdict = ("SCALES LINEARLY — the image prefill is being paid once per "
                   "sample. num_return_sequences expands the batch before "
                   "prefill, so the vision tower is re-encoding the same photo "
                   f"{args.samples} times. Prefill once and expand the cache, "
                   "or accept it and cut the tile count.")
    elif ratio < 2.0:
        verdict = ("FLAT — prefill really is shared, and the samples are nearly "
                   "free. The time is going into load or prefill, not sampling.")
    else:
        verdict = ("PARTIAL — some work is shared and some is not. Worth "
                   "profiling prefill separately from decode.")
    print(f"\n  {verdict}\n", file=sys.stderr)

    total_load = sum(d for label, d in marks if label != "warm-up (discarded)"
                     and "samples" not in label)
    print(f"Load total {total_load:.0f}s   |   one 10-sample run "
          f"{tn:.0f}s   |   ten such runs would be "
          f"{(total_load + 10 * tn) / 60:.0f} min\n", file=sys.stderr)

    if args.output:
        Path(args.output).write_text(json.dumps({
            "device": device, "dtype": args.dtype, "attn": args.attn,
            "max_tiles": args.max_tiles, "tile_kwarg": proc_kw or None,
            "pixel_values_shape": tiles, "prompt_tokens": n_text,
            "steps": dict(marks), "t_one_sample": round(t1, 2),
            "t_n_samples": round(tn, 2), "n": args.samples,
            "ratio": round(ratio, 2), "verdict": verdict,
        }, indent=2))
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
