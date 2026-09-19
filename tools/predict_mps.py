#!/usr/bin/env python3
"""Scanpath inference for DeepGaze3.5-VL on Apple Silicon (or CPU, or CUDA).

Upstream runs through vLLM, whose standard install requires CUDA — which
Apple Silicon does not have. vLLM is doing exactly one job there, fast batched
serving, and this workload is one image producing about sixty tokens. So this
script drops vLLM and drives the same weights through plain transformers.

The prompt construction is upstream's, reproduced verbatim in gaze_prompts.py
and verified against their source with `gaze_prompts.py --verify`. That is the
part that determines output quality, because the LoRA was fine-tuned on those
exact strings.

Setup (on the MacBook, NOT in the Codespace):

    git clone https://github.com/Susmit-A/DeepGaze3.5-VL   # needs git-lfs
    python3 -m venv .venv-model && source .venv-model/bin/activate
    pip install -r requirements-model.txt

Then:

    python tools/predict_mps.py --repo ../DeepGaze3.5-VL \
        --image data/images/street.jpg --mode freeview --num-fixations 5

First run downloads ~16GB of base model from HuggingFace. Expect 1-3 minutes
per image after that; slower on CPU. That is fine — see spec section 4.2.
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
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(repo: str, adapter: str, device: str, dtype_name: str):
    """Base model + LoRA adapter, merged, on the chosen device."""
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
    model = AutoModelForImageTextToText.from_pretrained(
        gp.BASE_MODEL, dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
    )

    print(f"Merging LoRA adapter: {adapter_path}", file=sys.stderr)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()

    model.to(device).eval()
    return model, processor


def predict(model, processor, image, prompt_text, n_samples, temperature,
            seed, device, max_new_tokens):
    """One prefill, n_samples decodes.

    Batching the samples in a single generate() call is what makes virtual
    observers affordable: the image prefill is thousands of vision tokens and
    is identical across samples, so paying it once instead of n times turns
    ten observers into roughly the cost of one.
    """
    import torch

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt_text}]}]
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=image, text=chat, return_tensors="pt").to(device)

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
    ap.add_argument("--mode", choices=["freeview", "search"], default="freeview")
    ap.add_argument("--target", help="Search target (required for --mode search)")
    ap.add_argument("--num-fixations", type=int, default=None,
                    help="Default 8 for freeview, 3 for search (upstream defaults)")
    ap.add_argument("--samples", type=int, default=1,
                    help="Virtual observers. Needs --temperature > 0")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy, one deterministic path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--output", help="Write result JSON here")
    args = ap.parse_args()

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
    prompt_text = gp.build_prompt(args.mode, n, args.target)

    from PIL import Image
    image = Image.open(args.image).convert("RGB")

    device = pick_device(args.device)
    print(f"Device: {device}", file=sys.stderr)
    if device == "cpu":
        print("CPU inference works but is slow (10-20 min/image is normal).",
              file=sys.stderr)

    adapter = "combined_adapter" if args.mode == "freeview" else "visual_search_adapter"
    model, processor = load_model(args.repo, adapter, device, args.dtype)

    t0 = time.time()
    texts = predict(model, processor, image, prompt_text, args.samples,
                    args.temperature, args.seed, device, max_new)
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
        "scanpath_grid": samples_grid[0],
        "scanpath_norm": gp.grid_to_norm(samples_grid[0]),
        "samples_grid": samples_grid,
        "samples_norm": [gp.grid_to_norm(s) for s in samples_grid],
        "source": "live",
        "model": f"{gp.BASE_MODEL} + {adapter}",
        "device": device,
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
