#!/usr/bin/env python3
"""What each machine can actually do, in one place.

Two machines run this: an M4 Air and an RTX 5070 Ti. They fail in opposite
directions, so a single set of defaults serves neither.

    M4 Air, 32GB unified   weights fit comfortably, but the memory is shared
                           with the OS and everything else, and there is no
                           hard boundary — overcommit does not fail, it
                           swaps, and swapping an 8B model is how one image
                           took eighty-five minutes.

    5070 Ti, 16GB VRAM     fast, and too small for the weights: 8B parameters
                           in 16-bit is 16GB before a single activation.
                           Overcommit here does not swap, it raises. The card
                           needs the model quantised to be usable at all.

The inference logic itself is NOT forked. The prompt strings and the
coordinate parsing are what determine output quality, and two copies of them
would drift — one machine would quietly start producing different scanpaths
from the other, and the comparison between sessions would be worthless.
What differs between machines is a handful of numbers, and those live here.
"""

from dataclasses import dataclass, field


@dataclass
class Profile:
    name: str
    dtype: str
    # How many samples to ask for in one generate() call. This is the lever
    # that matters most. num_return_sequences=n expands the batch before
    # prefill, so the KV cache — and probably the vision tower's work —
    # scales with n. Ten at once is what pushes a 32GB Mac into swap and what
    # puts a 16GB card over its limit.
    sample_chunk: int
    quant: str = "none"          # none | 8bit | 4bit
    attn: str = None             # None = let transformers decide
    max_tiles: int = None        # None = the processor's own default
    load_on_device: bool = False
    notes: list = field(default_factory=list)


def _cuda_profile(vram_gb: float, bf16: bool, capability=None) -> Profile:
    dtype = "bfloat16" if bf16 else "float16"
    weights = 16.0 if dtype in ("bfloat16", "float16") else 32.0
    p = Profile(name=f"cuda-{vram_gb:.0f}gb", dtype=dtype, sample_chunk=4,
                attn="sdpa", load_on_device=True)

    if vram_gb >= weights + 8:
        # Room for the weights and a generous cache.
        p.sample_chunk = 8
    elif vram_gb >= weights + 3:
        p.sample_chunk = 4
        p.notes.append("Tight but workable in 16-bit; keeping chunks small.")
    else:
        # The 5070 Ti lands here: 16GB of card against 16GB of weights.
        p.quant = "8bit"
        p.sample_chunk = 4
        p.max_tiles = 6
        p.notes.append(
            f"{vram_gb:.0f}GB cannot hold {weights:.0f}GB of 16-bit weights "
            f"with anything left for activations, so the model is loaded in "
            f"8-bit (about {weights / 2:.0f}GB). Use --quant 4bit if even "
            f"that is tight, or --quant none on a bigger card.")
    if capability and capability[0] >= 12:
        p.notes.append(
            "Blackwell (sm_120) needs PyTorch 2.7 or newer built against "
            "CUDA 12.8. An older wheel has no kernels for this card and "
            "fails with 'no kernel image is available'.")
    return p


def _mps_profile(total_gb: float) -> Profile:
    # Unified memory is shared, so the budget is not the machine's RAM. Leave
    # the OS and the browser their share; what is left has to hold the
    # weights AND every sample's cache at once.
    p = Profile(name=f"mps-{total_gb:.0f}gb", dtype="bfloat16",
                sample_chunk=3, attn="sdpa", load_on_device=True)
    usable = total_gb - 10.0
    if usable < 20:
        p.sample_chunk = 2
        p.max_tiles = 6
        p.notes.append(
            "Unified memory does not raise when it runs out, it swaps — and "
            "swapping 16GB of weights is indistinguishable from the model "
            "being slow. Small chunks and a tile cap keep it resident.")
    else:
        p.notes.append(
            "Samples run in small chunks: one call for all of them multiplies "
            "the cache by the sample count, which is what pushes this machine "
            "into swap.")
    return p


def detect(device: str, requested_quant: str = None,
           requested_chunk: int = None) -> Profile:
    """Build the profile for whatever this machine turns out to be.

    Falls back to conservative settings whenever it cannot tell, because the
    cost of guessing small is some wasted speed and the cost of guessing big
    is a crash or an hour of swapping.
    """
    if device == "cuda":
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / 1e9
            try:
                bf16 = torch.cuda.is_bf16_supported()
            except Exception:
                bf16 = props.major >= 8
            p = _cuda_profile(vram, bf16, (props.major, props.minor))
            p.notes.insert(0, f"{props.name}, {vram:.0f}GB")
        except Exception as exc:
            p = Profile(name="cuda-unknown", dtype="float16", sample_chunk=2,
                        attn="sdpa", quant="8bit",
                        notes=[f"Could not read the GPU ({exc}); assuming the "
                               f"cautious case."])
    elif device == "mps":
        try:
            import psutil
            total = psutil.virtual_memory().total / 1e9
        except Exception:
            try:
                import os
                total = (os.sysconf("SC_PAGE_SIZE")
                         * os.sysconf("SC_PHYS_PAGES")) / 1e9
            except Exception:
                total = 16.0
        p = _mps_profile(total)
        p.notes.insert(0, f"Apple Silicon, {total:.0f}GB unified")
    else:
        p = Profile(name="cpu", dtype="float32", sample_chunk=1,
                    notes=["CPU inference is minutes per sample. Fine for "
                           "checking the plumbing, not for a session."])

    if requested_quant:
        p.quant = requested_quant
    if requested_chunk:
        p.sample_chunk = requested_chunk
    return p


def describe(p: Profile) -> str:
    lines = [f"Profile: {p.name}",
             f"  dtype          {p.dtype}",
             f"  quantisation   {p.quant}",
             f"  samples/call   {p.sample_chunk}",
             f"  attention      {p.attn or 'default'}",
             f"  max tiles      {p.max_tiles or 'processor default'}"]
    for n in p.notes:
        lines.append(f"  - {n}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    print(describe(detect(dev)))
