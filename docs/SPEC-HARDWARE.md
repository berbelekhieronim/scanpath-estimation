# Two machines, one pipeline

The model runs on an M4 Air and on an RTX 5070 Ti. They fail in opposite
directions, so a single set of defaults serves neither.

| | M4 Air, 32GB unified | RTX 5070 Ti, 16GB VRAM |
|---|---|---|
| Weights (8B, 16-bit) | ~16GB — fits | ~16GB — **fills the card** |
| When memory runs out | **swaps**, silently | **raises**, immediately |
| Failure looks like | the model being slow | a stack trace |
| Fix | keep the working set resident | quantise |

The second row is the whole story. A card that is out of memory tells you so.
Unified memory does not: it pages to disk and the run simply takes an hour
and a half, which is exactly what happened on the first real run.

## What is shared and what is not

**The inference logic is not forked.** The prompt strings and the coordinate
parsing determine output quality, and two copies would drift — one machine
would quietly start producing different scanpaths from the other, and every
comparison between sessions would be worthless. Prompts are already pinned
byte-for-byte against upstream and verified by `gaze_prompts.py --verify`.

What differs between machines is a handful of numbers, and those live in
`tools/backends.py`. Ask any machine what it will do:

```bash
python tools/predict.py --profile
```

```
Profile: cuda-16gb
  dtype          bfloat16
  quantisation   8bit
  samples/call   4
  attention      sdpa
  max tiles      6
  - NVIDIA GeForce RTX 5070 Ti, 16GB
  - 16GB cannot hold 16GB of 16-bit weights with anything left for
    activations, so the model is loaded in 8-bit (about 8GB).
  - Blackwell (sm_120) needs PyTorch 2.7 or newer built against CUDA 12.8.
```

## The change that matters on both

Samples used to be one `generate()` call with `num_return_sequences=n`, and
the docstring claimed that paid the image prefill once. It does not pay it
once in *memory*: transformers expands the batch before prefill, so the KV
cache is multiplied by the sample count. Ten samples meant 16GB of weights
plus ten caches over thousands of vision tokens.

Samples now run in chunks sized to the machine. Each chunk pays its own
prefill — that is the cost — and in exchange the working set stays inside
the machine, which on both of these is worth far more than the prefill.

Two details that are easy to get wrong and were tested rather than assumed:

- **Each chunk gets its own seed** (`seed + chunk_index`). One seed across
  every chunk returns the same scanpath each time, collapsing ten observers
  into one while still producing ten entries that look like samples.
- **The cache is released between chunks**, or chunking buys nothing.

## Setup

**Mac** — `pip install -r requirements-mac.txt`.

**NVIDIA** — install torch from the right index *first*, then the rest:

```bash
# Blackwell (RTX 50-series) needs 2.7+ on CUDA 12.8. An older wheel has no
# kernels for sm_120 and fails at the first matmul.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-cuda.txt
```

`requirements-cuda.txt` includes bitsandbytes, which the 16GB profile needs
for its 8-bit load. Without it the run stops with an explanation rather than
an out-of-memory.

## How to use the two of them

**Mac, any time.** Small runs while iterating: one image, a handful of
samples. It is memory-comfortable and needs no quantisation, so its output is
the reference — full-precision weights, nothing approximated.

**5070 Ti, for a heavy session.** Sweeps from `SPEC-EXPERIMENTS.md`: many
images, many probes, many samples. Far faster per token, at the cost of
8-bit weights.

**One caveat about mixing them.** An 8-bit model is not bit-identical to a
16-bit one. For most of this that is irrelevant — the outputs are sampled at
temperature 0.7 anyway, so run-to-run variation already dwarfs quantisation
noise. But do not compare a Mac run against a CUDA run and attribute a
difference to anything but the hardware, unless the CUDA run was made with
`--quant none` on a card big enough to allow it. Every run records its
`profile`, `dtype` and `quant` in the output JSON so this is checkable after
the fact rather than remembered.

## Still unmeasured

Whether `num_return_sequences` repeats the *compute* of the prefill, as
opposed to its memory, is not settled — see `SPEC-EXPERIMENTS.md`. Chunking
is justified by the memory argument alone, which is certain. Run
`tools/bench_model.py` on each machine before trusting any timing estimate;
nothing in this document was measured on either, because neither is here.
