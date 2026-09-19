# scanpath-estimation

Demonstration of scanpath estimation.

A live-audience demo: participants open a URL on their phone and tap where they
think their eyes would go on a photo. Their collective guesses are displayed,
then compared against a state-of-the-art scanpath model
([DeepGaze3.5-VL](https://github.com/Susmit-A/DeepGaze3.5-VL), ECCV 2026).

**Status:** specification approved (2026-09-19), implementation not started.
Build proceeds from Phase 1 of the spec's build order.

## Start here

[`docs/SPEC.md`](docs/SPEC.md) — product and technical spec, including the
model's real capabilities and constraints, the architecture, and a phased
build order.

## Layout

| Path | Purpose |
|---|---|
| `docs/` | Specification |
| `public/` | Web app — this is what gets deployed |
| `data/` | SQLite database and precomputed model runs (not web-accessible) |
| `tools/` | Model precomputation, result push, analysis |

## Licence note

The scanpath model is released for **non-commercial research use only**, and its
base model (`OpenGVLab/InternVL3_5-8B-HF`) carries its own upstream licence.
