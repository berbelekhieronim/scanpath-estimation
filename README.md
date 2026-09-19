# scanpath-estimation

Demonstration of scanpath estimation.

A live-audience demo: participants scan a QR code, open a page on their phone,
and tap where they think their eyes would go on a photo. Their collective
guesses are displayed on a presenter screen, then compared against a
state-of-the-art scanpath model
([DeepGaze3.5-VL](https://github.com/Susmit-A/DeepGaze3.5-VL), ECCV 2026).

**Status:** Phases 1–4 complete. Participants join by QR and tap; the presenter
screen shows the aggregate live and can overlay the model's predicted scanpath.

**The inference path has not been run against the real model** — there is no GPU
in the environment it was written in. See *Running the model* below.

## Start here

[`docs/SPEC.md`](docs/SPEC.md) — product and technical spec: the model's real
capabilities and constraints, the architecture, the analysis method, and a
phased build order.

## Architecture in one line

A FastAPI + SQLite web app runs in a GitHub Codespace and hosts the live
session; the 8B scanpath model runs separately on an M4 MacBook and pushes
results in as JSON. The web app never imports torch.

## Layout

| Path | Purpose |
|---|---|
| `docs/` | Specification |
| `app/` | FastAPI web app — participant capture, presenter display, analysis |
| `data/` | SQLite database, stimulus images, precomputed model runs, session exports |
| `tools/` | Model inference on Apple Silicon, precomputation, result push, export |
| `.devcontainer/` | Codespace definition |

Two requirements files, deliberately: `requirements.txt` is the light web app
and is what the Codespace installs; `requirements-model.txt` carries
torch/transformers/peft and is only ever installed on the MacBook.

## Licence note

The scanpath model is released for **non-commercial research use only**, and its
base model (`OpenGVLab/InternVL3_5-8B-HF`) carries its own upstream licence.

## Running it

In a Codespace (or locally with Python 3.12):

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The console prints the control token and the URLs on startup. Put stimulus
images in `data/images/` — they are picked up at boot, or via **Rescan folder**
in `/admin`.

| Route | Who | Notes |
|---|---|---|
| `/` | Participants | Capture view — tap in order, undo, submit |
| `/qr` | Projector | Full-screen QR code to join |
| `/display` | Projector | Heatmap, individual paths, live response count |
| `/control?k=TOKEN` | Presenter | Layer toggles, image select, fresh round |
| `/admin?k=TOKEN` | Presenter | Image management |

`SCANPATH_CONTROL_TOKEN` pins the token; otherwise one is generated on first run
and persisted, so it stays stable across restarts.

**In a Codespace, port 8000 must be set to Public** or participants hit a GitHub
login. Verify it in the Ports panel before every session — see §4.3 of the spec.

Open `/qr` on the projector for people to join. It shows the join URL as text
too, and warns in red if that URL is one only this machine can reach — which is
what a misconfigured port looks like before anyone tries to scan it.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Running a session

1. Open `/control?k=TOKEN` on your laptop and `/display` on the projector.
2. Put `/qr` up while people join — the response count rises on both screens.
3. When the room has responded, reveal the layers in order: heatmap first,
   then individual paths.
4. **Clear** is a layer toggle, not a delete. Responses stay in the database
   and reappear if you switch the layer back on.
5. To run the same image with a second group, use **Start a fresh round** —
   the earlier responses are kept and stay exportable.

## Running the model

The model runs on your MacBook, not in the Codespace. One-time setup:

```bash
git clone https://github.com/Susmit-A/DeepGaze3.5-VL   # needs git-lfs
python3 -m venv .venv-model && source .venv-model/bin/activate
pip install -r requirements-model.txt
```

Confirm the prompt templates still match upstream (they are reproduced in
`tools/gaze_prompts.py` and must stay byte-identical, since the adapter was
fine-tuned on those exact strings):

```bash
python tools/gaze_prompts.py --verify ../DeepGaze3.5-VL
```

Single image:

```bash
python tools/predict_mps.py --repo ../DeepGaze3.5-VL \
    --image data/images/street.jpg --mode freeview --num-fixations 5
```

Precompute everything before a session (the recommended path):

```bash
python tools/precompute.py --repo ../DeepGaze3.5-VL \
    --modes freeview --num-fixations 5 --samples 10 --temperature 0.7
```

Results land in `data/model/` as JSON. Commit them, and the app loads them at
boot or via **Reload runs from disk** in `/control`.

First run downloads ~16GB of base model. Expect 1–3 minutes per image-config
on MPS, or 10–20 on CPU. Slow is fine — precomputing means nothing waits on it
during the talk.

### Synthetic placeholder data

`tools/precompute.py --synthetic` generates placeholder scanpaths with no model
at all, so the display can be developed without a GPU. Its output is stamped
`source: "synthetic"`, and the display shows a full-width red warning banner
whenever it renders one. **Never present synthetic output as a model
prediction** — delete `data/model/*.json` and regenerate before a real session.
