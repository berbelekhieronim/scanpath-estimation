# Scanpath Estimation Demo — Product & Technical Spec

**Status:** Draft for approval · **Owner:** bszperlinski1@st.swps.edu.pl
**Repo:** `berbelekhieronim/scanpath-estimation` · **Branch:** `claude/eye-gaze-annotation-app-fvoz8p`

---

## 1. What this is

A live-audience demo. People in a room open a URL on their phone, look at a
photo, and tap where they think they would look. Their collective guesses are
displayed on a screen. Then the same photo is run through a state-of-the-art
scanpath model, the model's predicted eye-movement path is overlaid, and the two
are compared numerically.

**The point being made:** a machine can predict human visual attention about as
well as humans can predict it about themselves.

### 1.1 One honest caveat that shapes the whole design

Taps are not fixations. What a participant *deliberately taps* is a slow,
top-down, reportable judgement. What their eyes *actually do* in the first 3
seconds is fast, largely bottom-up, and strongly centre-biased. These diverge in
known, replicable ways — real first fixations cluster near the image centre;
taps do not.

This matters because the demo narrative must survive a mediocre correlation. If
human taps and model output disagree, the defensible reading is **not** "the
model is wrong." The model was trained on real eye-tracking data (MIT, CAT2000,
COCO, Daemons, FIGRIM); the taps were not measured, they were guessed. A
disagreement is evidence that **people are poor introspectors of their own
gaze** — which is a better talk than "look, they match."

Design consequence: the UI should call the human layer *"where you predicted
you'd look"*, never *"where you looked"*, and the analysis panel should report
agreement without implying either side is ground truth. Section 7 specifies the
metrics accordingly.

---

## 2. The model: what it can and cannot do

Findings from reading [`Susmit-A/DeepGaze3.5-VL`](https://github.com/Susmit-A/DeepGaze3.5-VL)
(ECCV 2026, [arXiv:2607.02083](https://arxiv.org/abs/2607.02083)) directly.

| Property | Value |
|---|---|
| Architecture | LoRA adapter on `OpenGVLab/InternVL3_5-8B-HF` (8B vision-language model) |
| Free-viewing adapter | rank 32, trained on MIT + CAT + COCO + Daemons + FIGRIM |
| Visual-search adapter | rank 8, trained on COCO-Search18 |
| Output | Ordered `(x, y)` fixations on a 0–99 integer grid |
| Runtime | vLLM; **1 GPU required**, ~16 GB VRAM for bf16 8B |
| Base weights | Not bundled — pulled from HuggingFace on first run (~16 GB download) |
| First-run cost | Downloads base model, merges LoRA into `model/combined_adapter_merged/` |
| Licence | **Non-commercial research use only** (adapter + code); base model under its own upstream licence; bundled MIT/CAT2000 sample images under original dataset terms |

### 2.1 Text prompts — the answer is "yes, but within a trained envelope"

You asked whether this supports text prompts rather than a binary on/off. It
does, with an important qualification.

The interface is genuinely textual. `predict_scanpath.py` builds a plain-string
prompt and passes it to the VLM — there is no fixed API of flags underneath, it
really is language in, coordinates out. The paper's own framing is that prompt
modifications provide global conditioning (viewer identity, task).

**But** the two prompt templates in `predict_scanpath.py` are, per the source
comment, *"copied VERBATIM from the training data."* The LoRA was fine-tuned on
those exact strings. Free-text prompts outside that distribution will still
produce coordinates — the model always emits *something* — but quality is
unvalidated and silently degrades. There is no error, just worse predictions.

So the conditioning axes worth exposing in the UI, in descending order of
confidence:

| Axis | Status | Exposed in UI |
|---|---|---|
| Task: free-viewing vs. visual search | Trained, separate adapters | **Yes** — primary toggle |
| Search target (18 COCO-Search18 categories) | Trained | **Yes** — dropdown, fixed list |
| Number of fixations (`--num-fixations`) | Templated into the prompt; trained across lengths | **Yes** — slider, default 8 |
| Temperature + seed | Standard sampling | **Yes** — used to generate N "virtual observers" |
| Arbitrary free-text task prompt | Off-distribution, unvalidated | **Advanced panel, clearly labelled experimental** |

The 18 trained search targets: bottle, bowl, car, chair, clock, cup, fork,
keyboard, knife, laptop, microwave, mouse, oven, potted plant, sink, stop sign,
toilet, tv. The script warns on stdout for anything else but proceeds anyway.

**Demo recommendation.** The free-viewing vs. search contrast is the strongest
material you have and it is fully supported. On your attached street photo, run
free-viewing, then re-run with `--mode search --target car` and show the path
reorganise toward the vehicle. That is a vivid, legitimate demonstration of
task-driven attention, and it needs no off-distribution prompting.

The free-text box should exist — it is the more interesting research direction —
but it must be marked experimental so a bad output in front of an audience reads
as "we are probing the edges," not "the demo broke."

### 2.2 Your stimulus is well chosen

The attached photo contains a yellow warning triangle with a **capybara** on it.
That is a semantically incongruent object in a mundane street scene — the
classic scene-violation manipulation, and objects like it reliably attract early
fixations. If the model's free-viewing path goes to the sign, the demo lands. It
is worth having a second, "boring" control image with no violation so the
contrast is visible.

---

## 3. The constraint that decides the architecture

**The model needs a GPU. Your FTP host does not have one, and neither does a
standard GitHub Codespace.**

This is not a detail to engineer around — it is the fork in the road:

- FTP shared hosting: PHP, no GPU, no long-running process. Cannot run the model. Ever.
- GitHub Codespaces: CPU-only on all standard machine types. Cannot run an 8B VLM at usable speed. GPU Codespaces exist but are not in the default offering.
- vLLM engine startup is minutes, not milliseconds. Even *with* a GPU, this can never be a cold-start-per-request web endpoint. It needs a warm, persistent process.

Therefore the system **must** split into two independently deployable pieces:

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  WEB APP  (no GPU)          │        │  MODEL RUNNER  (GPU)         │
│  ─────────────────          │        │  ──────────────              │
│  • participant capture      │◄──────►│  • vLLM + InternVL3.5-8B     │
│  • presenter display        │  JSON  │  • LoRA adapters             │
│  • storage + analysis       │        │  • emits scanpath JSON       │
│  Runs anywhere. Tiny.       │        │  Runs where a GPU is.        │
└─────────────────────────────┘        └──────────────────────────────┘
```

The web app never imports torch. The model runner never serves participants.
They meet at a JSON contract (§6.3). This keeps the participant-facing path —
the non-negotiable part — completely immune to GPU availability, driver
problems, CUDA OOM, and HuggingFace being slow.

---

## 4. Recommended build

### 4.1 Web app: PHP + SQLite, served from your FTP host

**Recommendation.** Nearly every FTP-based shared host runs PHP with SQLite
compiled in. That gives you: no build step, no Node process to keep alive, no
container, drag-and-drop deployment over the FTP client you already use, and a
stable URL you control. For a room of 30 people tapping an image, this is
comfortably sufficient and has the fewest moving parts of any option.

Frontend is a single static HTML page per view with vanilla JS and inline SVG
overlays — no framework, no bundler, no npm. Total payload target under 50 KB
excluding images. This is the fastest thing to make work correctly on an
unpredictable mix of phones.

*Prerequisite to confirm before building:* that your host has PHP ≥ 8.0 with
`pdo_sqlite`, and a writable directory outside the web root. A three-line
`phpinfo()` upload settles it. **If the host is static-only (no PHP), fall back
to Option B.**

**Option B — Node/Express + SQLite on a small always-on host** (Fly.io, Render,
Railway; free-to-cheap tiers). Better local development story, needs a real
deploy pipeline. Choose this if PHP is unavailable or if you would rather work
in JS.

**On Codespaces:** excellent for *developing* this, poor for *hosting* the live
session. A forwarded port gives a long URL, requires the port be set public, and
sleeps after idle timeout. Develop there, deploy to the FTP host.

### 4.2 Model runner: precompute, with a live path available

Three tiers. **Tier A is the recommendation for the live session.**

**Tier A — Precompute (recommended).** Before the session, run every image
through the model on a GPU machine for each configuration you plan to show
(free-viewing, plus each search target, plus N seeds for virtual observers).
Commit the resulting JSON to `data/model/`. The web app reads static JSON and
renders instantly.

This is worth being precise about, because it touches your "or I could fake it"
remark: **precomputing is not faking.** It is the identical model, identical
weights, identical prompt, producing identical output — just computed on
Tuesday rather than during the talk. The only thing you lose is the theatre of
the progress bar. What you gain is a demo that cannot fail in front of an
audience because of a CUDA OOM or a slow download. Say out loud that the
scanpaths were generated ahead of time and nothing about the claim weakens.

The genuinely dishonest version would be hand-drawing plausible paths. Don't do
that. Run the real model; run it early.

**Tier B — Live inference service.** A ~100-line FastAPI wrapper holding the
vLLM engine warm, exposed over HTTPS with a shared-secret header, called by the
web app's "Run model" button. Genuinely live. Requires an always-on GPU: your
own card with ≥16 GB VRAM, or a rented cloud GPU at roughly $0.20–0.50/hour.
Build this *after* Tier A works, as an upgrade, never as the only path.

**Tier C — Laptop-in-the-loop.** You run `predict_scanpath.py` from your
terminal during the talk; a small `push_result.py` helper POSTs the JSON to the
web app, which is polling and renders it within two seconds. This is your
"run it from my terminal and show them" idea, and it is fully legitimate — the
inference is real and happening live. It gives you the theatre of Tier B with
the hardware requirements of a laptop with a decent GPU, and it degrades
gracefully: if it fails, the Tier A precomputed path is already sitting there.

**Ship Tier A and Tier C. Treat Tier B as a later upgrade.**

---

## 5. Views and user flows

Four URLs. Participant-facing ones stay dead simple.

### 5.1 `/` — Participant capture

The non-negotiable path. Must be flawless on a phone.

1. Lands on the currently active image, scaled to fit the viewport, no page scroll.
2. Prompt text above: *"Where do you think your eyes would go first? Tap 5 places, in order."*
3. Each tap drops a numbered marker. Undo removes the last one. Markers are ordered — sequence is the data.
4. Tapping is disabled after the target count, with a "Submit" button becoming active.
5. Submit posts and shows a "thanks — watch the screen" state. No account, no email, no personal data.
6. Anonymous `participant_id` (UUID v4) in `localStorage`, so a returning device is recognised without identifying a person.

Requirements: touch and mouse both work; markers land accurately on high-DPI
screens; works in portrait and landscape; coordinates stored **normalised to
0.0–1.0** against the image's own box so every display size renders identically;
no zoom-on-double-tap interfering with marker placement.

### 5.2 `/display` — Presenter screen

Projected. Read-only, polls every 2 s. Layers, independently toggleable:

| Layer | Content |
|---|---|
| 0 — Image | The stimulus, letterboxed |
| 1 — Human taps | Aggregate. Heatmap (Gaussian-blurred KDE) plus, optionally, per-participant numbered paths |
| 2 — Model scanpath | Numbered fixations, connected path, animated draw-on |
| 3 — Analysis | Agreement metrics panel (§7) |

Live participant counter ("17 responses") so you know when to move on.

### 5.3 `/control` — Presenter controls

Same data, plus the buttons. **Token-protected** (`?k=<secret>` or a cookie) so
a participant who guesses the URL cannot wipe your data mid-session.

- **Clear** — per-layer toggles. "Clear" hides a layer; it never deletes rows. Your flow is *show humans → clear → show model → clear → next image*, and that is layer visibility, not destruction. Deletion is a separate, confirmed, explicitly-labelled action.
- **Image select** — dropdown of all images in the asset folder; sets the active image, which participant devices pick up on their next poll.
- **Run model** — plays the precomputed scanpath (Tier A), or triggers live inference (Tier B/C).
- **Model config** — mode (free-viewing / search), target dropdown, fixation count, number of virtual observers.
- **Reset round** — archives current responses and opens a fresh round for the same image, so you can run the same stimulus with a second group.

### 5.4 `/admin` — Image management

Upload or rescan the assets folder, view thumbnails, set display order, see
which images have precomputed model runs and which don't.

---

## 6. Technical design

### 6.1 Repository layout

```
scanpath-estimation/
├── docs/
│   └── SPEC.md                  # this file
├── public/                      # web root — this is what goes on FTP
│   ├── index.php                # participant capture
│   ├── display.php              # presenter screen
│   ├── control.php              # presenter controls (token-gated)
│   ├── admin.php                # image management
│   ├── api/
│   │   ├── markers.php          # POST tap data, GET aggregate
│   │   ├── state.php            # GET/POST active image + layer state
│   │   ├── model.php            # GET precomputed run, POST pushed result
│   │   └── analysis.php         # GET agreement metrics
│   ├── assets/
│   │   ├── app.css
│   │   ├── capture.js
│   │   └── display.js
│   └── images/                  # stimulus images (your assets folder)
├── data/                        # NOT web-accessible
│   ├── app.sqlite
│   └── model/                   # precomputed scanpaths, one JSON per run
├── tools/
│   ├── precompute.py            # batch-run the model over all images
│   ├── push_result.py           # Tier C: POST a local run to the web app
│   └── analysis.py              # reference implementation of §7 metrics
├── .gitignore
└── README.md
```

`data/` must sit outside the web root, or be protected by `.htaccess` — the
SQLite file must never be downloadable.

### 6.2 Data model

```sql
images(id, filename, label, width, height, sort_order, active)
rounds(id, image_id, opened_at, closed_at, label)
participants(id, uuid, first_seen, user_agent)
markers(id, round_id, participant_id, seq, x, y, created_at)
   -- x, y are REAL in 0.0–1.0, normalised to the image box
model_runs(id, image_id, mode, target, n_fixations, seed, temperature,
           prompt_text, coords_json, source, created_at)
   -- source: 'precomputed' | 'live' | 'pushed'
app_state(key, value)   -- active_image_id, active_round_id, layer visibility
```

Normalised coordinates are load-bearing: a phone at 390 px wide and a projector
at 1920 px must produce and render comparable data.

### 6.3 The web ↔ model JSON contract

One shape, whether precomputed, pushed from a laptop, or returned live:

```json
{
  "image": "1.jpg",
  "mode": "freeview",
  "target": null,
  "n_fixations": 8,
  "seed": 42,
  "temperature": 0.0,
  "prompt_text": "Analyze this image and predict a human eye movement scanpath...",
  "scanpath_grid": [[51,46],[38,28],[72,33]],
  "scanpath_norm": [[0.51,0.46],[0.38,0.28],[0.72,0.33]],
  "source": "precomputed",
  "model": "InternVL3_5-8B-HF + combined_adapter r32",
  "created_at": "2026-09-19T14:03:00Z"
}
```

`scanpath_grid` is the model's native 0–99 output, preserved verbatim for
provenance. `scanpath_norm` is the 0.0–1.0 form the frontend renders and the
analysis consumes. Storing both means a rendering bug can never be mistaken for
a model bug.

### 6.4 Multiple virtual observers

A single greedy decode (`--temperature 0.0`) gives one path — visually thin next
to 30 human responses, and it invites the objection that you cherry-picked a
seed. Generating 10–20 samples at `temperature ≈ 0.7` with different seeds
yields a *distribution* of model scanpaths, which can be rendered as a heatmap
directly comparable to the human heatmap, and compared distribution-to-
distribution rather than path-to-crowd. This is a better comparison and a better
visual. `precompute.py` should do this by default.

---

## 7. Agreement analysis

Reported in the Layer 3 panel. Each metric answers a different question, and
none of them treats either side as truth.

**Spatial overlap — do they attend to the same regions?**
- Divide the image into an AOI grid (5 × 5 default, configurable).
- Human vector: proportion of taps per cell. Model vector: proportion of model fixations per cell, pooled across virtual observers.
- **Spearman ρ** between the two vectors — robust, interpretable, the headline number.
- **Normalised Scanpath Saliency (NSS)** of human tap locations against the model's fixation density map. Standard in the literature, so it is the comparable figure.

**Sequence similarity — do they go in the same order?**
- Encode each path as a string of AOI cell labels, compare with **Levenshtein distance**, normalised by length.
- **ScanMatch** (Cristino et al., 2010) if you want the established method — substitution matrix weighted by inter-cell distance.

**Baselines — is the agreement meaningful?** This is the part that makes the
analysis credible rather than decorative. Report alongside:
- **Random baseline:** uniformly sampled points, same count.
- **Centre-bias baseline:** a 2-D Gaussian at image centre. This one matters — centre bias alone explains a surprising amount of agreement in any fixation data, and a model that only beat *random* would not be impressive.
- **Human-to-human:** split participants into two halves, score one against the other. This is the ceiling. If model-vs-human approaches human-vs-human, the claim "the machine does this as well as you do" is quantitatively supported — and that sentence is your whole talk.

The human-to-human ceiling is the most valuable number on the screen. Compute it.

---

## 8. Build order

Each phase ends somewhere demonstrable.

| Phase | Deliverable | Gate |
|---|---|---|
| **1** | Repo scaffold, SQLite schema, image asset loader, `/admin` | Images visible in a list |
| **2** | `/` capture view — tap, order, undo, submit. Phone + desktop | **A phone can post markers.** The non-negotiable path works |
| **3** | `/display` + `/control` — heatmap, ordered paths, layer toggles, image switch, live poll | Full human half of the demo runs end to end |
| **4** | `tools/precompute.py` + model layer rendering | Model scanpaths overlay on the image |
| **5** | `tools/analysis.py` + Layer 3 panel with baselines | Numbers on screen, including the human-to-human ceiling |
| **6** | `tools/push_result.py` (Tier C live path) | Laptop inference appears on the projector |
| **7** | Optional: Tier B live service; free-text prompt panel | — |

Phases 1–3 need no GPU at all and deliver a working participant experience.
That ordering is deliberate: the part that cannot be faked or deferred gets
built first.

---

## 9. Operational notes

**Before a session**
- Precompute every image × config you might show. Verify each JSON renders.
- Confirm the participant URL works on the venue's guest wifi, from a phone, not just your laptop.
- Have a QR code to the participant URL on a slide.
- Short URL if possible; people mistype.

**Risks**
| Risk | Mitigation |
|---|---|
| Venue wifi blocks or throttles | Precomputed data is local to the server; capture degrades but display still works. Have screenshots as a final fallback |
| GPU unavailable on the day | Tier A means the GPU is never needed on the day |
| Very low participation | Seed with a couple of your own responses so the heatmap is not empty; human-to-human ceiling needs ≥ 6 responses to mean anything |
| Poor human/model agreement | Reframe per §1.1 — it is a finding, not a failure. Prepare that line in advance |
| Someone taps the control URL | Token-gate it |

**Licensing.** The model is non-commercial research use only — fine for an
academic demonstration, and worth a footnote on your slide. Do **not** commit
the bundled MIT/CAT2000 sample images into this repo; they carry the original
dataset terms. Use your own photographs as stimuli, which you appear to be
doing already.

**Privacy.** No personal data is collected: an anonymous UUID and tap
coordinates only. Worth one sentence on the participant screen — it costs
nothing and pre-empts the question. If this is ever run as actual research
rather than a demonstration, ethics approval and a consent screen apply.

---

## 10. Open decisions

Needed before Phase 1:

1. **Does your FTP host run PHP ≥ 8 with `pdo_sqlite`?** Decides web stack (§4.1). A `phpinfo()` upload answers it.
2. **Do you have a GPU with ≥ 16 GB VRAM**, on your laptop or otherwise? Decides whether Tier C is available, and how precomputation gets run.
3. **How many taps per participant?** Spec assumes 5. The model defaults to 8 fixations for free-viewing; matching the counts makes sequence comparison cleaner.
4. **How many stimulus images** in a session? Affects precompute time.
5. **Should participants see the results on their own phone** after submitting, or only on the projected screen? Spec currently assumes projector-only.
