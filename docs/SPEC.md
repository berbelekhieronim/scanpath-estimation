# Scanpath Estimation Demo — Product & Technical Spec

**Status:** Approved, 2026-09-19 · **Owner:** bszperlinski1@st.swps.edu.pl
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
- Your M4 MacBook has the memory but not CUDA, so the repo's own instructions do not run on it either (§4.2).
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

### 4.1 Web app: PHP, with a storage layer that works either way

**Recommendation.** Zenbox runs PHP, which is the only thing that must be true.
Frontend is a single static HTML page per view with vanilla JS and inline SVG
overlays — no framework, no bundler, no npm, no Node process to keep alive.
Deployment is a drag-and-drop over the FTP client you already use. Total payload
under 50 KB excluding images. For a room of 30 people tapping an image this is
comfortably sufficient and has the fewest moving parts of any option.

**The SQLite question is unblocked, not answered.** Zenbox's PHP module list
isn't publicly documented, so rather than block Phase 1 on finding out, storage
goes behind a thin `Storage` interface with two implementations:

- `SqliteStorage` — zero setup, a single file, preferred if `pdo_sqlite` is present.
- `MysqlStorage` — same schema, used if only `pdo_mysql` is available. Every
  shared host that offers PHP offers MySQL/MariaDB, so this is the guaranteed
  fallback.

Both speak the same ~8 queries. The choice is one config line, decided once when
you know which extension exists. This costs maybe an extra hour of work and
removes an unknown from the critical path entirely.

**To find out which:** upload `tools/check_host.php` to your Zenbox web root,
open it in a browser, and it reports PHP version, which PDO drivers are
available, whether the data directory is writable, and the upload limits. Delete
it afterwards — it reveals server configuration.

**Option B — Node/Express on a small always-on host** (Fly.io, Render, Railway)
remains the alternative if you would rather work in JS, but with the storage
layer abstracted there is no longer a reason to reach for it.

**On Codespaces:** excellent for *developing* this, poor for *hosting* the live
session — a forwarded port gives a long URL, must be set public, and sleeps after
idle timeout. Develop there, deploy to Zenbox.

### 4.2 Model runner on an M4 MacBook: bypass vLLM

A 32 GB M4 MacBook is enough memory to run this, but **not by following the
repo's instructions**. Two things are in the way:

1. **vLLM's standard install requires CUDA**, which does not exist on Apple
   Silicon. Running it on a Mac means either the community
   [`vllm-metal`](https://github.com/vllm-project/vllm-metal) plugin (MLX
   backend) or an experimental CPU build from source — neither with any
   guarantee of supporting InternVL3.5 multimodal input plus a merged LoRA.
2. `requirements.txt` pins `vllm==0.11.2` and `torch==2.9.0`, a CUDA-shaped
   dependency set.

**The way through: don't use vLLM at all.** vLLM is doing exactly one job here —
fast batched serving — and this workload is one image at a time producing ~60
tokens of output. None of that speed is needed.

Crucially, the repo makes this easy in a way worth pointing out:

- `FewShotPromptBuilder.build_prompt()` is ~15 lines and touches no vLLM code —
  it builds a standard `messages` list and calls `processor.apply_chat_template()`.
- `parse_scanpath_reduced()` is a regex over `(x, y)` tuples.
- **Every `vllm` import in `evaluate_vllm_unified.py` is lazy** (inside function
  bodies, never at module level).

So the module can be imported on a Mac with no vLLM installed, and the exact
prompt-construction and parsing code can be reused verbatim. Only the engine
gets swapped:

```
torch (MPS) + transformers + peft
  ├─ AutoProcessor.from_pretrained("OpenGVLab/InternVL3_5-8B-HF")
  ├─ AutoModelForImageTextToText.from_pretrained(..., dtype=torch.bfloat16)
  ├─ PeftModel.from_pretrained(model, "model/combined_adapter").merge_and_unload()
  ├─ FewShotPromptBuilder(processor).build_prompt(...)   ← reused as-is
  ├─ model.generate(..., max_new_tokens=96)
  └─ parse_scanpath_reduced(text)                        ← reused as-is
```

This becomes `tools/predict_mps.py`. It is a genuinely small script, and because
it reuses their prompt builder it produces the same prompt string the model was
trained on — which is the part that actually determines output quality.

**Memory.** 8B at bf16 ≈ 16 GB of weights. 32 GB unified memory fits that with
room for activations, but InternVL tiles images into up to ~3000 vision tokens,
so the prefill is the peak. If it OOMs, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
lifts the allocator cap. Expect roughly **1–3 minutes per image**, dominated by
prefill, not decode.

**The optimisation that makes precompute practical.** The virtual-observer
samples (§6.4) all share one image and one prompt, so the expensive prefill is
identical across them. Generating them in a single `generate()` call with
`num_return_sequences=N` pays the prefill **once**. Ten virtual observers then
cost barely more than one. A realistic precompute of 5 images × 3 configs is
~15 prefills, so **half an hour on your MacBook**, not an afternoon.

**Slow is acceptable** (owner's constraint, confirmed). This materially
changes the plan for the better: the 1–3 minute runtime is no longer a reason to
avoid live inference, and it means there is **no scenario in which the model
cannot be run at all** — if MPS fails, plain CPU on 32 GB of RAM still executes
the same script, just at perhaps 10–20 minutes per image. Slow beats absent, and
precompute makes runtime invisible at demo time anyway.

**Tier recommendation:**

- **Tier A — Precompute on the MacBook (primary).** Run `tools/precompute.py` ahead of the session, commit the JSON to `data/model/`, web app serves it instantly. No GPU present at demo time, nothing to fail in front of an audience. This is not faking: same weights, same prompt, same output, computed Tuesday instead of during the talk. The dishonest version would be hand-drawing plausible paths — don't do that, run the real model early.
- **Tier C — Laptop-in-the-loop (now a first-class demo element).** `tools/push_result.py` POSTs a local run to the web app, which is polling and renders it within two seconds. Genuinely live inference from your terminal, on a new image the audience picks. The 1–3 minute wait is workable if you narrate over it — it is arguably *better* theatre than an instant result, because the audience watches a real model think. Keep the Tier A result loaded as the silent fallback.
- **Tier B — Always-on GPU service.** Not needed. Only worth building if this later becomes something people use unattended.

**Fallback ladder**, in order, if the MPS path misbehaves:

1. `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to lift the allocator cap.
2. `device_map="cpu"` — slow but certain, and 32 GB is sufficient.
3. A rented 24 GB cloud GPU (RunPod, Vast.ai) for one hour, running the repo's instructions unmodified, for under a euro.

**First task of Phase 4** is validating `predict_mps.py` end to end on your
machine, because this is the one part of the plan that cannot be verified from
here. Everything upstream of it is independent of the outcome.

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
│   ├── lib/
│   │   ├── storage.php          # Storage interface
│   │   ├── storage_sqlite.php   # pdo_sqlite implementation
│   │   └── storage_mysql.php    # pdo_mysql fallback
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
│   ├── check_host.php           # one-off Zenbox capability probe (delete after use)
│   ├── predict_mps.py           # single-image inference on Apple Silicon (§4.2)
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

## 7. AOI definition and agreement analysis

With 20+ participants × 5 taps each you have ~100 human points per image, which
is enough to derive areas of interest from the data rather than imposing a grid.
This section replaces the fixed-grid approach.

### 7.1 The circularity trap — read this before choosing a method

If AOIs are derived by clustering human taps, and you then measure *"do humans
and the model hit the same AOIs"*, **the humans score near-perfectly by
construction**. The AOIs were drawn around their own points. Any comparison on
that basis is rigged in the humans' favour and the resulting number means
nothing.

This is easy to miss and would quietly invalidate the headline claim, so AOIs
must come from a source that does not privilege either side. Three valid
options:

| Method | How | Use for |
|---|---|---|
| **A. Semantic** | Hand-drawn boxes around objects: the capybara sign, the car, the pedestrian, the crossing, the traffic light | **The narrative.** Named AOIs on screen, independent of both sides, most legible to an audience |
| **B. Pooled clustering** | Mean-shift over human taps ∪ model fixations together | **The data-driven number.** Neither side privileged |
| **C. Split-half** | Derive AOIs from a random half of participants; score the other half *and* the model against them | **The rigorous version.** Yields the human-to-human ceiling for free |

**Recommendation: A for the talk, C for the numbers.** Semantic AOIs let you say
"humans ranked the capybara sign first; so did the model" — concrete and
memorable. Split-half gives you the defensible statistic behind it. Implement B
as well since it's ~20 lines once C exists.

For B and C, use **mean-shift** clustering: it doesn't require specifying the
number of clusters, and its bandwidth has a principled setting — roughly 5% of
the image diagonal, approximating 2° of visual angle at typical viewing
distance, i.e. about one foveal window. Expect 4–8 AOIs on a scene like the
street photo.

### 7.2 Metrics

Taps and fixations need not be equal in number — the spatial metrics below are
count-independent. Only the sequence metrics care, and they are length-normalised.

**Spatial — do they attend to the same regions?**
- Per-AOI proportion of human taps vs. per-AOI proportion of model fixations (pooled over virtual observers).
- **Spearman ρ** across AOIs — the headline number.
- **Rank agreement of the top AOI** — the single most legible line on the screen: *"humans ranked the sign #1, the model ranked it #1."*
- **NSS** of human tap locations against the model's fixation density map — the figure comparable to published work.

**Sequential — do they go in the same order?**
- Encode each path as a string of AOI labels; compare with **Levenshtein distance**, normalised by length.
- **ScanMatch** (Cristino et al., 2010) for the established method, with a substitution matrix weighted by inter-AOI distance.
- **AOI transition matrices**, human vs. model, displayed side by side. This visualises well and shows sequence structure a single number hides.

**Baselines — is the agreement meaningful?** This is what makes the analysis
credible rather than decorative. Report alongside every figure above:
- **Random** — uniformly sampled points, same count.
- **Centre-bias** — a 2-D Gaussian at image centre. This one matters: centre bias alone explains a surprising share of agreement in any fixation data, and beating only *random* would not be impressive.
- **Human-to-human** (the split-half from method C) — the ceiling. If model-vs-human approaches human-vs-human, then *"the machine predicts this as well as you do"* is quantitatively supported, and that sentence is the whole talk.

The human-to-human ceiling is the most valuable number you will put on screen.
It needs roughly ≥ 6 responses per half to be stable, so ≥ 12 participants; at
20+ you are fine.

---

## 8. Build order

Each phase ends somewhere demonstrable.

| Phase | Deliverable | Gate |
|---|---|---|
| **1** | Repo scaffold, SQLite schema, image asset loader, `/admin` | Images visible in a list |
| **2** | `/` capture view — tap, order, undo, submit. Phone + desktop | **A phone can post markers.** The non-negotiable path works |
| **3** | `/display` + `/control` — heatmap, ordered paths, layer toggles, image switch, live poll | Full human half of the demo runs end to end |
| **4** | `tools/predict_mps.py` validated on the MacBook, then `precompute.py` + model layer rendering | Model scanpaths overlay on the image |
| **5** | `tools/analysis.py` — AOI derivation + metrics + baselines, Layer 3 panel | Numbers on screen, including the human-to-human ceiling |
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
| Live Tier C run is slow or stalls mid-talk | Tier A result is already loaded; switch layers and carry on. Never make the live run the only path to a visible result |
| MPS inference path fails on the Mac | Fallback ladder in §4.2 — allocator cap, then CPU, then a rented GPU. Slow is acceptable, so this cannot become a blocker |
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

## 10. Decisions

All resolved. Nothing blocks Phase 1.

| # | Decision | Resolution |
|---|---|---|
| 1 | **Hosting** | Zenbox, PHP. PDO driver unknown, so storage is abstracted over SQLite and MySQL (§4.1). `tools/check_host.php` decides which — run it whenever, it is not on the critical path |
| 2 | **Model runtime** | M4 MacBook, 32 GB. transformers + MPS, reusing the repo's prompt builder (§4.2). **Slow is acceptable**, so the fallback ladder ends in plain CPU and the model can always be run |
| 3 | **Taps per participant** | 5. Comparison is AOI-based, so counts need not match. Model at 5 fixations for metrics, 8 for the display overlay |
| 4 | **AOI definition** | Derived, not gridded (§7). Semantic AOIs for the narrative, split-half derivation for the statistics, avoiding the circularity trap in §7.1 |
| 5 | **Stimulus images** | 4 to start: the capybara street scene, one control (§7 / below), two spares. Precompute cost is ~2 min per image per config, so this is cheap to expand |
| 6 | **Participant sees results?** | Projector-only. During a talk, a result on their own phone competes with the screen you want them watching. A "view results" state is a small later addition if wanted |
| 7 | **Control image** | Yes. A comparable street scene with no semantic violation, so the capybara-sign effect is *visible by contrast* rather than asserted. This is the difference between a claim and a demonstration |

### 10.1 What to do first

1. Run `tools/check_host.php` on Zenbox, note the verdict, delete the file.
2. Choose and photograph the control image (§10, decision 7) — a street scene matching the capybara shot in composition but with an ordinary sign.
3. Confirm the participant URL reaches your Zenbox host from a phone on mobile data, not just your laptop.

Items 2 and 3 are the ones with a lead time. Item 1 takes two minutes.

---

## 11. Approval

Spec approved by the owner on 2026-09-19 covering: the two-part architecture
(§3), PHP + abstracted storage (§4.1), transformers + MPS with slow runtime
accepted (§4.2), the four views (§5), derived AOIs with split-half scoring
(§7), and the seven-phase build order (§8).

Build proceeds from Phase 1. Phases 1–3 deliver the complete participant and
presenter experience with no GPU involved; Phase 4 is the first point at which
anything depends on model inference working on the MacBook.
