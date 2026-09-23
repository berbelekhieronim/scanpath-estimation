# Scope — what is built, what is next, what was deliberately abandoned

Written 2026-09. This is the planning document: §1 is the state of the app,
§2 is the work queued with its size and its open questions, §3 is the
graveyard — ideas that were tried or seriously considered and then rejected,
kept so they are not quietly reinvented.

The specs (`SPEC.md`, `SPEC-METRICS.md`, `SPEC-WEBCAM.md`, `SPEC-HARDWARE.md`,
`SPEC-EXPERIMENTS.md`) describe how the built parts work. This describes what
is and is not there.

---

## 1. What exists

### 1.1 The experiment

Between-subjects, assigned per round by a completion-balanced allocator. A
participant is either **tapped** (five taps, "where would you look?") or
**eye-tracked** (webcam gaze over a fixed viewing window). Never both — a
person in both conditions contaminates the comparison, and the join codes,
the routing and the thank-you screens all now enforce it in the same
direction.

Three sources are compared, pairwise, on a 3×3 grid of per-participant
density maps smoothed with one shared kernel:

| | tapped | measured | model |
|---|---|---|---|
| **tapped** | split-half ceiling | ✅ | ✅ |
| **measured** | | split-half ceiling | ✅ |
| **model** | | | split-half ceiling |

All three pairs, the ceilings, the correlation matrix and the per-pair
difference maps live on `/charts`. The projected `/display` carries the
picture and the layers, and no longer carries the agreement figures — they
were clutter at projector distance and are a thing to walk through, not
glance at.

### 1.2 The model

DeepGaze3.5-VL: LoRA adapters over InternVL3_5-8B-HF. `combined_adapter` for
free viewing, `visual_search_adapter` for the car search. Two tasks, both
trained; the seven untrained "probes" were removed (§3.4).

Runs are precomputed to `data/model/*.json` and loaded from disk. On disk
today:

| run | fixations | observers | notes |
|---|---|---|---|
| `street_capybara_sign__freeview__n5` | 5 | 10 | markedly centre-biased (r = 0.92 vs a centre blob) |
| `street_capybara_sign__freeview__n10` | 10 | 20 | r = 0.76 vs centre blob; ceiling 0.90 |

A finding worth keeping: asking for ten fixations changes the *first five*
too. `first-5-of-n10` correlates 0.94 with `n5` but carries n10's lower
centre bias (0.761 vs 0.764 for all ten). The model does not generate one
path and truncate it — the requested length changes the whole trajectory.
So runs at different `n` are **not** interchangeable, and the control says so.

### 1.3 The machines

`tools/backends.py` holds one profile per machine (M4 Air, 32GB unified;
RTX 5070 Ti, 16GB VRAM), because they fail in opposite directions —
overcommit swaps on one and raises on the other. The inference code is **not**
forked: two copies of the prompt strings would drift and the machines would
quietly produce different scanpaths.

`tools/run_session.py` is the script for a long run: it measures whether the
accelerator is actually being used, times one sample and extrapolates before
committing to the evening, and checkpoints so an hour-five crash does not
cost hours one to four.

### 1.4 Rounds

A round — one image, one sitting — exports as a self-contained JSON document
and imports back as a new round, never overwriting. Responses, calibration
grades, group assignments, model runs and the settings the round ran under
all travel. `/rounds` lists every sitting and puts any of them back on
screen. This is both the demo insurance and the substrate the round-to-round
comparison in §2.4 needs.

---

## 2. Queued

Sized S (an afternoon), M (a day or two), L (a week-ish). "Blocked on" means
a decision is needed before starting.

### 2.1 Calibration: tap-and-hold with a shrink-to-confirm — DONE

Today calibration asks the participant to look at a dot and tap it. Two
things go wrong. Aiming a finger at a 64px target pulls the eyes to the
finger, and a single instantaneous tap samples gaze at one moment, which is
the moment the thumb arrives rather than the moment the eyes settle.

The proposal: on a handset, **tap and hold anywhere except the dot**. The dot
shrinks over ~500ms while the finger is held and the gaze is on it, an
implosion that both paces the hold and gives the eye something to keep
fixating. Gaze is sampled across the hold window, not at one instant, so
each calibration point gets a median over several frames rather than one.
Release or look away early and the dot springs back.

Why it should be better: more samples per point, no finger for the eye to
chase, and an explicit "hold your gaze" instruction the interaction enforces
instead of just stating.

**Built.** Handsets only. Desktop keeps click-the-dot — the accuracy
problem is a thumb on a phone, and one interaction that works does not need
replacing to match one that is being fixed.

The rehearsal before Start is built too. It was deferred on the grounds that
the first calibration point being worst was a guess; it is not much of a
guess. That point is taken while the participant is still working out what
is being asked, and weighted the same as the eight after it. The intro
overlay now loops the gesture at its real 500ms, which is the only way to
teach the two things the text cannot: the finger goes somewhere that is
*not* the dot, and the press is held rather than tapped.

### 2.2 The validity threshold — S to decide, M to implement

Sessions are graded good / usable / poor / failed from calibration residual,
and anything below "usable" is excluded. Separately, gaze samples landing
off the picture are recorded with `on_image = 0` and dropped from the
spatial analysis.

The second is where the data actually goes. In the repository's development
database, two of eleven sessions lost 73% and 35% of their samples that way.

**This cannot be tuned from what is in the repository.** Every session in
`data/app.sqlite` is a headless test fixture or hand-seeded; there is no
sitting with a real eye behind it. Setting a threshold from that data would
be fitting to a simulation.

**Decided:** the work is to *stop discarding off-image gaze*, not to raise
the calibration gate. Today a sample that lands off the picture is dropped
from the analysis, so a participant who spent a third of the window looking
at the letterboxing contributes a third less weight than one who did not —
which is a measurement artefact wearing the costume of a preference.

The shape: keep off-image samples as an explicit "attention off the picture"
quantity per participant, report it, and stop letting it silently thin the
density map. Whether a session with a very high off-image fraction should
then be excluded *is* a threshold question, and that one still needs a real
round to set.

### 2.3 Display toggles on the display; controls as run setup — M

Today `/control` owns the layer toggles and `/display` is passive. During a
demo that means driving the projected screen from a second device, and the
full-screen layers (charts, raw JSON) are unreachable once they are up.

The proposal: layer toggles move onto `/display` itself, available on every
full-screen layer so they stay usable mid-demo, and `/control` becomes run
setup only — image, task, fixation count, viewing time, round management.
The legend shrinks to short descriptions of the items rather than a key.

**Decided:** the controls hide themselves. They fade in on pointer movement
or a key press and fade out after a few seconds, the way video player chrome
does — invisible to the audience in normal use, instantly reachable when the
presenter's own laptop is driving the projector. An always-visible strip
would be simpler and would sit on screen for the whole talk.

### 2.4 Round-to-round comparison — M

Rounds are now exportable, timestamped and labelled entities, so comparing
two sittings is mostly a selection problem. Requirements from the user,
which are the right ones:

- The comparison is **opt-in**. Not selected, not displayed.
- Never on `/display` unless explicitly turned on — a second round appearing
  unbidden during a reveal is the worst possible time.
- Rounds are selectable by date, label, and the parameters that make two
  rounds comparable at all: `n` fixations, which model run, viewing time.

The round label should be generated from those parameters rather than typed,
so two rounds are never distinguishable only by a timestamp.

### 2.5 Temporal analysis — L

The one genuinely new analysis. Both sources carry order: taps have a
sequence, gaze samples have timestamps, model fixations are numbered. Today
all three are collapsed to a density map and the order is thrown away.

The unit that makes them comparable is **rank, not seconds**: the first
third of a scanpath against the first third of a viewing window. Three bins
is the natural split.

What the frame rate allows, from the measured ~3.7–4 Hz of the browser
tracker:

| viewing window | samples/participant | 3 bins | 4 bins |
|---|---|---|---|
| 3 s (the model's prompt) | ~12 | 4 each — thin | 3 each — no |
| **5 s (today)** | ~18–20 | **6–7 each — works** | 4–5 each — thin |
| 8 s | ~30 | 10 each | 7–8 each — works |

So **a three-bin temporal analysis is possible with the viewing window as it
is**, and a four-bin one needs about eight seconds. Lengthening the window
widens the existing mismatch with the model's "free viewing for 3 seconds"
prompt, which is already stated when presenting and would need restating
more loudly.

On the model side, ten fixations splits three ways comfortably; with 25
observers that is 75–100 points per bin.

On the charts this wants animation — the three bins played in sequence
rather than shown as three static panels — because the thing being shown is
a trajectory.

**Decided: the window stays at five seconds and the analysis uses three
bins.** That works with what the tracker already produces and keeps the
humans as close to the model's three-second prompt as the frame rate allows.
Eight seconds would buy a fourth bin and cost a caveat — the humans looking
for nearly three times what the model was asked about — which is not worth
a bin.

### 2.6 A model run for the car search — M, next release

The search adapter is trained and the task is offered, but no search run has
been generated. Deliberately deferred: the free-viewing comparison is the
demo, and a second run is another evening of GPU time.

---

## 3. Deliberately abandoned

The point of this section: each of these looked reasonable, was built or
seriously planned, and turned out to be wrong for a stated reason. Rebuilding
one without a new reason is the mistake this section exists to prevent.

### 3.1 Synthetic model output as a placeholder

**Was:** `precompute.py --synthetic` generated plausible centre-biased
scanpaths without a GPU, stamped `source="synthetic"` and badged loudly in
the UI.

**Why it went:** it built the display, which was the job. Then it became the
largest risk in the room. The protection was a badge, and a badge is read by
someone checking; a presenter mid-demo reads the picture. Real runs exist
now, so the stand-in has nothing left to do.

**Would need to come back only if:** the display is rebuilt from scratch with
no runs on disk at all — and then it should be a fixture in `tests/`, not a
CLI flag on the tool that also produces real output.

### 3.2 The seven untrained probes

**Was:** danger, music, robots, people, roads, living things, counting
buildings — tasks offered in the UI and marked "experimental".

**Why it went:** the adapter never saw them. The model always returns
coordinates, so every one produced a confident-looking scanpath of
unvalidated quality. Labelling that is not fixing it, and a menu where seven
of nine entries are guesses invites picking one.

**Would need to come back only if:** one is validated against human data
first (§2.6 is the shape of that work). `build_probe_prompt()` is kept for
exactly that.

### 3.3 Offering the tap group the camera stage

**Was:** the tap thank-you screen ended with "now the other half — we can
measure where your eyes actually go", and a button into calibration.

**Why it went:** it put one person in both conditions, which is precisely the
contamination a between-subjects design exists to prevent, and it was one
tap away. Whoever is eye-tracked arrives by their own join code.

### 3.4 Agreement figures on the projected display

**Was:** an `analysis` layer on `/display` showing correlation numbers.

**Why it went:** unreadable at projector distance and clutter during the
reveal. The figures are something to walk an audience through, which is what
`/charts` is for.

### 3.5 Percentages printed in the projected attention grid

**Was:** each cell of the grid on `/display` carried `24%`.

**Why it went:** the shading already says which quadrants drew the eye, and a
room reads shading at a glance and a table not at all. It also spent the
analysis before the charts got to walk through it.

### 3.6 The "attention per cell, with uncertainty" dot plot

**Was:** a dot plot with error bars, on `/charts`.

**Why it went:** not readable without explanation. Replaced by the
correlation matrix, which answers the question people actually ask — how
much do these agree — directly.

### 3.7 A `gaze_fixations` table

**Was:** proposed in SPEC-WEBCAM §9: detect fixations from the gaze stream
and store them.

**Why it went:** the browser tracker runs near 4 Hz. Samples 250 ms apart
cannot separate fixations from saccades, so the table would have held
detections that were an artefact of the sampling rate. Raw samples are
stored and treated spatially instead.

**Would need to come back only if:** the tracker reaches a frame rate that
justifies it — the WebGL backend at ~13 Hz is the candidate, blocked on
`setBackend` not migrating weights (SPEC-WEBCAM §13.4).

### 3.8 Adding the same blur to every source to make maps comparable

**Was:** smoothing every source's density map with one identical sigma.

**Why it went:** it preserved exactly the differences it was meant to remove.
A noisier source stays noisier when everyone gets the same kernel. Errors add
in quadrature, so what is added is `sqrt(target² − own²)` — each source is
blurred *up to* a common total, not *by* a common amount.

### 3.9 Tap paths filtered to assigned tappers, with an all-or-nothing fallback

**Was:** keep only participants explicitly assigned to tap; if the assignment
table was empty, keep everyone.

**Why it went:** one stray assignment flipped it out of the fallback and
silently discarded every tapper who had submitted without one — eleven
participants to zero, with no error. Markers only ever come from tapping, so
the rule is now "everyone holding markers, except those assigned to gaze".

### 3.10 The PID file as the source of truth for the dev server

**Was:** `devserver.sh` recorded `$!` after `setsid nohup … &`.

**Why it went:** whether that is uvicorn's PID depends on whether setsid
forked or exec'd. When it forked, `stop` killed nothing, deleted the PID
file, and `start` launched a second server against a port the first still
held — so the *old* process kept serving, running whatever code it imported
at boot. That is the entire "stale server" family of bugs. The port is the
source of truth now.

### 3.12 Trusting that one press produces one pointerdown

**Was:** `calibrate.html` called `calib.tap()` straight from every
`pointerdown`, with no notion of the pointer being released in between.

**Why it went:** some devices re-fire `pointerdown` during a single
continuous press. The validation loop has no `await` between points —
`_recentPoint()` is synchronous — so each event landed on a freshly armed
`_resolveTap`, and one long touch walked the whole validation set in about
thirty milliseconds. Every point scored against **zero** gaze samples, and
the run reported itself as a completed calibration: the accuracy gate the
entire eye-tracked condition depends on, computed from nothing, presenting
as a pass.

Fixed in two independent places on purpose. The page requires a release
between confirmations, so one press is one confirmation whatever the device
emits. `Calibration.tap()` separately refuses anything inside
`MIN_TAP_GAP_MS` and counts the refusals, because the page is the part that
can be rebound or duplicated and the model has to hold anyway.

### 3.11 One tiled link per page on the start page

**Was:** three groups of tiles linking to every page.

**Why it went:** the top nav already reaches all of them. What is left on the
start page is what a nav bar cannot do — join QR codes, health checks,
diagnostics.
