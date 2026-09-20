# Webcam Gaze Capture — Spec Addendum

**Status:** Draft for approval · Extends [`SPEC.md`](SPEC.md)
**Owner:** bszperlinski1@st.swps.edu.pl

---

## 1. Why this is the most valuable thing left to build

§1.1 of the main spec opens with a caveat that has shaped everything since:

> Taps are not fixations. What a participant *deliberately taps* is a slow,
> top-down, reportable judgement. What their eyes *actually do* in the first 3
> seconds is fast, largely bottom-up, and strongly centre-biased.

Every number the demo currently produces sits behind that caveat. Webcam
capture removes it. Instead of *"where people said they'd look versus what the
model predicts"* the demo becomes *"where people **actually looked**, versus
where they **thought** they would, versus what the model predicts."*

That is three-way, and the third axis is the interesting one, because it is
**within-subject**. The same person, on the same image, produces a prediction
about themselves and a measurement of themselves. The gap between those two is
a result you can show an individual about their own eyes, in the room, in
thirty seconds.

**There is also a reason to expect the model to look better under this mode.**
DeepGaze3.5-VL was trained on real eye-tracking data — MIT, CAT2000, COCO,
Daemons, FIGRIM. It has learned the statistics of real fixations, centre bias
among them. Taps have no centre bias, because nobody deliberately reports
"I would look at the middle of the picture". So the model may well agree with
*measured* gaze substantially better than it agrees with *tapped* predictions.
If that happens it is not a coincidence and not a flattering artefact: it is
the demo's whole thesis showing up in the numbers.

---

## 2. The constraint that decides everything: accuracy

Read this section before any design opinions form. The available accuracy
determines what analysis is defensible, and it is tighter than intuition
suggests.

### 2.1 What the literature actually reports

| System | Reported error | Notes |
|---|---|---|
| [WebGazer.js](https://webgazer.cs.brown.edu/) | **4.17°** visual angle, 175–210 px | The mature, widely-used option. Its own documentation says fine-grained tracking is *not* feasible and that you can identify the screen **quadrant** |
| [WebEyeTrack](https://github.com/RedForestAi/WebEyeTrack) | **2.32 cm** (GazeCapture, in-distribution); **4.56 cm** cross-dataset | 2026, MIT licence, npm package, TensorFlow.js, 9-sample on-device personalisation, 2.4 ms inference on an iPhone 14 |
| [Google, *Nature Comms* 2020](https://www.nature.com/articles/s41467-020-18360-5) | **0.46 cm** (0.6–1°) | Comparable to Tobii glasses. **A research model, not a released library.** This is the ceiling, not an option |

### 2.2 What that means on an actual phone

A 6.1-inch phone is about 7 cm wide and is held at roughly 30 cm. The image in
our capture view occupies about 6.6 cm of that.

| System | Error | As % of image width | Resolvable columns |
|---|---|---|---|
| WebEyeTrack, in-distribution | 2.32 cm (4.4°) | 35% | **~3** |
| WebGazer | 2.19 cm (4.2°) | 33% | **~3** |
| WebEyeTrack, cross-dataset | 4.56 cm (8.6°) | 69% | **~1** |
| Google research model | 0.46 cm (0.9°) | 7% | ~14 |

**On a phone, expect a 3 × 3 grid. That is the honest ceiling.**

A laptop is roughly twice as good — a 30 cm screen at 60 cm subtends 28°, so
the same angular error covers 15% of the width and resolves about **6 columns**.
The difference is pure geometry: a bigger screen further away gives the same
eye rotation more pixels to land in.

### 2.3 Consequences you cannot design around

1. **The current AOI analysis does not transfer.** §7 of the main spec derives
   6–12 areas by mean-shift clustering. On phone gaze data those clusters would
   be smaller than the measurement error — the analysis would be reporting
   structure that is noise. Webcam mode needs its own, coarser analysis (§6).
2. **Sequence comparison mostly dies.** A five-step scanpath across a 3 × 3 grid
   carries very little information, and consecutive fixations will often fall in
   the same cell. Report transitions between coarse regions, not edit distance.
3. **The findings that survive are the big ones.** Centre bias, top-versus-bottom,
   whether the salient object drew attention at all, and first-fixation region.
   Those are large effects that 4° of error does not erase — and they are also
   the effects worth showing an audience.

**None of this makes the feature not worth building.** Coarse measured gaze
beats precise guesses for the demo's actual claim. It does mean the feature
must not be sold as "eye tracking" in the sense a psychophysics lab means it.

---

## 3. What the phones make possible

This is the part that is genuinely novel rather than merely a port of a lab
setup.

**Every participant becomes an eye tracker at once.** Twenty simultaneous
scanpaths on the same stimulus, gathered in under a minute, is something no
single-station lab setup can do. The per-participant data is coarse; the
aggregate over twenty people is not, because averaging twenty noisy estimates
of the same underlying distribution recovers a lot of what each one lost.

This is the key statistical point of the whole feature: **individual
measurements are poor, the group heatmap is good.** Show individuals as a
caveat and the aggregate as the result.

**Within-subject prediction versus measurement.** Each person produces both, so
the demo can show one anonymous participant their own two-panel result: here is
where you said you'd look; here is where you did. Nothing in the tap-only
version can do this.

**What phones do *not* enable:** tracking gaze onto the projected screen. The
tracker needs known geometry between camera, eye and stimulus, which only holds
when the image is on the same device as the camera. The image must be displayed
on each participant's own phone.

---

## 4. Library choice

**Recommendation: WebEyeTrack**, with WebGazer as a fallback.

- MIT licence — unencumbered, unlike the scanpath model's non-commercial terms.
- Published as an npm package with a TensorFlow.js browser build, explicitly targeting mobile.
- 9-sample few-shot personalisation, which is a short calibration rather than a long one.
- 670 KB model, 0.15 GFLOPs, ~2.4 ms inference on an iPhone 14 — comfortably real-time on a phone.
- On-device inference and calibration, which is what makes the privacy story true rather than aspirational (§7).

WebGazer is the conservative alternative: older, more widely validated in
published studies, similar angular accuracy, but heavier and less mobile-focused.

**Build the capture layer against an interface, not a library.** Both produce
`(x, y, timestamp, confidence)`; everything downstream should depend only on
that. Swapping trackers then costs one adapter, and the choice can be revisited
after the first real-room test — which is the only test that matters here.

---

## 5. Participant flow

Three stages, all on the participant's own phone. Stage 1 is the existing
capture view, unchanged.

### Stage 1 — Predict (existing)
Tap five places in order. Unchanged, and deliberately first: doing it *before*
seeing the image again under measurement keeps the prediction naive.

### Stage 2 — Consent and calibration

**Consent screen.** Plain language, no dark patterns, one screen:
what is captured (gaze coordinates), what is not (video, ever), where it is
processed (on the phone), and a decline that is as easy as accepting. Declining
keeps them in the demo — they keep their Stage 1 tap data and skip to the end.

**Calibration.** Nine points in a 3 × 3 grid, one at a time.

- A dot appears; the participant looks at it and **taps it**. The tap confirms fixation and gives the tracker a labelled sample.
- Dot order randomised, so learning the sequence cannot substitute for looking.
- Target 20–30 seconds total. Longer loses the room.
- Hold the phone still, at a comfortable reading distance, in that position for the whole task.

**Validation.** Four more points, not used for training, scored for error. This
produces a per-participant quality number, which §5.1 then acts on.

### Stage 3 — Free viewing

The image, full-frame, no markers, no buttons. Three to five seconds, a
countdown ring so the participant knows it will end. Gaze sampled throughout.
Then: "done — watch the screen."

### 5.1 Quality gating is mandatory, not optional

With twenty uncontrolled participants in a room, a meaningful fraction will
produce unusable data: glasses, low light, phone drifting, a face half out of
frame. **Data that failed calibration must be excluded from the aggregate and
the exclusion must be reported**, not silently averaged in.

- Validation error above a threshold (start at ~1.5× the expected 2.5 cm, tune on real data) → excluded, participant offered one retry.
- Tracking lost for more than ~30% of the viewing window → excluded.
- Fewer than N valid samples → excluded.
- The display shows both counts: *"17 of 23 tracked successfully."*

Reporting the exclusion rate is not an admission of weakness. Hiding it would
make every other number untrustworthy, and the rate is itself interesting —
it is the honest cost of doing this on consumer hardware in a real room.

---

## 6. Analysis changes

Webcam mode gets its own analysis path. It shares the baseline philosophy of
§7 in the main spec — nothing is ground truth, everything is reported against
baselines — but at a resolution the data supports.

**Coarse AOIs, fixed.** A 3 × 3 grid, not data-driven clusters. Mean-shift on
data this noisy would invent structure. The grid is also easier to narrate:
"top-left", "centre", "bottom-right" need no explanation.

**Metrics that survive the error budget:**

| Metric | Why it holds up |
|---|---|
| Distribution across the 9 cells | Aggregates over participants, so noise averages down |
| **Centre-bias index** — share of gaze in the central cell | A large effect, and the single clearest tap-versus-gaze difference to expect |
| First-fixation cell | Coarse, robust, and the most intuitive thing to show |
| Dwell time per cell | Duration is measured well even when position is not |
| Coarse transitions (9 × 9 matrix) | Structure without pretending to fine sequence |

**Metrics to drop in this mode:** mean-shift AOIs, edit-distance sequence
similarity, anything that assumes sub-degree precision.

**The three-way comparison** is the output that matters:

```
             tapped prediction   measured gaze   model prediction
  centre bias       low              HIGH             high?
  top AOI           ?                 ?                ?
  agreement    ──────────── the number the talk rests on ────────────
```

Report model-versus-measured and model-versus-tapped side by side. If the model
agrees better with measurement than with prediction — which §1 argues is likely
— that is the headline, and it is a stronger claim than anything the current
build can make.

Keep the split-half human-to-human ceiling. It matters more here, not less,
because it absorbs measurement noise: if measured gaze is noisy, human-to-human
agreement drops too, so the ratio stays interpretable.

---

## 7. Privacy, consent and ethics

Webcam capture changes this project's ethical footing, and the design has to
carry that rather than a disclaimer.

**Video never leaves the phone.** Inference runs client-side; the only thing
uploaded is a list of gaze coordinates. This must be *true*, not merely stated —
no frame uploads, no "diagnostic" image capture, no exceptions.

**Requirements:**
- Explicit opt-in before the camera is touched. Declining is one tap and keeps them in the demo.
- A visible recording indicator whenever the camera is live, plus a stop control on every screen.
- Camera released the instant Stage 3 ends. Not on navigation, not on timeout — immediately.
- No face embeddings, head-pose traces or identity-adjacent derivatives stored. Gaze coordinates and a calibration quality score, nothing else.
- HTTPS is mandatory for `getUserMedia`; Codespaces forwarded URLs already satisfy this.
- The existing anonymous UUID stays. No accounts, no emails.

**Where this crosses into research.** As a live demonstration with verbal
consent, this is ordinary. The moment the data is analysed for a paper, a
poster, or anything with a conclusion attached, it is human-subjects research
with biometric-adjacent capture and needs ethics approval before collection,
not after. Given the institutional email on this project, that line is worth
drawing explicitly now: **the export from a demo session is not research data
unless it was collected under approval.**

---

## 8. What will break

Ranked by how likely they are to bite in a real room.

| Risk | Reality | Mitigation |
|---|---|---|
| **Low light** | A darkened lecture theatre is exactly the condition front cameras fail in, and exactly the condition you create by projecting | Run the eye-tracking segment with the room lights **up**. Force phone screen brightness to maximum — the screen is also the fill light. Test in the actual room |
| **Glasses** | Reflections and refraction degrade or defeat iris tracking | Expect it, exclude by quality gate, mention the exclusion rate honestly |
| **Handheld drift** | The phone moves; calibration assumes it does not | Keep the viewing window short (3–5 s). Re-validate after. Consider asking people to brace elbows on a desk |
| **Permission refusal** | Some will decline, and should be able to | Full graceful path: they keep Stage 1 and see the result |
| **iOS/Android variance** | Camera APIs, frame rates and front-camera placement differ | Test on both before the session. Record device model with each session for post-hoc diagnosis |
| **Calibration fatigue** | Nine points is already at the limit of audience patience | Randomise, keep it under 30 s, show progress |
| **Phone thermals / battery** | TF.js inference for minutes is warm work | Only run the tracker during calibration and viewing, never idle |

---

## 9. Architecture

Fits the existing shape: the web app gains a capture mode, the model runner is
untouched, and nothing new needs a GPU.

**New data model:**

```sql
gaze_sessions(id, round_id, participant_id, started_at, ended_at,
              calibration_error, validation_points_json, tracker, tracker_version,
              device_label, screen_w, screen_h, viewport_w, viewport_h,
              excluded, exclusion_reason)

gaze_samples(id, session_id, t_ms, x, y, confidence)   -- normalised 0-1, ~10 Hz
gaze_fixations(id, session_id, seq, x, y, start_ms, duration_ms)  -- client-derived
```

Storing both raw samples and derived fixations is deliberate: the fixation
detection threshold is a parameter you will want to revisit, and only raw
samples let you revisit it.

**Volume** is trivial — 5 s × 10 Hz × 25 people ≈ 1,250 rows per round.

**New endpoints:**

| Endpoint | Purpose |
|---|---|
| `POST /api/gaze/session` | Open a session, store calibration quality, return an id or a rejection |
| `POST /api/gaze/samples` | Batch upload at end of viewing (not streaming — a dropped connection mid-stream should not lose the lot) |
| `GET /api/gaze/aggregate` | Coarse grid distribution for the display |
| `GET /api/analysis/three-way` | Tapped vs measured vs model |

**New participant routes:** `/consent`, `/calibrate`, `/view`.

**New display layers:** *Measured gaze* (its own heatmap), *Tapped vs measured*
(side-by-side), and a tracking-quality readout showing the exclusion count.

**Client bundle.** This is the one real departure: WebEyeTrack brings
TensorFlow.js, so the participant page stops being a dependency-free 50 KB.
Load the tracker **only** on the calibration and viewing routes, never on the
tap route, so Stage 1 stays as light as it is now.

---

## 10. Build order

| Phase | Deliverable | Gate |
|---|---|---|
| **W1** | Tracker adapter interface + WebEyeTrack behind it; a bare test page showing a live gaze dot | **Gate MET on a real phone, 2026-09-20** — dot tracks, face-found and FPS both above threshold, all dependencies reachable |
| **W2** | Consent screen, 9-point calibration, validation scoring, quality gate | **Built, see §13.** Verified end to end against the mock backend; untested against a real eye |
| **W3** | Stage 3 viewing, sampling, batch upload, session storage | Gaze data lands in the database with a quality score |
| **W4** | Coarse-grid analysis, measured-gaze display layer, exclusion reporting | The room's measured heatmap appears on the projector |
| **W5** | Three-way comparison (tapped / measured / model) and the per-participant two-panel result | The headline claim |

**Stop after W1 if the dot does not track in the actual room.** That single
check — on the venue's lighting, on a handful of real phones, with a few people
wearing glasses — decides whether the remaining four phases are worth building.
It is a day of work and it is the honest gate.

---

## 11. Open decisions

1. **Does the tap stage stay for everyone?** Recommended yes — the within-subject comparison is the strongest output, and it needs both.
2. **Laptop stations as well?** Two or three laptops give roughly double the resolution (§2.2) and a cleaner subset to report alongside the phone aggregate. Worth it if the room allows.
3. **Viewing duration?** 3 s matches the model's training prompt ("free viewing for 3 seconds"), which argues for 3 over 5.
4. **Show individuals their own result?** Powerful, but needs a per-participant results route and raises the "my data on screen" question. Recommend: yes, on their own phone only, never projected.
5. **Grid resolution: 3 × 3 or 3 × 2?** Start at 3 × 3 and check against real calibration error before trusting it.
6. **Is any of this going to become research output?** If yes, ethics approval must precede the first session that collects data (§7).

---

## 12. W1 as built

Implemented at `/gazetest`, with the adapter in `app/static/gaze/tracker.js`.

### 12.1 Two packaging problems, both fixed

**The published npm package cannot run as documented.** `dist/index.js`
references `index.worker.js`, which upstream's webpack build emits and the
published tarball omits. `WebEyeTrackProxy` — the worker-based class the
upstream README tells you to use — therefore cannot start. The fix is to use
the `WebEyeTrack` class, which runs on the main thread and needs no worker. At
the ~2.4 ms per frame the paper reports, that is affordable; it does mean
inference shares the UI thread, which is worth revisiting if upstream ships
the worker.

**The UMD bundle does not namespace its exports.** It copies each class
straight onto the global object, so they arrive as `window.WebEyeTrack` and
`window.WebcamClient`, not under a `window.webeyetrack` namespace. Found by
loading the bundle in a headless browser and diffing `Object.keys(window)`.

Both are pinned by tests, so an upstream change surfaces as a failure rather
than a mystery on the day.

### 12.2 Vendored, and what still is not

`webeyetrack@0.0.2`'s UMD bundle (2.7 MB) and the BlazeGaze weights (669 KB)
are committed. The weights must be served at exactly `/web/model.json` because
the bundle hard-codes that path; a test asserts both the mount and the
hard-coding.

**Two dependencies are still fetched from the internet at runtime:**
`cdn.jsdelivr.net` for the MediaPipe WASM runtime, and
`storage.googleapis.com` for the face landmark model. If the venue blocks
either, tracking will not start. `/gazetest` has a **Check** button that tests
every dependency and names which are unreachable — run it on the venue wifi.
Vendoring these two is the obvious follow-up.

### 12.3 The mock backend

`MockBackend` produces pointer-driven samples with noise at roughly the real
tracker's error. It is not a placeholder for its own sake: it lets W2–W5 —
calibration, sampling, storage, analysis — be built and tested without waiting
on camera hardware, and gives automated tests a deterministic source. Anything
built against it meets realistically noisy data rather than a clean signal.

### 12.4 What is verified, and what is not

Verified in a headless browser: page and asset serving, library loading,
the mock pipeline end to end (dot, trail, FPS and face-rate counters, start
and stop), camera release on stop, on tab-hide and on failure, and the error
messages for a blocked dependency.

**Not verified: that it tracks a real eye.** This environment has no camera and
no face, and its network blocks the MediaPipe CDN, so the tracker has never
completed initialisation here. Everything up to that point works; the gaze
estimate itself is unproven.

**The W1 gate is therefore still open.** It needs one session with a real phone
on the venue network:

1. Open `/gazetest` on a phone over the Codespace HTTPS URL.
2. Press **Check** — all four dependencies must read reachable.
3. Press **Start camera**, allow the prompt.
4. Look at each of the five targets in turn.

Judge it on: does the dot move in the right direction; does *face found* stay
above ~80%; does FPS hold above ~8. Absolute position will be off before
calibration (W2) — W1 is about whether the signal exists at all, not where it
lands. Try it with the room lights down as well as up, since that is the
condition that most likely defeats it.

---

## 13. W2 as built

`/consent` then `/calibrate`, with the calibration state machine in
`app/static/gaze/calibration.js` — deliberately independent of any tracker, so
the mock backend drives it in tests and a camera drives it in the room.

### 13.1 A silent data-loss bug in the tracker's defaults

`WebEyeTrack`'s constructor takes `maxPoints`, **which defaults to 5**, and
`pruneCalibData()` keeps only that many most-recent points. A nine-point
calibration built on the default would have **silently discarded the first
four points** and then reported a confident, wrong model — no error, no
warning, just a worse fit than the progress bar implied.

The adapter now constructs it with an explicit `maxPoints` of 16, and a test
asserts both that an argument is passed and that it is at least 9.

Two related behaviours in `handleClick()` also needed handling: it drops a
point silently if it arrives within 1000 ms of the previous one, or within
0.05 units of it. The calibration grid is spaced to clear the proximity
filter, the adapter enforces the interval, and a dropped point is retried once
rather than ignored — otherwise the model is weaker than the progress
indicator claims.

### 13.2 Validation samples before the tap, not after

Found by a failure-path test that would not fail. Sampling gaze *after* the
confirming tap measures wherever the participant drifted to once nothing held
their attention on the target, and scores that drift as tracker error. The
window now taken is the one immediately **preceding** the tap, which is the
only moment we know they were looking at it — and it matches what the tracker
itself does for calibration, adapting on the frame at click time.

### 13.3 Quality gating

Four validation points, none of them a calibration point, scored as mean
Euclidean error in fractions of viewport width:

| Grade | Mean error | Outcome |
|---|---|---|
| good | ≤ 0.18 | used |
| usable | ≤ 0.30 | used |
| poor | > 0.30 | excluded, one retry offered |
| failed | no face, or no samples | excluded, one retry offered |

**These thresholds are starting values, not measurements.** The literature puts
webcam error near 4°, which on a phone is about a third of the screen, so
"good" here is already coarse. Tune them once there is real session data.

Failed calibrations are **stored, not discarded**. The proportion of the room
whose tracking did not work is a number the presenter has to be able to state;
it cannot be stated if the failures were never written down. `/api/gaze/stats`
returns total, usable and excluded counts for the current round.

### 13.4 Consent

Five plain statements — processed on the device, no video uploaded, only
coordinates shared, no identity, camera released afterwards — with declining
presented as an equal-weight button that keeps the participant in the demo.
The choice is remembered locally so nobody is asked twice.

### 13.6 First real-device test, 2026-09-20

Tested on a phone. It hung during calibration and produced no result at all,
which is the worst possible failure: nothing to diagnose from. Four changes
came out of it.

**Resting the phone beats holding it, by a lot.** The owner's observation, and
it changes the instruction rather than the code: the opening screen now says
to put the phone on the desk. §8 listed handheld drift as a risk; this
promotes it from a risk to a design decision. For a seated audience, "rest
your phone on the desk" is easy to ask for and materially improves the data.
It also makes the tapping less tiring, which was the other complaint.

**The hang is now impossible to reproduce silently.** Upstream's frame loop
exits permanently the moment the video element pauses, and mobile browsers
pause off-screen or zero-opacity video — which is exactly what the page had,
a 1x1 transparent element. It is now a visible 74x56 thumbnail, which also
lets the participant see their own framing. On top of that a watchdog ends the
run with a message if faces stop arriving for 15 seconds, and every exit from
`run()` now returns a result. A hang with nothing to look at should not happen
again; if tracking does stop, the screen says so and reports frames seen, face
rate and points accepted.

**Taps are acknowledged immediately** — the target flashes, a ripple expands
from the finger, and the phone buzzes where supported. This is not decoration.
Upstream debounces calibration points at 1000 ms, so an unacknowledged tap
invites a second one, and the second is precisely the one silently dropped.

**Twelve taps instead of thirteen**, with tighter settle and sample windows.
Validation dropped from four points to three. Nine calibration points stay,
since that is what the tracker's few-shot adaptation expects.

A live face indicator now sits in the corner during calibration, so a stall is
visible while it happens rather than only in the result, and the diagnostics
are stored with each session.

### 13.7 Second real-device test: the tab crashed

The video fix worked — camera visible, launch clean, tap feedback good, taps
registering. Then, after calibration, Safari showed *"a problem repeatedly
occurred"*: the tab was killed. Almost certainly memory.

**This was probably caused by the previous fix.** Setting `maxPoints` to 16 to
stop calibration points being discarded also meant more retained support
tensors, and `adapt()` concatenates every retained point on each call. Each
point holds a 512x128x3 eye patch, roughly 786 KB. Upstream disposes almost
nothing — two `tf.dispose` calls against eleven tensor creations — so nine
points is about 7 MB of WebGL textures held live, re-concatenated nine times,
with Adam's optimiser state on top. iOS Safari kills tabs for less.

Three changes:

**`maxPoints` is now exactly 9**, the number of calibration points. Anything
higher buys nothing and costs memory.

**Calibration tensors are freed the moment calibration ends.** They exist only
to adapt again; the fitted transform is already applied. `releaseCalibrationMemory()`
disposes the retained support tensors from outside the library, at exactly the
point the crash happened.

**Memory is now instrumented.** The live chip during calibration shows tensor
count and megabytes alongside the face rate, the result screen reports peak
and freed counts, and both are stored. A monotonic climb in tensor count is
the signature to look for.

**And the diagnostics now survive a crash.** That is the real lesson of both
failed tests: when the tab dies, nothing is posted and there is nothing to
look at. Progress is now written to `localStorage` as it happens and any
unsent breadcrumb is posted on the next load, as a failed session with
`crashed_at_<stage>`. A crashed run reports how far it got, its face rate and
its tensor count. Verified by simulating a mid-run crash: the reloaded page
reported the dead run.

**`?light=1`** halves the calibration points as a fallback if memory is still
the culprit. A weaker fit, but half the retained tensors.

### 13.8 Third test: the actual bug was mine

Still crashing, and again with nothing to look at. But *"after a couple of
calibration attempts"* was the detail that mattered, and it pointed at this
project's code rather than the library's.

**Every retry built a whole new tracker.** `begin()` called `createTracker()`
each time, which constructs a fresh `WebEyeTrack`, which loads another
BlazeGaze model and another MediaPipe FaceLandmarker. `stop()` only stopped
the camera stream, and **upstream exposes no `dispose()` on either object**.
So attempt one held one model, attempt two held two, attempt three held
three — and the tab died. That matches the reported symptom exactly, in a way
the earlier tensor-leak theory did not.

The tracker is now created once and reused. `restart()` clears the fitted
calibration and opens a fresh camera client without touching the model.
Verified in a browser with a stubbed library: three attempts now produce
**one** model construction and **one** `initialize()`, against three of each
before.

`destroy()` is a genuine teardown for leaving the page, reaching past
upstream's missing API to `tf.LayersModel.dispose()` and MediaPipe's
`FaceLandmarker.close()`. The tap listener is also bound once rather than per
attempt, which previously meant the Nth attempt fired N taps per touch —
quietly corrupting calibration on every retry.

**The previous two diagnoses were wrong, and the pattern is worth naming.**
Each round produced a plausible theory from reading the library, and each
shipped a change that did not fix it. What broke the cycle was a detail in the
report — "a couple of attempts" — that no amount of source reading would have
supplied. The memory instrumentation and the crash breadcrumbs from those
rounds are still worth having, but they were not the fix.

The breadcrumb also now renders **on the start screen** rather than only
posting to the server, since two rounds ended with "no logs": a crashed tab
cannot report itself, and a server-side record is no use to the person holding
the phone.

### 13.5 Verified, and not

Driven end to end in a headless browser against the mock backend: consent flow
and storage, all 9 calibration and 4 validation targets appearing at 13
distinct positions, tapping anywhere counting as a confirmation, grading,
retry on failure, camera release, and storage of both a passing and a failing
session with the exclusion counted.

Discriminates correctly: a simulated participant looking straight at each
target grades *good*; one whose gaze sits ~150 px off grades *poor* and is
offered a retry.

**Untested against a real eye.** No camera here. In particular the real
tracker's post-calibration accuracy — the number the whole gate depends on —
is unknown until someone runs `/consent` on a phone.
