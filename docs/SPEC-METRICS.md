# Comparing Tapped, Measured and Model Viewing — Spec

**Status:** Draft for approval · Extends [`SPEC.md`](SPEC.md) §7 and
[`SPEC-WEBCAM.md`](SPEC-WEBCAM.md) §6

---

## 1. Why the existing analysis does not stretch

There are now three sources, and they differ in ways that make a naive
comparison meaningless rather than merely imprecise.

| | Tapped | Measured | Model |
|---|---|---|---|
| Observations per person | 5 | ~14 | 5–8 per virtual observer |
| Spatial precision | exact | **±19–23% of screen** | exact (its own output) |
| Sampling rate | — | ~2.7 Hz | — |
| Centre bias | none expected | strong, structural | learned from real fixations |
| Temporal meaning | deliberate ranking | position samples, saccades included | discrete fixations |
| Unit of observation | a person | a person | a sample of the model |

**Three of those rows break a naive comparison:**

**Precision.** A tap is where someone meant to point. A gaze sample is within
about a fifth of the screen of where they looked. Comparing them at any
resolution finer than the *worse* of the two credits the tap data with
agreement it never had the chance to show, or blames the gaze data for error
that is measurement rather than disagreement.

**Temporal meaning.** Tap order is a deliberate ranking — first tap means
*"this is what I'd notice first"*. A gaze sample at 2.7 Hz is wherever the eye
happened to be, including mid-saccade. Calling both a "scanpath" and running
an edit distance between them compares two different kinds of object.

**Unit of observation.** Twenty people times fourteen samples is not 280
independent observations, it is twenty. Pooling points and running a test over
them inflates significance enormously, and with n≈10 per group that is the
difference between a real result and a made-up one.

---

## 2. The common representation

Everything is reduced to the same object before anything is compared: a
**smoothed density map on a fixed grid**.

**Grid: 3 × 3.** Set by the measurement error, not by preference. At ~21% of
screen width, cells narrower than a third of the image would be reporting
noise as structure. 2 × 2 is the fallback if real calibration comes back worse.

**Smoothing: one Gaussian, the same for all three sources**, with σ equal to
the *worst* source's measurement error (currently the gaze data, ~0.21). This
is the step that makes the comparison fair. Tap data is deliberately blurred
to the resolution the gaze data can support; the alternative is comparing a
sharp map with a blurry one and calling the difference disagreement.

**Per-participant normalisation.** Each person's points become one density map
that sums to 1, and the group map is the **mean of those**, not a pool of
points. Otherwise the person who produced 19 gaze samples counts more than the
one who produced 9.

---

## 3. What to measure

### 3.1 Headline: spatial agreement

**Pearson correlation between group density maps**, computed over the 9 cells.
Standard in the saliency literature (usually written CC), symmetric, and needs
no assumption about which side is right.

Report it for all three pairs, and always against these baselines:

- **Uniform random** — the floor.
- **Centre-only** — a Gaussian at the image centre. This is the one that
  matters. Centre bias alone explains a lot of agreement in gaze data, and a
  result that only beats uniform has not shown anything.
- **Split-half within group** — the ceiling. Half the tap group against the
  other half tells you how well tapping agrees with *itself*, and no
  cross-group agreement can be expected to exceed it.

### 3.2 The one number the demo rests on

```
            CC(model, measured)
  ratio =  ─────────────────────
           CC(measured, measured)     ← split-half ceiling
```

*"The model matches the room as well as the room matches itself"* is a claim
this ratio either supports or does not. It is bounded, interpretable, and does
not require either side to be ground truth.

### 3.3 Centre-bias index

Share of each source's density in the central cell, with a per-participant
confidence interval.

This is the measure most likely to separate tapped from measured, and it is a
prediction the design makes in advance: **real fixations cluster centrally for
reasons unrelated to content, deliberate taps do not**, because nobody reports
"I would look at the middle". If that separation appears, it is direct evidence
that asking people where they would look does not recover where they do.

### 3.4 Sequence, handled honestly

Tap order and gaze samples are not the same object, so they get different
treatment:

- **Within tapped:** first-tap distribution — where does the *first* mark go?
  A clean, meaningful comparison against the model's first fixation.
- **Within measured:** first-second distribution — where was the eye in the
  first second? The nearest equivalent, and also the most centre-biased part
  of any real scanpath.
- **Across the two:** compare only those two summaries. **Do not run an edit
  distance between a five-tap ranking and a fourteen-sample time series.**

### 3.5 Statistical test

**A permutation test on participant labels**, not a t-test on pooled points.

Procedure: compute the observed between-group difference (CC, or centre-bias
difference); shuffle participants between the two groups 10,000 times,
recomputing each time; the p-value is the proportion of shuffles at least as
extreme as observed.

Two reasons this is the right tool. It respects the unit of observation — a
participant moves as a whole, so the non-independence of their own samples
cannot inflate anything. And it assumes nothing about distribution shape,
which matters at n≈10 per group where normality is untestable.

**Report the effect size regardless of the p-value.** With ten per group, an
underpowered null is uninformative; the difference and its interval are what
can actually be interpreted.

---

## 4. Visualisation

Four, in order of how much they earn their place.

**1. Three density maps, side by side, one shared colour scale.** Tapped,
measured, model. A shared scale is essential — three independently-scaled
heatmaps are three different questions.

**2. The difference map: measured − tapped.** Diverging scale, zero at white.
This is the single most informative image the project can produce, because it
shows *where* prediction and reality part company rather than reducing it to
one number. If the centre goes strongly one way and a salient object the
other, the story tells itself.

**3. Per-cell proportions, three series, with confidence intervals.** The
statistical companion to the maps: nine cells on the x-axis, proportion on the
y, three bars or points per cell, participant-level CIs. This is the chart
that makes the numbers checkable.

**4. Agreement matrix.** A 3 × 3 of pairwise CC with the split-half ceilings on
the diagonal. Compact, and puts every comparison in one place.

---

## 5. Is the chart feasible now?

**Partly, and the split is worth being precise about.**

**Buildable now, and worth building now:**

- The three-panel comparison and the difference map. The data structures exist,
  the rendering is straightforward, and having them *before* real data is
  useful — they will show whether the pipeline produces anything sane, and
  they are how you will notice a systematic problem at rehearsal rather than
  during the talk.
- The per-cell proportion chart with CIs.

**Should wait for real data:**

- **The permutation test and the inferential claims.** Not because they are
  hard, but because the right test depends on what the data looks like — how
  many per group survive the quality gate, how much spread there is, whether
  the gaze group has enough usable sessions to split in half at all. Building
  the statistics before seeing the distribution risks fitting the analysis to
  a guess, which is exactly the thing the baselines exist to prevent.

**Genuinely blocked:**

- **Anything involving the model.** Every model run is still synthetic
  placeholder data. Until `precompute.py` runs against the real weights, any
  model comparison is a comparison with noise, and a chart of it would look
  entirely convincing while meaning nothing.

**Recommended order:** build the visual comparison next (panels, difference
map, proportions chart), generate real model runs, run one session with
≥10 per group, then fit the statistics to what actually arrives.

---

## 6. Open decisions

1. **Grid at 3 × 3 or 2 × 2?** 3 × 3 is defensible at 21% error but is at the
   limit. Decide once there are several real calibrations.
2. **Minimum per group before showing metrics at all?** Suggest 6 — below that
   the split-half ceiling is noise and the panel should say so rather than
   print a number.
3. **Do excluded gaze sessions appear anywhere on the display?** Currently
   counted but not shown. The exclusion rate is honest and interesting;
   showing it costs a line.
4. **Should the measured group's first second be treated separately** in the
   main comparison, or only in the sequence section? It is the most centre-
   biased and most informative part of the recording.

---

## 7. What was built (2026-09-20)

All four visualisations from §4 now exist at `/charts`, fed by
`GET /api/compare/maps` (`analysis.comparison_maps`). Five things landed
differently from the spec above, each for a reason.

**1. The difference map's zero is grey, not white.** §4 says "zero at white".
White is the light-mode surface, so a zero cell would have disappeared into
the card, and in dark mode there is no white to diverge from at all. The
midpoint is now the neutral grey the palette specifies for each mode (light
`#f0efec`, dark `#383835`). Cells within 4% of the largest difference are
snapped to that midpoint deliberately: at eight participants a one-point gap
is not a finding, and a faint tint invites reading it as one.

**2. Per-cell proportions are dots, not bars, and the axes are swapped.**
§4 puts nine cells on the x-axis. Nine categorical labels on the x-axis at
phone width either rotate or truncate; cells run down the y-axis instead, and
each series is a dot with its interval drawn through it. Bars with error bars
would have needed a zero baseline per cell and three times the ink for the
same nine comparisons.

**3. The agreement matrix is a table, not a 3 × 3 grid.** With three sources
there are exactly three pairs. A matrix of three filled cells and three
mirrored duplicates is a table wearing a costume — the anti-pattern is a chart
where the number is the point. The split-half ceilings ride along as their own
column, which is where they are actually read.

**4. Colours were re-derived, and the display's are wrong.** The categorical
slots used here (`#3987e5`/`#d95926`/`#199e70` dark, blue swapped to `#2a78d6`
light) were chosen by running the palette validator against this app's own
surfaces. The display's existing overlay hues — tap `#5b8cff`, gaze `#b06bff`,
model `#ff7a45` — fail it: blue against purple separates by ΔE 1.5 under
deutan simulation and 13.2 with normal colour vision, below the floor of 15.
Roughly one man in twelve cannot tell the tapped layer from the measured layer
on the projected screen. Realigning the display to the validated slots is a
separate change, not yet made.

**5. There is no filter row.** Grid size and blur are query parameters on the
endpoint (`?grid=4&sigma=0.25`), clamped rather than validated so a stray URL
cannot take the page down mid-session. They are not exposed as controls
because §6 has not yet decided the grid, and a control implies the decision is
the viewer's to make.

The page refuses to overstate what it has: it names the model panel as
synthetic whenever the run is a placeholder, says so when a group has fewer
than four people and the ceiling cannot be estimated, and reports excluded
calibrations rather than dropping them. §5's "should wait" and "blocked"
items are unchanged — no inferential test is computed, and every model number
on the page is still a number about noise.


---

## 8. Two screens, one copy of the code (2026-09-20)

The charts are drawn on the projected display as well, as a layer the
presenter toggles from the control page like any other. Two screens showing
the same comparison could quietly disagree about what the data says, so the
drawing code is a single ES module (`app/static/charts.js`) and the
differences between the screens are options passed to it, not second copies.

**The projected view shows less on purpose.** It carries the three density
panels, the difference map, the legend and one plain-language headline
sentence. It leaves out the dot plot, the numbers table and every tooltip:
nobody hovers a mark or reads a nine-row table from the back of a room, and
including them would only make the two images that do work smaller. The
caveats stay — a synthetic model run and the count of excluded calibrations
are named in the footer, at projector size.

**Charts and raw JSON are mutually exclusive**, enforced in `LAYERS.set`
rather than in the control page, because it is the display that has the
constraint: both take over the whole screen, so two on at once means one is
invisible behind the other. Turning either on turns the other off. The image
overlays (heatmap, scanpaths, model, gaze) are untouched by the rule and keep
their state while a full-screen layer is up.

**The colour scale's direction flips with the mode**, and the captions follow
it. On a light page near-zero is the light end of the blue ramp, so darker
means more; on a dark page near-zero is the dark end, so lighter means more.
The word in the caption is derived from the mode (`MORE_IS`), because a
hard-coded "darker means more" is wrong half the time — it was, briefly.
