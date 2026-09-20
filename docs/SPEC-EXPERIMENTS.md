# What to model now that there is a GPU

Status: proposal, 2026-09-20. Nothing here is built.

The first real run took **85 minutes** on MPS for one image, one prompt, ten
samples. On an RTX that is minutes, and a whole class of question stops being
a budget decision. This is what to spend it on, roughly in order of what each
buys per hour of compute.

Everything below uses the model the app already drives
(`InternVL3_5-8B-HF` + the DeepGaze LoRA adapters) through
`tools/precompute.py`, which loads the model once and reuses it across jobs.
Sampling is already batched — one image prefill, *n* decodes — so ten virtual
observers cost roughly what one does. Fifty is not ten times fifty.

---

## 0. The finding that is already on the table

Before adding anything: the one real run we have is **+0.92 correlated with a
plain centre blob**, and all ten samples start their first fixation at
(0.49, 0.51) with a standard deviation of 0.01. Free-viewing eye movements
genuinely are centre-biased, so this is not proof of a broken model — but it
does mean "the model predicts where you look" is, on this evidence, close to
"attention is central".

**Every experiment below is really a way of asking the same question: does
this model have anything beyond a centre prior?** That framing is worth
keeping, because it is the question an audience will ask and the one a
reviewer will ask.

---

## 1. Yarbus, replicated (highest value, do this first)

**The question.** Yarbus showed in 1967 that the same painting produces
completely different scanpaths depending on the question the viewer is asked
— estimate the family's wealth, guess their ages, remember their clothes. It
is the founding demonstration that looking is task-driven, not stimulus-driven.
Does this model reproduce it?

**The method.** One image, every probe in the catalogue, 20+ samples each.
Compare the resulting density maps pairwise. The measure already exists:
`analysis.comparison_maps` gives correlation against each source's split-half
ceiling.

```bash
python tools/precompute.py --repo ../DeepGaze3.5-VL --device cuda \
    --all-probes --samples 24 --images street_capybara_sign.jpg
```

**What would count as a result.** Between-task correlation *well below* each
task's own split-half ceiling. If "find the cars" and "danger" produce maps
that correlate 0.95 with each other, the prompt is decorative and the model
is drawing the same saliency map every time with a different label on it.
That is a real finding and a publishable negative.

**Why it is the best demo.** It runs live. Split the room: half get "just
look", half get "find the cars". Their measured gaze should separate. Then
show whether the model's does too. Three-way, on one image, with the audience
as one of the three arms.

**Cost.** 10 probes x 24 samples on one image. Minutes.

---

## 2. Does task-direction break the centre bias?

**The question.** A model with genuine top-down control should move its
fixations *off* centre when told to find something peripheral. One that has
learned a centre prior will not.

**The method.** For every (image, probe) pair, compute the centre-bias index
already in `analysis.centre_bias_index` and the correlation against the
synthetic centre baseline in `comparison_maps().baselines.centre`. Plot
centre-bias by task.

**What would count as a result.** Free viewing sits high (we know: 0.92);
task probes sit measurably lower. If they all sit at 0.9, section 1's answer
is already settled and section 0's caveat becomes the headline.

**Cost.** Free — it is a different reading of section 1's output.

---

## 3. Are the virtual observers dispersed like real people?

**The question.** Ten samples at temperature 0.7 are being treated as ten
observers, and a split-half ceiling is computed from them exactly as it is
for humans. That is only fair if the model's spread means the same thing as
between-subject spread. It probably does not.

**The method.** Sweep temperature (0.3 / 0.7 / 1.0 / 1.3) at 50 samples.
For each, compute the split-half ceiling. Compare against the human ceiling
from a real session.

**What would count as a result.** The temperature at which model dispersion
matches human dispersion. Below it the model is over-confident and its
ceiling is inflated, which *flatters* every comparison against it; above it
the samples are noise.

**Why it matters more than it looks.** Every "% of ceiling" figure the charts
report depends on this. If the model ceiling is 0.99 because its samples are
near-identical, the model's scores are being divided by the wrong number.
This is the least glamorous item here and possibly the most important.

**Cost.** 4 temperatures x 50 samples x 1-2 images. Minutes.

---

## 4. Counterfactual edits — does it see the joke?

**The question.** The test image has a capybara on a road sign. A viewer
notices it because it is semantically wrong, not because it is visually
salient. Does the model?

**The method.** Make variants of one scene: (a) capybara sign, (b) an
ordinary deer/pedestrian sign in the same place, (c) sign removed, (d)
capybara standing in the road instead. Run `unexpected` ("what shouldn't be
here") and `freeview` on each.

**What would count as a result.** Attention to the sign region tracks the
*oddity*, not the pixels — high in (a) and (d), low in (b) and (c), with the
sign occupying identical pixels in (a) and (b). That would be strong evidence
of semantic rather than low-level attention, and it is the kind of result an
audience feels immediately.

**The honest risk.** The variants must be pixel-identical outside the edited
region or the comparison is worthless. That is image work, not model work,
and it is the real cost of this experiment.

**Cost.** Minutes of GPU. An hour or two of careful image editing.

---

## 5. Is it modelling a process, or generating a plausible list?

**The question.** Ask for 5 fixations, then 10, then 20. If the model
simulates a viewing process, the first five of a twenty-fixation request
should resemble a five-fixation request. If it is producing a
well-shaped list to fit the requested length, they will not.

**The method.** Same image and prompt, `--num-fixations` 5 / 10 / 20, 24
samples each. Compare the density of the first 5 across conditions, and the
sequence similarity already implemented in `analysis.sequence_similarity`.

**What would count as a result.** Either answer is worth having. Agreement
supports treating the output as a scanpath. Disagreement means the number of
fixations is a formatting parameter, and every comparison should fix it at
the participant tap count and never vary it — which is what the app does
today, by luck rather than by evidence.

**Cost.** Trivial. Do it alongside section 1.

---

## 6. How much does the wording matter?

**The question.** Eight of the ten probes are marked `experimental` because
they are outside the templates the adapter was trained on. Nobody has
measured what that costs.

**The method.** Three paraphrases per probe — "people", "where are the
people", "find the humans" — 24 samples each. Correlate the maps.

**What would count as a result.** A paraphrase-stability number per probe.
Anything that moves a lot under paraphrase should carry a warning on the
display, exactly as `prompt_kind: custom` already does, but earned rather
than assumed. Anything stable can be promoted out of "experimental".

**Cost.** 10 probes x 3 paraphrases x 24 samples. Still minutes.

---

## 7. Breadth: more than one picture

Everything above is single-image. One image cannot separate "this model
understands scenes" from "this model happens to fit this scene". Ten to
twenty images across a few categories (street scenes, interiors, faces,
text-heavy, empty landscapes) turn every result above from an anecdote into
a small study — and give the live session variety instead of one picture
shown nine ways.

**This is the cheapest thing on the list and the one that most changes how
much the results are worth.**

---

## What needs building

Ranked by what unblocks the most:

1. **A sweep runner.** `precompute.py` takes one configuration per run.
   Sections 3, 5 and 6 need parameter sweeps (temperature, fixation count,
   paraphrase) with results keyed so they can be compared. This is the only
   substantial piece of work here.
2. **Variant images as first-class.** Section 4 needs image (a) and image (b)
   to be known as variants of one scene, not two unrelated files, so the app
   can show them side by side.
3. **A model-vs-model comparison view.** `comparison_maps` compares tapped,
   measured and model. Sections 1, 3, 5 and 6 all compare *model against
   model*. The maths is identical; the endpoint takes three fixed sources.
   Generalising it to N named sources is a small change.
4. **Paraphrase support in the probe catalogue** — a probe becomes a list of
   phrasings rather than one.

Nothing here needs a new model, a new metric, or a change to how participants
are handled. It is all downstream of compute that used to be unaffordable.

---

## Caveats worth keeping in front

- **The adapter is non-commercial research-licensed.** These are experiments,
  not a product.
- **A GPU makes it cheap to generate far more output than anyone will check.**
  Section 3 exists precisely because a plausible-looking number (a 0.99
  ceiling) was about to be used without asking what it meant.
- **Faster does not mean the first run was wrong.** The MPS run and a CUDA
  run of the same seed and temperature should agree; if they do not, that is
  a finding about determinism worth chasing before trusting any sweep.
