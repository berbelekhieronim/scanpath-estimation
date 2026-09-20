# Runbook

Practical operating guide. Keep this open next to the terminal.

---

## The one thing to understand first

**There are no separate apps.** There is one server process. The participant
view, the admin page, the projector display and your controls are just
different URLs on it. Starting the server starts all of them at once.

```
        ONE process  (uvicorn app.main:app)
                 │
   ┌──────┬──────┼────────┬──────────┬─────────┐
   │      │      │        │          │         │
/start    /    /qr    /display   /control   /admin
 you    phones project project    laptop     laptop
```

**Open `/start` first.** It lists every page with a description of who it is
for, shows live status (images loaded, model runs, responses so far, the join
address) and warns about anything not ready. Paste the control token there once
and the presenter links unlock.

The same applies to the analysis: it is **not a separate program you run**. It
is a layer you switch on, recomputed from the database every two seconds.

---

## 1. Start it

In the Codespace terminal:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` matters. Without it the server only listens on localhost and
no phone can reach it.

Leave that terminal running. It prints the control token and the URLs, leading
with `/start`:

```
  images found:   2
  model runs:     2

  START HERE:     /start        <- links to every page

  participant:    /            (what phones scan into)
  join screen:    /qr          (project while people join)
  display:        /display     (project during the demo)
  controls:       /control?k=C5ufh8ZJ4MwP
  admin:          /admin?k=C5ufh8ZJ4MwP

  control token:  C5ufh8ZJ4MwP
```

**After every `git pull`, restart the server.** Pages are read from disk on
each request, so a pull updates the interface immediately — but Python keeps
running whatever it imported at startup. The result is an interface offering
features the backend has never heard of, which presents as several unrelated
bugs at once. `/start` detects this and says so in red, but the cheapest fix is
to add `--reload` while developing:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Copy the token.** You need it for `/control`, `/admin` and the tools. It is
stored in the database, so it stays the same across restarts.

*Alternative:* `tools/devserver.sh start` runs it in the background and writes
to `.devserver.log`. Use `stop` and `restart` likewise. Handy while setting up,
but during a live session prefer the plain command in a visible terminal so you
can see requests arriving.

### Make the port public — do not skip this

In the Codespace **Ports** panel, find port 8000 and set Visibility to
**Public**. If you don't, every participant hits a GitHub login screen instead
of your app.

Check it again after any restart. It can silently revert, and it is the single
most likely way for the demo to fail.

---

## 2. Load your images

Put image files in `data/images/` (`.jpg`, `.jpeg`, `.png`, `.webp`).

They are picked up at startup. To add more without restarting, open
`/admin?k=TOKEN` and press **Rescan folder**.

In `/admin` you also choose which image participants currently see. Setting an
active image opens a fresh round.

---

## 3. Get the model's predictions

This does **not** happen in the Codespace — there is no GPU there. Run it on
your MacBook, then bring the results over.

One-time setup on the Mac:

```bash
git clone https://github.com/Susmit-A/DeepGaze3.5-VL   # needs git-lfs
cd DeepGaze3.5-VL && git lfs pull && cd ..
python3 -m venv .venv-model && source .venv-model/bin/activate
pip install -r requirements-model.txt
python tools/gaze_prompts.py --verify ../DeepGaze3.5-VL   # expect "match exactly"
```

Then, with the same images present locally:

```bash
python tools/precompute.py --repo ../DeepGaze3.5-VL \
    --modes freeview --num-fixations 5 --samples 10 --temperature 0.7
```

This writes one JSON per image into `data/model/`. Commit and push them, pull
in the Codespace, then press **Reload runs from disk** in `/control`.

**Delete the synthetic placeholders first** — `rm data/model/*.json` — or the
display will show a red SYNTHETIC banner over your results.

---

## 4. Run the session

Open two browser windows:

| Window | URL | Where |
|---|---|---|
| Controls | `/control?k=TOKEN` | Your laptop, private |
| Display | `/display` | Projector |

Plus `/qr` on the projector while people join.

### The sequence

1. **`/qr` on screen.** People scan, land on the tap page, place five markers in
   order, submit. The response counter rises on `/qr`, `/display` and
   `/control`. Wait until it plateaus.

2. **Switch to `/display`.** The **Human heatmap** layer is on by default, so
   the room's answer appears.

3. **Turn on Individual paths** in `/control` for the second reveal — the same
   data, but showing that order exists, not just location.

4. **Turn both off, turn on Model scanpath.** This is your "clear" — it hides
   layers, it never deletes. The responses are still there and come back when
   you switch the layer on again.

   Turn on **Show the prompt** here so the audience sees what the model was
   asked, not just the dots it produced.

5. **Turn the heatmap back on** to show both together.

6. **Turn on Agreement metrics.** The panel appears under the image. Nothing to
   run — it computes from whatever is in the database right now.

7. **Next image:** pick it from the dropdown in `/control`. That opens a new
   round, participants' phones follow within a few seconds, and the display
   starts empty. The previous round is kept.

---

## 4b. Task probes

`/control` offers a **Task** dropdown in two groups.

**Trained — validated.** *Free viewing* and *Cars*. These use the exact
templates the adapter was fine-tuned on. `car` is one of the 18 COCO-Search18
targets, so it is genuinely on-distribution.

**Experimental — off-distribution.** *What shouldn't be here*, *People*,
*Roads*, *Count buildings*, *Find living things*, *Danger*, *Music*, *Robots*.
The adapter never saw these tasks. The model still returns coordinates — it
always does — but the quality is unvalidated, and there is no error to tell you
when it is wrong.

Each probe keeps the trained template's structure byte-for-byte and substitutes
only the task clause. A bare instruction like "danger" would not produce
coordinates at all; the model has to be told the output format.

**Use *Cars* as your control.** It is the one experimental-looking option that
is actually trained, so comparing it against *Danger* or *Robots* on the same
image shows the audience the difference between a validated prediction and a
plausible-looking guess. *Music* and *Robots* are useful negative controls on a
street scene — nothing there matches, so watch what the model does with an
impossible task.

Generate probe runs ahead of time:

```bash
python tools/precompute.py --repo ../DeepGaze3.5-VL --all-probes \
    --num-fixations 5 --samples 10 --temperature 0.7
python tools/precompute.py --list-probes      # see the catalogue
```

## 4c. Showing the prompt and the raw data

Two more toggles in `/control`, both for teaching:

**Show the prompt** puts the full text the model was given under the image,
with a green *Trained template* or amber *Experimental* badge and a one-line
note. It shows the prompt the displayed run actually used, which can differ
from the current selection if you changed it after the run was made.

**Raw data (JSON)** opens a full-screen overlay with four tabs — Model run,
Human taps, Agreement, Display state — showing exactly what the display is
drawn from. Useful for making the point that the picture is computed, not
illustrated. It refreshes while open, so taps arriving live appear in it.

### What each overlay looks like

One identity per source, the same three colours as the charts:

| Layer | On the picture |
|---|---|
| **Tapped — heatmap** | blue cloud: where people *said* they would look |
| **Tapped — scanpaths** | blue lines, first tap ringed pale |
| **Measured — heatmap** | orange **rings**, tighter rings = more looking |
| **Measured — scanpaths** | orange traces, start ringed pale |
| **Model — scanpath** | green numbered path, faint green ghosts behind it |

Measured gaze is drawn as rings rather than a second cloud on purpose: two
filled clouds on one picture hide each other, and a different shape reads as a
different kind of thing from the back of a room.

**Say out loud that blue and orange are different people.** Nobody both tapped
and was eye-tracked — tapping first would change where you then look. The
caption on screen says it, but an audience hears it better than it reads it.

---

## 5. Reading the analysis panel

**This panel is the tap group vs the model. Measured gaze is not in it.** It
is computed from taps, and always has been. The panel says so on screen. For
all three sources together, use the charts in section 5b.

The headline compares the model against the **tap-group-to-itself ceiling**,
not against perfection — neither side is ground truth, so "as well as those
people agree with each other" is the honest claim.

| Metric | Means |
|---|---|
| **Spearman ρ** | Do they emphasise the same areas? Compare it against the random and centre-bias figures underneath, not against 1.0 |
| **Tap group vs itself (ceiling)** | Split the tap group in half, score one against the other. This is the realistic maximum |
| **Order similarity** | Do they visit areas in the same sequence? Compare against the "people agree" figure beside it |
| **NSS** | Standard saliency measure. 0 is chance |
| **Areas of interest** | Clustered from pooled human *and* model points, so neither side defines its own yardstick |

**It needs 12 responses** before the ceiling appears; below that the panel says
so. With fewer than that the ceiling is noise, not a number.

If the model and the room disagree, that is a finding, not a failure. The model
was trained on real eye-tracking; the taps were guesses about one's own gaze.
Disagreement is evidence people are poor introspectors about where they look.

---

## 5b. The comparison charts

**`/charts`** — open it in a second tab or on a second screen. It is the
between-groups view: where the tappers said they would look, where the
tracked group actually looked, and where the model says they should have,
as three grids on one colour scale.

Read it in this order:

1. **The three panels.** Same scale, same blur, so a darker cell always means
   more attention. Each panel is averaged per person, so someone who produced
   thirty gaze samples does not outweigh someone who produced nine.
2. **The difference map.** Measured minus tapped. Red is where people looked
   more than they said; blue is where they said more than they looked. This
   is the picture the talk rests on.
3. **The dot plot.** The same numbers with 95% intervals. Where two groups'
   intervals do not overlap, the gap is unlikely to be noise. Wide intervals
   mean few people, not a strong result.
4. **The agreement table.** Each pair's correlation against the ceiling — how
   well a group agrees with *itself*, split in half. The "% of ceiling" column
   is the honest figure; the raw correlation alone is not.

It refreshes every five seconds, so it can be left open while the session
runs. It says so on the page when the model run is synthetic, when a group has
fewer than four people, and how many calibrations were excluded.

**To project it**, turn on **Comparison charts** under *Full screen* in
Controls. The display then shows the panels, the difference map and one
headline sentence at projector size — no table, no dot plot, nothing to hover.
Charts and Raw data (JSON) are mutually exclusive: turning one on turns the
other off, because each takes over the whole screen. Turn it off and the image
comes back with its overlays exactly as you left them.

`?grid=4` or `?sigma=0.25` change the grid and the blur if you want to see how
sensitive the picture is. **Do not present a grid finer than 3 × 3**: webcam
gaze on a phone is accurate to about a fifth of the screen width, which is
three resolvable columns and no more.

---

## 6. Live inference during the talk (optional)

To run the model in front of the audience, from the Mac:

```bash
export SCANPATH_URL=https://YOUR-CODESPACE-8000.app.github.dev
export SCANPATH_TOKEN=your-control-token

python tools/push_result.py --repo ../DeepGaze3.5-VL \
    --image data/images/street.jpg --mode freeview --num-fixations 5
```

It computes locally and pushes; the display picks it up within two seconds.
Takes 1–3 minutes — narrate over it, and keep the precomputed run on screen as
the fallback.

---

## 6b. Phones are less accurate than laptops, and that is geometry

Expect measured gaze on a phone to be roughly **two and a half times less
accurate than on a laptop**, as a fraction of the picture. It is not the
camera, and it is not a bug.

```bash
python tools/device_report.py
```

Prints the calibration error your own sessions recorded, split by device, plus
the geometry that explains it. In short: the tracker's error is **angular** —
it guesses a direction, and the library returns a point normalised to the
screen with no physical size anywhere in it, so all device geometry is
absorbed by the nine-point calibration. A 4:3 picture on a 13" laptop at 55cm
subtends about 25°; the same picture on an iPhone 12 mini at 30cm subtends
about 11°. The same angular error therefore covers about 22% of the phone
picture and about 10% of the laptop one.

The phone camera is not the limit: at 30cm the face fills *more* of the frame
than at 55cm on a laptop, so the eyes are sampled by more pixels, not fewer.

What follows from it:

- **Do not present a grid finer than 3 × 3.** This is the same constraint as
  before, and this is where it comes from.
- Landscape helps a little — the picture fills more of the screen, about 17%
  error instead of 22% — but it is not a fix, and it costs vertical room:
  a phone in landscape leaves roughly 290px once Safari's bars are counted.
- **Do not rotate the phone mid-run.** The calibration is fitted to one
  screen shape, so turning the device invalidates it. The app now pauses and
  asks for it to be turned back rather than recording samples that would look
  like data.
- If some participants can use laptops, their gaze data is materially better.
  Worth knowing when reading a thin measured group.

---

## 7. After the session — do not skip

```bash
python tools/export.py --token YOUR_CONTROL_TOKEN
```

Writes `data/exports/session-TIMESTAMP.json` and a CSV of every tap. **Commit
them.** A Codespace is eventually deleted with its database inside, and
participant responses are the only thing here that cannot be regenerated.

**Commit real model runs too.** `data/model/*.json` used to be gitignored on
the grounds that runs are regenerable. That holds for synthetic placeholders,
which take seconds. It does not hold for a real run: an hour or more of MPS
time against weights this Codespace does not have, so the JSON is the only
artifact of that compute. Check what you have before a session:

```bash
python3 -c "
import json,glob
for f in sorted(glob.glob('data/model/*.json')):
    d=json.load(open(f))
    print(f.split('/')[-1], d.get('source'), 'samples:', len(d.get('samples_norm') or []))
"
```

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Participants see a GitHub login | Port 8000 is not Public. Ports panel |
| `/qr` shows a red warning | The URL resolves only on this machine. Same fix, or set `SCANPATH_PUBLIC_URL` |
| Red SYNTHETIC banner | Placeholder data is loaded. `rm data/model/*.json`, precompute for real, reload |
| Amber EXPERIMENTAL strip | That run used a custom prompt, off the trained templates |
| "No run for this image and task" | Nothing precomputed for that image/mode. Run `precompute.py`, then **Reload runs from disk** |
| Analysis says "no human responses yet" | Nobody has submitted in the current round |
| Ceiling shows "—" | Fewer than 12 responses |
| Display frozen | Check the terminal is still running and the Codespace has not idled out |
| Not sure what is running | Open `/start` — it shows live status and warns about anything not ready |
| "Unknown layer" or a control that does nothing | The server is running older code than the pages. Restart it — see above |
| "Recorded, but not saved" | Should no longer happen: the upload now falls back to the participant id. If it does, the device has no calibration on record at all |
| Codespace idled out | Restart it (~30s). Data and URL survive. **Re-check the port is Public** |
| Phones show an old image | They poll every few seconds; give it a moment |

Raise the Codespaces idle timeout to 240 minutes in your personal settings
before the session, and keep the Codespace browser tab open throughout.
