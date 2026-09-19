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
    ┌────────┬───┴────┬──────────┬─────────┐
    │        │        │          │         │
    /      /qr    /display   /control   /admin
 phones  project  project    laptop     laptop
```

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

Leave that terminal running. It prints the control token and the URLs:

```
  images found:   2
  model runs:     2

  participant:    /
  presenter:      /display
  controls:       /control?k=C5ufh8ZJ4MwP
  admin:          /admin?k=C5ufh8ZJ4MwP

  control token:  C5ufh8ZJ4MwP
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

5. **Turn the heatmap back on** to show both together.

6. **Turn on Agreement metrics.** The panel appears under the image. Nothing to
   run — it computes from whatever is in the database right now.

7. **Next image:** pick it from the dropdown in `/control`. That opens a new
   round, participants' phones follow within a few seconds, and the display
   starts empty. The previous round is kept.

---

## 5. Reading the analysis panel

The headline compares the model against the **human-to-human ceiling**, not
against perfection — neither side is ground truth, so "as well as people agree
with each other" is the honest claim.

| Metric | Means |
|---|---|
| **Spearman ρ** | Do they emphasise the same areas? Compare it against the random and centre-bias figures underneath, not against 1.0 |
| **Human-to-human ceiling** | Split the room in half, score one against the other. This is the realistic maximum |
| **Order similarity** | Do they visit areas in the same sequence? Compare against the "people agree" figure beside it |
| **NSS** | Standard saliency measure. 0 is chance |
| **Areas of interest** | Clustered from pooled human *and* model points, so neither side defines its own yardstick |

**It needs 12 responses** before the ceiling appears; below that the panel says
so. With fewer than that the ceiling is noise, not a number.

If the model and the room disagree, that is a finding, not a failure. The model
was trained on real eye-tracking; the taps were guesses about one's own gaze.
Disagreement is evidence people are poor introspectors about where they look.

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

## 7. After the session — do not skip

```bash
python tools/export.py --token YOUR_CONTROL_TOKEN
```

Writes `data/exports/session-TIMESTAMP.json` and a CSV of every tap. **Commit
them.** A Codespace is eventually deleted with its database inside, and
participant responses are the only thing here that cannot be regenerated.

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
| Codespace idled out | Restart it (~30s). Data and URL survive. **Re-check the port is Public** |
| Phones show an old image | They poll every few seconds; give it a moment |

Raise the Codespaces idle timeout to 240 minutes in your personal settings
before the session, and keep the Codespace browser tab open throughout.
