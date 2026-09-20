# Vendored WebEyeTrack 0.0.2

Source: https://github.com/RedForestAi/WebEyeTrack (MIT) · npm `webeyetrack@0.0.2`

## Why vendored rather than installed

Three reasons, in order of how much they forced the decision:

1. **The published npm package is incomplete.** `dist/index.js` references
   `index.worker.js`, which is emitted by the upstream webpack build and was
   not included in the tarball. `WebEyeTrackProxy` — the worker-based class the
   upstream README documents — therefore cannot run from npm as published.
   We use the `WebEyeTrack` class instead, which runs on the main thread and
   needs no worker. At ~2.4 ms per frame (per the paper, iPhone 14) that is
   affordable.

2. **This app has no bundler**, by design. `dist/index.js` is a self-contained
   UMD bundle with TensorFlow.js inlined, so a plain `<script>` tag works.

3. **Venue wifi.** A pinned local copy cannot fail on the day, and 0.0.2 is an
   early version that could change under us.

## Files

| Path | What |
|---|---|
| `webeyetrack.umd.js` | The UMD bundle, 2.7 MB, unmodified |
| `index.js.LICENSE.txt` | Upstream's bundled dependency licences |
| `LICENSE` | WebEyeTrack's MIT licence |
| `../web/model.json`, `../web/group1-shard1of1.bin` | BlazeGaze weights (669 KB), from `js/examples/minimal-example/public/web/` in the repo. **The bundle hard-codes the path `/web/model.json`**, so these must be served at that exact URL |

## Still fetched from the internet at runtime

Vendoring is not complete. The bundle still pulls MediaPipe from two CDNs:

- `cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm`
- `storage.googleapis.com/mediapipe-models/face_landmarker/...`

**If the venue's wifi blocks either, tracking will not start.** Test on the
actual network. Vendoring these too is a known follow-up.

## Updating

Check whether upstream has shipped the worker bundle; if so, `WebEyeTrackProxy`
becomes usable and moves inference off the UI thread.
