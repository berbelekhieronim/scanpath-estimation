/* Gaze tracker adapter.
 *
 * Everything downstream depends on this interface, never on a tracker
 * library. WebEyeTrack is a 0.0.2 release and the first real-room test may
 * send us to WebGazer or elsewhere; when it does, that should cost one
 * backend file, not a rewrite.
 *
 * A backend emits samples shaped:
 *   { x, y, t, state, ok }
 *      x, y   viewport-relative, 0..1, top-left origin
 *      t      ms since tracking started
 *      state  'open' | 'closed'   (closed = blink; position is unreliable)
 *      ok     false when no face was found this frame
 */

export const STATUS = {
  IDLE: 'idle',
  LOADING: 'loading',       // fetching model weights
  PERMISSION: 'permission', // waiting on the camera prompt
  RUNNING: 'running',
  STOPPED: 'stopped',
  ERROR: 'error',
};

export class GazeTrackerBase {
  constructor() {
    this.status = STATUS.IDLE;
    this.error = null;
    this.onSample = () => {};
    this.onStatus = () => {};
  }

  _setStatus(status, error = null) {
    this.status = status;
    this.error = error;
    this.onStatus(status, error);
  }

  async start() { throw new Error('not implemented'); }
  stop() { throw new Error('not implemented'); }

  /** Teach the tracker that the eye is on (x, y), both 0..1 top-left. */
  async calibratePoint(_x, _y) { throw new Error('not implemented'); }
}


/* ------------------------------------------------------------------ *
 * WebEyeTrack backend
 * ------------------------------------------------------------------ */

export class WebEyeTrackBackend extends GazeTrackerBase {
  /** @param {HTMLVideoElement} videoEl - must already be in the DOM with an id */
  constructor(videoEl, {
    scriptUrl = '/static/vendor/webeyetrack/webeyetrack.umd.js',
    maxPoints = 9,
    backend = 'auto',
  } = {}) {
    super();
    this.videoEl = videoEl;
    this.scriptUrl = scriptUrl;
    // WebEyeTrack's maxPoints DEFAULTS TO 5, and pruneCalibData() keeps only
    // the most recent that many. A nine-point calibration built on the
    // default would silently throw away the first four points and report a
    // confident, wrong model. Verified in upstream's WebEyeTrack.ts.
    //
    // Set to exactly the number of calibration points, not more. Every
    // retained point holds a 512x128x3 eye patch and adapt() concatenates all
    // of them on each call, so a generous value buys nothing and costs real
    // memory — which on iOS Safari is how a tab gets killed.
    this.maxPoints = maxPoints;
    this.backendPref = backend;
    this.backend = null;
    this.name = 'webeyetrack';
    this.version = '0.0.2';
    this._t0 = 0;
    this._lastCalibAt = 0;
    this._lastCalibPt = null;
  }

  async _loadLibrary() {
    // The UMD wrapper copies each export straight onto the global object
    // rather than under a namespace, so the classes arrive as window.WebEyeTrack,
    // window.WebcamClient and so on. Verified by loading the bundle and
    // diffing Object.keys(window).
    const pick = () => (window.WebEyeTrack && window.WebcamClient
      ? { WebEyeTrack: window.WebEyeTrack, WebcamClient: window.WebcamClient }
      : null);

    const already = pick();
    if (already) return already;

    await new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = this.scriptUrl;
      s.onload = resolve;
      s.onerror = () => reject(new Error(`Could not load ${this.scriptUrl}`));
      document.head.appendChild(s);
    });

    const lib = pick();
    if (!lib) {
      throw new Error('The tracker bundle loaded but did not define WebEyeTrack ' +
                      'and WebcamClient on window — it may be the wrong file.');
    }
    return lib;
  }

  /* Restart the camera WITHOUT rebuilding the model.
   *
   * This is the important one. Constructing WebEyeTrack again loads another
   * BlazeGaze model and another MediaPipe FaceLandmarker, and upstream
   * exposes no dispose() for either — so a retry used to double the memory
   * held, and a third attempt tripled it. That is why it survived one
   * calibration and died on the next.
   */
  async restart() {
    if (!this.wet) return this.start();
    this.resetCalibration();
    try { this.cam && this.cam.stopWebcam(); } catch {}
    this._setStatus(STATUS.PERMISSION);
    const lib = await this._loadLibrary();
    this.cam = new lib.WebcamClient(this.videoEl.id);
    this._t0 = performance.now();
    this._lastCalibAt = 0;
    this._lastCalibPt = null;
    await this.cam.startWebcam(this._frameHandler);
    this._setStatus(STATUS.RUNNING);
  }

  /** Clear the fitted calibration so a retry starts clean. */
  resetCalibration() {
    this.releaseCalibrationMemory();
    try {
      if (this.wet) {
        this.wet.latestMouseClick = null;
        if (this.wet.affineMatrix && this.wet.affineMatrix.dispose) {
          this.wet.affineMatrix.dispose();
        }
        this.wet.affineMatrix = null;
      }
    } catch {}
  }

  /* Last-resort teardown. Only for leaving the page — after this the tracker
   * cannot be restarted without a full reload. */
  destroy() {
    this.stop();
    try {
      const bg = this.wet && this.wet.blazeGaze;
      if (bg && bg.model && bg.model.dispose) bg.model.dispose();
    } catch {}
    try {
      const fl = this.wet && this.wet.faceLandmarkerClient;
      if (fl && fl.faceLandmarker && fl.faceLandmarker.close) fl.faceLandmarker.close();
    } catch {}
    this.wet = null;
    this.cam = null;
  }

  async start() {
    try {
      this._setStatus(STATUS.LOADING);
      const lib = await this._loadLibrary();

      // WebEyeTrack (main thread), not WebEyeTrackProxy: the worker bundle
      // the Proxy needs is missing from the published package. See
      // vendor/webeyetrack/VENDORED.md.
      await this._selectBackend();

      this.wet = new lib.WebEyeTrack(this.maxPoints);
      await this.wet.initialize();   // BlazeGaze weights + MediaPipe FaceLandmarker

      this._setStatus(STATUS.PERMISSION);
      this.cam = new lib.WebcamClient(this.videoEl.id);
      this._t0 = performance.now();

      this._frameHandler = async (frame, timestamp) => {
        if (this.status === STATUS.STOPPED) return;
        let result;
        try {
          result = await this.wet.step(frame, timestamp);
        } catch {
          this.onSample({ x: null, y: null, t: performance.now() - this._t0,
                          state: 'open', ok: false });
          return;
        }
        this.onSample(this._toSample(result));
      };
      await this.cam.startWebcam(this._frameHandler);

      this._setStatus(STATUS.RUNNING);
    } catch (err) {
      this._setStatus(STATUS.ERROR, describeError(err));
      throw err;
    }
  }

  _toSample(result) {
    const t = performance.now() - this._t0;
    if (!result || !result.normPog) {
      return { x: null, y: null, t, state: 'open', ok: false };
    }
    // normPog is screen-relative in [-0.5, 0.5] with the origin at screen
    // centre and +y downwards. Converting to a 0..1 top-left fraction is the
    // arithmetic below; mapping screen space onto viewport space exactly is
    // not possible from JS on mobile, so this assumes the page fills the
    // screen. On a phone in normal use that is close to true, and the
    // residual offset is exactly what calibration (W2) learns out.
    const [nx, ny] = result.normPog;
    return {
      x: nx + 0.5,
      y: ny + 0.5,
      t,
      state: result.gazeState || 'open',
      ok: true,
    };
  }

  /* Upstream's handleClick() silently drops a point if it lands within
   * 1000 ms OR within 0.05 units of the previous one. Both are easy to trip
   * with an impatient participant, and a dropped point is invisible — so the
   * caller is told whether the point was actually taken. */
  async calibratePoint(x, y) {
    if (!this.wet || !this.wet.latestGazeResult) {
      return { accepted: false, reason: 'no face detected right now' };
    }
    const now = Date.now();
    const nx = x - 0.5, ny = y - 0.5;   // to the tracker's centre-origin space

    if (now - this._lastCalibAt < 1100) {
      return { accepted: false, reason: 'too soon after the previous point' };
    }
    if (this._lastCalibPt &&
        Math.abs(nx - this._lastCalibPt[0]) < 0.06 &&
        Math.abs(ny - this._lastCalibPt[1]) < 0.06) {
      return { accepted: false, reason: 'too close to the previous point' };
    }

    await this.wet.handleClick(nx, ny);
    this._lastCalibAt = now;
    this._lastCalibPt = [nx, ny];
    return { accepted: true, points: this.wet.calibData
      ? this.wet.calibData.supportX.length : null };
  }

  /* Choose the TF.js backend before the model loads.
   *
   * A real iPhone killed the tab mid-calibration holding only 26 MB of
   * tensors — far too little for a JS-heap exhaustion. On the WebGL backend
   * every tensor is a GPU texture, and a training step over a batch of
   * 512x128x3 eye patches allocates many transient ones; iOS Safari's texture
   * budget is much tighter than its heap. The CPU backend uses plain typed
   * arrays, so that entire failure mode disappears.
   *
   * The weights and the arithmetic are identical either way; only speed
   * differs, and BlazeGaze is small enough (0.15 GFLOPs) that CPU is viable.
   * Reliability beats frame rate here, so iOS gets CPU by default.
   */
  async _selectBackend() {
    const engine = window._tfengine;
    if (!engine || typeof engine.setBackend !== 'function') return;

    let want = this.backendPref;
    if (want === 'auto') want = isIOS() ? 'cpu' : 'webgl';
    if (want === 'default') { this.backend = engine.backendName; return; }

    try {
      const available = Object.keys(engine.registryFactory || {});
      if (!available.includes(want)) { this.backend = engine.backendName; return; }
      await engine.setBackend(want);
      this.backend = engine.backendName;
    } catch {
      this.backend = engine.backendName || null;
    }
  }

  /** TF.js tensor counts, for spotting a leak from outside the library. */
  memory() {
    try {
      const tf = window.tf || (window._tfengine && window._tfengine.registry && window.tf);
      if (tf && tf.memory) {
        const m = tf.memory();
        return { numTensors: m.numTensors, mb: +(m.numBytes / 1048576).toFixed(1) };
      }
      if (window._tfengine && window._tfengine.state) {
        return { numTensors: window._tfengine.state.numTensors,
                 mb: +(window._tfengine.state.numBytes / 1048576).toFixed(1) };
      }
    } catch {}
    return null;
  }

  /* Free the retained calibration tensors once calibration is finished.
   *
   * adapt() keeps every support point — an eye patch, a head vector and a
   * face origin per calibration point — and upstream disposes almost none of
   * them (two tf.dispose calls against eleven tensor creations). They are
   * only needed to adapt again; the fitted transform itself is already
   * applied. Holding roughly 7MB of WebGL textures for no reason is how an
   * iOS tab gets killed right after calibration.
   *
   * Only safe once no further calibration will happen.
   */
  releaseCalibrationMemory() {
    const before = this.memory();
    try {
      const cd = this.wet && this.wet.calibData;
      if (cd) {
        (cd.supportX || []).forEach((s) => {
          ['eyePatches', 'headVectors', 'faceOrigins3D'].forEach((k) => {
            try { s && s[k] && s[k].dispose && s[k].dispose(); } catch {}
          });
        });
        (cd.supportY || []).forEach((y) => {
          try { y && y.dispose && y.dispose(); } catch {}
        });
        cd.supportX = []; cd.supportY = [];
        cd.timestamps = []; cd.ptType = [];
      }
    } catch {}
    return { before, after: this.memory() };
  }

  stop() {
    try { this.cam && this.cam.stopWebcam(); } catch {}
    // Belt and braces: WebcamClient should release the stream, but a camera
    // left live after a demo is the one bug nobody forgives.
    try {
      const s = this.videoEl && this.videoEl.srcObject;
      if (s && s.getTracks) s.getTracks().forEach((t) => t.stop());
      if (this.videoEl) this.videoEl.srcObject = null;
    } catch {}
    this._setStatus(STATUS.STOPPED);
  }
}


/* ------------------------------------------------------------------ *
 * Mock backend
 * ------------------------------------------------------------------ */

export class MockBackend extends GazeTrackerBase {
  /* Pointer-driven, no camera.
   *
   * Not a toy: it lets calibration, sampling, storage and the analysis
   * (phases W2-W5) be built and tested without waiting on camera hardware or
   * a real face, and it gives a deterministic target for automated tests.
   * Noise is injected at roughly the error the real tracker reports, so
   * anything built against it meets realistic data rather than a clean signal.
   */
  constructor({ noise = 0.035, hz = 30, bias = [0.09, -0.06] } = {}) {
    super();
    this.name = 'mock';
    this.version = '1';
    this.noise = noise;
    this.hz = hz;
    // A systematic offset that calibration removes, so the quality gate is
    // exercised against something that actually improves rather than a
    // signal that was already perfect.
    this.bias = bias.slice();
    this._calibrated = 0;
    this._pos = { x: 0.5, y: 0.5 };
    this._onMove = (e) => {
      const p = e.touches ? e.touches[0] : e;
      this._pos = {
        x: p.clientX / window.innerWidth,
        y: p.clientY / window.innerHeight,
      };
    };
  }

  async start() {
    this._setStatus(STATUS.LOADING);
    window.addEventListener('pointermove', this._onMove, { passive: true });
    window.addEventListener('touchmove', this._onMove, { passive: true });
    this._t0 = performance.now();
    this._timer = setInterval(() => {
      const g = () => (Math.random() + Math.random() + Math.random() - 1.5) * this.noise;
      const shrink = Math.max(0, 1 - this._calibrated / 9);
      this.onSample({
        x: Math.min(1, Math.max(0, this._pos.x + g() + this.bias[0] * shrink)),
        y: Math.min(1, Math.max(0, this._pos.y + g() + this.bias[1] * shrink)),
        t: performance.now() - this._t0,
        state: 'open',
        ok: true,
      });
    }, 1000 / this.hz);
    this._setStatus(STATUS.RUNNING);
  }

  async calibratePoint() {
    this._calibrated += 1;
    return { accepted: true, points: this._calibrated };
  }

  async restart() { this._calibrated = 0; return this.start(); }
  resetCalibration() { this._calibrated = 0; }
  destroy() { this.stop(); }

  memory() { return { numTensors: 0, mb: 0 }; }
  get backend() { return 'mock'; }
  set backend(_v) { /* mock has no backend to set */ }
  releaseCalibrationMemory() { return { before: this.memory(), after: this.memory() }; }

  stop() {
    clearInterval(this._timer);
    window.removeEventListener('pointermove', this._onMove);
    window.removeEventListener('touchmove', this._onMove);
    this._setStatus(STATUS.STOPPED);
  }
}


/* ------------------------------------------------------------------ */

/* The tracker's dependencies, checked before starting so a failure names a
 * cause instead of surfacing as a stack trace mid-demo. The two remote ones
 * are the reason the spec says to test on the venue's actual wifi. */
export const DEPENDENCIES = [
  { name: 'Tracker bundle', url: '/static/vendor/webeyetrack/webeyetrack.umd.js', local: true },
  { name: 'Gaze model weights', url: '/web/model.json', local: true },
  { name: 'MediaPipe runtime (jsdelivr)', local: false,
    url: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm/vision_wasm_internal.js' },
  { name: 'Face model (Google)', local: false,
    url: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' },
];

/** Check each dependency is reachable. Returns one result per dependency. */
export async function preflight() {
  return Promise.all(DEPENDENCIES.map(async (dep) => {
    try {
      // no-cors still distinguishes "reachable" from "blocked" for the
      // cross-origin ones, which is all we need.
      await fetch(dep.url, { method: 'GET', mode: dep.local ? 'cors' : 'no-cors',
                             cache: 'no-store' });
      return { ...dep, ok: true };
    } catch (err) {
      return { ...dep, ok: false, detail: String(err && err.message || err) };
    }
  }));
}

export function isIOS() {
  const ua = navigator.userAgent || '';
  return /iPad|iPhone|iPod/.test(ua) ||
         (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
}

export function describeError(err) {
  // A failed script or resource load throws an Event, not an Error, and
  // String(Event) is "[object Event]" — useless in front of an audience.
  let msg;
  if (err instanceof Event || (err && err.type && err.target)) {
    const src = (err.target && (err.target.src || err.target.currentSrc)) || '';
    msg = `A resource failed to load${src ? `: ${src}` : ''}`;
    if (/jsdelivr|googleapis/.test(src) || !src) {
      return 'Could not fetch the MediaPipe runtime. The tracker needs ' +
             'cdn.jsdelivr.net and storage.googleapis.com — this network appears ' +
             'to block one of them. Run the dependency check below.';
    }
  } else {
    msg = String((err && err.message) || err);
  }
  if (msg === '[object Object]' || msg === 'undefined') {
    msg = 'The tracker failed to start and gave no reason. Run the dependency check below.';
  }
  if (/NotAllowedError|Permission denied/i.test(msg)) {
    return 'Camera permission was denied. Allow it in the browser’s site settings and reload.';
  }
  if (/NotFoundError|no camera/i.test(msg)) {
    return 'No camera was found on this device.';
  }
  if (/NotReadableError|in use/i.test(msg)) {
    return 'The camera is already in use by another app. Close it and reload.';
  }
  if (/getUserMedia|mediaDevices/i.test(msg)) {
    return 'This browser will not give camera access. It needs HTTPS — a Codespaces ' +
           'forwarded URL qualifies, plain http on a LAN address does not.';
  }
  if (/Could not load|exported nothing/i.test(msg)) {
    return msg + ' The tracker bundle may not be served correctly.';
  }
  if (/fetch|network|Failed to fetch|Load failed|ERR_/i.test(msg)) {
    return 'A network request failed. The tracker fetches MediaPipe from ' +
           'cdn.jsdelivr.net and storage.googleapis.com — this network may ' +
           'block them. Run the dependency check below.';
  }
  return msg;
}

export function createTracker(kind, videoEl, opts) {
  if (kind === 'mock') return new MockBackend(opts);
  return new WebEyeTrackBackend(videoEl, opts);
}
