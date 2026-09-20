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
}


/* ------------------------------------------------------------------ *
 * WebEyeTrack backend
 * ------------------------------------------------------------------ */

export class WebEyeTrackBackend extends GazeTrackerBase {
  /** @param {HTMLVideoElement} videoEl - must already be in the DOM with an id */
  constructor(videoEl, { scriptUrl = '/static/vendor/webeyetrack/webeyetrack.umd.js' } = {}) {
    super();
    this.videoEl = videoEl;
    this.scriptUrl = scriptUrl;
    this.name = 'webeyetrack';
    this.version = '0.0.2';
    this._t0 = 0;
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

  async start() {
    try {
      this._setStatus(STATUS.LOADING);
      const lib = await this._loadLibrary();

      // WebEyeTrack (main thread), not WebEyeTrackProxy: the worker bundle
      // the Proxy needs is missing from the published package. See
      // vendor/webeyetrack/VENDORED.md.
      this.wet = new lib.WebEyeTrack();
      await this.wet.initialize();   // BlazeGaze weights + MediaPipe FaceLandmarker

      this._setStatus(STATUS.PERMISSION);
      this.cam = new lib.WebcamClient(this.videoEl.id);
      this._t0 = performance.now();

      await this.cam.startWebcam(async (frame, timestamp) => {
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
      });

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
  constructor({ noise = 0.035, hz = 30 } = {}) {
    super();
    this.name = 'mock';
    this.version = '1';
    this.noise = noise;
    this.hz = hz;
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
      this.onSample({
        x: Math.min(1, Math.max(0, this._pos.x + g())),
        y: Math.min(1, Math.max(0, this._pos.y + g())),
        t: performance.now() - this._t0,
        state: 'open',
        ok: true,
      });
    }, 1000 / this.hz);
    this._setStatus(STATUS.RUNNING);
  }

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
