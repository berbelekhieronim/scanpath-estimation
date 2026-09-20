/* Nine-point calibration with four-point validation.
 *
 * Deliberately knows nothing about a tracker beyond the adapter interface, so
 * it can be driven by the mock backend in tests and by a real camera in the
 * room without changing.
 *
 * Coordinates throughout are 0..1 of the viewport, top-left origin.
 */

// 3x3 calibration grid, inset from the edges: gaze at the very corner of a
// screen is an extreme eye rotation and the worst-estimated part of the range.
export const CALIB_POINTS = [
  [0.15, 0.15], [0.50, 0.15], [0.85, 0.15],
  [0.15, 0.50], [0.50, 0.50], [0.85, 0.50],
  [0.15, 0.85], [0.50, 0.85], [0.85, 0.85],
];

// Validation points sit between the calibration points, never on them —
// scoring on a point the model was trained on measures memorisation.
export const VALIDATION_POINTS = [
  [0.32, 0.32], [0.68, 0.68], [0.68, 0.32],
];

/* Thresholds in fractions of viewport width.
 *
 * These are starting values, not measurements. The literature puts webcam
 * error near 4 degrees, which on a phone is about a third of the screen, so
 * "good" here is already coarse. Tune them on real sessions before trusting
 * the gate — see SPEC-WEBCAM.md section 5.1. */
export const QUALITY = {
  GOOD: 0.18,
  USABLE: 0.30,
};

export const PHASE = {
  IDLE: 'idle',
  WAITING_FOR_FACE: 'waiting_for_face',
  CALIBRATING: 'calibrating',
  VALIDATING: 'validating',
  DONE: 'done',
  FAILED: 'failed',
};

export function gradeError(err) {
  if (err == null) return 'failed';
  if (err <= QUALITY.GOOD) return 'good';
  if (err <= QUALITY.USABLE) return 'usable';
  return 'poor';
}

export class Calibration {
  /**
   * @param {object} tracker   a started GazeTrackerBase
   * @param {object} opts
   *   onPhase(phase, info)    phase changes
   *   onPoint(index, total, xy, kind)  a target should be shown
   *   onProgress(info)        per-target feedback
   *   settleMs                pause after a calibration tap before adapting
   *   sampleMs                length of the pre-tap window scored at a
   *                           validation point
   */
  constructor(tracker, opts = {}) {
    this.tracker = tracker;
    this.onPhase = opts.onPhase || (() => {});
    this.onPoint = opts.onPoint || (() => {});
    this.onProgress = opts.onProgress || (() => {});
    this.settleMs = opts.settleMs ?? 150;
    this.sampleMs = opts.sampleMs ?? 700;
    // If samples stop arriving the run must end with a message, never hang.
    // Upstream's frame loop exits permanently when the video element pauses —
    // which mobile browsers do to off-screen video — so this is a real state,
    // not a defensive nicety.
    this.stallMs = opts.stallMs ?? 15000;
    this.shuffle = opts.shuffle !== false;

    this.phase = PHASE.IDLE;
    this.samples = [];
    this.accepted = 0;
    this.rejected = [];
    this.validation = [];
    this._resolveTap = null;

    this.lastSampleAt = 0;
    this.lastFaceAt = 0;
    this.framesSeen = 0;
    this.facesSeen = 0;

    this._onSample = (s) => {
      this.framesSeen++;
      this.lastSampleAt = performance.now();
      if (s.ok) { this.facesSeen++; this.lastFaceAt = this.lastSampleAt; }
      if (s.ok && s.x != null && s.state !== 'closed') this.samples.push(s);
      if (this.samples.length > 400) this.samples.shift();
    };
  }

  _setPhase(phase, info) {
    this.phase = phase;
    this.onPhase(phase, info);
  }

  /** The page calls this when the participant taps the shown target. */
  tap() {
    if (this._resolveTap) { this._resolveTap(); this._resolveTap = null; }
  }

  _awaitTap() {
    return new Promise((resolve, reject) => {
      this._resolveTap = resolve;
      const check = setInterval(() => {
        if (!this._resolveTap) { clearInterval(check); return; }
        const since = performance.now() - this.lastFaceAt;
        if (this.lastFaceAt && since > this.stallMs) {
          clearInterval(check);
          this._resolveTap = null;
          reject(new StallError('tracking stopped'));
        }
      }, 1000);
      const orig = this._resolveTap;
      this._resolveTap = () => { clearInterval(check); orig(); };
    });
  }

  diagnostics() {
    return {
      framesSeen: this.framesSeen,
      facesSeen: this.facesSeen,
      faceRate: this.framesSeen ? this.facesSeen / this.framesSeen : 0,
      msSinceFace: this.lastFaceAt ? Math.round(performance.now() - this.lastFaceAt) : null,
      phase: this.phase,
      accepted: this.accepted,
      rejected: this.rejected.length,
    };
  }

  _recentMean(sinceMs) {
    const cut = performance.now() - sinceMs;
    const pts = this.samples.filter((s) => s.wall >= cut);
    if (!pts.length) return null;
    return [
      pts.reduce((a, s) => a + s.x, 0) / pts.length,
      pts.reduce((a, s) => a + s.y, 0) / pts.length,
    ];
  }

  async run() {
    const prevHandler = this.tracker.onSample;
    this.tracker.onSample = (s) => {
      s.wall = performance.now();
      this._onSample(s);
      prevHandler && prevHandler(s);
    };

    try {
      return await this._run();
    } catch (err) {
      if (err instanceof StallError) {
        this._setPhase(PHASE.FAILED, { reason: 'stalled' });
        return this.result('stalled');
      }
      this._setPhase(PHASE.FAILED, { reason: 'error', error: String(err) });
      return this.result('error');
    } finally {
      this.tracker.onSample = prevHandler || (() => {});
    }
  }

  async _run() {
    {
      this._setPhase(PHASE.WAITING_FOR_FACE);
      const found = await this._waitForFace(12000);
      if (!found) {
        this._setPhase(PHASE.FAILED, { reason: 'no_face' });
        return this.result('no_face');
      }

      // --- calibration ---
      this._setPhase(PHASE.CALIBRATING);
      const points = this.shuffle ? shuffled(CALIB_POINTS) : CALIB_POINTS.slice();
      for (let i = 0; i < points.length; i++) {
        this.onPoint(i, points.length, points[i], 'calibration');
        await this._awaitTap();
        await sleep(this.settleMs);
        const res = await this.tracker.calibratePoint(points[i][0], points[i][1]);
        if (res && res.accepted) {
          this.accepted++;
        } else {
          // A silently dropped point would leave the model weaker than the
          // progress bar claims, so it is retried once rather than ignored.
          this.rejected.push({ point: points[i], reason: res && res.reason });
          await sleep(1200);
          const retry = await this.tracker.calibratePoint(points[i][0], points[i][1]);
          if (retry && retry.accepted) this.accepted++;
        }
        this.onProgress({ accepted: this.accepted, total: points.length,
                          rejected: this.rejected.length });
      }

      // --- validation: measured, never trained on ---
      this._setPhase(PHASE.VALIDATING);
      for (let i = 0; i < VALIDATION_POINTS.length; i++) {
        const target = VALIDATION_POINTS[i];
        this.onPoint(i, VALIDATION_POINTS.length, target, 'validation');
        // Sample the window BEFORE the tap, not after. The tap is the only
        // moment we know the participant was actually looking at the target;
        // once they have tapped, nothing holds their gaze there. Sampling
        // afterwards measures where they drifted to and scores it as error.
        // This also matches what the tracker does for calibration, which
        // adapts on the frame at click time.
        await this._awaitTap();
        const mean = this._recentMean(this.sampleMs);
        this.validation.push({
          target,
          measured: mean,
          error: mean ? Math.hypot(mean[0] - target[0], mean[1] - target[1]) : null,
        });
        this.onProgress({ validated: this.validation.length,
                          total: VALIDATION_POINTS.length });
      }

      this._setPhase(PHASE.DONE);
      return this.result();
    }
  }

  async _waitForFace(timeoutMs) {
    const start = performance.now();
    while (performance.now() - start < timeoutMs) {
      if (this.samples.length >= 5) return true;
      await sleep(200);
    }
    return this.samples.length > 0;
  }

  result(failure = null) {
    const errs = this.validation.map((v) => v.error).filter((e) => e != null);
    const meanError = errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : null;
    const worstError = errs.length ? Math.max(...errs) : null;
    const grade = failure ? 'failed' : gradeError(meanError);
    return {
      ok: grade === 'good' || grade === 'usable',
      grade,
      failure,
      diagnostics: this.diagnostics(),
      meanError,
      worstError,
      pointsAccepted: this.accepted,
      pointsTotal: CALIB_POINTS.length,
      pointsRejected: this.rejected,
      validation: this.validation,
      validationLost: VALIDATION_POINTS.length - errs.length,
      tracker: this.tracker.name,
      trackerVersion: this.tracker.version,
    };
  }
}

function shuffled(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export class StallError extends Error {}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
