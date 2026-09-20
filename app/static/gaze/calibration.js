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

/* Validation points sit between the calibration points, never on them —
 * scoring on a point the model was trained on measures memorisation.
 *
 * Three, not more: twelve taps in total was already reported as feeling long
 * on a phone. Three is also enough for what the correction needs, which is
 * two distinct target levels on each axis — these give x in {0.32, 0.68} and
 * y in {0.32, 0.68}. That is the minimum that can measure a *gain* and not
 * just an offset, and gain is the one that matters: the usual webcam failure
 * is not that predictions are shifted but that they huddle toward the middle
 * of the screen, which no offset can fix. */
export const VALIDATION_POINTS = [
  [0.32, 0.32], [0.68, 0.68], [0.68, 0.32],
];

/* Thresholds in fractions of viewport width.
 *
 * These are starting values, not measurements. The literature puts webcam
 * error near 4 degrees, which on a phone is about a third of the screen, so
 * "good" here is already coarse. Tune them on real sessions before trusting
 * the gate — see SPEC-WEBCAM.md section 5.1. */
/* A gain outside this band is not a measurement, it is one bad sample. */
export const GAIN_LIMITS = [0.45, 2.2];

/* How far a validation window may disagree with itself, as a fraction of the
   viewport, before the point is treated as noise rather than a reading. */
export const MAX_SPREAD = 0.25;

/* Invert the fit: from what the tracker said back to where the eye was.
 *
 * Exported because the viewing stage has to apply exactly the same
 * correction the calibration measured. Two copies of this arithmetic would
 * eventually disagree, and the disagreement would look like tracker noise.
 */
export function applyFit(measured, fit) {
  if (!fit) return measured;
  return [
    (measured[0] - fit.offset[0]) / (fit.gain[0] || 1),
    (measured[1] - fit.offset[1]) / (fit.gain[1] || 1),
  ];
}

export const QUALITY = {
  GOOD: 0.18,
  USABLE: 0.30,
};

export const PHASE = {
  IDLE: 'idle',
  WAITING_FOR_FACE: 'waiting_for_face',
  CALIBRATING: 'calibrating',
  FITTING: 'fitting',
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
    this.points = opts.points || CALIB_POINTS;

    this.phase = PHASE.IDLE;
    this.samples = [];
    this.accepted = 0;
    this.rejected = [];
    this.validation = [];
    this._resolveTap = null;

    this.memoryPeak = null;
    this.memoryFreed = null;
    // Per-point tensor counts. A single snapshot cannot distinguish a steady
    // state from a climb, and the climb is what matters.
    this.memorySeries = [];
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
    const mem = this.tracker.memory ? this.tracker.memory() : null;
    return {
      memory: mem,
      backend: this.tracker.backend || null,
      memorySeries: this.memorySeries,
      memoryPeak: this.memoryPeak,
      memoryFreed: this.memoryFreed || null,
      framesSeen: this.framesSeen,
      facesSeen: this.facesSeen,
      faceRate: this.framesSeen ? this.facesSeen / this.framesSeen : 0,
      msSinceFace: this.lastFaceAt ? Math.round(performance.now() - this.lastFaceAt) : null,
      phase: this.phase,
      accepted: this.accepted,
      rejected: this.rejected.length,
    };
  }

  /* The middle of the recent window, not its average.
   *
   * At the frame rate this runs at, a 700ms window holds two or three
   * predictions. A mean over three samples moves most of the way toward a
   * single bad one — a half-blink, a frame where the face was half out of
   * shot — and these three measurements are what the offset and gain
   * corrections are fitted to, so one of them being wrong tilts the whole
   * recording. The median of three ignores it entirely.
   *
   * The spread comes back too: a point whose samples disagree with each
   * other is not a measurement of anything, and the fit drops it.
   */
  _recentPoint(sinceMs) {
    const cut = performance.now() - sinceMs;
    const pts = this.samples.filter((s) => s.wall >= cut);
    if (!pts.length) return null;

    const mid = (xs) => {
      const a = xs.slice().sort((p, q) => p - q);
      const h = a.length >> 1;
      return a.length % 2 ? a[h] : (a[h - 1] + a[h]) / 2;
    };
    const xs = pts.map((s) => s.x), ys = pts.map((s) => s.y);
    const point = [mid(xs), mid(ys)];
    // Largest distance from the middle: how much the window disagreed.
    const spread = Math.max(...pts.map((s) =>
      Math.hypot(s.x - point[0], s.y - point[1])));
    return { point, spread, n: pts.length };
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
      const points = this.shuffle ? shuffled(this.points) : this.points.slice();
      for (let i = 0; i < points.length; i++) {
        this.onPoint(i, points.length, points[i], 'calibration');
        await this._awaitTap();
        await sleep(this.settleMs);
        // Collect only — the fit happens once, after every point (see
        // collectCalibrationSample). No 1000 ms debounce applies to
        // collection, so the pace is set by the participant, not the library.
        const res = await this.tracker.collectCalibrationSample(
          points[i][0], points[i][1]);
        if (res && res.accepted) {
          this.accepted++;
        } else {
          // A silently dropped point would leave the model weaker than the
          // progress bar claims, so it is retried once rather than ignored.
          this.rejected.push({ point: points[i], reason: res && res.reason });
          await sleep(400);
          const retry = await this.tracker.collectCalibrationSample(
            points[i][0], points[i][1]);
          if (retry && retry.accepted) this.accepted++;
        }
        // Watch tensor count across calibration: a monotonic climb is the
        // signature of the leak that kills the tab.
        const mem = this.tracker.memory ? this.tracker.memory() : null;
        if (mem) {
          this.memorySeries.push([this.accepted, mem.numTensors, mem.mb]);
          if (!this.memoryPeak || mem.numTensors > this.memoryPeak.numTensors) {
            this.memoryPeak = { ...mem, afterPoint: this.accepted };
          }
        }
        this.onProgress({ accepted: this.accepted, total: points.length,
                          rejected: this.rejected.length, memory: mem });
      }

      // One adaptation over every point collected.
      this._setPhase(PHASE.FITTING);
      const t0 = performance.now();
      const fit = await this.tracker.applyCalibration();
      this.fitMs = Math.round(performance.now() - t0);
      this.fitted = (fit && fit.fitted) || 0;

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
        const m = this._recentPoint(this.sampleMs);
        const mean = m ? m.point : null;
        this.validation.push({
          target,
          measured: mean,
          samples: m ? m.n : 0,
          spread: m ? +m.spread.toFixed(4) : null,
          error: mean ? Math.hypot(mean[0] - target[0], mean[1] - target[1]) : null,
        });
        this.onProgress({ validated: this.validation.length,
                          total: VALIDATION_POINTS.length });
      }

      // Free the retained calibration tensors before anything else runs.
      // This is the point the page crashed on a real iPhone.
      if (this.tracker.releaseCalibrationMemory) {
        this.memoryFreed = this.tracker.releaseCalibrationMemory();
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

  /* Mean residual offset between where the tracker said the eye was and
   * where the target actually sat.
   *
   * The adapter is fine-tuned from a pretrained prior in a few gradient
   * steps, so a systematic offset can survive calibration — reported on a
   * real phone as gaze registering consistently above where the person was
   * looking. The validation points already measure exactly that residual, so
   * it can be subtracted from subsequent samples instead of being lived with.
   *
   * Only the SHARED component is removed. Scatter around it is genuine
   * measurement error and stays in the numbers.
   */
  /* Validation points steady enough to fit anything to. A point whose own
     window disagreed with itself by more than this is measuring the moment,
     not the eye. */
  _usable() {
    return this.validation.filter((v) =>
      v.measured && (v.spread == null || v.spread <= MAX_SPREAD));
  }

  bias() {
    const pts = this._usable();
    if (pts.length < 2) return null;
    const dx = pts.reduce((a, v) => a + (v.measured[0] - v.target[0]), 0) / pts.length;
    const dy = pts.reduce((a, v) => a + (v.measured[1] - v.target[1]), 0) / pts.length;
    return [dx, dy];
  }

  /* A per-axis straight line from where the eye was to where the tracker said
   * it was, fitted on the validation points and then inverted to correct.
   *
   * The offset alone cannot fix the usual failure, which is not that the
   * predictions are shifted but that they are *squashed*: the eye sweeps the
   * whole screen and the tracker reports a huddle near the middle. That is a
   * gain below one, and it needs two distinct target levels per axis to see
   * at all — which is why there are four validation points.
   *
   * Refuses to correct a gain it cannot believe. With four samples a single
   * bad one can produce an absurd slope, and multiplying the real data by an
   * absurd slope is far worse than leaving it alone, so anything outside a
   * plausible band falls back to the offset-only correction.
   */
  fit() {
    const pts = this._usable();
    if (pts.length < 3) {
      // Not enough steady points to fit a slope. An offset still works on
      // two, and is better than no correction at all.
      const b = this.bias();
      return b ? { gain: [1, 1], offset: b, gainUsed: false } : null;
    }

    const axis = (i) => {
      const t = pts.map((v) => v.target[i]);
      const m = pts.map((v) => v.measured[i]);
      const tBar = t.reduce((a, x) => a + x, 0) / t.length;
      const mBar = m.reduce((a, x) => a + x, 0) / m.length;
      let cov = 0, varT = 0;
      for (let k = 0; k < t.length; k++) {
        cov += (t[k] - tBar) * (m[k] - mBar);
        varT += (t[k] - tBar) ** 2;
      }
      // No spread in the targets, or a slope that would amplify noise more
      // than it removes error: keep the offset, drop the gain.
      const gain = varT > 1e-6 ? cov / varT : 1;
      const usable = gain >= GAIN_LIMITS[0] && gain <= GAIN_LIMITS[1];
      const g = usable ? gain : 1;
      return { gain: g, offset: mBar - g * tBar, gainUsed: usable };
    };

    const x = axis(0), y = axis(1);
    // All or nothing. One bad validation sample skews both axes, so a gain
    // that is implausible on either is evidence the whole set is untrustworthy
    // — not a reason to keep the half that happens to look reasonable.
    const gainUsed = x.gainUsed && y.gainUsed;
    if (!gainUsed) {
      const b = this.bias() || [0, 0];
      return { gain: [1, 1], offset: b, gainUsed: false };
    }
    return {
      gain: [x.gain, y.gain],
      offset: [x.offset, y.offset],
      gainUsed: true,
    };
  }

  result(failure = null) {
    const errs = this.validation.map((v) => v.error).filter((e) => e != null);
    const meanError = errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : null;
    const worstError = errs.length ? Math.max(...errs) : null;

    // Error that would remain once the shared offset is removed. This is the
    // honest figure for what the corrected data can resolve; meanError stays
    // as the uncorrected measurement.
    const b = this.bias();
    const f = this.fit();
    const residuals = f ? this.validation.filter((v) => v.measured).map((v) =>
      Math.hypot(applyFit(v.measured, f)[0] - v.target[0],
                 applyFit(v.measured, f)[1] - v.target[1])) : [];
    const residualError = residuals.length
      ? residuals.reduce((a, x) => a + x, 0) / residuals.length : null;
    const grade = failure ? 'failed'
      : gradeError(residualError != null ? residualError : meanError);
    return {
      ok: grade === 'good' || grade === 'usable',
      grade,
      failure,
      diagnostics: this.diagnostics(),
      bias: b,
      fit: f,
      residualError,
      meanError,
      worstError,
      pointsAccepted: this.accepted,
      fitted: this.fitted || 0,
      fitMs: this.fitMs || null,
      pointsTotal: this.points.length,
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
