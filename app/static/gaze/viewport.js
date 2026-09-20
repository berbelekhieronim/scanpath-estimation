/* One measured idea of "the screen", shared by the layout and by the maths.
 *
 * iOS Safari sizes `position: fixed; inset: 0` and `100vh` against the LARGE
 * viewport — the one you get with the toolbars hidden. In landscape the
 * toolbars are showing and never collapse, because the page also sets
 * `overflow: hidden` so there is nothing to scroll. The result is a page
 * roughly a quarter taller than the visible area, with its bottom edge behind
 * the toolbar and no way to reach it. That hid the bottom row of calibration
 * targets and the bottom of the picture.
 *
 * It was not only a layout bug. The tracker returns gaze normalised to the
 * screen, and the page converted that with window.innerHeight. When the
 * visible area and the layout viewport disagree, so do those two, and every
 * sample is stretched against a box the participant could not see. Both now
 * read the same measured numbers.
 *
 * visualViewport is the only thing that reports what is actually visible.
 */

const listeners = new Set();
let box = measure();

function measure() {
  const vv = window.visualViewport;
  return {
    width: Math.round(vv ? vv.width : window.innerWidth),
    height: Math.round(vv ? vv.height : window.innerHeight),
    // Which way up the device is, at the moment of measuring. A calibration
    // is a mapping onto one screen geometry; rotating produces a different
    // one, and the old mapping does not survive it.
    orientation: (window.innerWidth >= window.innerHeight)
      ? 'landscape' : 'portrait',
  };
}

/** The current visible viewport. Never cached by callers — it changes. */
export function viewport() {
  return box;
}

function publish() {
  const next = measure();
  const changed = next.width !== box.width || next.height !== box.height
    || next.orientation !== box.orientation;
  const previous = box;
  box = next;
  apply();
  if (changed) listeners.forEach((fn) => fn(next, previous));
}

/** Expose the measurement to CSS as --app-w / --app-h. */
function apply() {
  const s = document.documentElement.style;
  s.setProperty('--app-w', box.width + 'px');
  s.setProperty('--app-h', box.height + 'px');
}

/** Call once, early. Returns the first measurement. */
export function track() {
  apply();
  const vv = window.visualViewport;
  if (vv) {
    vv.addEventListener('resize', publish);
    // Toolbars sliding in and out move the visible box without resizing it.
    vv.addEventListener('scroll', publish);
  }
  window.addEventListener('resize', publish);
  window.addEventListener('orientationchange', () => {
    // iOS reports the old size for a frame or two after the rotation event.
    setTimeout(publish, 120);
    setTimeout(publish, 400);
  });
  return box;
}

/** Notified on every real change, with (current, previous). */
export function onChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
