/* Browser and screen details, collected once per participant.
 *
 * Tracking quality varies a lot with screen size, pixel ratio and browser, so
 * without this there is no way to tell afterwards whether a poor session was
 * the room, the phone or the person. Nothing here identifies anyone — it is
 * the same information any website reads on page load.
 */
export function deviceInfo() {
  const ua = navigator.userAgent || '';
  const browser =
    /CriOS/.test(ua) ? 'Chrome iOS'
    : /FxiOS/.test(ua) ? 'Firefox iOS'
    : /EdgiOS|Edg\//.test(ua) ? 'Edge'
    : /Chrome\//.test(ua) ? 'Chrome'
    : /Firefox\//.test(ua) ? 'Firefox'
    : /Safari\//.test(ua) ? 'Safari'
    : 'other';

  let orientation = null;
  try {
    orientation = (screen.orientation && screen.orientation.type) ||
                  (window.innerWidth > window.innerHeight ? 'landscape' : 'portrait');
  } catch {}

  const q = (s) => {
    try { return window.matchMedia(s).matches; } catch { return null; }
  };

  return {
    ua: ua.slice(0, 200),
    platform: navigator.platform || null,
    browser,
    screen_w: (screen && screen.width) || null,
    screen_h: (screen && screen.height) || null,
    viewport_w: window.innerWidth,
    viewport_h: window.innerHeight,
    dpr: window.devicePixelRatio || 1,
    orientation,
    touch_points: navigator.maxTouchPoints || 0,
    languages: (navigator.languages || []).slice(0, 3).join(','),
    timezone: (() => {
      try { return Intl.DateTimeFormat().resolvedOptions().timeZone; }
      catch { return null; }
    })(),
    reduced_motion: q('(prefers-reduced-motion: reduce)'),
    color_scheme: q('(prefers-color-scheme: dark)') ? 'dark' : 'light',
  };
}
