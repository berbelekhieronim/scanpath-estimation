/* One navigation bar for the operator pages.
 *
 * Every page had its own ad-hoc handful of links — Controls offered four,
 * Charts two, Admin none — so getting from one to another meant knowing the
 * URLs or going back to the start page. Presenting is the wrong moment to be
 * remembering paths.
 *
 * Not used on the projected screens (/display, /qr) or on anything a
 * participant sees: a navigation bar there is either a distraction or an
 * invitation to wander off mid-study.
 */

const PAGES = [
  { path: '/start',   label: 'Start' },
  { path: '/control', label: 'Controls', token: true },
  { path: '/charts',  label: 'Charts' },
  { path: '/rounds',  label: 'Rounds', token: true },
  { path: '/admin',   label: 'Images', token: true },
  { path: '/display', label: 'Display', blank: true, token: true },
  { path: '/qr',      label: 'Join screen', blank: true },
];

const TOKEN_KEY = 'scanpath_control_token';

function token() {
  try {
    return new URLSearchParams(location.search).get('k')
      || localStorage.getItem(TOKEN_KEY) || '';
  } catch {
    return new URLSearchParams(location.search).get('k') || '';
  }
}

/** @param {string} current path of the page drawing the bar */
export function renderNav(current) {
  const host = document.getElementById('nav');
  if (!host) return;
  const k = token();

  host.innerHTML = PAGES.map((p) => {
    const here = p.path === current;
    const href = p.token && k
      ? `${p.path}?k=${encodeURIComponent(k)}`
      : p.path;
    // The projected screens open in their own tab: the presenter is driving
    // from this one and losing it mid-session would be the worst moment for
    // it. Everything else replaces the page, because they are alternatives.
    const attrs = p.blank && !here ? ' target="_blank" rel="noopener"' : '';
    return `<a class="nav-item${here ? ' on' : ''}"${attrs} href="${href}"
              ${here ? 'aria-current="page"' : ''}>${p.label}</a>`;
  }).join('');
}
