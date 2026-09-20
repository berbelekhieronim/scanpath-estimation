/* The comparison charts, as reusable pieces.
 *
 * Two screens draw these: /charts, read at arm's length with a mouse, and the
 * projected display, read from the back of a room with nobody hovering
 * anything. They must never disagree about what the data says, so there is one
 * copy of the drawing code and the differences are options.
 *
 * Every builder takes the /api/compare/maps payload and returns a DOM node.
 * None of them fetch, and none of them own a container — the page decides
 * where things go.
 */

const LIGHT = window.matchMedia("(prefers-color-scheme: light)").matches;

export const SERIES = [
  {key: "tapped",   label: "Tapped",
   who: "phones, where people said they would look"},
  {key: "measured", label: "Measured",
   who: "webcam gaze, where they actually looked"},
  {key: "model",    label: "Model",
   who: "DeepGaze prediction"},
];

/* Series identity. The validated categorical slots 1-3, stepped per mode
   rather than flipped: dark (surface #1c1e28) worst all-pairs CVD separation
   9.4, normal-vision 20.9; light (surface #fff) 9.4 and 21.6, all above 3:1
   against their own surface. Checked by running the palette validator, not by
   looking at them. One source of truth, so the legend, the panels and the
   table cannot drift apart. */
export const HEX = LIGHT
  ? {tapped: "#2a78d6", measured: "#d95926", model: "#199e70"}
  : {tapped: "#3987e5", measured: "#d95926", model: "#199e70"};

/* The same three identities, for marks drawn on top of the photograph. Fixed
   rather than mode-dependent: the surface there is the picture, not the page,
   so flipping them with the browser's colour scheme would be answering the
   wrong question. */
export const OVERLAY = {
  tapped: "#3987e5", measured: "#d95926", model: "#199e70",
  tappedInk: "#cde2fb", measuredInk: "#ffd9c4",
};

/* Sequential: one hue, ordered from "near zero" to "most". Near zero is the
   step that recedes toward this mode's surface — the light end on a light
   page, the dark end on a dark one. A selected ramp per mode, not a flip. */
const BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
              "#184f95", "#0d366b"];
export const SEQ = LIGHT ? BLUE : BLUE.slice().reverse();

/* Diverging: two poles that read as opposite, with a neutral grey midpoint —
   never a hue in the middle. Stepped from each mode's own midpoint. */
export const DIV = LIGHT
  ? {mid: "#f0efec",
     lo: ["#afc8e8", "#70a1e0", "#2a78d6"],
     hi: ["#f3bbb3", "#ee867e", "#e34948"]}
  : {mid: "#383835",
     lo: ["#3d536e", "#3d6da8", "#3987e5"],
     hi: ["#714a46", "#aa5956", "#e66767"]};

/* Cell names follow the grid, because the grid is a query parameter. Hard-
   coding nine names once meant a 4x4 run labelled its second cell
   "top-centre", which it is not. */
const BANDS = {
  2: {rows: ["top", "bottom"], cols: ["left", "right"]},
  3: {rows: ["top", "middle", "bottom"], cols: ["left", "centre", "right"]},
};
export function cellNames(n) {
  const b = BANDS[n];
  const out = [];
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      if (!b) { out.push(`row ${r + 1}, column ${c + 1}`); continue; }
      const rw = b.rows[r], cl = b.cols[c];
      out.push(rw === "middle" && cl === "centre" ? "centre" : `${rw}-${cl}`);
    }
  }
  return out;
}

/* Which end of the sequential ramp means "more" flips with the mode, because
   near-zero is the step that recedes toward the surface. Any caption that
   says "darker means more" is therefore wrong half the time, so the word is
   derived rather than typed. */
export const MORE_IS = LIGHT ? "darker" : "lighter";

export const pct = (v) => (v * 100).toFixed(1) + "%";
export const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

/* ---------- hover ---------- */

let tipEl = null;
function showTip(html, ev) {
  if (!tipEl) {
    tipEl = el("div", "chart-tip");
    document.body.appendChild(tipEl);
  }
  tipEl.innerHTML = html;
  tipEl.hidden = false;
  const pad = 14, r = tipEl.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
  tipEl.style.left = Math.max(8, x) + "px";
  tipEl.style.top = Math.max(8, y) + "px";
}
const hideTip = () => { if (tipEl) tipEl.hidden = true; };

function hoverable(node, html, on) {
  if (!on) return;
  node.addEventListener("pointerenter", (e) => showTip(html, e));
  node.addEventListener("pointermove", (e) => showTip(html, e));
  node.addEventListener("pointerleave", hideTip);
  node.setAttribute("tabindex", "0");
  node.addEventListener("focus", () => {
    const r = node.getBoundingClientRect();
    showTip(html, {clientX: r.left + r.width / 2, clientY: r.top});
  });
  node.addEventListener("blur", hideTip);
}

/* ---------- colour ---------- */

export function seqColour(v, max) {
  if (!(max > 0)) return SEQ[0];
  return SEQ[Math.min(SEQ.length - 1,
                      Math.floor((v / max) * SEQ.length * 0.999))];
}

export function divColour(v, max) {
  // A cell within a few percent of zero is "no difference", and gets the
  // neutral midpoint rather than a faint hue that invites over-reading.
  if (!(max > 0) || Math.abs(v) < max * 0.04) return DIV.mid;
  const arm = v > 0 ? DIV.hi : DIV.lo;
  return arm[Math.min(arm.length - 1,
                      Math.floor((Math.abs(v) / max) * arm.length * 0.999))];
}

// Ink on a filled cell: the ramp spans light to dark, so the label has to
// follow the fill or it disappears at one end.
function inkFor(hex) {
  const n = parseInt(hex.slice(1), 16);
  const lum = (0.2126 * (n >> 16) + 0.7152 * ((n >> 8) & 255)
               + 0.0722 * (n & 255)) / 255;
  return lum > 0.55 ? "#12131a" : "#f2f4f8";
}

function svg(tag, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  return node;
}

/* ---------- the grid mark ---------- */

export function gridSvg(cells, n, o) {
  // The grid divides the picture, so it carries the picture's shape. Drawn
  // square over a 4:3 photograph it would not line up with the thing it
  // describes.
  const W = 300, H = Math.round(300 / (o.aspect || 1));
  const gap = 2;
  const cw = (W - gap * (n - 1)) / n, ch = (H - gap * (n - 1)) / n;
  const root = svg("svg", {viewBox: `0 0 ${W} ${H}`, role: "img",
                           "aria-label": o.aria});
  if (o.image) {
    // A faint plate of the scene underneath, so the audience remembers what
    // the cells are cells *of*. Desaturated and dimmed so it never competes
    // with the fills it sits behind.
    root.appendChild(svg("image", {
      href: o.image, x: 0, y: 0, width: W, height: H,
      preserveAspectRatio: "none", opacity: o.plate ?? 0.55,
      style: "filter:saturate(.35)",
    }));
  }
  cells.forEach((v, i) => {
    const r = Math.floor(i / n), c = i % n;
    const x = c * (cw + gap), y = r * (ch + gap);
    const fill = o.colour(v);
    // A 2px surface gap between fills, never a border, so two cells of a
    // similar shade still read as two cells.
    // Translucent whenever there is something behind worth seeing: the
    // embedded plate, or — for the grid drawn straight onto the projected
    // picture — the photograph itself.
    const alpha = o.fillOpacity ?? (o.image ? 0.78 : 1);
    const rect = svg("rect", {x, y, width: cw, height: ch, rx: 4, fill,
                              "fill-opacity": alpha});
    root.appendChild(rect);
    const t = svg("text", {
      x: x + cw / 2, y: y + ch / 2 + ch * 0.055, "text-anchor": "middle",
      "font-size": Math.round(Math.min(cw, ch) * 0.155), "font-weight": 600,
      fill: inkFor(fill), "font-family": "inherit", "pointer-events": "none",
      // Over a photograph the fill alone no longer decides legibility, so the
      // number carries its own contrast.
      ...(alpha < 1 ? {stroke: inkFor(fill) === "#12131a" ? "#fff" : "#000",
                       "stroke-width": 3, "paint-order": "stroke"} : {}),
    });
    t.textContent = o.label(v);
    root.appendChild(t);
    hoverable(rect, o.tip(i, v), o.hover);
  });
  return root;
}

/* ---------- 1. density panels ---------- */

export function densityPanels(data, opts = {}) {
  const hover = opts.hover !== false;
  const names = cellNames(data.grid);
  const maps = data.maps || {};
  const host = el("div", "panels");

  // One shared maximum across all three panels. Per-panel scales would make a
  // flat map look identical to a peaked one.
  let max = 0;
  for (const k in maps) max = Math.max(max, ...maps[k].cells);

  SERIES.forEach((s) => {
    const m = maps[s.key];
    const card = el("div", "panel" + (m ? "" : " empty"));
    const h = el("h3");
    h.innerHTML = `<span class="dot" style="background:${HEX[s.key]}"></span>`
                + esc(s.label);
    card.appendChild(h);
    card.appendChild(el("div", "who", m
      ? `${m.n} ${m.n === 1 ? "person" : "people"} — ${s.who}`
      : "no data yet"));
    if (m) {
      card.appendChild(gridSvg(m.cells, data.grid, {
        aria: `${s.label}: attention per grid cell`,
        aspect: opts.aspect,
        // Fainter than the difference map's plate: three of these sit side
        // by side and have to be compared to each other, not studied.
        image: opts.image, plate: 0.34,
        colour: (v) => seqColour(v, max),
        label: (v) => Math.round(v * 100) + "%",
        hover,
        tip: (i, v) => {
          const ci = m.ci[i];
          return `<div class="k">${esc(s.label)} — ${esc(names[i])}</div>`
               + `<strong>${pct(v)}</strong> of attention`
               + `<div class="k">95% interval ${pct(ci[0])}–${pct(ci[1])}`
               + ` · n=${m.n}</div>`;
        },
      }));
    } else {
      card.appendChild(el("div", null, s.key === "model"
        ? "Run precompute.py to fill this panel."
        : "No completed sessions in this group yet."));
    }
    host.appendChild(card);
  });
  return host;
}

export function sequentialScale(lo = "less looked at", hi = "more") {
  const n = el("div", "scale");
  // SEQ runs near-zero first and the legend reads left to right, so the
  // gradient takes it in order in either mode.
  n.innerHTML = `<span>${esc(lo)}</span><div class="bar" style="background:`
    + `linear-gradient(to right, ${SEQ.join(",")})"></div>`
    + `<span>${esc(hi)}</span>`;
  return n;
}

/* ---------- 2. difference map ---------- */

export function differenceMap(data, opts = {}) {
  const d = data.difference;
  if (!d) return null;
  const hover = opts.hover !== false;
  const names = cellNames(data.grid);
  const max = Math.max(...d.map(Math.abs)) || 1;

  const row = el("div", "diff-row");
  const map = el("div", "map");
  map.appendChild(gridSvg(d, data.grid, {
    aria: "Measured minus tapped, per grid cell",
    image: opts.image, aspect: opts.aspect,
    colour: (v) => divColour(v, max),
    label: (v) => (v > 0 ? "+" : "") + Math.round(v * 100) + "%",
    hover,
    tip: (i, v) => `<div class="k">${esc(names[i])}</div>`
      + `<strong>${v > 0 ? "+" : ""}${pct(v)}</strong>`
      + `<div class="k">${v > 0 ? "looked at more than expected"
                                : "expected more than looked at"}</div>`,
  }));
  const scale = el("div", "scale");
  scale.innerHTML = `<span>said&nbsp;more</span><div class="bar" style="background:`
    + `linear-gradient(to right, ${DIV.lo.slice().reverse().join(",")},`
    + `${DIV.mid},${DIV.hi.join(",")})"></div><span>looked&nbsp;more</span>`;
  map.appendChild(scale);
  row.appendChild(map);

  if (opts.key !== false) {
    const biggest = d.indexOf(Math.max(...d));
    const smallest = d.indexOf(Math.min(...d));
    const key = el("div", "key");
    key.innerHTML =
      `<p class="note" style="margin-top:0">Percentage points of each group's`
      + ` total attention. Positive and negative cells must cancel out — both`
      + ` maps sum to 100%, so somewhere has to lose what somewhere else`
      + ` gains.</p>`
      + `<p class="note">Biggest gap: <strong>${esc(names[biggest])}</strong>`
      + ` drew ${pct(d[biggest])} more measured gaze than tapped guesses,`
      + ` while <strong>${esc(names[smallest])}</strong> drew`
      + ` ${pct(Math.abs(d[smallest]))} less.</p>`
      + `<p class="note">Cells within a few percent of zero are drawn neutral`
      + ` grey: at this sample size they are not a difference worth`
      + ` reading.</p>`;
    row.appendChild(key);
  }
  return row;
}

/* ---------- 3. dot plot with intervals ---------- */

/* Pick one row per series to label directly, so identity never rests on
   colour alone — but never a row where the labels would sit on top of each
   other. Each series takes the free row where it is furthest from its
   neighbours, and no row is used twice. */
function labelRows(present, maps, cells) {
  const taken = new Set(), chosen = {};
  present.forEach((s) => {
    let best = -1, bestGap = -1;
    for (let i = 0; i < cells; i++) {
      if (taken.has(i)) continue;
      const v = maps[s.key].cells[i];
      let gap = Infinity;
      present.forEach((o) => {
        if (o.key !== s.key) {
          gap = Math.min(gap, Math.abs(maps[o.key].cells[i] - v));
        }
      });
      if (gap > bestGap) { bestGap = gap; best = i; }
    }
    if (best >= 0) { taken.add(best); chosen[s.key] = best; }
  });
  return chosen;
}

export function dotPlot(data, opts = {}) {
  const hover = opts.hover !== false;
  const names = cellNames(data.grid);
  const maps = data.maps || {};
  const present = SERIES.filter((s) => maps[s.key]);
  const host = el("div");
  if (!present.length) { host.textContent = "No data yet."; return host; }

  const cells = data.grid * data.grid;
  const rowH = 42, padL = 112, padR = 96, padT = 24, padB = 34, W = 780;
  const H = padT + cells * rowH + padB;
  let hi = 0;
  present.forEach((s) => maps[s.key].ci.forEach((c) => { hi = Math.max(hi, c[1]); }));
  hi = Math.max(hi, 0.05);
  const x = (v) => padL + (v / hi) * (W - padL - padR);

  const root = svg("svg", {viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Share of attention per grid cell, by group, with 95% intervals"});

  // Hairline grid, one shade off the surface. Solid, never dashed.
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = (hi / ticks) * i;
    root.appendChild(svg("line", {x1: x(v), x2: x(v), y1: padT - 6, y2: H - padB,
      stroke: "var(--border)", "stroke-width": 1}));
    const t = svg("text", {x: x(v), y: H - padB + 20, "text-anchor": "middle",
      "font-size": 11, fill: "var(--muted)", "font-family": "inherit"});
    t.textContent = Math.round(v * 100) + "%";
    root.appendChild(t);
  }

  const labels = labelRows(present, maps, cells);

  for (let i = 0; i < cells; i++) {
    const cy = padT + i * rowH + rowH / 2;
    const lab = svg("text", {x: padL - 14, y: cy + 4, "text-anchor": "end",
      "font-size": 12, fill: "var(--muted)", "font-family": "inherit"});
    lab.textContent = names[i];
    root.appendChild(lab);

    present.forEach((s, si) => {
      const m = maps[s.key];
      const off = (si - (present.length - 1) / 2) * 11;
      const v = m.cells[i], ci = m.ci[i];
      const g = svg("g", {});
      // An invisible hit band, so the tooltip does not demand a direct hit on
      // a 10px dot.
      g.appendChild(svg("rect", {x: padL - 10, y: cy + off - 8,
        width: W - padL - 10, height: 16, fill: "transparent"}));
      g.appendChild(svg("line", {
        x1: x(ci[0]), x2: x(ci[1]), y1: cy + off, y2: cy + off,
        stroke: HEX[s.key], "stroke-width": 2, "stroke-linecap": "round",
        opacity: .5}));
      // A 2px surface ring, not a border, keeps overlapping dots separable.
      g.appendChild(svg("circle", {cx: x(v), cy: cy + off, r: 5,
        fill: HEX[s.key], stroke: "var(--surface)", "stroke-width": 2}));
      if (labels[s.key] === i) {
        const t = svg("text", {x: Math.max(x(ci[1]), x(v)) + 9,
          y: cy + off + 4, "font-size": 11, fill: "var(--muted)",
          "font-family": "inherit"});
        t.textContent = s.label;
        g.appendChild(t);
      }
      hoverable(g, `<div class="k">${esc(s.label)} — ${esc(names[i])}</div>`
        + `<strong>${pct(v)}</strong>`
        + `<div class="k">95% interval ${pct(ci[0])}–${pct(ci[1])}`
        + ` · n=${m.n}</div>`, hover);
      root.appendChild(g);
    });
  }

  const scroller = el("div", "scroll-x");
  scroller.appendChild(root);
  host.appendChild(scroller);
  host.appendChild(el("div", "note",
    `Share of that group's total attention. All ${cells} cells add up to 100% `
    + "within each group, so the groups are comparable even though they are "
    + "different sizes."));
  return host;
}

/* ---------- 4. agreement ---------- */

const numCell = (v, fmt) => (v === null || v === undefined)
  ? '<td class="na">—</td>' : `<td>${esc(fmt(v))}</td>`;

export function agreementTable(data) {
  const pairs = data.pairs || {};
  const keys = Object.keys(pairs);
  const host = el("div");
  if (!keys.length) {
    host.textContent = "Needs at least two sources.";
    return host;
  }
  const label = (k) => (SERIES.find((s) => s.key === k) || {label: k}).label;

  const t = el("table", "chart-table");
  t.innerHTML = "<thead><tr><th>Pair</th><th>Correlation</th>"
    + "<th>Ceiling</th><th>% of ceiling</th></tr></thead>";
  const tb = el("tbody");
  keys.forEach((k) => {
    const [a, b] = k.split("|"), p = pairs[k];
    const ceil = Math.max(data.ceilings[a] ?? 0, data.ceilings[b] ?? 0) || null;
    const tr = el("tr");
    tr.innerHTML =
      `<td><span class="swatch" style="background:${HEX[a]}"></span>`
      + `${esc(label(a))} vs `
      + `<span class="swatch" style="background:${HEX[b]}"></span>`
      + `${esc(label(b))}</td>`
      + numCell(p.cc, (v) => v.toFixed(2))
      + numCell(ceil, (v) => v.toFixed(2))
      + numCell(p.of_ceiling, (v) => Math.round(v * 100) + "%");
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  host.appendChild(t);

  const known = Object.values(data.ceilings || {}).some((v) => v !== null);
  host.appendChild(el("div", "note", known
    ? "Correlation runs from −1 (opposite) through 0 (unrelated) to 1 "
      + "(identical). The ceiling needs at least four people in a group; "
      + "without it the percentage column stays blank."
    : "Ceilings need at least four people per group. Until then only the raw "
      + "correlation is shown, and it cannot be judged good or bad."));
  return host;
}

/* ---------- 5. table view ---------- */

export function numbersTable(data) {
  const names = cellNames(data.grid);
  const maps = data.maps || {};
  const present = SERIES.filter((s) => maps[s.key]);
  const host = el("div");
  if (!present.length) { host.textContent = "No data yet."; return host; }

  const t = el("table", "chart-table");
  let head = "<thead><tr><th>Cell</th>";
  present.forEach((s) => { head += `<th>${esc(s.label)} (n=${maps[s.key].n})</th>`; });
  if (data.difference) head += "<th>Measured − tapped</th>";
  t.innerHTML = head + "</tr></thead>";

  const tb = el("tbody");
  for (let i = 0; i < data.grid * data.grid; i++) {
    let row = `<td>${esc(names[i])}</td>`;
    present.forEach((s) => {
      const m = maps[s.key];
      row += `<td>${pct(m.cells[i])}<span class="na"> `
           + `(${pct(m.ci[i][0])}–${pct(m.ci[i][1])})</span></td>`;
    });
    if (data.difference) {
      const v = data.difference[i];
      row += `<td>${v > 0 ? "+" : ""}${pct(v)}</td>`;
    }
    const tr = el("tr");
    tr.innerHTML = row;
    tb.appendChild(tr);
  }
  t.appendChild(tb);
  const scroller = el("div", "scroll-x");
  scroller.appendChild(t);
  host.appendChild(scroller);
  return host;
}

/* ---------- legend ---------- */

export function legendRow(data) {
  const host = el("div", "chart-legend");
  SERIES.forEach((s) => {
    const m = (data.maps || {})[s.key];
    const item = el("div", "item" + (m ? "" : " off"));
    item.innerHTML = `<span class="dot" style="background:${HEX[s.key]}"></span>`
      + `<span>${esc(s.label)}</span>`
      + `<span class="n">${m ? "n=" + m.n : "none yet"}</span>`;
    host.appendChild(item);
  });
  return host;
}
