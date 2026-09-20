/* Minimal KDE heatmap on a canvas. No dependency, ~60 lines.
 *
 * Two passes, which is the standard trick: first stamp a radial alpha
 * gradient per point so overlapping points accumulate density, then map the
 * accumulated alpha through a colour ramp. Doing it in one pass with coloured
 * blobs would make overlaps muddy rather than hot.
 *
 * The ramp is a parameter because this canvas is used twice on the same
 * picture — once for where people said they would look, once for where they
 * actually looked. Drawing both with one ramp made two clouds that no
 * audience could tell apart, which is what it did until now. Each source gets
 * one hue, running deep-and-transparent at low density to bright-and-opaque
 * at high, so the photograph still shows through the tails.
 */

export const RAMPS = {
  // Tapped — categorical slot 1.
  blue: [
    [0.00, [13, 54, 107, 0]],
    [0.20, [24, 79, 149, 150]],
    [0.50, [42, 120, 214, 195]],
    [0.75, [109, 167, 236, 220]],
    [1.00, [205, 226, 251, 240]],
  ],
  // Measured — categorical slot 2. Separated from blue by all-pairs CVD
  // deltaE 9.4, unlike the purple it was nominally (but never actually) drawn
  // in.
  orange: [
    [0.00, [88, 33, 12, 0]],
    [0.20, [140, 57, 24, 150]],
    [0.50, [217, 89, 38, 195]],
    [0.75, [240, 150, 110, 220]],
    [1.00, [255, 217, 196, 240]],
  ],
};

function rampLookup(ramp) {
  // 256-entry lookup table, built once per draw.
  const c = document.createElement('canvas');
  c.width = 256; c.height = 1;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 256, 0);
  for (const [stop, [r, gr, b, a]] of ramp) {
    grad.addColorStop(stop, `rgba(${r},${gr},${b},${a / 255})`);
  }
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 1);
  return g.getImageData(0, 0, 256, 1).data;
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {Array<[number,number]>} points normalised 0..1
 * @param {number} radiusFraction blob radius as a fraction of the smaller side
 * @param {string} rampName which hue this source owns — see RAMPS
 * @param {string} mode 'fill' for a solid cloud, 'contour' for nested rings
 */
export function drawHeatmap(canvas, points, radiusFraction = 0.085,
                            rampName = 'blue', mode = 'fill') {
  const w = canvas.width, h = canvas.height;
  // The first tick can fire before the image has laid out, leaving a 0x0
  // canvas that getImageData refuses to read.
  if (!w || !h) return;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.clearRect(0, 0, w, h);
  if (!points.length) return;

  const radius = Math.max(12, Math.min(w, h) * radiusFraction);

  // Pass 1 — accumulate density as alpha.
  const stamp = document.createElement('canvas');
  stamp.width = stamp.height = radius * 2;
  const sc = stamp.getContext('2d');
  const g = sc.createRadialGradient(radius, radius, 0, radius, radius, radius);
  g.addColorStop(0, 'rgba(0,0,0,1)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  sc.fillStyle = g;
  sc.fillRect(0, 0, radius * 2, radius * 2);

  // Alpha per point falls as the crowd grows, so 5 responses and 50 responses
  // both produce a readable map instead of one saturating to solid red.
  ctx.globalAlpha = Math.max(0.12, Math.min(0.5, 3.2 / Math.sqrt(points.length)));
  for (const [x, y] of points) {
    ctx.drawImage(stamp, x * w - radius, y * h - radius);
  }
  ctx.globalAlpha = 1;

  // Pass 2 — recolour by accumulated alpha.
  const img = ctx.getImageData(0, 0, w, h);
  const px = img.data;
  const lut = rampLookup(RAMPS[rampName] || RAMPS.blue);

  if (mode === 'contour') {
    contour(px, w, h, lut);
  } else {
    for (let i = 0; i < px.length; i += 4) {
      const a = px[i + 3];
      if (a === 0) continue;
      const o = a * 4;
      px[i] = lut[o];
      px[i + 1] = lut[o + 1];
      px[i + 2] = lut[o + 2];
      px[i + 3] = lut[o + 3];
    }
  }
  ctx.putImageData(img, 0, 0);
}

const BANDS = 5;        // how many density levels get a ring
const THICKNESS = 3;    // how far to look for a level change, in pixels

/* Draw the density as nested rings with nothing inside them.
 *
 * Two filled clouds on one photograph cannot both be seen: whichever is
 * painted second wins, and blending them into each other washed both out
 * against a bright sky. Giving the second source a different *form* rather
 * than only a different hue solves both at once — the lines read as a
 * separate kind of thing at a glance, and the map underneath shows through
 * the gaps.
 *
 * Quantise the accumulated density into bands, then keep only the pixels
 * where the band changes: those are the level boundaries.
 */
function contour(px, w, h, lut) {
  const n = w * h;
  const band = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    band[i] = Math.min(BANDS, (px[i * 4 + 3] * BANDS / 255) | 0);
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, b = band[i], o = i * 4;
      if (b === 0) { px[o + 3] = 0; continue; }
      const right = x + THICKNESS < w ? band[i + THICKNESS] : b;
      const down = y + THICKNESS < h ? band[i + w * THICKNESS] : b;
      if (right === b && down === b) { px[o + 3] = 0; continue; }
      // Colour the ring by the level it encloses, so the ramp still reads
      // from sparse to dense.
      const c = Math.min(255, (b * 255 / BANDS) | 0) * 4;
      px[o] = lut[c];
      px[o + 1] = lut[c + 1];
      px[o + 2] = lut[c + 2];
      // Outer rings fainter than inner ones, so the nesting reads as a
      // gradient rather than as five equally loud outlines.
      px[o + 3] = 110 + ((b / BANDS) * 125) | 0;
    }
  }
}
