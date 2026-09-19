/* Minimal KDE heatmap on a canvas. No dependency, ~60 lines.
 *
 * Two passes, which is the standard trick: first stamp a radial alpha
 * gradient per point so overlapping points accumulate density, then map the
 * accumulated alpha through a colour ramp. Doing it in one pass with coloured
 * blobs would make overlaps muddy rather than hot.
 */

const RAMP = [
  [0.00, [43, 58, 160, 0]],
  [0.20, [43, 58, 160, 150]],
  [0.45, [30, 158, 138, 190]],
  [0.70, [216, 209, 60, 215]],
  [1.00, [232, 69, 46, 235]],
];

function rampLookup() {
  // 256-entry lookup table, built once per draw.
  const c = document.createElement('canvas');
  c.width = 256; c.height = 1;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 256, 0);
  for (const [stop, [r, gr, b, a]] of RAMP) {
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
 */
export function drawHeatmap(canvas, points, radiusFraction = 0.085) {
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
  const lut = rampLookup();
  for (let i = 0; i < px.length; i += 4) {
    const a = px[i + 3];
    if (a === 0) continue;
    const o = a * 4;
    px[i] = lut[o];
    px[i + 1] = lut[o + 1];
    px[i + 2] = lut[o + 2];
    px[i + 3] = lut[o + 3];
  }
  ctx.putImageData(img, 0, 0);
}
