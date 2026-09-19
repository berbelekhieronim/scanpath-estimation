"""Agreement between what people said they'd look at and what the model predicts.

Spec section 7. Three things this module is careful about:

1. **Neither side is ground truth.** Human taps are a slow, deliberate,
   reportable judgement; the model was trained on real eye-tracking. When they
   disagree, that is a finding about introspection, not a model failure. So
   every metric here is symmetric and reported against baselines.

2. **The circularity trap.** If areas of interest are clustered from human
   taps and humans are then scored on whether they hit those areas, they score
   near-perfectly by construction. AOIs are therefore derived either from the
   pooled human+model points (neither side privileged) or, for the headline
   number, from one half of the participants with the other half and the model
   scored against them.

3. **Ties inflate rank correlation.** When one side visits areas the other
   never does, both sides share a zero there, and Spearman counts that
   shared zero as agreement. With real model output the two sides overlap
   heavily and this is minor; with scattered output it is not. The report
   therefore includes `aois.unshared` — how many areas only one side visited
   — so an inflated correlation is visible rather than silent.

4. **Baselines decide whether a number means anything.** Centre bias alone
   explains a surprising share of agreement in any fixation data, so beating
   random is not evidence. The human-to-human split-half ceiling is the
   comparison that matters: if model-vs-human approaches it, the claim that
   the machine predicts this as well as people do is supported.
"""

from typing import List, Optional, Sequence

import numpy as np

Path = List[Sequence[float]]      # ordered points, each (x, y) in 0..1
Points = List[Sequence[float]]

# Roughly 2 degrees of visual angle at typical viewing distance — about one
# foveal window. The spec sets this at 5% of the image DIAGONAL; coordinates
# here are normalised to a unit square whose diagonal is sqrt(2), so the
# figure is 0.05 * sqrt(2). Using a bare 0.05 over-segments badly: on a test
# scene with 5 true regions it found 11 AOIs instead of 6.
DEFAULT_BANDWIDTH = 0.0707

MIN_PARTICIPANTS_FOR_CEILING = 6   # per half; below this the ceiling is noise

# Beyond a dozen or so areas the panel stops being readable and the per-AOI
# proportions get too thin to correlate stably. Scattered model output (or
# genuinely diffuse looking) can push mean-shift well past that, so the
# bandwidth is widened until the count is manageable.
MAX_AOIS = 12


# ---------------------------------------------------------------------------
# AOIs
# ---------------------------------------------------------------------------

def derive_aois(points: Points, bandwidth: float = DEFAULT_BANDWIDTH,
                max_aois: int = MAX_AOIS) -> np.ndarray:
    """Cluster points into AOI centres with mean-shift.

    Mean-shift rather than k-means because the number of interesting regions
    in a scene is not known in advance, and its bandwidth has a principled
    setting (above) instead of being a free parameter.

    If the principled bandwidth yields more than `max_aois` regions, it is
    widened until it does not. That is a readability compromise, and the
    bandwidth actually used is reported so the compromise is visible.
    """
    from sklearn.cluster import MeanShift

    arr = np.asarray(points, dtype=float)
    if len(arr) == 0:
        return np.empty((0, 2))
    if len(arr) < 3:
        return arr.copy()

    # bin_seeding speeds up large sets but falls back with a warning on small
    # ones, so only ask for it when there is enough data to bin.
    bw = bandwidth
    centres = arr[:1].copy()
    for _ in range(8):
        ms = MeanShift(bandwidth=bw, bin_seeding=len(arr) >= 50, cluster_all=True)
        try:
            ms.fit(arr)
            found = ms.cluster_centers_
        except Exception:
            break
        if len(found):
            centres = found
        if len(centres) <= max_aois:
            break
        bw *= 1.3
    derive_aois.last_bandwidth = bw
    return centres if len(centres) else arr[:1].copy()


def assign(points: Points, centres: np.ndarray) -> np.ndarray:
    """Index of the nearest AOI centre for each point."""
    arr = np.asarray(points, dtype=float)
    if len(arr) == 0 or len(centres) == 0:
        return np.empty(0, dtype=int)
    d = np.linalg.norm(arr[:, None, :] - centres[None, :, :], axis=2)
    return np.argmin(d, axis=1)


def distribution(points: Points, centres: np.ndarray) -> np.ndarray:
    """Proportion of points falling in each AOI."""
    idx = assign(points, centres)
    counts = np.bincount(idx, minlength=len(centres)).astype(float)
    total = counts.sum()
    return counts / total if total else counts


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Rank correlation between two AOI distributions."""
    from scipy.stats import spearmanr

    if len(a) < 3 or len(b) < 3:
        return None          # a correlation over two points is meaningless
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None          # no variance to correlate
    rho, _ = spearmanr(a, b)
    return None if np.isnan(rho) else float(rho)


def density_map(points: Points, size: int = 64,
                sigma_frac: float = DEFAULT_BANDWIDTH) -> np.ndarray:
    """Gaussian-smoothed density over a square grid."""
    from scipy.ndimage import gaussian_filter

    grid = np.zeros((size, size), dtype=float)
    for x, y in points:
        gx = min(size - 1, max(0, int(x * size)))
        gy = min(size - 1, max(0, int(y * size)))
        grid[gy, gx] += 1.0
    return gaussian_filter(grid, sigma=max(1.0, sigma_frac * size))


def nss(points: Points, reference: Points, size: int = 64) -> Optional[float]:
    """Normalised Scanpath Saliency of `points` against `reference`'s density.

    The standard measure, so it is the figure comparable with published work:
    z-normalise the reference density map, then average its value at the
    locations being scored. Zero means chance.
    """
    if not points or not reference:
        return None
    m = density_map(reference, size)
    sd = m.std()
    if sd < 1e-12:
        return None
    z = (m - m.mean()) / sd
    vals = [z[min(size - 1, max(0, int(y * size))),
              min(size - 1, max(0, int(x * size)))] for x, y in points]
    return float(np.mean(vals))


def _levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sequence_similarity(p1: Path, p2: Path, centres: np.ndarray) -> Optional[float]:
    """1 - normalised edit distance between two paths' AOI label strings.

    1.0 is an identical visiting order, 0.0 is no shared structure. Length
    normalisation is what lets a 5-tap human path be compared with an
    8-fixation model path.
    """
    if not p1 or not p2 or len(centres) == 0:
        return None
    s1, s2 = assign(p1, centres), assign(p2, centres)
    denom = max(len(s1), len(s2))
    return float(1.0 - _levenshtein(list(s1), list(s2)) / denom) if denom else None


def transition_matrix(paths: List[Path], centres: np.ndarray) -> List[List[float]]:
    """Row-normalised AOI-to-AOI transition probabilities."""
    n = len(centres)
    if n == 0:
        return []
    m = np.zeros((n, n), dtype=float)
    for path in paths:
        idx = assign(path, centres)
        for a, b in zip(idx[:-1], idx[1:]):
            m[a, b] += 1
    rows = m.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(rows > 0, m / rows, 0.0)
    return m.tolist()


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def random_points(n: int, seed: int = 0) -> Points:
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)).tolist()


def centre_bias_points(n: int, seed: int = 0, sigma: float = 0.18) -> Points:
    """A 2-D Gaussian at the image centre.

    The baseline that actually matters. Real fixations cluster centrally for
    reasons that have nothing to do with scene content, so a model that only
    beat uniform random would not have demonstrated anything.
    """
    rng = np.random.default_rng(seed)
    pts = rng.normal(0.5, sigma, size=(n, 2))
    return np.clip(pts, 0.0, 1.0).tolist()


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

def _pair_scores(a_paths, b_paths, centres) -> dict:
    a_pts = [p for path in a_paths for p in path]
    b_pts = [p for path in b_paths for p in path]
    sims = [s for s in (sequence_similarity(ap, bp, centres)
                        for ap in a_paths for bp in b_paths) if s is not None]
    return {
        "spearman": spearman(distribution(a_pts, centres),
                             distribution(b_pts, centres)),
        "nss": nss(a_pts, b_pts),
        "sequence_similarity": float(np.mean(sims)) if sims else None,
    }


def analyse(human_paths: List[Path], model_paths: List[Path],
            bandwidth: float = DEFAULT_BANDWIDTH, seed: int = 0) -> dict:
    """Compare human taps with model scanpaths, against three baselines."""
    human_paths = [p for p in human_paths if p]
    model_paths = [p for p in model_paths if p]

    if not human_paths:
        return {"ok": False, "reason": "no human responses yet"}
    if not model_paths:
        return {"ok": False, "reason": "no model run for this image"}

    human_pts = [p for path in human_paths for p in path]
    model_pts = [p for path in model_paths for p in path]

    # AOIs from the pooled set, so neither side defines the yardstick it is
    # then measured against (see the circularity note in this module's docstring).
    centres = derive_aois(human_pts + model_pts, bandwidth)
    used_bandwidth = getattr(derive_aois, "last_bandwidth", bandwidth)
    if len(centres) == 0:
        return {"ok": False, "reason": "could not derive areas of interest"}

    human_dist = distribution(human_pts, centres)
    model_dist = distribution(model_pts, centres)

    human_only = int(np.sum((human_dist > 0) & (model_dist == 0)))
    model_only = int(np.sum((model_dist > 0) & (human_dist == 0)))

    n_human = len(human_pts)
    baselines = {}
    for name, pts in (("random", random_points(n_human, seed)),
                      ("centre_bias", centre_bias_points(n_human, seed))):
        baselines[name] = {
            "spearman": spearman(distribution(pts, centres), model_dist),
            "nss": nss(pts, model_pts),
        }

    # Human-to-human ceiling: split participants in half and score one against
    # the other. This is the number the demo's claim rests on.
    ceiling = None
    if len(human_paths) >= 2 * MIN_PARTICIPANTS_FOR_CEILING:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(human_paths))
        half = len(order) // 2
        a = [human_paths[i] for i in order[:half]]
        b = [human_paths[i] for i in order[half:]]
        ceiling = _pair_scores(a, b, centres)
        ceiling["n_per_half"] = half
    else:
        ceiling = {
            "unavailable": True,
            "needed": 2 * MIN_PARTICIPANTS_FOR_CEILING,
            "have": len(human_paths),
        }

    model_vs_human = _pair_scores(human_paths, model_paths, centres)

    top_human = int(np.argmax(human_dist))
    top_model = int(np.argmax(model_dist))

    return {
        "ok": True,
        "n_participants": len(human_paths),
        "n_model_samples": len(model_paths),
        "aois": {
            "centres": centres.tolist(),
            "count": len(centres),
            "bandwidth": round(used_bandwidth, 4),
            "bandwidth_widened": used_bandwidth > bandwidth * 1.001,
            "human_share": human_dist.tolist(),
            "model_share": model_dist.tolist(),
            "unshared": {
                "human_only": human_only,
                "model_only": model_only,
                "total": human_only + model_only,
            },
        },
        "model_vs_human": model_vs_human,
        "top_aoi": {
            "human": top_human,
            "model": top_model,
            "agree": top_human == top_model,
            "human_share": float(human_dist[top_human]),
            "model_share": float(model_dist[top_model]),
        },
        "baselines": baselines,
        "human_ceiling": ceiling,
        "transitions": {
            "human": transition_matrix(human_paths, centres),
            "model": transition_matrix(model_paths, centres),
        },
    }
