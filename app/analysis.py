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

import math

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


# ---------------------------------------------------------------------------
# Between-subjects comparison
# ---------------------------------------------------------------------------

def coarse_grid(points: Points, n: int = 3) -> np.ndarray:
    """Proportion of points in each cell of an n x n grid.

    A fixed grid, not derived clusters. Measured gaze carries roughly 20% of
    screen width in error, so cells finer than about a third of the image
    would be reporting noise as structure.
    """
    counts = np.zeros(n * n, dtype=float)
    for x, y in points:
        cx = min(n - 1, max(0, int(x * n)))
        cy = min(n - 1, max(0, int(y * n)))
        counts[cy * n + cx] += 1
    total = counts.sum()
    return counts / total if total else counts


def centre_bias_index(points: Points, n: int = 3) -> Optional[float]:
    """Share of points in the central cell.

    The measure most likely to separate the groups. Real fixations cluster
    centrally for reasons unrelated to scene content; deliberate taps do not,
    because nobody reports "I would look at the middle of the picture".
    """
    if not points:
        return None
    g = coarse_grid(points, n)
    return float(g[(n // 2) * n + (n // 2)])


def compare_sources(sources: dict, grid: int = 3, seed: int = 0) -> dict:
    """Pairwise comparison of two or more sets of viewing data.

    `sources` maps a label ("tap", "gaze", "model") to a list of paths. Every
    pair present is compared on the same coarse grid, so the numbers are on a
    common footing even though the sources differ wildly in precision.
    """
    from itertools import combinations

    pooled, grids, present = {}, {}, []
    for label, paths in sources.items():
        pts = [p for path in (paths or []) for p in path]
        if not pts:
            continue
        present.append(label)
        pooled[label] = pts
        grids[label] = coarse_grid(pts, grid)

    if len(present) < 2:
        return {"ok": False, "reason": "need at least two sources with data",
                "available": present}

    pairs = {}
    for a, bl in combinations(present, 2):
        pairs[f"{a}_vs_{bl}"] = {
            "spearman": spearman(grids[a], grids[bl]),
            "nss": nss(pooled[a], pooled[bl]),
            "top_cell_agree": int(np.argmax(grids[a])) == int(np.argmax(grids[bl])),
        }

    return {
        "ok": True,
        "grid": grid,
        "sources": {
            label: {
                "n_participants": len(sources[label]),
                "n_points": len(pooled[label]),
                "cells": grids[label].tolist(),
                "top_cell": int(np.argmax(grids[label])),
                "centre_bias": centre_bias_index(pooled[label], grid),
            } for label in present
        },
        "pairs": pairs,
        "baselines": {
            "random": {
                label: spearman(coarse_grid(random_points(len(pooled[label]), seed), grid),
                                grids[label]) for label in present
            },
            "centre_bias": {
                label: spearman(coarse_grid(centre_bias_points(len(pooled[label]), seed), grid),
                                grids[label]) for label in present
            },
        },
    }


# ---------------------------------------------------------------------------
# Comparable density maps (SPEC-METRICS.md)
# ---------------------------------------------------------------------------

FINE = 32          # working resolution before aggregating to the coarse grid
SIGMA_DEFAULT = 0.21   # worst source's measurement error, as a fraction of width


def _fine_map(points: Points, sigma: float) -> np.ndarray:
    """One participant's points as a smoothed density, summing to 1."""
    from scipy.ndimage import gaussian_filter

    grid = np.zeros((FINE, FINE), dtype=float)
    for x, y in points:
        gx = min(FINE - 1, max(0, int(x * FINE)))
        gy = min(FINE - 1, max(0, int(y * FINE)))
        grid[gy, gx] += 1.0
    if grid.sum() == 0:
        return grid
    sm = gaussian_filter(grid, sigma=max(0.5, sigma * FINE), mode="constant")
    total = sm.sum()
    return sm / total if total else sm


def _to_coarse(fine: np.ndarray, n: int) -> np.ndarray:
    """Aggregate the fine map down to an n x n grid."""
    edges = [round(i * FINE / n) for i in range(n + 1)]
    out = np.zeros(n * n, dtype=float)
    for r in range(n):
        for c in range(n):
            out[r * n + c] = fine[edges[r]:edges[r + 1], edges[c]:edges[c + 1]].sum()
    return out


def blur_to_reach(target: float, own_error: Optional[float]) -> float:
    """How much blur to ADD so a map ends at the common resolution.

    The shared kernel exists so every source is read at one resolution. Adding
    the same sigma to everybody does not achieve that — it preserves the
    differences it was meant to remove. A laptop calibration accurate to 0.08
    and a phone one accurate to 0.22 both gain 0.21 and end at 0.22 and 0.30:
    still apart, and now both blurrier than intended.

    Errors add in quadrature, so reaching a common target means adding
    sqrt(target^2 - own^2) — a lot for a precise measurement, nothing for one
    already coarser than the target. Someone already worse than the target
    cannot be sharpened, and is left alone rather than blurred further.
    """
    if not own_error or own_error <= 0:
        return target
    if own_error >= target:
        return 0.0
    return math.sqrt(target ** 2 - own_error ** 2)


def participant_maps(paths: List[Path], n: int = 3,
                     sigma: float = SIGMA_DEFAULT,
                     errors: Optional[Sequence[Optional[float]]] = None) -> np.ndarray:
    """One coarse density map per participant, each summing to 1.

    Per participant, not pooled. Someone who produced nineteen gaze samples
    must not outweigh someone who produced nine — the unit of observation is
    the person (SPEC-METRICS.md section 1).

    `errors` is each participant's own measured calibration error, when it is
    known. Given it, each map is blurred only as far as the common resolution
    rather than by a flat amount, which is what makes "one shared kernel"
    true instead of merely intended. Without it the behaviour is unchanged.

    Deliberately NOT weighted by how many points a participant contributed.
    It was tried: add a k^(-1/6) sampling term in quadrature so a four-point
    map counts as rougher than an eighteen-point one. At the measurement
    errors this study actually sees it moves a participant's total from
    0.100 to 0.103, which is a statistic that looks like rigour and does
    nothing. The shared kernel is already ~0.2 of image width, which smooths
    a four-point map into a broad blob on its own, and each map sums to 1 so
    nobody outweighs anybody. If sparse maps ever do need discounting it
    should follow from measured data, not from a constant chosen to make the
    effect visible.
    """
    out = []
    for i, p in enumerate(paths):
        if not p:
            continue
        own = errors[i] if errors is not None and i < len(errors) else None
        m = _to_coarse(_fine_map(p, blur_to_reach(sigma, own)), n)
        if m.sum() > 0:
            out.append(m)
    return np.array(out) if out else np.empty((0, n * n))


def group_map(paths: List[Path], n: int = 3, sigma: float = SIGMA_DEFAULT,
              errors: Optional[Sequence[Optional[float]]] = None) -> dict:
    """Mean density map across participants, with per-cell intervals."""
    mats = participant_maps(paths, n, sigma, errors)
    if not len(mats):
        return {"cells": [0.0] * (n * n), "ci": [[0.0, 0.0]] * (n * n), "n": 0}

    mean = mats.mean(axis=0)
    if len(mats) > 1:
        sem = mats.std(axis=0, ddof=1) / np.sqrt(len(mats))
        lo = np.clip(mean - 1.96 * sem, 0, 1)
        hi = np.clip(mean + 1.96 * sem, 0, 1)
    else:
        lo = hi = mean
    return {
        "cells": mean.tolist(),
        "ci": [[float(a), float(b)] for a, b in zip(lo, hi)],
        "n": int(len(mats)),
    }


def correlate(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Pearson correlation between two density maps.

    The saliency literature's CC. Symmetric, so neither side is treated as
    ground truth — which is the whole point here.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return None if np.isnan(r) else r


def split_half(paths: List[Path], n: int = 3, sigma: float = SIGMA_DEFAULT,
               seed: int = 0, repeats: int = 40) -> Optional[float]:
    """How well a group agrees with itself — the ceiling for any comparison.

    Averaged over repeated random splits, because a single split of ten people
    is noisy enough to mislead.
    """
    mats = participant_maps(paths, n, sigma)
    if len(mats) < 4:
        return None
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        order = rng.permutation(len(mats))
        half = len(order) // 2
        a = mats[order[:half]].mean(axis=0)
        b = mats[order[half:]].mean(axis=0)
        r = correlate(a, b)
        if r is not None:
            scores.append(r)
    return float(np.mean(scores)) if scores else None


def comparison_maps(sources: dict, n: int = 3,
                    sigma: float = SIGMA_DEFAULT, seed: int = 0,
                    errors: Optional[dict] = None) -> dict:
    """Everything the charts need, in one shape.

    All sources are smoothed with the SAME kernel, sized to the worst source's
    error. Comparing a sharp map with a blurry one and calling the difference
    disagreement is the mistake this prevents.
    """
    present = {k: v for k, v in sources.items() if v}
    if len(present) < 1:
        return {"ok": False, "reason": "no data yet"}

    errs = errors or {}
    maps = {k: group_map(v, n, sigma, errs.get(k)) for k, v in present.items()}
    ceilings = {k: split_half(v, n, sigma, seed) for k, v in present.items()}

    pairs = {}
    labels = list(present)
    for i, a in enumerate(labels):
        for bl in labels[i + 1:]:
            r = correlate(maps[a]["cells"], maps[bl]["cells"])
            ceil_a, ceil_b = ceilings.get(a), ceilings.get(bl)
            ref = max([c for c in (ceil_a, ceil_b) if c is not None], default=None)
            pairs[f"{a}|{bl}"] = {
                "cc": r,
                # Against the ceiling, not against 1.0: no comparison can be
                # expected to beat how well a group agrees with itself.
                "of_ceiling": (r / ref) if (r is not None and ref and ref > 0) else None,
            }

    # A difference map for every pair present, not just the two human ones.
    # "Where does the model part company with the people who were actually
    # measured" is the question the whole project is for, and it had no
    # picture — only a correlation, which says how much they differ and
    # never where.
    differences = {}
    labels_all = list(maps)
    for i, a in enumerate(labels_all):
        for b in labels_all[i + 1:]:
            differences[f"{a}|{b}"] = (np.array(maps[b]["cells"])
                                       - np.array(maps[a]["cells"])).tolist()

    # Kept under its old name so nothing that reads it breaks; it is simply
    # the measured-minus-tapped entry of the map above.
    difference = differences.get("tapped|measured")

    centre = (n // 2) * n + (n // 2)
    return {
        "ok": True,
        "grid": n,
        "sigma": sigma,
        "maps": maps,
        "ceilings": ceilings,
        "pairs": pairs,
        "difference": difference,
        "differences": differences,
        "centre_bias": {k: maps[k]["cells"][centre] for k in maps},
        "baselines": {
            "random": {k: correlate(
                group_map([random_points(12, seed) for _ in range(8)], n, sigma)["cells"],
                maps[k]["cells"]) for k in maps},
            "centre": {k: correlate(
                group_map([centre_bias_points(12, seed) for _ in range(8)], n, sigma)["cells"],
                maps[k]["cells"]) for k in maps},
        },
    }


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------
#
# Every source carries order and until now all three threw it away. Taps have
# a sequence, gaze samples have timestamps, model fixations are numbered, and
# the density map collapses all of it into "where", losing "when".
#
# The unit that makes them comparable is RANK, not seconds. The model has no
# clock — its fixations are first, second, third — and the humans have no
# fixation count. What both have is a proportion through their own looking:
# the first third of a scanpath against the first third of a viewing window.
#
# Three bins, and that number is set by the tracker, not by taste. At the
# measured ~4Hz a five-second window gives about eighteen to twenty samples a
# person, so three bins hold six or seven each. Four bins would hold four or
# five, which is thin, and the eight-second window that would fix it widens
# the mismatch with the model's "free viewing for 3 seconds" prompt — a
# caveat that costs more than the extra bin is worth (SCOPE.md 2.5).

BINS_DEFAULT = 3


def split_by_rank(path: Path, bins: int = BINS_DEFAULT) -> List[Path]:
    """Cut one path into equal-sized bins by position along it.

    By rank rather than by clock so the three sources can be laid side by
    side: a tap sequence and a gaze stream and a list of model fixations all
    have a beginning, a middle and an end, and nothing else in common.

    A path shorter than `bins` does not get padded or dropped. Its points
    land in the bins they fall into and the empty ones stay empty, because
    inventing a fixation to fill a bin would put attention somewhere nobody
    looked.
    """
    out: List[Path] = [[] for _ in range(bins)]
    n = len(path)
    if not n:
        return out
    for i, pt in enumerate(path):
        # i/n scaled into [0, bins); the min() guards the final point, which
        # would otherwise land in bin `bins` and be lost.
        out[min(bins - 1, int(i * bins / n))].append(pt)
    return out


def temporal_maps(sources: dict, n: int = 3, sigma: float = SIGMA_DEFAULT,
                  bins: int = BINS_DEFAULT,
                  errors: Optional[dict] = None) -> dict:
    """The comparison, once per time bin, on one shared scale.

    Same kernel as the untimed comparison and the same per-participant
    weighting, so a bin can be read against the whole and against the other
    sources. What it adds is that each source is now three maps, and the
    pairwise correlation is computed per bin: "the model and the people agree
    at the start and part company by the end" is a claim this can support and
    the pooled map could not.
    """
    present = {k: v for k, v in sources.items() if v}
    if not present:
        return {"ok": False, "reason": "no data yet"}

    errs = errors or {}
    out_bins = []
    for b in range(bins):
        maps, kept_errors = {}, {}
        for key, paths in present.items():
            per_person = [split_by_rank(p, bins)[b] for p in paths]
            src_err = errs.get(key) or []
            # Keep each participant's own error aligned with their slice, and
            # drop anyone who contributed nothing to this bin.
            pairs = [(p, src_err[i] if i < len(src_err) else None)
                     for i, p in enumerate(per_person) if p]
            maps[key] = [p for p, _ in pairs]
            kept_errors[key] = [e for _, e in pairs]

        bin_maps = {k: group_map(v, n, sigma, kept_errors[k])
                    for k, v in maps.items() if v}
        pairs_cc = {}
        labels = list(bin_maps)
        for i, a in enumerate(labels):
            for b2 in labels[i + 1:]:
                pairs_cc[f"{a}|{b2}"] = correlate(bin_maps[a]["cells"],
                                                  bin_maps[b2]["cells"])
        out_bins.append({
            "index": b,
            "label": _bin_label(b, bins),
            "maps": bin_maps,
            "pairs": pairs_cc,
        })

    return {
        "ok": True,
        "grid": n,
        "bins": bins,
        "sigma": sigma,
        "frames": out_bins,
        "sources": list(present),
    }


def _bin_label(i: int, bins: int) -> str:
    if bins == 3:
        return ("First third", "Middle third", "Last third")[i]
    if bins == 2:
        return ("First half", "Second half")[i]
    return f"Part {i + 1} of {bins}"
