"""Phase 5 tests: AOI derivation, metrics, baselines, and the endpoint.

Where possible these use cases with a known right answer — identical inputs
must score 1.0, disjoint inputs must score negatively — rather than asserting
whatever the implementation happens to produce.
"""

import importlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import analysis as A

REGIONS = [(0.90, 0.38), (0.63, 0.12), (0.31, 0.72), (0.75, 0.70), (0.50, 0.50)]
ELSEWHERE = [(0.08, 0.10), (0.12, 0.90), (0.05, 0.50), (0.20, 0.20), (0.10, 0.70)]


def paths(weights, n=24, k=5, spread=0.03, seed=1, regions=REGIONS):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = []
        for idx in rng.choice(len(regions), size=k, p=weights):
            cx, cy = regions[idx]
            p.append([float(np.clip(rng.normal(cx, spread), 0, 1)),
                      float(np.clip(rng.normal(cy, spread), 0, 1))])
        out.append(p)
    return out


HUMAN_WEIGHTS = [.34, .20, .22, .14, .10]


# --- AOI derivation --------------------------------------------------------

def test_clusters_recover_the_true_regions():
    pts = [p for path in paths(HUMAN_WEIGHTS, n=40) for p in path]
    centres = A.derive_aois(pts)
    assert 4 <= len(centres) <= 8, f"5 true regions, got {len(centres)}"

    # every true region should have a centre near it
    for cx, cy in REGIONS:
        d = np.min(np.linalg.norm(centres - np.array([cx, cy]), axis=1))
        assert d < 0.12, f"no AOI near true region ({cx}, {cy})"


def test_aoi_count_is_capped():
    rng = np.random.default_rng(0)
    scattered = rng.random((400, 2)).tolist()
    assert len(A.derive_aois(scattered)) <= A.MAX_AOIS


def test_bandwidth_matches_five_percent_of_the_diagonal():
    assert A.DEFAULT_BANDWIDTH == pytest.approx(0.05 * np.sqrt(2), abs=0.001)


def test_degenerate_inputs_do_not_crash():
    assert len(A.derive_aois([])) == 0
    assert len(A.derive_aois([[0.5, 0.5]])) == 1
    assert len(A.assign([], np.empty((0, 2)))) == 0


# --- metrics ---------------------------------------------------------------

def test_identical_paths_have_perfect_sequence_similarity():
    p = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]
    centres = A.derive_aois(p)
    assert A.sequence_similarity(p, p, centres) == 1.0


def test_reversed_path_scores_below_identical():
    p = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]
    centres = A.derive_aois(p)
    assert A.sequence_similarity(p, list(reversed(p)), centres) < 1.0


def test_distribution_sums_to_one():
    pts = [p for path in paths(HUMAN_WEIGHTS) for p in path]
    centres = A.derive_aois(pts)
    assert A.distribution(pts, centres).sum() == pytest.approx(1.0)


def test_nss_is_positive_for_matching_and_negative_for_disjoint():
    human = [p for path in paths(HUMAN_WEIGHTS, seed=1) for p in path]
    same = [p for path in paths(HUMAN_WEIGHTS, n=10, seed=2) for p in path]
    other = [p for path in paths([.2] * 5, n=10, seed=3, regions=ELSEWHERE) for p in path]
    assert A.nss(human, same) > 1.0
    assert A.nss(human, other) < 0.0


def test_spearman_needs_variance():
    assert A.spearman(np.array([0.2] * 5), np.array([0.2] * 5)) is None
    assert A.spearman(np.array([0.5, 0.5]), np.array([0.5, 0.5])) is None


# --- full report -----------------------------------------------------------

def test_a_matching_model_approaches_the_human_ceiling():
    human = paths(HUMAN_WEIGHTS, n=24, seed=1)
    model = paths(HUMAN_WEIGHTS, n=10, seed=99)
    r = A.analyse(human, model)

    assert r["ok"] and r["top_aoi"]["agree"]
    assert r["model_vs_human"]["spearman"] > 0.7
    ceiling = r["human_ceiling"]["spearman"]
    assert r["model_vs_human"]["spearman"] > 0.7 * ceiling


def test_a_disjoint_model_scores_negatively():
    human = paths(HUMAN_WEIGHTS, n=24, seed=1)
    model = paths([.2] * 5, n=10, seed=5, regions=ELSEWHERE)
    r = A.analyse(human, model)
    assert r["model_vs_human"]["spearman"] < 0
    assert r["model_vs_human"]["nss"] < 0
    assert not r["top_aoi"]["agree"]


def test_baselines_are_reported():
    r = A.analyse(paths(HUMAN_WEIGHTS), paths(HUMAN_WEIGHTS, n=10, seed=2))
    assert set(r["baselines"]) == {"random", "centre_bias"}
    for b in r["baselines"].values():
        assert "spearman" in b and "nss" in b


def test_ceiling_requires_enough_participants():
    model = paths(HUMAN_WEIGHTS, n=10, seed=2)
    few = A.analyse(paths(HUMAN_WEIGHTS, n=4), model)
    assert few["human_ceiling"]["unavailable"] is True
    assert few["human_ceiling"]["needed"] == 2 * A.MIN_PARTICIPANTS_FOR_CEILING

    many = A.analyse(paths(HUMAN_WEIGHTS, n=20), model)
    assert "unavailable" not in many["human_ceiling"]
    assert many["human_ceiling"]["n_per_half"] == 10


def test_unshared_aois_are_reported():
    """Shared zeros inflate rank correlation; the count must be visible."""
    human = paths(HUMAN_WEIGHTS, n=20, seed=1)
    model = paths([.2] * 5, n=10, seed=5, regions=ELSEWHERE)
    r = A.analyse(human, model)
    assert r["aois"]["unshared"]["total"] > 0


def test_missing_inputs_return_a_reason_not_a_crash():
    assert A.analyse([], [[[0.5, 0.5]]])["reason"] == "no human responses yet"
    assert A.analyse([[[0.5, 0.5]]], [])["reason"] == "no model run for this image"


def test_transition_matrix_rows_normalise():
    p = paths(HUMAN_WEIGHTS, n=10)
    centres = A.derive_aois([q for path in p for q in path])
    m = np.array(A.transition_matrix(p, centres))
    for row in m:
        assert row.sum() == pytest.approx(0.0) or row.sum() == pytest.approx(1.0)


# --- endpoint --------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("SCANPATH_DATA_DIR", tmp)
    monkeypatch.setenv("SCANPATH_CONTROL_TOKEN", "test-token")
    root = Path(tmp)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "model").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), "red").save(root / "images" / "a.jpg")

    model_paths = paths(HUMAN_WEIGHTS, n=6, seed=99)
    (root / "model" / "a.json").write_text(json.dumps({
        "image": "a.jpg", "mode": "freeview", "target": None, "n_fixations": 5,
        "scanpath_norm": model_paths[0], "samples_norm": model_paths,
        "source": "precomputed", "model": "test", "created_at": "2026-09-19T00:00:00Z",
    }))

    from app import config, db, urls, main
    for mod in (config, db, urls, main):
        importlib.reload(mod)
    with TestClient(main.app) as c:
        yield c


def test_endpoint_reports_missing_humans(client):
    d = client.get("/api/analysis").json()
    assert d["ok"] is False and "human" in d["reason"]


def test_endpoint_computes_once_humans_respond(client):
    rid = client.get("/api/state").json()["round_id"]
    for i, path in enumerate(paths(HUMAN_WEIGHTS, n=14, seed=1)):
        client.post("/api/markers", json={
            "round_id": rid, "participant_uuid": f"participant-{i:04d}", "points": path})

    d = client.get("/api/analysis").json()
    assert d["ok"] is True
    assert d["n_participants"] == 14
    assert d["model_vs_human"]["spearman"] > 0.5
    assert d["model_source"] == "precomputed"
    assert d["human_ceiling"]["n_per_half"] == 7


def test_endpoint_is_public(client):
    assert client.get("/api/analysis").status_code == 200
