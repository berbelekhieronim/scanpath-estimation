"""Phase 4 tests: model run loading, resolution, config, synthetic labelling."""

import importlib
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def make_run(image="a.jpg", mode="freeview", target=None, n=5,
             source="precomputed", samples=3):
    paths = [[[0.1 * (i + 1), 0.2 * (j + 1)] for j in range(n)] for i in range(samples)]
    return {
        "image": image, "mode": mode, "target": target, "n_fixations": n,
        "seed": 42, "temperature": 0.7, "prompt_text": "…",
        "scanpath_grid": [[int(x * 100), int(y * 100)] for x, y in paths[0]],
        "scanpath_norm": paths[0],
        "samples_norm": paths,
        "source": source, "model": "test-model",
        "created_at": "2026-09-19T00:00:00Z",
    }


@pytest.fixture()
def env(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("SCANPATH_DATA_DIR", tmp)
    monkeypatch.setenv("SCANPATH_CONTROL_TOKEN", "test-token")
    root = Path(tmp)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "model").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), "red").save(root / "images" / "a.jpg")
    Image.new("RGB", (640, 480), "blue").save(root / "images" / "b.png")
    return root


def start(env):
    from app import config, db, urls, main
    for mod in (config, db, urls, main):
        importlib.reload(mod)
    return TestClient(main.app)


AUTH = {"X-Control-Token": "test-token"}


def test_runs_on_disk_are_loaded_at_boot(env):
    (env / "model" / "a__freeview__n5.json").write_text(json.dumps(make_run()))
    with start(env) as c:
        d = c.get("/api/model").json()
        assert d["run"] is not None
        assert d["run"]["n_fixations"] == 5
        assert len(d["run"]["samples_norm"]) == 3


def test_no_run_is_a_normal_state_not_an_error(env):
    with start(env) as c:
        d = c.get("/api/model").json()
        assert d["run"] is None
        assert "reason" in d
        assert c.get("/api/model").status_code == 200


def test_run_for_a_missing_image_is_reported_orphaned(env):
    (env / "model" / "ghost.json").write_text(json.dumps(make_run(image="ghost.jpg")))
    with start(env) as c:
        r = c.post("/api/control/rescan-model", headers=AUTH).json()["result"]
        assert r["orphaned"] == ["ghost.json"]


def test_malformed_run_files_are_reported_not_fatal(env):
    (env / "model" / "bad.json").write_text("{ not json")
    (env / "model" / "incomplete.json").write_text(json.dumps({"image": "a.jpg"}))
    (env / "model" / "good.json").write_text(json.dumps(make_run()))
    with start(env) as c:
        r = c.post("/api/control/rescan-model", headers=AUTH).json()["result"]
        assert sorted(r["invalid"]) == ["bad.json", "incomplete.json"]
        assert r["loaded"] == ["good.json"], "a bad file must not block a good one"


def test_config_selects_which_run_is_shown(env):
    (env / "model" / "fv.json").write_text(json.dumps(make_run(mode="freeview")))
    (env / "model" / "se.json").write_text(
        json.dumps(make_run(mode="search", target="car")))
    with start(env) as c:
        assert c.get("/api/model").json()["run"]["mode"] == "freeview"
        c.post("/api/control/model-config",
               json={"mode": "search", "target": "car"}, headers=AUTH)
        run = c.get("/api/model").json()["run"]
        assert run["mode"] == "search" and run["target"] == "car"


def test_search_run_is_not_returned_for_the_wrong_target(env):
    (env / "model" / "se.json").write_text(
        json.dumps(make_run(mode="search", target="car")))
    with start(env) as c:
        c.post("/api/control/model-config",
               json={"mode": "search", "target": "laptop"}, headers=AUTH)
        assert c.get("/api/model").json()["run"] is None


def test_a_run_at_another_fixation_count_still_resolves(env):
    """Better to show an 8-fixation run than nothing when 5 was requested."""
    (env / "model" / "n8.json").write_text(json.dumps(make_run(n=8)))
    with start(env) as c:
        c.post("/api/control/model-config", json={"n_fixations": 5}, headers=AUTH)
        run = c.get("/api/model").json()["run"]
        assert run is not None and run["n_fixations"] == 8


def test_rerunning_a_config_replaces_rather_than_duplicates(env):
    (env / "model" / "a.json").write_text(json.dumps(make_run(samples=3)))
    with start(env) as c:
        (env / "model" / "a.json").write_text(json.dumps(make_run(samples=7)))
        c.post("/api/control/rescan-model", headers=AUTH)
        assert len(c.get("/api/model/runs").json()["runs"]) == 1
        assert len(c.get("/api/model").json()["run"]["samples_norm"]) == 7


def test_synthetic_source_is_preserved_all_the_way_through(env):
    """The UI badge depends on this surviving the round trip."""
    (env / "model" / "s.json").write_text(json.dumps(make_run(source="synthetic")))
    with start(env) as c:
        assert c.get("/api/model").json()["run"]["source"] == "synthetic"


def test_invalid_mode_is_rejected(env):
    with start(env) as c:
        assert c.post("/api/control/model-config", json={"mode": "nonsense"},
                      headers=AUTH).status_code == 400


def test_model_config_is_token_gated(env):
    with start(env) as c:
        assert c.post("/api/control/model-config", json={"mode": "search"}).status_code == 403


def test_model_endpoint_is_public(env):
    with start(env) as c:
        assert c.get("/api/model").status_code == 200


def test_model_layer_defaults_off(env):
    with start(env) as c:
        assert c.get("/api/state").json()["layers"]["model"] is False


# --- prompt fidelity -------------------------------------------------------

def test_prompt_templates_and_parser():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import gaze_prompts as gp

    p = gp.build_freeview_prompt(5)
    assert "exactly 5 fixation points" in p
    assert "free viewing for 3 seconds" in p
    assert "<image>" not in p, "the chat template adds the image, not the prompt"

    s = gp.build_search_prompt("car", 3)
    assert "searching for a car" in s and "exactly 3 fixation points" in s

    assert gp.parse_scanpath("[(51,46),(38,28)]") == [(51, 46), (38, 28)]
    assert gp.parse_scanpath("[(1,2,300),(4,5,600)]") == [(1, 2), (4, 5)]
    assert gp.parse_scanpath("[(150,200)]") == [(99, 99)], "clipped to the grid"
    assert gp.parse_scanpath("no coordinates here") == []
    assert gp.grid_to_norm([(50, 25)]) == [[0.5, 0.25]]
    assert len(gp.COCO_SEARCH18_TARGETS) == 18


# --- experimental prompts (Phase 7) ----------------------------------------

def test_prompt_kind_defaults_to_trained(env):
    (env / "model" / "a.json").write_text(json.dumps(make_run()))
    with start(env) as c:
        c.post("/api/model/push", json={
            "image": "a.jpg", "mode": "freeview", "n_fixations": 3,
            "scanpath_norm": [[0.1, 0.2]],
        }, headers=AUTH)
        assert c.get("/api/model").json()["run"]["prompt_kind"] == "trained"


def test_custom_prompt_kind_survives_the_round_trip(env):
    """The display's experimental warning strip depends on this."""
    with start(env) as c:
        c.post("/api/model/push", json={
            "image": "a.jpg", "mode": "freeview", "n_fixations": 3,
            "scanpath_norm": [[0.1, 0.2]],
            "prompt_kind": "custom",
            "prompt_text": "where would a curious person glance?",
        }, headers=AUTH)
        run = c.get("/api/model").json()["run"]
        assert run["prompt_kind"] == "custom"
        assert "curious person" in run["prompt_text"]


def test_jobs_are_ordered_so_each_adapter_loads_once():
    """Switching adapters re-reads sixteen gigabytes and re-merges the LoRA.
    Built image-major, a mixed run alternated once per image: ten images
    asking for free viewing and any probe meant twenty loads where two do."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import gaze_prompts as gp
    import precompute

    images = [f"img{i}.jpg" for i in range(10)]
    configs = [(p["mode"], p["target"]) for p in gp.PROBES]
    jobs = [(i, m, t) for i in images for (m, t) in configs]

    def loads(js):
        cur, n = None, 0
        for _, mode, _ in js:
            a = precompute.adapter_for(mode)
            if a != cur:
                n += 1
                cur = a
        return n

    assert loads(jobs) == 20                       # what it used to do
    jobs.sort(key=lambda j: precompute.adapter_for(j[1]))
    assert loads(jobs) == 2                        # what it does now
    # Every job survives the sort; only the order changes.
    assert len(jobs) == len(images) * len(configs)


def test_free_viewing_and_task_modes_map_to_their_own_adapters():
    """The sort and the run loop must agree, or the sort stops helping."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import precompute

    assert precompute.adapter_for("freeview") == "combined_adapter"
    for mode in ("search", "probe"):
        assert precompute.adapter_for(mode) == "visual_search_adapter"


def test_dtype_is_chosen_for_the_hardware_not_hard_coded():
    """bfloat16 needs compute capability 8.0. It was the unconditional
    default, so a GTX card or an RTX 20-series — neither of which has native
    bfloat16 — would have hit it."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import predict

    # An explicit choice is always honoured.
    for d in ("bfloat16", "float16", "float32"):
        assert predict.pick_dtype(d, "cuda") == d
    # CPU never gets bfloat16: it is slower there than the format it was
    # meant to speed up.
    assert predict.pick_dtype("auto", "cpu") == "float32"
    assert predict.pick_dtype("auto", "mps") == "bfloat16"


def test_pre_ampere_cards_fall_back_to_float16(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import predict

    class FakeCuda:
        @staticmethod
        def is_bf16_supported():
            return False

        @staticmethod
        def get_device_capability():
            return (6, 1)          # Pascal, as in a GTX 10-series

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    assert predict.pick_dtype("auto", "cuda") == "float16"


def test_ampere_and_newer_keep_bfloat16(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import predict

    class FakeCuda:
        @staticmethod
        def is_bf16_supported():
            return True

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    assert predict.pick_dtype("auto", "cuda") == "bfloat16"


def test_cuda_is_preferred_over_mps_when_both_are_present(monkeypatch):
    """The ordering used to hand an NVIDIA machine to Apple's backend."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import predict

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return True

        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return True

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    assert predict.pick_device("auto") == "cuda"


# --- hardware profiles ----------------------------------------------------

def _backends():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import backends
    return backends


def test_a_16gb_card_is_quantised_because_the_weights_alone_fill_it():
    """8B parameters in 16-bit is ~16GB. A 5070 Ti has exactly 16GB, so
    loading unquantised leaves nothing for activations and the run raises."""
    b = _backends()
    p = b._cuda_profile(16.0, bf16=True, capability=(12, 0))
    assert p.quant == "8bit"
    assert p.max_tiles is not None          # smaller prefill too
    assert any("8-bit" in n for n in p.notes)
    # Blackwell needs a torch new enough to have kernels for it.
    assert any("2.7" in n for n in p.notes)


def test_a_24gb_card_is_left_alone():
    """Quantising a card that does not need it costs accuracy for nothing."""
    b = _backends()
    p = b._cuda_profile(24.0, bf16=True, capability=(8, 9))
    assert p.quant == "none"
    assert p.sample_chunk >= 8
    assert not any("2.7" in n for n in p.notes)


def test_apple_silicon_never_asks_for_every_sample_at_once():
    """Unified memory does not raise when it runs out, it swaps — and one
    generate() call for ten samples multiplies the KV cache by ten."""
    b = _backends()
    for total in (16.0, 32.0, 64.0):
        p = b._mps_profile(total)
        assert p.quant == "none"            # the weights do fit
        assert 1 <= p.sample_chunk <= 4, total
    # The smallest machine is the most cautious.
    assert b._mps_profile(16.0).sample_chunk < b._mps_profile(64.0).sample_chunk


def test_an_unreadable_gpu_gets_the_cautious_profile():
    """Guessing small costs some speed; guessing big costs the run."""
    b = _backends()
    p = b.detect("cuda")                    # no torch in this container
    assert p.quant == "8bit"
    assert p.sample_chunk <= 2
    assert any("cautious" in n for n in p.notes)


def test_explicit_choices_override_the_profile():
    b = _backends()
    p = b.detect("cpu", requested_quant="4bit", requested_chunk=7)
    assert p.quant == "4bit" and p.sample_chunk == 7


def test_chunked_sampling_varies_the_seed_per_chunk():
    """One seed across every chunk returns the same scanpath each time and
    collapses ten observers into one — silently, since the output still
    looks like ten samples."""
    src = (Path(__file__).resolve().parent.parent
           / "tools" / "predict.py").read_text()
    assert "torch.manual_seed(seed + i)" in src
    # And the cache is handed back between chunks, or chunking buys nothing.
    assert "_release(device)" in src


def test_greedy_decoding_does_not_pretend_to_sample():
    """At temperature 0 every sample is identical, so asking for ten is ten
    times the work for one answer."""
    src = (Path(__file__).resolve().parent.parent
           / "tools" / "predict.py").read_text()
    assert "n_samples = 1" in src


def test_more_than_one_fixation_count_can_coexist_and_be_chosen(env):
    """Two real runs of the same task at different fixation counts are not
    interchangeable, and there was no way to pick between them."""
    import json as _json

    run = {"image": "a.jpg", "mode": "freeview", "target": None,
           "n_fixations": 10, "scanpath_norm": [[0.5, 0.5]] * 10,
           "samples_norm": [[[0.5, 0.5]] * 10] * 4, "source": "precomputed"}
    (env / "model" / "a__freeview__n10.json").write_text(_json.dumps(run))
    run5 = dict(run, n_fixations=5, scanpath_norm=[[0.3, 0.3]] * 5,
                samples_norm=[[[0.3, 0.3]] * 5] * 4)
    (env / "model" / "a__freeview__n5.json").write_text(_json.dumps(run5))

    with start(env) as client:
        counts = {r["n_fixations"]
                  for r in client.get("/api/model/runs").json()["runs"]}
        assert counts == {5, 10}

        for n in (5, 10):
            client.post("/api/control/model-config", json={"n_fixations": n},
                        headers=AUTH)
            assert client.get("/api/model").json()["run"]["n_fixations"] == n


def test_the_control_page_offers_the_counts_that_exist():
    """Listing counts that have no run on disk would offer a dead choice."""
    page = (Path(__file__).resolve().parent.parent
            / "app" / "static" / "control.html").read_text()
    assert "renderFixationChoice" in page
    assert "/api/model/runs" in page
    # And it says why the two are not like for like.
    assert "not because it is" in page
