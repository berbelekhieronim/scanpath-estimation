"""Prompt templates and output parsing for DeepGaze3.5-VL.

These are copied VERBATIM from the upstream repository
(https://github.com/Susmit-A/DeepGaze3.5-VL, predict_scanpath.py), which in
turn notes that they are verbatim from its training data. That exactness is
load-bearing: the LoRA adapter was fine-tuned on these strings, and a prompt
outside that distribution still returns coordinates but of unvalidated
quality, with no error to warn you.

They are reproduced here rather than imported so this tool has no hard
dependency on the upstream checkout's import surface (which pulls matplotlib,
scipy and tqdm at module load). Run
`python tools/gaze_prompts.py --verify /path/to/DeepGaze3.5-VL`
to confirm these copies still match upstream.
"""

import re
from typing import List, Optional, Tuple

BASE_MODEL = "OpenGVLab/InternVL3_5-8B-HF"

# The 18 categories the visual-search adapter was trained on. Anything else is
# off-distribution; upstream warns and proceeds, and so do we.
COCO_SEARCH18_TARGETS = [
    "bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard",
    "knife", "laptop", "microwave", "mouse", "oven", "potted plant", "sink",
    "stop sign", "toilet", "tv",
]


def build_freeview_prompt(n: int) -> str:
    """Free-viewing prompt for exactly n fixations."""
    return (
        f"Analyze this image and predict a human eye movement scanpath during free viewing for 3 seconds.\n"
        f"A scanpath is the temporal sequence of fixation points showing where a person looks over time.\n"
        f"Consider visual saliency, semantic importance, and how attention naturally flows across a scene.\n"
        f"\n"
        f"Generate a scanpath of exactly {n} fixation points in temporal order as a list of tuples: (x, y)\n"
        f"- x: horizontal position (0-100, 0=left, 100=right)\n"
        f"- y: vertical position (0-100, 0=top, 100=bottom)\n"
        f"- Points should be ordered from first fixation to last fixation.\n"
        f"\n"
        f"Output ONLY a Python list of tuples:\n"
        f"[(51,46),(38,28),...]"
    )


def build_search_prompt(target: str, n: int) -> str:
    """Visual-search prompt for a given target and n fixations."""
    return (
        f"Analyze this image and predict a human eye movement scanpath while searching for a {target}.\n"
        f"A scanpath is the temporal sequence of fixation points showing where a person looks over time.\n"
        f"Consider the search target, visual saliency, and how attention naturally flows during visual search.\n"
        f"\n"
        f"Generate a scanpath of exactly {n} fixation points in temporal order as a list of tuples: (x, y)\n"
        f"- x: horizontal position (0-100, 0=left, 100=right)\n"
        f"- y: vertical position (0-100, 0=top, 100=bottom)\n"
        f"- Points should be ordered from first fixation to last fixation.\n"
        f"\n"
        f"Output ONLY a Python list of tuples:\n"
        f"[(51,46),(38,28),...]"
    )


def build_probe_prompt(phrase: str, n: int) -> str:
    """An experimental task prompt, built on the trained search scaffolding.

    A bare instruction ("danger") would not produce coordinates at all — the
    model has to be told the output format. So a probe keeps the search
    template's structure byte-for-byte and substitutes only the task clause.
    That stays as close to the trained distribution as a custom task can,
    which maximises the chance of usable output.

    It is still off-distribution, and is labelled as such everywhere it
    surfaces.
    """
    return (
        f"Analyze this image and predict a human eye movement scanpath while {phrase}.\n"
        f"A scanpath is the temporal sequence of fixation points showing where a person looks over time.\n"
        f"Consider the task, visual saliency, and how attention naturally flows during visual search.\n"
        f"\n"
        f"Generate a scanpath of exactly {n} fixation points in temporal order as a list of tuples: (x, y)\n"
        f"- x: horizontal position (0-100, 0=left, 100=right)\n"
        f"- y: vertical position (0-100, 0=top, 100=bottom)\n"
        f"- Points should be ordered from first fixation to last fixation.\n"
        f"\n"
        f"Output ONLY a Python list of tuples:\n"
        f"[(51,46),(38,28),...]"
    )


# The probe catalogue offered in the UI.
#
# `kind` is the honest bit. "trained" probes use a template the LoRA was
# fine-tuned on, with a target from COCO-Search18.
#
# There were eight more, marked "experimental": danger, music, robots,
# counting buildings and so on, built on build_probe_prompt() below. They
# asked the adapter for something it never saw. The model always returns
# coordinates, so each one produced a confident-looking scanpath of
# unvalidated quality, and the UI could only label that fact rather than fix
# it. Offering nine tasks of which seven are guesses is not a menu, it is a
# way to show an audience a prediction nobody can defend. They are gone.
#
# build_probe_prompt() stays because the next release means to earn one of
# these back properly — run it, measure it against human data, and add it
# only if it holds up.
PROBES = [
    {"id": "freeview", "label": "Free viewing", "kind": "trained",
     "mode": "freeview", "target": None,
     "note": "The trained free-viewing template. No task given."},

    {"id": "cars", "label": "Find a car", "kind": "trained",
     "mode": "search", "target": "car",
     "note": "A trained COCO-Search18 target. The one task the adapter was "
             "actually fine-tuned to do, alongside free viewing."},
]

PROBES_BY_ID = {p["id"]: p for p in PROBES}


def probe_for(mode: str, target: Optional[str] = None) -> Optional[dict]:
    """Find the catalogue entry matching a stored (mode, target) pair."""
    for p in PROBES:
        if p["mode"] == mode and (p["target"] or None) == (target or None):
            return p
    return None


def build_prompt(mode: str, n: int, target: str = None) -> str:
    if mode == "freeview":
        return build_freeview_prompt(n)
    if mode == "search":
        if not target:
            raise ValueError("search mode requires a target")
        return build_search_prompt(target, n)
    if mode == "probe":
        probe = PROBES_BY_ID.get(target) or probe_for("probe", target)
        if not probe:
            raise ValueError(f"unknown probe: {target!r}")
        return build_probe_prompt(probe["phrase"], n)
    raise ValueError(f"unknown mode: {mode!r}")


def parse_scanpath(text: str) -> List[Tuple[int, int]]:
    """Parse '[(51,46),(38,28)]' into [(51, 46), (38, 28)], clipped to 0-99.

    Mirrors upstream's parse_scanpath_reduced: tries the 3-tuple (x, y, t)
    form first, then the 2-tuple form, always returning (x, y) pairs.
    """
    matches = re.findall(r"\((\d+),\s*(\d+),\s*\d+\)", text)
    if matches:
        return [(min(int(x), 99), min(int(y), 99)) for x, y in matches]
    matches = re.findall(r"\((\d+),\s*(\d+)\)", text)
    return [(min(int(x), 99), min(int(y), 99)) for x, y in matches]


def grid_to_norm(coords) -> List[List[float]]:
    """Model's 0-99 grid to the 0.0-1.0 space the web app stores and renders."""
    return [[x / 100.0, y / 100.0] for x, y in coords]


def _verify_against_upstream(repo: str) -> int:
    """Diff our copies against upstream's, without importing upstream.

    Importing predict_scanpath.py would drag in matplotlib, scipy and tqdm at
    module load, which is exactly the dependency this module exists to avoid.
    Instead the source is parsed and only the two prompt builders and the
    target list are compiled and executed.
    """
    import ast

    path = f"{repo.rstrip('/')}/predict_scanpath.py"
    try:
        source = open(path).read()
    except OSError as exc:
        print(f"Could not read {path}: {exc}")
        return 2

    tree = ast.parse(source)
    wanted_fns = {"build_freeview_prompt", "build_search_prompt"}
    wanted_var = "COCO_SEARCH18_TARGETS"
    extracted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_fns:
            extracted.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == wanted_var for t in node.targets
        ):
            extracted.append(node)

    found_fns = {n.name for n in extracted if isinstance(n, ast.FunctionDef)}
    if found_fns != wanted_fns:
        print(f"Upstream file did not contain the expected builders "
              f"(found {sorted(found_fns)}). Its layout may have changed.")
        return 2

    ns: dict = {}
    exec(compile(ast.Module(body=extracted, type_ignores=[]), path, "exec"), ns)

    problems = 0
    for n in (3, 5, 8):
        if ns["build_freeview_prompt"](n) != build_freeview_prompt(n):
            print(f"MISMATCH: freeview prompt, n={n}")
            problems += 1
        if ns["build_search_prompt"]("car", n) != build_search_prompt("car", n):
            print(f"MISMATCH: search prompt, n={n}")
            problems += 1
    if sorted(ns.get(wanted_var, [])) != sorted(COCO_SEARCH18_TARGETS):
        print("MISMATCH: COCO-Search18 target list")
        problems += 1

    print("Prompts match upstream exactly."
          if not problems else f"{problems} mismatch(es) — update this file")
    return 1 if problems else 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Verify prompt fidelity.")
    ap.add_argument("--verify", metavar="DEEPGAZE_REPO",
                    help="Path to a DeepGaze3.5-VL checkout to diff against")
    args = ap.parse_args()
    raise SystemExit(_verify_against_upstream(args.verify) if args.verify else 0)
