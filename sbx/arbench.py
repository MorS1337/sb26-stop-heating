"""Development benchmark for the controller line in ``policies/ctrl_x.py``.

    OMP_NUM_THREADS=1 starter-kit/.venv/bin/python -m sbx.arbench

The workload is fixed; a new workload starts a new, incomparable series of scores.

* **Scenarios.** Ten of them, two per climate station: one shipped building and one
  generated training building each. They span the seasons and 30 to 79 zones.
* **Length.** Seven days each: 2016 steps from the scenario's own Sunday 06:00 start.
* **Simulator.** The *corrected* tree (starter-kit PR #2 + PR #3), where the
  boiler-billing and return-air seams are closed, so a gain measured here is physics,
  not accounting.
* **Occupancy.** Seed 12345, not the default 17321: the final evaluation reseeds
  occupancy, and this keeps the tuning off the default draw.
* **Execution.** The exact C step (``--cenv fused --fast``, bit-identical to the
  Python step), two processes, one BLAS thread each, so a fanless laptop stays
  responsive.

Primary metric: ``score``, the challenge score on this workload, i.e. the mean over
scenarios of |mean per-step reward|. Lower is better.

The run exits 1, and prints no ``score``, when:

* a scenario fails;
* any zone falls below ``FROST_MIN_K`` at any step. The shipped 278 K frost floor
  holds the coldest room near 277.8 K; +4 C allows the controller's tracking error,
  not a lower floor;
* the module closure of the policy imports anything beyond numpy and the standard
  library, or a non-docstring string in it names a building, a weather strip or a
  station. A submission must stay numpy-only and must not key on scenario identity.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

POLICY = "policies.ctrl_x:CtrlX"
SIM = "corrected"
STEPS = 2016
OCCUPANCY_SEED = 12345
WORKERS = 2
FROST_MIN_K = 277.15
OUT_DIR = WORKSPACE / "scratch" / "arbench" / "last"

# (scenario, origin). Listed most expensive first, so two workers stay balanced.
WORKLOAD: tuple[tuple[str, str], ...] = (
    ("building_train_5/singapore_4", "shipped"),   # 79 zones, June
    ("building_train_4/denver_2", "shipped"),      # 54 zones, February heating
    ("building_gen_007/denver_g4", "generated"),   # 49 zones, August cooling
    ("building_gen_009/atlanta_g4", "generated"),  # 49 zones, August cooling
    ("building_gen_012/phoenix_g0", "generated"),  # 44 zones, March, mild
    ("building_gen_008/fairbanks_g6", "generated"),  # 43 zones, October, cold
    ("building_train_1/fairbanks_1", "shipped"),   # 39 zones, January, -26 C mean
    ("building_train_7/atlanta_8", "shipped"),     # 36 zones, November heating
    ("building_train_8/phoenix_4", "shipped"),     # 30 zones, June cooling
    ("building_gen_004/singapore_g6", "generated"),  # 30 zones
)

STATIONS = ("fairbanks", "phoenix", "singapore", "atlanta", "denver")
_ID_PATTERN = re.compile(
    r"building_(train|gen|hold)_\d+|\b(" + "|".join(STATIONS + ("minneapolis",)) + r")(_g?\d+)?\b",
    re.IGNORECASE,
)
_ALLOWED_THIRD_PARTY = {"numpy"}


# --------------------------------------------------------------------------- #
# static policy checks
# --------------------------------------------------------------------------- #
def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def check_policy_closure(module: str = POLICY.split(":")[0]) -> list[str]:
    """Imports and identifiers of the policy module and every sibling it imports."""
    problems: list[str] = []
    policies_dir = WORKSPACE / "policies"
    stdlib = set(sys.stdlib_module_names)
    pending = [policies_dir / (module.split(".")[-1] + ".py")]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                match = _ID_PATTERN.search(node.value)
                if match:
                    problems.append(f"{path.name}:{node.lineno}: identifier {match.group(0)!r} in a string")
            names: list[tuple[str, int]] = []
            if isinstance(node, ast.Import):
                names = [(alias.name, 0) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "", node.level)]
            for name, level in names:
                root = name.split(".")[0]
                sibling = policies_dir / (root + ".py")
                if level > 0 or (root and sibling.exists()):
                    if level > 0 and not root:  # "from . import x"
                        continue
                    if not sibling.exists():
                        problems.append(f"{path.name}:{node.lineno}: relative import of missing {name!r}")
                        continue
                    pending.append(sibling)
                elif root not in stdlib and root not in _ALLOWED_THIRD_PARTY and root != "__future__":
                    problems.append(f"{path.name}:{node.lineno}: imports {name!r} (numpy and stdlib only)")
    return problems


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _station(scenario: str) -> str:
    weather = scenario.split("/")[1]
    return re.split(r"_g?\d+$", weather)[0]


def main() -> int:
    problems = check_policy_closure()
    if problems:
        for problem in problems:
            print(f"[arbench] POLICY CHECK FAILED: {problem}", flush=True)
        return 1

    from sbx import evalrun  # noqa: PLC0415 - after sys.path setup

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    started = time.time()
    summary = evalrun.run(
        policy=POLICY,
        scenarios=[scenario for scenario, _ in WORKLOAD],
        steps=STEPS,
        sim=SIM,
        workers=WORKERS,
        out=OUT_DIR,
        tag="arbench",
        components="parity",
        occupancy_seed=OCCUPANCY_SEED,
        cenv="fused",
        fast=True,
    )
    wall = time.time() - started
    if summary["n_failed"]:
        print(f"[arbench] FAILED: {summary['n_failed']} scenario(s) failed", flush=True)
        return 1

    origin = dict(WORKLOAD)
    records = []
    for scenario, _ in WORKLOAD:
        path = OUT_DIR / (scenario.replace("/", "__") + ".json")
        records.append(json.loads(path.read_text()))

    def mean(values):
        values = list(values)
        return sum(values) / len(values)

    per = {r["scenario"]: abs(r["mean_per_step_reward"]) for r in records}
    score = mean(per.values())
    min_zone_k = min(r["comfort"]["min_zone_temp_all_steps"] for r in records)
    coldest = min(records, key=lambda r: r["comfort"]["min_zone_temp_all_steps"])["scenario"]

    for scenario, value in per.items():
        print(f"[arbench] {scenario:34s} {value:.7f}", flush=True)
    if min_zone_k < FROST_MIN_K:
        print(
            f"[arbench] FROST CHECK FAILED: a zone reached {min_zone_k:.3f} K "
            f"({min_zone_k - 273.15:+.2f} C) on {coldest}; limit {FROST_MIN_K} K",
            flush=True,
        )
        return 1

    print(f"METRIC score={score:.9f}")
    print(f"METRIC score_shipped_bldg={mean(v for s, v in per.items() if origin[s] == 'shipped'):.9f}")
    print(f"METRIC score_generated_bldg={mean(v for s, v in per.items() if origin[s] == 'generated'):.9f}")
    for station in STATIONS:
        print(f"METRIC s_{station}={mean(v for s, v in per.items() if _station(s) == station):.9f}")
    print(f"METRIC comfort_e4={1e4 * mean(-r['components']['per_step_comfort'] for r in records):.6f}")
    print(f"METRIC energy_e3={1e3 * mean(-r['components']['per_step_energy'] for r in records):.6f}")
    print(f"METRIC carbon_e3={1e3 * mean(-r['components']['per_step_carbon'] for r in records):.6f}")
    print(f"METRIC min_zone_C={min_zone_k - 273.15:.4f}")
    # Policy CPU per step: the Kaggle budget is ~9 h for 30 thirty-day episodes.
    print(f"METRIC policy_ms={1e3 * mean(r['policy_cpu_seconds'] / r['steps'] for r in records):.3f}")
    print(f"METRIC wall_s={wall:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
