"""Environment construction shared by the fastsim tools.

Mirrors ``sbx.evalrun._run_scenario`` exactly (same loader, same two attribute
overrides) so anything measured here is the scored configuration and not a
look-alike.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]


def pin_blas() -> None:
    """One BLAS thread per process, before numpy loads."""
    for var in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def boot(sim: str = "shipped") -> None:
    """Select the simulator tree and import ``smart_buildings``."""
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    from sbx import simload  # noqa: PLC0415  (needs WORKSPACE on sys.path)

    simload.use_sim(sim)


def resolve_out(path: str | Path, sim: str = "shipped") -> Path:
    """Absolute output path for a CLI run.

    ``boot`` chdirs into the simulator tree, so a relative ``--out`` would land
    inside the (pristine) copy of the simulator. Anything relative is resolved
    against the workspace root instead.
    """
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE / p


def scenario_config(scenario: str, sim: str = "shipped") -> Path:
    """Absolute path of a ``building_x/weather_y`` scenario's config.gin."""
    from sbx import scenarios  # noqa: PLC0415

    return scenarios.config_path(sim, scenario)


def build_env(scenario: str, sim: str = "shipped"):
    """Returns ``(env, meta)`` for a scenario id such as building_train_8/fairbanks_8."""
    from smart_buildings import evaluate  # noqa: PLC0415

    cfg = scenario_config(scenario, sim)
    env = evaluate.load_hybrid_environment(cfg)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = evaluate.create_building_meta_data(env, cfg.parent)
    return env, meta


def make_policy(env, meta, spec: str = "policies.ctrl_w:CtrlW", params=None):
    """Builds the policy the same way the runner does."""
    import importlib  # noqa: PLC0415

    from smart_buildings import evaluate  # noqa: PLC0415

    module_name, _, cls_name = spec.partition(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    if params:
        cls.PARAMS = {**(getattr(cls, "PARAMS", None) or {}), **params}
    return cls, evaluate.initialize_policy(cls, meta, env.action_spec())
