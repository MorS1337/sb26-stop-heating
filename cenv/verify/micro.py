"""Where the per-step time goes, Python vs cenv, on a live episode.

Two views, both measured inside a real ``env.step`` loop rather than assumed:

* **per-phase, per step** -- the class methods are wrapped so the time spent
  *inside* each patched function is accumulated over N steps, once with the
  shipped Python implementations and once with the C ones. This is the number
  that answers "what does the component save per step".
* **per call** -- the same functions called back to back on one fixed state, for
  the micro cost of a single invocation.

    starter-kit/.venv/bin/python -m cenv.verify.micro --scenario building_train_8/fairbanks_8

``get_zone_temp_stats`` is only ever called from ``get_zone_average_temps`` in a
real step, so it is reported as a per-call cost inside that phase, not as extra
per-step work.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]


def _time_calls(fn, n: int) -> float:
    """Best-of-three microseconds per call."""
    best = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best.append((time.perf_counter() - t0) / n * 1e6)
    return min(best)


class _Phases:
    """Accumulate process CPU time spent inside named methods during a loop."""

    def __init__(self):
        self.totals: dict[str, float] = {}
        self.saved: list[tuple] = []

    def wrap(self, obj, name: str, key: str):
        current = getattr(obj, name)
        self.saved.append((obj, name, current))

        def wrapper(*args, **kwargs):
            t0 = time.process_time()
            try:
                return current(*args, **kwargs)
            finally:
                self.totals[key] = self.totals.get(key, 0.0) + (
                    time.process_time() - t0
                )

        setattr(obj, name, wrapper)

    def restore(self):
        for obj, name, fn in reversed(self.saved):
            setattr(obj, name, fn)
        self.saved.clear()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", action="append", default=None)
    ap.add_argument("--sim", default="shipped")
    ap.add_argument("--policy", default="policies.ctrl_w:CtrlW")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--configs", default="bookkeeping,observation")
    args = ap.parse_args(argv)
    scenarios = args.scenario or ["building_train_8/fairbanks_8"]
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))

    for scenario in scenarios:
        _measure_scenario(scenario, args)
    return 0


def _measure_scenario(scenario: str, args) -> None:
    import cenv
    from cenv.verify.probe import build_env
    from sbx.evalrun import _resolve_policy

    env, evaluate = build_env(args.sim, scenario)
    sim_root = WORKSPACE / "sims" / args.sim
    spec = env.action_spec()
    policy_cls, _ = _resolve_policy(args.policy, evaluate)
    meta = evaluate.create_building_meta_data(
        env, sim_root / "smart_buildings" / "training_scenarios" / scenario
    )
    policy = evaluate.initialize_policy(policy_cls, meta, spec)
    action = evaluate.validate_action(
        evaluate.get_policy_action(policy, env.reset()[0]), spec
    )
    env.step(action)

    building_cls = __import__(
        "smart_buildings.smart_control.simulator.building", fromlist=["x"]
    ).FloorPlanBasedBuilding
    sim_cls = type(env.building.simulator)
    env_cls = type(env)
    phases = {
        "step": (env_cls, "step"),
        "zone_means": (building_cls, "get_zone_average_temps"),
        "observation": (env_cls, "_get_observation"),
        "reward_info": (sim_cls, "reward_info"),
        "compute_reward": (type(env.reward_function), "compute_reward"),
    }

    results = {}
    for label, config in (("python", ""), ("cenv", args.configs)):
        if config:
            cenv.install(config, sim_root=sim_root)
        tracker = _Phases()
        for key, (cls, name) in phases.items():
            tracker.wrap(cls, name, key)
        for _ in range(max(1, args.warmup)):
            env.step(action)
        tracker.totals.clear()
        t0 = time.process_time()
        for _ in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
        wall_cpu = time.process_time() - t0
        tracker.restore()
        per_step = {k: v / args.steps * 1000.0 for k, v in tracker.totals.items()}
        per_step["total"] = wall_cpu / args.steps * 1000.0
        results[label] = per_step
        if config:
            cenv.uninstall(config.split(",") if isinstance(config, str) else config)

    # per-call costs on a fixed state (no trajectory in between)
    stats_cls = building_cls
    stats_fn = stats_cls.get_zone_temp_stats
    rooms = [
        z for z in env.building.simulator.building.room_dict
        if not __import__(
            "smart_buildings.smart_control.simulator.constants", fromlist=["x"]
        ).is_non_physical_space(z)
    ]
    per_call = {}
    if rooms:
        zone = rooms[0]
        per_call["get_zone_temp_stats (python)"] = _time_calls(
            lambda: stats_fn(env.building.simulator.building, zone), 300
        )

    keys = ["total", "zone_means", "observation", "reward_info", "compute_reward"]
    print(f"\n{scenario}  {args.sim}  {args.policy}  {args.steps} steps/phase "
          f"(zones={len(rooms)})")
    print(f"{'phase (ms/step)':<18} {'python':>9} {'cenv':>9} {'saved':>9} {'x':>6}")
    for key in keys:
        py = results["python"].get(key, float("nan"))
        c = results["cenv"].get(key, float("nan"))
        saved = py - c
        speed = py / c if c else float("inf")
        print(f"{key:<18} {py:9.3f} {c:9.3f} {saved:9.3f} {speed:5.2f}x")
    if per_call:
        print("per call: " + ", ".join(f"{k} {v:.1f} us" for k, v in per_call.items()))


if __name__ == "__main__":
    raise SystemExit(main())
