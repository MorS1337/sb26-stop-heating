"""Interleaved A/B of per-step CPU: pure Python vs ``--cenv`` components.

One interpreter, one environment, alternating blocks of ``--steps`` steps with
the components installed and removed. Interleaving inside a single process is
what makes a 2-5% effect visible: the machine is shared with other agents, so two
separate processes measured minutes apart differ by more than the effect.

    starter-kit/.venv/bin/python -m cenv.verify.ab \
        --scenario building_train_8/fairbanks_8 --steps 100 --repeats 3 \
        --config "" bookkeeping bookkeeping,observation

Reports median process-CPU seconds per ``env.step`` for each configuration
(``""`` = pure Python) and the speed-up against it, plus the reward of every
block so a change in behaviour shows up as a reward mismatch.
"""

from __future__ import annotations

import argparse
import importlib
import statistics
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]


def _apply(config: str, sim_root: Path) -> list[str]:
    """Make exactly ``config``'s components live, and assert that they are."""
    import cenv

    entries = [c.strip() for c in config.split(",") if c.strip()] if config else []
    for module_name in cenv.COMPONENTS:
        if module_name not in entries:
            cenv.uninstall([module_name])
    if entries:
        cenv.install(entries, sim_root=sim_root)
    installed = cenv.live()
    if sorted(installed) != sorted(entries):
        raise RuntimeError(
            f"_apply({config!r}) left {installed} installed; the measurement "
            "would be attributed to the wrong configuration"
        )
    return installed


def run(sim: str, scenario: str, steps: int, policy_spec: str,
        configs: list[str], repeats: int, warmup: int = 30,
        components: str = "parity", params: dict | None = None) -> dict:
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    from cenv.verify.probe import build_env
    from sbx.evalrun import _resolve_policy, _StepRecorder

    env, evaluate = build_env(sim, scenario)
    sim_root = WORKSPACE / "sims" / sim
    policy_cls, _label = _resolve_policy(policy_spec, evaluate)
    if params:
        policy_cls.PARAMS = {**(getattr(policy_cls, "PARAMS", None) or {}), **params}
    meta = evaluate.create_building_meta_data(
        env, sim_root / "smart_buildings" / "training_scenarios" / scenario
    )
    spec = env.action_spec()
    policy = evaluate.initialize_policy(policy_cls, meta, spec)
    recorder = _StepRecorder(env, meta, mode=components)

    import smart_buildings  # noqa: F401

    _apply("", sim_root)
    obs, _ = env.reset()
    for _ in range(warmup):
        action = evaluate.validate_action(
            evaluate.get_policy_action(policy, obs), spec
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        recorder.record(float(reward))
        if terminated or truncated:
            obs, _ = env.reset()

    samples: dict[str, list[float]] = {c or "python": [] for c in configs}
    rewards: dict[str, list[float]] = {c or "python": [] for c in configs}
    for _ in range(max(1, repeats)):
        for config in configs:
            _apply(config, sim_root)
            name = config or "python"
            total_reward = 0.0
            total_cpu = 0.0
            for _ in range(steps):
                action = evaluate.validate_action(
                    evaluate.get_policy_action(policy, obs), spec
                )
                t0 = time.process_time()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_cpu += time.process_time() - t0
                recorder.record(float(reward))
                total_reward += float(reward)
                if terminated or truncated:
                    obs, _ = env.reset()
            samples[name].append(total_cpu / steps)
            rewards[name].append(total_reward / steps)

    base = statistics.median(samples["python"]) if "python" in samples else None
    table = {}
    for name, values in samples.items():
        med = statistics.median(values)
        table[name] = {
            "step_cpu_s": med,
            "steps_per_cpu_s": 1.0 / med if med else float("inf"),
            "speedup": (base / med) if base and med else float("inf"),
            "reward_per_step": statistics.median(rewards[name]),
            "samples": values,
        }
    print(f"\n{scenario}  {sim}  {policy_spec}  {steps} steps x {repeats} blocks "
          f"(components={components})")
    print(f"{'installed':<26} {'s/step':>9} {'steps/s':>8} {'speedup':>8} "
          f"{'reward/step':>12}")
    for name, row in table.items():
        print(f"{name:<26} {row['step_cpu_s']:9.5f} {row['steps_per_cpu_s']:8.2f} "
              f"{row['speedup']:7.2f}x {row['reward_per_step']:12.6f}")
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", action="append", default=None)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--policy", default="policies.ctrl_w:CtrlW")
    ap.add_argument("--sim", default="shipped")
    ap.add_argument("--params", default=None)
    ap.add_argument("--config", action="append", default=None,
                    help="a --cenv value ('' = pure Python); repeatable")
    ap.add_argument("--eval-components", default="parity",
                    choices=["parity", "hook"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    import json

    configs = args.config or ["", "all"]
    scenarios = args.scenario or ["building_train_8/fairbanks_8"]
    params = json.loads(args.params) if args.params else None
    out = {}
    for scenario in scenarios:
        out[scenario] = run(scenario=scenario, sim=args.sim, steps=args.steps,
                            policy_spec=args.policy, configs=configs,
                            repeats=args.repeats, warmup=args.warmup,
                            components=args.eval_components, params=params)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
