"""Timing: per env-step CPU for plain, fused (single env) and the batch.

One process per measurement set, because a batch and a single-environment run
cannot be built in the same interpreter reliably (the simulator's occupancy and
convection generators are process-global on the Python side, so a second
environment changes the first one's draws -- which is the reason the batch owns
per-environment C streams in the first place). Inside a set the repetitions are
interleaved, and the reported number is the median of ``reps``.

    python -m cenv.verify.batch_bench single --scenario building_train_8/fairbanks_8
    python -m cenv.verify.batch_bench single --cenv fused --scenario ...
    python -m cenv.verify.batch_bench batch --scenario ... --M 16
    python -m cenv.verify.batch_bench table --steps 60 --reps 3   # the whole table

"env only" feeds the step a fixed action (the policy's decision at step 0,
reused), so the number is ``env.step`` and nothing else; "env+CtrlW" recomputes
the action every step and the policy's own CPU time is reported separately.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
PY = str(WS / "starter-kit" / ".venv" / "bin" / "python")
if str(WS) not in sys.path:
    sys.path.insert(0, str(WS))

POLICY = "policies.ctrl_w:CtrlW"


def _config(sim: str, scenario: str) -> Path:
    return (WS / "sims" / sim / "smart_buildings" / "training_scenarios"
            / scenario / "config.gin")


def _load() -> list[float]:
    try:
        return [round(x, 2) for x in __import__("os").getloadavg()]
    except Exception:
        return []


def _action(evaluate, policy, obs, spec):
    return evaluate.validate_action(evaluate.get_policy_action(policy, obs), spec)


# --------------------------------------------------------------------------- #
# one environment, plain and fused, interleaved
# --------------------------------------------------------------------------- #
def bench_single(sim: str, scenario: str, steps: int, reps: int,
                 cenv_name: str | None = None, warmup: int = 10) -> dict:
    """One single-environment configuration, ``cenv_name`` installed or not.

    Its own process: installing ``fused`` patches the simulator classes, so a
    plain baseline cannot share an interpreter with it.
    """
    from sbx import simload

    simload.use_sim(sim)
    if cenv_name:
        import cenv

        cenv.install(cenv_name, sim_root=simload.sim_root(sim), verbose=False)

    from sbx.evalrun import _load_environment, _resolve_policy
    from smart_buildings import evaluate

    config = _config(sim, scenario)
    policy_cls, _label = _resolve_policy(POLICY, evaluate)
    env = _load_environment(evaluate, config, {})
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = evaluate.create_building_meta_data(env, config.parent)
    spec = env.action_spec()
    obs, _ = env.reset()
    policy = evaluate.initialize_policy(policy_cls, meta, spec)
    fixed = _action(evaluate, policy, obs, spec)

    env_only = []
    for _rep in range(reps):
        for _ in range(warmup):
            env.step(fixed)
        t0 = time.process_time()
        for _ in range(steps):
            env.step(fixed)
        env_only.append((time.process_time() - t0) / steps * 1e3)

    both = []
    policy_ms = []
    for _rep in range(reps):
        obs, _ = env.reset()
        for _ in range(warmup):
            obs, _, _, _, _ = env.step(_action(evaluate, policy, obs, spec))
        t_pol = 0.0
        t0 = time.process_time()
        for _ in range(steps):
            q0 = time.process_time()
            act = _action(evaluate, policy, obs, spec)
            t_pol += time.process_time() - q0
            obs, _, _, _, _ = env.step(act)
        dt = time.process_time() - t0
        both.append(dt / steps * 1e3)
        policy_ms.append(t_pol / steps * 1e3)
    out = {
        "sim": sim,
        "scenario": scenario,
        "cenv": cenv_name,
        "steps": steps,
        "reps": reps,
        "load": _load(),
        "env_only_ms": statistics.median(env_only),
        "env_only_ms_all": env_only,
        "env_plus_policy_ms": statistics.median(both),
        "policy_ms": statistics.median(policy_ms),
    }
    if cenv_name == "fused":
        from cenv import fused as fused_mod

        prof = dict(fused_mod.PROFILE)
        out["fused_c_ms_per_step"] = (prof.get("c_seconds", 0.0)
                                      / max(prof.get("steps", 1), 1) * 1e3)
    print(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------- #
# the batch
# --------------------------------------------------------------------------- #
def bench_batch(sim: str, scenario: str, m: int, steps: int, reps: int,
                warmup: int = 10) -> dict:
    from sbx import simload

    simload.use_sim(sim)   # before smart_buildings is imported anywhere
    from cenv.batch import BatchEnv, PROFILE
    from sbx.evalrun import _resolve_policy
    from smart_buildings import evaluate

    config = _config(sim, scenario)
    env = BatchEnv(config, sim, m)
    policy_cls, _label = _resolve_policy(POLICY, evaluate)
    policies = [evaluate.initialize_policy(policy_cls, meta,
                                          env.envs[0].action_spec())
                for meta in env.metas]
    obs = env.reset()
    spec = env.envs[0].action_spec()
    fixed = [_action(evaluate, policies[k], obs[k], spec) for k in range(m)]

    env_only = []
    for _rep in range(reps):
        for _ in range(warmup):
            env.step(fixed)
        t0 = time.process_time()
        for _ in range(steps):
            env.step(fixed)
        env_only.append((time.process_time() - t0) / steps / m * 1e3)

    both = []
    policy_ms = []
    for _rep in range(reps):
        obs = env.reset()
        for _ in range(warmup):
            acts = [_action(evaluate, policies[k], obs[k], spec)
                    for k in range(m)]
            obs, _, _ = env.step(acts)
        t_pol = 0.0
        t0 = time.process_time()
        for _ in range(steps):
            q0 = time.process_time()
            acts = [_action(evaluate, policies[k], obs[k], spec)
                    for k in range(m)]
            t_pol += time.process_time() - q0
            obs, _, _ = env.step(acts)
        dt = time.process_time() - t0
        both.append(dt / steps / m * 1e3)
        policy_ms.append(t_pol / steps / m * 1e3)
    out = {
        "sim": sim,
        "scenario": scenario,
        "M": m,
        "steps": steps,
        "reps": reps,
        "load": _load(),
        "env_only_ms": statistics.median(env_only),
        "env_only_ms_all": env_only,
        "env_plus_policy_ms": statistics.median(both),
        "policy_ms": statistics.median(policy_ms),
        "c_ms_per_env_step": PROFILE["c_seconds"] / max(PROFILE["steps"], 1)
        / max(m, 1) * 1e3,
        "solve_phase_ms_per_step": env.tbatch.phase(),
        "tbatch_k": env.tbatch.k,
        "tbatch_n": env.tbatch.n,
        "tbatch_n_full": env.tbatch.n_full,
    }
    env.close()
    print(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
# M per building. train_2 costs ~250 MB of simulator state per environment
# (~95 MB on train_8), so M=64 would need 16 GB on its own; 32 is the ceiling
# this machine takes with the rest of the box in use.
MS = {8: [1, 4, 16, 32, 64], 2: [1, 4, 16, 32]}


def table(scenarios=("building_train_8/fairbanks_8",
                     "building_train_2/fairbanks_2"), steps=100, reps=3,
          out: Path | None = None) -> dict:
    results = {}
    for scenario in scenarios:
        building = scenario.rsplit("/", 1)[0]
        budget = MS.get(int(building.split("_")[-1]), [1, 4, 16])
        plain = _json([PY, "-m", "cenv.verify.batch_bench", "single", "--scenario", scenario,
                       "--steps", str(steps), "--reps", str(reps)])
        fused = _json([PY, "-m", "cenv.verify.batch_bench", "single", "--cenv", "fused",
                       "--scenario", scenario, "--steps", str(steps),
                       "--reps", str(reps)])
        results[scenario] = {"single": {"python": plain, "fused": fused},
                             "batch": {}}
        for m in budget:
            results[scenario]["batch"][m] = _json(
                [PY, "-m", "cenv.verify.batch_bench", "batch", "--scenario", scenario, "--M", str(m),
                 "--steps", str(steps), "--reps", str(reps)])
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(results, indent=1))
    markdown(results)
    return results


def markdown(results: dict) -> None:
    """The table the ticket asks for, one row per (building, configuration)."""
    print("\n| building | config | M | env only ms/env-step | env+CtrlW "
          "ms/env-step | policy ms | C call ms | solve ms/step | load |")
    print("|---|---|---|---|---|---|---|---|---|")
    for scenario, data in results.items():
        single = data["single"]
        for name in ("python", "fused"):
            row = single[name]
            print(f"| {scenario.split('/')[0]} | {name} | 1 |"
                  f" {row['env_only_ms']:.2f} | {row['env_plus_policy_ms']:.2f} |"
                  f" {row['policy_ms']:.2f} |"
                  f" {row.get('fused_c_ms_per_step', float('nan')):.2f} | - |"
                  f" {row['load'][0]} |")
        for m, row in data["batch"].items():
            solve = row["solve_phase_ms_per_step"]
            print(f"| {scenario.split('/')[0]} | batch | {m} |"
                  f" {row['env_only_ms']:.2f} | {row['env_plus_policy_ms']:.2f} |"
                  f" {row['policy_ms']:.2f} | {row['c_ms_per_env_step']:.2f} |"
                  f" {solve['solve']:.1f} (all {m}) | {row['load'][0]} |")


def _json(argv) -> dict:
    cmd = [str(a) for a in argv]
    print(f"[bench] {' '.join(cmd[1:])}", flush=True)
    proc = subprocess.run(cmd, cwd=str(WS), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed: {proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("single", "batch"):
        p = sub.add_parser(name)
        p.add_argument("--cenv", default=None)
        p.add_argument("--sim", default="shipped")
        p.add_argument("--scenario", default="building_train_8/fairbanks_8")
        p.add_argument("--steps", type=int, default=100)
        p.add_argument("--reps", type=int, default=3)
        if name == "batch":
            p.add_argument("--M", type=int, default=16)
    t = sub.add_parser("table")
    t.add_argument("--steps", type=int, default=100)
    t.add_argument("--reps", type=int, default=3)
    t.add_argument("--out", default=str(WS / "scratch" / "CEnvBatch"
                                        / "timing.json"))
    args = ap.parse_args(argv)
    if args.cmd == "single":
        bench_single(args.sim, args.scenario, args.steps, args.reps,
                     cenv_name=args.cenv)
        return 0
    if args.cmd == "batch":
        bench_batch(args.sim, args.scenario, args.M, args.steps, args.reps)
        return 0
    table(steps=args.steps, reps=args.reps, out=Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
