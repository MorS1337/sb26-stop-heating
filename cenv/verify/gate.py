"""Full-episode gate: official Python vs ``--cenv`` components.

Two episodes of the same policy and scenario are run in independent processes --
one pure Python, one with the C components installed -- and their trajectories
are compared step by step: reward (every step), the reward response's scalars,
and the per-zone temperatures. This is the whole-episode gate the ticket asks
for (``<=1e-9`` relative on totals, ``<=1e-8`` K on zone temperatures); the
per-function gate is :mod:`cenv.shadow` and the per-call cost is
:mod:`cenv.verify.micro`.

    # run one side (writes a trace)
    python -m cenv.verify.gate trace --scenario building_train_8/fairbanks_8 \
        --steps 8641 --cenv bookkeeping,observation --out /tmp/cenv_gate/t8_c

    # compare the two traces
    python -m cenv.verify.gate compare /tmp/cenv_gate/t8_py.npz /tmp/cenv_gate/t8_c.npz

    # the whole regression matrix (both trees, both start offsets, a generated
    # building), each case in its own pair of processes
    python -m cenv.verify.gate suite

``--start-offset-days`` shifts ``start_timestamp`` exactly as ``sbx.evalrun``
does (``_shifted_config``); a nonzero offset is a distinct regression axis, not
a cosmetic one. An episode that starts *inside* the occupancy model's
arrival/departure draw windows (06:00-19:00 local on a work day) is the only
one in which the occupant chain's first interval is ticked, so a reward path
that skips that tick agrees at offset 0 and disagrees from the second interval
on. That is what the suite's offset cases exist to catch.

Both sides are ``--sim shipped`` unless told otherwise, because the gate is about
reproducing the official simulator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]

# (sim, scenario, start_offset_days, steps). The last four are the regression
# this list exists for: offsets 1 and 2 start the episode at 06:00 on a Monday
# and a Tuesday, inside the arrival window on a work day.
SUITE = (
    ("shipped", "building_train_8/atlanta_1", 0, 600),
    ("corrected", "building_train_8/atlanta_1", 0, 600),
    ("shipped", "building_train_2/phoenix_6", 0, 600),
    ("corrected", "building_train_2/phoenix_6", 0, 600),
    ("shipped", "building_train_8/fairbanks_8", 1, 600),
    ("shipped", "building_train_8/fairbanks_8", 2, 600),
    ("corrected", "building_train_8/fairbanks_8", 1, 600),
    ("corrected", "building_train_8/fairbanks_8", 2, 600),
    ("shipped", "building_gen_001/singapore_g2", 1, 600),
    ("corrected", "building_gen_001/singapore_g2", 1, 600),
)

def trace(sim: str, scenario: str, steps: int, policy_spec: str,
          cenv_spec: str | None, out: Path, start_offset_days: int = 0,
          params: dict | None = None, components: str = "parity",
          occupancy_seed: int | None = None, log_every: int = 500) -> dict:
    """Run one episode and write its trajectory (npz) plus a JSON summary."""
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    import time as _time

    started_cpu = _time.process_time()
    started_wall = _time.time()

    from sbx import simload

    simload.use_sim(sim)
    if cenv_spec:
        import cenv

        cenv.install(cenv_spec, sim_root=simload.sim_root(sim), verbose=True)

    from smart_buildings import evaluate
    from sbx.evalrun import _StepRecorder, _resolve_policy, _load_environment

    config = (
        WORKSPACE / "sims" / sim / "smart_buildings" / "training_scenarios"
        / scenario / "config.gin"
    )
    # The scenario directory (building meta data, relative data paths) is the
    # *original* one: ``_shifted_config`` only rewrites ``start_timestamp`` into
    # a private copy, exactly as ``sbx.evalrun._run_scenario`` does it (it takes
    # ``scenario_dir`` before shifting too).
    scenario_dir = config.parent
    if start_offset_days:
        from sbx.evalrun import _shifted_config  # noqa: PLC0415

        config = _shifted_config(config, int(start_offset_days),
                                 Path(out).resolve().parent / "_gin")
        print(f"[gate] start offset {start_offset_days} d -> {config}", flush=True)
    policy_cls, _label = _resolve_policy(policy_spec, evaluate)
    if params:
        policy_cls.PARAMS = {**(getattr(policy_cls, "PARAMS", None) or {}), **params}
    bindings = {}
    if occupancy_seed is not None:
        bindings["RandomizedArrivalDepartureOccupancy.seed"] = int(occupancy_seed)
    env = _load_environment(evaluate, config, bindings)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0

    meta = evaluate.create_building_meta_data(env, scenario_dir)
    spec = env.action_spec()
    policy = evaluate.initialize_policy(policy_cls, meta, spec)
    recorder = _StepRecorder(env, meta, mode=components)

    obs, _ = env.reset()
    n = int(steps)
    rewards = np.zeros(n, dtype=np.float64)
    zone_names = None
    zone_temps = None
    obs_trace = None
    step_cpu = 0.0
    done = 0
    for i in range(n):
        action = evaluate.validate_action(
            evaluate.get_policy_action(policy, obs), spec
        )
        t0 = _time.process_time()
        obs, reward, terminated, truncated, _ = env.step(action)
        step_cpu += _time.process_time() - t0
        recorder.record(float(reward))
        rewards[i] = float(reward)
        averages = env.building.simulator.building.get_zone_average_temps()
        if zone_names is None:
            zone_names = list(averages)
            zone_temps = np.zeros((n, len(zone_names)), dtype=np.float64)
            obs_trace = np.zeros((n, int(np.asarray(obs).size)), dtype=np.float64)
        zone_temps[i] = [averages[name] for name in zone_names]
        obs_trace[i] = np.asarray(obs, dtype=np.float64)
        done = i + 1
        if terminated or truncated:
            break
        if log_every and (i + 1) % log_every == 0:
            print(f"[gate] {scenario} {i + 1}/{n} steps", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        rewards=rewards[:done],
        zone_temps=zone_temps[:done] if zone_temps is not None else np.zeros(0),
        obs=obs_trace[:done] if obs_trace is not None else np.zeros(0),
        zone_names=np.array(zone_names or [], dtype=object),
        meta=json.dumps(
            {
                "scenario": scenario,
                "sim": sim,
                "policy": policy_spec,
                "params": params or {},
                "cenv": cenv_spec,
                "components": components,
                "steps": done,
                "start_offset_days": int(start_offset_days),
                "occupancy_seed": occupancy_seed,
                "step_cpu_seconds": step_cpu,
                "cpu_seconds": _time.process_time() - started_cpu,
                "wall_seconds": _time.time() - started_wall,
                "total_reward": float(rewards[:done].sum()),
            }
        ),
    )
    summary = json.loads(str(np.load(out)["meta"]))
    print(f"[gate] wrote {out}: steps={done} total_reward={summary['total_reward']:.6f} "
          f"step_cpu={summary['step_cpu_seconds']:.1f}s "
          f"({summary['step_cpu_seconds'] / max(done, 1) * 1000:.2f} ms/step)", flush=True)
    return summary


def compare(path_a: str | Path, path_b: str | Path,
            rel_tol: float = 1e-9, temp_tol: float = 1e-8) -> dict:
    """Compare two traces; ``a`` is the reference (official Python)."""
    a = np.load(path_a, allow_pickle=True)
    b = np.load(path_b, allow_pickle=True)
    ma, mb = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
    result: dict = {"a": str(path_a), "b": str(path_b),
                    "scenario": ma["scenario"], "steps": [ma["steps"], mb["steps"]],
                    "policy": ma["policy"], "sim": ma["sim"]}
    ra, rb = a["rewards"], b["rewards"]
    n = min(ra.size, rb.size)
    result["total_reward"] = [float(ra[:n].sum()), float(rb[:n].sum())]
    base = abs(result["total_reward"][0])
    result["total_reward_rel_diff"] = (
        abs(result["total_reward"][1] - result["total_reward"][0]) / base
        if base else abs(result["total_reward"][1] - result["total_reward"][0])
    )
    d = np.abs(ra[:n] - rb[:n])
    result["max_step_reward_abs"] = float(d.max()) if n else 0.0
    result["first_reward_diff_step"] = int(np.argmax(d > 0)) if (d > 0).any() else -1
    za, zb = a["zone_temps"], b["zone_temps"]
    if za.size and zb.size and za.shape == zb.shape:
        dz = np.abs(za - zb)
        result["max_zone_temp_abs_K"] = float(dz.max())
        result["max_zone_temp_step"] = int(np.argmax(dz.max(axis=1)))
        result["zone_names"] = list(map(str, a["zone_names"]))
        names = list(map(str, a["zone_names"]))
        if dz.max() > 0:
            worst = np.unravel_index(int(np.argmax(dz)), dz.shape)
            result["worst_zone"] = names[worst[1]]
        bit_identical = int(np.count_nonzero(za != zb))
        result["zone_temp_values_differing"] = bit_identical
        result["zone_temp_values_total"] = int(za.size)
    else:
        result["max_zone_temp_abs_K"] = None
    result["passed"] = bool(
        result["total_reward_rel_diff"] <= rel_tol
        and (result["max_zone_temp_abs_K"] is None
             or result["max_zone_temp_abs_K"] <= temp_tol)
    )
    result["step_cpu_seconds"] = [ma["step_cpu_seconds"], mb["step_cpu_seconds"]]
    result["ms_per_step"] = [
        1000.0 * ma["step_cpu_seconds"] / max(ma["steps"], 1),
        1000.0 * mb["step_cpu_seconds"] / max(mb["steps"], 1),
    ]
    result["speedup"] = (
        result["ms_per_step"][0] / result["ms_per_step"][1]
        if result["ms_per_step"][1] else None
    )
    print(f"\n{result['scenario']}  {result['policy']}  {result['sim']}  "
          f"steps={result['steps']}")
    print(f"total_reward      {result['total_reward'][0]:14.6f} vs "
          f"{result['total_reward'][1]:14.6f}  rel={result['total_reward_rel_diff']:.3e}")
    print(f"max |dreward|     {result['max_step_reward_abs']:.3e}  "
          f"(first differing step {result['first_reward_diff_step']})")
    if result["max_zone_temp_abs_K"] is not None:
        print(f"max |dT_zone|     {result['max_zone_temp_abs_K']:.3e} K  "
              f"(step {result['max_zone_temp_step']}, "
              f"{result['zone_temp_values_differing']}/"
              f"{result['zone_temp_values_total']} values differ)")
    print(f"ms/step           {result['ms_per_step'][0]:.2f} -> "
          f"{result['ms_per_step'][1]:.2f}  ({result['speedup']:.2f}x)")
    print("GATE " + ("PASS" if result["passed"] else "FAIL"))
    return result


def run_suite(out_dir: str | Path, cases=SUITE, policy_spec: str = "policies.ctrl_w:CtrlW",
              rel_tol: float = 1e-9, temp_tol: float = 1e-8,
              verbose: bool = True) -> list[dict]:
    """Run every (sim, scenario, offset, steps) case in its own two processes."""
    import os

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for sim, scenario, offset, steps in cases:
        tag = f"{sim}__{scenario.replace('/', '__')}__off{offset}"
        plain = out_dir / f"{tag}__plain.npz"
        cenv = out_dir / f"{tag}__cenv.npz"
        for side, extra in (("plain", []), ("cenv", ["--cenv", "fused"])):
            dest = plain if side == "plain" else cenv
            cmd = [sys.executable, "-m", "cenv.verify.gate", "trace",
                   "--sim", sim, "--scenario", scenario, "--steps", str(steps),
                   "--start-offset-days", str(offset), "--policy", policy_spec,
                   "--out", str(dest), "--log-every", "0", *extra]
            proc = subprocess.run(cmd, cwd=str(WORKSPACE), env={**os.environ},
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"gate trace {side} {tag} failed ({proc.returncode}):\n"
                    f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        result = compare(plain, cenv, rel_tol=rel_tol, temp_tol=temp_tol)
        result["case"] = dict(sim=sim, scenario=scenario,
                              start_offset_days=offset, steps=steps)
        result["json"] = str(out_dir / f"{tag}__cmp.json")
        (out_dir / f"{tag}__cmp.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        if verbose:
            print(f"GATE {'PASS' if result['passed'] else 'FAIL'} "
                  f"{sim} {scenario} off={offset} steps={steps} "
                  f"rel={result['total_reward_rel_diff']:.3e} "
                  f"dT={result['max_zone_temp_abs_K']:.3e} K", flush=True)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--scenario", required=True)
    t.add_argument("--steps", type=int, default=8641)
    t.add_argument("--policy", default="policies.ctrl_w:CtrlW")
    t.add_argument("--params", default=None)
    t.add_argument("--sim", default="shipped")
    t.add_argument("--cenv", default=None)
    t.add_argument("--start-offset-days", type=int, default=0)
    t.add_argument("--occupancy-seed", type=int, default=None)
    t.add_argument("--components", default="parity", choices=["parity", "hook"])
    t.add_argument("--out", required=True)
    t.add_argument("--log-every", type=int, default=500)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--rel-tol", type=float, default=1e-9)
    c.add_argument("--temp-tol", type=float, default=1e-8)
    c.add_argument("--json", default=None)
    s = sub.add_parser("suite")
    s.add_argument("--out", default="scratch/cenv_gate_suite")
    s.add_argument("--policy", default="policies.ctrl_w:CtrlW")
    s.add_argument("--rel-tol", type=float, default=1e-9)
    s.add_argument("--temp-tol", type=float, default=1e-8)
    s.add_argument("--only", default=None,
                   help="substring filter on '<sim> <scenario> off<N>', so the "
                        "matrix can be run in chunks (one chunk per process on "
                        "a machine with a wall-clock budget)")
    args = ap.parse_args(argv)

    if args.cmd == "trace":
        params = json.loads(args.params) if args.params else None
        summary = trace(
            sim=args.sim, scenario=args.scenario, steps=args.steps,
            policy_spec=args.policy, cenv_spec=args.cenv, out=Path(args.out),
            start_offset_days=args.start_offset_days, params=params,
            components=args.components, occupancy_seed=args.occupancy_seed,
            log_every=args.log_every,
        )
        return 0
    if args.cmd == "suite":
        cases = [c for c in SUITE
                 if args.only is None
                 or args.only in f"{c[0]} {c[1]} off{c[2]}"]
        if not cases:
            print(f"gate suite: --only {args.only!r} matched no case")
            return 2
        results = run_suite(args.out, cases=cases, policy_spec=args.policy,
                            rel_tol=args.rel_tol, temp_tol=args.temp_tol)
        failed = [r for r in results if not r["passed"]]
        print(f"\nGATE SUITE {'PASS' if not failed else 'FAIL'} "
              f"({len(results) - len(failed)}/{len(results)} cases)")
        return 0 if not failed else 1
    result = compare(args.a, args.b, rel_tol=args.rel_tol, temp_tol=args.temp_tol)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
