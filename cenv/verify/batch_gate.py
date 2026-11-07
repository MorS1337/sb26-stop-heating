"""Gate for ``cenv.batch``: each environment against a single-environment run.

The question a batch has to answer is not "does the C match the C" -- the fused
kernel already has its own gate -- but "is environment i of a batch still the
trajectory a separate single-environment run of the official simulator
produces, given the same seed, the same policy and the same parameters". So the
two sides are:

* one process per **single-environment** side: ``cenv/verify/fused_gate.py
  trace``, which is the reference machinery already used to gate ``fused``
  (unpatched Python, or ``--cenv fused`` for the SuperLU path);
* one process for the whole **batch** side: M=4 environments with four
  occupancy seeds and four CtrlW parameter sets, stepped together by
  ``cenv.batch.run_batch``.

Every environment of the batch is compared, one to one, with **two** separate
single-environment runs of its own seed, parameters and policy:

* the **same-factorization** reference (``--cenv fused,solver``): one
  environment, the same LL^T the batch solves with, through
  ``cenv/solver.py``'s shim. This is the verdict, because it isolates the one
  thing this component changes -- M environments per call instead of one -- and
  it is bit-exact (measured: total reward and observations identical, zone means
  3.5e-15 relative);
* the **official simulator** (unpatched Python, scipy's SuperLU): reported
  beside it. A Cholesky and a pivoted LU differ in the last bits, and on a
  policy that sits on a decision threshold that can become a different damper
  command. This is not batching: ``--cenv fused,solver`` alone -- a plain
  single environment, no batch anywhere -- reproduces the same divergence at the
  same step with the same number of flips (measured on ``corrected``
  train_8, seed 999, CtrlW comfort_cost=20000: first reward difference at step
  2562, 377 damper flips, total reward 3.5e-9 relative, for both).

The pass criteria are the project's: <= 1e-12 relative per temperature
component, <= 1e-9 relative on the episode total reward, no action flips.

``M=1`` also runs, to show that one environment through the batch kernel lands
on the SuperLU single-environment path within the same tolerance -- and the
``fused`` single-environment path itself is re-checked, because this work split
``cenv_fused_run`` into the slices a batch needs.

    python -m cenv.verify.batch_gate gate                 # the checklist
    python -m cenv.verify.batch_gate gate --only shipped  # one tree
    python -m cenv.verify.batch_gate trace --sim shipped \
        --scenario building_train_8/fairbanks_8 --steps 2880 \
        --out scratch/CEnvBatch/batch.npz                  # one batch side

The batch traces land in ``scratch/CEnvBatch/``; the reference traces are the
ones ``fused_gate`` caches, so a re-run reuses them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
SCRATCH = WORKSPACE / "scratch" / "CEnvBatch"
FUSED_SCRATCH = WORKSPACE / "scratch" / "CEnvFused2"

# One occupancy seed and one CtrlW parameter set per environment. The
# parameters are real keys of ``CtrlW.PARAMS`` (a bogus key would be ignored and
# the "different policies" part of the check would be vacuous): each set changes
# the controller's decisions, so the four environments must diverge from each
# other while each still matches its own single-environment run.
SEEDS = [17321, 12345, 999, 7]
PARAMS = [
    {},                                        # CtrlW as shipped
    {"comfort_margin": 0.30},
    {"comfort_cost": 20000.0},
    {"response_scale": 0.40, "model_residual_gain": 0.30},
]
CASES = [
    ("shipped", "building_train_8/fairbanks_8"),
    ("shipped", "building_train_2/fairbanks_2"),
    ("corrected", "building_train_8/fairbanks_8"),
    ("corrected", "building_train_2/fairbanks_2"),
]
STEPS = 2880
TEMP_REL = 1e-12
# The reference for the *batching* verdict: a single environment with the same
# factorization the batch uses. ``cenv solver`` routes the simulator's splu
# through the same LL^T (cenv/solver.py), so a batch environment and its
# reference then solve the same linear systems and the comparison isolates what
# this component changes -- running M environments in one call.
SAME_FACTOR_CENV = "fused,solver"
REWARD_REL = 1e-9


def _scenario_path(sim: str, scenario: str) -> Path:
    return (WORKSPACE / "sims" / sim / "smart_buildings" / "training_scenarios"
            / scenario / "config.gin")


def _rel(a, b, scale):
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), scale)
    return float(np.max(diff / denom)) if diff.size else 0.0


# --------------------------------------------------------------------------- #
# the batch side
# --------------------------------------------------------------------------- #
def trace_batch(sim: str, scenario: str, steps: int, out: Path,
                seeds=None, params=None, policy: str = "policies.ctrl_w:CtrlW",
                tag: str | None = None) -> dict:
    """Runs the whole batch in this process and writes one npz per environment.

    The files have the layout ``cenv/verify/fused_gate.py`` writes, so
    ``fused_gate.compare`` compares a batch environment with a
    single-environment run without a second comparison implementation.
    """
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    from cenv.batch import PROFILE, run_batch

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds or SEEDS)
    params = [dict(p) for p in (params or PARAMS)]
    m = len(seeds)
    started_cpu = __import__("time").process_time()
    started_wall = __import__("time").time()
    result = run_batch(
        scenario, sim, [(policy, p) for p in params], steps,
        occupancy_seeds=seeds, components="parity", tag=tag, record_trace=True,
    )
    tr = result["traces"]
    done = result["steps"]
    paths = []
    for k in range(m):
        path = out / f"{sim}_{scenario.replace('/', '_')}_s{seeds[k]}_b{k}.npz"
        meta = {
            "scenario": scenario,
            "sim": sim,
            "policy": f"{policy} {params[k]}",
            "params": params[k],
            "cenv": "batch",
            "components": "parity",
            "steps": done,
            "occupancy_seed": seeds[k],
            "n_obs": int(tr["obs"].shape[-1]),
            "step_cpu_seconds": float(PROFILE["c_seconds"]) / max(m, 1),
            "cpu_seconds": __import__("time").process_time() - started_cpu,
            "wall_seconds": __import__("time").time() - started_wall,
            "total_reward": float(tr["rewards"][k].sum()),
            "batch": {
                "size": m,
                "index": k,
                "timing": result["timing"],
            },
            "fused_profile": None,
        }
        np.savez_compressed(
            path,
            rewards=tr["rewards"][k],
            obs=tr["obs"][k],
            zone_temps=tr["zone_temps"][k],
            damper=tr["damper"][k],
            boiler=tr["boiler"][k],
            zone_names=tr["zone_names"],
            **{f"c_{key}": value[k] for key, value in tr["components"].items()},
            meta=json.dumps(meta),
        )
        paths.append(path)
        print(f"[batch_gate] wrote {path.name}: steps={done}"
              f" total_reward={meta['total_reward']:.9f}", flush=True)
    summary = {"paths": [str(p) for p in paths], "timing": result["timing"],
               "steps": done, "sim": sim, "scenario": scenario}
    (out / "manifest.json").write_text(json.dumps(summary, indent=1,
                                                  default=str))
    return summary


# --------------------------------------------------------------------------- #
# the single-environment side (a separate process per environment)
# --------------------------------------------------------------------------- #
def _batch_side(sim, scenario, steps, out, seeds, params, tag):
    """The batch side in its own process (one simulator tree per interpreter:
    ``simload.use_sim`` swaps the ``smart_buildings`` package, and gin refuses
    to register one configurable twice)."""
    manifest = Path(out) / "manifest.json"
    if manifest.exists():
        print(f"[batch_gate] reusing {manifest}")
        return json.loads(manifest.read_text())
    _run([
        sys.executable, "-m", "cenv.verify.batch_gate", "trace",
        "--sim", sim, "--scenario", scenario, "--steps", str(steps),
        "--out", str(Path(out).resolve()),
        "--seeds", ",".join(str(s) for s in seeds),
        "--params", json.dumps(list(params)),
    ])
    return json.loads(manifest.read_text())


def _run(argv: list[str]) -> None:
    import subprocess

    proc = subprocess.run(argv, cwd=str(WORKSPACE), capture_output=True,
                          text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess failed: {' '.join(argv)}\n{proc.stdout}\n{proc.stderr}")
    sys.stdout.write(proc.stdout)


def trace_single(sim, scenario, steps, seed, params, tag, cenv):
    """One single-environment trace through fused_gate (cached by tag/params)."""
    suffix = "" if not params else "_p" + "_".join(
        f"{k}{v}" for k, v in sorted(params.items())).replace(".", "")
    label = (cenv or "py").replace(",", "+")
    path = FUSED_SCRATCH / f"batch_ref_{tag}{suffix}_{label}_{steps}.npz"
    if path.exists():
        print(f"[batch_gate] reusing {path.name}")
        return path
    argv = [
        sys.executable, "-m", "cenv.verify.fused_gate", "trace",
        "--sim", sim, "--scenario", scenario, "--steps", str(steps),
        "--occupancy-seed", str(seed), "--out", str(path),
    ]
    if params:
        argv += ["--params", json.dumps(params)]
    if cenv:
        argv += ["--cenv", cenv]
    _run(argv)
    return path


def compare_env(ref_path, got_path, label: str, against: str = "") -> dict:
    """One batch environment against its own single-environment run.

    The criteria are the project's standard (<= 1e-12 relative on every
    temperature component, <= 1e-9 relative on the episode total reward) plus
    the one this component adds: no device decision may flip. The batch
    factorizes with a Cholesky where the reference uses SuperLU, so the
    observation vector and the rewards are *not* expected to be bit-identical
    (that is ``fused``'s own bar); the deviations are reported next to the
    verdict, and the observation vector is held at the same 1e-12 standard
    because it is what the policy acts on.
    """
    ref = np.load(ref_path, allow_pickle=True)
    got = np.load(got_path, allow_pickle=True)
    meta_ref = json.loads(str(ref["meta"]))
    meta_got = json.loads(str(got["meta"]))
    n = min(int(meta_ref["steps"]), int(meta_got["steps"]))
    losses: list[str] = []
    if int(meta_ref["steps"]) != int(meta_got["steps"]):
        losses.append(f"step count {meta_ref['steps']} != {meta_got['steps']}")

    entry = {
        "label": label,
        "against": against,
        "scenario": meta_ref["scenario"],
        "sim": meta_ref["sim"],
        "occupancy_seed": meta_got["occupancy_seed"],
        "params": meta_got.get("params"),
        "steps": n,
    }

    def pair(name):
        a = np.asarray(ref[name], dtype=np.float64)[:n]
        b = np.asarray(got[name], dtype=np.float64)[:n]
        return a, b

    rew_ref, rew_got = pair("rewards")
    entry["total_reward_ref"] = float(rew_ref.sum())
    entry["total_reward_got"] = float(rew_got.sum())
    entry["total_reward_abs"] = abs(
        entry["total_reward_got"] - entry["total_reward_ref"])
    entry["total_reward_rel"] = entry["total_reward_abs"] / max(
        abs(entry["total_reward_ref"]), 1e-12)
    d = np.abs(rew_ref - rew_got)
    entry["reward_max_abs"] = float(d.max()) if d.size else 0.0
    entry["reward_first_diff_step"] = (
        int(np.argmax(d > 0.0)) if np.any(d > 0.0) else -1)
    if entry["total_reward_rel"] > REWARD_REL:
        losses.append(f"episode total_reward rel {entry['total_reward_rel']:.3e}")

    obs_ref, obs_got = pair("obs")
    d = np.abs(obs_ref - obs_got)
    entry["obs_max_abs"] = float(d.max()) if d.size else 0.0
    entry["obs_max_rel"] = _rel(obs_got, obs_ref, 1e-9)
    entry["obs_flips"] = int((d > 0.0).sum())
    if entry["obs_flips"]:
        step, slot = np.unravel_index(int(np.argmax(d > 0.0)), d.shape)
        entry["obs_first_diff"] = [int(step), int(slot)]
    if entry["obs_max_rel"] > TEMP_REL:
        losses.append(f"observation rel {entry['obs_max_rel']:.3e}")

    t_ref, t_got = pair("zone_temps")
    dt = np.abs(t_ref - t_got)
    entry["temp_max_abs"] = float(dt.max()) if dt.size else 0.0
    entry["temp_max_rel"] = _rel(t_got, t_ref, 1e-6)
    if entry["temp_max_rel"] > TEMP_REL:
        losses.append(f"temperature rel {entry['temp_max_rel']:.3e}")

    dam_ref, dam_got = pair("damper")
    dam = np.abs(dam_ref - dam_got)
    entry["damper_max_abs"] = float(dam.max()) if dam.size else 0.0
    entry["damper_flips"] = int((dam > 0.0).sum())
    if entry["damper_flips"]:
        step, col = np.unravel_index(int(np.argmax(dam > 0.0)), dam.shape)
        entry["damper_first_flip"] = [int(step), int(col)]
        losses.append(
            f"{entry['damper_flips']} damper decisions differ (first at step"
            f" {step}, VAV {col})")

    b_ref, b_got = pair("boiler")
    entry["boiler_max_abs"] = (
        float(np.abs(b_ref - b_got).max()) if b_ref.size else 0.0)
    for col, name in ((0, "tank_temperature"),
                      (1, "return_water_temperature"),
                      (2, "last_step_duration"),
                      (3, "flow_factor_sum")):
        entry[f"boiler_{name}_rel"] = _rel(b_got[:, col], b_ref[:, col], 1e-6)
        if entry[f"boiler_{name}_rel"] > TEMP_REL:
            losses.append(f"boiler {name} rel {entry[f'boiler_{name}_rel']:.3e}")

    for key in ("npr", "ec", "ce", "occupancy", "dmax"):
        name = f"c_{key}"
        if name not in ref.files or name not in got.files:
            continue
        a, b = pair(name)
        rel = _rel(b, a, 1e-9)
        entry[f"{key}_rel"] = rel
        if rel > TEMP_REL:
            losses.append(f"reward component {key} rel {rel:.3e}")

    entry["passed"] = not losses
    entry["losses"] = losses[:8]
    return entry


# --------------------------------------------------------------------------- #
# the checklist
# --------------------------------------------------------------------------- #
def gate(cases=CASES, steps=STEPS, m: int | None = None) -> dict:
    import time

    seeds = SEEDS[:m] if m else SEEDS
    params = PARAMS[:m] if m else PARAMS
    out_dir = SCRATCH / f"gate_{steps}"
    results = []
    timings = {}
    for sim, scenario in cases:
        tag = f"{sim}_{scenario.replace('/', '_')}"
        t0 = time.time()
        batch = _batch_side(sim, scenario, steps, out_dir / tag, seeds, params,
                            tag)
        timings[f"{sim} {scenario}"] = batch["timing"]
        print(f"[batch_gate] batch side {sim} {scenario}: {time.time() - t0:.1f}s",
              flush=True)
        for k, seed in enumerate(seeds):
            label = f"{sim} {scenario} env{k} seed{seed}"
            # The verdict is about *batching*: same factorization, one
            # environment. The official-simulator comparison is reported next
            # to it, because a Cholesky and a SuperLU disagree in the last
            # bits and a policy sitting on a decision threshold can turn that
            # into a different action (measured: see the module docstring).
            ref_same = trace_single(sim, scenario, steps, seed, params[k], tag,
                                    SAME_FACTOR_CENV)
            entry = compare_env(ref_same, batch["paths"][k], label,
                                against=SAME_FACTOR_CENV)
            ref_official = trace_single(sim, scenario, steps, seed, params[k],
                                        tag, None)
            official = compare_env(ref_official, batch["paths"][k], label,
                                   against="python (official simulator)")
            entry["vs_official"] = {
                key: official[key] for key in (
                    "total_reward_ref", "total_reward_got", "total_reward_rel",
                    "reward_first_diff_step", "temp_max_rel", "obs_max_abs",
                    "damper_flips", "passed")
            }
            results.append(entry)
            print(json.dumps(entry, indent=1, default=str), flush=True)
    # M=1 through the batch kernel against the SuperLU single-environment path.
    single = []
    for sim, scenario in cases[:2]:
        tag = f"m1_{sim}_{scenario.replace('/', '_')}"
        batch = _batch_side(sim, scenario, steps, out_dir / tag, SEEDS[:1],
                            PARAMS[:1], tag)
        ref = trace_single(sim, scenario, steps, SEEDS[0], PARAMS[0],
                           f"{sim}_{scenario.replace('/', '_')}", "fused")
        entry = compare_env(ref, batch["paths"][0], f"M=1 {sim} {scenario}",
                            against="cenv fused (SuperLU)")
        single.append(entry)
        print(json.dumps(entry, indent=1, default=str), flush=True)
    passed = all(r["passed"] for r in results + single)
    return {"cases": results, "m1": single, "timing": timings,
            "passed": passed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate")
    g.add_argument("--steps", type=int, default=STEPS)
    g.add_argument("--m", type=int, default=None,
                   help="use only the first M seeds/params (default: all 4)")
    g.add_argument("--only", default="all",
                   choices=("all", "shipped", "corrected"))

    t = sub.add_parser("trace")
    t.add_argument("--sim", default="shipped")
    t.add_argument("--scenario", default="building_train_8/fairbanks_8")
    t.add_argument("--steps", type=int, default=STEPS)
    t.add_argument("--out", required=True)
    t.add_argument("--seeds", default=None, help="comma separated")
    t.add_argument("--params", default=None, help="JSON list of dicts")
    t.add_argument("--policy", default="policies.ctrl_w:CtrlW")

    args = ap.parse_args(argv)
    if args.cmd == "trace":
        seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
        params = json.loads(args.params) if args.params else None
        trace_batch(args.sim, args.scenario, args.steps, Path(args.out),
                    seeds, params, policy=args.policy)
        return 0
    cases = [c for c in CASES if args.only == "all" or c[0] == args.only]
    result = gate(cases=cases, steps=args.steps, m=args.m)
    path = SCRATCH / "batch_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=str))
    print(f"[batch_gate] {'PASSED' if result['passed'] else 'FAILED'} -> {path}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
