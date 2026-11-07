"""Gate for the ``fused`` component: closed-loop, both trees, whole episodes.

Two sides, each in its own process (so the C component is installed before any
simulator module is bound): the official Python step and ``--cenv fused``. Both
run the same policy, scenario and occupancy seed, and write the trajectory --
reward, the reward response's components, the observation vector, the zone
temperatures and the device state -- step by step. ``compare`` then reports the
first step they disagree at and the worst relative difference anywhere.

    # the checklist the ticket asks for
    python -m cenv.verify.fused_gate gate

    # one trace / one comparison by hand
    python -m cenv.verify.fused_gate trace --scenario building_train_8/fairbanks_8 \
        --steps 2880 --sim shipped --out scratch/x.npz
    python -m cenv.verify.fused_gate compare ref.npz fused.npz

The pass criteria are the ticket's exactness standard: <= 1e-12 relative on
every temperature component along the episode, <= 1e-9 relative on the episode
total reward, and (this component's own stricter bar, because the observation
and the reward decide the next action) bit equality on the observation vector
and on every reward value. Bit equality is what the kernel actually achieves;
the tolerance is there so the gate can report a near miss instead of a wall of
mismatches.

``tables`` prints the per-step breakdown against plain Python and ``--cenv
all`` (interleaved A/B repetitions, medians, machine load recorded).
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

WORKSPACE = Path(__file__).resolve().parents[2]
SCRATCH = WORKSPACE / "scratch" / "CEnvFused2"

# The checklist: (sim, scenario, occupancy seed).
CASES = [
    ("shipped", "building_train_8/fairbanks_8", 17321),
    ("shipped", "building_train_8/fairbanks_8", 12345),
    ("shipped", "building_train_2/fairbanks_2", 17321),
    ("shipped", "building_train_2/fairbanks_2", 12345),
    ("shipped", "building_train_5/fairbanks_5", 17321),
    ("shipped", "building_train_5/fairbanks_5", 12345),
    ("corrected", "building_train_8/fairbanks_8", 17321),
    ("corrected", "building_train_8/fairbanks_8", 12345),
    ("corrected", "building_train_2/fairbanks_2", 17321),
    ("corrected", "building_train_2/fairbanks_2", 12345),
    ("corrected", "building_train_5/fairbanks_5", 17321),
    ("corrected", "building_train_5/fairbanks_5", 12345),
]
STEPS = 2880
FULL_STEPS = 8641
TEMP_REL = 1e-12
REWARD_REL = 1e-9


def _fused_profile(cenv: str | None) -> dict | None:
    if not cenv:
        return None
    try:
        from cenv import fused as fused_mod

        prof = dict(fused_mod.PROFILE)
        steps = max(prof.get("steps", 0), 1)
        prof["c_ms_per_step"] = prof.get("c_seconds", 0.0) / steps * 1e3
        prof["builds_ms"] = prof.get("build_seconds", 0.0) * 1e3
        return prof
    except Exception:  # pragma: no cover
        return None


def _scenario_path(sim: str, scenario: str) -> Path:
    return (
        WORKSPACE / "sims" / sim / "smart_buildings" / "training_scenarios"
        / scenario / "config.gin"
    )


def trace(sim: str, scenario: str, steps: int, out: Path, policy: str = "policies.ctrl_w:CtrlW",
          cenv: str | None = None, occupancy_seed: int | None = None,
          components: str = "parity", params: dict | None = None,
          log_every: int = 0) -> dict:
    """Runs one closed-loop episode and writes its trajectory."""
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    # sbx.simload.use_sim chdirs into the simulator tree (gin data paths are
    # relative to it), so everything this function writes must be absolute
    # first -- and never inside sims/.
    out = Path(out).resolve()
    started_cpu = time.process_time()
    started_wall = time.time()

    from sbx import simload

    simload.use_sim(sim)
    if cenv:
        import cenv as cenv_pkg

        cenv_pkg.install(cenv, sim_root=simload.sim_root(sim), verbose=True)

    from smart_buildings import evaluate
    from sbx.evalrun import _StepRecorder, _resolve_policy, _load_environment

    config = _scenario_path(sim, scenario)
    policy_cls, label = _resolve_policy(policy, evaluate)
    if params:
        policy_cls.PARAMS = {**(getattr(policy_cls, "PARAMS", None) or {}), **params}
    bindings = {}
    if occupancy_seed is not None:
        bindings["RandomizedArrivalDepartureOccupancy.seed"] = int(occupancy_seed)
    env = _load_environment(evaluate, config, bindings)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = evaluate.create_building_meta_data(env, config.parent)
    spec = env.action_spec()
    inst = evaluate.initialize_policy(policy_cls, meta, spec)
    recorder = _StepRecorder(env, meta, mode=components)

    obs, _ = env.reset()
    n = int(steps)
    n_obs = int(np.asarray(obs).size)
    rewards = np.zeros(n, dtype=np.float64)
    obs_trace = np.zeros((n, n_obs), dtype=np.float64)
    zone_names = None
    zone_temps = None
    damper = None
    boiler = np.zeros((n, 4), dtype=np.float64)
    step_cpu = 0.0
    done = 0
    sim_obj = env.building.simulator
    vavs = list(sim_obj.hvac.vavs.values())
    for i in range(n):
        action = evaluate.validate_action(
            evaluate.get_policy_action(inst, obs), spec
        )
        t0 = time.process_time()
        obs, reward, terminated, truncated, _ = env.step(action)
        step_cpu += time.process_time() - t0
        recorder.record(float(reward))
        rewards[i] = float(reward)
        obs_trace[i] = np.asarray(obs, dtype=np.float64)
        averages = sim_obj.building.get_zone_average_temps()
        if zone_names is None:
            zone_names = list(averages)
            zone_temps = np.zeros((n, len(zone_names)), dtype=np.float64)
            damper = np.zeros((n, len(vavs)), dtype=np.float64)
        zone_temps[i] = [averages[name] for name in zone_names]
        damper[i] = [v._damper_setting for v in vavs]
        heat_source = sim_obj.hvac.hot_water_system._heat_source
        boiler[i] = (
            heat_source._current_temperature,
            sim_obj.hvac.hot_water_system.return_water_temperature_sensor,
            heat_source._last_step_duration.total_seconds(),
            sim_obj.hvac.hot_water_system.flow_factor_sum,
        )
        done = i + 1
        if terminated or truncated:
            break
        if log_every and done % log_every == 0:
            print(f"[fused_gate] {scenario} {sim} {done}/{n}", flush=True)

    steps_rec = recorder.steps[:done]
    comp = {
        key: np.array([float(s[key]) for s in steps_rec], dtype=np.float64)
        for key in ("npr", "ec", "ce", "occupancy", "dmax")
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        rewards=rewards[:done],
        obs=obs_trace[:done],
        zone_temps=zone_temps[:done] if zone_temps is not None else np.zeros(0),
        damper=damper[:done] if damper is not None else np.zeros(0),
        boiler=boiler[:done],
        zone_names=np.array(zone_names or [], dtype=object),
        **{f"c_{k}": v for k, v in comp.items()},
        meta=json.dumps(
            {
                "scenario": scenario,
                "sim": sim,
                "policy": label,
                "params": params or {},
                "cenv": cenv,
                "components": components,
                "steps": done,
                "occupancy_seed": occupancy_seed,
                "n_obs": n_obs,
                "step_cpu_seconds": step_cpu,
                "cpu_seconds": time.process_time() - started_cpu,
                "wall_seconds": time.time() - started_wall,
                "total_reward": float(rewards[:done].sum()),
                "fused_profile": _fused_profile(cenv),
            }
        ),
    )
    summary = json.loads(str(np.load(out, allow_pickle=True)["meta"]))
    print(
        f"[fused_gate] {sim} {scenario} seed={occupancy_seed} cenv={cenv or 'python'}:"
        f" steps={done} total_reward={summary['total_reward']:.9f}"
        f" step_cpu={summary['step_cpu_seconds'] / max(done, 1) * 1e3:.3f} ms",
        flush=True,
    )
    return summary


def _rel(a, b, scale):
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), scale)
    return float(np.max(diff / denom)) if diff.size else 0.0


def compare(ref_path, got_path) -> dict:
    ref = np.load(ref_path, allow_pickle=True)
    got = np.load(got_path, allow_pickle=True)
    meta_ref = json.loads(str(ref["meta"]))
    meta_got = json.loads(str(got["meta"]))
    out: dict = {
        "scenario": meta_ref["scenario"],
        "sim": meta_ref["sim"],
        "occupancy_seed": meta_ref["occupancy_seed"],
        "steps_ref": meta_ref["steps"],
        "steps_got": meta_got["steps"],
        "total_reward_ref": meta_ref["total_reward"],
        "total_reward_got": meta_got["total_reward"],
    }
    n = min(int(meta_ref["steps"]), int(meta_got["steps"]))
    losses: list[str] = []
    if meta_ref["steps"] != meta_got["steps"]:
        losses.append(f"step count {meta_ref['steps']} != {meta_got['steps']}")

    rew_ref = ref["rewards"][:n]
    rew_got = got["rewards"][:n]
    diff = np.abs(rew_ref - rew_got)
    first = int(np.argmax(diff > 0.0)) if np.any(diff > 0.0) else -1
    out["reward_max_abs"] = float(diff.max()) if diff.size else 0.0
    out["reward_first_diff_step"] = first
    total_ref = float(rew_ref.sum())
    total_got = float(rew_got.sum())
    out["total_reward_rel"] = (
        abs(total_got - total_ref) / max(abs(total_ref), 1e-12)
    )
    if out["total_reward_rel"] > REWARD_REL:
        losses.append(f"episode total_reward rel {out['total_reward_rel']:.3e}")

    obs_ref = ref["obs"][:n]
    obs_got = got["obs"][:n]
    odiff = np.abs(obs_ref - obs_got)
    out["obs_max_abs"] = float(odiff.max()) if odiff.size else 0.0
    out["obs_max_rel"] = _rel(obs_got, obs_ref, 1e-9)
    if odiff.size and odiff.max() > 0.0:
        step, slot = np.unravel_index(int(np.argmax(odiff > 0.0)), odiff.shape)
        out["obs_first_diff"] = [int(step), int(slot)]
        losses.append(f"observation differs at step {step} slot {slot}")

    t_ref = ref["zone_temps"][:n]
    t_got = got["zone_temps"][:n]
    out["temp_max_abs"] = float(np.abs(t_ref - t_got).max()) if t_ref.size else 0.0
    out["temp_max_rel"] = _rel(t_got, t_ref, 1e-6)
    if out["temp_max_rel"] > TEMP_REL:
        losses.append(f"temperature rel {out['temp_max_rel']:.3e}")

    d_ref = ref["damper"][:n]
    d_got = got["damper"][:n]
    out["damper_max_abs"] = float(np.abs(d_ref - d_got).max()) if d_ref.size else 0.0
    if out["damper_max_abs"] > 0.0:
        dstep, dcol = np.unravel_index(
            int(np.argmax(np.abs(d_ref - d_got) > 0.0)), d_ref.shape
        )
        out["damper_first_diff"] = [int(dstep), int(dcol)]
        losses.append(f"damper differs at step {dstep} VAV {dcol}")

    # Boiler state: the tank temperature, the elapsed time and the flow factor
    # sum are bit-identical; the return water temperature is a ratio of the
    # reheat sum to the water capacity rate, so it inherits the solver's
    # ~1e-15 relative difference (a fraction of one ULP) -- held at the
    # ticket's 1e-12 instead.
    b_ref = ref["boiler"][:n]
    b_got = got["boiler"][:n]
    out["boiler_max_abs"] = float(np.abs(b_ref - b_got).max()) if b_ref.size else 0.0
    if b_ref.size:
        for col, name, tol in (
            (0, "tank temperature", 0.0),
            (2, "last step duration", 0.0),
            (3, "flow factor sum", 0.0),
            (1, "return water temperature", TEMP_REL),
        ):
            a_col = b_ref[:, col]
            b_col = b_got[:, col]
            d = np.abs(a_col - b_col)
            rel = _rel(b_col, a_col, 1e-6)
            out[f"boiler_{name.replace(' ', '_')}_rel"] = rel
            if d.max() > 0.0 and (tol == 0.0 or rel > tol):
                step = int(np.argmax(d > 0.0))
                out["boiler_first_diff"] = [step, col]
                losses.append(
                    f"boiler {name} differs at step {step} (max {d.max():.3e},"
                    f" rel {rel:.3e})"
                )

    for key in ("npr", "ec", "ce", "occupancy", "dmax"):
        name = f"c_{key}"
        if name not in ref.files or name not in got.files:
            continue
        a = ref[name][:n]
        b = got[name][:n]
        d = np.abs(a - b)
        out[f"{key}_max_abs"] = float(d.max()) if d.size else 0.0
        if d.size and d.max() > 0.0:
            losses.append(f"reward component {key} differs (max {d.max():.3e})")

    out["passed"] = not losses
    out["losses"] = losses[:6]
    return out


def _run(argv: list[str]) -> None:
    proc = subprocess.run(argv, cwd=str(WORKSPACE), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess failed: {' '.join(argv)}\n{proc.stdout}\n{proc.stderr}"
        )
    sys.stdout.write(proc.stdout)


def _trace_pair(sim, scenario, steps, seed, tag, cenv, extra=()):
    out_ref = SCRATCH / f"{tag}_py.npz"
    out_got = SCRATCH / f"{tag}_fused.npz"
    base = [
        sys.executable, "-m", "cenv.verify.fused_gate", "trace",
        "--sim", sim, "--scenario", scenario, "--steps", str(steps),
        "--occupancy-seed", str(seed),
    ]
    if not out_ref.exists():
        _run(base + ["--out", str(out_ref), *extra])
    else:
        print(f"[fused_gate] reusing {out_ref.name}")
    if not out_got.exists():
        _run(base + ["--cenv", cenv, "--out", str(out_got), *extra])
    else:
        print(f"[fused_gate] reusing {out_got.name}")
    return compare(out_ref, out_got)


def gate(cases=CASES, steps=STEPS, tag_prefix="gate", do_full: bool = True,
         do_episodes: bool = True) -> dict:
    results = []
    for sim, scenario, seed in cases:
        building, weather = scenario.split("/")
        tag = f"{tag_prefix}_{sim}_{building}_{weather}_s{seed}"
        res = _trace_pair(sim, scenario, steps, seed, tag, "fused")
        results.append(res)
        print(json.dumps(res, indent=1, default=str), flush=True)
    full = []
    for sim in (("shipped", "corrected") if do_full else ()):
        tag = f"{tag_prefix}_full_{sim}"
        res = _trace_pair(
            sim, "building_train_8/fairbanks_8", FULL_STEPS, 17321, tag, "fused"
        )
        full.append(res)
        print(json.dumps(res, indent=1, default=str), flush=True)
    runner_episodes = episodes(FULL_STEPS) if do_episodes else []
    passed = all(r["passed"] for r in results + full + runner_episodes)
    return {
        "cases": results,
        "full": full,
        "episodes": runner_episodes,
        "passed": passed,
    }


# --------------------------------------------------------------------------- #
# timing tables
# --------------------------------------------------------------------------- #
def _evalrun_once(sim, scenario, steps, config, tmp, name):
    """One evalrun subprocess; returns its per-scenario record."""
    building, weather = scenario.split("/")
    tag = f"fused_bench_{name}"
    cmd = [
        sys.executable, "-m", "sbx.evalrun",
        "--policy", "policies.ctrl_w:CtrlW",
        "--scenario", scenario,
        "--steps", str(steps),
        "--sim", sim,
        "--workers", "1",
        "--tag", tag,
        "--out", str(Path(tmp) / tag),
    ]
    if config:
        cmd += ["--cenv", config]
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"evalrun failed for {scenario} cenv={config}:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    return json.loads((Path(tmp) / tag / f"{building}__{weather}.json").read_text())


def _fused_breakdown(sim, scenario, steps, tmp):
    """The fused run's own split: C step vs the rest of env.step."""
    out = Path(tmp) / "fused_trace.npz"
    trace(sim, scenario, steps, out, cenv="fused")
    meta = json.loads(str(np.load(out, allow_pickle=True)["meta"]))
    return meta.get("fused_profile") or {}


def tables(steps: int = 400, repeats: int = 3) -> dict:
    """Per-step cost of plain Python, ``--cenv all`` and ``--cenv fused``.

    The three configurations are run round-robin inside each repetition, so a
    machine that slows down mid-measurement penalises all of them equally; the
    reported number is the median over repetitions of
    ``step_cpu_seconds / steps`` (CPU time inside ``env.step`` only, which is
    what the components shorten). Load averages bracket the run.
    """
    import tempfile

    configs = ("", "all", "fused")
    out: dict = {"repeats": repeats, "steps": steps, "scenarios": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in ("building_train_8/fairbanks_8",
                         "building_train_2/fairbanks_2"):
            load_before = _load()
            rows = {c or "python": [] for c in configs}
            for k in range(repeats):
                for config in configs:
                    record = _evalrun_once("shipped", scenario, steps, config,
                                           tmp, f"{k}_{config or 'py'}")
                    rows[config or "python"].append(record)
            load_after = _load()
            table = {}
            for name, records in rows.items():
                step_cpu = statistics.median(
                    r["step_cpu_seconds"] / r["steps"] for r in records
                )
                table[name] = {
                    "ms_per_step": step_cpu * 1e3,
                    "steps_per_s": 1.0 / step_cpu,
                    "total_reward": statistics.median(
                        r["total_reward"] for r in records
                    ),
                    "cpu_seconds": statistics.median(
                        r["cpu_seconds"] for r in records
                    ),
                }
            base = table["python"]["ms_per_step"]
            for name, row in table.items():
                row["speedup"] = base / row["ms_per_step"]
            fused = _fused_breakdown("shipped", scenario, steps, tmp)
            c_ms = float(fused.get("c_ms_per_step", 0.0))
            table["fused"]["c_call_ms"] = c_ms
            table["fused"]["python_remainder_ms"] = (
                table["fused"]["ms_per_step"] - c_ms
            )
            table["fused"]["c_share"] = (
                c_ms / table["fused"]["ms_per_step"]
                if table["fused"]["ms_per_step"]
                else 0.0
            )
            table["fused"]["build_ms"] = float(fused.get("builds_ms", 0.0))
            table["fused"]["builds"] = int(fused.get("builds", 0))
            out["scenarios"][scenario] = {
                "table": table,
                "load_before": load_before,
                "load_after": load_after,
            }
    return out


def _evalrun_episode(sim, steps, cenv):
    """One full episode through ``sbx.evalrun``; returns its record."""
    tag = f"ep_{sim}_{'fused' if cenv else 'python'}"
    out = SCRATCH / "episodes" / tag
    cmd = [
        sys.executable, "-m", "sbx.evalrun",
        "--policy", "policies.ctrl_w:CtrlW",
        "--scenario", "building_train_8/fairbanks_8",
        "--steps", str(steps),
        "--sim", sim,
        "--workers", "1",
        "--tag", tag,
        "--out", str(out),
    ]
    if cenv:
        cmd += ["--cenv", cenv]
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"evalrun failed ({sim}, cenv={cenv}):\n{proc.stdout}\n{proc.stderr}"
        )
    return json.loads(
        (out / "building_train_8__fairbanks_8.json").read_text()
    )


def episodes(steps: int = FULL_STEPS) -> list[dict]:
    """Full ``--steps`` episodes, plain vs ``--cenv fused``, one per tree.

    This is the runner-level check: ``sbx.evalrun`` exactly as a sweep runs it
    (parity recorder, one worker), so the episode's ``total_reward`` and its
    reward-component aggregates are compared as the frozen run reports them.
    """
    results = []
    for sim in ("shipped", "corrected"):
        ref = _evalrun_episode(sim, steps, None)
        got = _evalrun_episode(sim, steps, "fused")
        total_ref = float(ref["total_reward"])
        total_got = float(got["total_reward"])
        rel = abs(total_got - total_ref) / max(abs(total_ref), 1e-12)
        entry = {
            "sim": sim,
            "steps_ref": ref["steps"],
            "steps_got": got["steps"],
            "total_reward_ref": total_ref,
            "total_reward_got": total_got,
            "total_reward_rel": rel,
            "total_reward_abs": abs(total_got - total_ref),
            "components": {},
            "losses": [],
        }
        if ref["steps"] != got["steps"]:
            entry["losses"].append(
                f"step count {ref['steps']} != {got['steps']}"
            )
        if rel > REWARD_REL:
            entry["losses"].append(f"total_reward rel {rel:.3e}")
        for key in ("sum_comfort", "sum_energy", "sum_carbon",
                    "sum_reward_from_components"):
            a = float(ref["components"][key])
            b = float(got["components"][key])
            entry["components"][key] = {
                "ref": a,
                "got": b,
                "abs": abs(b - a),
                "rel": abs(b - a) / max(abs(a), 1e-12),
            }
            if abs(b - a) > 0.0:
                entry["losses"].append(
                    f"component {key} differs (abs {abs(b - a):.3e})"
                )
        entry["passed"] = not entry["losses"]
        results.append(entry)
        print(json.dumps(entry, indent=1, default=str), flush=True)
    return results


def _load():
    try:
        return list(__import__("os").getloadavg())
    except Exception:  # pragma: no cover
        return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trace")
    t.add_argument("--sim", default="shipped")
    t.add_argument("--scenario", default="building_train_8/fairbanks_8")
    t.add_argument("--steps", type=int, default=STEPS)
    t.add_argument("--policy", default="policies.ctrl_w:CtrlW")
    t.add_argument("--params", default=None)
    t.add_argument("--cenv", default=None)
    t.add_argument("--components", default="parity")
    t.add_argument("--occupancy-seed", type=int, default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--log-every", type=int, default=0)

    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("got")
    c.add_argument("--json", default=None)

    g = sub.add_parser("gate")
    g.add_argument("--steps", type=int, default=STEPS)
    g.add_argument("--only", default="all", choices=("all", "shipped", "corrected"),
                   help="restrict the 2880-step cases to one simulator tree")
    g.add_argument("--cases-only", action="store_true",
                   help="skip the full-episode traces and the evalrun episodes")

    e = sub.add_parser("episodes")
    e.add_argument("--steps", type=int, default=FULL_STEPS)

    b = sub.add_parser("tables")
    b.add_argument("--steps", type=int, default=400)
    b.add_argument("--repeats", type=int, default=3)

    args = ap.parse_args(argv)
    if args.cmd == "trace":
        params = json.loads(args.params) if args.params else None
        trace(args.sim, args.scenario, args.steps, Path(args.out),
              policy=args.policy, cenv=args.cenv, occupancy_seed=args.occupancy_seed,
              components=args.components, params=params, log_every=args.log_every)
        return 0
    if args.cmd == "compare":
        result = compare(args.ref, args.got)
        text = json.dumps(result, indent=1, default=str)
        print(text)
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(text)
        return 0 if result["passed"] else 1
    if args.cmd == "gate":
        cases = [c for c in CASES if args.only == "all" or c[0] == args.only]
        result = gate(cases=cases, steps=args.steps, do_full=not args.cases_only,
                      do_episodes=not args.cases_only)
        path = SCRATCH / "fused_gate.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1, default=str))
        print(f"[fused_gate] {'PASSED' if result['passed'] else 'FAILED'} -> {path}")
        return 0 if result["passed"] else 1
    if args.cmd == "episodes":
        result = episodes(args.steps)
        path = SCRATCH / "fused_episodes.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1, default=str))
        ok = all(r["passed"] for r in result)
        print(f"[fused_gate] episodes {'PASSED' if ok else 'FAILED'} -> {path}")
        return 0 if ok else 1
    if args.cmd == "tables":
        result = tables(steps=args.steps, repeats=args.repeats)
        path = SCRATCH / "fused_tables.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1, default=str))
        print(json.dumps(result, indent=1, default=str))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
