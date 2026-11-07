"""Per-step CPU before/after for one or more cenv component sets.

Runs ``sbx.evalrun`` as a subprocess once per configuration (so each measurement
gets a clean interpreter, a cold numba cache and one scenario in one process),
then reports the record's ``step_cpu_seconds / steps`` -- process CPU time spent
*inside* ``env.step`` only, which is what the C components shorten -- together
with the total reward so a speed-up that changed behaviour is visible.

    starter-kit/.venv/bin/python -m cenv.verify.bench \
        --scenario building_train_8/fairbanks_8 --steps 300 \
        --policy policies.ctrl_w:CtrlW --config "" bookkeeping --repeats 2

Configurations are ``--cenv`` values, so ``""`` is pure Python, ``bookkeeping``
is one component and ``all`` is every gate-passed component.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]


def run_one(scenario: str, steps: int, policy: str, sim: str, cenv: str,
            start_offset_days: int, out_dir: Path, components: str = "parity",
            extra: list[str] | None = None) -> dict:
    tag = f"bench_{scenario.replace('/', '_')}_{cenv or 'py'}_{int(time.time() * 1000) % 10**9}"
    cmd = [
        sys.executable, "-m", "sbx.evalrun",
        "--policy", policy,
        "--scenario", scenario,
        "--steps", str(steps),
        "--sim", sim,
        "--workers", "1",
        "--components", components,
        "--start-offset-days", str(start_offset_days),
        "--tag", tag,
        "--out", str(out_dir / tag),
    ]
    if cenv:
        cmd += ["--cenv", cenv]
    if extra:
        cmd += extra
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"evalrun failed:\n{proc.stdout}\n{proc.stderr}")
    building, weather = scenario.split("/")
    record = out_dir / tag / f"{building}__{weather}.json"
    return json.loads(record.read_text())


def bench(scenario: str, steps: int, policy: str, sim: str, configs: list[str],
          repeats: int = 1, start_offset_days: int = 0,
          components: str = "parity", extra: list[str] | None = None,
          verbose: bool = True) -> dict:
    results: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        for cenv in configs:
            runs = []
            for _ in range(max(1, repeats)):
                runs.append(
                    run_one(scenario, steps, policy, sim, cenv,
                            start_offset_days, out_dir, components, extra)
                )
            results[cenv] = runs
    table = {}
    base = None
    for cenv, runs in results.items():
        step_cpu = statistics.median(r["step_cpu_seconds"] / r["steps"] for r in runs)
        total = statistics.median(r["total_reward"] for r in runs)
        cpu = statistics.median(r["cpu_seconds"] for r in runs)
        if base is None:
            base = step_cpu
        table[cenv or "python"] = {
            "step_cpu_s": step_cpu,
            "steps_per_cpu_s": 1.0 / step_cpu if step_cpu else float("inf"),
            "speedup": (base / step_cpu) if step_cpu else float("inf"),
            "cpu_seconds": cpu,
            "total_reward": total,
            "mean_per_step_reward": total / runs[0]["steps"],
            "steps": runs[0]["steps"],
            "sim": sim,
            "scenario": scenario,
            "policy": runs[0]["policy"],
        }
    if verbose:
        print(f"\n{scenario}  {sim}  {policy}  {runs[0]['steps']} steps  "
              f"components={components}")
        print(f"{'cenv':<22} {'s/step':>10} {'steps/s':>9} {'speedup':>8} "
              f"{'total_reward':>14}")
        for name, row in table.items():
            print(f"{name:<22} {row['step_cpu_s']:10.5f} {row['steps_per_cpu_s']:9.2f} "
                  f"{row['speedup']:7.2f}x {row['total_reward']:14.4f}")
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", action="append",
                    default=["building_train_8/fairbanks_8"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--policy", default="policies.ctrl_w:CtrlW")
    ap.add_argument("--sim", default="shipped")
    ap.add_argument("--config", action="append", default=None,
                    help="a --cenv value to measure ('' for pure Python); "
                         "repeatable, default '' then 'all'")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--start-offset-days", type=int, default=0)
    ap.add_argument("--eval-components", default="parity",
                    choices=["parity", "hook"])
    ap.add_argument("--extra", action="append", default=None,
                    help="extra evalrun flags, e.g. --extra=--fast")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    configs = args.config if args.config is not None else ["", "all"]
    out = {}
    for scenario in args.scenario:
        out[scenario] = bench(
            scenario, args.steps, args.policy, args.sim, configs,
            repeats=args.repeats, start_offset_days=args.start_offset_days,
            components=args.eval_components, extra=args.extra,
        )
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
