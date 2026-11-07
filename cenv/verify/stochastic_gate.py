"""Gate for cenv.stochastic.

Three layers, in increasing cost:

``unit``      the pieces in isolation -- a million MT19937 draws per stream
              against numpy's legacy RandomState and numba's np.random, the
              weather CSV/interp/kelvin chain against pandas and np.interp for
              every station, the convection shim against the numba kernel (and
              its candidate CSR against ``_room_arrays``) on four room shapes,
              and the occupancy shim against the Python model driven with the
              simulator's own call pattern for 2880 steps, generator state
              included;
``episode``   full 2880-step shadow runs on building_train_8/fairbanks_8 with
              occupancy seeds 17321 and 12345, plus a non-shadow run of the same
              episode: shadow proves component-level equality every step, and
              the pair proves the episode totals are bit-identical. Writes
              ``results/cenv_stochastic/episode_gate.json``;
``bench``     800-step CPU per step of building_train_8/fairbanks_8 and
              building_train_2/fairbanks_2 with policies.ctrl_w:CtrlW, cenv off
              vs on, plus the totals of the two runs.

Usage::

    starter-kit/.venv/bin/python -m cenv.verify.stochastic_gate unit
    starter-kit/.venv/bin/python -m cenv.verify.stochastic_gate episode
    starter-kit/.venv/bin/python -m cenv.verify.stochastic_gate bench
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
PY = str(WORKSPACE / "starter-kit" / ".venv" / "bin" / "python")
SIM = "corrected"
STEPS_EPISODE = 2880
STEPS_BENCH = 800
POLICY = "policies.ctrl_w:CtrlW"
RESULTS = WORKSPACE / "results" / "cenv_stochastic"


def _bootstrap(sim: str = SIM):
    """Load the simulator tree the way sbx.evalrun does."""
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    import sbx.simload as simload

    simload.use_sim(sim)
    return simload


# --------------------------------------------------------------------------- #
# unit
# --------------------------------------------------------------------------- #
def _unit_streams() -> bool:
    from cenv.verify import stochastic_stream

    return stochastic_stream.main() == 0


def _unit_weather() -> bool:
    import ctypes

    import numpy as np
    import pandas as pd

    from cenv import stochastic as shim
    from cenv.verify.stochastic_stream import _cmp
    from smart_buildings.smart_control.simulator import weather_controller as wxm

    ok = True
    files = sorted(
        (WORKSPACE / "sims" / SIM / "smart_buildings" / "weather" / "csv").glob("*.csv")
    )
    checked = 0
    for path in files:
        handle = shim._wx_load(str(path).encode())
        if handle <= 0:
            print(f"  FAIL parse {path.name}: no handle")
            ok = False
            continue
        try:
            lo = ctypes.c_double()
            hi = ctypes.c_double()
            n = ctypes.c_int64()
            shim._wx_bounds(handle, ctypes.byref(lo), ctypes.byref(hi), ctypes.byref(n))
            xp = np.empty(n.value)
            tf = np.empty(n.value)
            hu = np.empty(n.value)
            shim._wx_copy(handle, xp, tf, hu)

            df = pd.read_csv(path)
            df["Time"] = pd.to_datetime(df["Time"], utc=True)
            index = (df["Time"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
            parsed_same = (
                n.value == len(df)
                and np.array_equal(xp, index.to_numpy())
                and np.array_equal(tf, df["TempF"].to_numpy())
                and np.array_equal(hu, df["Humidity"].to_numpy())
            )
            if not parsed_same:
                print(f"  FAIL csv {path.name}")
                ok = False
                continue

            # The simulator queries whole seconds only, so probe whole seconds:
            # every node, node + 137 s, node + 300 s (which clamps at the end)
            # and 500 random instants, against np.interp.
            rng = np.random.default_rng(20261105)
            probes = np.concatenate(
                [
                    xp,
                    xp + 137.0,
                    xp + 300.0,
                    np.floor(rng.uniform(xp[0], xp[-1], 500)),
                ]
            ).astype(np.int64)
            out = ctypes.c_double()
            got = np.empty(probes.size)
            for i, seconds in enumerate(probes):
                rc = shim._wx_query(handle, int(seconds) * 1_000_000_000, 0,
                                    ctypes.byref(out))
                if rc not in (0, 1):  # 1 == outside the table, np.interp clamps
                    raise AssertionError(f"cenv_weather_query rc={rc} at {seconds}")
                got[i] = out.value
            ref = np.interp(probes.astype(np.float64), xp, tf)
            if not np.array_equal(got, ref):
                bad = np.flatnonzero(got != ref)
                print(
                    f"  FAIL interp {path.name}: {bad.size} of {probes.size} "
                    f"differ, first x={probes[bad[0]]} "
                    f"c={got[bad[0]]!r} py={ref[bad[0]]!r}"
                )
                ok = False
                continue

            # get_current_temp == fahrenheit_to_kelvin(np.interp(...))
            temps = np.array([shim._f2k(v) for v in got])
            ref_temps = np.array([wxm.utils.fahrenheit_to_kelvin(v) for v in ref])
            if not np.array_equal(temps, ref_temps):
                print(f"  FAIL kelvin {path.name}")
                ok = False
                continue
            checked += 1
        finally:
            shim._wx_destroy(handle)

    print(f"  weather: {checked}/{len(files)} stations parsed + interp + kelvin")
    return ok


def _unit_convection() -> bool:
    import numpy as np

    from cenv import stochastic as shim
    from smart_buildings.smart_control.simulator import (
        stochastic_convection_simulator as conv,
    )

    def coords(kind, nrows=60, ncols=60):
        if kind == "rect":
            return [(r, c) for r in range(3, 43) for c in range(5, 35)]
        if kind == "single":
            return [(7, 7)]
        if kind == "lshape":
            return [
                (r, c)
                for r in range(2, 30)
                for c in range(4, 20)
                if r < 12 or c > 10
            ]
        if kind == "line":
            return [(9, c) for c in range(10, 25)]
        raise ValueError(kind)

    sim = conv.StochasticConvectionSimulator(p=0.1, distance=2, seed=5)
    ok = True
    trials = 0

    # The candidate CSR itself, against StochasticConvectionSimulator
    # ._room_arrays, and candidates_for() over the C rooms.
    for kind in ("rect", "single", "lshape", "line"):
        rooms = {kind: coords(kind)}
        temp = np.zeros((60 * 60,))
        temp = temp.reshape(60, 60)
        sim.apply_convection(rooms, temp)          # builds the C rooms
        cvs, ptr, idx = sim._room_arrays(kind, rooms[kind], 2, False)
        st = sim.__dict__["_cenv_conv"]
        handle = st.rooms[0]
        nnz = shim._conv_room_nnz(handle)
        got_ptr = np.zeros(cvs.shape[0] + 1, dtype=np.int64)
        got_idx = np.zeros(max(nnz, 1), dtype=np.int64)
        rc = shim._conv_room_csr(handle, got_ptr, got_idx[:max(nnz, 1)])
        same = (
            rc == 0
            and nnz == idx.size
            and np.array_equal(got_ptr, ptr)
            and np.array_equal(got_idx[:nnz], idx)
        )
        if not same:
            print(f"  FAIL csr {kind}: rc={rc} nnz={nnz} py_nnz={idx.size}")
            ok = False
        probe = rooms[kind][len(rooms[kind]) // 2]
        if sim.candidates_for(kind, 2, probe) != [
            rooms[kind][j] for j in idx[ptr[rooms[kind].index(probe)]:
                                        ptr[rooms[kind].index(probe) + 1]]
        ]:
            print(f"  FAIL candidates_for {kind}")
            ok = False
    print("  convection: candidate CSR identical for 4 room shapes")

    for kind in ("rect", "single", "lshape", "line"):
        rooms = {kind: coords(kind)}
        for p in (0.1, 0.5, 1.0):
            sim._p = p
            for trial in range(3):
                base = (
                    np.arange(60 * 60, dtype=np.float64).reshape(60, 60) * 1e-3
                    + 290.0
                )
                # shadow mode: the patched method runs the numba kernel on a
                # copy and the C kernel on `temp`, then compares.
                conv._seed_kernel_rng(5)
                shim.GATE.reset()
                old = sim.__dict__.pop("_cenv_conv", None)
                if old is not None:
                    shim._conv_dispose(old)
                temp = base.copy()
                sim.apply_convection(rooms, temp)
                report = shim.GATE.report()
                st = report.get("convection")
                trials += 1
                if not st or not st["passed"]:
                    print(f"  FAIL convection {kind} p={p} trial={trial}: {st}")
                    ok = False
                    break
    print(f"  convection: {trials} room/p/trial cases, numba kernel vs C")
    return ok


def _unit_occupancy() -> bool:
    import numpy as np
    import pandas as pd

    from cenv import stochastic as shim
    from smart_buildings.smart_control.simulator import (
        randomized_arrival_departure_occupancy as occ,
    )

    ok = True
    for seed in (17321, 12345):
        model = occ.RandomizedArrivalDepartureOccupancy(
            zone_assignment=2,
            earliest_expected_arrival_hour=6,
            latest_expected_arrival_hour=12,
            earliest_expected_departure_hour=13,
            latest_expected_departure_hour=19,
            time_step_sec=300,
            seed=seed,
            time_zone="US/Pacific",
        )
        zones = [f"room_{i + 1}" for i in range(30)]
        now = pd.Timestamp("2023-10-22 06:00")
        step = pd.Timedelta(300, unit="s")
        shim.GATE.reset()
        values = 0
        for _ in range(STEPS_EPISODE):
            # the simulator's own call pattern: the forward interval (internal
            # gains, then the reward), then the trailing-window observation.
            for zone in zones:
                model.average_zone_occupancy(zone, now, now + step)
            for zone in zones:
                model.average_zone_occupancy(zone, now, now + step)
            after = now + step
            for zone in zones:
                model.average_zone_occupancy(zone, after - step, after)
            now = after
            values += 1
        report = shim.GATE.report().get("occupancy")
        # the C generator must also be at the same word as the Python one
        st = model._cenv_occ
        words, index = shim._mt_export(st.rng)
        py_words, py_index = shim._numpy_mt_state(model._random_state)
        stream_same = index == py_index and np.array_equal(words, py_words)
        if not report or not report["passed"] or not stream_same:
            print(
                f"  FAIL occupancy seed={seed}: {report} "
                f"stream_same={stream_same} pos c={index} py={py_index}"
            )
            ok = False
        else:
            print(
                f"  occupancy: seed {seed}, {values} steps x 30 zones "
                f"({report['calls']} draws), generator state identical"
            )
    return ok


def run_unit() -> bool:
    simload = _bootstrap()
    import smart_buildings  # noqa: F401  (the tree install() checks against)
    from cenv import stochastic as shim

    shim.install(simload.sim_root(SIM), shadow="convection,occupancy,weather",
                 verbose=True)
    print("unit: streams")
    ok = _unit_streams()
    print("unit: weather")
    ok = _unit_weather() and ok
    print("unit: convection")
    ok = _unit_convection() and ok
    print("unit: occupancy")
    ok = _unit_occupancy() and ok
    return ok


# --------------------------------------------------------------------------- #
# episodes
# --------------------------------------------------------------------------- #
def _evalrun(
    *,
    tag: str,
    steps: int,
    scenario: str,
    occupancy_seed: int | None,
    cenv: bool,
    shadow: str,
    workers: int = 1,
) -> dict:
    env = dict(os.environ)
    env["CENV_SHADOW"] = shadow
    cmd = [
        PY, "-m", "sbx.evalrun",
        "--policy", POLICY,
        "--scenario", scenario,
        "--steps", str(steps),
        "--sim", SIM,
        "--workers", str(workers),
        "--components", "parity",
        "--tag", tag,
        "--out", str(RESULTS / tag),
    ]
    if occupancy_seed is not None:
        cmd += ["--occupancy-seed", str(occupancy_seed)]
    if cenv:
        cmd += ["--cenv", "stochastic"]
    proc = subprocess.run(cmd, cwd=WORKSPACE, env=env, capture_output=True, text=True)
    out = {"cmd": " ".join(cmd[1:]), "rc": proc.returncode}
    record = RESULTS / tag / (scenario.replace("/", "__") + ".json")
    if proc.returncode != 0 or not record.exists():
        out["error"] = (proc.stderr or proc.stdout)[-2000:]
        return out
    rec = json.loads(record.read_text())
    out.update(
        {
            "total_reward": rec["total_reward"],
            "step_cpu_seconds": rec["step_cpu_seconds"],
            "cpu_seconds": rec["cpu_seconds"],
            "steps": rec["steps"],
        }
    )
    gate = None
    for line in proc.stderr.splitlines():
        if line.startswith("cenv-gate: "):
            payload = json.loads(line[len("cenv-gate: "):])
            gate = payload.get("components", payload)
            out["occupancy_zones"] = payload.get("occupancy_zones")
    if gate is not None:
        out["gate"] = gate
    return out


def run_episode() -> bool:
    scenario = "building_train_8/fairbanks_8"
    ok = True
    evidence = {"scenario": scenario, "steps": STEPS_EPISODE, "sim": SIM,
                "policy": POLICY, "seeds": {}}
    for seed in (17321, 12345):
        records = {}
        base = f"gate_{'shadow' if seed == 17321 else 'shadow'}_s{seed}"
        records["ref"] = _evalrun(
            tag=f"stoch_ref_s{seed}", steps=STEPS_EPISODE, scenario=scenario,
            occupancy_seed=seed, cenv=False, shadow="",
        )
        records["shadow"] = _evalrun(
            tag=base, steps=STEPS_EPISODE, scenario=scenario,
            occupancy_seed=seed, cenv=True, shadow="convection,occupancy,weather",
        )
        records["on"] = _evalrun(
            tag=f"stoch_on_s{seed}", steps=STEPS_EPISODE, scenario=scenario,
            occupancy_seed=seed, cenv=True, shadow="",
        )
        ref = records["ref"].get("total_reward")
        shadows = records["shadow"].get("total_reward")
        on = records["on"].get("total_reward")
        gate = records["shadow"].get("gate") or {}
        passed = (
            ref is not None
            and shadows == ref
            and on == ref
            and gate
            and all(v["passed"] for v in gate.values())
        )
        print(f"\nseed {seed} on {scenario} ({STEPS_EPISODE} steps)")
        for name in ("ref", "shadow", "on"):
            r = records[name]
            if "error" in r:
                print(f"  {name:<6} FAILED: {r['error'].splitlines()[-1]}")
            else:
                print(
                    f"  {name:<6} total={r['total_reward']:+.10f} "
                    f"step_cpu={r['step_cpu_seconds']:.2f}s"
                )
        zones = records["shadow"].get("occupancy_zones")
        print(f"  occupancy zone ids used: {zones}")
        for name, st in sorted(gate.items()):
            print(
                f"  gate {name:<28} calls={st['calls']:>7} "
                f"mismatches={st['mismatches']} max|d|={st['max_abs_diff']:.3e} "
                f"{'PASS' if st['passed'] else 'FAIL ' + str(st['first'])}"
            )
        same_total = ref is not None and shadows == ref and on == ref
        print(
            f"  episode totals identical: {same_total} "
            f"(ref={ref!r} shadow={shadows!r} on={on!r})"
        )
        evidence["seeds"][str(seed)] = {
            "total_reward": {"ref": ref, "shadow": shadows, "on": on},
            "identical": bool(same_total),
            "gate": gate,
            "occupancy_zones": zones,
            "step_cpu_seconds": {
                name: records[name].get("step_cpu_seconds")
                for name in ("ref", "shadow", "on")
            },
            "passed": bool(passed),
        }
        ok = ok and bool(passed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "episode_gate.json").write_text(
        json.dumps(evidence, indent=1, sort_keys=True)
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #
def run_bench() -> bool:
    rows = []
    scenarios = ("building_train_8/fairbanks_8", "building_train_2/fairbanks_2")
    for scenario in scenarios:
        for cenv in (False, True):
            name = scenario.replace("/", "__")
            rec = _evalrun(
                tag=f"bench_{'on' if cenv else 'off'}_{name}",
                steps=STEPS_BENCH,
                scenario=scenario,
                occupancy_seed=None,
                cenv=cenv,
                shadow="",
            )
            rows.append((scenario, cenv, rec))
    print(f"\n{POLICY} on --sim {SIM}, {STEPS_BENCH} steps")
    print(f"{'scenario':<34} {'cenv':<5} {'total_reward':>16} {'step s':>9} {'steps/s':>8}")
    ok = True
    for scenario, cenv, rec in rows:
        if "error" in rec:
            print(f"{scenario:<34} {'on' if cenv else 'off':<5} FAILED: "
                  f"{rec['error'].splitlines()[-1]}")
            ok = False
            continue
        print(
            f"{scenario:<34} {'on' if cenv else 'off':<5} "
            f"{rec['total_reward']:>16.6f} {rec['step_cpu_seconds']:>9.2f} "
            f"{rec['steps'] / rec['step_cpu_seconds']:>8.2f}"
        )
    for i in range(0, len(rows), 2):
        off, on = rows[i][2], rows[i + 1][2]
        if "error" in off or "error" in on:
            continue
        speedup = off["step_cpu_seconds"] / on["step_cpu_seconds"]
        same = off["total_reward"] == on["total_reward"]
        print(
            f"{rows[i][0]:<34} speedup {speedup:.3f}x "
            f"({off['step_cpu_seconds']:.2f}s -> {on['step_cpu_seconds']:.2f}s) "
            f"totals identical: {same}"
        )
        ok = ok and same
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("unit", "episode", "bench", "all"))
    args = parser.parse_args(argv)
    ok = True
    if args.mode in ("unit", "all"):
        ok = run_unit() and ok
    if args.mode in ("episode", "all"):
        ok = run_episode() and ok
    if args.mode in ("bench", "all"):
        ok = run_bench() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
