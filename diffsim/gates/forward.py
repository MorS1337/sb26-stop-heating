"""Forward-exactness gate: the twin's step against the shipped kernel step.

Runs one closed-loop episode (a policy, both simulator trees) and after every
step compares the twin, fed the *same* command block, against the fused kernel:
solved temperature field, interior mass, tank temperature, return temperature
and the scalar reward.

    starter-kit/.venv/bin/python -m diffsim.gates.forward --sim corrected \
        --scenario building_train_1/fairbanks_1 --steps 600

Exit status 0 and ``GATE PASS`` when every relative deviation is below
``--rtol`` (default 1e-12) and the episode total agrees to ``--ttol`` (1e-9).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def split_sync(sync, nv, na):
    b = 2 * nv
    act = np.empty(2 * nv + 3 * na + 2)
    act[:nv] = sync[:nv]
    act[nv:2 * nv] = sync[nv:2 * nv]
    for a in range(na):
        act[2 * nv + a] = sync[b + 4 * a + 1]
        act[2 * nv + na + a] = sync[b + 4 * a + 2]
        act[2 * nv + 2 * na + a] = sync[b + 4 * a + 3]
    act[2 * nv + 3 * na] = sync[b + 4 * na + 1]
    act[2 * nv + 3 * na + 1] = sync[b + 4 * na + 2]
    runs = np.empty(na + 1)
    for a in range(na):
        runs[a] = sync[b + 4 * a]
    runs[na] = sync[b + 4 * na + 3]
    return act, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="corrected")
    ap.add_argument("--scenario", default="building_train_1/fairbanks_1")
    ap.add_argument("--policy", default="policies.ctrl_w5:CtrlW5")
    ap.add_argument("--params", default='{"standby_minimum": 262.0}')
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--ttol", type=float, default=1e-9)
    ap.add_argument("--reward-rtol", type=float, default=1e-6,
                    help="the reward is built by the simulator from float32 "
                         "RewardInfo fields, so its per-step relative error "
                         "cannot be 1e-12; the episode total is still gated")
    ap.add_argument("--offset-days", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from diffsim.plan import extract
    from diffsim.twin import Twin
    from sbx import simload, scenarios
    from sbx.evalrun import _load_environment, _resolve_policy

    simload.use_sim(a.sim)
    from smart_buildings import evaluate
    from cenv import install as cenv_install
    from cenv.fused import FusedState, _state_of

    cenv_install(["fused"], sim_root=simload.sim_root(a.sim), verbose=False)
    cfg = scenarios.config_path(a.sim, a.scenario)
    if a.offset_days:
        from pathlib import Path
        from sbx.evalrun import _shifted_config
        cfg = _shifted_config(cfg, a.offset_days, Path(ROOT, "scratch/ExactOpt/_gin"))
    env = _load_environment(evaluate, cfg, {})
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = evaluate.create_building_meta_data(env, cfg.parent)
    spec = env.action_spec()
    cls, label = _resolve_policy(a.policy, evaluate)
    params = json.loads(a.params) if a.params else {}
    if params:
        cls.PARAMS = {**(getattr(cls, "PARAMS", None) or {}), **params}
    policy = evaluate.initialize_policy(cls, meta, spec)

    captured = []
    orig = FusedState._sync_step

    def patched(self):
        orig(self)
        captured.append(self.sync.copy())

    FusedState._sync_step = patched
    obs, _ = env.reset()
    st = _state_of(env)
    plan = extract(env, a.steps + 1, a.sim, a.scenario)
    twin = Twin(plan, quantize=True)
    state = twin.initial_state()
    sim_obj = env.building.simulator
    solved = plan["solved"]

    worst = {"temp": 0.0, "mass": 0.0, "tank": 0.0, "ret": 0.0, "reward": 0.0}
    total_env = 0.0
    total_twin = 0.0
    t0 = time.process_time()
    n = 0
    for i in range(a.steps):
        act_in = evaluate.validate_action(evaluate.get_policy_action(policy, obs), spec)
        obs, rew, term, trunc, _ = env.step(act_in)
        sync = captured[-1]
        act, runs = split_sync(sync, plan["nv"], plan["na"])
        rew_t, pen_t, viol_t, tape, state = twin.forward(
            act[None, :], runs[None, :], state=state, i0=i)
        ref_T = np.asarray(sim_obj.building.temp, dtype=np.float64).reshape(-1)[solved]
        err_t = np.max(np.abs(state["T"] - ref_T)) / max(1e-12, np.max(np.abs(ref_T)))
        mass_ref = np.asarray(st.mass_array, dtype=np.float64)
        err_m = np.max(np.abs(state["mass"] - mass_ref)) / max(1e-12, np.max(np.abs(mass_ref)))
        err_tank = abs(float(state["tank"][0]) - float(st.o_boiler_temp[0])) / max(
            1e-12, abs(float(st.o_boiler_temp[0])))
        err_ret = abs(float(state["ret"][0]) - float(st.o_hws_return[0])) / max(
            1e-12, abs(float(st.o_hws_return[0])))
        err_r = abs(float(rew_t[0]) - float(rew)) / max(1e-12, abs(float(rew)))
        worst["temp"] = max(worst["temp"], float(err_t))
        worst["mass"] = max(worst["mass"], float(err_m))
        worst["tank"] = max(worst["tank"], float(err_tank))
        worst["ret"] = max(worst["ret"], float(err_ret))
        worst["reward"] = max(worst["reward"], float(err_r))
        total_env += float(rew)
        total_twin += float(rew_t[0])
        n += 1
        if term or trunc:
            break
    cpu = time.process_time() - t0
    tot_err = abs(total_twin - total_env) / max(1e-12, abs(total_env))
    print(f"sim={a.sim} scenario={a.scenario} steps={n} policy={label}")
    print("worst relative deviations:", json.dumps({k: f"{v:.3e}" for k, v in worst.items()}))
    print(f"episode total: kernel {total_env:.12f} twin {total_twin:.12f} "
          f"relative {tot_err:.3e}")
    print(f"cpu {cpu:.1f}s ({cpu / max(n, 1) * 1e3:.2f} ms/step)")
    state_ok = all(worst[k] <= a.rtol for k in ("temp", "mass", "tank", "ret"))
    ok = (state_ok and worst["reward"] <= a.reward_rtol and tot_err <= a.ttol)
    print(f"state rtol {a.rtol:g}: {'ok' if state_ok else 'FAIL'}; "
          f"reward per-step <= {a.reward_rtol:g}: "
          f"{'ok' if worst['reward'] <= a.reward_rtol else 'FAIL'}; "
          f"episode total <= {a.ttol:g}: {'ok' if tot_err <= a.ttol else 'FAIL'}")
    print("GATE PASS" if ok else "GATE FAIL")
    if a.out:
        out_path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dict(sim=a.sim, scenario=a.scenario, steps=n,
                           worst=worst, total_env=total_env, total_twin=total_twin,
                           total_rel=tot_err, cpu=cpu, pass_=ok), f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    import os as _os
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # the C kernel's teardown corrupts the glibc heap on exit (a segfault
    # after the results are written, WSL2/glibc only); skip it
    _os._exit(int(_rc))
