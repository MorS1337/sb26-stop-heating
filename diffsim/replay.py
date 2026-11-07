"""Drive the fused kernel and the simulator's own reward with arbitrary actions.

The twin is used to *find* a trajectory; the numbers reported for it must come
from the simulator itself.  This module replays a trajectory by calling the
fused kernel step directly (``cenv_fused_run``) and then the simulator's own
reward path, which is what ``sbx.evalrun`` records:

    response = building.request_action(action_request)   # sets the devices
    building.wait_time()                                 # one simulated step
    observation = env._get_observation()
    reward      = env._get_reward()

The only part replaced is the action request: the command block the kernel
consumes (damper/valve/AHU/HWS commands) is written straight into ``sync``
instead of being routed through the device setters, so a *native* command that
a setter would clamp (recirculation above the unit's maximum, for one) reaches
the kernel unclamped.  The optimiser's box bounds keep every command inside the
range the shipped setters accept, and :func:`verify_setters` checks that the
two paths agree on a warm-start trajectory.
"""
from __future__ import annotations

import ctypes
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAP = os.path.join(ROOT, "scratch", "ExactOpt", "captures")


def to_sync(act, run, nv, na):
    """The kernel's command block for one step (ts fields filled in by caller)."""
    sync = np.zeros(2 * nv + 4 * na + 6, dtype=np.float64)
    sync[:nv] = act[:nv]
    sync[nv:2 * nv] = act[nv:2 * nv]
    o = 2 * nv
    b = 2 * nv
    for a in range(na):
        sync[b + 4 * a] = run[a]
        sync[b + 4 * a + 1] = act[o + a]
        sync[b + 4 * a + 2] = act[o + na + a]
        sync[b + 4 * a + 3] = act[o + 2 * na + a]
    http = b + 4 * na
    sync[http] = run[na]
    sync[http + 1] = act[o + 3 * na]
    sync[http + 2] = act[o + 3 * na + 1]
    sync[http + 3] = run[na]
    return sync


def from_sync(sync, nv, na):
    act = np.empty(2 * nv + 3 * na + 2)
    act[:nv] = sync[:nv]
    act[nv:2 * nv] = sync[nv:2 * nv]
    b = 2 * nv
    o = 2 * nv
    for a in range(na):
        act[o + a] = sync[b + 4 * a + 1]
        act[o + na + a] = sync[b + 4 * a + 2]
        act[o + 2 * na + a] = sync[b + 4 * a + 3]
    act[o + 3 * na] = sync[b + 4 * na + 1]
    act[o + 3 * na + 1] = sync[b + 4 * na + 2]
    run = np.empty(na + 1)
    for a in range(na):
        run[a] = sync[b + 4 * a]
    run[na] = sync[b + 4 * na + 3]
    return act, run


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #
def capture(sim: str, scenario: str, policy: str, params: dict, steps: int,
            offset_days: int = 0, tag: str | None = None) -> dict:
    """Runs ``policy`` through the environment and records every command block."""
    from sbx import scenarios, simload
    from sbx.evalrun import (_load_environment, _resolve_policy,
                             _shifted_config)
    from pathlib import Path

    simload.use_sim(sim)
    from smart_buildings import evaluate  # noqa: PLC0415
    from cenv import install as cenv_install  # noqa: PLC0415
    from cenv.fused import FusedState, _state_of  # noqa: PLC0415

    cenv_install(["fused"], sim_root=simload.sim_root(sim), verbose=False)
    cfg = scenarios.config_path(sim, scenario)
    if offset_days:
        cfg = _shifted_config(cfg, offset_days,
                              Path(ROOT, "scratch/ExactOpt/_gin"))
    env = _load_environment(evaluate, cfg, {})
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = evaluate.create_building_meta_data(env, cfg.parent)
    spec = env.action_spec()
    cls, label = _resolve_policy(policy, evaluate)
    if params:
        cls.PARAMS = {**(getattr(cls, "PARAMS", None) or {}), **params}
    pol = evaluate.initialize_policy(cls, meta, spec)

    rec = []
    orig = FusedState._sync_step

    def patched(self):
        orig(self)
        rec.append(self.sync.copy())

    FusedState._sync_step = patched
    try:
        obs, _ = env.reset()
        st = _state_of(env)
        rewards = []
        for i in range(steps):
            act_in = evaluate.validate_action(
                evaluate.get_policy_action(pol, obs), spec)
            obs, rew, term, trunc, _ = env.step(act_in)
            rewards.append(float(rew))
            if term or trunc:
                break
    finally:
        FusedState._sync_step = orig
    sync = np.asarray(rec, dtype=np.float64)
    nv, na = st.n_vav, st.n_ahu
    acts = np.empty((len(sync), 2 * nv + 3 * na + 2))
    runs = np.empty((len(sync), na + 1))
    for i, s in enumerate(sync):
        acts[i], runs[i] = from_sync(s, nv, na)
    rewards = np.asarray(rewards, dtype=np.float64)
    out = dict(acts=acts, runs=runs, rewards=rewards, sync=sync,
               policy=label, params=params, sim=sim, scenario=scenario,
               offset_days=offset_days, nv=nv, na=na)
    os.makedirs(CAP, exist_ok=True)
    name = tag or ("%s__%s__off%d__%d" % (sim, scenario.replace("/", "__"),
                                          offset_days, len(sync)))
    np.savez_compressed(os.path.join(CAP, name + ".npz"),
                        acts=acts, runs=runs, rewards=rewards, sync=sync)
    with open(os.path.join(CAP, name + ".json"), "w") as f:
        json.dump(dict(policy=label, params=params, sim=sim,
                       scenario=scenario, offset_days=offset_days,
                       steps=len(sync), total=float(rewards.sum())), f, indent=2)
    return out


def load_capture(sim, scenario, steps, offset_days=0) -> dict:
    """The captured warm start, truncated/extended to ``steps``."""
    import glob

    base = os.path.join(CAP, "%s__%s" % (sim, scenario.replace("/", "__")))
    paths = sorted(glob.glob(base + "__off%d__*.npz" % offset_days))
    if not paths:
        paths = sorted(glob.glob(base + ".npz"))
    if not paths:
        raise FileNotFoundError(f"no capture for {sim} {scenario}: run "
                                f"diffsim.optim capture first")
    path = paths[-1]
    with np.load(path) as d:
        acts = d["acts"]
        runs = d["runs"]
        rewards = d["rewards"]
    if acts.shape[0] < steps:
        raise ValueError(f"capture has {acts.shape[0]} steps, need {steps}")
    return dict(acts=acts[:steps].copy(), runs=runs[:steps].copy(),
                rewards=rewards[:steps].copy(), path=path)


# --------------------------------------------------------------------------- #
# exact replay
# --------------------------------------------------------------------------- #
class ExactRunner:
    """Steps the fused kernel with an explicit command block."""

    def __init__(self, sim: str, scenario: str, offset_days: int = 0,
                 horizon: int | None = None):
        from sbx import scenarios, simload
        from sbx.evalrun import _load_environment, _shifted_config
        from pathlib import Path

        simload.use_sim(sim)
        from smart_buildings import evaluate  # noqa: PLC0415
        from cenv import install as cenv_install  # noqa: PLC0415
        from cenv.fused import FusedState, _state_of  # noqa: PLC0415
        from cenv.bindings import lib  # noqa: PLC0415

        cenv_install(["fused"], sim_root=simload.sim_root(sim), verbose=False)
        cfg = scenarios.config_path(sim, scenario)
        if offset_days:
            cfg = _shifted_config(cfg, offset_days,
                                  Path(ROOT, "scratch/ExactOpt/_gin"))
        self.env = _load_environment(evaluate, cfg, {})
        self.env.metrics_path = None
        self.env._occupancy_normalization_constant = 125.0
        self.sim = sim
        self.scenario = scenario
        self.obs, _ = self.env.reset()
        self.st = _state_of(self.env)
        self.lib = lib
        self.nv = self.st.n_vav
        self.na = self.st.n_ahu
        self.sim_obj = self.env.building.simulator
        self.temp = self.sim_obj.building.temp
        self.q = self.sim_obj.building.input_q
        self.i = 0

    def step(self, act, run):
        sync = to_sync(np.asarray(act, dtype=np.float64),
                       np.asarray(run, dtype=np.float64), self.nv, self.na)
        b = 2 * self.nv + 4 * self.na
        sync[b + 4] = self.i          # boiler observation timestamp index
        sync[b + 5] = self.i          # boiler action timestamp index
        sync = np.ascontiguousarray(sync, dtype=np.float64)
        rc = self.lib.cenv_fused_run(
            self.st.handle, self.i,
            self.temp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            sync.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if rc != 0:
            raise RuntimeError(f"cenv_fused_run failed ({rc}) at step {self.i}")
        self.st._post_step_bookkeeping(self.i)
        self.i += 1
        return float(self.env._get_reward())

    def run(self, acts, runs, steps=None):
        acts = np.asarray(acts)
        runs = np.asarray(runs)
        n = acts.shape[0] if steps is None else int(steps)
        out = np.empty(n)
        for k in range(n):
            out[k] = self.step(acts[k], runs[k])
        return out

    def close(self):
        self.st.close()


def replay(sim, scenario, acts, runs, offset_days=0, steps=None):
    runner = ExactRunner(sim, scenario, offset_days)
    try:
        rewards = runner.run(acts, runs, steps)
    finally:
        runner.close()
    return rewards
