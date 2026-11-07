"""Build a self-contained plan for the differentiable twin of one scenario.

Everything here is extraction: the simulator environment is built exactly as
``sbx.evalrun`` builds it, the fused C kernel (``cenv.fused.FusedState``) has
already validated every array this code reads, and the plan keeps only what the
twin needs:

* geometry (zone cells, VAV -> zone relation, per-VAV and per-AHU parameters);
* the thermal system (``A`` and its factorisation, the air-side ``P``/``Q``,
  ``R = A^-1 P`` and ``S = Q^T A^-1 P``), all taken from the simulator's own
  solver, so the twin's solve is the same linear system;
* the precomputed per-step tables (ambient, comfort flag, occupancy, energy
  prices and carbon rates);
* the initial state (temperature field, applied power, interior mass);
* the convection permutation of every step, which is action-independent: it is
  drawn once from a copy of the kernel's own MT19937 state, by running
  ``cenv_conv_apply`` on the *index* field.  This reuses the exact generator
  instead of re-implementing it.

The plan is JSON-invisible (numpy only) and picklable; :func:`load_or_build`
caches it under ``scratch/ExactOpt/cache``.
"""
from __future__ import annotations

import ctypes
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "scratch", "ExactOpt", "cache")


def _bind_mt():
    from cenv.bindings import lib
    from cenv.stochastic import _mt_create, _mt_set_state, _conv_apply
    return lib, _mt_create, _mt_set_state, _conv_apply


def _build_env(sim: str, scenario: str, offset_days: int = 0):
    from sbx import simload, scenarios
    from sbx.evalrun import _load_environment, _shifted_config
    from pathlib import Path

    simload.use_sim(sim)
    from smart_buildings import evaluate  # noqa: PLC0415
    from cenv import install as cenv_install  # noqa: PLC0415

    cenv_install(["fused"], sim_root=simload.sim_root(sim), verbose=False)
    cfg = scenarios.config_path(sim, scenario)
    if offset_days:
        cfg = _shifted_config(cfg, offset_days, Path(CACHE, "_gin"))
    env = _load_environment(evaluate, cfg, {})
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    return env, evaluate, scenarios


def build(sim: str, scenario: str, steps: int, offset_days: int = 0,
          cache: bool = True) -> dict:
    """Builds the environment and extracts the plan (cached on disk)."""
    key = f"{sim}__{scenario.replace('/', '__')}__{steps}__off{offset_days}.pkl"
    path = os.path.join(CACHE, key)
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            plan = pickle.load(f)
        plan["factor"] = _refactor(plan)
        return plan
    os.makedirs(CACHE, exist_ok=True)
    env, evaluate, scenarios = _build_env(sim, scenario, offset_days)
    # the kernel and the reward path both need a *reset* environment (the
    # episode clock, the occupancy grid and the building's initial state); the
    # gate scripts get this from env.reset() before extract()
    env.reset()
    plan = extract(env, steps, sim, scenario, offset_days)
    if cache:
        stored = dict(plan)
        stored["factor"] = None
        with open(path, "wb") as f:
            pickle.dump(stored, f, protocol=4)
    return plan


def _refactor(plan):
    import scipy.sparse as sp  # noqa: PLC0415
    from cenv.solver import Factor  # noqa: PLC0415

    A = sp.csc_matrix(plan["_A"])
    return Factor(A)


def extract(env, steps: int, sim: str = "", scenario: str = "",
            offset_days: int = 0) -> dict:
    """Extracts the plan from an already-built environment."""
    from cenv.fused import FusedState  # noqa: PLC0415
    from cenv.bindings import lib  # noqa: PLC0415
    from cenv.solver import Factor  # noqa: PLC0415
    import scipy.sparse as sp  # noqa: PLC0415

    sim_obj = env.building.simulator
    st = FusedState(env)
    solver = sim_obj._solver
    bld = sim_obj.building
    n_full = int(bld.temp.size)
    nrows, ncols = (int(x) for x in bld.temp.shape)
    solved = np.ascontiguousarray(np.asarray(solver._solved), dtype=np.int32)
    n = int(solved.size)
    solved_pos = np.full(n_full, -1, dtype=np.int32)
    solved_pos[solved] = np.arange(n, dtype=np.int32)
    pinned = solved_pos < 0

    nz = st.n_zones
    nv = st.n_vav
    na = st.n_ahu
    k = st.thermal.k
    dt = float(st.dt)
    step_ns = int(st.step_ns)
    origin_ns = int(st.origin_ns)

    # ---- geometry ------------------------------------------------------
    zone_off = np.asarray(st.zone_off, dtype=np.int32)
    zone_cells = np.asarray(st.zone_cells, dtype=np.int32)
    zone_area = np.asarray(st.zone_area, dtype=np.int32)
    if np.any(pinned[zone_cells]):
        raise NotImplementedError("a zone control volume is pinned to ambient")

    vav = st.vav_cfg
    grab = lambda f, dtp=np.float64: np.array([getattr(vav[i], f) for i in range(nv)], dtype=dtp)
    vav_max_flow = grab("max_air_flow_rate")
    vav_min_damper = grab("min_damper")
    vav_max_discharge = grab("max_discharge_temp")
    vav_design_static = grab("max_flow_static_pressure")
    vav_reheat_max = grab("reheat_max_flow_factor")
    vav_ahu = grab("ahu", np.int32)
    vav_zoff = grab("zone_off", np.int32)
    vav_zn = grab("zone_n", np.int32)
    vav_area = grab("area", np.int32)
    vav_area_all = grab("area_all", np.int32)
    vav_zone = np.asarray(st.vav_zone, dtype=np.int32)
    coupled_vav = np.asarray(st.coupled_vav, dtype=np.int32)
    zone_vav_off = np.asarray(st.zone_vav_off, dtype=np.int32)
    zone_vav = np.asarray(st.zone_vav, dtype=np.int32)

    ahu = st.ahu_cfg
    agrab = lambda f: np.array([getattr(ahu[i], f) for i in range(na)], dtype=np.float64)
    ahu_max_flow = agrab("max_air_flow_rate")
    ahu_min_sp = agrab("min_static_pressure")
    ahu_max_recirc = agrab("max_recirculation")
    ahu_fan_eff = agrab("fan_efficiency")
    ahu_cop = agrab("cooling_cop")

    hws = st.hws
    boiler = st.boiler
    pump = st.pump
    fixed = dict(
        energy_balance=bool(st.hws_energy_balance),
        has_tank=bool(st.hws_has_tank),
        header_resistance=float(hws._header_resistance),
        heating_rate=float(getattr(boiler, "_heating_rate", 0.0)),
        cooling_rate=float(getattr(boiler, "_cooling_rate", 0.0)),
        conv_coef=float(getattr(boiler, "_convection_coefficient", 0.0)),
        tank_len=float(getattr(boiler, "_tank_length", 0.0)),
        tank_radius=float(getattr(boiler, "_tank_radius", 0.0)),
        water_capacity=float(getattr(boiler, "_water_capacity", 0.0)),
        insul_cond=float(getattr(boiler, "_insulation_conductivity", 0.0)),
        insul_thick=float(getattr(boiler, "_insulation_thickness", 0.0)),
        pump_eff=float(pump._water_pump_efficiency),
    )
    internal_gain = float(getattr(sim_obj, "_internal_gain_w_per_m2", 0.0))
    if internal_gain <= 0.0:
        raise NotImplementedError(
            "the twin relies on every zone being written every step by the "
            "internal gains (building.apply_thermal_power_zone); this "
            "scenario has internal_gain_w_per_m2 == 0")
    occupant_heat = float(getattr(sim_obj, "_occupant_sensible_heat_w", 0.0))
    cv_area = (float(bld.cv_size_cm) / 100.0) ** 2
    schedule = st.hvac.schedule

    # ---- thermal -------------------------------------------------------
    h = float(st.thermal_h)
    A = sp.csc_matrix(solver.matrix(h))
    factor = Factor(A)
    cap = np.ascontiguousarray(solver._capacitance_rate, dtype=np.float64)
    dirichlet = np.ascontiguousarray(solver._dirichlet_conductance, dtype=np.float64)
    exposure = np.ascontiguousarray(solver.ambient_conductance(h), dtype=np.float64)
    mc = np.ascontiguousarray(solver._mass_coupling, dtype=np.float64)
    mr = np.ascontiguousarray(solver._mass_retained, dtype=np.float64)
    mg = np.ascontiguousarray(solver._mass_gained, dtype=np.float64)
    P = sp.csc_matrix(solver._applied) if k else sp.csc_matrix((n, 0))
    Q = sp.csr_matrix(solver._sensed) if k else sp.csr_matrix((0, n))
    if k:
        R = np.empty((n, k), dtype=np.float64)
        for j in range(k):
            b = np.zeros(n, dtype=np.float64)
            col = P.getcol(j).tocoo()
            b[col.row] = col.data
            R[:, j] = factor.solve(b)
        S = np.ascontiguousarray(Q @ R, dtype=np.float64)
        # cross-check against the kernel's own S
        sresp = np.asarray(st.thermal.sresp, dtype=np.float64).reshape(k, k)
        sdiff = float(np.max(np.abs(S - sresp)) / max(1e-30, np.max(np.abs(sresp))))
        if sdiff > 1e-12:
            raise RuntimeError(f"twin S disagrees with the kernel: {sdiff}")
    else:
        R = np.zeros((n, 0))
        S = np.zeros((0, 0))

    # ---- per-step tables ----------------------------------------------
    n_h = int(st.n_horizon)
    if steps + 2 > n_h:
        raise ValueError("plan: steps+2 exceeds the kernel's horizon table")
    ambient = np.ascontiguousarray(st.ambient[: steps + 2], dtype=np.float64)
    comfort = np.asarray(st.comfort[: steps + 2], dtype=bool)

    # occupancy over [k, k+1) for k = 0 .. steps+1, per zone.
    # A *separate* C handle: the occupancy chain is stateful and advances as it
    # is queried, so the episode's own handle must not be consumed here.  The
    # fresh handle is seeded from the model's RandomState (the same state the
    # kernel copied) and queried in the kernel's own order -- per window,
    # zones in occ_id order, ticks ascending -- so the values agree.
    import cenv.stochastic as _stoch  # noqa: PLC0415

    occ = np.zeros((steps + 2, nz), dtype=np.float64)
    occ_ids = np.asarray(st.occ_id, dtype=np.int32)
    occ_state = _stoch._OccState(sim_obj._occupancy)
    occ_state.ensure(origin_ns)
    occ_state.ensure(origin_ns + (steps + 2) * step_ns)
    occ_handle = ctypes.c_int64(occ_state.handle)
    val = ctypes.c_double()
    lib.cenv_occ_average.restype = ctypes.c_int32
    lib.cenv_occ_average.argtypes = [ctypes.c_int64, ctypes.c_int32,
                                     ctypes.c_int64, ctypes.c_int64,
                                     ctypes.POINTER(ctypes.c_double)]
    for kk in range(steps + 2):
        s0 = origin_ns + kk * step_ns
        s1 = s0 + step_ns
        for z in range(nz):
            if lib.cenv_occ_average(occ_handle, int(occ_ids[z]), s0, s1,
                                    ctypes.byref(val)) != 0:
                raise RuntimeError("cenv_occ_average failed")
            occ[kk, z] = val.value

    # Occupancy as the *reward* sees it.  ``--cenv fused`` does not install the
    # stochastic component, so ``average_zone_occupancy`` keeps running the
    # model's own Python chain (its own MT19937 stream, advanced once per step
    # per zone in room_dict order), while the kernel's occupant gains use a
    # copy of that stream inside C.  The reward is built by the simulator's
    # Python ``reward_info`` (the flexible-geometry class overrides the method
    # cenv patches), so its occupancy is the Python one: reproduce it on a deep
    # copy, queried in the same order and windows.
    occ_reward = _python_occupancy(sim_obj, st.zone_names, origin_ns, step_ns,
                                   steps + 2)

    # energy price / carbon unit coefficients per step
    import pandas as pd  # noqa: PLC0415

    rf = env.reward_function
    e_cost = rf._electricity_energy_cost
    g_cost = rf._natural_gas_energy_cost
    t0 = pd.Timestamp(sim_obj.current_timestamp)
    e_unit = np.zeros(steps)
    e_carb = np.zeros(steps)
    g_unit = np.zeros(steps)
    g_carb = np.zeros(steps)
    max_e_cost = np.zeros(steps)
    max_e_carb = np.zeros(steps)
    max_g_cost = np.zeros(steps)
    max_g_carb = np.zeros(steps)
    for kk in range(steps):
        a = t0 + pd.Timedelta(seconds=dt * kk)
        b = a + pd.Timedelta(seconds=dt)
        e_unit[kk] = e_cost.cost(a, b, 1.0)
        e_carb[kk] = e_cost.carbon(a, b, 1.0)
        g_unit[kk] = g_cost.cost(a, b, 1.0)
        g_carb[kk] = g_cost.carbon(a, b, 1.0)
        max_e_cost[kk] = e_cost.cost(a, b, rf._max_electricity_rate)
        max_e_carb[kk] = e_cost.carbon(a, b, rf._max_electricity_rate)
        max_g_cost[kk] = g_cost.cost(a, b, rf._max_natural_gas_rate)
        max_g_carb[kk] = g_cost.carbon(a, b, rf._max_natural_gas_rate)

    reward = dict(
        max_productivity=float(rf._max_productivity_personhour_usd),
        min_productivity=float(rf._min_productivity_personhour_usd),
        deviation_ref=float(rf._productivity_deviation_ref),
        w_p=float(rf._productivity_weight),
        w_e=float(rf._energy_cost_weight),
        w_c=float(rf._carbon_emission_weight),
        max_electricity_rate=float(rf._max_electricity_rate),
        max_natural_gas_rate=float(rf._max_natural_gas_rate),
        dt=dt,
    )

    # ---- initial state -------------------------------------------------
    temp_full = np.ascontiguousarray(bld.temp, dtype=np.float64).reshape(-1)
    T0 = np.ascontiguousarray(temp_full[solved], dtype=np.float64)
    q_full = np.ascontiguousarray(bld.input_q, dtype=np.float64).reshape(-1)
    q0 = np.ascontiguousarray(q_full[solved], dtype=np.float64)
    mass0 = np.ascontiguousarray(solver._mass_temp, dtype=np.float64).copy()
    tank0 = float(boiler._current_temperature)
    ret0 = float(boiler.return_water_temperature_sensor)

    # ---- convection permutations --------------------------------------
    conv = _conv_perms(st, solved, solved_pos, steps)

    plan = dict(
        sim=sim, scenario=scenario, steps=int(steps), offset_days=int(offset_days),
        n_full=n_full, nrows=nrows, ncols=ncols, n=n, nz=nz, nv=nv, na=na, k=k,
        dt=dt, step_ns=step_ns, origin_ns=origin_ns,
        solved=solved, solved_pos=solved_pos, pinned_count=int(pinned.sum()),
        zone_off=zone_off, zone_cells=zone_cells,
        zone_cells_solved=np.ascontiguousarray(solved_pos[zone_cells]),
        zone_area=zone_area,
        vav_max_flow=vav_max_flow, vav_min_damper=vav_min_damper,
        vav_max_discharge=vav_max_discharge,
        vav_design_static=vav_design_static, vav_reheat_max=vav_reheat_max,
        vav_ahu=vav_ahu, vav_zoff=vav_zoff, vav_zn=vav_zn,
        vav_area=vav_area, vav_area_all=vav_area_all, vav_zone=vav_zone,
        coupled_vav=coupled_vav, zone_vav_off=zone_vav_off, zone_vav=zone_vav,
        ahu_max_flow=ahu_max_flow, ahu_min_sp=ahu_min_sp,
        ahu_max_recirc=ahu_max_recirc, ahu_fan_eff=ahu_fan_eff,
        ahu_cop=ahu_cop, hws=fixed,
        internal_gain=internal_gain, occupant_heat=occupant_heat,
        cv_area=cv_area,
        comfort_heat=float(schedule.comfort_temp_window[0]),
        comfort_cool=float(schedule.comfort_temp_window[1]),
        eco_heat=float(schedule.eco_temp_window[0]),
        eco_cool=float(schedule.eco_temp_window[1]),
        cap=cap, dirichlet=dirichlet, exposure=exposure, mc=mc, mr=mr, mg=mg,
        P=P, Q=Q, R=R, S=S, factor=factor,
        ambient=ambient, comfort=comfort, occ=occ, occ_reward=occ_reward,
        e_unit=e_unit, e_carb=e_carb, g_unit=g_unit, g_carb=g_carb,
        max_e_cost=max_e_cost, max_e_carb=max_e_carb,
        max_g_cost=max_g_cost, max_g_carb=max_g_carb,
        reward=reward,
        T0=T0, q0=q0, mass0=mass0, tank0=tank0, ret0=ret0,
        conv=conv,
        action_names=list(env.action_names),
        _A=A,
    )
    st.close()
    return plan


def _python_occupancy(sim_obj, zone_names, origin_ns, step_ns, n):
    """Reward occupancy over [k, k+1) for k < n, per zone, Python stream."""
    import copy  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    model = copy.deepcopy(sim_obj._occupancy)
    model.__dict__.pop("_cenv_occ", None)
    out = np.zeros((n, len(zone_names)), dtype=np.float64)
    # The simulator's first reward-info window is [t1, t2) -- the one after the
    # reset -- so the chain's first walk is over that window.  Starting the
    # table at k = 0 would consume draws the episode never consumes and shift
    # the whole stream.
    for k in range(1, n):
        a = pd.Timestamp(origin_ns + k * step_ns)
        b = pd.Timestamp(origin_ns + (k + 1) * step_ns)
        for z, name in enumerate(zone_names):
            out[k, z] = float(model.average_zone_occupancy(name, a, b))
    return out


def _conv_perms(st, solved, solved_pos, steps):
    """Per-step convection maps restricted to the solved destinations.

    The kernel applies, per room, one accepted double per CV, a partner draw and
    a Fisher-Yates shuffle, then sequential swaps.  That whole map is independent
    of the temperature field, so running ``cenv_conv_apply`` on the *index*
    field on a copy of the kernel's MT19937 state yields the permutation
    directly: ``out[i] = index_source[i]``.

    Returns (offsets, idx, src) with, for every step, the solved destinations
    whose value changes (``idx``) and their source solved position (``src``;
    ``-1`` means the source is pinned to ambient).
    """
    from cenv.bindings import lib
    from cenv.stochastic import _mt_create, _mt_set_state, _conv_apply

    state_words, state_index = export_mt_state(st)
    handle = int(_mt_create(0))
    _mt_set_state(handle, state_words.view(np.uint8), int(state_index))
    field = np.arange(st.n_full, dtype=np.float64)
    rooms = np.ascontiguousarray(st.conv_rooms, dtype=np.int64)
    offs = np.zeros(steps + 1, dtype=np.int64)
    idx_all = []
    src_all = []
    n_total = 0
    for s in range(steps):
        # a fresh index field every step: the swap sequence is a different
        # permutation each step, and composing them would be wrong
        field[:] = np.arange(st.n_full, dtype=np.float64)
        rc = _conv_apply(rooms, int(st.conv_rooms.size), int(handle), field,
                         int(st.nrows), int(st.ncols), float(st.conv_p))
        if rc != 0:
            raise RuntimeError(f"cenv_conv_apply failed ({rc})")
        src = field[solved].astype(np.int64)
        moved = src != solved
        ii = np.nonzero(moved)[0].astype(np.int32)
        ss = src[moved]
        ss_solved = solved_pos[ss].astype(np.int32)
        idx_all.append(ii)
        src_all.append(ss_solved)
        n_total += ii.size
        offs[s + 1] = n_total
    from cenv.stochastic import _mt_destroy
    _mt_destroy(handle)
    idx = np.concatenate(idx_all) if idx_all else np.zeros(0, np.int32)
    src = np.concatenate(src_all) if src_all else np.zeros(0, np.int32)
    return dict(offsets=offs, idx=np.ascontiguousarray(idx),
                src=np.ascontiguousarray(src), n_moved=int(n_total))


def export_mt_state(st):
    """Reads the convection generator's MT19937 state (624 words + index)."""
    from cenv.stochastic import _mt_export  # noqa: PLC0415
    return _mt_export(int(st.conv_rng))


def load_or_build(sim: str, scenario: str, steps: int, offset_days: int = 0) -> dict:
    return build(sim, scenario, steps, offset_days, cache=True)
