"""``fused``: the whole per-step work of ``env.step`` as one C call.

What one step is, in order
--------------------------
This is the map the kernel reproduces (``cenv/src/step.c``; the Python it
transcribes is named in each block). ``timestep`` is 300 s on every shipped
scenario, ``t`` is the step's timestamp and ``T = t + timestep``.

``Environment.step(action)``::

    action   = _format_action(action, _action_names)          # no-op
    request  = _create_action_request(action)                 # protobuf, normaliser
    response = building.request_action(request)               # device.set_action
    building.wait_time()                                      # <- the C call
    observation = _get_observation()                          # cached from the call
    reward      = _get_reward()                               # cached reward info
    ... counters, episode-end flag, logging

``building.wait_time()`` is ``simulator.execute_step_sim``::

   1. avg_temps = building.get_zone_average_temps()      zone means, numpy pairwise
   2. ambient   = weather.get_current_temp(t)            precomputed table
   3. for each VAV: update_settings(area-weighted zone temp, t)
        schedule.get_temperature_window(t) -> (heating, cooling) setpoint window
        thermostat.update(zone_temp, t)  -> HEAT / COOL / OFF / PASSIVE_COOL
        damper, valve from the mode, then the agent's commands, then
        damper = max(damper, min_damper)                  # ventilation floor
      and (shipped tree) the area-weighted AHU temperature sums
   4. (corrected tree) _recirculation_temps: airflow-weighted per AHU
   5. supply_air_temp per AHU = supply setpoint if mixed air is warmer and the
      unit runs, else the mixed (recirculation, ambient) temperature
   6. convection coefficient: precomputed table (asserted constant)
   7. air_handler.reset_demand(), hot_water_system.reset_demand()
   8. hot_water_system.add_demand(vav.reheat_flow_factor) for every VAV
   9. unrestricted air demand per AHU -> vav.supply_air_flow_ratio, and
      vav.flow_rate_demand = unrestricted * ratio
  10. for each air-side column: reheat energy (reads the boiler's supply-water
      sensor), the air-side conductance, the reheat share per zone, and
      air_handler.add_demand(flow)
  11. internal gains (lighting/equipment, comfort-mode square wave)
  12. occupant gains: occupancy(zone, t, T) * occupant_sensible_heat_w
  13. building.apply_thermal_power_zone for every zone touched (ASSIGNS
      input_q, so untouched zones keep the previous step's value)
  14. finite_differences_timestep: the SuperLU + Woodbury solve
  15. air_side_q = g * (T_supply - sensed @ temp) folded back into input_q
  16. building.apply_convection() -- MT19937 draws and the swap kernel
  17. return water temperature from the reheat energy (reads the boiler's
      supply-water sensor again)
  18. (corrected tree) hot_water_system.apply_heat_draw: tank energy balance
  19. np.clip(temp, 150, 500) if the solve diverged
  20. current_timestamp += timestep

``_get_observation()`` at ``T``: every requested device measurement (in the
response order, which matters because reading the hot water system's
supply-water sensor advances the shipped boiler's tank), then 2 * num_hod and
2 * num_dow phase-shifted cos/sin features, then comfort_mode_now,
comfort_mode_soon and num_occupants (the occupancy sum over [T - 5 min, T],
which is the interval step 12 already queried, so it is a cache hit).

``_get_reward()`` at ``T``: ``Simulator.reward_info(occupancy)`` ->
per-zone setpoints, zone temperature, airflow setpoint/actual and occupancy
over [T, T + timestep]; per-AHU blower (intake + exhaust) and cooling
electrical energy rates; boiler thermal rate and pump power. Then
``reward_function.compute_reward`` (C, via the ``bookkeeping`` component).

Two facts drive the whole design:

* the shipped boiler's ``supply_water_temperature_sensor`` *mutates* the tank
  (``_set_current_temperature``) on every read, and it is read once per
  air-side column plus once by the observation and once more by the return
  water calculation. The kernel keeps the same count and order; the read is a
  function there, not a property;
* ``apply_thermal_power_zone`` assigns, so a zone the step does not touch keeps
  last step's ``input_q``. The kernel tracks which zones were touched.

What is in C and what stayed in Python
--------------------------------------
In C (one ``cenv_fused_run`` call per step): everything in items 1-20 above,
the observation vector, and the reward-info numbers. The already-gated pieces
are reused, not re-derived: ``thermal.c`` (whole solve, exported factors),
``hvac.c`` (the scalar device maths), ``convection.c``/``occupancy.c``
(the RNG streams, on the same handles the ``stochastic`` shims use),
``core.c`` (numpy's pairwise summation), ``observation``'s normalisation
constants and ``bookkeeping``'s reward arithmetic.

Stays in Python, with the reason:

* gin, construction, factorisation and export -- once per episode, and scipy's
  SuperLU is the reference the factors must come from;
* the action request protobuf, the normalisers and ``device.set_action``
  (~0.4 ms/step): it is the policy-facing half of the step, it keeps every
  Python device object's ``_action_timestamp`` exact, and ``env.step`` needs
  the ``ActionResponse`` for the acceptance flag and the rejection path;
* per-step timestamp arithmetic, the tz database, DST, holidays and the
  cos/sin time features: precomputed once per episode as a table (the kernel
  only indexes it), because pandas is the reference and a 1-ULP difference in
  a float64 feature is not worth re-deriving;
* the ``RewardInfo``/``RewardResponse`` protobufs, filled from the kernel's
  arrays (~0.2 ms/step): they are the environment's API and what
  ``evaluate.py``'s recorder reads;
* device write-back: after each step the kernel's device state is written back
  onto the Python VAVs, AHUs and boiler, so anything that reads them (the
  reward-info fallback, a debugger, another component) sees the same numbers.

Not supported (``install`` raises instead of silently doing something else):
heat-pump hot water systems (``ASHP``), occupancy models other than
``RandomizedArrivalDepartureOccupancy``, a weather controller whose convection
coefficient varies over the episode (the timestep matrix would have to be
refactorised), and non-``SimulatorFlexibleGeometries`` simulators.

Gate
----
There is no ``shadow_pairs``: a per-function shadow hook is what the other
components need, but this one *is* the step, so its gate is
``cenv/verify/fused_gate.py`` -- closed-loop episodes compared against the
unpatched environment step by step (observation, reward, reward components,
zone temperatures, device state), plus full ``sbx.evalrun`` episodes. Run it
with ``python -m cenv.verify.fused_gate gate``.

Usage
-----
    starter-kit/.venv/bin/python -m sbx.evalrun --cenv fused \
        --policy policies.ctrl_w:CtrlW --scenario building_train_8/fairbanks_8 \
        --steps 8641 --sim shipped --workers 1 --tag myrun

    python -m cenv.verify.fused_gate run     # the gate (see that module)

``install`` also installs ``bookkeeping`` (the reward arithmetic) unless it is
already live; nothing else is required. ``--cenv all`` includes ``fused`` once
its gate passes.

Coverage and limits
-------------------
* Single-AHU and ``AirHandlerSystem`` both work; every VAV may serve several
  zones and every zone several VAVs (the general zone<->VAV relation is
  precomputed, nothing assumes 1:1).
* The zone mean, the air-side coupling, the convection draws and the occupancy
  chains use the same handles as the standalone components, so ``--cenv fused``
  and ``--cenv all`` cannot double-apply anything: the fused step replaces
  ``execute_step_sim`` wholesale, so the Python bodies never run.
* The kernel needs the episode's step grid up front (``n_horizon`` =
  ``_num_timesteps_in_episode + 2``); running past it extends the tables in
  Python before the next call.
* One environment per process on this path: the convection generator continues
  numba's process-global ``np.random`` stream, so two environments built in one
  process would share that stream and neither would match a standalone run.
  ``sbx/evalrun.py`` runs one scenario per process (``--workers`` forks), which
  is the intended use. M environments of one scenario are ``cenv/batch.py``:
  it gives every environment its own occupancy model and convection handle and
  shares only the factorization, then steps all of them with one C call.
* ``GATE_PASSED`` is set once ``cenv/verify/fused_gate.py`` has passed in this
  checkout; naming ``fused`` explicitly opts in regardless.
* The batching this kernel was written for is ``cenv/batch.py``: ``step.c`` is
  split into the four slices a batch needs (prelude / thermal / foldback /
  epilogue, exported as ``cenv_fused_batch_*``), the batch drives them for M
  handles around one ``cenv_tbatch_step_many``, and ``FusedState(env,
  tbatch=...)`` is the batch's form of the state -- it keeps no factor of its
  own and refuses to be stepped by ``env.step``.

Gate (``python -m cenv.verify.fused_gate gate``, passed 2026-11-06)
--------------------------------------------------------------------
Closed-loop CtrlW, fused vs the unpatched simulator, both trees, occupancy
seeds 17321 and 12345, 2880 steps each: building_train_8/fairbanks_8,
building_train_2/fairbanks_2 (sparse thermal path), building_train_5/fairbanks_5
-- 12/12 cases with observations and rewards bit-identical (max |Δreward| 0.0),
zone temperatures <= 1.2e-15 relative. Full 8641-step traces on train_8, both
trees, and the runner-level pair (``sbx/evalrun.py --cenv fused`` vs plain):
total_reward -208.8193519315464 (shipped) and -201.76287243400293 (corrected),
identical to the last bit. Verdict file: ``scratch/CEnvFused2/fused_gate.json``.

Measured
--------
CPU per ``env.step`` on a quiet-ish M3, plain -> fused: train_8 20.5 -> 9.6 ms
(2.1x), train_5 49.8 -> 27.8 ms (1.8x), train_2 67.6 -> 42.0 ms (1.6x); the
thermal solve is 68-81 % of the kernel, convection 16-28 %. Timings on a loaded
M3 are unreliable (threads land on E-cores), compare A/B back-to-back only.
``cenv/verify/fused_gate.py tables`` prints the per-step breakdown against
plain Python and ``--cenv all`` on this machine.
"""

from __future__ import annotations

import ctypes
import time

import numpy as np

COMPONENT = "fused"
GATE_PASSED = True

_COMFORT_NOW = "comfort_mode_now"
_COMFORT_SOON = "comfort_mode_soon"
_NUM_OCCUPANTS = "num_occupants"

# Observation slot kinds, in the order of cenv/include/fused.h.
(_VAV_DAMPER, _VAV_VALVE, _VAV_MAX_FLOW, _VAV_SETPOINT_HEAT,
 _VAV_SETPOINT_COOL, _VAV_ZONE_TEMP, _AHU_COOLING_REQUESTS,
 _AHU_STATIC_SETPOINT, _AHU_FLOW, _AHU_SUPPLY_TEMP_SENSOR,
 _AHU_SUPPLY_TEMP_SETPOINT, _AHU_RUN_COMMAND, _AHU_FAN_SPEED, _AHU_OUTSIDE_FLOW,
 _AHU_RECIRC, _AHU_OUTSIDE_TEMP, _HWS_DIFFP_SENSOR, _HWS_DIFFP_SETPOINT,
 _HWS_HEATING_REQUESTS, _HWS_RUN_STATUS, _HWS_SUPPLY_WATER_TEMP,
 _HWS_SUPPLY_WATER_SETPOINT, _AUX_COMFORT_NOW, _AUX_COMFORT_SOON,
 _AUX_NUM_OCCUPANTS, _TIME) = range(26)

_VAV_KINDS = {
    "supply_air_damper_percentage_command": _VAV_DAMPER,
    "reheat_valve_percentage_command": _VAV_VALVE,
    "supply_air_flowrate_setpoint": _VAV_MAX_FLOW,
    "zone_air_heating_temperature_setpoint_sensor": _VAV_SETPOINT_HEAT,
    "zone_air_cooling_temperature_setpoint_sensor": _VAV_SETPOINT_COOL,
    "zone_air_temperature_sensor": _VAV_ZONE_TEMP,
}

# AirHandler / AirHandlerSystem observable attribute -> slot kind.
_AHU_KINDS = {
    "cooling_request_count": _AHU_COOLING_REQUESTS,
    "supply_air_static_pressure_setpoint": _AHU_STATIC_SETPOINT,
    "supply_air_static_pressure_sensor": _AHU_STATIC_SETPOINT,
    "supply_air_flowrate_setpoint": _AHU_FLOW,
    "supply_air_flowrate_sensor": _AHU_FLOW,
    "supply_air_temperature_sensor": _AHU_SUPPLY_TEMP_SENSOR,
    "supply_air_temperature_setpoint": _AHU_SUPPLY_TEMP_SETPOINT,
    "supply_fan_run_command": _AHU_RUN_COMMAND,
    "exhaust_fan_run_command": _AHU_RUN_COMMAND,
    "supervisor_run_command": _AHU_RUN_COMMAND,
    "supply_fan_speed_percentage_command": _AHU_FAN_SPEED,
    "discharge_fan_speed_percentage_command": _AHU_FAN_SPEED,
    "outside_air_flowrate_sensor": _AHU_OUTSIDE_FLOW,
    "recirculation_percentage_command": _AHU_RECIRC,
    "outside_air_temperature_sensor": _AHU_OUTSIDE_TEMP,
}

_HWS_KINDS = {
    "differential_pressure_sensor": _HWS_DIFFP_SENSOR,
    "differential_pressure_setpoint": _HWS_DIFFP_SETPOINT,
    "heating_request_count": _HWS_HEATING_REQUESTS,
    "run_status": _HWS_RUN_STATUS,
    "supervisor_run_command": _HWS_RUN_STATUS,
    "supply_water_temperature_sensor": _HWS_SUPPLY_WATER_TEMP,
    "supply_water_temperature_setpoint": _HWS_SUPPLY_WATER_SETPOINT,
}

# CPU seconds inside the C call (and the one-time build), for the timing table.
# process_time, not perf_counter: the gate compares this against sbx.evalrun's
# step_cpu_seconds, which is CPU time, and this machine is shared.
PROFILE = {"c_seconds": 0.0, "steps": 0, "build_seconds": 0.0, "builds": 0}


# --------------------------------------------------------------------------- #
# ctypes surface
# --------------------------------------------------------------------------- #
class _ObsSlot(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("index", ctypes.c_int32),
        ("slot", ctypes.c_int32),
        ("mean", ctypes.c_double),
        ("scale", ctypes.c_double),
    ]


class _VavCfg(ctypes.Structure):
    _fields_ = [
        ("max_air_flow_rate", ctypes.c_double),
        ("min_damper", ctypes.c_double),
        ("max_discharge_temp", ctypes.c_double),
        ("max_flow_static_pressure", ctypes.c_double),
        ("reheat_max_flow_factor", ctypes.c_double),
        ("ahu", ctypes.c_int32),
        ("coupled", ctypes.c_int32),
        ("zone_off", ctypes.c_int32),
        ("zone_n", ctypes.c_int32),
        ("area", ctypes.c_int32),
        ("area_all", ctypes.c_int32),
    ]


class _AhuCfg(ctypes.Structure):
    _fields_ = [
        ("recirculation", ctypes.c_double),
        ("supply_setpoint", ctypes.c_double),
        ("fan_static_pressure", ctypes.c_double),
        ("fan_efficiency", ctypes.c_double),
        ("max_air_flow_rate", ctypes.c_double),
        ("min_static_pressure", ctypes.c_double),
        ("max_recirculation", ctypes.c_double),
        ("cooling_cop", ctypes.c_double),
        ("run_command", ctypes.c_int32),
    ]


def _i32p(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


class _Cfg(ctypes.Structure):
    _fields_ = [
        ("nrows", ctypes.c_int32),
        ("ncols", ctypes.c_int32),
        ("n_zones", ctypes.c_int32),
        ("zone_off", ctypes.POINTER(ctypes.c_int32)),
        ("zone_cells", ctypes.POINTER(ctypes.c_int32)),
        ("zone_of_cv", ctypes.POINTER(ctypes.c_int32)),
        ("zone_area", ctypes.POINTER(ctypes.c_int32)),
        ("n_coupled", ctypes.c_int32),
        ("coupled_vav", ctypes.POINTER(ctypes.c_int32)),
        ("n_vav_zones", ctypes.c_int32),
        ("n_zone_vav", ctypes.c_int32),
        ("thermal", ctypes.c_void_p),
        ("n_vav", ctypes.c_int32),
        ("vavs", ctypes.POINTER(_VavCfg)),
        ("vav_zone", ctypes.POINTER(ctypes.c_int32)),
        ("zone_vav_off", ctypes.POINTER(ctypes.c_int32)),
        ("zone_vav", ctypes.POINTER(ctypes.c_int32)),
        ("n_ahu", ctypes.c_int32),
        ("ahus", ctypes.POINTER(_AhuCfg)),
        ("hws_energy_balance", ctypes.c_int32),
        ("hws_has_tank", ctypes.c_int32),
        ("hws_run", ctypes.c_int32),
        ("pump_run", ctypes.c_int32),
        ("hws_header_resistance", ctypes.c_double),
        ("boiler_heating_rate", ctypes.c_double),
        ("boiler_cooling_rate", ctypes.c_double),
        ("boiler_convection_coef", ctypes.c_double),
        ("boiler_tank_length", ctypes.c_double),
        ("boiler_tank_radius", ctypes.c_double),
        ("boiler_water_capacity", ctypes.c_double),
        ("boiler_insulation_conductivity", ctypes.c_double),
        ("boiler_insulation_thickness", ctypes.c_double),
        ("boiler_setpoint", ctypes.c_double),
        ("boiler_return_temp", ctypes.c_double),
        ("pump_head", ctypes.c_double),
        ("pump_efficiency", ctypes.c_double),
        ("occ_handle", ctypes.c_int64),
        ("occ_zone_id", ctypes.POINTER(ctypes.c_int32)),
        ("n_occ_all", ctypes.c_int32),
        ("occ_all", ctypes.POINTER(ctypes.c_int32)),
        ("n_rooms", ctypes.c_int32),
        ("rooms", ctypes.POINTER(ctypes.c_int64)),
        ("conv_rng", ctypes.c_int64),
        ("conv_p", ctypes.c_double),
        ("step_ns", ctypes.c_int64),
        ("origin_ns", ctypes.c_int64),
        ("n_horizon", ctypes.c_int32),
        ("ambient", ctypes.POINTER(ctypes.c_double)),
        ("comfort", ctypes.POINTER(ctypes.c_uint8)),
        ("conv_coef", ctypes.POINTER(ctypes.c_double)),
        ("feat_slot", ctypes.POINTER(ctypes.c_int32)),
        ("feat", ctypes.POINTER(ctypes.c_float)),
        ("n_feat", ctypes.c_int32),
        ("comfort_heat", ctypes.c_double),
        ("comfort_cool", ctypes.c_double),
        ("eco_heat", ctypes.c_double),
        ("eco_cool", ctypes.c_double),
        ("internal_gain_w_per_m2", ctypes.c_double),
        ("internal_gain_standby_fraction", ctypes.c_double),
        ("occupant_heat_w", ctypes.c_double),
        ("cv_area_m2", ctypes.c_double),
        ("time_step_sec", ctypes.c_double),
        ("n_obs", ctypes.c_int32),
        ("obs", ctypes.POINTER(_ObsSlot)),
        # Appended last: exactly one of `thermal` and `thermal_many` is set.
        ("thermal_many", ctypes.c_void_p),
    ]


class _Views(ctypes.Structure):
    _fields_ = [
        ("obs", ctypes.POINTER(ctypes.c_float)),
        ("mass", ctypes.POINTER(ctypes.c_double)),
        ("zone_temp", ctypes.POINTER(ctypes.c_double)),
        ("zone_heat_sp", ctypes.POINTER(ctypes.c_double)),
        ("zone_cool_sp", ctypes.POINTER(ctypes.c_double)),
        ("zone_flow_sp", ctypes.POINTER(ctypes.c_double)),
        ("zone_flow", ctypes.POINTER(ctypes.c_double)),
        ("zone_occupancy", ctypes.POINTER(ctypes.c_double)),
        ("ahu_intake", ctypes.POINTER(ctypes.c_double)),
        ("ahu_exhaust", ctypes.POINTER(ctypes.c_double)),
        ("ahu_cooling", ctypes.POINTER(ctypes.c_double)),
        ("vav_damper", ctypes.POINTER(ctypes.c_double)),
        ("vav_valve", ctypes.POINTER(ctypes.c_double)),
        ("vav_zone_temp", ctypes.POINTER(ctypes.c_double)),
        ("vav_heat_sp", ctypes.POINTER(ctypes.c_double)),
        ("vav_cool_sp", ctypes.POINTER(ctypes.c_double)),
        ("vav_flow", ctypes.POINTER(ctypes.c_double)),
        ("vav_ratio", ctypes.POINTER(ctypes.c_double)),
        ("vav_mode", ctypes.POINTER(ctypes.c_int32)),
        ("vav_prev", ctypes.POINTER(ctypes.c_int32)),
        ("ahu_flow", ctypes.POINTER(ctypes.c_double)),
        ("ahu_requests", ctypes.POINTER(ctypes.c_int32)),
        ("hws_flow_factor_sum", ctypes.POINTER(ctypes.c_double)),
        ("hws_requests", ctypes.POINTER(ctypes.c_int32)),
        ("hws_return", ctypes.POINTER(ctypes.c_double)),
        ("hws_heating", ctypes.POINTER(ctypes.c_double)),
        ("hws_pump", ctypes.POINTER(ctypes.c_double)),
        ("boiler_temp", ctypes.POINTER(ctypes.c_double)),
        ("boiler_change", ctypes.POINTER(ctypes.c_double)),
        ("boiler_last_dur", ctypes.POINTER(ctypes.c_double)),
        ("boiler_burner", ctypes.POINTER(ctypes.c_double)),
        ("flags", ctypes.POINTER(ctypes.c_int32)),
        ("diag", ctypes.POINTER(ctypes.c_double)),
        ("phase", ctypes.POINTER(ctypes.c_double)),
        ("n_phase", ctypes.POINTER(ctypes.c_int32)),
        ("counts", ctypes.POINTER(ctypes.c_int32)),
    ]


_BOUND = {"done": False}


def _bind():
    if _BOUND["done"]:
        return
    from .bindings import lib

    lib.cenv_fused_create.restype = ctypes.c_void_p
    lib.cenv_fused_create.argtypes = [ctypes.POINTER(_Cfg)]
    lib.cenv_fused_free.restype = None
    lib.cenv_fused_free.argtypes = [ctypes.c_void_p]
    lib.cenv_fused_views_get.restype = ctypes.c_int32
    lib.cenv_fused_views_get.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Views)]
    lib.cenv_fused_sync_full.restype = ctypes.c_int32
    lib.cenv_fused_sync_full.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_double, ctypes.c_int32,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_int64, ctypes.c_int64,
    ]
    lib.cenv_fused_run.restype = ctypes.c_int32
    lib.cenv_fused_run.argtypes = [
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _BOUND["done"] = True


def _dp(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _ary(ptr, count, dtype):
    if not ptr or count <= 0:
        return np.zeros(0, dtype=dtype)
    return np.ctypeslib.as_array(ptr, shape=(int(count),))


# --------------------------------------------------------------------------- #
# the state                                                                    #
# --------------------------------------------------------------------------- #
class FusedState:
    """One C handle plus the Python-side plans and buffers behind it."""

    def __init__(self, env, tbatch=None):
        self.env = env
        self.tbatch = tbatch   # cenv.batch.TBatch, or None for one environment
        self.n_rows = 0
        self.sim = env.building.simulator
        self.building = self.sim.building
        self.hvac = self.sim.hvac
        self.handle = None
        self.i = 0
        self.obs_ts = None
        self.reward_ts = None
        self._build()

    # -- construction ---------------------------------------------------- #
    def _build(self):
        _bind()
        from . import stochastic as stoch
        from .thermal import State as ThermalState

        env = self.env
        sim, building, hvac = self.sim, self.building, self.hvac
        t0 = time.process_time()
        if type(sim).__name__ != "SimulatorFlexibleGeometries":
            raise NotImplementedError(
                f"cenv fused: unsupported simulator {type(sim).__name__}"
            )
        solver = sim._solver
        self.dt = float(sim.time_step_sec)
        self.step_ns = int(round(self.dt * 1e9))
        self.origin_ns = int(sim.current_timestamp.value)
        self.origin_ts = sim.current_timestamp
        self.nrows, self.ncols = (int(x) for x in building.temp.shape)
        self.n_full = self.nrows * self.ncols

        # -- zones ------------------------------------------------------- #
        from smart_buildings.smart_control.simulator import constants as simconst

        self.zone_names = [
            z for z in building.room_dict if not simconst.is_non_physical_space(z)
        ]
        zone_index = {name: i for i, name in enumerate(self.zone_names)}
        self.n_zones = n_zones = len(self.zone_names)
        off = np.zeros(n_zones + 1, dtype=np.int32)
        cells = []
        zone_area = np.zeros(n_zones, dtype=np.int32)
        zone_of_cv = np.full(self.n_full, -1, dtype=np.int32)
        for i, name in enumerate(self.zone_names):
            coords = building.room_dict[name]
            flat = [int(r) * self.ncols + int(c) for r, c in coords]
            cells.extend(flat)
            zone_area[i] = len(flat)
            off[i + 1] = off[i] + len(flat)
            if flat:
                zone_of_cv[np.asarray(flat, dtype=np.int64)] = i
        self.zone_off = off
        self.zone_cells = np.ascontiguousarray(cells, dtype=np.int32)
        self.zone_area = zone_area
        self.zone_of_cv = zone_of_cv

        # -- VAVs -------------------------------------------------------- #
        vav_ids = list(hvac.vavs.keys())
        self.vav_ids = vav_ids
        self.vavs = [hvac.vavs[v] for v in vav_ids]
        self.n_vav = n_vav = len(vav_ids)
        ahu_of = {}
        self.ahus = (
            list(hvac.air_handler.ahus)
            if hasattr(hvac.air_handler, "ahus")
            else [hvac.air_handler]
        )
        self.n_ahu = len(self.ahus)
        for i, ahu in enumerate(self.ahus):
            ahu_of[id(ahu)] = i
        self.ahu_index = [ahu_of[id(v.air_handler)] for v in self.vavs]

        coupled = list(getattr(sim, "_coupled_vavs", []))
        self.coupled_ids = coupled
        col_of = {vid: c for c, vid in enumerate(coupled)}

        vav_zone = []
        vav_cfg = (_VavCfg * max(n_vav, 1))()
        for i, (vid, vav) in enumerate(zip(vav_ids, self.vavs)):
            all_zones = list(hvac.get_zones_for_vav(vid))
            zones = [z for z in all_zones if z in zone_index]
            idxs = [zone_index[z] for z in zones]
            area_all = sum(len(building.room_dict[z]) for z in all_zones
                           if z in building.room_dict)
            c = vav_cfg[i]
            c.max_air_flow_rate = float(vav._max_air_flow_rate)
            c.min_damper = float(vav._min_damper_setting)
            c.max_discharge_temp = float(vav._max_discharge_air_temperature)
            c.max_flow_static_pressure = float(vav._max_air_flow_static_pressure)
            c.reheat_max_flow_factor = float(vav._reheat_max_water_flow_factor)
            c.ahu = int(self.ahu_index[i])
            c.coupled = int(col_of.get(vid, -1))
            c.zone_off = len(vav_zone)
            c.zone_n = len(idxs)
            c.area = int(sum(zone_area[z] for z in idxs))
            c.area_all = int(area_all)
            vav_zone.extend(idxs)
        self.vav_cfg = vav_cfg
        self.vav_zone = np.ascontiguousarray(vav_zone, dtype=np.int32)

        # zone -> VAVs, in get_vav_ids_for_zone order
        zoff = np.zeros(n_zones + 1, dtype=np.int32)
        zone_vav = []
        for i, name in enumerate(self.zone_names):
            ids = list(hvac.get_vav_ids_for_zone(name))
            zone_vav.extend(vav_ids.index(v) for v in ids)
            zoff[i + 1] = len(zone_vav)
        self.zone_vav_off = zoff
        self.zone_vav = np.ascontiguousarray(zone_vav, dtype=np.int32)

        self.coupled_vav = np.ascontiguousarray(
            [vav_ids.index(v) for v in coupled], dtype=np.int32
        )

        # -- air handlers ------------------------------------------------ #
        ahu_cfg = (_AhuCfg * max(self.n_ahu, 1))()
        for i, ahu in enumerate(self.ahus):
            c = ahu_cfg[i]
            c.recirculation = float(ahu._recirculation)
            c.supply_setpoint = float(ahu._supply_air_temperature_setpoint)
            c.fan_static_pressure = float(ahu._fan_static_pressure)
            c.fan_efficiency = float(ahu._fan_efficiency)
            c.max_air_flow_rate = float(ahu._max_air_flow_rate)
            c.min_static_pressure = float(ahu._min_supply_air_static_pressure)
            c.max_recirculation = float(ahu._max_recirculation)
            c.cooling_cop = float(ahu._cooling_cop)
            c.run_command = int(ahu._run_command)
        self.ahu_cfg = ahu_cfg

        # -- hot water --------------------------------------------------- #
        hws = hvac.hot_water_system
        boiler = hws._heat_source
        from smart_buildings.smart_control.proto import (
            smart_control_building_pb2 as building_pb2,
        )

        if (
            hws.heat_source_device_type
            != building_pb2.DeviceInfo.DeviceType.BLR
        ):
            raise NotImplementedError(
                "cenv fused: only boiler hot water systems are supported"
            )
        self.boiler = boiler
        self.pump = hws._pump
        self.hws = hws
        self.hws_energy_balance = bool(getattr(boiler, "energy_balance", False))
        self.hws_has_tank = bool(getattr(boiler, "_has_tank", False))
        self.hws_id = hws.device_id()
        self.ahu_system_id = hvac.air_handler.device_id()

        # -- stochastic -------------------------------------------------- #
        occ = sim._occupancy
        if occ is None:
            raise NotImplementedError("cenv fused: the simulator has no occupancy")
        state = occ.__dict__.get("_cenv_occ")
        if state is None:
            state = stoch._OccState(occ)
            occ._cenv_occ = state
        self.occ_state = state
        occ_id = np.zeros(n_zones, dtype=np.int32)
        for i, name in enumerate(self.zone_names):
            if name not in state.zones:
                state.zones[name] = len(state.zones)
            occ_id[i] = state.zones[name]
        self.occ_id = occ_id
        zone_names_ordered = [z.zone_id for z in env.building.zones]
        occ_all = [
            state.zones.setdefault(name, len(state.zones))
            for name in zone_names_ordered
        ]
        self.occ_all = np.ascontiguousarray(occ_all, dtype=np.int32)

        conv = building.convection_simulator
        if conv is not None:
            self.conv_p = float(conv.p)
            cs = stoch._conv_state(conv, building.room_dict, building.temp.shape)
            self.conv_rooms = np.ascontiguousarray(cs.rooms, dtype=np.int64)
            self.conv_rng = int(cs.rng)
            self.conv_state = cs
        else:
            self.conv_p = 0.0
            self.conv_rooms = np.zeros(0, dtype=np.int64)
            self.conv_rng = 0
            self.conv_state = None

        # -- thermal ----------------------------------------------------- #
        # One factorization per (solver, h). A single environment owns it here
        # (scipy's SuperLU factors, exported to the C kernel); a batch shares
        # one Cholesky over all M environments and this handle keeps none (see
        # cenv/batch.py), so the batched solve -- not this object -- steps it.
        h = float(
            sim.weather_controller.get_air_convection_coefficient(
                sim.current_timestamp
            )
        )
        self.thermal_h = h
        if self.tbatch is None:
            self.thermal = ThermalState(solver, h)
            self.n_rows = int(self.thermal.n)
        else:
            self.thermal = None
            self.n_rows = int(self.tbatch.n)

        # -- per-step time tables ---------------------------------------- #
        n_horizon = int(env._num_timesteps_in_episode) + 2
        self.n_horizon = n_horizon
        self._build_tables(n_horizon)
        # Cover the whole episode on the occupancy model's local-time grid:
        # the kernel queries absolute instants, and a query off the grid is an
        # error there rather than a pandas dance.
        self._ensure_occupancy_grid()

        # -- observation plan -------------------------------------------- #
        self._build_obs_plan()

        # -- the C handle ------------------------------------------------ #
        self._create()

        # -- initial device state from the Python objects ----------------- #
        self._sync_full()
        PROFILE["build_seconds"] += time.process_time() - t0
        PROFILE["builds"] += 1

    # -- time tables ----------------------------------------------------- #
    def _ensure_occupancy_grid(self):
        """Cover the whole table horizon on the occupancy local-time grid."""
        self.occ_state.ensure(self.origin_ns)
        self.occ_state.ensure(self.origin_ns + self.n_horizon * self.step_ns)

    def _build_tables(self, n_horizon):
        import pandas as pd

        sim = self.sim
        env = self.env
        n = int(n_horizon)
        self.n_horizon = n
        idx = pd.to_datetime(
            [self.origin_ns + j * self.step_ns for j in range(n)]
        )
        weather = sim.weather_controller
        schedule = self.hvac.schedule
        ambient = np.empty(n, dtype=np.float64)
        comfort = np.empty(n, dtype=np.uint8)
        conv = np.empty(n, dtype=np.float64)
        for j in range(n):
            ts = idx[j]
            # The tables are filled eagerly (and doubled on extension), the clock
            # is not: entries past the end of the weather data are NaN, and step()
            # raises the simulator's own error only if a step really reaches one
            # (building_train_4/phoenix_8's data ends exactly at the episode end).
            try:
                ambient[j] = weather.get_current_temp(ts)
            except ValueError:
                ambient[j] = np.nan
            try:
                conv[j] = weather.get_air_convection_coefficient(ts)
            except ValueError:
                conv[j] = np.nan
            comfort[j] = 1 if schedule.is_comfort_mode(ts) else 0
        known = conv[~np.isnan(conv)]
        if known.size and np.any(known != known[0]):
            raise NotImplementedError(
                "cenv fused: the convection coefficient varies over the episode"
            )
        self.ambient = ambient
        self.comfort = comfort
        self.conv_coef = conv
        self._timestamps = idx

        from smart_buildings.smart_control.utils import (
            conversion_utils,
            regression_building_utils,
        )

        n_hod = int(env._num_hod_features)
        n_dow = int(env._num_dow_features)
        from smart_buildings.smart_control.environment import environment as env_mod

        hod_label = env_mod.HOD_LABEL
        dow_label = env_mod.DOW_LABEL
        hod_names = regression_building_utils.get_time_feature_names(
            n_hod, hod_label
        )
        dow_names = regression_building_utils.get_time_feature_names(
            n_dow, dow_label
        )
        feat_names = list(hod_names) + list(dow_names)
        feat = np.empty((len(feat_names), n), dtype=np.float32)
        for j in range(n):
            ts = idx[j]
            hod_rad = conversion_utils.get_radian_time(
                ts, conversion_utils.TimeIntervalEnum.HOUR_OF_DAY
            )
            dow_rad = conversion_utils.get_radian_time(
                ts, conversion_utils.TimeIntervalEnum.DAY_OF_WEEK
            )
            values = list(
                regression_building_utils.expand_time_features(
                    n_hod, hod_rad, hod_label
                ).values()
            ) + list(
                regression_building_utils.expand_time_features(
                    n_dow, dow_rad, dow_label
                ).values()
            )
            feat[:, j] = np.asarray(values, dtype=np.float32)
        self.feat_names = [f"{a}_{b}" for a, b in feat_names]
        self.feat_table = np.ascontiguousarray(feat)

    # -- observation plan ------------------------------------------------ #
    def _build_obs_plan(self):
        env = self.env
        inv = env._id_map.inv
        normalizer = env._observation_normalizer
        names = list(env.field_names)
        self.obs_names = names
        self.n_obs = n_obs = len(names)
        devices = env.building._device_map
        vav_pos = {v.device_id(): i for i, v in enumerate(self.vavs)}
        ahu_pos = {a.device_id(): i for i, a in enumerate(self.ahus)}
        sys_id = self.ahu_system_id
        feat_row = {name: i for i, name in enumerate(self.feat_names)}
        slots = np.zeros(n_obs, dtype=[("kind", "i4"), ("index", "i4"),
                                       ("slot", "i4"), ("mean", "f8"),
                                       ("scale", "f8")])
        feat_slot = []
        for i, name in enumerate(names):
            mean, scale = 0.0, 0.0
            if name in inv:
                device_id, measurement = inv[name]
                constants = normalizer._get_normalization_constants(measurement)
                mean = float(constants.sample_mean)
                variance = float(constants.sample_variance)
                scale = float(np.sqrt(variance)) if variance > 0.0 else 0.0
                device = devices[device_id]
                if device_id in vav_pos:
                    kind = _VAV_KINDS.get(measurement)
                    if kind is None:
                        raise NotImplementedError(
                            f"cenv fused: unhandled VAV observable {measurement!r}"
                        )
                    index = vav_pos[device_id]
                elif device_id == sys_id and hasattr(device, "ahus"):
                    parts = measurement.split("_", 2)
                    prefix = parts[0]
                    base = parts[2] if len(parts) == 3 else ""

                    if prefix != "ahu" or base not in _AHU_KINDS:
                        raise NotImplementedError(
                            f"cenv fused: unhandled AHU observable {measurement!r}"
                        )
                    kind = _AHU_KINDS[base]
                    index = int(measurement.split("_", 2)[1]) - 1
                elif device_id in ahu_pos:
                    kind = _AHU_KINDS.get(measurement)
                    if kind is None:
                        raise NotImplementedError(
                            f"cenv fused: unhandled AHU observable {measurement!r}"
                        )
                    index = ahu_pos[device_id]
                elif device_id == self.hws_id:
                    kind = _HWS_KINDS.get(measurement)
                    if kind is None:
                        raise NotImplementedError(
                            f"cenv fused: unhandled HWS observable {measurement!r}"
                        )
                    index = 0
                else:
                    raise NotImplementedError(
                        f"cenv fused: unhandled device {device_id!r} "
                        f"({measurement!r})"
                    )
                slots[i] = (kind, int(index), i, mean, scale)
            elif name in (_COMFORT_NOW, _COMFORT_SOON, _NUM_OCCUPANTS):
                kind = {
                    _COMFORT_NOW: _AUX_COMFORT_NOW,
                    _COMFORT_SOON: _AUX_COMFORT_SOON,
                    _NUM_OCCUPANTS: _AUX_NUM_OCCUPANTS,
                }[name]
                slots[i] = (kind, 0, i, 0.0, 1.0)
            elif name in feat_row:
                slots[i] = (_TIME, 0, i, 0.0, 0.0)
                feat_slot.append(i)
            else:
                raise NotImplementedError(
                    f"cenv fused: unhandled observation field {name!r}"
                )
        self.slots = slots
        self.feat_slot = np.ascontiguousarray(feat_slot, dtype=np.int32)
        self.n_feat = len(feat_slot)
        self._slot_struct = (_ObsSlot * n_obs)()
        for i in range(n_obs):
            row = slots[i]
            self._slot_struct[i].kind = int(row["kind"])
            self._slot_struct[i].index = int(row["index"])
            self._slot_struct[i].slot = int(row["slot"])
            self._slot_struct[i].mean = float(row["mean"])
            self._slot_struct[i].scale = float(row["scale"])

    # -- C handle -------------------------------------------------------- #
    def _create(self):
        from .bindings import lib

        env = self.env
        hvac = self.hvac
        sim = self.sim
        boiler = self.boiler
        pump = self.pump
        cfg = _Cfg()
        cfg.nrows = self.nrows
        cfg.ncols = self.ncols
        cfg.n_zones = self.n_zones
        cfg.zone_off = _i32p(self.zone_off)
        cfg.zone_cells = _i32p(self.zone_cells) if self.zone_cells.size else _i32p(
            np.zeros(1, dtype=np.int32)
        )
        cfg.zone_of_cv = _i32p(self.zone_of_cv)
        cfg.zone_area = _i32p(self.zone_area) if self.n_zones else _i32p(
            np.zeros(1, dtype=np.int32)
        )
        cfg.n_coupled = int(self.coupled_vav.size)
        cfg.coupled_vav = _i32p(self.coupled_vav) if self.coupled_vav.size else _i32p(
            np.zeros(1, dtype=np.int32)
        )
        cfg.n_vav_zones = int(self.vav_zone.size)
        cfg.n_zone_vav = int(self.zone_vav.size)
        cfg.thermal = ctypes.c_void_p(
            self.thermal.handle if self.thermal is not None else 0
        )
        cfg.thermal_many = ctypes.c_void_p(
            self.tbatch.handle if self.tbatch is not None else 0
        )
        cfg.n_vav = self.n_vav
        cfg.vavs = self.vav_cfg
        cfg.vav_zone = _i32p(self.vav_zone) if self.vav_zone.size else _i32p(
            np.zeros(1, dtype=np.int32)
        )
        cfg.zone_vav_off = _i32p(self.zone_vav_off)
        cfg.zone_vav = _i32p(self.zone_vav) if self.zone_vav.size else _i32p(
            np.zeros(1, dtype=np.int32)
        )
        cfg.n_ahu = self.n_ahu
        cfg.ahus = self.ahu_cfg
        cfg.hws_energy_balance = int(self.hws_energy_balance)
        cfg.hws_has_tank = int(self.hws_has_tank)
        cfg.hws_run = int(self.hws._run_command)
        cfg.pump_run = int(pump._run_command)
        cfg.hws_header_resistance = float(self.hws._header_resistance)
        cfg.boiler_heating_rate = float(getattr(boiler, "_heating_rate", 0.0))
        cfg.boiler_cooling_rate = float(getattr(boiler, "_cooling_rate", 0.0))
        cfg.boiler_convection_coef = float(
            getattr(boiler, "_convection_coefficient", 0.0)
        )
        cfg.boiler_tank_length = float(getattr(boiler, "_tank_length", 0.0))
        cfg.boiler_tank_radius = float(getattr(boiler, "_tank_radius", 0.0))
        cfg.boiler_water_capacity = float(getattr(boiler, "_water_capacity", 0.0))
        cfg.boiler_insulation_conductivity = float(
            getattr(boiler, "_insulation_conductivity", 0.0)
        )
        cfg.boiler_insulation_thickness = float(
            getattr(boiler, "_insulation_thickness", 0.0)
        )
        cfg.boiler_setpoint = float(boiler._supply_water_temperature_setpoint)
        cfg.boiler_return_temp = float(boiler.return_water_temperature_sensor)
        cfg.pump_head = float(pump._water_pump_differential_head)
        cfg.pump_efficiency = float(pump._water_pump_efficiency)
        cfg.occ_handle = ctypes.c_int64(self.occ_state.handle)
        cfg.occ_zone_id = _i32p(self.occ_id) if self.n_zones else _i32p(
            np.zeros(1, dtype=np.int32)
        )
        cfg.n_occ_all = int(self.occ_all.size)
        cfg.occ_all = _i32p(self.occ_all) if self.occ_all.size else _i32p(
            np.zeros(1, dtype=np.int32)
        )
        cfg.n_rooms = int(self.conv_rooms.size)
        cfg.rooms = self.conv_rooms.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int64)
        ) if self.conv_rooms.size else ctypes.cast(
            ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_int64)
        )
        cfg.conv_rng = ctypes.c_int64(self.conv_rng)
        cfg.conv_p = float(self.conv_p)
        cfg.step_ns = self.step_ns
        cfg.origin_ns = self.origin_ns
        cfg.n_horizon = self.n_horizon
        cfg.ambient = _dp(self.ambient)
        cfg.comfort = self.comfort.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )
        cfg.conv_coef = _dp(self.conv_coef)
        cfg.feat_slot = _i32p(self.feat_slot) if self.feat_slot.size else _i32p(
            np.zeros(1, dtype=np.int32)
        )
        cfg.feat = self.feat_table.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cfg.n_feat = int(self.n_feat)
        schedule = hvac.schedule
        cfg.comfort_heat = float(schedule.comfort_temp_window[0])
        cfg.comfort_cool = float(schedule.comfort_temp_window[1])
        cfg.eco_heat = float(schedule.eco_temp_window[0])
        cfg.eco_cool = float(schedule.eco_temp_window[1])
        cfg.internal_gain_w_per_m2 = float(
            getattr(sim, "_internal_gain_w_per_m2", 0.0)
        )
        cfg.internal_gain_standby_fraction = float(
            getattr(sim, "_internal_gain_standby_fraction", 0.2)
        )
        cfg.occupant_heat_w = float(getattr(sim, "_occupant_sensible_heat_w", 0.0))
        cfg.cv_area_m2 = (float(self.building.cv_size_cm) / 100.0) ** 2
        cfg.time_step_sec = self.dt
        cfg.n_obs = self.n_obs
        cfg.obs = self._slot_struct
        handle = lib.cenv_fused_create(ctypes.byref(cfg))
        if not handle:
            raise MemoryError("cenv_fused_create failed")
        self.handle = ctypes.c_void_p(handle)
        views = _Views()
        rc = lib.cenv_fused_views_get(self.handle, ctypes.byref(views))
        if rc != 0:
            raise RuntimeError(f"cenv_fused_views_get failed ({rc})")
        self.views = views
        counts = _ary(views.counts, 4, np.int32)
        self.counts = counts
        self.obs_array = _ary(views.obs, self.n_obs, np.float32)
        self.mass_array = _ary(views.mass, self.n_rows, np.float64)
        self.o_zone_temp = _ary(views.zone_temp, self.n_zones, np.float64)
        self.o_zone_heat = _ary(views.zone_heat_sp, self.n_zones, np.float64)
        self.o_zone_cool = _ary(views.zone_cool_sp, self.n_zones, np.float64)
        self.o_zone_flow_sp = _ary(views.zone_flow_sp, self.n_zones, np.float64)
        self.o_zone_flow = _ary(views.zone_flow, self.n_zones, np.float64)
        self.o_zone_occ = _ary(views.zone_occupancy, self.n_zones, np.float64)
        self.o_ahu_intake = _ary(views.ahu_intake, self.n_ahu, np.float64)
        self.o_ahu_exhaust = _ary(views.ahu_exhaust, self.n_ahu, np.float64)
        self.o_ahu_cooling = _ary(views.ahu_cooling, self.n_ahu, np.float64)
        self.o_vav_damper = _ary(views.vav_damper, self.n_vav, np.float64)
        self.o_vav_valve = _ary(views.vav_valve, self.n_vav, np.float64)
        self.o_vav_temp = _ary(views.vav_zone_temp, self.n_vav, np.float64)
        self.o_vav_heat = _ary(views.vav_heat_sp, self.n_vav, np.float64)
        self.o_vav_cool = _ary(views.vav_cool_sp, self.n_vav, np.float64)
        self.o_vav_flow = _ary(views.vav_flow, self.n_vav, np.float64)
        self.o_vav_ratio = _ary(views.vav_ratio, self.n_vav, np.float64)
        self.o_vav_mode = _ary(views.vav_mode, self.n_vav, np.int32)
        self.o_vav_prev = _ary(views.vav_prev, self.n_vav, np.int32)
        self.o_ahu_flow = _ary(views.ahu_flow, self.n_ahu, np.float64)
        self.o_ahu_requests = _ary(views.ahu_requests, self.n_ahu, np.int32)
        self.o_hws_fsum = _ary(views.hws_flow_factor_sum, 1, np.float64)
        self.o_hws_requests = _ary(views.hws_requests, 1, np.int32)
        self.o_hws_return = _ary(views.hws_return, 1, np.float64)
        self.o_hws_heating = _ary(views.hws_heating, 1, np.float64)
        self.o_hws_pump = _ary(views.hws_pump, 1, np.float64)
        self.o_boiler_temp = _ary(views.boiler_temp, 1, np.float64)
        self.o_boiler_change = _ary(views.boiler_change, 1, np.float64)
        self.o_boiler_dur = _ary(views.boiler_last_dur, 1, np.float64)
        self.o_boiler_burner = _ary(views.boiler_burner, 1, np.float64)
        self.o_flags = _ary(views.flags, 1, np.int32)
        self.diag = _ary(views.diag, 4, np.float64)
        self.phase = _ary(views.phase, 16, np.float64)
        self.sync = np.zeros(
            2 * self.n_vav + 4 * self.n_ahu + 4 + 2, dtype=np.float64
        )

    # -- device state <-> kernel ----------------------------------------- #
    def _index_of(self, ts):
        if ts is None:
            return -2
        return int((int(ts.value) - self.origin_ns) // self.step_ns)

    def _sync_full(self, import_mass=True):
        """Push the Python device state into the kernel (once per reset).

        ``import_mass`` is off when the kernel is rebuilt mid-episode: the
        interior mass nodes then live in C, and ``solver._mass_temp`` is
        whatever the last reset left there.
        """
        from .bindings import lib

        vav = self.vavs
        n = self.n_vav
        damper = np.ascontiguousarray(
            [v._damper_setting for v in vav], dtype=np.float64
        )
        valve = np.ascontiguousarray(
            [v._reheat_valve_setting for v in vav], dtype=np.float64
        )
        zone_temp = np.ascontiguousarray(
            [v._zone_air_temperature for v in vav], dtype=np.float64
        )
        heat = np.ascontiguousarray(
            [v._zone_air_heating_temperature_setpoint for v in vav],
            dtype=np.float64,
        )
        cool = np.ascontiguousarray(
            [v._zone_air_cooling_temperature_setpoint for v in vav],
            dtype=np.float64,
        )
        ratio = np.ascontiguousarray(
            [v._supply_air_flow_ratio for v in vav], dtype=np.float64
        )
        mode = np.ascontiguousarray(
            [int(v._thermostat._current_mode.value) for v in vav], dtype=np.int32
        )
        prev = np.ascontiguousarray(
            [self._index_of(v._thermostat._previous_timestamp) for v in vav],
            dtype=np.int32,
        )
        ahu_flow = np.ascontiguousarray(
            [a._air_flow_rate for a in self.ahus], dtype=np.float64
        )
        ahu_req = np.ascontiguousarray(
            [int(a._cooling_request_count) for a in self.ahus], dtype=np.int32
        )
        boiler = self.boiler
        rc = lib.cenv_fused_sync_full(
            self.handle,
            _dp(damper), _dp(valve), _dp(zone_temp), _dp(heat), _dp(cool),
            _dp(ratio),
            mode.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            prev.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            _dp(ahu_flow),
            ahu_req.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            float(self.hws._flow_factor_sum),
            int(self.hws._heating_request_count),
            float(boiler._current_temperature),
            float(boiler._supply_water_temperature_setpoint),
            float(boiler.return_water_temperature_sensor),
            float(boiler._step_tank_temperature_change),
            float(boiler._last_step_duration.total_seconds()),
            float(getattr(boiler, "_burner_heat_rate", 0.0)),
            self._index_of(boiler._observation_timestamp),
            self._index_of(boiler._action_timestamp),
        )
        if rc != 0:
            raise RuntimeError(f"cenv_fused_sync_full failed ({rc})")
        # The interior mass nodes are the solver's, not the building's; an
        # episode reset puts them at the temperature of their own CV.
        if import_mass:
            mass = self.sim._solver._mass_temp
            if mass is not None and self.mass_array.size == np.asarray(mass).size:
                np.copyto(self.mass_array, np.asarray(mass, dtype=np.float64))

    def _sync_step(self):
        """The commands the action request just applied to the devices."""
        sync = self.sync
        n = self.n_vav
        for i, v in enumerate(self.vavs):
            d = v._damper_command
            sync[i] = np.nan if d is None else d
            r = v._reheat_valve_command
            sync[n + i] = np.nan if r is None else r
        base = 2 * n
        for i, a in enumerate(self.ahus):
            b = base + 4 * i
            sync[b] = 1.0 if a._run_command else 0.0
            sync[b + 1] = a._supply_air_temperature_setpoint
            sync[b + 2] = a._fan_static_pressure
            sync[b + 3] = a._recirculation
        b = base + 4 * self.n_ahu
        sync[b] = 1.0 if self.hws._run_command else 0.0
        sync[b + 1] = self.boiler._supply_water_temperature_setpoint
        sync[b + 2] = self.pump._water_pump_differential_head
        sync[b + 3] = 1.0 if self.pump._run_command else 0.0
        sync[b + 4] = self._index_of(self.boiler._observation_timestamp)
        action_ts = self.boiler._action_timestamp
        sync[b + 5] = -2 if action_ts is None else self._index_of(action_ts)

    def _write_back(self):
        """Keep the Python device objects equal to the kernel's state."""
        sim = self.sim
        ts = sim._current_timestamp
        damper = self.o_vav_damper
        valve = self.o_vav_valve
        zone_temp = self.o_vav_temp
        heat = self.o_vav_heat
        cool = self.o_vav_cool
        ratio = self.o_vav_ratio
        mode = self.o_vav_mode
        for i, v in enumerate(self.vavs):
            v._damper_setting = float(damper[i])
            v._reheat_valve_setting = float(valve[i])
            v._zone_air_temperature = float(zone_temp[i])
            v._zone_air_heating_temperature_setpoint = float(heat[i])
            v._zone_air_cooling_temperature_setpoint = float(cool[i])
            v._supply_air_flow_ratio = float(ratio[i])
            thermostat = v._thermostat
            thermostat._current_mode = thermostat.Mode(int(mode[i]))
            thermostat._previous_timestamp = ts
        for a, ahu in enumerate(self.ahus):
            ahu._air_flow_rate = float(self.o_ahu_flow[a])
            ahu._cooling_request_count = int(self.o_ahu_requests[a])
        self.hws._flow_factor_sum = float(self.o_hws_fsum[0])
        self.hws._heating_request_count = int(self.o_hws_requests[0])
        self.hws.return_water_temperature_sensor = float(self.o_hws_return[0])
        boiler = self.boiler
        boiler._current_temperature = float(self.o_boiler_temp[0])
        boiler._step_tank_temperature_change = float(self.o_boiler_change[0])
        boiler._last_step_duration = pd_Timedelta(
            float(self.o_boiler_dur[0]), unit="s"
        )
        boiler._burner_heat_rate = float(self.o_boiler_burner[0])
        boiler._observation_timestamp = ts
        self.hws._observation_timestamp = ts

    # -- the step -------------------------------------------------------- #
    def step(self):
        if self.tbatch is not None:
            raise RuntimeError(
                "cenv batch: this environment is stepped by BatchEnv"
                " (cenv/batch.py), which issues one solve for all M; call"
                " BatchEnv.step([...]) instead."
            )
        from .bindings import lib

        sim = self.sim
        i = self.i
        if i + 2 >= self.n_horizon:
            self._extend_horizon(i + 64)
        for j in (i, i + 1):
            if np.isnan(self.ambient[j]) or np.isnan(self.conv_coef[j]):
                # The plain step queries the weather at this clock and fails here.
                sim.weather_controller.get_current_temp(self._timestamps[j])
                sim.weather_controller.get_air_convection_coefficient(self._timestamps[j])
        self._sync_step()
        temp = self.building.temp
        if temp.dtype != np.float64 or not temp.flags["C_CONTIGUOUS"]:
            temp = np.ascontiguousarray(temp, dtype=np.float64)
            self.building.temp = temp
        q = self.building.input_q
        if q.dtype != np.float64 or not q.flags["C_CONTIGUOUS"]:
            q = np.ascontiguousarray(q, dtype=np.float64)
            self.building.input_q = q
        t0 = time.process_time()
        rc = lib.cenv_fused_run(self.handle, i, _dp(temp), _dp(q), _dp(self.sync))
        PROFILE["c_seconds"] += time.process_time() - t0
        PROFILE["steps"] += 1
        if rc != 0:
            raise RuntimeError(f"cenv_fused_run failed ({rc}) at step {i}")
        self._post_step_bookkeeping(i)

    def _post_step_bookkeeping(self, i: int) -> None:
        """Advance the clock and publish the step's outputs (both paths).

        The batched driver calls this between its solve and the environments'
        observation/reward halves, so the order the single-environment step
        uses -- clock, device write-back, then the observation -- is kept.
        """
        sim = self.sim
        sim._current_timestamp = sim._current_timestamp + pd_Timedelta(
            self.dt, unit="s"
        )
        self.i = i + 1
        self._write_back()
        self.obs_ts = sim._current_timestamp
        self.reward_ts = sim._current_timestamp
        if self.o_flags[0]:
            self.o_flags[0] = 0
            import logging

            logging.warning(
                "cenv fused: temperature field left physical bounds and was"
                " clipped at step %d.", i
            )

    def _extend_horizon(self, want):
        # The interior mass nodes are the kernel's between resets, so carry
        # them across the rebuild.
        mass = self.mass_array.copy()
        self._build_tables(max(2 * self.n_horizon, int(want)))
        self._ensure_occupancy_grid()
        if self.handle is not None:
            from .bindings import lib

            lib.cenv_fused_free(self.handle)
            self.handle = None
        self._create()
        self._sync_full(import_mass=False)
        if mass.size == self.mass_array.size:
            np.copyto(self.mass_array, mass)

    # -- reward info ----------------------------------------------------- #
    def build_reward_info(self, occupancy_function=None):
        """The RewardInfo the Python reward path would build, from the kernel."""
        from smart_buildings.smart_control.proto import (
            smart_control_reward_pb2 as reward_pb2,
        )
        from smart_buildings.smart_control.utils import conversion_utils

        ts = self.sim._current_timestamp
        info = reward_pb2.RewardInfo()
        info.start_timestamp.CopyFrom(
            conversion_utils.pandas_to_proto_timestamp(ts)
        )
        info.end_timestamp.CopyFrom(
            conversion_utils.pandas_to_proto_timestamp(
                ts + pd_Timedelta(self.dt, unit="s")
            )
        )
        zone_infos = info.zone_reward_infos
        heat = self.o_zone_heat
        cool = self.o_zone_cool
        temp = self.o_zone_temp
        flow_sp = self.o_zone_flow_sp
        flow = self.o_zone_flow
        occ = self.o_zone_occ
        for i, name in enumerate(self.zone_names):
            zi = zone_infos[name]
            zi.heating_setpoint_temperature = float(heat[i])
            zi.cooling_setpoint_temperature = float(cool[i])
            zi.zone_air_temperature = float(temp[i])
            zi.air_flow_rate_setpoint = float(flow_sp[i])
            zi.air_flow_rate = float(flow[i])
            zi.average_occupancy = float(occ[i])
        blower = sum(self.o_ahu_intake) + sum(self.o_ahu_exhaust)
        cooling = sum(self.o_ahu_cooling)
        shim = info.air_handler_reward_infos[self.ahu_system_id]
        shim.blower_electrical_energy_rate = float(blower)
        shim.air_conditioning_electrical_energy_rate = float(cooling)
        boiler_info = info.boiler_reward_infos[self.hws_id]
        boiler_info.natural_gas_heating_energy_rate = float(
            self.o_hws_heating[0]
        )
        boiler_info.pump_electrical_energy_rate = float(self.o_hws_pump[0])
        return info

    def phase_table(self):
        """CPU ms per step for each phase of the kernel (see fused.h)."""
        names = [
            "zone_means", "vav_update", "recirc", "ahu_demand", "coupled",
            "gains", "input_q", "thermal", "air_foldback", "convection",
            "return_water", "heat_draw", "clip", "observation", "reward_info",
            "device_state",
        ]
        steps = max(self.i, 1)
        return {
            name: float(self.phase[k]) / steps * 1e3
            for k, name in enumerate(names)
            if k < self.phase.size
        }

    # -- lifetime -------------------------------------------------------- #
    def close(self):
        from .bindings import lib

        if self.handle is not None:
            lib.cenv_fused_free(self.handle)
            self.handle = None

    def key(self):
        sim = self.sim
        return (
            id(sim),
            id(sim._solver),
            id(sim.building),
            id(sim.hvac),
            tuple(np.shape(sim.building.temp)),
        )

    def refresh(self):
        """Start a new episode on the same objects (after env.reset)."""
        sim = self.sim
        self.i = 0
        self.obs_ts = None
        self.reward_ts = None
        origin = int(sim.current_timestamp.value)
        if origin != self.origin_ns:
            self.origin_ns = origin
            self.origin_ts = sim.current_timestamp
            self._build_tables(self.n_horizon)
            self._ensure_occupancy_grid()
            if self.handle is not None:
                from .bindings import lib

                lib.cenv_fused_free(self.handle)
                self.handle = None
            self._create()
        self._sync_full()


def pd_Timedelta(value, unit="s"):
    """pd.Timedelta, imported lazily so this module loads without pandas."""
    import pandas as pd

    return pd.Timedelta(value, unit=unit)


# --------------------------------------------------------------------------- #
# installation
# --------------------------------------------------------------------------- #
_MARK = "_cenv_fused"
_ORIG = {}
_patched = False


def _state_of(env):
    return env.building.simulator.__dict__.get(_MARK)


def _fused_wait_time(self):
    state = self.simulator.__dict__.get(_MARK)
    if state is None:
        return _ORIG["wait_time"](self)
    return state.step()


def _fused_get_observation(self):
    state = _state_of(self)
    if state is not None and state.obs_ts is not None:
        if state.obs_ts == self.building.current_timestamp:
            return state.obs_array.copy()
    return _ORIG["observation"](self)


def _fused_reward_info(self, occupancy_function):
    state = self.__dict__.get(_MARK)
    if state is not None and state.reward_ts is not None:
        if state.reward_ts == self._current_timestamp:
            return state.build_reward_info(occupancy_function)
    return _ORIG["reward_info"](self, occupancy_function)


def _fused_reset(self, *, seed=None, options=None):
    out = _ORIG["reset"](self, seed=seed, options=options)
    ensure_state(self)
    return out


def _patch_classes():
    """(Re)apply the method patches; idempotent and order independent."""
    global _patched
    from smart_buildings.smart_control.environment import environment
    from smart_buildings.smart_control.simulator import simulator
    from smart_buildings.smart_control.simulator import simulator_building

    env_cls = environment.Environment
    sim_cls = simulator.Simulator
    bld_cls = simulator_building.SimulatorBuilding
    if not _patched:
        _ORIG["wait_time"] = bld_cls.wait_time
        _ORIG["observation"] = env_cls._get_observation
        _ORIG["reward_info"] = sim_cls.reward_info
        _ORIG["reset"] = env_cls.reset
        _patched = True
    # Always rewrite, so a component installed after this one cannot take the
    # methods back: fused is the step, and its fallbacks are the originals.
    bld_cls.wait_time = _fused_wait_time
    env_cls._get_observation = _fused_get_observation
    sim_cls.reward_info = _fused_reward_info
    env_cls.reset = _fused_reset


def _structural_key(env):
    sim = env.building.simulator
    return (
        id(sim),
        id(sim._solver),
        id(sim.building),
        id(sim.hvac),
        tuple(np.shape(sim.building.temp)),
    )


def ensure_state(env):
    """Build (or refresh) the kernel for this environment after a reset."""
    sim = env.building.simulator
    _patch_classes()
    state = sim.__dict__.get(_MARK)
    if state is not None and state.key() == _structural_key(env):
        state.refresh()
        return state
    if state is not None:
        state.close()
    state = FusedState(env)
    sim._cenv_fused = state
    return state


def install(sim_root=None, verbose: bool = False):
    """Install ``fused``: one C call per step, both simulator trees."""
    from . import install as cenv_install
    from . import live
    from smart_buildings.smart_control.environment import environment
    from smart_buildings.smart_control.simulator import simulator
    from smart_buildings.smart_control.simulator import simulator_building

    if "bookkeeping" not in live():
        # The reward arithmetic on the RewardInfo this module fills.
        cenv_install(["bookkeeping"], sim_root=sim_root, verbose=verbose)
    _patch_classes()
    if verbose:
        print("cenv: installed fused (env.step -> one C call)")
    return {
        "patched": [
            "SimulatorBuilding.wait_time",
            "Environment._get_observation",
            "Environment.reset",
            "Simulator.reward_info",
        ],
        "restore": [
            (simulator_building.SimulatorBuilding, "wait_time",
             _ORIG["wait_time"]),
            (environment.Environment, "_get_observation", _ORIG["observation"]),
            (environment.Environment, "reset", _ORIG["reset"]),
            (simulator.Simulator, "reward_info", _ORIG["reward_info"]),
        ],
        "supersedes": ["observation"],
    }
