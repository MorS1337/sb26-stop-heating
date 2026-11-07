"""Integrity audit: what the simulator's accounting artifacts are worth, measured.

Two artifacts of the shipped tree are quantified here, on the same episodes the
leaderboard runs:

* **B2 -- the return-air weighting.** ``simulator_flexible_floor_plan.py`` gives
  each air handler the *floor-area* mean of the temperatures of the zones its
  VAVs serve, while the zones exchange heat with the supply air at their own
  airflow. The recirculated stream therefore carries an enthalpy no air stream
  delivered, and the handler's mixing box can generate or destroy heat for free.
  For an air handler with recirculation ``r``, total mass flow ``mdot`` and the
  two candidate return temperatures the imbalance is

      dQ = r * mdot * c_p * (T_area_weighted - T_airflow_weighted)

  which is Kozzen's PR #3 formula (topic 753804) and is what this module
  records, together with the part of it that actually reaches the zones
  (``supply_gap``: the delivered supply-air temperature against the one the
  airflow-weighted return would give) and the mixing-box entropy generation
  (negative is thermodynamically impossible).

* **B1 -- the boiler bill.** ``boiler.compute_thermal_energy_rate`` bills
  ``cw * rho_w * flow * (supply_water_setpoint - return_water)`` while the reheat
  coils draw at the tank's own temperature, so the burned gas and the heat the
  coils delivered are two different numbers. Both are captured here, at the
  point the reward reads them, so their ratio is the billing mismatch.

Everything is read-only instrumentation. No hook changes an argument, a return
value or a simulator attribute: the wrappers call the original and return its
result unchanged, and the only state they touch is the recorder's own counters.
The one quantity that must *not* be read directly is the boiler's
``supply_water_temperature_sensor`` -- in the shipped tree every read advances
the tank -- so the burned gas is taken from a wrapper on the model method that
already ran, never from a second sensor read.

Usage::

    # per-step B2/B1 account for one policy, one scenario
    python -m sbx.integrity run --policy policies.ctrl_w:CtrlW \
        --scenario building_train_1/fairbanks_1 --steps 2016 --sim shipped \
        --fast --tag ia_anchor_ctrlw

Results per scenario: ``results/<tag>/<building>__<weather>.json`` plus an
``.npz`` with the per-step series.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# One BLAS thread per process, before numpy loads (spawned workers re-import).
for _blas_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from sbx import evalrun, simload  # noqa: E402

DEFAULT_STEPS = 2016

# One per-step row, in a fixed order so an unhooked control run and an
# instrumented run produce arrays that can be compared element by element.
ROW_KEYS = (
    "hour",
    "weekday",
    "reward",
    "npr",
    "ec",
    "ce",
    "occupancy",
    "reheat",
    "gas_drawn",
    "gas_billed",
    "imbalance",
    "supply_gap",
    "s_mix",
    "s_mix_negative",
    "recirc",
    "flow",
)

# Columns of one air handler's row: flow, recirculation, T_area, T_flow, T_mix,
# T_supply, imbalance, supply gap, mixing entropy, VAVs, coldest zone, warmest,
# rated airflow.
AHU_COLS = 13
DEFAULT_SCENARIO = "building_train_1/fairbanks_1"


# --------------------------------------------------------------------------- #
# recorder
# --------------------------------------------------------------------------- #
class IntegrityRecorder:
    """Samples the B2 imbalance and the B1 billing mismatch once per step.

    The sample is taken inside ``execute_step_sim``, at the moment the handler
    computes its supply-air temperature -- after ``update_settings`` has set the
    dampers and before the thermal solve -- because that is the only point at
    which both the return temperature the simulator uses and the flows the VAVs
    asked for are simultaneously current.
    """

    def __init__(self, env, b2: bool = True, b1: bool = True, mode: str = "all"):
        from smart_buildings.smart_control.simulator import (  # noqa: PLC0415
            air_handler as air_handler_py,
        )
        from smart_buildings.smart_control.simulator import (  # noqa: PLC0415
            constants as sim_const,
        )
        from smart_buildings.smart_control.utils import (  # noqa: PLC0415
            constants as util_const,
        )
        from smart_buildings.smart_control.utils import (  # noqa: PLC0415
            conversion_utils,
        )

        self._rho_cp = sim_const.AIR_DENSITY * sim_const.AIR_HEAT_CAPACITY
        self._cw_rho_w = util_const.WATER_HEAT_CAPACITY * util_const.WATER_DENSITY
        self._proto_to_ts = conversion_utils.proto_to_pandas_timestamp

        # "all": every wrapper; "reward": only the reward wrapper, which is
        # what a control run needs to log the reward's own components while the
        # simulator runs untouched; "none": nothing installed at all.
        self.mode = str(mode)
        if self.mode not in ("all", "reward", "none"):
            raise ValueError(f"unknown recorder mode {mode!r}")
        self.hook = self.mode != "none"
        self.b2 = bool(b2) and self.mode == "all"
        self.b1 = bool(b1) and self.mode == "all"
        self.env = env
        self.sim = env.building.simulator
        self.hvac = self.sim.hvac
        system = self.hvac.air_handler
        self.ahus = (
            list(system.ahus)
            if isinstance(system, air_handler_py.AirHandlerSystem)
            else [system]
        )
        self.vavs = dict(self.hvac.vavs)
        self.boiler = self.hvac.hot_water_system._heat_source
        self.step_sec = float(self.sim.time_step_sec)

        self._in_step = False
        self._sampled = False
        self._reheat = 0.0
        self._gas: dict[str, float] = {}
        self.reward_info = None
        self.reward_response = None
        self.rows: list[dict[str, float]] = []
        self.ahu_rows: list[list[float]] = []
        self.ahu_ids = [a.device_id() for a in self.ahus]
        if self.hook:
            self._install()

    # -- hooks ------------------------------------------------------------- #
    def _install(self) -> None:
        if self.mode == "all":
            self._install_sim_hooks()
        rf = self.env.reward_function
        orig_reward = rf.compute_reward

        def compute_reward(info):
            response = orig_reward(info)
            self.reward_info = info
            self.reward_response = response
            return response

        rf.compute_reward = compute_reward

    def _install_sim_hooks(self) -> None:
        sim = self.sim
        orig_step = sim.execute_step_sim
        orig_supply = sim.hvac.air_handler.get_supply_air_temp

        def execute_step_sim(*args, **kwargs):
            self._in_step = True
            self._sampled = False
            self._reheat = 0.0
            try:
                return orig_step(*args, **kwargs)
            finally:
                self._in_step = False

        def get_supply_air_temp(recirculation, ambient):
            supply = orig_supply(recirculation, ambient)
            if self._in_step and not self._sampled:
                self._sampled = True
                if self.b2:
                    self._record_b2(recirculation, ambient, supply)
            return supply

        sim.execute_step_sim = execute_step_sim
        sim.hvac.air_handler.get_supply_air_temp = get_supply_air_temp

        for vav in self.vavs.values():
            orig_reheat = vav.compute_reheat_energy_rate

            def reheat(supply_air_temp, input_water_temp, _f=orig_reheat):
                rate = _f(supply_air_temp, input_water_temp)
                if self._in_step:
                    self._reheat += float(rate)
                return rate

            vav.compute_reheat_energy_rate = reheat

        if self.b1:
            boiler = self.boiler
            if hasattr(boiler, "apply_heat_draw"):
                orig_draw = boiler.apply_heat_draw

                def apply_heat_draw(coil_heat_w, outside_temp, duration_s):
                    orig_draw(coil_heat_w, outside_temp, duration_s)
                    self._gas = {
                        "drawn": float(coil_heat_w),
                        "billed": float(getattr(boiler, "_burner_heat_rate", 0.0)),
                    }
                    return None

                boiler.apply_heat_draw = apply_heat_draw
            else:
                orig_rate = boiler.compute_thermal_energy_rate

                def compute_thermal_energy_rate(
                    return_water_temp, outside_temp, total_flow_rate
                ):
                    billed = orig_rate(
                        return_water_temp, outside_temp, total_flow_rate
                    )
                    setpoint = boiler._supply_water_temperature_setpoint
                    supply = (
                        setpoint if setpoint > return_water_temp else return_water_temp
                    )
                    self._gas = {
                        "drawn": float(
                            self._cw_rho_w
                            * total_flow_rate
                            * (supply - return_water_temp)
                        ),
                        "billed": float(billed),
                    }
                    return billed

                boiler.compute_thermal_energy_rate = compute_thermal_energy_rate

    # -- B2 ----------------------------------------------------------------- #
    def _record_b2(self, recirculation, ambient: float, supply) -> None:
        sim, hvac = self.sim, self.sim.hvac
        temps = sim.building.get_zone_average_temps()
        areas = {z: len(c) for z, c in sim.building.room_dict.items()}
        ahu_row: list[float] = []
        for ahu in self.ahus:
            ahu_id = ahu.device_id()
            # ``recirculation`` is a per-handler map for an AirHandlerSystem and
            # a scalar for a single handler; the same holds for the supply.
            t_used = (
                float(recirculation[ahu_id])
                if isinstance(recirculation, dict)
                else float(recirculation)
            )
            t_supply = (
                float(supply[ahu_id]) if isinstance(supply, dict) else float(supply)
            )
            # The streams each VAV extracts, at the airflow it actually asked
            # for. A VAV the step skips (closed damper, no rated flow) delivers
            # no air and contributes no weight; the handler's flow cap scales
            # every VAV it serves by the same ratio, so it cancels in t_flow.
            flow: list[float] = []
            temp: list[float] = []
            for v_id, vav in hvac.vavs.items():
                if vav.air_handler.device_id() != ahu_id:
                    continue
                if not (vav.damper_setting > 0 and vav.max_air_flow_rate > 0):
                    continue
                zones = [z for z in hvac.get_zones_for_vav(v_id) if z in temps]
                area = sum(areas[z] for z in zones)
                if area <= 0:
                    continue
                flow.append(float(vav.unrestricted_flow_rate_demand))
                temp.append(
                    sum(temps[z] * areas[z] for z in zones) / area
                )
            if not flow:
                ahu_row.extend((0.0,) * AHU_COLS)
                continue
            f = np.asarray(flow)
            t = np.asarray(temp)
            f = f * min(1.0, float(ahu.max_air_flow_rate) / f.sum())
            g = self._rho_cp * f
            g_total = float(g.sum())
            t_flow = float(g @ t / g_total)
            flow_total = float(f.sum())
            r = float(ahu.recirculation)
            t_mix = r * t_used + (1.0 - r) * ambient
            t_mix_true = r * t_flow + (1.0 - r) * ambient
            imbalance = r * g_total * (t_used - t_flow)
            # The part of it that survives the cooling coil: when the mixed air
            # is below the supply setpoint the coil passes it through untouched,
            # so the zones receive air that is dQ/(1) warmer than the physics of
            # their own extract streams allows.
            setpoint = float(ahu.supply_air_temperature_setpoint)
            # smart_device.RunStatus.ON == 1.
            running = int(ahu.run_command) == 1
            t_supply_true = (
                setpoint if (t_mix_true > setpoint and running) else t_mix_true
            )
            # The handler's own ``air_flow_rate`` is still the reset value at
            # this point in the step (the VAVs add their demand afterwards), so
            # the delivered flow is the sum of the capped VAV streams.
            supply_gap = flow_total * self._rho_cp * (t_supply - t_supply_true)
            # Entropy generated in the mixing box by the streams that really
            # exist (recirculated extract at each VAV's zone temperature plus
            # outdoor air) ending at the mixed temperature the simulator uses.
            s_mix = r * float(g @ np.log(t_mix / t)) + (1.0 - r) * g_total * float(
                np.log(t_mix / ambient)
            )
            ahu_row.extend(
                (
                    flow_total,
                    r,
                    t_used,
                    t_flow,
                    t_mix,
                    t_supply,
                    imbalance,
                    supply_gap,
                    s_mix,
                    float(len(flow)),
                    float(t.min()),
                    float(t.max()),
                    float(ahu.max_air_flow_rate),
                )
            )
        self.ahu_rows.append(ahu_row)

    # -- per-step ---------------------------------------------------------- #
    def sample(self, reward: float) -> None:
        if self.mode == "none":
            self.rows.append(
                {name: 0.0 for name in ROW_KEYS} | {"reward": float(reward)}
            )
            return
        if self.reward_info is None:
            raise RuntimeError("the reward hook never fired for this step")
        info = self.reward_info
        start = self._proto_to_ts(info.start_timestamp)
        response = self.reward_response
        row = {
            "hour": float(start.hour),
            "weekday": float(start.weekday()),
            "reward": float(reward),
            "npr": float(response.normalized_productivity_regret),
            "ec": float(response.normalized_energy_cost),
            "ce": float(response.normalized_carbon_emission),
            "occupancy": float(response.total_occupancy),
            "reheat": float(self._reheat),
            "gas_drawn": float(self._gas.get("drawn", 0.0)),
            "gas_billed": float(self._gas.get("billed", 0.0)),
        }
        if self.ahu_rows and len(self.ahu_rows) > len(self.rows):
            arr = np.asarray(self.ahu_rows[-1], dtype=float).reshape(-1, AHU_COLS)
            row["imbalance"] = float(arr[:, 6].sum())
            row["supply_gap"] = float(arr[:, 7].sum())
            row["s_mix"] = float(arr[:, 8].sum())
            row["s_mix_negative"] = float((arr[:, 8] < 0.0).sum())
            row["recirc"] = float(arr[:, 1].mean())
            row["flow"] = float(arr[:, 0].sum())
        else:  # B2 off
            row["imbalance"] = 0.0
            row["supply_gap"] = 0.0
            row["s_mix"] = 0.0
            row["s_mix_negative"] = 0.0
            row["recirc"] = 0.0
            row["flow"] = 0.0
        self.rows.append(row)

    # -- aggregate --------------------------------------------------------- #
    def aggregate(self) -> dict[str, Any]:
        steps = len(self.rows)
        data = {k: np.array([r[k] for r in self.rows]) for k in self.rows[0]}
        hours = self.step_sec / 3600.0
        kwh = hours / 1e3  # W * h -> kWh

        def kwh_of(values: np.ndarray, sign: int = 0) -> float:
            if sign > 0:
                values = np.maximum(values, 0.0)
            elif sign < 0:
                values = np.maximum(-values, 0.0)
            return float(values.sum() * kwh)

        reheat_pos = kwh_of(data["reheat"], +1)
        per_ahu: dict[str, Any] = {}
        if self.ahu_rows:
            arr = np.asarray(self.ahu_rows, dtype=float).reshape(
                len(self.ahu_rows), len(self.ahus), AHU_COLS
            )
            for i, ahu_id in enumerate(self.ahu_ids):
                a = arr[:, i, :]
                per_ahu[ahu_id] = {
                    "imbalance_kw_mean": float(a[:, 6].mean() / 1e3),
                    "imbalance_kwh_net": float(a[:, 6].sum() * kwh),
                    "imbalance_kwh_heat": kwh_of(a[:, 6], +1),
                    "imbalance_kwh_cool": kwh_of(a[:, 6], -1),
                    "imbalance_kw_max": float(a[:, 6].max() / 1e3),
                    "imbalance_kw_min": float(a[:, 6].min() / 1e3),
                    "supply_gap_kwh_heat": kwh_of(a[:, 7], +1),
                    "supply_gap_kwh_cool": kwh_of(a[:, 7], -1),
                    "t_area_mean": float(a[:, 2].mean()),
                    "t_flow_mean": float(a[:, 3].mean()),
                    "delta_t_mean": float((a[:, 2] - a[:, 3]).mean()),
                    "recirculation_mean": float(a[:, 1].mean()),
                    "airflow_mean": float(a[:, 0].mean()),
                    "t_coldest_mean": float(a[:, 10][a[:, 9] > 0].mean())
                    if (a[:, 9] > 0).any()
                    else None,
                    "t_warmest_mean": float(a[:, 11][a[:, 9] > 0].mean())
                    if (a[:, 9] > 0).any()
                    else None,
                    "abs_delta_t_mean": float(np.abs(a[:, 2] - a[:, 3]).mean()),
                    # Same-state ceiling: the whole rated flow of the handler
                    # pushed through its coldest zone, which is the most
                    # favourable split the current temperature field allows.
                    "ceiling_kwh_heat": float(
                        (
                            a[:, 1]
                            * a[:, 12]
                            * self._rho_cp
                            * np.maximum(a[:, 2] - a[:, 10], 0.0)
                        ).sum()
                        * kwh
                    ),
                    "steps_negative_entropy": int((a[:, 8] < 0.0).sum()),
                }
        return {
            "steps": steps,
            "total_reward": float(data["reward"].sum()),
            "mean_per_step_reward": float(data["reward"].mean()),
            "sum_comfort": float(data["npr"].sum()),
            "sum_energy": float(data["ec"].sum()),
            "sum_carbon": float(data["ce"].sum()),
            "b2": {
                "imbalance_kwh_net": kwh_of(data["imbalance"], 0),
                "imbalance_kwh_heat": kwh_of(data["imbalance"], +1),
                "imbalance_kwh_cool": kwh_of(data["imbalance"], -1),
                "imbalance_kw_mean": float(data["imbalance"].mean() / 1e3),
                "imbalance_kw_max": float(data["imbalance"].max() / 1e3),
                "imbalance_kw_min": float(data["imbalance"].min() / 1e3),
                "supply_gap_kwh_heat": kwh_of(data["supply_gap"], +1),
                "supply_gap_kwh_cool": kwh_of(data["supply_gap"], -1),
                "steps_negative_entropy": int((data["s_mix_negative"] > 0).sum()),
                "step_fraction_negative_entropy": float(
                    (data["s_mix_negative"] > 0).mean()
                ),
                "reheat_kwh": reheat_pos,
                "share_of_reheat": (
                    kwh_of(data["imbalance"], +1) / reheat_pos
                    if reheat_pos > 0.0
                    else None
                ),
                "passthru_share_of_reheat": (
                    kwh_of(data["supply_gap"], +1) / reheat_pos
                    if reheat_pos > 0.0
                    else None
                ),
                "recirculation_mean": float(data["recirc"].mean()),
                "flow_mean": float(data["flow"].mean()),
                "per_ahu": per_ahu,
            },
            "b1": {
                "gas_billed_kwh": kwh_of(data["gas_billed"], 0),
                "gas_drawn_kwh": kwh_of(data["gas_drawn"], 0),
                "billed_over_drawn": (
                    kwh_of(data["gas_billed"], 0) / kwh_of(data["gas_drawn"], 0)
                    if kwh_of(data["gas_drawn"], 0) > 0.0
                    else None
                ),
                "reheat_kwh": reheat_pos,
            },
        }

    def arrays(self) -> dict[str, np.ndarray]:
        out = {
            k: np.array([r[k] for r in self.rows], dtype=np.float64)
            for k in self.rows[0]
        }
        if self.ahu_rows:
            out["ahu"] = np.asarray(self.ahu_rows, dtype=np.float64).reshape(
                len(self.rows), len(self.ahus), AHU_COLS
            )
        return out


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    started_wall = time.time()
    simload.use_sim(task["sim"])
    from smart_buildings import evaluate  # noqa: PLC0415

    fast_report = None
    if task.get("fast"):
        from sbx import fastpatch  # noqa: PLC0415

        fast_report = fastpatch.apply(simload.sim_root(task["sim"]))

    policy_cls, policy_label = evalrun._resolve_policy(task["policy"], evaluate)
    config_path = Path(task["config_path"])
    scenario_dir = config_path.parent
    bindings: dict[str, Any] = {}
    if task.get("occupancy_seed") is not None:
        bindings["RandomizedArrivalDepartureOccupancy.seed"] = int(
            task["occupancy_seed"]
        )
    env = evalrun._load_environment(evaluate, config_path, bindings)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = evaluate.create_building_meta_data(env, scenario_dir)
    action_spec = env.action_spec()
    if task["params"]:
        policy_cls.PARAMS = {
            **(getattr(policy_cls, "PARAMS", None) or {}),
            **task["params"],
        }
    policy = evaluate.initialize_policy(policy_cls, meta, action_spec)
    recorder = IntegrityRecorder(
        env,
        b2=not task.get("no_b2"),
        b1=not task.get("no_b1"),
        mode=(
            "none"
            if task.get("no_hooks")
            else ("reward" if task.get("reward_only") else "all")
        ),
    )

    oracle = None
    observe_oracle = getattr(policy, "observe_oracle", None)
    sources = evalrun.parse_oracle(task.get("oracle"))
    if sources:
        oracle = evalrun._OracleChannel(
            env,
            meta,
            policy,
            sources,
            int(task.get("oracle_horizon") or evalrun.DEFAULT_ORACLE_HORIZON),
            scenario=task["scenario"],
            oracle_file=task.get("oracle_file"),
            record_dir=task.get("oracle_record"),
        )

    obs, _ = env.reset()
    rewards: list[float] = []
    started_cpu = time.process_time()
    for _ in range(int(task["steps"])):
        if oracle is not None:
            payload = oracle.info(
                recorder.reward_info, env.current_simulation_timestamp
            )
            if callable(observe_oracle):
                observe_oracle(payload)
        action = evaluate.validate_action(
            evaluate.get_policy_action(policy, obs), action_spec
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        recorder.sample(float(reward))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    cpu = time.process_time() - started_cpu
    if oracle is not None:
        oracle.finish()

    record = {
        "scenario": task["scenario"],
        "policy": policy_label,
        "params": task["params"] or {},
        "sim": task["sim"],
        "fast": list(fast_report or ()),
        "tag": task["tag"],
        "start_offset_days": int(task.get("start_offset_days") or 0),
        "occupancy_seed": task.get("occupancy_seed"),
        "oracle": list(sources),
        "oracle_horizon": int(task.get("oracle_horizon") or 0) or None,
        "requested_steps": int(task["steps"]),
        "wall_seconds": time.time() - started_wall,
        "cpu_seconds": cpu,
        **recorder.aggregate(),
    }
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = task["scenario"].replace("/", "__")
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2))
    np.savez_compressed(out_dir / f"{stem}.npz", **recorder.arrays())
    record["record_path"] = str(out_dir / f"{stem}.json")
    return record


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    try:
        return _run_one(task)
    except Exception:  # noqa: BLE001 - reported, not swallowed
        return {"scenario": task.get("scenario"), "error": traceback.format_exc()}


def run(
    policy: str = "policies.ctrl_w:CtrlW",
    scenarios_: list[str] | None = None,
    steps: int = DEFAULT_STEPS,
    sim: str = "shipped",
    workers: int = 1,
    tag: str = "ia",
    out: str | Path | None = None,
    params: dict[str, Any] | None = None,
    fast: bool = False,
    start_offset_days: int = 0,
    occupancy_seed: int | None = None,
    oracle: Any = None,
    oracle_horizon: int | None = None,
    oracle_file: str | None = None,
    oracle_record: str | None = None,
    b2: bool = True,
    b1: bool = True,
    mode: str = "all",
) -> dict[str, Any]:
    """Runs instrumented episodes and returns a summary."""
    if sim not in simload.SIM_NAMES:
        raise ValueError(f"unknown sim {sim!r}")
    items = list(scenarios_) or [DEFAULT_SCENARIO]
    pairs = evalrun._expand_scenarios(items, sim)
    if not pairs:
        raise ValueError("no scenarios selected")
    out_dir = Path(out) if out else WORKSPACE / "results" / tag
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "sim": sim,
            "scenario": scenario,
            "config_path": str(path.resolve()),
            "out_dir": str(out_dir),
            "policy": policy,
            "params": dict(params or {}),
            "steps": int(steps),
            "tag": tag,
            "fast": bool(fast),
            "start_offset_days": int(start_offset_days),
            "occupancy_seed": occupancy_seed,
            "oracle": oracle,
            "oracle_horizon": oracle_horizon,
            "oracle_file": oracle_file,
            "oracle_record": oracle_record,
            "no_b2": not b2,
            "no_b1": not b1,
            "no_hooks": mode == "none",
            "reward_only": mode == "reward",
        }
        for scenario, path in pairs
    ]
    print(
        f"[integrity] tag={tag} sim={sim} policy={policy} scenarios={len(tasks)} "
        f"steps={steps} workers={workers} -> {out_dir}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    started = time.time()
    # One process per scenario, always when there is more than one: gin only
    # registers its configurables once per interpreter, so a second scenario in
    # the same process fails to import ``smart_buildings`` again.
    if workers <= 1 and len(tasks) == 1:
        for task in tasks:
            result = _worker(task)
            results.append(result)
            print(f"[integrity] {result.get('scenario')} done", flush=True)
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=max(1, workers), maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(_worker, tasks, chunksize=1):
                results.append(result)
                print(
                    f"[integrity] {result.get('scenario')} "
                    f"{'FAILED' if 'error' in result else 'done'}",
                    flush=True,
                )
    summary = {
        "tag": tag,
        "policy": policy,
        "params": params or {},
        "sim": sim,
        "steps": int(steps),
        "fast": bool(fast),
        "out_dir": str(out_dir),
        "wall_seconds": time.time() - started,
        "mean_total_reward": (
            float(np.mean([r["total_reward"] for r in results if "error" not in r]))
            if any("error" not in r for r in results)
            else None
        ),
        "scenarios": [
            {
                "scenario": r.get("scenario"),
                "error": r.get("error"),
                "total_reward": r.get("total_reward"),
                "b2": r.get("b2"),
                "b1": r.get("b1"),
            }
            for r in results
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    bad = [r for r in results if "error" in r]
    print(
        f"[integrity] {len(results) - len(bad)}/{len(results)} ok, "
        f"wall {time.time() - started:.0f}s -> {out_dir / 'summary.json'}",
        flush=True,
    )
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="instrumented rollout")
    p.add_argument("--policy", default="policies.ctrl_w:CtrlW")
    p.add_argument("--scenario", nargs="+", action="append", default=None)
    p.add_argument("--set", dest="scenario_set", default=None)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--sim", default="shipped", choices=list(simload.SIM_NAMES))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tag", default="ia")
    p.add_argument("--out", default=None)
    p.add_argument("--params", default=None)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--start-offset-days", type=int, default=0)
    p.add_argument("--occupancy-seed", type=int, default=None)
    p.add_argument("--oracle", default=None)
    p.add_argument("--oracle-horizon", type=int, default=None)
    p.add_argument("--oracle-file", default=None)
    p.add_argument("--oracle-record", default=None)
    p.add_argument("--no-b2", action="store_true")
    p.add_argument("--no-b1", action="store_true")
    p.add_argument("--no-hooks", action="store_true",
                   help="control run: install no wrapper at all")
    p.add_argument("--reward-only", action="store_true",
                   help="control run: reward wrapper only, simulator untouched")
    args = parser.parse_args(argv)
    if args.command == "run":
        scenarios_ = None
        if args.scenario:
            scenarios_ = [s for group in args.scenario for s in group]
        if args.scenario_set:
            scenarios_ = (scenarios_ or []) + [args.scenario_set]
        run(
            policy=args.policy,
            scenarios_=scenarios_,
            steps=args.steps,
            sim=args.sim,
            workers=args.workers,
            tag=args.tag,
            out=args.out,
            params=json.loads(args.params) if args.params else None,
            fast=args.fast,
            start_offset_days=args.start_offset_days,
            occupancy_seed=args.occupancy_seed,
            oracle=args.oracle,
            oracle_horizon=args.oracle_horizon,
            oracle_file=args.oracle_file,
            oracle_record=args.oracle_record,
            b2=not args.no_b2,
            b1=not args.no_b1,
            mode=(
                "none"
                if args.no_hooks
                else ("reward" if args.reward_only else "all")
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
