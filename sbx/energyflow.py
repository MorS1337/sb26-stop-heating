"""Energy-flow instrumentation and decomposition of a CtrlW rollout.

The paper's central energy claim needs a per-joule account of what a policy is
billed for, so this module instruments a normal :mod:`sbx.evalrun` episode and
records, at every 5-minute step, the physical quantities behind the reward's two
fuel terms:

* per air handler -- supply airflow, recirculation (and hence the outside-air
  fraction), mixed-air, return-air and supply-air temperatures, the coil's
  thermal load, its electrical draw, and the supply and exhaust fan powers;
* per VAV -- airflow, reheat-coil thermal power, and the served zones' mean
  temperature;
* hot water -- billed gas, the coil draw the water loop carried, the tank's
  standing loss, the tank's storage term and the pump power;
* per zone -- mean temperature and occupancy, straight from the reward's own
  ``RewardInfo``;
* per step -- outside temperature and the reward's energy/carbon/comfort terms.

Two things make the numbers trustworthy rather than approximate:

1. **Nothing is re-derived by a second sensor read.** In the shipped simulator
   every read of the boiler's supply-water sensor advances the tank, and the
   reward path reads it more than once per step. So the quantities that are only
   valid at the moment the reward is computed -- the coil draw the boiler bills,
   the tank's step change, the entering-water temperature each coil actually saw
   -- are captured by wrapping the *model* methods (``compute_thermal_energy_rate``,
   ``compute_reheat_energy_rate``, ``apply_heat_draw``) and recording their
   arguments and returns as the step computes them. The wrappers add no state of
   their own.
2. **The decomposition is a counterfactual the simulator can answer.** The
   ventilation share of the heating and cooling load is defined as the marginal
   load the outside-air fraction imposes: the same episode re-evaluated as if
   the mixing box recirculated everything (``get_supply_air_temp(T_return,
   T_return)``), which is the one-line counterfactual the model already
   supports. That is exact -- the outside-air fraction cannot be zero while the
   handler is running, so the counterfactual is the right limit -- and it
   automatically respects the coil's saturation. The analytic cross-check
   ``(1 - recirculation) * airflow * rho * c_p * (T_return - T_out)`` is logged
   too: it is the heat the outside air carries, which is an upper bound on what
   the coils can be relieved of.

Usage::

    # one scenario, full episode, instrumented
    python -m sbx.energyflow run --policy policies.ctrl_w:CtrlW \
        --scenario building_train_1/fairbanks_1 --steps 8641 --sim shipped \
        --workers 2 --tag ef_shipped

    # table + figure from stored rollouts
    python -m sbx.energyflow report --tag ef_shipped

Results per scenario: ``results/<tag>/<building>__<weather>.json`` (aggregates,
reconciliation and the decomposition by bucket and by hour) and
``...__<weather>.npz`` (the per-step series, for reuse by other analyses).
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

DEFAULT_STEPS = 8641

# Buckets: the runner's weekday comfort window, plus the two windows that matter
# for this policy. ``CtrlW`` starts its preheat ramp two hours before the
# comfort window opens (``early_preheat_hours = 2``), so the preheat window is
# 04:00-06:00 on a weekday; night is everything on a weekday outside both.
BUCKETS = ("weekday_occupied", "preheat", "weekday_night", "weekend")

# The decomposition. Every entry is a non-negative power in W -- thermal for the
# two gas-heat entries and the two chiller entries, electrical for the rest --
# except where a signed share is reported separately.
GAS_COMPONENTS = ("heat_vent", "heat_rest", "boiler_standby")

COMPONENTS = (
    "heat_vent",  # gas: coil duty the outside-air fraction imposes
    "heat_rest",  # gas: remaining coil duty (envelope, internal gains, storage)
    "boiler_standby",  # gas: everything billed that the coils did not draw
    "cool_vent",  # electricity: chiller draw the outside-air fraction imposes
    "cool_rest",  # electricity: the rest of the chiller draw
    "fan_supply",  # electricity: supply (intake) fans
    "fan_exhaust",  # electricity: exhaust fans, proportional to outside air
    "pump",  # electricity: hot-water pump
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _area_weighted_temps(
    sim, hvac, ahu_zones: dict[str, list[str]], avg: dict[str, float] | None = None
) -> dict[str, float]:
    """Returns each air handler's mixed return-air temperature in K.

    Mirrors the shipped simulator's ``_get_air_handler_reward_infos``: the zones
    a VAV serves are averaged by floor area, and the VAVs of an air handler by
    their zones' total area. ``sims/corrected`` replaces this with an
    airflow-weighted mixture (PR #3) and exposes it as ``_recirculation_temps``;
    where that exists it is used, so the logged return temperature is the one
    the reward actually billed against.
    """
    if avg is None:
        avg = sim.building.get_zone_average_temps()
    if hasattr(sim, "_recirculation_temps"):
        return dict(sim._recirculation_temps(avg))
    areas = {z: len(c) for z, c in sim.building.room_dict.items()}
    out = {}
    for ahu_id, zones in ahu_zones.items():
        present = [z for z in zones if z in avg]
        area = sum(areas[z] for z in present)
        out[ahu_id] = (
            sum(avg[z] * areas[z] for z in present) / area
            if area > 0
            else float(sim.building.temp.mean())
        )
    return out


def _bucket(hour: int, weekday: int, morning: int, evening: int) -> str:
    if weekday >= 5:
        return "weekend"
    if morning <= hour < evening:
        return "weekday_occupied"
    if max(0, morning - 2) <= hour < morning:
        return "preheat"
    return "weekday_night"


# --------------------------------------------------------------------------- #
# recorder
# --------------------------------------------------------------------------- #
class FlowRecorder:
    """Instruments one environment in place and samples it after each step."""

    def __init__(self, env, meta: dict[str, Any]):
        from smart_buildings.smart_control.simulator import constants as sim_const
        from smart_buildings.smart_control.utils import constants as util_const
        from smart_buildings.smart_control.utils import conversion_utils

        self._proto_to_ts = conversion_utils.proto_to_pandas_timestamp
        self._rho_air = sim_const.AIR_DENSITY
        self._cp_air = sim_const.AIR_HEAT_CAPACITY
        self._cw_rho_w = util_const.WATER_HEAT_CAPACITY * util_const.WATER_DENSITY

        self.env = env
        self.sim = env.building.simulator
        self.hvac = self.sim.hvac
        self.hws = self.hvac.hot_water_system
        self.boiler = self.hws._heat_source
        system = self.hvac.air_handler
        self.ahus = list(getattr(system, "ahus", [system]))
        self.ahus_by_id = {a.device_id(): a for a in self.ahus}
        if hasattr(system, "ahu_zones_map"):
            self.ahu_zones = {
                a.device_id(): list(system.ahu_zones_map.get(a, []))
                for a in self.ahus
            }
        else:  # single handler: every zone
            self.ahu_zones = {
                self.ahus[0].device_id(): list(self.sim.building.room_dict)
            }
        self.vavs = dict(self.hvac.vavs)
        self.vav_ids = list(self.vavs)
        self.vav_ahu = {
            v_id: v.air_handler.device_id() for v_id, v in self.vavs.items()
        }
        self.zones = list(self.sim.building.room_dict)
        self.morning = int(meta["morning_start_hour"])
        self.evening = int(meta["evening_start_hour"])
        self.step_sec = float(self.sim.time_step_sec)
        self.sim_name: str | None = None

        self.reward_info = None
        self.reward_response = None
        self._reheat: dict[str, float] = {v_id: 0.0 for v_id in self.vav_ids}
        self._reheat_water: dict[str, float] = {v_id: 0.0 for v_id in self.vav_ids}
        self._gas: dict[str, Any] = {}
        self._gas_calls = 0
        self.rows: list[dict[str, float]] = []
        self.ahu_rows: list[list[float]] = []
        self.vav_rows: list[list[float]] = []
        self.zone_rows: list[list[float]] = []

        self._orig_reward = env.reward_function.compute_reward
        self._orig_reheat = {
            v_id: v.compute_reheat_energy_rate for v_id, v in self.vavs.items()
        }
        self._install_hooks()

    # -- hooks ------------------------------------------------------------- #
    def _install_hooks(self) -> None:
        rf = self.env.reward_function

        def compute_reward(info):
            response = self._orig_reward(info)
            self.reward_info = info
            self.reward_response = response
            return response

        rf.compute_reward = compute_reward

        for v_id, vav in self.vavs.items():
            orig = self._orig_reheat[v_id]

            def reheat(supply_air_temp, input_water_temp, _id=v_id, _f=orig):
                rate = _f(supply_air_temp, input_water_temp)
                self._reheat[_id] = rate
                self._reheat_water[_id] = input_water_temp
                return rate

            vav.compute_reheat_energy_rate = reheat

        boiler = self.boiler
        if hasattr(boiler, "apply_heat_draw"):
            # sims/corrected: gas is the burner heat of an energy-balanced tank.
            orig_draw = boiler.apply_heat_draw

            def apply_heat_draw(coil_heat_w, outside_temp, duration_s):
                before = boiler._current_temperature
                orig_draw(coil_heat_w, outside_temp, duration_s)
                self._gas_calls += 1
                self._gas = {
                    "coil": float(coil_heat_w),
                    "dissipation": float(
                        boiler.compute_thermal_dissipation_rate(
                            before, outside_temp
                        )
                    ),
                    "storage": (
                        float(
                            self._cw_rho_w
                            * boiler._water_capacity
                            * (boiler._current_temperature - before)
                            / duration_s
                        )
                        if duration_s > 0
                        else 0.0
                    ),
                    "billed": float(getattr(boiler, "_burner_heat_rate", 0.0)),
                }
                return None

            boiler.apply_heat_draw = apply_heat_draw
        else:
            # sims/shipped: gas = coil draw + standing loss + tank step change,
            # floored at zero because a gas boiler cannot burn negative fuel.
            orig_rate = boiler.compute_thermal_energy_rate

            def compute_thermal_energy_rate(
                return_water_temp, outside_temp, total_flow_rate
            ):
                billed = orig_rate(return_water_temp, outside_temp, total_flow_rate)
                setpoint = boiler._supply_water_temperature_setpoint
                supply = (
                    setpoint if setpoint > return_water_temp else return_water_temp
                )
                coil = (
                    self._cw_rho_w
                    * total_flow_rate
                    * (supply - return_water_temp)
                )
                seconds = boiler._last_step_duration.total_seconds()
                storage = (
                    self._cw_rho_w
                    * boiler._water_capacity
                    * boiler._step_tank_temperature_change
                    / seconds
                    if seconds > 0
                    else 0.0
                )
                self._gas_calls += 1
                self._gas = {
                    "coil": float(coil),
                    "dissipation": float(
                        boiler.compute_thermal_dissipation_rate(supply, outside_temp)
                    ),
                    "storage": float(storage),
                    "billed": float(billed),
                }
                return billed

            boiler.compute_thermal_energy_rate = compute_thermal_energy_rate

    # -- per-step ---------------------------------------------------------- #
    def begin_step(self) -> None:
        for v_id in self.vav_ids:
            self._reheat[v_id] = 0.0
        self._gas = {}
        self._gas_calls = 0
        self.reward_info = None
        self.reward_response = None

    def sample(self, reward: float) -> None:
        info = self.reward_info
        response = self.reward_response
        if info is None or response is None:
            raise RuntimeError("the reward hook never fired for this step")

        start = self._proto_to_ts(info.start_timestamp)
        end = self._proto_to_ts(info.end_timestamp)
        # The reward path reads outside air at the building's current
        # (already advanced) timestamp, so the logged value is the same one.
        t_out = float(
            self.sim._weather_controller.get_current_temp(self.sim.current_timestamp)
        )
        avg = self.sim.building.get_zone_average_temps()
        ret_temps = _area_weighted_temps(self.sim, self.hvac, self.ahu_zones, avg)

        fan_supply = fan_exhaust = p_chiller = 0.0
        q_coil = q_coil_cf = 0.0
        q_vent_analytic = 0.0
        ahu_row: list[float] = []
        for ahu in self.ahus:
            ahu_id = ahu.device_id()
            t_ret = float(ret_temps[ahu_id])
            flow = float(ahu.air_flow_rate)
            recirc = float(ahu.recirculation)
            t_supp = float(ahu.get_supply_air_temp(t_ret, t_out))
            t_mix = float(ahu.get_mixed_air_temp(t_ret, t_out))
            # Counterfactual: the same unit recirculating everything, which is
            # the limit the outside-air floor holds it away from.
            t_supp_cf = float(ahu.get_supply_air_temp(t_ret, t_ret))
            q_this = float(ahu.compute_thermal_energy_rate(t_ret, t_out))
            q_this_cf = float(ahu.compute_thermal_energy_rate(t_ret, t_ret))
            p_this = float(ahu.compute_cooling_electrical_energy_rate(t_ret, t_out))
            p_supply = float(ahu.compute_intake_fan_energy_rate())
            p_exhaust = float(ahu.compute_exhaust_fan_energy_rate())
            q_vent = (
                (1.0 - recirc) * flow * self._rho_air * self._cp_air * (t_out - t_ret)
            )
            q_coil += q_this
            q_coil_cf += q_this_cf
            p_chiller += p_this
            fan_supply += p_supply
            fan_exhaust += p_exhaust
            q_vent_analytic += q_vent
            ahu_row.extend(
                (
                    flow,
                    recirc,
                    t_ret,
                    t_out,
                    t_mix,
                    t_supp,
                    t_supp_cf,
                    q_this,
                    q_this_cf,
                    p_this,
                    p_supply,
                    p_exhaust,
                    q_vent,
                )
            )

        # Heating counterfactual per VAV: the same coil, same water, same
        # airflow, but air entering at the return temperature instead of the
        # mixture. The difference is the duty the outside air forces.
        q_reheat = q_reheat_cf = 0.0
        coil_capacity = 0.0  # W the air streams could absorb at the water temperature
        vav_row: list[float] = []
        zone_temp = {z: avg.get(z, np.nan) for z in self.zones}
        occ = {
            zid: float(zone.average_occupancy)
            for zid, zone in info.zone_reward_infos.items()
        }
        for v_id in self.vav_ids:
            vav = self.vavs[v_id]
            ahu = self.ahus_by_id[self.vav_ahu[v_id]]
            t_ret = float(ret_temps[ahu.device_id()])
            t_supp = float(ahu.get_supply_air_temp(t_ret, t_out))
            t_supp_cf = float(ahu.get_supply_air_temp(t_ret, t_ret))
            water = self._reheat_water[v_id]
            actual = self._reheat[v_id]
            cf = self._orig_reheat[v_id](t_supp_cf, water)
            q_reheat += actual
            q_reheat_cf += cf
            coil_capacity += (
                float(vav.flow_rate_demand)
                * self._rho_air
                * self._cp_air
                * max(0.0, water - t_supp)
            )
            vav_row.extend(
                (
                    float(vav.flow_rate_demand),
                    float(actual),
                    float(cf),
                    float(vav.damper_setting),
                    float(vav.reheat_valve_setting),
                    float(vav.zone_air_temperature),
                )
            )

        gas = self._gas
        coil_gas = float(gas.get("coil", 0.0))
        standby = float(gas.get("dissipation", 0.0)) + float(gas.get("storage", 0.0))
        billed_gas = float(gas.get("billed", 0.0))
        gas_floor = billed_gas - (coil_gas + standby)

        # Reward units: the exact per-step denominators and prices the reward
        # used, so every component is priced on the same basis.
        prices = self._prices(start, end)

        # The ventilation share of each load is the marginal duty the outside
        # air adds: the same episode with the mixing box recirculating
        # everything. Cooling only exists when the coil actually cooled -- with
        # no cooling there is nothing to attribute, and the counterfactual can
        # otherwise invent some (the supply-air setpoint often sits below the
        # return temperature in winter).
        q_cool = max(0.0, -q_coil)
        q_cool_cf = max(0.0, -q_coil_cf)
        cool_vent_thermal = min(q_cool, max(0.0, q_cool - q_cool_cf))
        cop = float(getattr(self.ahus[0], "_cooling_cop", 1.0)) or 1.0
        cool_vent_elec = cool_vent_thermal / cop
        cool_rest_elec = max(0.0, abs(p_chiller)) - cool_vent_elec
        heat_vent_thermal = min(max(0.0, q_reheat - q_reheat_cf), max(0.0, coil_gas))
        heat_rest = max(0.0, coil_gas) - heat_vent_thermal

        # Standby is everything the boiler billed that the coils did not draw:
        # the tank's standing loss, whatever it charged into storage, and the
        # negative clip. Negative when stored heat, not gas, covered the coils.
        components = {
            "heat_vent": heat_vent_thermal,
            "heat_rest": heat_rest,
            "boiler_standby": billed_gas - coil_gas,
            "cool_vent": cool_vent_elec,
            "cool_rest": cool_rest_elec,
            "fan_supply": fan_supply,
            "fan_exhaust": fan_exhaust,
            "pump": self.hws.compute_pump_power(),
        }
        row = {
            "hour": float(start.hour),
            "weekday": float(start.weekday()),
            "t_out": t_out,
            "occupancy": float(response.total_occupancy),
            "reward": float(reward),
            "npr": float(response.normalized_productivity_regret),
            "ec": float(response.normalized_energy_cost),
            "ce": float(response.normalized_carbon_emission),
            "weighted_energy_cost": float(
                -prices["w_e"] * response.normalized_energy_cost
            ),
            "weighted_carbon": float(
                -prices["w_c"] * response.normalized_carbon_emission
            ),
            "den_energy": float(prices["den_energy"]),
            "den_carbon": float(prices["den_carbon"]),
            "weighted_comfort": float(prices["w_p"] * response.normalized_productivity_regret),
            "p_chiller": abs(p_chiller),
            "p_chiller_vent": cool_vent_elec,
            "q_coil": q_coil,
            "q_coil_cf": q_coil_cf,
            "q_reheat": q_reheat,
            "q_reheat_cf": q_reheat_cf,
            "coil_capacity": coil_capacity,
            "q_vent_analytic": q_vent_analytic,
            "fan_supply": fan_supply,
            "fan_exhaust": fan_exhaust,
            "pump": components["pump"],
            "gas_coil": coil_gas,
            "gas_standby": standby,
            "gas_floor": gas_floor,
            "gas_billed": billed_gas,
            "gas_calls": float(self._gas_calls),
            "water_flow": float(self.hws.total_flow_rate),
            "water_supply": float(self.boiler._current_temperature),
            "water_return": float(self.hws.return_water_temperature_sensor),
        }
        # Every component is priced linearly, so the parts add up to the whole
        # exactly: an electrical component in W, a gas component in W of heat.
        for name in COMPONENTS:
            if name in GAS_COMPONENTS:
                kwargs = {"p_gas": components[name]}
            else:
                kwargs = {"p_elec": components[name]}
            row[f"loss_{name}"] = prices["loss"](**kwargs)
            row[f"energy_{name}"] = prices["energy_part"](**kwargs)
            row[f"carbon_{name}"] = prices["carbon_part"](**kwargs)
        total_elec = (
            abs(fan_supply) + abs(fan_exhaust) + abs(p_chiller) + abs(components["pump"])
        )
        row["loss_reconcile"] = prices["loss"](
            p_elec=total_elec, p_gas=billed_gas
        )
        row["energy_reconcile"] = prices["energy_part"](
            p_elec=total_elec, p_gas=billed_gas
        )
        row["carbon_reconcile"] = prices["carbon_part"](
            p_elec=total_elec, p_gas=billed_gas
        )

        self.rows.append(row)
        self.ahu_rows.append(ahu_row)
        self.vav_rows.append(vav_row)
        self.zone_rows.append(
            [float(zone_temp[z]) for z in self.zones]
            + [float(occ.get(z, 0.0)) for z in self.zones]
        )

    # -- pricing ----------------------------------------------------------- #
    def _prices(self, start, end) -> dict[str, Any]:
        """Returns per-watt price/carbon closures in reward units for one step.

        The reward's own models are queried at exactly 1 W, which makes the
        result linear in the power: ``cost(rate) = cost(1 W) * rate`` for any
        rate of the same sign, and both schedules are hour-of-week functions
        fixed by the step's start time. So a component's reward loss is its
        power times a per-watt constant and the components add up to the fuel
        total with no residual, which is what makes the reconciliation below a
        check on the physics rather than on the accounting.
        """
        rf = self.env.reward_function
        elec = rf._electricity_energy_cost
        gas = rf._natural_gas_energy_cost
        w_e = float(rf.energy_cost_weight)
        w_c = float(rf.carbon_emission_weight)
        den_energy = elec.cost(start, end, rf._max_electricity_rate) + gas.cost(
            start, end, rf._max_natural_gas_rate
        )
        den_carbon = elec.carbon(start, end, rf._max_electricity_rate) + gas.carbon(
            start, end, rf._max_natural_gas_rate
        )
        unit = {
            "elec_cost": elec.cost(start, end, 1.0),
            "elec_carbon": elec.carbon(start, end, 1.0),
            "gas_cost": gas.cost(start, end, 1.0),
            "gas_carbon": gas.carbon(start, end, 1.0),
        }
        scale_e = w_e / den_energy if den_energy > 0.0 else 0.0
        scale_c = w_c / den_carbon if den_carbon > 0.0 else 0.0

        def energy_part(p_elec: float = 0.0, p_gas: float = 0.0) -> float:
            """Reward energy term (<=0) of an electrical and a gas power."""
            cost = unit["elec_cost"] * p_elec + unit["gas_cost"] * p_gas
            return -scale_e * cost

        def carbon_part(p_elec: float = 0.0, p_gas: float = 0.0) -> float:
            """Reward carbon term (<=0) of an electrical and a gas power."""
            carbon = unit["elec_carbon"] * p_elec + unit["gas_carbon"] * p_gas
            return -scale_c * carbon

        def loss(p_elec: float = 0.0, p_gas: float = 0.0) -> float:
            """Reward loss (<=0) of an electrical and a gas power, in reward units."""
            return energy_part(p_elec, p_gas) + carbon_part(p_elec, p_gas)

        return {
            "w_e": w_e,
            "w_c": w_c,
            "w_p": float(rf.productivity_weight),
            "den_energy": den_energy,
            "den_carbon": den_carbon,
            "loss": loss,
            "energy_part": energy_part,
            "carbon_part": carbon_part,
        }

    # -- output ------------------------------------------------------------ #
    def aggregate(self, scenario: str, meta: dict[str, Any]) -> dict[str, Any]:
        keys = [
            k
            for k in self.rows[0]
            if k.startswith(("loss_", "energy_", "carbon_", "energy_reconcile", "carbon_reconcile"))
        ] + [
            "reward",
            "weighted_comfort",
            "weighted_energy_cost",
            "weighted_carbon",
            "q_coil",
            "q_coil_cf",
            "q_reheat",
            "q_reheat_cf",
            "coil_capacity",
            "q_vent_analytic",
            "p_chiller",
            "p_chiller_vent",
            "fan_supply",
            "fan_exhaust",
            "pump",
            "gas_coil",
            "gas_standby",
            "gas_billed",
            "t_out",
            "occupancy",
        ]
        data = {k: np.array([r[k] for r in self.rows], dtype=np.float64) for k in keys}
        buckets = np.array(
            [
                _bucket(int(r["hour"]), int(r["weekday"]), self.morning, self.evening)
                for r in self.rows
            ]
        )

        def by_bucket(values: np.ndarray) -> dict[str, float]:
            out = {}
            for name in BUCKETS:
                mask = buckets == name
                out[name] = float(values[mask].sum())
                out[f"{name}__steps"] = int(mask.sum())
            return out

        per_component = {
            name: {
                "loss_total": float(data[f"loss_{name}"].sum()),
                "energy_total": float(data[f"energy_{name}"].sum()),
                "carbon_total": float(data[f"carbon_{name}"].sum()),
                "loss_by_bucket": by_bucket(data[f"loss_{name}"]),
            }
            for name in COMPONENTS
        }
        total_loss = float(data["loss_reconcile"].sum())
        actual_loss = float(
            (data["weighted_energy_cost"] + data["weighted_carbon"]).sum()
        )
        actual_energy = float(data["weighted_energy_cost"].sum())
        actual_carbon = float(data["weighted_carbon"].sum())
        parts_energy = float(data["energy_reconcile"].sum())
        parts_carbon = float(data["carbon_reconcile"].sum())

        def rel(a: float, b: float) -> float:
            return float(abs(a - b) / abs(b)) if b != 0.0 else 0.0

        # How much of the coil duty the water loop could have carried, and how
        # often the outside air alone exceeded what the coils actually added.
        coil_heat = np.maximum(0.0, data["gas_coil"])
        coil_capacity = np.maximum(0.0, data["coil_capacity"])
        vent_heat = np.maximum(0.0, -data["q_vent_analytic"])
        heating = coil_heat > 0.0
        saturation = (
            float(np.median(coil_heat[heating] / coil_capacity[heating]))
            if (heating & (coil_capacity > 0.0)).any()
            else None
        )
        ratio = (
            vent_heat[heating] / coil_heat[heating]
            if heating.any()
            else np.zeros(0)
        )
        parts = sum(v["loss_total"] for v in per_component.values())
        # Raw powers, so the physical quantities can be quoted in energy units.
        fuel = {
            # Gas, all as watts of heat, summable to joules.
            "gas_billed_sum": float(data["gas_billed"].sum()),
            "gas_coil_sum": float(data["gas_coil"].sum()),
            "gas_standby_sum": float(data["gas_standby"].sum()),
            # Coil duty the outside air forces, and the duty that survives
            # 100% recirculation (both from the counterfactual).
            "reheat_sum": float(data["q_reheat"].sum()),
            "reheat_cf_sum": float(data["q_reheat_cf"].sum()),
            # The analytic ventilation load: (1-r) * airflow * rho * cp * dT.
            "vent_analytic_sum": float(data["q_vent_analytic"].sum()),
            # Chiller (electric), and its ventilation share.
            "chiller_electrical_sum": float(data["p_chiller"].sum()),
            "chiller_vent_sum": float(data["p_chiller_vent"].sum()),
            "coil_thermal_sum": float(data["q_coil"].sum()),
            "coil_cf_thermal_sum": float(data["q_coil_cf"].sum()),
            "fan_supply_sum": float(data["fan_supply"].sum()),
            "fan_exhaust_sum": float(data["fan_exhaust"].sum()),
            "pump_sum": float(data["pump"].sum()),
        }

        by_hour = {}
        for hour in range(24):
            mask = np.array([int(r["hour"]) == hour for r in self.rows])
            by_hour[str(hour)] = {
                name: float(data[f"loss_{name}"][mask].sum())
                for name in COMPONENTS
            }
            by_hour[str(hour)]["total"] = float(data["loss_reconcile"][mask].sum())

        ahu_arr = np.array(self.ahu_rows, dtype=np.float64)
        vav_arr = np.array(self.vav_rows, dtype=np.float64)
        zone_arr = np.array(self.zone_rows, dtype=np.float64)
        n_ahu = len(self.ahus)
        ahu_offset = 13
        ahu_means = {
            ahu.device_id(): {
                "airflow_mean": float(ahu_arr[:, i * ahu_offset + 0].mean()),
                "recirculation_mean": float(ahu_arr[:, i * ahu_offset + 1].mean()),
                "outside_air_fraction_mean": float(
                    1.0 - ahu_arr[:, i * ahu_offset + 1].mean()
                ),
                "return_temp_mean": float(ahu_arr[:, i * ahu_offset + 2].mean()),
                "mixed_temp_mean": float(ahu_arr[:, i * ahu_offset + 4].mean()),
                "supply_temp_mean": float(ahu_arr[:, i * ahu_offset + 5].mean()),
                "coil_thermal_mean": float(ahu_arr[:, i * ahu_offset + 7].mean()),
                "chiller_electrical_mean": float(
                    ahu_arr[:, i * ahu_offset + 9].mean()
                ),
                "fan_supply_mean": float(ahu_arr[:, i * ahu_offset + 10].mean()),
                "fan_exhaust_mean": float(ahu_arr[:, i * ahu_offset + 11].mean()),
            }
            for i, ahu in enumerate(self.ahus)
        }

        return {
            "scenario": scenario,
            "sim": getattr(self, "sim_name", None),
            "steps": len(self.rows),
            "num_zones": len(self.zones),
            "num_ahus": n_ahu,
            "num_vavs": len(self.vav_ids),
            "first_local_timestamp": str(self.rows[0]["hour"]),
            "total_reward": float(data["reward"].sum()),
            "mean_per_step_reward": float(data["reward"].mean()),
            "loss_comfort": float(data["weighted_comfort"].sum()),
            "loss_energy": float(data["weighted_energy_cost"].sum()),
            "loss_carbon": float(data["weighted_carbon"].sum()),
            "reconciliation": {
                "loss_from_components": parts,
                "loss_from_reward_terms": actual_loss,
                "loss_recomputed_from_powers": total_loss,
                "energy_from_powers": parts_energy,
                "energy_from_reward": actual_energy,
                "carbon_from_powers": parts_carbon,
                "carbon_from_reward": actual_carbon,
                # Re-deriving cost and carbon from the logged powers must give
                # back the reward's own two terms step by step; both are exact
                # up to float summation order.
                "rel_error_vs_reward_terms": rel(parts, actual_loss),
                "rel_error_energy": rel(parts_energy, actual_energy),
                "rel_error_carbon": rel(parts_carbon, actual_carbon),
                "rel_error_combined_powers_vs_reward": rel(total_loss, actual_loss),
            },
            "diagnostics": {
                # Median of (coil duty billed) / (heat the air streams could have
                # carried at the water temperature). Well below 1 means the hot
                # water side, not the air side, sets the heating.
                "coil_saturation_median": saturation,
                "heating_steps": int(heating.sum()),
                "vent_exceeds_coil_share": (
                    float((ratio > 1.0).mean()) if ratio.size else None
                ),
                "vent_to_coil_median": (
                    float(np.median(ratio)) if ratio.size else None
                ),
            },
            "fuel_totals": fuel,
            "components": per_component,
            "by_bucket_total": by_bucket(data["loss_reconcile"]),
            "by_hour": by_hour,
            "ahu": ahu_means,
            "vav_columns": [
                "flow",
                "reheat",
                "reheat_cf",
                "damper",
                "valve",
                "zone_temp",
            ],
            "vav_means": {
                v_id: {
                    "flow_mean": float(vav_arr[:, i * 6 + 0].mean()),
                    "reheat_mean": float(vav_arr[:, i * 6 + 1].mean()),
                    "damper_mean": float(vav_arr[:, i * 6 + 3].mean()),
                    "valve_mean": float(vav_arr[:, i * 6 + 4].mean()),
                    "zone_temp_mean": float(vav_arr[:, i * 6 + 5].mean()),
                }
                for i, v_id in enumerate(self.vav_ids)
            },
            "zone_names": self.zones,
            "occupancy_sum": float(data["occupancy"].sum()),
            "outside_temp_mean": float(data["t_out"].mean()),
            "outside_temp_min": float(data["t_out"].min()),
            "outside_temp_max": float(data["t_out"].max()),
        }

    # -- arrays ------------------------------------------------------------ #
    def arrays(self) -> dict[str, np.ndarray]:
        rows = self.rows
        scalar_keys = [k for k in rows[0]]
        out = {k: np.array([r[k] for r in rows], dtype=np.float32) for k in scalar_keys}
        out["ahu"] = np.array(self.ahu_rows, dtype=np.float32).reshape(
            len(rows), len(self.ahus), -1
        )
        out["vav"] = np.array(self.vav_rows, dtype=np.float32).reshape(
            len(rows), len(self.vav_ids), -1
        )
        out["zone"] = np.array(self.zone_rows, dtype=np.float32).reshape(
            len(rows), 2, len(self.zones)
        )
        return out


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    started_wall = time.time()
    started_cpu = time.process_time()
    simload.use_sim(task["sim"])
    from smart_buildings import evaluate  # noqa: PLC0415

    policy_cls, policy_label = evalrun._resolve_policy(task["policy"], evaluate)
    config_path = Path(task["config_path"])
    env = evalrun._load_environment(evaluate, config_path, {})
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = evaluate.create_building_meta_data(env, config_path.parent)
    action_spec = env.action_spec()
    if task["params"]:
        policy_cls.PARAMS = {**(getattr(policy_cls, "PARAMS", None) or {}), **task["params"]}
    policy = evaluate.initialize_policy(policy_cls, meta, action_spec)

    recorder = FlowRecorder(env, meta)
    recorder.sim_name = task["sim"]
    obs, _ = env.reset()
    rewards = []
    for _ in range(int(task["steps"])):
        action = evaluate.validate_action(
            evaluate.get_policy_action(policy, obs), action_spec
        )
        recorder.begin_step()
        obs, reward, terminated, truncated, _ = env.step(action)
        recorder.sample(float(reward))
        rewards.append(float(reward))
        if terminated or truncated:
            break

    record = recorder.aggregate(task["scenario"], meta)
    record.update(
        {
            "policy": policy_label,
            "tag": task["tag"],
            "requested_steps": int(task["steps"]),
            "wall_seconds": time.time() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "config_path": str(config_path.resolve()),
        }
    )
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
        return {"scenario": task["scenario"], "error": traceback.format_exc()}


def run(
    policy: str = "policies.ctrl_w:CtrlW",
    scenarios_: list[str] | None = None,
    steps: int = DEFAULT_STEPS,
    sim: str = "shipped",
    workers: int = 2,
    tag: str = "ef",
    out: str | Path | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Runs an instrumented episode over every scenario and returns a summary."""
    if sim not in simload.SIM_NAMES:
        raise ValueError(f"unknown sim {sim!r}")
    items = list(scenarios_) or ["building_train_1/fairbanks_1"]
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
        }
        for scenario, path in pairs
    ]
    print(
        f"[energyflow] tag={tag} sim={sim} policy={policy} "
        f"scenarios={len(tasks)} steps={steps} workers={workers} -> {out_dir}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    started = time.time()
    if workers <= 1:
        for task in tasks:
            result = _worker(task)
            results.append(result)
            print(f"[energyflow] {result.get('scenario')} done", flush=True)
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=workers, maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(_worker, tasks, chunksize=1):
                results.append(result)
                print(
                    f"[energyflow] {result.get('scenario')} "
                    f"{'FAILED' if 'error' in result else 'done'}",
                    flush=True,
                )
    summary = {
        "tag": tag,
        "policy": policy,
        "sim": sim,
        "steps": int(steps),
        "out_dir": str(out_dir),
        "wall_seconds": time.time() - started,
        "scenarios": [
            {
                "scenario": r.get("scenario"),
                "error": r.get("error"),
                "mean_per_step_reward": r.get("mean_per_step_reward"),
                "total_reward": r.get("total_reward"),
                "reconciliation": r.get("reconciliation"),
            }
            for r in results
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    bad = [r for r in results if "error" in r]
    print(
        f"[energyflow] {len(results) - len(bad)}/{len(results)} ok, "
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
    p_run = sub.add_parser("run", help="instrumented rollout")
    p_run.add_argument("--policy", default="policies.ctrl_w:CtrlW")
    p_run.add_argument("--scenario", nargs="+", required=True)
    p_run.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_run.add_argument("--sim", default="shipped", choices=list(simload.SIM_NAMES))
    p_run.add_argument("--workers", type=int, default=2)
    p_run.add_argument("--tag", default="ef")
    p_run.add_argument("--out", default=None)
    p_run.add_argument("--params", default=None)
    args = parser.parse_args(argv)
    if args.command == "run":
        params = json.loads(args.params) if args.params else None
        run(
            policy=args.policy,
            scenarios_=args.scenario,
            steps=args.steps,
            sim=args.sim,
            workers=args.workers,
            tag=args.tag,
            out=args.out,
            params=params,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
