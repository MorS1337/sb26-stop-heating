"""Reference policy: malyshevdanil's "Smart Buildings: per-zone PI + AHU cooling hold" (zt-best).

Notebook (public, best-scoring public controller of the competition):
    https://www.kaggle.com/code/malyshevdanil/smart-buildings-probe-zt-best
Author: malyshevdanil. Notebook licence: Apache-2.0 (the notebook builds on the
organizer `PIDWaterResetPolicy`, also Apache-2.0).

This module is a faithful re-implementation of the notebook's `SubmissionPolicy`
(= `APResearchPolicy` with the constants below), ported so that it runs in our
runner without importing `smart_buildings`:

  * `_PIDPolicy` ... `_PIDWaterResetPolicy` are a verbatim copy of the
    organizer chain `smart_buildings/baselines/pid.py::PIDPolicy` and
    `smart_buildings/baselines/pid_plus.py::PIDWorstZonePolicy` ->
    `PIDEconomizerPolicy` -> `PIDProportionalReheatPolicy` ->
    `PIDWaterResetPolicy` from the starter kit
    (https://github.com/smart-buildings-2026/starter-kit, Apache-2.0).
    Only the imports and the module docstrings were changed; every constant and
    every line of logic is the shipped organizer code.
  * `_BaselinePolicy` ... `RefZtBest` are the notebook's controller chain, in
    the notebook's own inheritance order, with the notebook's constants:
    `KP=0.1, KI=0.005, HOLD_STEPS=288, NIGHT_COOL_CEILING=False,
    ZONE_SHIFT_K=0.6, DEMAND_QUANTILE=0.5, ZONE_EMA_STEPS=144,
    WEEKEND_EXTRA_DROP_K=6.0, PRESSURE_MIN_ALL=True`.

Notebook-logged reference numbers (Kaggle gateway run of the competition
simulator, `building_train_1/atlanta_2`, 8,640 steps): mean reward -0.00556,
public LB 0.01420.
"""

import re
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Vendored organizer baselines (starter-kit `smart_buildings/baselines/pid.py`
# and `pid_plus.py`, Apache-2.0). Copied so this module is self-contained.
# --------------------------------------------------------------------------- #

# Native ranges from the action normalizer in every shipped config.
SUPPLY_AIR_MIN_K = 285.0
SUPPLY_AIR_MAX_K = 305.0
STATIC_PRESSURE_MAX_PA = 2000.0
# The simulator holds this floor while a handler runs, so a lower command
# still costs this much fan.
MIN_STATIC_PRESSURE_PA = 200.0
# Kelvin of outside-air advantage needed per pascal of fan static pressure
# before drawing outside air is cheaper than recirculating:
#     COP / (fan_efficiency * air_density * air_heat_capacity)
#   = 3.5 / (0.9 * 1.2 * 1006)  =  0.00322 K/Pa
ECON_K_PER_PA = 0.00322


class _PIDPolicy:
    """Distributed per-AHU PID controller with proportional fan pressure."""

    def __init__(
        self,
        building_meta_data,
        action_spec,
        kp: float = 0.5,
        ki: float = 0.02,
        kd: float = 0.1,
        integral_limit: float = 10.0,
        # Excursion in K at which a zone's damper is commanded fully open.
        zone_damper_full_k: float = 2.0,
    ):
        self.metadata = building_meta_data
        self.action_spec = action_spec

        self.comfort_temp_window = building_meta_data["comfort_temp_window"]
        self.eco_temp_window = building_meta_data["eco_temp_window"]
        self.morning_start_hour = building_meta_data["morning_start_hour"]
        self.evening_start_hour = building_meta_data["evening_start_hour"]

        self.n_discrete = building_meta_data["n_discrete_action"]
        self.n_continuous = building_meta_data["n_continuous_action"]
        self.temp_indices = list(building_meta_data["temp_indices"])
        self.pressure_indices = list(building_meta_data["pressure_indices"])
        self.water_indices = list(building_meta_data["water_indices"])
        self.recirculation_indices = list(building_meta_data["recirculation_indices"])
        self.zone_action_indices = building_meta_data["zone_action_indices"]
        self.field_names = building_meta_data["observation_field_names"]
        self.ahu_zone_assignments = building_meta_data["ahu_zone_assignments"]

        self.observation_normalization_metadata = building_meta_data[
            "observation_normalization_metadata"
        ]
        self.zone_air_temperature_sensor_mean = self.observation_normalization_metadata[
            "zone_air_temperature_sensor"
        ]["sample_mean"]
        self.zone_air_temperature_sensor_std = self.observation_normalization_metadata[
            "zone_air_temperature_sensor"
        ]["sample_std"]

        self.hod_sin_index = self.field_names.index("hod_sin_000")
        self.hod_cos_index = self.field_names.index("hod_cos_000")
        self.dow_sin_index = self.field_names.index("dow_sin_000")
        self.dow_cos_index = self.field_names.index("dow_cos_000")

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.zone_damper_full_k = zone_damper_full_k

        self._build_ahu_mapping()
        self.reset()

    def _build_ahu_mapping(self) -> None:
        zone_temp_field_index = {
            name: i
            for i, name in enumerate(self.field_names)
            if "zone_air_temperature" in name
        }

        ahu_ids = sorted(self.ahu_zone_assignments.keys(), key=self._numeric_suffix)
        self.n_ahu = len(self.temp_indices)
        if len(ahu_ids) != self.n_ahu:
            self.n_ahu = min(self.n_ahu, len(ahu_ids))
            ahu_ids = ahu_ids[: self.n_ahu]

        self.ahu_controllers: list[dict[str, Any]] = []
        for i, ahu_id in enumerate(ahu_ids):
            rooms = self.ahu_zone_assignments[ahu_id]
            # (vav id, observation index, action indices) for each zone this AHU
            # serves that the building actually reports.
            zones: list[tuple[str, int, dict[str, int]]] = []
            for room in rooms:
                vav_id = room.replace("room_", "VAV_")
                vav_field = vav_id + "_zone_air_temperature_sensor"
                if vav_field in zone_temp_field_index:
                    zones.append(
                        (
                            vav_id,
                            zone_temp_field_index[vav_field],
                            self.zone_action_indices.get(vav_id, {}),
                        )
                    )

            self.ahu_controllers.append(
                {
                    "ahu_id": ahu_id,
                    "temp_idx": self.temp_indices[i],
                    "pressure_idx": self.pressure_indices[i]
                    if i < len(self.pressure_indices)
                    else None,
                    "run_discrete_idx": i,
                    "zones": zones,
                    "zone_obs_indices": [obs_idx for _, obs_idx, _ in zones],
                }
            )

        self.water_temp_idx = self.water_indices[0] if self.water_indices else None
        self.water_pressure_idx = (
            self.pressure_indices[-1] if len(self.pressure_indices) > self.n_ahu else None
        )
        self.water_discrete_idx = self.n_discrete - 1

    @staticmethod
    def _numeric_suffix(name: str) -> int:
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else 0

    @property
    def name(self) -> str:
        return "pid"

    def reset(self) -> None:
        self.pid_state = {i: {"integral": 0.0, "prev_error": 0.0} for i in range(self.n_ahu)}

    def _cyclic_feature_to_index(self, sin_value: float, cos_value: float, period: int) -> int:
        angle = np.arctan2(sin_value, cos_value)
        if angle < 0:
            angle += 2 * np.pi
        return int(np.floor(period * angle / (2 * np.pi)))

    def _get_hour_and_day(self, obs: np.ndarray):
        hour = self._cyclic_feature_to_index(obs[self.hod_sin_index], obs[self.hod_cos_index], period=24)
        day = self._cyclic_feature_to_index(obs[self.dow_sin_index], obs[self.dow_cos_index], period=7)
        return hour, day

    def _denormalize_zone_temp(self, normalized_value: float) -> float:
        return normalized_value * self.zone_air_temperature_sensor_std + self.zone_air_temperature_sensor_mean

    def _deadband_error(self, avg_temp: float, low: float, high: float) -> float:
        if avg_temp < low:
            return low - avg_temp
        if avg_temp > high:
            return high - avg_temp
        return 0.0

    def _command_zones(
        self,
        continuous: np.ndarray,
        zones: list[tuple[str, int, dict[str, int]]],
        obs: np.ndarray,
        low: float,
        high: float,
    ) -> None:
        """Sets each VAV's damper and reheat valve from its own zone temperature."""
        for _, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            zone_error = self._deadband_error(
                self._denormalize_zone_temp(obs[obs_idx]), low, high
            )
            damper = min(1.0, abs(zone_error) / self.zone_damper_full_k)

            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 1.0 if zone_error > 0 else -1.0

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        hour, day = self._get_hour_and_day(obs)
        is_weekday = day < 5
        is_work_hour = self.morning_start_hour <= hour < self.evening_start_hour

        if is_weekday and is_work_hour:
            low_temp, high_temp = self.comfort_temp_window
        else:
            low_temp, high_temp = self.eco_temp_window

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)

        any_heating = False
        any_active = False

        for i, ctrl in enumerate(self.ahu_controllers):
            zone_obs_indices = ctrl["zone_obs_indices"]
            if not zone_obs_indices:
                continue

            normalized_temps = obs[zone_obs_indices]
            avg_temp = float(np.mean([self._denormalize_zone_temp(v) for v in normalized_temps]))

            self._command_zones(continuous, ctrl["zones"], obs, low_temp, high_temp)

            error = self._deadband_error(avg_temp, low_temp, high_temp)

            state = self.pid_state[i]
            if error == 0.0:
                # Inside the band: idle, same as bang_bang.
                state["integral"] *= 0.5
                state["prev_error"] = 0.0
                continuous[ctrl["temp_idx"]] = -1.0
                if ctrl["pressure_idx"] is not None:
                    continuous[ctrl["pressure_idx"]] = -1.0
                discrete[ctrl["run_discrete_idx"]] = 0
                continue

            state["integral"] = float(
                np.clip(state["integral"] + error, -self.integral_limit, self.integral_limit)
            )
            derivative = error - state["prev_error"]
            state["prev_error"] = error

            u = self.kp * error + self.ki * state["integral"] + self.kd * derivative
            u = float(np.clip(u, -1.0, 1.0))

            # Pressure scales with control effort: |u|=0 -> pressure=-1 (min),
            # |u|=1 -> pressure=+1 (max).
            pressure = float(np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0))

            continuous[ctrl["temp_idx"]] = u
            if ctrl["pressure_idx"] is not None:
                continuous[ctrl["pressure_idx"]] = pressure
            discrete[ctrl["run_discrete_idx"]] = 1

            any_active = True
            if error > 0:
                any_heating = True

        for i in self.recirculation_indices:
            continuous[i] = 1.0

        if self.water_temp_idx is not None:
            continuous[self.water_temp_idx] = 1.0 if any_heating else -1.0
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = 1.0 if any_active else -1.0
        if 0 <= self.water_discrete_idx < self.n_discrete:
            discrete[self.water_discrete_idx] = 1 if any_active else 0

        return {"continuous_action": continuous, "discrete_action": discrete}


class _PIDWorstZonePolicy(_PIDPolicy):
    """`pid`, but the AHU loop tracks its worst zone rather than the average."""

    USE_SENSOR_BAND = False

    @property
    def name(self) -> str:
        return "pid_worst"

    def _zone_band(self, obs: np.ndarray, vav_id: str, low: float, high: float):
        """This zone's band. Falls back to the schedule band when unavailable."""
        if not self.USE_SENSOR_BAND:
            return low, high
        idx = self._sensor_band_indices.get(vav_id)
        if idx is None:
            return low, high
        lo_idx, hi_idx = idx
        lo = float(obs[lo_idx]) if lo_idx is not None else low
        hi = float(obs[hi_idx]) if hi_idx is not None else high
        # A band that came back inverted or empty is not usable.
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return low, high
        return lo, hi

    def _build_sensor_band_indices(self) -> None:
        """Index of each zone's own heating and cooling setpoint sensor.

        These are raw kelvin, unlike the zone temperature, which is normalized.
        """
        self._sensor_band_indices: dict[str, tuple[int | None, int | None]] = {}
        names = list(self.field_names)

        def find(field: str):
            try:
                return names.index(field)
            except ValueError:
                return None

        for ctrl in self.ahu_controllers:
            for vav_id, _, _ in ctrl["zones"]:
                self._sensor_band_indices[vav_id] = (
                    find(f"{vav_id}_zone_air_heating_temperature_setpoint_sensor"),
                    find(f"{vav_id}_zone_air_cooling_temperature_setpoint_sensor"),
                )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_sensor_band_indices()
        self._attach_recirculation_and_weather()

    def _attach_recirculation_and_weather(self) -> None:
        """`pid` builds no recirculation or outside-air entry; add them."""
        names = list(self.field_names)
        for i, ctrl in enumerate(self.ahu_controllers):
            ctrl["recirc_idx"] = (
                self.recirculation_indices[i]
                if i < len(self.recirculation_indices)
                else None
            )
            field = f"ahs_{ctrl['ahu_id']}_outside_air_temperature_sensor"
            ctrl["oat_idx"] = names.index(field) if field in names else None

    def _worst_zone_error(self, obs: np.ndarray, ctrl, low: float, high: float) -> float:
        """Signed deadband error of the zone furthest outside its band."""
        worst = 0.0
        for vav_id, obs_idx, _ in ctrl["zones"]:
            lo, hi = self._zone_band(obs, vav_id, low, high)
            err = self._deadband_error(
                self._denormalize_zone_temp(obs[obs_idx]), lo, hi
            )
            if abs(err) > abs(worst):
                worst = err
        return worst

    def _command_zones(self, continuous, zones, obs, low, high) -> None:
        """Same as `pid`, except each zone is judged against its own band."""
        if not self.USE_SENSOR_BAND:
            return super()._command_zones(continuous, zones, obs, low, high)
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            lo, hi = self._zone_band(obs, vav_id, low, high)
            zone_error = self._deadband_error(
                self._denormalize_zone_temp(obs[obs_idx]), lo, hi
            )
            damper = min(1.0, abs(zone_error) / self.zone_damper_full_k)
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 1.0 if zone_error > 0 else -1.0

    # `pid.get_action` reduces the AHU's zones with np.mean. Rather than copy
    # the whole method to change one line, override the reduction it uses.
    def _ahu_error(self, obs: np.ndarray, ctrl, low: float, high: float) -> float:
        return self._worst_zone_error(obs, ctrl, low, high)

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        hour, day = self._get_hour_and_day(obs)
        is_weekday = day < 5
        is_work_hour = self.morning_start_hour <= hour < self.evening_start_hour
        if is_weekday and is_work_hour:
            low_temp, high_temp = self.comfort_temp_window
        else:
            low_temp, high_temp = self.eco_temp_window

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)
        any_heating = False
        any_active = False
        self._last_efforts: list[float] = []

        # Default every handler to the ventilation cap, then let the loop below
        # overwrite the ones it decides about.
        for idx in self.recirculation_indices:
            continuous[idx] = 1.0

        for i, ctrl in enumerate(self.ahu_controllers):
            if not ctrl["zone_obs_indices"]:
                continue

            self._command_zones(continuous, ctrl["zones"], obs, low_temp, high_temp)
            error = self._ahu_error(obs, ctrl, low_temp, high_temp)

            state = self.pid_state[i]
            if error == 0.0:
                state["integral"] *= 0.5
                state["prev_error"] = 0.0
                continuous[ctrl["temp_idx"]] = -1.0
                if ctrl["pressure_idx"] is not None:
                    continuous[ctrl["pressure_idx"]] = -1.0
                if ctrl.get("recirc_idx") is not None:
                    continuous[ctrl["recirc_idx"]] = 1.0
                discrete[ctrl["run_discrete_idx"]] = 0
                continue

            state["integral"] = float(np.clip(
                state["integral"] + error, -self.integral_limit, self.integral_limit))
            derivative = error - state["prev_error"]
            state["prev_error"] = error

            u = float(np.clip(
                self.kp * error + self.ki * state["integral"] + self.kd * derivative,
                -1.0, 1.0))

            continuous[ctrl["temp_idx"]] = u
            # Recorded for the hot water reset in PIDWaterResetPolicy.
            self._last_efforts.append(u)
            if ctrl["pressure_idx"] is not None:
                continuous[ctrl["pressure_idx"]] = float(
                    np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0))
            discrete[ctrl["run_discrete_idx"]] = 1

            if ctrl.get("recirc_idx") is not None:
                continuous[ctrl["recirc_idx"]] = self._recirculation(
                    obs, ctrl, u, low_temp, high_temp)

            any_active = True
            if error > 0:
                any_heating = True

        if self.water_temp_idx is not None:
            continuous[self.water_temp_idx] = 1.0 if any_heating else -1.0
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = 1.0 if any_active else -1.0
        if 0 <= self.water_discrete_idx < self.n_discrete:
            discrete[self.water_discrete_idx] = 1 if any_active else 0

        return {"continuous_action": continuous, "discrete_action": discrete}

    def _recirculation(self, obs, ctrl, u: float, low: float, high: float) -> float:
        """Hold the cap. Subclasses that economize override this."""
        return 1.0


class _PIDEconomizerPolicy(_PIDWorstZonePolicy):
    """`pid` plus a proportional economizer. Average-zone loop, as `pid` has."""

    USE_WORST_ZONE = False

    @property
    def name(self) -> str:
        return "pid_econ"

    def _ahu_error(self, obs: np.ndarray, ctrl, low: float, high: float) -> float:
        if self.USE_WORST_ZONE:
            return self._worst_zone_error(obs, ctrl, low, high)
        temps = [self._denormalize_zone_temp(obs[j]) for j in ctrl["zone_obs_indices"]]
        return self._deadband_error(float(np.mean(temps)), low, high)

    def _recirculation(self, obs, ctrl, u: float, low: float, high: float) -> float:
        oat_idx = ctrl.get("oat_idx")
        # Heating, or no weather reading: hold the heat in.
        if u >= 0.0 or oat_idx is None:
            return 1.0

        outside = float(obs[oat_idx])
        inside = float(np.mean(
            [self._denormalize_zone_temp(obs[j]) for j in ctrl["zone_obs_indices"]]))
        if not np.isfinite(outside) or outside >= inside:
            return 1.0

        pressure_pa = (self._commanded_pressure(u) + 1.0) * 0.5 * STATIC_PRESSURE_MAX_PA
        pressure_pa = max(pressure_pa, MIN_STATIC_PRESSURE_PA)
        breakeven_k = pressure_pa * ECON_K_PER_PA
        if (inside - outside) <= breakeven_k:
            return 1.0

        target = SUPPLY_AIR_MIN_K + (u + 1.0) * 0.5 * (SUPPLY_AIR_MAX_K - SUPPLY_AIR_MIN_K)
        # Never mix below the band's lower edge.
        target = max(target, low)
        if inside <= target:
            return 1.0

        f = (inside - target) / (inside - outside)
        recirculation = float(np.clip(1.0 - f, 0.0, 1.0))
        return 2.0 * recirculation - 1.0

    @staticmethod
    def _commanded_pressure(u: float) -> float:
        """The normalized pressure this controller is about to write."""
        return float(np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0))


class _PIDProportionalReheatPolicy(_PIDEconomizerPolicy):
    """`pid` with a modulating reheat valve instead of an on/off one."""

    # Shortfall in K at which the valve is fully open.
    REHEAT_FULL_K = 2.0

    @property
    def name(self) -> str:
        return "pid_prop"

    def _command_zones(self, continuous, zones, obs, low, high) -> None:
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            lo, hi = self._zone_band(obs, vav_id, low, high)
            zone_error = self._deadband_error(
                self._denormalize_zone_temp(obs[obs_idx]), lo, hi
            )
            damper = min(1.0, abs(zone_error) / self.zone_damper_full_k)
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                # zone_error > 0 means below the heating setpoint.
                opening = min(1.0, max(0.0, zone_error) / self.REHEAT_FULL_K)
                continuous[reheat_idx] = 2.0 * opening - 1.0


class _PIDWaterResetPolicy(_PIDProportionalReheatPolicy):
    """Proportional reheat valve AND the hot water temperature reset on demand."""

    # Proportional band: mean zone shortfall, in K, that alone commands full
    # water.
    RESET_FULL_K = 4.0
    # Weight on the air-side saturation term.
    RESET_EFFORT_WEIGHT = 1.0
    # Which handler to believe when they disagree, as a quantile over the
    # per-handler heating efforts.
    RESET_EFFORT_QUANTILE = 0.25
    # Integral gain and decay. Both left in place, and the gain defaults to
    # zero, because the integrator was built, measured and rejected.
    RESET_DEADBAND_K = 0.25
    RESET_INTEGRAL_GAIN = 0.0
    RESET_INTEGRAL_DECAY = 0.99

    @property
    def name(self) -> str:
        return "pid_water"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._water_integral = 0.0

    def _heating_shortfall(self, obs, low: float, high: float) -> float:
        """Building-wide heating demand, as the MEAN zone shortfall."""
        shortfalls = []
        for ctrl in self.ahu_controllers:
            for vav_id, obs_idx, _ in ctrl["zones"]:
                lo, _hi = self._zone_band(obs, vav_id, low, high)
                shortfalls.append(
                    max(0.0, lo - self._denormalize_zone_temp(obs[obs_idx]))
                )
        return float(np.mean(shortfalls)) if shortfalls else 0.0

    def get_action(self, obs):
        action = super().get_action(obs)
        if self.water_temp_idx is None:
            return action

        hour, day = self._get_hour_and_day(obs)
        is_work = day < 5 and self.morning_start_hour <= hour < self.evening_start_hour
        low, high = self.comfort_temp_window if is_work else self.eco_temp_window
        shortfall = self._heating_shortfall(obs, low, high)

        if shortfall > self.RESET_DEADBAND_K:
            self._water_integral += self.RESET_INTEGRAL_GAIN * (
                shortfall - self.RESET_DEADBAND_K)
        else:
            self._water_integral *= self.RESET_INTEGRAL_DECAY
        self._water_integral = float(np.clip(self._water_integral, 0.0, 1.0))

        cont = action["continuous_action"]
        if cont[self.water_temp_idx] <= 0.0:
            return action  # not heating; leave the loop cold

        frac = self._water_fraction(shortfall)
        cont[self.water_temp_idx] = np.float32(2.0 * frac - 1.0)
        return action

    def _water_fraction(self, shortfall: float) -> float:
        """How hot to run the loop, as a fraction of the 310-350 K range."""
        demand = shortfall / self.RESET_FULL_K
        efforts = getattr(self, "_last_efforts", None)
        saturation = 0.0
        if efforts and self.RESET_EFFORT_WEIGHT:
            heating = [max(0.0, u) for u in efforts]
            saturation = self.RESET_EFFORT_WEIGHT * float(
                np.quantile(heating, self.RESET_EFFORT_QUANTILE))
        return float(np.clip(max(demand, saturation) + self._water_integral,
                             0.0, 1.0))


# --------------------------------------------------------------------------- #
# The notebook's controller chain, in the notebook's inheritance order
# (malyshevdanil/smart-buildings-probe-zt-best, Apache-2.0).
# --------------------------------------------------------------------------- #
class _BaselinePolicy(_PIDWaterResetPolicy):
    """First baseline: average-zone control, sensor comfort bands, decoded weather."""

    USE_SENSOR_BAND = True
    # Worst-zone aggregation increased loss in both full-length pilot scenarios.
    USE_WORST_ZONE = False

    @property
    def name(self):
        return "transfer"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        stats = self.observation_normalization_metadata.get(
            "outside_air_temperature_sensor", {}
        )
        self.outside_mean = float(stats.get("sample_mean", 0.0))
        self.outside_std = float(stats.get("sample_std", 1.0))

    def _recirculation(self, obs, ctrl, u, low, high):
        # The parent's economizer expects kelvin; the observation is standardized.
        idx = ctrl.get("oat_idx")
        if idx is None:
            return 1.0
        decoded = obs.copy()
        decoded[idx] = obs[idx] * self.outside_std + self.outside_mean
        return super()._recirculation(decoded, ctrl, u, low, high)

    def get_action(self, obs):
        action = super().get_action(obs)
        # Enforce the gateway's dtype, shape and bounds contract.
        for key, dtype in (("continuous_action", np.float32),
                           ("discrete_action", np.int32)):
            space = self.action_spec[key]
            action[key] = np.clip(action[key], space.low, space.high).astype(dtype)
        return action


class _CoolingHoldPolicy(_BaselinePolicy):
    """The first baseline plus a cooling-run hold for air handlers."""

    HOLD_STEPS = 288
    # Outside work hours use (eco low, comfort high): no overnight drift up to
    # the eco ceiling in hot climates, full heating setback kept in cold ones.
    NIGHT_COOL_CEILING = False

    @property
    def name(self):
        return f"cooling_hold_{self.HOLD_STEPS}"

    def _base_action(self, obs):
        if not self.NIGHT_COOL_CEILING:
            return super().get_action(obs)
        hour, day = self._get_hour_and_day(obs)
        if day < 5 and self.morning_start_hour <= hour < self.evening_start_hour:
            return super().get_action(obs)
        saved = self.eco_temp_window
        self.eco_temp_window = (saved[0], self.comfort_temp_window[1])
        try:
            return super().get_action(obs)
        finally:
            self.eco_temp_window = saved

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._on_steps = [0] * len(self.ahu_controllers)
        self._last_on = [None] * len(self.ahu_controllers)

    def get_action(self, obs):
        action = self._base_action(obs)
        cont, disc = action["continuous_action"], action["discrete_action"]
        for i, ctrl in enumerate(self.ahu_controllers):
            if not ctrl["zone_obs_indices"]:
                continue
            run = ctrl["run_discrete_idx"]
            # Supply temperature first: its sign tells cooling from heating.
            idx = [j for j in (ctrl["temp_idx"], ctrl["pressure_idx"], ctrl.get("recirc_idx"))
                   if j is not None]
            if (not disc[run] and self._on_steps[i] > 0
                    and self._on_steps[i] < self.HOLD_STEPS and self._last_on[i][0] < 0):
                disc[run] = 1
                cont[idx] = self._last_on[i]
            if disc[run]:
                self._on_steps[i] += 1
                self._last_on[i] = cont[idx].copy()
            else:
                self._on_steps[i] = 0
        return action


class _ZonePIPolicy(_CoolingHoldPolicy):
    """v5 controller with per-zone PI control to the band centre in work hours."""

    HOLD_STEPS = 288
    NIGHT_COOL_CEILING = True
    KP = 0.2
    KI = 0.02
    PRESTART_H = 1
    # Hours before the end of the work day at which the PI loops stop (coast).
    EVENING_STOP_H = 0
    FORCED_SAT = -0.7       # normalized supply temp for a forced-on AHU (~15 C)
    FORCED_PRESSURE = -0.6  # normalized static pressure for a forced-on AHU
    KEY = "zone_pi"

    @property
    def name(self):
        return self.KEY

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._zone_integral = {}
        self._zone_u = {}
        self._pi_active = False

    def _pi_hours(self, obs):
        hour, day = self._get_hour_and_day(obs)
        return (day < 5 and self.morning_start_hour - self.PRESTART_H <= hour
                < self.evening_start_hour - self.EVENING_STOP_H)

    def _command_zones(self, continuous, zones, obs, low, high):
        if not self._pi_active:
            for vav_id, _, _ in zones:
                self._zone_integral[vav_id] = 0.0
            return super()._command_zones(continuous, zones, obs, low, high)
        lo_c, hi_c = self.comfort_temp_window
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            lo, hi = self._zone_band(obs, vav_id, lo_c, hi_c)
            # Before occupancy the sensor band is still the eco band; aim at comfort.
            if hi - lo > (hi_c - lo_c) + 1e-6:
                lo, hi = lo_c, hi_c
            self._cur_vav = vav_id
            err = self._denormalize_zone_temp(obs[obs_idx]) - self._zone_target(obs, lo, hi)
            integral = self._zone_integral.get(vav_id, 0.0)
            u_raw = self.KP * err + self.KI * integral
            # Conditional integration: stop winding up while saturated.
            integral *= 1.0 - self.INTEGRAL_LEAK
            if abs(u_raw) < 1.0 or np.sign(err) != np.sign(u_raw):
                limit = self.INTEGRAL_LIMIT_FRAC / self.KI
                integral = float(np.clip(integral + err, -limit, limit))
            self._zone_integral[vav_id] = integral
            u = float(np.clip(self.KP * err + self.KI * integral, -1.0, 1.0))
            if u < 0:
                u *= self.HEAT_U_SCALE
            self._max_heat = max(self._max_heat, -u)
            self._zone_u[vav_id] = u
            gain = self.HEAT_DAMPER_GAIN if u < 0 else 1.0
            damper = self.MIN_DAMPER + (1.0 - self.MIN_DAMPER) * abs(u) * gain
            reheat = max(0.0, -u)
            if self.REHEAT_FULL_AT is not None and u < 0:
                # Hottest possible discharge for the air that is moved.
                reheat = min(1.0, -u / self.REHEAT_FULL_AT)
            split = self.HEAT_SEQUENCE_SPLIT
            if split is not None and u < 0:
                # Reheat first at minimum air; add air only once reheat is full.
                reheat = min(1.0, -u / split)
                damper = self.MIN_DAMPER + (1.0 - self.MIN_DAMPER) * max(0.0, (-u - split) / (1.0 - split))
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 2.0 * reheat - 1.0

    MIN_DAMPER = 0.1
    # Damper opening per unit of heating demand (reheated air costs gas).
    HEAT_DAMPER_GAIN = 1.0
    # Anti-windup: integral term limited to this share of full output, and a
    # per-step leak towards zero (both neutral by default).
    INTEGRAL_LIMIT_FRAC = 1.0
    INTEGRAL_LEAK = 0.0
    # Heating sequence: None = damper and reheat both follow |u|.
    HEAT_SEQUENCE_SPLIT = None
    # Reheat valve fully open once heating demand reaches this |u| (None =
    # valve follows |u|).
    REHEAT_FULL_AT = None
    # Scale of the PI output when heating (softer loop for hot-discharge heating).
    HEAT_U_SCALE = 1.0
    # In PI hours, every running AHU (not only forced ones) at minimum pressure.
    PRESSURE_MIN_ALL = False

    def _zone_target(self, obs, lo, hi):
        """Temperature each zone's PI loop aims at; the band centre by default."""
        return 0.5 * (lo + hi)

    def _base_action(self, obs):
        self._pi_active = self._pi_hours(obs)
        self._max_heat = 0.0
        self._forced = []
        self._zone_u = {}
        action = super()._base_action(obs)
        if self._pi_active:
            cont, disc = action["continuous_action"], action["discrete_action"]
            # Zone reheat needs hot water even when no AHU loop asks for heat.
            if self.water_temp_idx is not None and self._max_heat > 0:
                cont[self.water_temp_idx] = max(cont[self.water_temp_idx],
                                                np.float32(2.0 * self._max_heat - 1.0))
                if self.water_pressure_idx is not None:
                    cont[self.water_pressure_idx] = 1.0
            if self.PRESSURE_MIN_ALL:
                for ctrl in self.ahu_controllers:
                    if ctrl["pressure_idx"] is not None and disc[ctrl["run_discrete_idx"]]:
                        cont[ctrl["pressure_idx"]] = -1.0
            for i, ctrl in enumerate(self.ahu_controllers):
                if ctrl["zone_obs_indices"] and not disc[ctrl["run_discrete_idx"]]:
                    self._forced.append(i)
                    disc[ctrl["run_discrete_idx"]] = 1
                    cont[ctrl["temp_idx"]] = self.FORCED_SAT
                    if ctrl["pressure_idx"] is not None:
                        cont[ctrl["pressure_idx"]] = self.FORCED_PRESSURE
            if 0 <= self.water_discrete_idx < self.n_discrete:
                disc[self.water_discrete_idx] = 1
        return action

    def get_action(self, obs):
        action = super().get_action(obs)
        # A forced-on unit is not a cooling run; do not let the hold extend it.
        for i in self._forced:
            if self._last_on[i] is not None:
                self._last_on[i][0] = 0.0
        return action


def _outside_kelvin(policy, value):
    """AHU outside-air sensors arrive in raw kelvin; decode only if normalized."""
    value = float(value)
    return value if value > 150.0 else value * policy.outside_std + policy.outside_mean


class _ZonePIEconFixPolicy(_ZonePIPolicy):
    """pe_lowp with a working economizer (raw-kelvin outside air temperature)."""

    KP = 0.10
    KI = 0.005
    FORCED_PRESSURE = -1.0
    WARM_SAT = None
    WARM_BELOW_K = 283.15
    KEY = "econ_fix"

    def _recirculation(self, obs, ctrl, u, low, high):
        idx = ctrl.get("oat_idx")
        if idx is None:
            return 1.0
        fixed = obs.copy()
        fixed[idx] = _outside_kelvin(self, obs[idx])
        # Skip TransferPolicy's decoding; call the organizer economizer directly.
        return super(_BaselinePolicy, self)._recirculation(fixed, ctrl, u, low, high)

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if self.WARM_SAT is None:
            return action
        cont = action["continuous_action"]
        for i in self._forced:
            ctrl = self.ahu_controllers[i]
            idx = ctrl.get("oat_idx")
            if idx is not None and _outside_kelvin(self, obs[idx]) < self.WARM_BELOW_K:
                cont[ctrl["temp_idx"]] = self.WARM_SAT
        return action


class _DemandSatPolicy(_ZonePIEconFixPolicy):
    """Supply-air temperature follows the zones' own PI demand in work hours."""

    RAMP = 0.0
    COLD_SAT = -0.7
    # Statistic of the zones' PI outputs that sets the supply temperature:
    # None = mean; a number = that quantile.
    DEMAND_QUANTILE = None
    KEY = "dsat_bin"

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self._pi_active:
            return action
        cont, disc = action["continuous_action"], action["discrete_action"]
        for ctrl in self.ahu_controllers:
            us = [self._zone_u[v] for v, _, _ in ctrl["zones"] if v in self._zone_u]
            if not us or not disc[ctrl["run_discrete_idx"]]:
                continue
            d = (float(np.mean(us)) if self.DEMAND_QUANTILE is None
                 else float(np.quantile(us, self.DEMAND_QUANTILE)))
            frac = (float(np.clip(d / self.RAMP, 0.0, 1.0)) if self.RAMP > 0
                    else (1.0 if d > 0 else 0.0))
            cont[ctrl["temp_idx"]] = np.float32(1.0 + frac * (self.COLD_SAT - 1.0))
        return action


class _TunedPIPolicy(_ZonePIEconFixPolicy):
    """econ_fix_warmcold plus switchable energy measures, all off by default."""

    WARM_SAT = -0.4
    NIGHT_HEAT_DROP_K = 0.0
    WATER_SCALE = 1.0
    TARGET_SHIFT_K = 0.0
    OCC_GATE = False
    # Shift the target towards the low edge only in heating weather.
    COLD_SHIFT_ONLY = False
    # In PI hours, run the hot-water loop at full temperature whenever any
    # zone calls for heat (more heat per unit of cold mixed air).
    WATER_MAX_PI = False
    KEY = "tuned"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        names = list(self.field_names)
        self._occ_idx = names.index("num_occupants") if "num_occupants" in names else None
        self._oat_any = next((c["oat_idx"] for c in self.ahu_controllers
                              if c.get("oat_idx") is not None), None)

    def _pi_hours(self, obs):
        active = super()._pi_hours(obs)
        if active and self.OCC_GATE and self._occ_idx is not None:
            hour, _ = self._get_hour_and_day(obs)
            if hour >= self.morning_start_hour + 2 and float(obs[self._occ_idx]) < 0.5:
                return False
        return active

    def _zone_target(self, obs, lo, hi):
        centre = 0.5 * (lo + hi)
        if self.TARGET_SHIFT_K <= 0 or self._oat_any is None:
            return centre
        outside = _outside_kelvin(self, obs[self._oat_any])
        shift = min(self.TARGET_SHIFT_K, 0.4 * (hi - lo))
        if outside < centre:
            return centre - shift
        return centre if self.COLD_SHIFT_ONLY else centre + shift

    def _zone_band(self, obs, vav_id, low, high):
        lo, hi = super()._zone_band(obs, vav_id, low, high)
        if self.NIGHT_HEAT_DROP_K and not self._pi_active:
            lo -= self.NIGHT_HEAT_DROP_K
        return lo, hi

    def _ahu_error(self, obs, ctrl, low, high):
        if self.NIGHT_HEAT_DROP_K and not self._pi_active:
            low -= self.NIGHT_HEAT_DROP_K
        return super()._ahu_error(obs, ctrl, low, high)

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if (self.WATER_MAX_PI and self._pi_active and self.water_temp_idx is not None
                and self._max_heat > 0):
            action["continuous_action"][self.water_temp_idx] = np.float32(1.0)
        if self.WATER_SCALE != 1.0 and self.water_temp_idx is not None:
            cont = action["continuous_action"]
            frac = (float(cont[self.water_temp_idx]) + 1.0) * 0.5 * self.WATER_SCALE
            cont[self.water_temp_idx] = np.float32(2.0 * frac - 1.0)
        return action


class _TunedDemandSatPolicy(_TunedPIPolicy, _DemandSatPolicy):
    """dsat_bin with the TunedPIPolicy switches (all off by default)."""

    WARM_SAT = None  # dsat_bin did not use the cold-weather warm supply
    KEY = "tuned_dsat"


class _AdaptiveColdPolicy(_TunedDemandSatPolicy):
    """Pick the heating measures from a smoothed outside temperature."""

    NIGHT_HEAT_DROP_K = 2.0
    PRESTART_H = 2
    NIGHT_COOL_CEILING = False
    EXTREME_BELOW_C = -15.0
    MILD_BELOW_C = 12.0
    EXTREME_PRESTART_H = 4
    EXTREME_WATER_MAX = True
    EXTREME_HEAT_DAMPER = 0.5
    MILD_SHIFT_K = 0.5
    EMA_STEPS = 288
    # Full hot-water temperature in PI hours whenever a zone heats, any weather.
    ALWAYS_WATER_MAX = False
    KEY = "adaptive_cold"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._oat_ema = None
        self._base_prestart = self.PRESTART_H

    def _update_regime(self, obs):
        if self._oat_any is not None:
            outside = _outside_kelvin(self, obs[self._oat_any]) - 273.15
            if self._oat_ema is None:
                self._oat_ema = outside
            else:
                self._oat_ema += (outside - self._oat_ema) / self.EMA_STEPS
        ema = self._oat_ema
        extreme = ema is not None and ema < self.EXTREME_BELOW_C
        mild = ema is not None and not extreme and ema < self.MILD_BELOW_C
        self.PRESTART_H = self.EXTREME_PRESTART_H if extreme else self._base_prestart
        self.WATER_MAX_PI = bool((extreme and self.EXTREME_WATER_MAX) or self.ALWAYS_WATER_MAX)
        self.HEAT_DAMPER_GAIN = self.EXTREME_HEAT_DAMPER if extreme else 1.0
        self.TARGET_SHIFT_K = self.MILD_SHIFT_K if mild else 0.0
        self.COLD_SHIFT_ONLY = True

    def _base_action(self, obs):
        self._update_regime(obs)
        return super()._base_action(obs)


class _APResearchPolicy(_AdaptiveColdPolicy):
    """adapt_pre_only plus switchable hypotheses (all neutral by default)."""

    EXTREME_WATER_MAX = False
    EXTREME_HEAT_DAMPER = 1.0
    ZONE_SHIFT_K = 0.0
    ZONE_EMA_STEPS = 288
    ZONE_SHIFT_FULL_U = 0.2
    DYN_START_RATE = None
    DYN_START_MAX_H = 5
    NUDGE = 0.0
    # Idle window: in PI hours stop an AHU while every zone it serves has
    # |PI output| below IDLE_U (None = never).
    IDLE_U = None
    # Extra heating setback on weekend days (no occupancy, comfort not scored).
    WEEKEND_EXTRA_DROP_K = 0.0
    # Extra pre-start hours on Monday morning after a weekend setback.
    MONDAY_EXTRA_PRESTART_H = 0
    KEY = "ap_research"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._u_ema = {}
        self._all_zone_obs = [j for c in self.ahu_controllers for j in c["zone_obs_indices"]]

    def _zone_target(self, obs, lo, hi):
        target = super()._zone_target(obs, lo, hi)
        if self.ZONE_SHIFT_K <= 0:
            return target
        ema = self._u_ema.get(getattr(self, "_cur_vav", None))
        if ema is None:
            return target
        frac = float(np.clip(ema / self.ZONE_SHIFT_FULL_U, -1.0, 1.0))
        shift = min(self.ZONE_SHIFT_K, 0.4 * (hi - lo)) * frac
        return float(np.clip(target + shift, lo + 0.3, hi - 0.3))

    def _weekend_drop(self, obs):
        if not self.WEEKEND_EXTRA_DROP_K or self._pi_active:
            return 0.0
        _, day = self._get_hour_and_day(obs)
        return self.WEEKEND_EXTRA_DROP_K if day >= 5 else 0.0

    def _zone_band(self, obs, vav_id, low, high):
        lo, hi = super()._zone_band(obs, vav_id, low, high)
        return lo - self._weekend_drop(obs), hi

    def _ahu_error(self, obs, ctrl, low, high):
        return super()._ahu_error(obs, ctrl, low - self._weekend_drop(obs), high)

    def _pi_hours(self, obs):
        active = super()._pi_hours(obs)
        if not active and self.MONDAY_EXTRA_PRESTART_H:
            hour, day = self._get_hour_and_day(obs)
            if day == 0 and self.morning_start_hour - self.PRESTART_H - self.MONDAY_EXTRA_PRESTART_H <= hour < self.morning_start_hour:
                return True
        if active or self.DYN_START_RATE is None or not self._all_zone_obs:
            return active
        hour, day = self._get_hour_and_day(obs)
        left = self.morning_start_hour - hour
        if day >= 5 or left <= 0 or left > self.DYN_START_MAX_H:
            return False
        mean_t = float(np.mean([self._denormalize_zone_temp(obs[j]) for j in self._all_zone_obs]))
        return mean_t < self.comfort_temp_window[0] - self.DYN_START_RATE * left

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if self.ZONE_SHIFT_K > 0:
            for v, u in self._zone_u.items():
                prev = self._u_ema.get(v, u)
                self._u_ema[v] = prev + (u - prev) / self.ZONE_EMA_STEPS
        if self.IDLE_U is not None and self._pi_active:
            cont, disc = action["continuous_action"], action["discrete_action"]
            for i, ctrl in enumerate(self.ahu_controllers):
                us = [abs(self._zone_u[v]) for v, _, _ in ctrl["zones"] if v in self._zone_u]
                if us and max(us) < self.IDLE_U and disc[ctrl["run_discrete_idx"]]:
                    disc[ctrl["run_discrete_idx"]] = 0
                    cont[ctrl["temp_idx"]] = -1.0
                    if ctrl["pressure_idx"] is not None:
                        cont[ctrl["pressure_idx"]] = -1.0
                    if i in self._forced:
                        self._forced.remove(i)
        if self.NUDGE:
            cont, disc = action["continuous_action"], action["discrete_action"]
            for ctrl in self.ahu_controllers:
                idx = ctrl["pressure_idx"]
                if idx is not None and disc[ctrl["run_discrete_idx"]]:
                    cont[idx] += np.float32(-self.NUDGE if cont[idx] > 0 else self.NUDGE)
        return action


class RefZtBest(_APResearchPolicy):
    """The notebook's `SubmissionPolicy`: the exact constants of zt-best."""

    KP = 0.1
    KI = 0.005
    HOLD_STEPS = 288
    NIGHT_COOL_CEILING = False
    ZONE_SHIFT_K = 0.6
    DEMAND_QUANTILE = 0.5
    ZONE_EMA_STEPS = 144
    WEEKEND_EXTRA_DROP_K = 6.0
    PRESSURE_MIN_ALL = True
