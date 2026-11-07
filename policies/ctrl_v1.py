"""CtrlV1 -- a building-agnostic structured HVAC controller (numpy only).

Everything is derived from ``building_meta_data`` and the live observation, so
one parameter vector controls 3-9 air handlers and 28-85 zones. The module is
deliberately self-contained (no ``smart_buildings`` import) so it can be pasted
into a Kaggle notebook.

Structure (top to bottom)
-------------------------
1. Index tables built in ``__init__`` from metadata: per-zone temperature,
   live setpoint bands, damper/reheat action indices, AHU membership, AHU
   temperature/pressure/recirculation indices, outside-air index, hot-water
   indices, discrete layout, time decoding and occupancy.
2. Comfort hours: per-zone PI (conditional integration, anti-windup) on a
   target inside the *live* band, shifted toward the cheap edge per zone, run
   on a one-step extrapolation of the lagging sensor. One signed effort drives
   the damper (both directions, never below the ventilation floor) and the
   reheat valve (heating direction only); the valve leads the damper so the
   coil, whose discharge temperature is capped, is fed air as soon as it can
   use it. A zone inside its deadband takes the ventilation minimum only:
   in a cold climate air, not water, is what the gas bill is made of.
3. Air handler: runs only when its zones need it (hysteresis, minimum on-time,
   cooling-run hold). Supply-air temperature from the *median* cooling demand
   of its zones, no chiller when the box is heating-dominated. Recirculation at
   the cap when heating, economizer only when the weather pays for the fan.
   Static pressure held near the simulator floor, scaled up only for a large
   unmet demand: measured, the fan is a bigger loss than the extra cooling.
4. Hot water: on only when some zone holds reheat demand. Supply temperature
   reset by the worst-zone shortfall and the air-side saturation (both are the
   weather, read through the plant), pump differential pressure near minimum
   unless demand is high.
5. Unoccupied: plant off, building floats, with a floor-temperature guard and
   an optimal-start pre-heat/pre-cool computed from the current worst-zone
   deficit and an online estimate of this building's own recovery rate. Late
   afternoon with no occupants left: stop conditioning and coast. An opt-in
   night cooling ceiling exists but is off by default (see known weaknesses).
6. Every output is clipped, nan-to-num'ed and cast; ``get_action`` never
   raises.

Known weaknesses (measured, see results/ctrl_v1/)
-------------------------------------------------
* The morning ramp still sets most of the comfort loss. In the hot climate the
  first occupied hour carries ~40% of the episode's comfort regret: the
  building soaks up heat overnight and the plant cannot remove it fast enough
  at 06:00. A night cooling ceiling was tried and costs 2-4x what it saves;
  the remaining fix is lookahead, not a better setpoint.
* Holidays are invisible. ``comfort_mode_soon`` is holiday-aware but only one
  hour ahead, and the decoded clock has no day of year, so an optimal start on
  a holiday pre-conditions an empty building. On building_train_1/fairbanks_1
  (2023-01-01, whose first Monday is a holiday) that is the whole cost of the
  two-day smoke test.
* Deep cold: the floor guard and the warm-up are the only defence against a
  cold soak, and the recovery rate is learned from this building's own warm-ups,
  so the first day of an episode starts on the prior.
* Per-zone occupancy is unobservable, so every zone is held and the comfort
  budget cannot be spent where people actually are.
* The cooling-run hold and the AHU on-time counters are per-handler constants;
  they are not adapted to a building's thermal mass.

PARAMS (all dimensionless or physical; one vector works for every building:
nothing below is keyed to zone counts, AHU counts, weather names or ids)
--------------------------------------------------------------------------
Zone loop -- one PI per zone on a target inside the live comfort band
  KP                    0.08    proportional gain, effort per kelvin of error
  KI                    0.06    integral gain, effort per (kelvin * hour)
  INTEGRAL_LIMIT_KH     8.0     anti-windup clamp on the integral state
  MARGIN_K              0.50    target stays at least this far inside the band
  ZONE_SHIFT_MAX_K      0.60    cap on the per-zone economic target shift
  ZONE_SHIFT_MAX_FRAC   0.40    same cap as a fraction of the band half-width
  ZONE_SHIFT_FULL_U     0.20    effort EMA magnitude that earns the full shift
  ZONE_EMA_STEPS      144.0     time constant of the per-zone effort EMA (steps)
  ZONE_EMA_RAMP_STEPS 288.0     steps over which that EMA replaces the prior
  OAT_NEUTRAL_K       292.0     outside air at which neither edge is cheaper
  OAT_SPAN_K            8.0     kelvin of outside air per unit of prior shift
  LAG_ALPHA             1.0     weight on the one-step extrapolation of the
                                sensor lag (1.0 = full dead-reckoning)
  LAG_MAX_K             1.0     cap on that extrapolation
  MIN_DAMPER            0.10    simulator ventilation floor for a damper
  DAMPER_GAIN           1.0     cooling damper opening per unit of effort
  DAMPER_U0             0.10    effort below which a zone takes no air at all
  HEAT_SEQ_SPLIT        0.10    heating sequence: the valve is fully open at
                                this effort, air only opens beyond it
  HEAT_DAMPER_GAIN      1.0     heating damper opening per unit of effort
                                past HEAT_SEQ_SPLIT
  REHEAT_FULL_U         1.0     effort scale of the reheat valve
Air handler -- one row per AHU
  NEED_U                0.12    effort below which a zone asks for no air
  NEED_K                0.20    band excursion that always counts as a demand
  MIN_ON_STEPS           6      minimum air-handler on time (5-minute steps)
  COOL_HOLD_STEPS      144      cooling-run latch (steps; 12 h)
  DEMAND_FULL           0.5     cooling demand that earns the coldest supply air
  DEMAND_QUANTILE       0.5     quantile of the zones' demand that sets it
  SAT_MIN_K           285.0     coldest supply-air setpoint (native, K)
  SAT_MAX_K           305.0     warmest supply-air setpoint (no chiller, K)
  PRESS_MIN_FRAC        0.10    static pressure when a box runs idle (of range)
  PRESS_MAX_FRAC        1.00    static pressure cap
  PRESS_DEADBAND        0.80    effort the pressure ignores
  PRESS_GAIN            1.0     pressure rise per unit of effort past the
                                deadband, as a fraction of the range
  ECON_MIN_ADV_K        0.50    outside-vs-return advantage needed to economize
  ECON_K_PER_PA     0.00322     outside-air advantage needed per pascal of fan
Hot water -- building-wide
  WATER_MIN_K         310.0     supply-water setpoint when the loop is idle
  WATER_MAX_K         350.0     supply-water setpoint at full demand
  WATER_RESET_FULL_K    4.0     worst-zone shortfall that earns full water
  WATER_EFFORT_WEIGHT   1.0     weight on the air-side saturation term
  WATER_EFFORT_QUANT 0.25       quantile over AHUs of the heating effort
  DP_MIN_FRAC           0.15    pump differential pressure when idle (of range)
  DP_GAIN               1.0     pressure rise per unit of reheat demand
  COLD_EMA_STEPS      288.0     time constant of the outside-air EMA (steps)
  WATER_COLD_EMA_K   258.15     outside-air EMA below which the explicit
                                weather term applies
  WATER_COLD_FRAC       0.0     water fraction that weather term forces. Zero by
                                default: the demand and saturation terms
                                already carry the weather in through the plant,
                                and forcing full water in deep cold measured
                                3% worse (tank and return-temperature losses
                                beat the faster recovery)
Unoccupied -- float, guard, optimal start, early stop
  FLOOR_TEMP_K        285.0     minimum zone temperature the guard defends
  FLOOR_BAND_K          0.5     the guard aims this far above FLOOR_TEMP_K
  FLOOR_PBAND_K         1.0     shortfall at which the guard is at full effort
  CEILING_MARGIN_K     None     opt-in night cooling ceiling: cool above
                                (comfort high - this). Measured on a hot
                                10-day episode: on costs 2-4x what it saves, so
                                the default leaves the plant off overnight and
                                the building floats; the guard is kept for
                                experiments (e.g. 0.5)
  CEILING_PBAND_K       2.0     excess at which the ceiling guard is at full
                                effort
  CEILING_GAIN          1.0     ceiling-guard effort scale
  RECOVERY_PRIOR_K_PER_H 1.5    prior heating recovery rate (K/h)
  COOL_RATE_PRIOR_K_PER_H 1.5   prior cooling recovery rate (K/h)
  RATE_EMA_STEPS       36.0     time constant of the recovery-rate estimate
  RATE_MIN_K_PER_H      0.2     lower clamp on a learned rate
  RATE_MAX_K_PER_H      6.0     upper clamp on a learned rate
  PRESTART_MAX_H        8.0     never pre-condition earlier than this
  PRESTART_SAFETY       1.2     lead-time safety factor
  PRESTART_SLACK_H      0.25    start this many hours early
  EARLY_STOP_H          1.0     stop this many hours before evening start
  EARLY_STOP_OCC        0.5     occupancy at or below which "nobody is left"
Timing
  DT_H              1.0/12.0    control interval in hours (5 minutes)
"""

from __future__ import annotations

import os

import numpy as np

DEFAULTS = {
    # Zone loop.
    "KP": 0.08,
    "KI": 0.06,
    "INTEGRAL_LIMIT_KH": 8.0,
    "MARGIN_K": 0.50,
    "ZONE_SHIFT_MAX_K": 0.60,
    "ZONE_SHIFT_MAX_FRAC": 0.40,
    "ZONE_SHIFT_FULL_U": 0.20,
    "ZONE_EMA_STEPS": 144.0,
    "ZONE_EMA_RAMP_STEPS": 288.0,
    "OAT_NEUTRAL_K": 292.0,
    "OAT_SPAN_K": 8.0,
    "LAG_ALPHA": 1.0,
    "LAG_MAX_K": 1.0,
    "MIN_DAMPER": 0.10,
    "DAMPER_GAIN": 1.0,
    "DAMPER_U0": 0.10,
    "HEAT_SEQ_SPLIT": 0.10,
    "HEAT_DAMPER_GAIN": 1.0,
    "REHEAT_FULL_U": 1.0,
    # Air handler.
    "NEED_U": 0.12,
    "NEED_K": 0.20,
    "MIN_ON_STEPS": 6,
    "COOL_HOLD_STEPS": 144,
    "DEMAND_FULL": 0.5,
    "DEMAND_QUANTILE": 0.5,
    "SAT_MIN_K": 285.0,
    "SAT_MAX_K": 305.0,
    "PRESS_MIN_FRAC": 0.10,
    "PRESS_MAX_FRAC": 1.0,
    "PRESS_DEADBAND": 0.80,
    "PRESS_GAIN": 1.0,
    "ECON_MIN_ADV_K": 0.5,
    "ECON_K_PER_PA": 0.00322,
    # Hot water.
    "WATER_MIN_K": 310.0,
    "WATER_MAX_K": 350.0,
    "WATER_RESET_FULL_K": 4.0,
    "WATER_EFFORT_WEIGHT": 1.0,
    "WATER_EFFORT_QUANT": 0.25,
    "DP_MIN_FRAC": 0.15,
    "DP_GAIN": 1.0,
    "COLD_EMA_STEPS": 288.0,
    "WATER_COLD_EMA_K": 258.15,
    "WATER_COLD_FRAC": 0.0,
    # Unoccupied.
    "FLOOR_TEMP_K": 285.0,
    "FLOOR_BAND_K": 0.5,
    "FLOOR_PBAND_K": 1.0,
    "CEILING_MARGIN_K": None,
    "CEILING_PBAND_K": 2.0,
    "CEILING_GAIN": 1.0,
    "RECOVERY_PRIOR_K_PER_H": 1.5,
    "COOL_RATE_PRIOR_K_PER_H": 1.5,
    "RATE_EMA_STEPS": 36.0,
    "RATE_MIN_K_PER_H": 0.2,
    "RATE_MAX_K_PER_H": 6.0,
    "PRESTART_MAX_H": 8.0,
    "PRESTART_SAFETY": 1.2,
    "PRESTART_SLACK_H": 0.25,
    "EARLY_STOP_H": 1.0,
    "EARLY_STOP_OCC": 0.5,
    # Timing.
    "DT_H": 1.0 / 12.0,
}

# Fallback native actuator ranges, used only when the metadata does not carry
# the matching action normalizer. These are the shipped configs' values.
_FALLBACK_RANGES = {
    "supply_air_temperature_setpoint": (285.0, 305.0),
    "supply_air_static_pressure_setpoint": (0.0, 2000.0),
    "recirculation_percentage_command": (0.0, 1.0),
    "supply_water_temperature_setpoint": (310.0, 350.0),
    "differential_pressure_setpoint": (0.0, 20.0),
    "supply_air_damper_percentage_command": (0.0, 1.0),
    "reheat_valve_percentage_command": (0.0, 1.0),
}


def _angle01(sin_value: float, cos_value: float) -> float:
    """Fraction of a full cycle implied by a (sin, cos) feature pair."""
    if not (np.isfinite(sin_value) and np.isfinite(cos_value)):
        return 0.0
    angle = float(np.arctan2(sin_value, cos_value))
    if angle < 0.0:
        angle += 2.0 * np.pi
    return angle / (2.0 * np.pi)


def _numeric_suffix(name: str) -> int:
    """Trailing-or-first integer in a device id, for stable ordering."""
    digits = ""
    for char in reversed(str(name)):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    return int(digits) if digits else 0


class CtrlV1:
    """Structured rule controller: per-zone PI, demand-based plant, optimal start."""

    PARAMS = dict(DEFAULTS)
    # Set CTRLV1_DEBUG=1 to surface internal errors instead of the fallback.
    DEBUG = bool(os.environ.get("CTRLV1_DEBUG"))

    # ------------------------------------------------------------------ setup
    def __init__(self, building_meta_data, action_spec=None):
        params = dict(DEFAULTS)
        params.update(getattr(type(self), "PARAMS", None) or {})
        self.p = params
        self.action_spec = action_spec
        self.meta = building_meta_data
        self.name = "ctrl_v1"

        self.n_cont = int(building_meta_data["n_continuous_action"])
        self.n_disc = int(building_meta_data["n_discrete_action"])
        self.field_names = list(building_meta_data["observation_field_names"])
        self._name_to_idx = {name: i for i, name in enumerate(self.field_names)}

        self._build_clock_indices()
        self._build_action_ranges()
        self._build_zone_tables()
        self._build_ahu_tables()
        self._build_plant_tables()
        self._init_state()

    def _find(self, name: str) -> int:
        return self._name_to_idx.get(name, -1)

    def _build_clock_indices(self) -> None:
        self.i_hod_sin = self._find("hod_sin_000")
        self.i_hod_cos = self._find("hod_cos_000")
        self.i_dow_sin = self._find("dow_sin_000")
        self.i_dow_cos = self._find("dow_cos_000")
        self.i_comfort_now = self._find("comfort_mode_now")
        self.i_comfort_soon = self._find("comfort_mode_soon")
        self.i_occupants = self._find("num_occupants")

    def _range_for(self, suffix: str, device: str | None = None) -> tuple[float, float]:
        """Native (min, max) of one actuator, from metadata with a fallback."""
        anorm = self.meta.get("action_normalization_metadata", {}) or {}
        keys = []
        if device is not None:
            keys.append(f"{device}_{suffix}")
        keys.append(suffix)
        for key in keys:
            entry = anorm.get(key)
            if isinstance(entry, dict):
                lo = float(entry.get("min_native_value", np.nan))
                hi = float(entry.get("max_native_value", np.nan))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    return lo, hi
        # Unknown key name: accept any normalizer that ends with the suffix.
        for key, entry in anorm.items():
            if str(key).endswith(suffix) and isinstance(entry, dict):
                lo = float(entry.get("min_native_value", np.nan))
                hi = float(entry.get("max_native_value", np.nan))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    return lo, hi
        return _FALLBACK_RANGES.get(suffix, (0.0, 1.0))

    def _build_action_ranges(self) -> None:
        self.rng_pressure = self._range_for("supply_air_static_pressure_setpoint")
        self.rng_recirc = self._range_for("recirculation_percentage_command")
        # The water and pump normalizers are plant-wide; the supply-air
        # temperature range is per air handler and resolved in _build_ahu_tables.
        self.rng_water = self._range_for("supply_water_temperature_setpoint")
        self.rng_dp = self._range_for("differential_pressure_setpoint")

    def _norm(self, value: float, rng: tuple[float, float]) -> float:
        lo, hi = rng
        if not np.isfinite(value) or hi <= lo:
            return 0.0
        return float(np.clip(2.0 * (value - lo) / (hi - lo) - 1.0, -1.0, 1.0))

    def _build_zone_tables(self) -> None:
        """Per-zone observation and action indices, and AHU membership."""
        zone_actions = self.meta.get("zone_action_indices", {}) or {}

        # AHU ids in the order the module's position lists use. The action
        # normalizers are keyed by ``ahu_<n>_*`` and the metadata's positional
        # lists follow the same device order, so a numeric sort of the ids is
        # the ordering those lists are in.
        assignments = self.meta.get("ahu_zone_assignments")
        if not assignments:
            assignments = self.meta.get("aha_zone_assignments", {}) or {}
        self.ahu_ids = sorted(assignments.keys(), key=_numeric_suffix)
        ahu_pos = {ahu_id: i for i, ahu_id in enumerate(self.ahu_ids)}
        zone_to_ahu: dict[str, int] = {}
        for ahu_id, rooms in assignments.items():
            for room in rooms:
                zone_to_ahu[str(room).replace("room_", "VAV_")] = ahu_pos[ahu_id]

        obs_mean = 295.0
        obs_std = 5.0
        onorm = self.meta.get("observation_normalization_metadata", {}) or {}
        entry = onorm.get("zone_air_temperature_sensor", {}) or {}
        mean = float(entry.get("sample_mean", np.nan))
        std = float(entry.get("sample_std", np.nan))
        if np.isfinite(mean) and np.isfinite(std) and std > 0.0:
            obs_mean, obs_std = mean, std
        self.temp_mean = obs_mean
        self.temp_std = obs_std

        vavs, temp_idx, lo_idx, hi_idx, damp_idx, reheat_idx, ahu_of = [], [], [], [], [], [], []
        orphans = []
        for vav in vavs_from(zone_actions):
            actions = zone_actions.get(vav, {}) or {}
            damp = actions.get("supply_air_damper_percentage_command")
            reheat = actions.get("reheat_valve_percentage_command")
            temp_i = self._find(f"{vav}_zone_air_temperature_sensor")
            if temp_i < 0:
                if damp is not None or reheat is not None:
                    orphans.append((damp, reheat))
                continue
            vavs.append(vav)
            temp_idx.append(temp_i)
            lo_idx.append(self._find(f"{vav}_zone_air_heating_temperature_setpoint_sensor"))
            hi_idx.append(self._find(f"{vav}_zone_air_cooling_temperature_setpoint_sensor"))
            damp_idx.append(-1 if damp is None else int(damp))
            reheat_idx.append(-1 if reheat is None else int(reheat))
            ahu_of.append(zone_to_ahu.get(vav, 0))

        self.zone_ids = vavs
        self.z_temp = np.asarray(temp_idx, dtype=np.int64)
        self.z_lo = np.asarray(lo_idx, dtype=np.int64)
        self.z_hi = np.asarray(hi_idx, dtype=np.int64)
        self.z_damp = np.asarray(damp_idx, dtype=np.int64)
        self.z_reheat = np.asarray(reheat_idx, dtype=np.int64)
        self.z_ahu = np.asarray(ahu_of, dtype=np.int64)
        self.n_zones = len(vavs)
        self._orphans = orphans

        self.sched_comfort = self._schedule_window("comfort_temp_window", (294.0, 297.0))
        self.sched_eco = self._schedule_window("eco_temp_window", (289.0, 298.0))
        self.morning_start = float(self.meta.get("morning_start_hour", 6))
        self.evening_start = float(self.meta.get("evening_start_hour", 19))

    def _schedule_window(self, key: str, default: tuple[float, float]) -> tuple[float, float]:
        window = self.meta.get(key, default)
        try:
            lo, hi = float(window[0]), float(window[1])
        except (TypeError, IndexError, ValueError):
            return default
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo + 0.5:
            return default
        return lo, hi

    def _build_ahu_tables(self) -> None:
        temp_indices = [int(i) for i in self.meta.get("temp_indices", [])]
        pressure_indices = [int(i) for i in self.meta.get("pressure_indices", [])]
        recirc_indices = [int(i) for i in self.meta.get("recirculation_indices", [])]
        self.n_ahu = len(temp_indices)
        if not self.n_ahu:
            # No plant actions at all: nothing to control but the zones.
            self.temp_idx = []
            self.press_idx = []
            self.recirc_idx = []
            self.oat_idx = []
            self.sat_rng = []
            self.ahu_zones = []
            return

        # An extra pressure action beyond the AHUs is the pump's differential
        # pressure; it is handled as a plant action, not an air handler.
        self.temp_idx = temp_indices
        self.press_idx = pressure_indices[: self.n_ahu]
        self.recirc_idx = recirc_indices[: self.n_ahu]
        self.oat_idx = []
        self.sat_rng = []
        self.ahu_zones = []
        for i in range(self.n_ahu):
            ahu_id = self.ahu_ids[i] if i < len(self.ahu_ids) else f"ahu_{i + 1}"
            self.oat_idx.append(self._find(f"ahs_{ahu_id}_outside_air_temperature_sensor"))
            self.sat_rng.append(
                self._range_for("supply_air_temperature_setpoint", ahu_id)
            )
            self.ahu_zones.append(np.flatnonzero(self.z_ahu == i))
        # Pad the per-AHU action lists so indexing never runs off the end.
        while len(self.press_idx) < self.n_ahu:
            self.press_idx.append(-1)
        while len(self.recirc_idx) < self.n_ahu:
            self.recirc_idx.append(-1)

    def _build_plant_tables(self) -> None:
        water_indices = [int(i) for i in self.meta.get("water_indices", [])]
        self.water_temp_idx = water_indices[0] if water_indices else -1
        pressure_indices = [int(i) for i in self.meta.get("pressure_indices", [])]
        self.water_dp_idx = (
            pressure_indices[self.n_ahu] if len(pressure_indices) > self.n_ahu else -1
        )
        # Discrete layout: one run command per air handler in AHU order, then
        # the hot-water system. Read from the tail, which is where the plant
        # switch sits in every shipped config.
        self.water_run_idx = self.n_disc - 1 if self.n_disc > self.n_ahu else -1

    def _init_state(self) -> None:
        n = self.n_zones
        self.integ = np.zeros(n)
        self.u = np.zeros(n)
        self.u_ema = np.zeros(n)
        self.t_prev = None
        self.rate_heat = np.full(n, float(self.p["RECOVERY_PRIOR_K_PER_H"]))
        self.rate_cool = np.full(n, float(self.p["COOL_RATE_PRIOR_K_PER_H"]))
        self.oat_ema = None
        self.steps = 0
        self.ahu_run = [False] * max(self.n_ahu, 1)
        self.ahu_on_steps = [0] * max(self.n_ahu, 1)
        self.ahu_hold = [0] * max(self.n_ahu, 1)
        self._last_action = None
        self.diag = {}

    # ------------------------------------------------------------------ policy
    def get_action(self, obs):
        """Return a valid hybrid action; never raises."""
        try:
            action = self._act(obs)
            self._last_action = action
            return action
        except Exception:  # pylint: disable=broad-except
            if self.DEBUG:
                raise
            if self._last_action is not None:
                return self._last_action
            return self._safe_action()

    def _safe_action(self):
        cont = np.full(self.n_cont, 2.0 * self.p["MIN_DAMPER"] - 1.0, dtype=np.float32)
        disc = np.zeros(self.n_disc, dtype=np.int32)
        if self.water_temp_idx >= 0:
            cont[self.water_temp_idx] = -1.0
        return {"continuous_action": cont, "discrete_action": disc}

    def _act(self, obs):
        p = self.p
        dt = float(p["DT_H"])
        obs = np.nan_to_num(
            np.asarray(obs, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.steps += 1

        # -- clock ---------------------------------------------------------
        hour = 24.0 * _angle01(obs[self.i_hod_sin], obs[self.i_hod_cos])
        # Day of week is a discrete 1/7 step; round so a value a float epsilon
        # below Sunday's integer does not read as Saturday.
        day = int(round(7.0 * _angle01(obs[self.i_dow_sin], obs[self.i_dow_cos]))) % 7
        comfort_now = self.i_comfort_now >= 0 and obs[self.i_comfort_now] > 0.5
        comfort_soon = self.i_comfort_soon >= 0 and obs[self.i_comfort_soon] > 0.5
        occupants = float(obs[self.i_occupants]) if self.i_occupants >= 0 else 0.0

        oat = self._outside_temperature(obs)
        if self.oat_ema is None:
            self.oat_ema = oat
        else:
            self.oat_ema += (oat - self.oat_ema) / max(1.0, float(p["COLD_EMA_STEPS"]))

        # -- zone state ----------------------------------------------------
        temp = obs[self.z_temp] * self.temp_std + self.temp_mean
        temp = np.nan_to_num(temp, nan=self.temp_mean)
        if self.t_prev is None or self.t_prev.shape != temp.shape:
            slope = np.zeros_like(temp)
        else:
            slope = np.clip(
                (temp - self.t_prev) * float(p["LAG_ALPHA"]),
                -float(p["LAG_MAX_K"]),
                float(p["LAG_MAX_K"]),
            )
        predict = temp + slope

        # -- live band -----------------------------------------------------
        soon_mode = comfort_now or comfort_soon
        sched_lo, sched_hi = self.sched_comfort if soon_mode else self.sched_eco
        lo = self._band_axis(obs, self.z_lo, sched_lo)
        hi = self._band_axis(obs, self.z_hi, sched_hi)
        bad = ~(hi > lo + 0.5)
        lo = np.where(bad, sched_lo, lo)
        hi = np.where(bad, sched_hi, hi)
        if soon_mode:
            # Before occupancy the reported band is still the eco band; the
            # pre-conditioning target must be the comfort band.
            widen = (hi - lo) > (self.sched_comfort[1] - self.sched_comfort[0]) + 1e-6
            lo = np.where(widen, self.sched_comfort[0], lo)
            hi = np.where(widen, self.sched_comfort[1], hi)

        # -- occupancy regime ---------------------------------------------
        occupied = comfort_now or occupants > 0.0
        late = hour >= self.evening_start - float(p["EARLY_STOP_H"])
        empty_evening = comfort_now and late and occupants <= float(p["EARLY_STOP_OCC"])
        hours_left, next_is_workday = self._hours_to_comfort(hour, day)
        need_k = float(p["NEED_K"])
        deficit = np.clip(lo + float(p["MARGIN_K"]) - predict, 0.0, None)
        excess = np.clip(predict - (hi - float(p["MARGIN_K"])), 0.0, None)
        lead_heat = float(np.max(deficit / np.maximum(self.rate_heat, 1e-3))) if self.n_zones else 0.0
        lead_cool = float(np.max(excess / np.maximum(self.rate_cool, 1e-3))) if self.n_zones else 0.0
        lead = max(lead_heat, lead_cool)
        prestart = False
        if next_is_workday and not occupied and not empty_evening:
            within_window = hours_left <= float(p["PRESTART_SAFETY"]) * min(
                lead, float(p["PRESTART_MAX_H"])
            )
            within_window = within_window and hours_left <= float(p["PRESTART_MAX_H"])
            arrived_soon = comfort_soon and lead > 0.0
            prestart = within_window or arrived_soon
            # Nothing is wrong yet: no lead time means no reason to run.
            prestart = prestart and lead > float(p["PRESTART_SLACK_H"]) / float(
                p["PRESTART_SAFETY"]
            )

        conditioning = (occupied and not empty_evening) or prestart

        # -- float guards ---------------------------------------------------
        # The building is allowed to drift overnight, but not without bound:
        # too cold and the morning cannot recover, too hot and the first
        # occupied hour is spent removing heat that nobody was being charged
        # for. Both guards act only when the building is otherwise unoccupied.
        floor_target = float(p["FLOOR_TEMP_K"]) + float(p["FLOOR_BAND_K"])
        floor_short = np.clip(floor_target - predict, 0.0, None)
        guard = floor_short > 0.0
        ceiling_margin = p["CEILING_MARGIN_K"]
        if ceiling_margin is None:
            ceiling_excess = np.zeros(self.n_zones)
        else:
            ceiling_target = self.sched_comfort[1] - float(ceiling_margin)
            ceiling_excess = np.clip(predict - ceiling_target, 0.0, None)
        ceil_guard = ceiling_excess > 0.0

        # -- per-zone PI ---------------------------------------------------
        width = hi - lo
        centre = 0.5 * (lo + hi)
        ema_weight = min(1.0, self.steps / max(1.0, float(p["ZONE_EMA_RAMP_STEPS"])))
        prior = float(np.clip(
            (oat - float(p["OAT_NEUTRAL_K"])) / max(1.0, float(p["OAT_SPAN_K"])), -1.0, 1.0
        ))
        ema_frac = np.clip(self.u_ema / max(1e-6, float(p["ZONE_SHIFT_FULL_U"])), -1.0, 1.0)
        shift_frac = np.clip(
            ema_weight * ema_frac + (1.0 - ema_weight) * prior, -1.0, 1.0
        )
        shift_max = np.minimum(
            float(p["ZONE_SHIFT_MAX_K"]),
            np.maximum(0.0, float(p["ZONE_SHIFT_MAX_FRAC"]) * width),
        )
        margin = np.minimum(float(p["MARGIN_K"]), 0.45 * width)
        target = np.clip(centre + shift_frac * shift_max, lo + margin, hi - margin)

        err = predict - target
        saturated = np.abs(self.u) >= 0.999
        winding = saturated & (np.sign(err) == np.sign(self.u))
        integ = np.where(winding, self.integ, self.integ + err * dt)
        integ = np.clip(integ, -float(p["INTEGRAL_LIMIT_KH"]), float(p["INTEGRAL_LIMIT_KH"]))
        u_raw = np.clip(float(p["KP"]) * err + float(p["KI"]) * integ, -1.0, 1.0)

        active = np.full(self.n_zones, conditioning, dtype=bool)
        if not conditioning:
            u_eff = np.zeros(self.n_zones)
            # Floor guard: heat only, for the zones that are actually drifting.
            if np.any(guard):
                drive = np.clip(
                    floor_short / max(1e-3, float(p["FLOOR_PBAND_K"])), 0.0, 1.0
                )
                u_eff = np.where(guard, -drive, 0.0)
                active = guard
                integ = np.where(guard, integ, 0.0)
            else:
                integ = np.zeros(self.n_zones)
            if np.any(ceil_guard):
                drive = np.clip(
                    ceiling_excess / max(1e-3, float(p["CEILING_PBAND_K"])), 0.0, 1.0
                ) * float(p["CEILING_GAIN"])
                u_eff = np.where(ceil_guard, drive, u_eff)
                active = active | ceil_guard
                integ = np.where(ceil_guard, 0.0, integ)
        else:
            u_eff = u_raw

        # -- zone actuators -------------------------------------------------
        # A zone inside its deadband sits at the ventilation floor: any air it
        # gets has to be heated from the mixed temperature in a cold climate,
        # so air is the expensive resource, not water.
        cooling = np.clip(u_eff, 0.0, 1.0)
        heat_mag = np.clip(-u_eff, 0.0, 1.0)
        heat_gain = float(p["HEAT_DAMPER_GAIN"])
        min_damper = float(p["MIN_DAMPER"])
        u0 = float(p["DAMPER_U0"])
        cool_open = np.clip((cooling - u0) / max(1e-6, 1.0 - u0), 0.0, 1.0)
        split = float(p["HEAT_SEQ_SPLIT"])
        # Heating sequence: valve first at the ventilation floor, air only once
        # the coil has nothing left to give (its discharge is capped, so extra
        # air above the floor buys heat only at the price of cold-air ingress).
        reheat = np.clip(heat_mag / max(1e-6, split), 0.0, 1.0)
        reheat = reheat / max(1e-6, float(p["REHEAT_FULL_U"]))
        heat_air = np.clip((heat_mag - split) / max(1e-6, 1.0 - split), 0.0, 1.0)
        damper = min_damper + (1.0 - min_damper) * np.where(
            u_eff < 0.0,
            heat_air * heat_gain,
            cool_open * float(p["DAMPER_GAIN"]),
        )
        damper = np.clip(damper, min_damper, 1.0)
        damper = np.where(active, damper, min_damper)
        reheat = np.where(active, np.clip(reheat, 0.0, 1.0), 0.0)

        # -- air handlers ----------------------------------------------------
        need_zone = active & (
            (np.abs(u_eff) > float(p["NEED_U"]))
            | (predict > hi + need_k)
            | (predict < lo - need_k)
        )
        cont = np.full(self.n_cont, 2.0 * min_damper - 1.0, dtype=np.float64)
        disc = np.zeros(self.n_disc, dtype=np.int32)
        damp_ok = self.z_damp >= 0
        if np.any(damp_ok):
            cont[self.z_damp[damp_ok]] = 2.0 * damper[damp_ok] - 1.0
        rht_ok = self.z_reheat >= 0
        if np.any(rht_ok):
            cont[self.z_reheat[rht_ok]] = 2.0 * reheat[rht_ok] - 1.0
        self._write_orphans(cont)

        heat_by_ahu = np.zeros(max(self.n_ahu, 1))
        for i in range(self.n_ahu):
            idx = self.ahu_zones[i]
            cool_i = float(np.max(cooling[idx])) if idx.size else 0.0
            heat_i = float(np.max(heat_mag[idx])) if idx.size else 0.0
            need_i = bool(np.any(need_zone[idx])) if idx.size else False
            heat_by_ahu[i] = heat_i

            hold = self.ahu_hold[i]
            running = self.ahu_run[i]
            if running and need_i and cool_i >= heat_i:
                hold = int(p["COOL_HOLD_STEPS"])
            elif hold > 0:
                hold -= 1
            if not running and need_i:
                running = True
                self.ahu_on_steps[i] = 0
                if cool_i >= heat_i:
                    hold = int(p["COOL_HOLD_STEPS"])
            if running:
                self.ahu_on_steps[i] += 1
                if (
                    not need_i
                    and self.ahu_on_steps[i] >= int(p["MIN_ON_STEPS"])
                    and hold <= 0
                ):
                    running = False
                    self.ahu_on_steps[i] = 0
                    hold = 0
            self.ahu_run[i] = running
            self.ahu_hold[i] = hold

            sat_k, press_frac, recirc = self._ahu_setpoints(
                i, idx, cooling[idx], heat_i,
                float(np.max(np.abs(u_eff[idx]))) if idx.size else 0.0,
                predict, oat,
            )
            self._write_ahu(cont, disc, i, running, sat_k, press_frac, recirc)

        # -- hot water -------------------------------------------------------
        heat_demand = float(np.max(heat_mag[active])) if np.any(active) else 0.0
        water_on = heat_demand > float(p["NEED_U"])
        shortfall = float(np.max(
            np.clip(target - predict, 0.0, None)[active]
        )) if np.any(active) else 0.0
        demand_frac = float(np.clip(shortfall / max(0.1, float(p["WATER_RESET_FULL_K"])), 0.0, 1.0))
        if self.n_ahu:
            effort = float(np.quantile(heat_by_ahu, float(p["WATER_EFFORT_QUANT"])))
        else:
            effort = 0.0
        weather_frac = 0.0
        if self.oat_ema is not None and self.oat_ema < float(p["WATER_COLD_EMA_K"]):
            weather_frac = float(p["WATER_COLD_FRAC"])
        frac = float(np.clip(
            max(demand_frac, float(p["WATER_EFFORT_WEIGHT"]) * effort, weather_frac),
            0.0, 1.0,
        ))
        water_k = float(p["WATER_MIN_K"]) + frac * (
            float(p["WATER_MAX_K"]) - float(p["WATER_MIN_K"])
        )
        dp_min = float(p["DP_MIN_FRAC"])
        dp_frac = float(np.clip(
            dp_min + (1.0 - dp_min) * heat_demand * float(p["DP_GAIN"]), dp_min, 1.0
        ))
        if not water_on:
            water_k = float(p["WATER_MIN_K"])
            dp_frac = float(p["DP_MIN_FRAC"])
        if self.water_temp_idx >= 0:
            cont[self.water_temp_idx] = self._norm(water_k, self.rng_water)
        if self.water_dp_idx >= 0:
            dp_lo, dp_hi = self.rng_dp
            cont[self.water_dp_idx] = self._norm(
                dp_lo + dp_frac * (dp_hi - dp_lo), self.rng_dp
            )
        if 0 <= self.water_run_idx < self.n_disc:
            disc[self.water_run_idx] = 1 if water_on else 0

        # -- state update ----------------------------------------------------
        self.integ = integ
        self.u = u_eff
        self.u_ema += (u_eff - self.u_ema) / max(1.0, float(p["ZONE_EMA_STEPS"]))
        if self.t_prev is not None and self.t_prev.shape == temp.shape:
            rate = (temp - self.t_prev) / dt
            alpha = 1.0 / max(1.0, float(p["RATE_EMA_STEPS"]))
            warming = (self.u < -0.05) & (rate > 0.0)
            cooling_now = (self.u > 0.05) & (rate < 0.0)
            self.rate_heat = np.where(
                warming,
                self.rate_heat + alpha * (rate - self.rate_heat),
                self.rate_heat,
            )
            self.rate_cool = np.where(
                cooling_now,
                self.rate_cool + alpha * (-rate - self.rate_cool),
                self.rate_cool,
            )
            self.rate_heat = np.clip(
                self.rate_heat, float(p["RATE_MIN_K_PER_H"]), float(p["RATE_MAX_K_PER_H"])
            )
            self.rate_cool = np.clip(
                self.rate_cool, float(p["RATE_MIN_K_PER_H"]), float(p["RATE_MAX_K_PER_H"])
            )
        self.t_prev = temp.copy()
        self.diag = {
            "hour": hour,
            "day": day,
            "conditioning": bool(conditioning),
            "prestart": bool(prestart),
            "lead_h": float(lead),
            "hours_left": float(hours_left),
            "water_on": bool(water_on),
            "n_run": int(sum(1 for r in self.ahu_run if r)),
            "max_error": float(np.max(np.abs(err))) if self.n_zones else 0.0,
            "floor_guard": bool(np.any(guard)),
            "ceiling_guard": bool(np.any(ceil_guard)),
        }
        return self._finalize(cont, disc)

    # ---------------------------------------------------------------- helpers
    def _outside_temperature(self, obs) -> float:
        values = []
        for idx in self.oat_idx:
            if idx >= 0:
                value = float(obs[idx])
                if np.isfinite(value):
                    values.append(value)
        if not values:
            return 292.0
        oat = float(np.mean(values))
        if oat < 150.0:  # a standardized reading slipped through a name miss
            oat = oat * 10.0 + 295.0
        return float(np.clip(oat, 200.0, 340.0))

    def _band_axis(self, obs, indices, default: float):
        """Read one live setpoint sensor per zone (raw kelvin) with fallback."""
        out = np.full(self.n_zones, float(default))
        if not self.n_zones:
            return out
        safe = np.maximum(indices, 0)
        values = obs[safe]
        valid = (indices >= 0) & (values > 150.0) & (values < 450.0)
        return np.where(valid, values, default)

    def _hours_to_comfort(self, hour: float, day: int) -> tuple[float, bool]:
        """Hours until the next comfort-mode start and its weekday-ness."""
        start = self.morning_start
        if hour < start:
            return start - hour, day < 5
        return 24.0 - hour + start, ((day + 1) % 7) < 5

    def _ahu_setpoints(self, i, idx, cooling, heat_i, u_abs, predict, oat):
        """Supply-air temperature, static pressure fraction and recirculation."""
        p = self.p
        if idx.size == 0:
            return float(p["SAT_MAX_K"]), float(p["PRESS_MIN_FRAC"]), 1.0

        cool_i = float(np.max(cooling)) if cooling.size else 0.0
        # A quantile of the zones' cooling demand, so one outlier zone cannot
        # set the box for all of them; 0.5 is the median.
        median_cool = float(np.quantile(
            np.clip(cooling, 0.0, 1.0),
            float(np.clip(p["DEMAND_QUANTILE"], 0.0, 1.0)),
        ))
        frac = float(np.clip(median_cool / max(1e-6, float(p["DEMAND_FULL"])), 0.0, 1.0))
        sat_lo, sat_hi = self.sat_rng[i]
        sat_k = sat_hi - frac * (sat_hi - sat_lo)
        if heat_i > cool_i:
            sat_k = sat_hi  # heating-dominated: no chiller
        sat_k = float(np.clip(sat_k, sat_lo, sat_hi))

        deadband = float(p["PRESS_DEADBAND"])
        span = max(1e-6, 1.0 - deadband)
        press_frac = float(p["PRESS_MIN_FRAC"]) + float(p["PRESS_GAIN"]) * max(
            0.0, u_abs - deadband
        ) / span
        press_frac = float(np.clip(press_frac, float(p["PRESS_MIN_FRAC"]), float(p["PRESS_MAX_FRAC"])))
        press_pa = self.rng_pressure[0] + press_frac * (self.rng_pressure[1] - self.rng_pressure[0])

        recirc = 1.0
        if cool_i > float(p["NEED_U"]) and oat < 380.0:
            return_air = float(np.mean(predict[idx]))
            advantage = return_air - oat
            breakeven = max(float(p["ECON_MIN_ADV_K"]), press_pa * float(p["ECON_K_PER_PA"]))
            if advantage > breakeven and return_air > sat_k:
                outside_fraction = float(np.clip(
                    (return_air - sat_k) / max(1e-6, advantage), 0.0, 1.0
                ))
                recirc = float(np.clip(1.0 - outside_fraction, 0.0, 1.0))
        return sat_k, press_frac, recirc

    def _write_ahu(self, cont, disc, i, running, sat_k, press_frac, recirc) -> None:
        rng = self.rng_pressure
        if i < len(self.temp_idx) and self.temp_idx[i] >= 0:
            cont[self.temp_idx[i]] = self._norm(sat_k, self.sat_rng[i])
        if i < len(self.press_idx) and self.press_idx[i] >= 0:
            pa = rng[0] + press_frac * (rng[1] - rng[0])
            cont[self.press_idx[i]] = self._norm(pa, rng)
        if i < len(self.recirc_idx) and self.recirc_idx[i] >= 0:
            cont[self.recirc_idx[i]] = self._norm(recirc, self.rng_recirc)
        if i < self.n_disc:
            disc[i] = 1 if running else 0

    def _write_orphans(self, cont) -> None:
        """Zones without a temperature sensor: hold the ventilation minimum."""
        idle = 2.0 * float(self.p["MIN_DAMPER"]) - 1.0
        for damp, reheat in self._orphans:
            if damp is not None:
                cont[int(damp)] = idle
            if reheat is not None:
                cont[int(reheat)] = -1.0

    def _finalize(self, cont, disc):
        """Clip, sanitize and cast to the gateway contract."""
        cont = np.nan_to_num(
            np.clip(cont, -1.0, 1.0), nan=0.0, posinf=1.0, neginf=-1.0
        ).astype(np.float32)
        disc = np.where(np.asarray(disc) > 0, 1, 0).astype(np.int32)
        return {"continuous_action": cont, "discrete_action": disc}


def vavs_from(zone_actions):
    """Zone ids in a stable, numeric-suffix order."""
    return sorted(zone_actions.keys(), key=_numeric_suffix)
