"""CtrlV2 -- a building-agnostic structured HVAC controller (numpy only).

Everything is derived from ``building_meta_data`` and the live observation, so
one parameter vector controls 3-9 air handlers and 28-85 zones. The module is
deliberately self-contained (no ``smart_buildings`` import) so it can be pasted
into a Kaggle notebook.

What changed from CtrlV1, and why (all measured on full 8640-step episodes of
sims/shipped; v1's numbers are results/v1_dev30_shipped/)
-----------------------------------------------------------------------
The controller is the same shell as v1 -- per-zone PI on a target inside the
live band, one row per air handler, demand-based hot water -- with five
structural changes. The comfort term is what the score is made of, so every
trade-off below was settled by measurement, not by preference.

1. Readiness is a *deadline*, not a lead time.
   v1 started pre-conditioning when the worst zone's own lead time came within
   8 h, and it aimed at the *eco* band until the comfort window opened, so the
   first occupied hour always arrived with every zone 2-5 K short (on
   phoenix_5 hour 06 carried 0.049 of the 0.0104 comfort loss per step, more
   than half of the whole episode's). v2 computes, per zone, how far outside
   the comfort band that zone still has to travel and how fast it can travel
   there, and turns that into a *terminal band*: the band edge plus/minus the
   distance the plant can still close in the time that is left. Far from the
   deadline both edges are wide and the building floats; as the deadline
   approaches they converge on the comfort band, so the ramp starts exactly
   late enough to finish on time. A deadline that has already been missed
   switches the plant to full output on every zone that is still wrong.
   Because the ramp aims at the comfort band itself, the pre-start no longer
   spends the night heating to the eco band and then failing the morning.

2. The recovery rate is learned per zone *and* against the weather.
   ``rate = rate_estimate + slope * (zone - outside)``: the level is v1's EMA
   of the observed recovery rate (updated only while the plant is saturated,
   because a half-open damper says nothing about capacity) and the slope is a
   forgetting-factor least squares over the same samples. In deep cold the
   slope is strongly negative, which is exactly what makes the ramp start
   earlier there. Without enough samples the slope is zero and this is v1.

3. Pre-cooling exists, is model-timed, and is OFF by default.
   The same terminal band mirrored above the comfort band: the ceiling edge
   rises with the cooling rate and the hours left, so the building would be
   pulled down to the band only in the last few hours before the deadline, and
   never outside ``CEILING_WINDOW_H``. Measured on full episodes, it loses in
   every climate tested: on the ten mild/hot dev30 scenarios it *raised* the
   comfort loss five-fold (0.0010 -> 0.0055 per step) as well as the energy,
   because the supply air that cools one end of a building over-cools the other
   and the reheat coils then pay for it. ``CEILING_WINDOW_H = 0`` disables it,
   which is the default; the mechanism is kept because it is the only thing
   that helps the first occupied hour of a hot episode and the trade may yet
   come out positive with a supply-air law that does not fight itself.

4. Zone airflow is budgeted; the valve is not.
   v1 drove the damper from the same effort as the valve, so a recovery -- every
   zone 5-9 K short at once -- opened every damper. Every cubic metre per
   second of supply air is 30% outside air (the simulator clamps recirculation
   at 0.7) that the reheat coils must lift from the weather to 308 K, and the
   coil heat is capped by the pump, so past the point where the coils can carry
   the air to the discharge limit extra air *dilutes*: on fairbanks_1 at -40 C
   v1 held 38 m3/s and 0.9 MW of gas and the air still arrived colder than the
   zones. v2 caps the mean heating damper of each handler at HEAT_FLOW_FRAC of
   its design airflow and keeps the deficit-proportional split, which is worth
   20% of the weekend gas on fairbanks_1 (weekend bucket -137 -> -47).

5. Static pressure is trimmed, not pinned -- but rarely needed.
   Pressure lifts the flow of every damper at once (flow goes as sqrt(P)) and
   the fan is billed linearly in pressure, so v2 raises it only when the mean
   damper of a handler is already past PRESS_OPEN_AT, responds immediately and
   releases slowly. Measured: the dampers alone carry the cooling load in every
   climate tested, so the ceiling is PRESS_COOL_MAX; pushing it further bought
   comfort worth 0.012 for energy worth 0.021. Heating never raises it.

6. Deep cold also lowers the float floor and loads the coil properly.
   The floor guard is now climate-adaptive: FLOOR_TEMP_K plus up to
   FLOOR_ADAPT_K K as the outside air falls below FLOOR_OAT_REF_K, because a
   mild night needs no floor and holding one costs gas every hour of it
   (measured on atlanta_2). The hot-water loop is enabled by any real heating
   drive rather than at NEED_U, which had silently starved the guard, and the
   supply temperature is reset by the worst-zone shortfall as before.

Also: an absent morning is detected. ``comfort_mode_soon`` is one hour ahead,
so a weekday morning whose last hour shows no comfort mode never starts; v1
pre-conditioned straight through such a holiday. That morning is removed from
the deadline (and from the pre-start) for the rest of the day.

Known weaknesses (measured)
--------------------------
* Holidays are only detected one hour ahead, so the night before a holiday is
  still spent pre-conditioning the deadline of a morning that will not come.
  The recovered floor keeps the cost to a few hours of plant.
* Per-zone occupancy is unobservable, so every zone is held and the comfort
  budget cannot be spent where the people actually are.
* Buildings whose zone spread is large (10 K between a sunny and a shaded
  zone, as on phoenix_5) cannot have every zone in band at once; the supply
  temperature then has to choose between over-cooling the shaded zones and
  under-cooling the sunny ones. v2 lets the coldest heating zone set a floor
  under the supply temperature, which is a compromise, not a solution.
* The static-pressure response is capped at PRESS_COOL_MAX. Without the cap a
  zone 1 K too warm asked for the whole range and the fan bill tripled.

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
  HEAT_FLOW_FRAC        0.25    mean heating damper an air handler may run, as
                                a fraction of its design airflow (item 4)
Air handler -- one row per AHU
  NEED_U                0.12    effort below which a zone asks for no air
  NEED_K                0.20    band excursion that always counts as a demand
  MIN_ON_STEPS           6      minimum air-handler on time (5-minute steps)
  COOL_HOLD_STEPS      144      cooling-run latch (steps; 12 h)
  DEMAND_FULL           0.5     cooling demand that earns the coldest supply air
  DEMAND_QUANTILE       0.5     quantile of the zones' demand that sets it
  SAT_MAX_K           305.0     warmest supply-air setpoint (no chiller, K)
  SAT_CRIT_WEIGHT       0.6     weight of the critical zone in the supply reset
  SAT_HEAT_FLOOR_K      0.5     the supply stays this far above the coldest zone
                                that is asking for heat
  SAT_EMA_STEPS        12.0     time constant of the supply-air reset (steps)
  PRESS_MIN_FRAC        0.10    static pressure when a box runs idle (of range)
  PRESS_MAX_FRAC        1.00    static pressure cap
  PRESS_DEADBAND        0.80    effort the pressure ignores
  PRESS_RESPOND_K       1.0     unused by the damper-saturation request; kept as
                                the scale of the effort term
  PRESS_TRIM_STEP       0.01    pressure released per step when unrequested
  PRESS_MIN_ON_K        0.20    excursion a request must exceed
  PRESS_OPEN_AT         0.8     mean cooling damper that counts as a request
  PRESS_COOL_MAX        0.12    cap on the pressure the cooling request earns
  PRESS_PRECOOL_KEEP    0.9     fraction of the range the pre-cool may suppress
  ECON_MIN_ADV_K        0.50    outside-vs-return advantage needed to economize
  ECON_K_PER_PA     0.00322     outside-air advantage needed per pascal of fan
Hot water -- building-wide
  WATER_MIN_K         310.0     supply-water setpoint when the loop is idle
  WATER_MAX_K         350.0     supply-water setpoint at full demand
  WATER_RESET_FULL_K    4.0     worst-zone shortfall that earns full water
  WATER_EFFORT_WEIGHT   1.0     weight on the air-side saturation term
  WATER_EFFORT_QUANT 0.25       quantile over AHUs of the heating effort
  WATER_ON_U           0.02     heating drive that keeps the water loop on
  DP_MIN_FRAC           0.15    pump differential pressure when idle (of range)
  DP_GAIN               1.0     pressure rise per unit of reheat demand
  COLD_EMA_STEPS      288.0     time constant of the outside-air EMA (steps)
  WATER_COLD_EMA_K   258.15     outside-air EMA below which the weather term
                                applies
  WATER_COLD_FRAC       0.0     water fraction that weather term forces. Zero by
                                default: the demand and saturation terms carry
                                the weather in through the plant
Unoccupied -- deadline ramp, climate-adaptive float, holiday detection
  HEAT_MARGIN_K         0.6     the heating ramp aims this far inside the band
  COOL_MARGIN_K         1.2     the pre-cool aims this far inside the band
  PRESTART_MAX_H        8.0     the ramp model never looks further than this
  PRESTART_SAFETY       1.2     safety factor on the projected ramp
  HEAT_LEAD_MARGIN_H    1.0     the heating ramp finishes this early
  COOL_LEAD_MARGIN_H    3.0     the pre-cool finishes this early
  FLOOR_TEMP_K        285.0     minimum zone temperature the guard defends
  FLOOR_ADAPT_K         3.0     extra floor earned by deep cold
  FLOOR_OAT_REF_K     278.0     outside air above which the extra floor is zero
  FLOOR_OAT_SPAN_K     20.0    kelvin of outside air per unit of extra floor
  FLOOR_SLACK_K         9.0     floor never sits more than this below the band
  FLOOR_BAND_K          0.5     the guard aims this far above its floor
  FLOOR_PBAND_K         1.0     shortfall at which the guard is at full effort
  CEILING_MAX_K       299.0     absolute ceiling the pre-cool may defend
  CEILING_SLACK_K       8.0     ceiling never sits more than this above the band
  CEILING_RAMP_H        3.0     hours of cooling the ceiling ramp assumes
  CEILING_WINDOW_H      0.0     hours before the deadline the pre-cool may
                                run; 0 disables it (measured best)
  CEILING_PBAND_K       2.0     excess at which the ceiling guard is at full
  CEILING_GAIN          1.0     ceiling-guard effort scale
  RECOVERY_PRIOR_K_PER_H 1.5    prior heating recovery rate (K/h)
  COOL_RATE_PRIOR_K_PER_H 1.5   prior cooling recovery rate (K/h)
  RATE_EMA_STEPS       144.0    time constant of the recovery-rate estimate
  RATE_MIN_K_PER_H      0.15    lower clamp on a learned rate
  RATE_MAX_K_PER_H      3.0     upper clamp on a learned rate
  RATE_SLOPE_CLIP       0.25    clamp on d(rate)/d(zone - outside)
  RATE_FIT_STEPS      576.0     forgetting time constant of that fit
  RATE_FIT_MIN_W       24.0     fit weight at which the slope is fully used
  SPRINT_H              1.5     deadline window for the full-capacity sprint
  SPRINT_TOL_K          0.25    deviation that counts as out of band
  GUARD_NEED_U          0.02    drive that keeps an air handler alive
  OCC_EPS               0.01    occupancy at or below which the building is empty
Timing
  DT_H              1.0/12.0    control interval in hours (5 minutes)
"""

from __future__ import annotations

import os

import numpy as np

DEFAULTS = {
    # Zone loop: proportional allocation on the two comfort-band edges.
    "KP": 0.08,
    "KI": 0.06,
    "INTEGRAL_LIMIT_KH": 8.0,
    "MARGIN_K": 0.50,
    "DAMPER_U0": 0.10,
    "HEAT_SEQ_SPLIT": 0.10,
    "REHEAT_FULL_U": 1.0,
    "ZONE_SHIFT_MAX_K": 0.60,
    "ZONE_SHIFT_MAX_FRAC": 0.40,
    "ZONE_SHIFT_FULL_U": 0.20,
    "ZONE_EMA_STEPS": 144.0,
    "ZONE_EMA_RAMP_STEPS": 288.0,
    "OAT_NEUTRAL_K": 292.0,
    "OAT_SPAN_K": 8.0,
    "FLOOR_PBAND_K": 1.0,
    "CEILING_PBAND_K": 2.0,
    "CEILING_GAIN": 1.0,
    "LAG_ALPHA": 1.0,
    "LAG_MAX_K": 1.0,
    "MIN_DAMPER": 0.10,
    "DAMPER_GAIN": 1.0,
    "HEAT_FLOW_FRAC": 0.25,
    "HEAT_DAMPER_GAIN": 1.0,
    # Air handler.
    "NEED_U": 0.12,
    "NEED_K": 0.20,
    "MIN_ON_STEPS": 6,
    "COOL_HOLD_STEPS": 144,
    "DEMAND_FULL": 0.5,
    "DEMAND_QUANTILE": 0.5,
    "SAT_MAX_K": 305.0,
    "PRESS_MIN_FRAC": 0.10,
    "PRESS_MAX_FRAC": 1.0,
    "PRESS_DEADBAND": 0.80,
    "PRESS_RESPOND_K": 1.0,
    "PRESS_TRIM_STEP": 0.01,
    "PRESS_MIN_ON_K": 0.20,
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
    "WATER_ON_U": 0.02,
    "GUARD_NEED_U": 0.02,
    # Deadline-driven readiness (replaces v1's floor/ceiling guards and the
    # optimal-start window).
    "HEAT_MARGIN_K": 0.6,
    "COOL_MARGIN_K": 1.2,
    "PRESTART_MAX_H": 8.0,
    "PRESTART_SAFETY": 1.2,
    "FLOOR_TEMP_K": 285.0,
    "FLOOR_SLACK_K": 9.0,
    "CEILING_SLACK_K": 8.0,
    "CEILING_MAX_K": 299.0,
    "CEILING_RAMP_H": 3.0,
    "CEILING_WINDOW_H": 0.0,
    "FLOOR_OAT_REF_K": 278.0,
    "FLOOR_OAT_SPAN_K": 20.0,
    "FLOOR_ADAPT_K": 3.0,
    "HEAT_LEAD_MARGIN_H": 1.0,
    "COOL_LEAD_MARGIN_H": 3.0,
    "PRESS_PRECOOL_KEEP": 0.9,
    "PRESS_COOL_MAX": 0.12,
    "PRESS_OPEN_AT": 0.8,
    "SAT_CRIT_WEIGHT": 0.6,
    "SAT_HEAT_FLOOR_K": 0.5,
    "SAT_EMA_STEPS": 12.0,
    "SPRINT_H": 1.5,
    "SPRINT_TOL_K": 0.25,
    "RECOVERY_PRIOR_K_PER_H": 1.5,
    "COOL_RATE_PRIOR_K_PER_H": 1.5,
    "RATE_EMA_STEPS": 144.0,
    "RATE_MIN_K_PER_H": 0.15,
    "RATE_MAX_K_PER_H": 3.0,
    "RATE_SLOPE_CLIP": 0.25,
    "RATE_FIT_STEPS": 576.0,
    "RATE_FIT_MIN_W": 24.0,
    # Occupancy.
    "OCC_EPS": 0.01,
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


class CtrlV2:
    """Deadline-driven readiness, trim-and-respond pressure, adaptive floor."""

    PARAMS = dict(DEFAULTS)
    # Set CTRLV2_DEBUG=1 to surface internal errors instead of the fallback.
    DEBUG = bool(os.environ.get("CTRLV2_DEBUG"))

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
        # v2: static-pressure trim state, one fraction per air handler.
        self.press = np.full(max(self.n_ahu, 1), float(self.p["PRESS_MIN_FRAC"]))
        # v2: filtered supply-air cooling demand, one per air handler.
        self.sat_frac = np.zeros(max(self.n_ahu, 1))
        # v2: online slope of the recovery rate against (zone - outside) air,
        # one regression per zone and direction, plus the time bookkeeping that
        # turns the decoded clock into a deadline.
        self._fit = {
            "heat": np.zeros((5, n)),
            "cool": np.zeros((5, n)),
        }
        self._prev_hour = None
        self._mornings = 0
        self._holidays: set[int] = set()
        self._holiday_seen = False

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

        # -- deadline ------------------------------------------------------
        # The decoded clock has no day of year, so mornings are counted: every
        # crossing of the morning start hour advances the counter, and the
        # deadline is the next crossing that the schedule will actually turn
        # comfort mode on. A morning whose last hour shows no comfort mode
        # ahead is a holiday and is skipped (see _track_clock).
        self._track_clock(hour, day, comfort_now, comfort_soon)
        hours_left, next_is_workday = self._hours_to_comfort(hour, day)
        if comfort_now:
            # Comfort can begin at any step while the window is open, so the
            # deadline is now.
            hours_left = 0.0
        occupied = comfort_now or occupants > float(p["OCC_EPS"])
        need_k = float(p["NEED_K"])

        # -- recovery rates and the terminal band ---------------------------
        # Every zone must be inside the comfort band by the deadline, so the
        # allowed band is what this zone's own recovery rate can still close in
        # the time that is left. Far from the deadline both edges are wide and
        # the building floats; as hours_left shrinks the edges converge on the
        # comfort band, so the ramp starts exactly late enough to finish on
        # time, and deep cold (slower rate) starts it earlier. This replaces
        # v1's floor guard, its ceiling guard and its optimal-start window.
        rate_heat, rate_cool = self._rates(predict, oat)
        c_lo, c_hi = self.sched_comfort
        h_heat = np.clip(
            hours_left - float(p["HEAT_LEAD_MARGIN_H"]), 0.0, float(p["PRESTART_MAX_H"])
        )
        h_cool = np.clip(
            hours_left - float(p["COOL_LEAD_MARGIN_H"]), 0.0, float(p["CEILING_RAMP_H"])
        )
        safety = float(p["PRESTART_SAFETY"])
        # The float floor is climate-adaptive: a mild night needs nothing, and
        # holding a building 3 K warmer than it would float all night costs gas
        # every hour of it (measured on atlanta_2: nearly all of v2's first
        # energy regression came from a fixed 288 K floor where v1 held 285.5).
        # Deep cold is the opposite case -- there the floor has to be high
        # enough that the recovery can still finish by the deadline.
        warmth = np.clip(
            (float(p["FLOOR_OAT_REF_K"]) - oat) / max(1.0, float(p["FLOOR_OAT_SPAN_K"])),
            0.0, 1.0,
        )
        floor_lo = float(p["FLOOR_TEMP_K"]) + float(p["FLOOR_ADAPT_K"]) * float(warmth)
        floor_lo = max(floor_lo, c_lo - float(p["FLOOR_SLACK_K"]))
        ceil_hi = min(float(p["CEILING_MAX_K"]), c_hi + float(p["CEILING_SLACK_K"]))
        m_heat = float(p["HEAT_MARGIN_K"])
        m_cool = float(p["COOL_MARGIN_K"])
        floor_target = np.clip(
            c_lo + m_heat - rate_heat * h_heat / safety, floor_lo, c_lo + m_heat
        )
        ceiling_target = np.clip(
            c_hi - m_cool + rate_cool * h_cool / safety, c_hi - m_cool, ceil_hi
        )
        if hours_left > float(p["CEILING_WINDOW_H"]):
            # No pre-cooling days ahead: v1 measured a standing night ceiling at
            # 2-4x what it saves, and the same trap was worth 0.003 per step in
            # phoenix_5 when the ramp ran from the weekend.
            ceiling_target = np.full(self.n_zones, 400.0)

        # The guard reads the measured temperature for its threshold and the
        # extrapolation only for the drive: on a ramp the two guards otherwise
        # fire against each other (the cooling zones all extrapolate downward).
        floor_short = np.clip(floor_target - predict, 0.0, None)
        guard = temp < floor_target
        ceiling_excess = np.clip(predict - ceiling_target, 0.0, None)
        ceil_guard = temp > ceiling_target

        # -- last-chance sprint ---------------------------------------------
        # Inside the deadline window with zones still outside the live comfort
        # band, the plant runs flat out: every actuator at its most aggressive
        # setting, because an occupant arriving into a 10 K excursion costs far
        # more than any amount of plant power.
        tol = float(p["SPRINT_TOL_K"])
        # Measured, not extrapolated: during a pre-cool every zone has a
        # negative slope, and the one-step lead then makes the whole building
        # look 1 K colder than it is. With the sprint reading that, it heated
        # while the chiller was cooling and burned 1.1 MW of gas in 20 minutes.
        sprint_mask = (temp < c_lo - tol) | (temp > c_hi + tol)
        # The sprint is a pre-occupancy mechanism only. A building with a wide
        # spread across its zones -- one side in sun, one in shade -- can never
        # have every zone in band at once, and a sprint that fires on any
        # excursion would then hold the plant at full output all day: measured
        # on phoenix_5, that cost 0.34 of the comfort term per occupied step and
        # more than twice v1's energy, because the airflow it forced in was
        # 30%-hot outside air that the chiller then had to remove.
        sprint = bool(
            not occupied
            and hours_left <= float(p["SPRINT_H"])
            and np.any(sprint_mask)
        )
        sprint_heat = sprint_mask & (temp < c_lo)
        sprint_cool = sprint_mask & (temp > c_hi)

        conditioning = bool(
            occupied or np.any(guard) or np.any(ceil_guard) or sprint
        )

        # -- per-zone control -----------------------------------------------
        # Occupied: a PI on a target inside the live band, exactly as v1 (the
        # integral is what removes the standing offset: a purely proportional
        # law held phoenix_5 about 1 K outside the band all day, three times
        # v1's comfort loss). Unoccupied: the terminal band is the drive.
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

        if occupied:
            u_eff = u_raw
            active = np.full(self.n_zones, True, dtype=bool)
        else:
            u_eff = np.zeros(self.n_zones)
            if np.any(guard):
                drive = np.clip(floor_short / max(1e-3, float(p["FLOOR_PBAND_K"])), 0.0, 1.0)
                u_eff = np.where(guard, -drive, 0.0)
            if np.any(ceil_guard):
                drive = np.clip(
                    ceiling_excess / max(1e-3, float(p["CEILING_PBAND_K"])), 0.0, 1.0
                ) * float(p["CEILING_GAIN"])
                u_eff = np.where(ceil_guard, drive, u_eff)
            active = guard | ceil_guard
            integ = np.where(active, 0.0, integ)
        if sprint:
            # Deadline missed and the zone is still outside the band: flat out
            # in the direction it is wrong.
            u_eff = np.where(sprint_heat, -1.0, u_eff)
            u_eff = np.where(sprint_cool, 1.0, u_eff)
            active = active | sprint_mask

        # -- zone actuators -------------------------------------------------
        # v1's sequence, with the air side budgeted: the valve leads the damper
        # in heating (the coil's discharge is capped, so extra air above the
        # floor buys heat only at the price of cold-air ingress), and the
        # heating dampers of a handler are scaled to a mean that its water flow
        # can actually heat.
        cooling = np.clip(u_eff, 0.0, 1.0)
        heat_mag = np.clip(-u_eff, 0.0, 1.0)
        heat_gain = float(p["HEAT_DAMPER_GAIN"])
        min_damper = float(p["MIN_DAMPER"])
        u0 = float(p["DAMPER_U0"])
        cool_open = np.clip((cooling - u0) / max(1e-6, 1.0 - u0), 0.0, 1.0)
        split = float(p["HEAT_SEQ_SPLIT"])
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
        u_abs = np.maximum(heat_mag, cooling)
        heating = (u_eff < 0.0) & active

        # -- air handlers ----------------------------------------------------
        need_u = float(p["NEED_U"])
        guard_u = float(p["GUARD_NEED_U"])
        need_zone = active & (
            (np.abs(u_eff) > np.where(occupied, need_u, guard_u))
            | (predict > hi + need_k)
            | (predict < lo - need_k)
        )
        cont = np.full(self.n_cont, 2.0 * min_damper - 1.0, dtype=np.float64)
        disc = np.zeros(self.n_disc, dtype=np.int32)
        rht_ok = self.z_reheat >= 0
        if np.any(rht_ok):
            cont[self.z_reheat[rht_ok]] = 2.0 * reheat[rht_ok] - 1.0
        self._write_orphans(cont)

        heat_by_ahu = np.zeros(max(self.n_ahu, 1))
        # v2: per-air-handler cooling request. The static pressure answers how
        # *much* of the box is above its band, not how far one zone is: a
        # single hot zone gets its damper opened and a colder supply, while ten
        # hot zones need the fan. Raised pressure is expensive and, in a mixed
        # building, it pushes minimum-damper air into the zones that are already
        # too cold, so the response is scaled by the mean demand.
        heat_flow_budget = float(p["HEAT_FLOW_FRAC"])
        for i in range(self.n_ahu):
            idx = self.ahu_zones[i]
            cool_i = float(np.mean(cooling[idx])) if idx.size else 0.0
            heat_i = float(np.mean(heat_mag[idx])) if idx.size else 0.0
            need_i = bool(np.any(need_zone[idx])) if idx.size else False
            heat_by_ahu[i] = heat_i
            if idx.size:
                # A request only exists once the dampers are nearly wide open:
                # static pressure is the last resort, because the fan is billed
                # linearly in pressure and the extra airflow drags in the
                # weather (30% outside air) that the coil then has to remove.
                cool_req = float(np.clip(
                    cool_i / max(1e-3, float(p["PRESS_OPEN_AT"])), 0.0, 1.0
                ))
                if sprint:
                    cool_req = max(cool_req, float(np.any(sprint_cool[idx])))
                # Heating airflow budget: the mean heating damper a handler may
                # run, i.e. the fraction of its design airflow.
                heat_zone = heating[idx]
                total_heat = float(np.sum(heat_air[idx][heat_zone])) if heat_zone.any() else 0.0
                allowed = heat_flow_budget * float(idx.size)
                if total_heat > allowed > 0.0:
                    scale = allowed / total_heat
                    damper[idx] = np.where(
                        heat_zone,
                        min_damper + (damper[idx] - min_damper) * scale,
                        damper[idx],
                    )
            else:
                cool_req = 0.0

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
                predict, oat, cool_req, occupied, heating[idx],
            )
            self._write_ahu(cont, disc, i, running, sat_k, press_frac, recirc)

        # Zone dampers are written after the per-handler airflow budget has been
        # applied to them.
        damp_ok = self.z_damp >= 0
        if np.any(damp_ok):
            cont[self.z_damp[damp_ok]] = 2.0 * damper[damp_ok] - 1.0

        # -- hot water -------------------------------------------------------
        # Any real heating drive needs the loop on: v1 gated this at NEED_U,
        # which silently starved the floor guard (drives of 0.1) of hot water.
        water_u = float(p["WATER_ON_U"])
        heating_zones = heat_mag > water_u
        heat_demand = float(np.max(heat_mag[heating_zones])) if np.any(heating_zones) else 0.0
        water_on = bool(np.any(heating_zones))
        heat_target = np.where(occupied, target, floor_target)
        shortfall = float(np.max(
            np.clip(heat_target - predict, 0.0, None)[heating_zones]
        )) if np.any(heating_zones) else 0.0
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
        if sprint:
            frac = 1.0
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
        if self.t_prev is not None and self.t_prev.shape == temp.shape:
            rate = (temp - self.t_prev) / dt
            alpha = 1.0 / max(1.0, float(p["RATE_EMA_STEPS"]))
            # Only a saturated plant says what a *recovery* costs in time. A
            # half-open damper warming a zone is a control action, not a
            # capacity, and using it made the deadline model claim recoveries
            # that the plant cannot deliver.
            warming = (self.u < -0.6) & (rate > 0.0)
            cooling_now = (self.u > 0.6) & (rate < 0.0)
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
            # v2: which way the recovery rate depends on how far the zone is
            # from the weather. Only samples where the plant is really working
            # are used, since a floating zone says nothing about recovery.
            contrast = 0.5 * (temp + self.t_prev) - oat
            self._fit_update("heat", self.u < -0.3, contrast, rate, dt)
            self._fit_update("cool", self.u > 0.3, contrast, -rate, dt)
        self.integ = integ
        self.u = u_eff
        self.u_ema += (u_eff - self.u_ema) / max(1.0, float(p["ZONE_EMA_STEPS"]))
        self.t_prev = temp.copy()
        self.diag = {
            "hour": hour,
            "day": day,
            "conditioning": bool(conditioning),
            "prestart": bool(not occupied and conditioning),
            "hours_left": float(hours_left),
            "floor_target": float(np.min(floor_target)) if self.n_zones else 0.0,
            "ceiling_target": float(np.max(ceiling_target)) if self.n_zones else 0.0,
            "rate_heat": float(np.median(rate_heat)) if self.n_zones else 0.0,
            "rate_cool": float(np.median(rate_cool)) if self.n_zones else 0.0,
            "sprint": sprint,
            "press": [round(float(v), 3) for v in self.press],
            "water_on": bool(water_on),
            "water_k": float(water_k),
            "n_run": int(sum(1 for r in self.ahu_run if r)),
            "max_error": float(np.max(np.abs(err))) if self.n_zones else 0.0,
            "u_raw": [round(float(v), 2) for v in u_raw[:6]],
            "integ": [round(float(v), 2) for v in integ[:6]],
            "err": [round(float(v), 2) for v in err[:6]],
            "u_eff": [round(float(v), 2) for v in u_eff[:6]],
            "heat_mag": [round(float(v), 2) for v in heat_mag[:6]],
            "occupied": bool(occupied),
            "floor_guard": bool(np.any(guard)),
            "ceiling_guard": bool(np.any(ceil_guard)),
            "mornings": int(self._mornings),
            "holiday": bool(self._mornings in self._holidays),
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

    def _track_clock(self, hour: float, day: int, comfort_now: bool, comfort_soon: bool) -> None:
        """Counts mornings and flags the ones that will not start comfort mode.

        The observation carries only the time of day and a 1/7 day-of-week
        step, so the number of morning starts seen so far is the only absolute
        time reference available. ``comfort_mode_soon`` is one hour ahead: a
        weekday morning that reaches its last hour without it is a holiday, and
        the deadline moves to the next working morning.
        """
        start = self.morning_start
        prev = self._prev_hour
        self._prev_hour = hour
        if prev is None:
            return
        if prev <= start < hour or prev > hour:  # crossing forward, or a reset
            self._mornings += 1
        if (start - 1.0) <= hour < start and not comfort_soon and not comfort_now:
            if day < 5:
                self._holidays.add(self._mornings)

    def _rates(self, predict, oat):
        """Per-zone recovery rates at this zone/outside contrast.

        The base level is v1's online EMA of the observed recovery rate; the
        fitted slope moves it with (zone - outside) air, because the same plant
        recovers a zone much faster in mild weather than at -40 C. Without
        enough samples the slope is zero and this is exactly v1's estimate.
        """
        p = self.p
        x = predict - oat
        clip = float(p["RATE_SLOPE_CLIP"])
        out = []
        for key, base in (("heat", self.rate_heat), ("cool", self.rate_cool)):
            w, sx, sy, sxx, sxy = self._fit[key]
            safe_w = np.maximum(w, 1e-6)
            blend = np.clip(w / max(1.0, float(p["RATE_FIT_MIN_W"])), 0.0, 1.0)
            x_mean = np.where(w > 0.5, sx / safe_w, x)
            sxx_c = sxx - sx * sx / safe_w
            sxy_c = sxy - sx * sy / safe_w
            slope = np.where(sxx_c > 1e-3, sxy_c / np.maximum(sxx_c, 1e-3), 0.0)
            slope = np.clip(slope, -clip, clip)
            rate = base + blend * slope * (x - x_mean)
            out.append(np.clip(
                rate, float(p["RATE_MIN_K_PER_H"]), float(p["RATE_MAX_K_PER_H"])
            ))
        return out[0], out[1]

    def _fit_update(self, key: str, mask, x, y, dt: float) -> None:
        """Forgetting-factor least squares of rate against (zone - outside)."""
        p = self.p
        tau = max(1.0, float(p["RATE_FIT_STEPS"]))
        decay = float(np.exp(-dt / tau))
        f = self._fit[key]
        f *= decay
        m = np.asarray(mask, dtype=np.float64)
        f[0] += m
        f[1] += m * x
        f[2] += m * y
        f[3] += m * x * x
        f[4] += m * x * y

    def _hours_to_comfort(self, hour: float, day: int) -> tuple[float, bool]:
        """Hours until the next comfort-mode start that will actually happen.

        Mornings already known to be holidays (see ``_track_clock``) are
        skipped, and so is any morning that cannot start a comfort window.
        """
        start = self.morning_start
        if hour < start:
            hours = start - hour
            offset = 0
            index = self._mornings
        else:
            hours = 24.0 - hour + start
            offset = 1
            index = self._mornings + 1
        for _ in range(8):
            workday = ((day + offset) % 7) < 5
            if workday and index not in self._holidays:
                return hours, True
            offset += 1
            hours += 24.0
            index += 1
        return hours, False

    def _ahu_setpoints(self, i, idx, cooling, heat_i, u_abs, predict, oat,
                       cool_req=0.0, occupied=True, heating=None):
        """Supply-air temperature, static pressure fraction and recirculation."""
        p = self.p
        if idx.size == 0:
            return float(p["SAT_MAX_K"]), float(p["PRESS_MIN_FRAC"]), 1.0

        # Supply air: the critical zone sets how cold the box has to be, but a
        # single outlier would then freeze the building, so the median demand
        # also counts. Both are fractions of the demand scale.
        crit = float(np.max(cooling)) if cooling.size else 0.0
        mean_cool = float(np.mean(cooling)) if cooling.size else 0.0
        median_cool = float(np.quantile(
            np.clip(cooling, 0.0, 1.0),
            float(np.clip(p["DEMAND_QUANTILE"], 0.0, 1.0)),
        ))
        frac = float(np.clip(median_cool / max(1e-6, float(p["DEMAND_FULL"])), 0.0, 1.0))
        frac = max(frac, float(p["SAT_CRIT_WEIGHT"]) * float(np.clip(crit, 0.0, 1.0)))
        # Filter the supply-air reset. The band is 3 K wide and the supply
        # spans 20 K, so an instantaneous reset on the critical zone slams a
        # whole air handler from full cooling to full heating every few steps
        # and the zones swing 8 K in 20 minutes; measured on phoenix_5 that
        # cycle alone was 0.27 of the comfort term per occupied step.
        tau = max(1.0, float(p["SAT_EMA_STEPS"]))
        self.sat_frac[i] += (float(np.clip(frac, 0.0, 1.0)) - self.sat_frac[i]) / tau
        frac_s = float(self.sat_frac[i])
        sat_lo, sat_hi = self.sat_rng[i]
        sat_k = sat_hi - frac_s * (sat_hi - sat_lo)
        if heat_i > mean_cool and frac_s <= 0.02:
            sat_k = sat_hi  # heating-dominated: no chiller
        # A supply-air temperature below the temperature of a zone that is
        # asking for heat *cools* that zone: the reheat coil can only lift the
        # air to the 308 K discharge limit, so at a wide damper the air arrives
        # colder than the room. In a mixed building -- one side in sun, one in
        # shade -- the critical-zone supply temperature did exactly that and
        # froze the shaded zones 7 K below their band. The supply never goes
        # below the warmest zone that needs heat.
        # The coldest zone that wants heat sets the floor, not the warmest: with
        # the warmest, one slightly cool zone lifted the supply above every hot
        # zone in its handler and the handler stopped cooling at all.
        heat_zones = np.flatnonzero(np.asarray(heating, dtype=bool)) if heating is not None else None
        if heat_zones is not None and heat_zones.size:
            sat_k = max(sat_k, float(np.min(predict[heat_zones])) + float(p["SAT_HEAT_FLOOR_K"]))
        sat_k = float(np.clip(sat_k, sat_lo, sat_hi))

        # Static pressure: respond to an unmet request immediately, release it
        # slowly otherwise. v1 held it at the floor unless the effort passed
        # 0.8, which in a hot climate meant a zone 3 K too warm received a third
        # of its design airflow and could never be pulled back.
        deadband = float(p["PRESS_DEADBAND"])
        span = max(1e-6, 1.0 - deadband)
        press_min = float(p["PRESS_MIN_FRAC"])
        press_max = float(p["PRESS_MAX_FRAC"])
        request = max(
            float(np.clip(cool_req, 0.0, 1.0)),
            float(np.clip((u_abs - deadband) / span, 0.0, 1.0)),
        )
        if not occupied:
            # Pre-cooling keeps the pressure on the floor: the dampers of the
            # zones that are actually too warm carry it, and the zones at their
            # ventilation floor are not flooded with cold air.
            request = min(request, 1.0 - float(p["PRESS_PRECOOL_KEEP"]))
        else:
            request = min(request, float(p["PRESS_COOL_MAX"]))
        held = float(self.press[i]) if len(self.press) > i else press_min
        if request > 0.0:
            held = max(held, press_min + (press_max - press_min) * request)
        else:
            held = max(press_min, held - float(p["PRESS_TRIM_STEP"]))
        self.press[i] = held
        press_frac = float(np.clip(held, press_min, press_max))
        press_pa = self.rng_pressure[0] + press_frac * (self.rng_pressure[1] - self.rng_pressure[0])

        recirc = 1.0
        if mean_cool > float(p["NEED_U"]) and oat < 380.0:
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
