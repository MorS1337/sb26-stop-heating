"""CtrlW -- Wu's controller plus a supervisory layer, tunable, submission-safe.

Wu's `TransferPolicy`/`ResidualPolicy` (see `policies/wu_core.py`, kaggle
notebook `danielwuu/cpu-a-alpha075-control-20261022`, Apache-2.0) is the base.
It scores 0.00750 on dev30/shipped -- better than anything we had built
(zt-best 0.01042, our ctrl_v1 0.01483) -- so the base is kept bit-exact and
everything we add is a switch on top of it:

    CtrlW(PARAMS flags all off) == RefWu   (total -28.360879105 on
    building_train_1/atlanta_2, 8641 steps, shipped sim)

That equality is the contract: `_band` and `_energy_weights` return the
baseline's own numbers whenever their flag is off, so a disabled addition does
not even perturb floating-point summation order.

What the base gets wrong, and what each addition does about it
-------------------------------------------------------------
(a) `release_enabled` -- Wu reads `comfort_mode_now` (a *schedule* flag) but
    never `num_occupants`, so from 06:00 to 19:00 he holds the comfort band
    even with an empty building. Comfort regret is the occupancy-weighted mean
    shortfall, so an empty zone is free to drift: once `num_occupants` has been
    zero for `release_empty_steps` consecutive 5-minute steps and the clock is
    past `release_hour`, the zone bands fall back to the standby band -- the
    band Wu already applies at night, so recovery behaviour is unchanged --
    until the comfort mode ends. Any occupant re-arms the comfort band in the
    same step; a single person arriving at 15:00 into a cold building would
    otherwise be charged the full shortfall.

    Measured: exactly zero, on all 30 dev30 scenarios (identical totals to the
    last bit). `RandomizedArrivalDepartureOccupancy` puts arrivals in 06:00-12:00
    and departures in 13:00-19:00 per zone and `num_occupants` is the sum over
    zones, so the building is occupied from the first arrival to the last
    departure -- the count never returns to 0 inside the comfort window, and
    after `release_hour` the latch can never arm. The mechanism is kept because
    it is the correct guard (a probe with the occupancy test forced to "empty"
    is 6x worse, -3.80 against -0.62 over 600 steps: the first-arrival effect is
    real and expensive), and because a hidden set with a sparser occupancy
    draw would expose the opposite failure.

(b) `reward_weights_enabled` -- Wu's AHU search minimises

        fan + chiller + 0.30 * gas + comfort_cost * violation

    which prices 1 kW of gas heat at a third of 1 kW of electricity. The
    simulator's reward does not agree. With `normalized_energy_cost` and
    `normalized_carbon_emission` over `max_electricity_rate = 80 kW/AHU` and
    `max_natural_gas_rate = 205 kW/AHU`, and `natural_gas_heating_energy_rate`
    being coil heat, the marginal reward of one watt is

        e(h) = 0.2 * ( P_e(h)/D_e(h) + C_e(h)/D_c(h) )
        g(h) = 0.2 * ( P_g/D_e(h)   + C_g/D_c(h)     )
        D_e(h) = 80 kW*P_e(h) + 205 kW*P_g,   D_c(h) = 80 kW*C_e(h) + 205 kW*C_g

    with `P_e` the 16/18/20 c/kWh time-of-use schedule (weekday), `C_e` the
    87.8-104.7 kg/MWh hourly carbon, and `P_g`, `C_g` from
    `natural_gas_energy_cost.py` (monthly $/1000 ft3, `GAS_CO2 = 53.12
    kg/MMBtu`). Dividing by `e(12:00)` gives this module's two tables:

        ELEC_REL[h]  = e(h) / e(12)          0.930 ... 1.003
        GAS_RATIO[h] = g(h) / e(h)           0.443 ... 0.509, mean 0.481

    so gas is worth 1.6x what Wu assumes and the price ratio moves ~13% across
    the day. The objective becomes `w_e*(fan + chiller) + w_e*GAS_RATIO*gas`,
    which is the reward's own trade-off in Wu's units.

    Measured: 0.0075035 against 0.0075040 -- 0.007% better, i.e. the AHU
    search sits on a plateau in this direction. Gas fell 0.14 and energy rose
    0.05 in reward units, which is the trade the corrected price asks for, just
    a very small one. `gas_ratio_scale` and `gas_cost_ratio` are both in the
    tuning space, so CMA-ES can test a much larger correction than the derived
    one.

    Shipped vs corrected: `sims/corrected` bills the boiler as burner heat from
    an energy balance (draw + standing loss + charging) while `sims/shipped`
    bills `flow_heat + dissipation + tank_temperature_change`. Both are watts
    of gas heat through the same `NaturalGasEnergyCost`, so the *prices* -- and
    therefore this ratio -- are identical; only the amount attributed to a step
    differs, and neither changes the marginal ratio between fuels.

(c) `holiday_guard_enabled` -- pre-conditioning is armed only when today's
    schedule can still contain a comfort window: not a weekend (decoded from
    the `dow_sin_000`/`dow_cos_000` features), and either comfort mode has
    already been seen today or it was seen on the previous day. `SetpointSchedule
    .is_comfort_mode` returns False on a holiday, so on a weekday holiday a
    policy that preheats from the clock alone heats an empty building for
    nothing. Measured: exactly zero. Every shipped scenario leaves the
    schedule's `holidays` set empty and Wu's ramp is already gated by
    `comfort_mode_soon`, so there is nothing for the guard to suppress; it
    exists to keep (d)'s clock-driven ramp from firing on a day the schedule
    has declared non-comfort.

(d) `early_preheat_enabled` -- the one that pays. Wu's preheat ramp lives
    entirely inside `comfort_mode_soon` (`is_comfort_mode(now + 60 min)`), so
    `precondition_hours` above 1.0 has no effect at all: the ramp always spans
    one hour. Yet 25% of RefWu's dev30 loss is spent in the single hour 05-06
    (energy+carbon 0.356 of the 1.945 total) and the largest comfort loss of the
    whole day is hour 06 (0.050 of 0.055): the building reaches the band edge
    exactly as the occupants arrive. This flag lets the *same* `eco -> comfort`
    ramp start `early_preheat_hours` before the comfort window opens, using
    `morning_start_hour` from the metadata and the `hod_*` features already in
    the observation. The horizon is `min(early_preheat_hours,
    max(1, precondition_hours))`, so the ramp is continuous into Wu's own
    (he reuses `precondition_hours` in the same `1 - remaining/duration`
    formula). Dawn is the cheapest part of the day -- 16 c/kWh and 88.2 kg/MWh
    are both the daily minimum -- and the building is coldest then, so the ramp
    is also the cheapest way to hit the arrival:

        horizon   1 h (Wu)   2 h      3 h      4 h
        dev30    0.007504   0.007426  0.007491 0.007561

    A quadratic through the first three points puts the minimum at ~2.1 h. The
    whole gain is the arrival-hour comfort term: comfort loss falls from -55.4
    to -14.3 for ~21 units of extra energy+carbon.

Measured, dev30/shipped, full 8641-step episodes (results/ctrw_*/)
------------------------------------------------------------------
    RefWu                      0.0075040
    + (a) release              0.0075040   identical, no empty comfort hour
    + (b) reward weights       0.0075035   -0.007%
    + (c) holiday guard        0.0075040   identical
    + (d) preheat, 1 h (Wu)    0.0075040
    + (d) preheat, 2 h         0.0074263   -1.04%
    + (d) preheat, 3 h         0.0074906
    + (d) preheat, 4 h         0.0075614
    CtrlW.PARAMS (a+b+c+d 2 h) 0.0073694   -1.80%, better on 30 of 30
    same config, corrected sim 0.0071449

Nothing here reads a building id, a weather file, the occupancy seed or the
future; every quantity is either in the observation, in the public metadata, or
a constant of the published simulator.
"""

from __future__ import annotations

import math

import numpy as np

try:  # packaged notebook: the embedded modules are top-level names
    from wu_core import ResidualEnsemble, ResidualPolicy
    from wu_weights import members as _wu_members
except ImportError:  # workspace: policies/ is a package
    from .wu_core import ResidualEnsemble, ResidualPolicy
    from .wu_weights import members as _wu_members

# --------------------------------------------------------------------------- #
# Reward-consistent fuel weights (derivation in the module docstring).
# --------------------------------------------------------------------------- #
ELEC_REL = (
    0.93025, 0.92959, 0.92973, 0.92966, 0.92994, 0.93025, 0.95806, 0.96493,
    0.97265, 0.97566, 0.97659, 0.97720, 1.00000, 1.00131, 1.00239, 1.00331,
    0.99919, 0.99133, 0.98583, 0.93715, 0.93597, 0.93430, 0.93280, 0.93156,
)
GAS_RATIO = (
    0.50873, 0.50936, 0.50923, 0.50930, 0.50903, 0.50873, 0.48264, 0.47642,
    0.46955, 0.46689, 0.46607, 0.46554, 0.44603, 0.44493, 0.44404, 0.44327,
    0.44670, 0.45334, 0.45805, 0.50211, 0.50323, 0.50483, 0.50627, 0.50747,
)


class CtrlW(ResidualPolicy):
    """Wu's residual-corrected transfer policy with a supervised band/fuel layer."""

    name = "ctrl_w"

    PARAMS = {
        # --- Wu's TransferPolicy.DEFAULTS, unchanged -------------------------
        "comfort_margin": 0.12,
        "precondition_hours": 2.0,
        "recovery_steps": 1.5,
        "forgetting_factor": 0.997,
        "model_residual_gain": 0.16,
        "comfort_cost": 65000.0,
        "gas_cost_ratio": 0.30,
        "hot_water_target": 345.0,
        "standby_minimum": 282.0,
        "standby_maximum": 307.0,
        "observation_delay": 1,
        "response_scale": 0.75,
        "disturbance_rate": 0.35,
        # --- Wu's notebook residual guards -----------------------------------
        "residual_alpha": 0.75,
        "correction_limit": 0.25,
        "disagreement_limit": 0.15,
        # --- (a) occupancy-aware release -------------------------------------
        "release_enabled": True,
        "release_hour": 12.0,
        "release_empty_steps": 3,
        # --- (b) reward-consistent fuel weights ------------------------------
        "reward_weights_enabled": True,
        "gas_ratio_scale": 1.0,
        "elec_hour_blend": 1.0,
        # --- (c) holiday guard -----------------------------------------------
        "holiday_guard_enabled": True,
        # --- (d) extended preheat --------------------------------------------
        "early_preheat_enabled": True,
        "early_preheat_hours": 2.0,
    }
    WU_PARAMETERS = (
        "comfort_margin",
        "precondition_hours",
        "recovery_steps",
        "forgetting_factor",
        "model_residual_gain",
        "comfort_cost",
        "gas_cost_ratio",
        "hot_water_target",
        "standby_minimum",
        "standby_maximum",
        "observation_delay",
        "response_scale",
        "disturbance_rate",
    )

    def __init__(self, building_meta_data, action_spec=None):
        params = dict(self.PARAMS)
        self.release_enabled = bool(params["release_enabled"])
        self.release_hour = float(params["release_hour"])
        self.release_empty_steps = max(1, int(params["release_empty_steps"]))
        self.reward_weights_enabled = bool(params["reward_weights_enabled"])
        self.gas_ratio_scale = float(params["gas_ratio_scale"])
        self.elec_hour_blend = float(params["elec_hour_blend"])
        self.holiday_guard_enabled = bool(params["holiday_guard_enabled"])
        self.early_preheat_enabled = bool(params["early_preheat_enabled"])
        self.early_preheat_hours = max(0.0, float(params["early_preheat_hours"]))
        self.dow_sin_index = None
        self.dow_cos_index = None
        super().__init__(
            building_meta_data,
            action_spec,
            parameters={key: params[key] for key in self.WU_PARAMETERS},
            ensemble=_ensemble(),
            alpha=float(params["residual_alpha"]),
            correction_limit=float(params["correction_limit"]),
            disagreement_limit=float(params["disagreement_limit"]),
        )
        self.dow_sin_index = self.field_index.get("dow_sin_000")
        self.dow_cos_index = self.field_index.get("dow_cos_000")

    # ------------------------------------------------------------------ state
    def reset(self):
        super().reset()
        self._empty_steps = 0
        self._released = False
        self._comfort_seen_today = False
        self._comfort_seen_previous_day = False
        self._hour_cache = int(self.start_hour) % 24
        self._hour_cache_step = -1
        self._last_hour = None

    # ------------------------------------------------------------------ clock
    def _hour_index(self) -> int:
        """Local hour of day, cached for the step (the AHU search calls us often)."""
        if self._hour_cache_step != self.step_count:
            obs = getattr(self, "_r2_obs", None)
            hour = self._hour(obs) if obs is not None else self.start_hour
            self._hour_cache = int(math.floor(hour)) % 24
            self._hour_cache_step = self.step_count
        return self._hour_cache

    def _weekday(self, obs):
        """0 = Monday ... 6 = Sunday, from the public day-of-week features."""
        if self.dow_sin_index is None or self.dow_cos_index is None:
            return None
        angle = math.atan2(float(obs[self.dow_sin_index]), float(obs[self.dow_cos_index]))
        return int(round((angle % (2.0 * math.pi)) * 7.0 / (2.0 * math.pi))) % 7

    # ------------------------------------------------------------- (b) weights
    def _energy_weights(self):
        if not self.reward_weights_enabled:
            return 1.0, float(self.parameters["gas_cost_ratio"])
        hour = self._hour_index()
        blend = min(max(self.elec_hour_blend, 0.0), 1.0)
        weight_e = 1.0 + blend * (ELEC_REL[hour] - 1.0)
        ratio = GAS_RATIO[hour] * self.gas_ratio_scale
        return weight_e, weight_e * ratio

    # ---------------------------------------------------------------- (a) / (c)
    def _occupied(self, obs) -> bool:
        """Whether the building holds anyone, from `num_occupants`.

        The environment writes `(n - C) / (C + 1)` with `C` its
        `occupancy_normalization_constant`, whose gin default is 0.0 and which
        no scenario config overrides -- so the field is the raw count of
        occupants in the trailing 5 minutes and the test is `> 0`. (The
        `env._occupancy_normalization_constant = 125.0` that `evaluate.py`
        sets is a different attribute; the observation code never reads it.)
        `n` is an int truncation of the zone-average sum, so it reads 0 while a
        zone still holds 0.9 of a person; the reward charges that zone's
        shortfall at full weight, which is exactly why the release below is
        re-armed by any occupant instead of by a rate.
        """
        index = self.occupancy_index
        if index is None:
            return True  # cannot tell: assume people are there
        return float(obs[index]) > 0.5

    def _standby_band(self, temperature):
        """Wu's own night band, including its one-step recovery clamp."""
        low = np.full(self.n_zones, self.parameters["standby_minimum"])
        high = np.full(self.n_zones, self.parameters["standby_maximum"])
        recovery = max(1.0, self.parameters["recovery_steps"])
        low = np.minimum(low, temperature + np.maximum((low - temperature) / recovery, 0.0))
        high = np.maximum(high, temperature - np.maximum((temperature - high) / recovery, 0.0))
        return low, high

    def _comfort_day(self, obs, now: bool) -> bool:
        """Whether today's schedule can still contain a comfort window (c)."""
        weekday = self._weekday(obs)
        if weekday is not None and weekday >= 5:
            return False
        if now or self._comfort_seen_today:
            return True
        if self._comfort_seen_previous_day:
            return True
        # First morning of the episode: nothing observed yet, trust the weekday.
        return weekday is not None

    def _track_day(self, obs, now: bool) -> None:
        hour = self._hour(obs)
        if self._last_hour is not None and hour < self._last_hour - 12.0:
            self._comfort_seen_previous_day = self._comfort_seen_today
            self._comfort_seen_today = bool(now)
        elif now:
            self._comfort_seen_today = True
        self._last_hour = hour

    def _release_active(self, obs, now: bool) -> bool:
        if not now:  # the comfort mode the release belongs to has ended
            self._empty_steps = 0
            self._released = False
            return False
        if self._occupied(obs):
            self._empty_steps = 0
            self._released = False
            return False
        self._empty_steps += 1
        if not self._released:
            armed = self._empty_steps >= self.release_empty_steps
            if armed and self._hour(obs) >= self.release_hour:
                self._released = True
        return self._released

    # ------------------------------------------------------------------- band
    def _band(self, obs, temperature):
        low, high, now = super()._band(obs, temperature)
        if not (self.release_enabled or self.early_preheat_enabled
                or self.holiday_guard_enabled):
            return low, high, now
        self._track_day(obs, now)
        if now:
            if self.release_enabled and self._release_active(obs, now):
                low, high = self._standby_band(temperature)
            return low, high, now
        self._release_active(obs, now)  # clears the latch between modes
        if not self.early_preheat_enabled:
            return low, high, now
        soon = (bool(obs[self.comfort_soon_index] > 0.5)
                if self.comfort_soon_index is not None else False)
        if soon:  # Wu's own ramp is already running
            return low, high, now
        if self.holiday_guard_enabled and not self._comfort_day(obs, now):
            return low, high, now
        horizon = min(self.early_preheat_hours,
                      max(1.0, float(self.parameters["precondition_hours"])))
        if horizon <= 0.0:
            return low, high, now
        remaining = (self.start_hour - self._hour(obs)) % 24.0
        if remaining > horizon:
            return low, high, now
        progress = float(np.clip(
            1.0 - remaining / max(0.1, float(self.parameters["precondition_hours"])),
            0.0, 1.0))
        if progress <= 0.0:
            return low, high, now
        low = (self.eco_band[0] + progress * (self.comfort_band[0] - self.eco_band[0])
               + np.zeros(self.n_zones))
        high = (self.eco_band[1] + progress * (self.comfort_band[1] - self.eco_band[1])
                + np.zeros(self.n_zones))
        margin = float(self.parameters["comfort_margin"])
        low += margin
        high -= margin
        recovery = max(1.0, self.parameters["recovery_steps"])
        low = np.minimum(low, temperature + np.maximum((low - temperature) / recovery, 0.0))
        high = np.maximum(high, temperature - np.maximum((temperature - high) / recovery, 0.0))
        return low, high, now


_ENSEMBLE = None


def _ensemble():
    """The two embedded residual members, decoded once per process."""
    global _ENSEMBLE
    if _ENSEMBLE is None:
        _ENSEMBLE = ResidualEnsemble(_wu_members())
    return _ENSEMBLE
