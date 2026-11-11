"""CtrlX -- the paper's controller: CtrlW5 with the shipped 278 K frost floor.

`python -m sbx.arbench` scores this class. It started as exactly CtrlW5-278 (Kaggle
submission 58059955). Development changed this module, and may add sibling modules,
but never `ctrl_w5.py`, so CtrlW5 stays reproducible.

Every version must keep these constraints (enforced by `sbx/arbench.py`):
* numpy and the standard library only, and no building or weather identifiers;
* no zone below +4 C (277.15 K) at any step of the benchmark workload;
* nothing that depends on the simulator's accounting seams or on the occupancy seed.
  The benchmark runs the corrected tree, where the seams are closed, and a
  non-default occupancy seed.

Mechanisms added on top of CtrlW5-278:

1. Release on an empty comfort day ("nobody came").
   - The schedule runs comfort mode on every weekday, holidays included. The
     occupancy model keeps everyone home on holidays: a US calendar, about one
     holiday in a 30-day window, e.g. 2 January 2023.
   - CtrlW's release waits for `release_hour` (noon), so a holiday morning is
     conditioned for nobody.
   - On a work day arrivals start at 06:00, about 1/36 per occupant per step, so a
     building that has stayed empty for K steps since comfort mode began is empty
     for the day: P(false) = (35/36)^(K x occupants).
   - K is sized from the zone count for one occupant per zone and a false-release
     probability of 1e-4. Any arrival re-arms the comfort band, exactly as CtrlW's
     own release does.

2. Load-driven hot-water reset (the boiler-reset practice, priced from the reward).
   - Pump power is exactly `PUMP_W * sum(valve) * dp**1.5` (fitted to 1e-7 on two
     buildings), so for a fixed heat demand it falls as (Tw - Ts)**-3. The tank's
     standing loss rises only `TANK_UA_W_PER_K` per kelvin. Every scenario config
     shares the pump and boiler parameters.
   - Each step the controller prices its own pump draw as c = P * (Tw - Ts)**3,
     which does not depend on Tw. c is smoothed with a slow rise (24 h) and a fast
     decay (3 h). A short morning burst in a mild week then does not charge the
     tank at the dearest gas hour (05:00) only to discharge it at noon. A sustained
     cold-week load reaches the ceiling within an hour through the fourth root.
     Once the load goes, the stored heat is drawn down rather than kept to the end.
   - Minimising w_e * c / x**3 + w_g * UA * x over x = Tw - Ts gives the target
     x* = (3 w_e c / (w_g UA))**(1/4), clipped to [`water_floor`, `water_ceiling`].
   - Wu's idle branch (310 K after two hours without heat demand) is kept.
   - B1 guard: under load, the setpoint never sits more than 0.5 K (one shipped
     cooling step) below the observed tank. On the corrected tree that only caps
     how fast the tank gives back its stored heat. On the shipped tree it keeps the
     setpoint-billed seam (B1) out of reach.

3. No early pre-cooling, unless outside air can do it.
   - CtrlW's extended preheat (and CtrlW2's per-zone version, up to 3 h ahead)
     replaces the night standby band with the eco->comfort ramp for both bounds.
     The upper bound therefore drops from the standby maximum (307 K) to about
     298 K hours before anyone can arrive.
   - In hot-night weeks that holds the building near comfort-high from 04:00 by
     chiller, at near-daytime cost (phoenix_4 and singapore_g6: 12-17 % of the week
     in 04:00-06:00), with zero comfort cost at 06:00. So there the early window
     moves only the lower bound, and Wu's own one-hour ramp (`comfort_mode_soon`)
     brings the upper bound down before arrivals.
   - When outside air is at least `precool_free_margin_K` below comfort-high (and
     below the zones), the economizer pre-cools for fan energy only, and the
     pre-cooled mass cuts daytime cooling by more than that costs. CtrlW's early
     ramp is kept in that case. 30-day dev30 checks: phoenix_3 (Apr) -0.29 %,
     denver_5 (Jul) -0.19 %, atlanta_3 (Apr) -0.73 %, denver_6 (Aug) -0.65 %.
     Tropical nights never qualify.

4. Arrival lead.
   - Every zone trails the pre-arrival ramp by about half a kelvin, so heating weeks
     reach 06:00 below the heating setpoint just as the first occupants arrive.
   - For the last `arrival_lead_hours` of `comfort_mode_soon`, the lower bound is
     the full comfort-low plus the margin (recovery-limited), so the zones arrive
     on the setpoint.

5. Model gain by regime.
   - Wu never learns his zone model online: `_learn` only updates a residual EMA,
     so each zone's HVAC response gain `theta[:, 5]` stays at its geometric
     prior x `response_scale` for the whole episode.
   - At Wu's 0.75 the one-step controller over-acts, and many zones fall into a
     limit cycle: an RMS step change of 0.14-0.20 K against 0.01 K for a settled
     zone. This hit 22 of 49 zones on a hot Denver week, 6-13 zones on milder
     heating weeks, and 2 of 39 on a -37 C week.
   - That comfort cost is pure control error. At 0.80-0.85 the cycles stop and the
     comfort cost goes with them, with energy unchanged. 0.95 is too much.
   - A higher gain under-heats when the hot-water side is saturated, as on
     deep-cold mornings (pump at 20 from 05:00). The controller expects more rise
     per watt, asks for less heat, and the pump leaves capacity unused. Milder
     heating weeks never saturate the pump, and they also prefer the higher gain.
   - So while the last commanded pump pressure is at least
     `heating_gain_saturation_dp`, zones that received reheat in the last step, or
     sit below their last lower bound, run at `heating_response_scale`. Every other
     zone, and every zone when the plant is not saturated, runs at `response_scale`.

6. Comfort margin 0.12 -> 0.08 K.
   - Wu's margin also absorbed the limit cycles. Once zones settle, a 0.08 K margin
     keeps the comfort cost small and saves energy on every benchmark scenario.
   - 0.04 K goes past the knee: the scenarios split, and the comfort cost rises 68 %.

7. Night lift sized by the setback.
   - CtrlW4 lifts floor-bound zones 2 K before every comfort day. That pays on
     Mondays and on the day after a holiday, when the building has spent more than a
     day near the frost floor.
   - After an ordinary night the building is still warm, and a 1 K lift is cheaper
     on every such morning. 30-day fairbanks_1: a uniform 1 K lift gains +0.12..0.19
     units per mid-week morning but loses 0.06-0.32 on Mondays and the post-holiday
     Tuesday.
   - The controller tracks the last step anyone was in the building (the observed
     count). It lifts `night_lift_K` after `night_lift_long_hours` empty, otherwise
     `night_lift_short_K`. 30-day fairbanks_1: -0.135 % cost, better than either
     fixed lift.

8. Per-zone margin, earned by staying in the comfort window.
   - Halving the margin everywhere (0.08 -> 0.04 K) helped six benchmark scenarios
     and hurt four. The losers' zones sometimes exceed the cooling setpoint in the
     afternoon (0.3-0.8 % of occupied zone-steps, against ~0.01 % in the winners).
     That is systematic drift, not one-step noise: Wu's model_error is lower there.
   - So in settled comfort hours (from `margin_relax_settle_hours` after comfort
     starts) each zone earns relaxation (up to `margin_relax_max`, by
     `margin_relax_up` per step) while it stays inside the comfort window. It loses
     `margin_relax_down` per step outside it. Both bounds move by it, with Wu's
     recovery clamp re-applied.
   - Nothing is relaxed before the settle time, and nothing is learned from it.
     The first arrivals weigh the most while the morning recovery tracks loosely.
     Relaxing then, using the margin carried over from the day before, cost about
     1 % of a scenario in 06:00-08:00 comfort on 30-day dev30 shoulder-season weeks
     (denver_3, phoenix_1, denver_7, fairbanks_4). With the gate those four are
     0.4-0.8 % better than without the mechanism.
   - Benchmark: -0.26 %, 10 of 10 scenarios better.

9. Late start for groups whose fabric is still warm.
   - The zone air is light: 3 kW moves it about 2.5 K in a step, and with the plant
     off it relaxes to the fabric within a step. The fabric is what cools overnight
     (about 0.07 K per step at -35 C) and what the morning must recharge.
   - A pre-arrival ramp (Wu's 1 h, CtrlW/CtrlW2's 2-3 h) charges the fabric with the
     air held part-way, which is cheap per watt in deep cold: gas per net watt
     1 + 0.3 (T_zone - T_out)/(T_dis - T_zone) is about half of its value at comfort.
     But every ramp step ventilates every zone of the group with 30 % outside air at
     the minimum flow. In a mild week the fabric has barely cooled, there is little
     to charge, and the ramp is mostly that overhead: 57 % of denver_2's 04-10h gas
     is outside air, and the 03-06h window is 27-28 % of the atlanta_8 and denver_2
     bills.
   - So when the window opens (`late_start_window_hours` before comfort) each AHU
     group whose mean gap to comfort-low is within `late_start_gap_K` drops the
     CtrlW/CtrlW2 readiness ramps (2-3 h) and holds the standby band (with the
     night lift) until Wu's own last-hour ramp, which calls every colder zone at
     once at 05:00 (291.6 K) and the warmer ones as it passes them, followed by
     the 0.5 h arrival lead (`late_start_ramp`). A colder group keeps all ramps.
   - What lost: the full comfort-low from 0.75-1 h before comfort instead of Wu's
     ramp (30-day corrected -0.13 %, but +0.33 % on the shipped tree and mild weeks
     +0.8 %); leads scaled by the group's coldest gap; per-zone call times and
     earned per-zone leads (inside a group the plant then runs for one zone while
     the others are ventilated); an own ramp from eco-low; a warm-started or faster
     residual during the lead.
   - The gate: a first-order optimum-start balance (charge saved at the lower air
     temperature against ventilating the group for the extra hours) puts the
     break-even gap at 2-5 K whatever the weather. 4 -> 6 K: benchmark -0.15 %,
     30-day corrected -0.07 % (10 better / 19 identical / 1 worse); 7-8 K is worse.
   - 30-day dev30, paired vs the controller without mechanisms 9-10: corrected
     -0.33 % at 4 K (14 better / 14 identical / 2 within 0.01 %), shipped -0.18 %,
     49 shipped holdouts -0.22 %. Without a tank hold the shipped tree re-bills the
     tank recharge at the morning call, so there the 6 K gate is +0.03 %.

10. Free-drift clip widened on the cold side (-5 -> -8 K/step).
   - Wu clips the predicted free drift (geometry prior + residual EMA) to +-5 K per
     step. For the large exterior zones of a -35 C week the prior alone is
     -3.2..-3.5 K/step and the residual another -2.5..-3, so the clip binds and the
     one-step inversion under-requests by (sum + 5)/gain, about 3 kW per zone.
   - fairbanks_1: the clip binds on 11-34 % of comfort steps in exactly the four
     zones that sit 1-2 K below the setpoint for hours (mean shortfall 0.48 K when
     binding, 0.01 K otherwise). With -8 the worst morning shortfall falls from
     2.1 K to 0.9 K: comfort 1.19 -> 0.40 per week for 1.0 of energy.
   - Benchmark: -0.15 % (fairbanks_1 -0.32 %, every other scenario bit-identical),
     comfort 0.50 -> 0.27. -12 is identical to -8 (the sum never passes -8).
"""

from __future__ import annotations

import math

import numpy as np

try:  # packaged notebook: the embedded modules are top-level names
    from ctrl_w5 import CtrlW5
except ImportError:  # workspace: policies/ is a package
    from .ctrl_w5 import CtrlW5

PUMP_W = 3.0612  # W per (valve fraction x (differential pressure)**1.5)
TANK_UA_W_PER_K = 6.25  # Boiler defaults (2 m x 0.5 m tank, 6 cm insulation); no config overrides them


class CtrlX(CtrlW5):
    """CtrlX: CtrlW5-278 plus the mechanisms in the module docstring."""

    name = "ctrl_x"

    PARAMS = {
        **CtrlW5.PARAMS,
        "standby_minimum": 278.0,
        # Empty-day release: occupant-steps without an arrival before the day counts as
        # empty. -ln(1e-4) / ln(36/35) = 327, rounded up.
        "empty_day_release_enabled": True,
        "empty_day_occupant_steps": 330.0,
        "empty_day_min_steps": 3,
        "empty_day_max_steps": 12,
        # Load-driven hot-water reset.
        "water_reset_enabled": True,
        "water_floor": 330.0,
        "water_ceiling": 350.0,
        "water_rise_steps": 288.0,
        "water_decay_steps": 36.0,
        # Rise guard: never command a hot-water setpoint more than this above the observed tank
        # (the burner's 0.5 K/min x one step; every scenario config shares it). 0 disables.
        "water_rise_guard_K": 0.0,
        # Off-peak floor (0 disables): outside the morning window (`water_morning_before_hours`
        # before the comfort start to `water_morning_after_hours` after it, comfort days only)
        # the reset's floor is this value instead of `water_floor`. The 330 K floor is morning
        # capacity; held all day it is standing loss (6.25 W/K) that the load never asked for.
        "water_offpeak_floor": 310.0,
        "water_morning_before_hours": 2.0,
        "water_morning_after_hours": 1.5,
        # Seam-neutral tank options (off in the honest line; see docs/ctrlx_experiments.md).
        # water_idle_hold: keep the tank at water_floor through Wu's idle branch (310 K after two
        # hours without demand) while there was heat demand within water_idle_hold_hours, instead of
        # letting it drop and re-billing the recharge at the first morning demand (1e9 hours = always,
        # the variant submitted as 58173725). water_peak_hold_hours > 0: never let the setpoint fall
        # below its recent peak for that long (the shipped tree re-bills every re-rise).
        "water_idle_hold": False,
        "water_idle_hold_hours": 72.0,
        "water_peak_hold_hours": 0.0,
        # No early pre-cooling: the pre-arrival window moves only the lower bound.
        "early_precool_enabled": False,
        # Hold the full comfort-low band for this long before comfort mode starts.
        "arrival_lead_hours": 0.5,
        # Morning start per AHU group, decided late_start_window_hours before comfort: a group
        # whose zones are on average within late_start_gap_K of comfort-low skips the pre-arrival
        # ramps and holds the standby band until late_start_lead_hours before comfort, then the
        # full comfort-low; a colder group keeps the ramps (its fabric is charged while the air
        # is cold).
        "late_start_lead_hours": 0.75,
        # True: the warm groups keep Wu's own last-hour ramp (each zone is called when the ramp passes
        # it) and only the earlier CtrlW/CtrlW2 ramps are dropped; False: the full comfort-low from
        # late_start_lead_hours before comfort.
        "late_start_ramp": True,
        "late_start_gap_K": 6.0,
        "late_start_window_hours": 3.0,
        # Wu clips the predicted free drift to +-5 K/step; the cold side is widened to this.
        "drift_clip_K": 8.0,
        # Online identification of Wu's zone model (prototype, off by default): per-zone RLS on his
        # six regressors plus the previous step's heat and change; the fitted drift and gain replace
        # the frozen prior + residual EMA in the one-step search.
        "ident_enabled": False,
        "ident_forget": 0.998,
        "ident_prior_sd": 0.3,
        "ident_gain_min": 0.4,
        "ident_gain_max": 1.6,
        "ident_comfort_only": True,
        "response_scale": 0.8,
        # Model gain of heated zones while the hot-water pump is near its maximum (20).
        "heating_response_scale": 0.75,
        "heating_gain_saturation_dp": 15.0,
        "comfort_margin": 0.08,
        # Size-aware comfort: the reward weighs occupied zones by headcount and the challenge's
        # occupancy model puts the same headcount in every zone, while a kelvin costs heat in
        # proportion to the zone's size (areas span 3.5-9x inside one building). So a kelvin of
        # comfort insurance is cheap in a small zone and dear in a large one: the comfort margin
        # and the arrival lead are scaled per zone by (median area / area)^power, clipped to
        # [1/size_weight_clip, size_weight_clip]. Power 0 disables.
        "size_margin_power": 0.5,
        "size_lead_power": 0.5,
        "size_weight_clip": 3.0,
        # Allow CtrlW's early upper-bound ramp while outside air is this far below comfort-high
        # (economizer pre-cooling for fan energy only); otherwise no early pre-cooling.
        "precool_free_margin_K": 2.0,
        # Night lift: CtrlW4's 2 K only after a long empty stretch (weekend, holiday);
        # after an ordinary night the building is warmer and a smaller lift pays.
        "night_lift_short_K": 1.0,
        "night_lift_long_hours": 24.0,
        # Per-zone margin relaxation in settled comfort: up to margin_relax_max K, grown by
        # margin_relax_up per clean step, cut by margin_relax_down per violating step.
        "margin_relax_max": 0.04,
        "margin_relax_up": 0.0004,
        "margin_relax_down": 0.01,
        "margin_relax_settle_hours": 1.0,
    }

    def __init__(self, building_meta_data, action_spec=None):
        params = dict(self.PARAMS)
        self.empty_day_release_enabled = bool(params["empty_day_release_enabled"])
        self._empty_day_occupant_steps = float(params["empty_day_occupant_steps"])
        self._empty_day_min_steps = int(params["empty_day_min_steps"])
        self._empty_day_max_steps = int(params["empty_day_max_steps"])
        self.water_reset_enabled = bool(params["water_reset_enabled"])
        self.water_floor = float(params["water_floor"])
        self.water_ceiling = float(params["water_ceiling"])
        self.water_rise_steps = max(1.0, float(params["water_rise_steps"]))
        self.water_decay_steps = max(1.0, float(params["water_decay_steps"]))
        self.water_idle_hold = bool(params["water_idle_hold"])
        self.water_idle_hold_steps = max(0.0, float(params["water_idle_hold_hours"])) * 12.0
        self.water_peak_hold_steps = max(0.0, float(params["water_peak_hold_hours"])) * 12.0
        self.water_rise_guard_K = max(0.0, float(params["water_rise_guard_K"]))
        self.water_offpeak_floor = max(0.0, float(params["water_offpeak_floor"]))
        self.water_morning_before_hours = max(0.0, float(params["water_morning_before_hours"]))
        self.water_morning_after_hours = max(0.0, float(params["water_morning_after_hours"]))
        self.early_precool_enabled = bool(params["early_precool_enabled"])
        self.arrival_lead_hours = max(0.0, float(params["arrival_lead_hours"]))
        self.late_start_lead_hours = max(0.0, float(params["late_start_lead_hours"]))
        self.late_start_ramp = bool(params["late_start_ramp"])
        self.late_start_gap_K = float(params["late_start_gap_K"])
        self.late_start_window_hours = max(self.late_start_lead_hours, float(params["late_start_window_hours"]))
        self.drift_clip_K = float(params["drift_clip_K"])
        self.ident_enabled = bool(params["ident_enabled"])
        self.ident_forget = min(max(float(params["ident_forget"]), 0.9), 1.0)
        self.ident_prior_sd = max(1e-3, float(params["ident_prior_sd"]))
        self.ident_gain_min = max(0.05, float(params["ident_gain_min"]))
        self.ident_gain_max = max(self.ident_gain_min, float(params["ident_gain_max"]))
        self.ident_comfort_only = bool(params["ident_comfort_only"])
        self.heating_response_scale = float(params["heating_response_scale"])
        self.heating_gain_saturation_dp = float(params["heating_gain_saturation_dp"])
        self.night_lift_long_K = max(0.0, float(params["night_lift_K"]))
        self.night_lift_short_K = max(0.0, float(params["night_lift_short_K"]))
        self.night_lift_long_steps = max(0.0, float(params["night_lift_long_hours"])) * 12.0
        self.margin_relax_max = max(0.0, float(params["margin_relax_max"]))
        self.margin_relax_up = max(0.0, float(params["margin_relax_up"]))
        self.margin_relax_down = max(0.0, float(params["margin_relax_down"]))
        self.margin_relax_settle_hours = max(0.0, float(params["margin_relax_settle_hours"]))
        self.size_margin_power = max(0.0, float(params["size_margin_power"]))
        self.size_lead_power = max(0.0, float(params["size_lead_power"]))
        self.size_weight_clip = max(1.0, float(params["size_weight_clip"]))
        super().__init__(building_meta_data, action_spec)
        steps = math.ceil(self._empty_day_occupant_steps / max(1, self.n_zones))
        self.empty_day_steps = min(max(steps, self._empty_day_min_steps), self._empty_day_max_steps)

    def reset(self):
        super().reset()
        self._seen_this_window = False
        self._pump_coef = 0.0
        self._supply_ref = 285.0
        self._water_peak = None
        self._water_peak_step = None
        self._gain_cool = self.theta[:, 5].copy()
        self._gain_heat = self._gain_cool * (self.heating_response_scale / float(self.parameters["response_scale"]))
        self._last_low = None
        self._last_pump_dp = 0.0
        self._last_occupied_step = -10 ** 9  # the episode starts on a Sunday: a long setback
        self._relax = np.zeros(self.n_zones)
        self._late_start = None
        rel = float(np.median(self.zone_area)) / np.maximum(np.asarray(self.zone_area, dtype=float), 1e-6)
        clip = self.size_weight_clip
        self._size_w_margin = np.clip(rel ** self.size_margin_power, 1.0 / clip, clip)
        self._size_w_lead = np.clip(rel ** self.size_lead_power, 1.0 / clip, clip)
        self._id_w = None

    def _release_active(self, obs, now: bool) -> bool:
        released = super()._release_active(obs, now)
        if not now:
            self._seen_this_window = False
            return released
        if self._occupied(obs):
            self._seen_this_window = True
            return released
        if (
            not released
            and self.empty_day_release_enabled
            and not self._seen_this_window
            and self._empty_steps >= self.empty_day_steps
        ):
            self._released = True
            return True
        return released

    def _band(self, obs, temperature):
        if self._occupied(obs):
            self._last_occupied_step = self.step_count
        long_setback = self.step_count - self._last_occupied_step > self.night_lift_long_steps
        self.night_lift_K = self.night_lift_long_K if long_setback else self.night_lift_short_K
        low, high, now = super()._band(obs, temperature)
        if not now:
            soon = bool(obs[self.comfort_soon_index] > 0.5) if self.comfort_soon_index is not None else False
            if not soon and not self.early_precool_enabled and not self._free_cooling(obs, temperature):
                high = np.maximum(high, float(self.parameters["standby_maximum"]))
            low = self._lead_band(obs, low, temperature)
        if now and not self._released and self.size_margin_power > 0.0:
            shift = float(self.parameters["comfort_margin"]) * (self._size_w_margin - 1.0)
            recovery = max(1.0, float(self.parameters["recovery_steps"]))
            raw_low = np.where(low > temperature, temperature + recovery * (low - temperature), low) + shift
            raw_high = np.where(high < temperature, temperature - recovery * (temperature - high), high) - shift
            low = np.minimum(raw_low, temperature + np.maximum(raw_low - temperature, 0.0) / recovery)
            high = np.maximum(raw_high, temperature - np.maximum(temperature - raw_high, 0.0) / recovery)
        if now and not self._released and self.margin_relax_max > 0.0:
            low, high = self._relax_band(obs, low, high, temperature)
        self._last_low = low.copy()
        return low, high, now

    def _lead_band(self, obs, low, temperature):
        """Late start for groups whose fabric is already warm; the ramps for the others.

        The zone air is light and relaxes to the fabric within a step: a ramp that holds the air
        part-way for two hours charges the fabric at a low air temperature, which is cheap per
        watt in deep cold (gas per net watt 1 + 0.3 (T_zone - T_out)/(T_dis - T_zone) is about
        half of its value at comfort), but every step of it ventilates every zone of the group
        with 30 % outside air at the minimum flow. When the fabric has barely cooled overnight
        (mild weeks) there is little to charge and the ramp is mostly that overhead, so the
        group stays on the standby band until `arrival_lead_hours` before comfort and then
        asks for the full comfort-low, which the plant reaches within a few steps. The choice
        is latched per group when the window opens, from the group's mean gap to comfort-low.
        """
        remaining = (self.start_hour - self._hour(obs)) % 24.0
        if remaining > self.late_start_window_hours or not self._comfort_day(obs, False):
            self._late_start = None
            return low
        if self._late_start is None:
            gap = self.comfort_band[0] - temperature
            self._late_start = np.zeros(self.n_zones, dtype=bool)
            for group in self.groups:
                if group.size and float(np.mean(gap[group])) <= self.late_start_gap_K:
                    self._late_start[group] = True
        full = self.comfort_band[0] + float(self.parameters["comfort_margin"]) * self._size_w_margin
        recovery = max(1.0, float(self.parameters["recovery_steps"]))
        target = np.minimum(full, temperature + np.maximum((full - temperature) / recovery, 0.0))
        # the ramp ends exactly at comfort start; arrive on the setpoint
        low = np.where(remaining <= self.arrival_lead_hours * self._size_w_lead, np.maximum(low, target), low)
        if not np.any(self._late_start):
            return low
        if self.late_start_ramp:
            # Wu's own last-hour ramp (plus the arrival lead) calls each zone when the ramp passes it;
            # only the earlier CtrlW/CtrlW2 ramps are dropped for the warm groups.
            soon = bool(obs[self.comfort_soon_index] > 0.5) if self.comfort_soon_index is not None else False
            if soon:
                return low
            base = self._night_lift(obs, self._standby_band(temperature)[0], temperature)
            return np.where(self._late_start, base, low)
        base = self._night_lift(obs, self._standby_band(temperature)[0], temperature)
        if remaining <= self.late_start_lead_hours:
            base = np.maximum(base, target)
        return np.where(self._late_start, base, low)

    def _relax_band(self, obs, low, high, temperature):
        """Per-zone comfort margin: relax zones that keep inside the comfort window.

        Halving the margin everywhere helped six benchmark scenarios and hurt four. The
        losers are the buildings whose zones occasionally exceed the cooling setpoint in
        the afternoon: 0.3-0.8 % of occupied zone-steps, against ~0.01 % in the winners.
        That is systematic drift, not one-step noise (Wu's model_error is lower in the
        losers). So each zone earns relaxation while it stays inside the comfort window
        and loses it fast when it leaves.
        """
        # Arrivals run from 06:00 and the first arrivals weigh most, while the morning
        # recovery tracks loosely: relax neither then, and do not learn from it.
        if (self._hour(obs) - self.start_hour) % 24.0 < self.margin_relax_settle_hours:
            return low, high
        outside_band = (temperature > self.comfort_band[1]) | (temperature < self.comfort_band[0])
        self._relax = np.clip(self._relax + np.where(outside_band, -self.margin_relax_down, self.margin_relax_up),
                              0.0, self.margin_relax_max)
        if not np.any(self._relax > 0.0):
            return low, high
        recovery = max(1.0, float(self.parameters["recovery_steps"]))
        # Invert Wu's recovery clamp, shift the raw bounds, and re-apply the clamp.
        raw_low = np.where(low > temperature, temperature + recovery * (low - temperature), low) - self._relax
        raw_high = np.where(high < temperature, temperature - recovery * (temperature - high), high) + self._relax
        low = np.minimum(raw_low, temperature + np.maximum(raw_low - temperature, 0.0) / recovery)
        high = np.maximum(raw_high, temperature - np.maximum(temperature - raw_high, 0.0) / recovery)
        return low, high

    def _free_cooling(self, obs, temperature) -> bool:
        """Whether outside air alone can pre-cool the zones to the comfort ceiling.

        True when outside air is at least `precool_free_margin_K` below comfort-high and
        below the zones. The economizer then pre-cools for fan energy only. Tropical nights
        stay above the ceiling, so only the chiller could reach it there.
        """
        margin = float(self.PARAMS.get("precool_free_margin_K", -1.0))
        if margin < 0.0 or not self.outside_obs:
            return False
        outside = float(np.mean([self._temperature(obs, i, "outside_air_temperature_sensor", 290.0)
                                 for i in self.outside_obs]))
        return outside < min(self.comfort_band[1] - margin, float(np.mean(temperature)))

    # ------------------------------------------------------------ model gain
    def _select_gain(self, obs):
        """This step's per-zone response gain: the heating prior for heated zones at capacity."""
        if self.heating_response_scale == float(self.parameters["response_scale"]):
            return
        heating = np.zeros(self.n_zones, dtype=bool)
        if self._last_pump_dp >= self.heating_gain_saturation_dp:
            if self.history:
                heating |= self.history[-1]["reheat"] > 1.0
            if self._last_low is not None:
                heating |= self._zone_temperatures(obs) < self._last_low
        self.theta[:, 5] = np.where(heating, self._gain_heat, self._gain_cool)
        if self.ident_enabled and self._id_w is not None:
            now = bool(obs[self.comfort_now_index] > 0.5) if self.comfort_now_index is not None else True
            if now or not self.ident_comfort_only:
                prior = self._gain_cool
                self.theta[:, 5] = np.clip(self._id_w[:, 5], self.ident_gain_min * prior, self.ident_gain_max * prior)

    # ------------------------------------------------------- identification
    def _ident_init(self):
        n = self.n_zones
        self._id_w = np.hstack((self.theta.copy(), np.zeros((n, 2))))  # [theta0..5, Q(k-1), dT(k-1)]
        scale = np.abs(self._id_w) * self.ident_prior_sd + 1e-3
        scale[:, 6] = 0.1
        scale[:, 7] = 0.3
        self._id_P = np.einsum("ni,ij->nij", scale ** 2, np.eye(8))
        self._id_prev_q = np.zeros(n)
        self._id_prev_delta = np.zeros(n)
        self._id_residual = np.zeros(n)
        self._id_updates = 0

    def _learn(self, observed):
        if self.ident_enabled:
            if self._id_w is None:
                self._ident_init()
            delay = int(self.parameters["observation_delay"])
            if self.previous_observed is not None and len(self.history) > delay:
                applied = list(self.history)[-(delay + 1)]
                old = self.previous_observed
                power = (applied["conductance"] * (applied["supply"] - observed) + applied["reheat"]) / 1000.0
                features = self._features(old, applied["outside"], applied["mode"], applied["slow"])
                x = np.column_stack((features, power, self._id_prev_q, self._id_prev_delta))
                y = np.clip(observed - old, -15.0, 15.0)
                err = y - np.einsum("ni,ni->n", x, self._id_w)
                px = np.einsum("nij,nj->ni", self._id_P, x)
                denom = self.ident_forget + np.einsum("ni,ni->n", x, px)
                gain = px / denom[:, None]
                self._id_w = self._id_w + gain * err[:, None]
                self._id_P = (self._id_P - np.einsum("ni,nj->nij", gain, px)) / self.ident_forget
                rate = float(self.parameters["disturbance_rate"])
                post = y - np.einsum("ni,ni->n", x, self._id_w)
                self._id_residual = np.clip((1.0 - rate) * self._id_residual + rate * post, -5.0, 5.0)
                self._id_prev_q = power
                self._id_prev_delta = y
                self._id_updates += 1
        super()._learn(observed)

    def _free_drift(self, temperature, outside, mode):
        """Wu's free drift with the -5 K/step clip widened to -`drift_clip_K`.

        Wu clips the predicted free drift (geometry prior + residual EMA) to +-5 K per step.
        For the large exterior zones of a -35 C week the prior alone is -3.2..-3.5 K/step
        and the residual another -2.5..-3, so the clip binds; the one-step inversion then
        under-requests by (sum + 5) / gain, about 3 kW, and those zones sit 1-2 K below the
        setpoint for hours (fairbanks_1: the four zones where the clip binds on 11-34 % of
        comfort steps carry a 0.48 K mean shortfall, every other zone-step 0.01 K).
        """
        drift = super()._free_drift(temperature, outside, mode)
        if self.ident_enabled and self._id_w is not None and (bool(mode) or not self.ident_comfort_only):
            w = self._id_w
            raw = (np.einsum("ij,ij->i", w[:, :5], self._features(temperature, outside, mode, self.slow_temperature))
                   + w[:, 6] * self._id_prev_q + w[:, 7] * self._id_prev_delta + self._id_residual)
            return np.clip(raw, -self.drift_clip_K, 5.0)
        if self.drift_clip_K <= 5.0:
            return drift
        raw = np.einsum("ij,ij->i", self.theta[:, :5], self._features(temperature, outside, mode, self.slow_temperature))
        raw = raw + self.residual
        return np.where(raw < -5.0, np.maximum(raw, -self.drift_clip_K), drift)

    # ------------------------------------------------------------ hot water
    def get_action(self, obs):
        self._select_gain(np.asarray(obs, dtype=float).reshape(-1))
        action = super().get_action(obs)
        continuous = action["continuous_action"]
        if len(self.pressure_indices) > self.num_ahus:
            self._last_pump_dp = float(np.clip(self._native_value(
                "differential_pressure_setpoint", continuous[self.pressure_indices[-1]], 0.0, 20.0), 0.0, 20.0))
        if self.water_reset_enabled and self.water_indices and len(self.pressure_indices) > self.num_ahus:
            self._reset_water(np.asarray(obs, dtype=float).reshape(-1), continuous)
        return action

    def _native_value(self, kind, normalized, low, high):
        """Inverse of `_native_action`: a normalised action entry in native units."""
        spec = self.action_norm.get(kind)
        if spec is None:
            spec = next((v for k, v in self.action_norm.items() if str(k).endswith(kind)), None)
        if spec:
            low = float(spec.get("min_native_value", low))
            high = float(spec.get("max_native_value", high))
            out_low = float(spec.get("min_normalized_value", -1.0))
            out_high = float(spec.get("max_normalized_value", 1.0))
        else:
            out_low, out_high = -1.0, 1.0
        return low + (np.asarray(normalized, dtype=float) - out_low) * (high - low) / max(out_high - out_low, 1e-12)

    def _reset_water(self, obs, continuous):
        index = self.water_indices[0]
        wu_target = float(self._native_value("supply_water_temperature_setpoint", continuous[index], 310.0, 350.0))
        valves = np.clip(self._native_value("reheat_valve_percentage_command",
                                            continuous[self.valve_indices], 0.0, 1.0), 0.0, 1.0)
        pressure = float(np.clip(self._native_value("differential_pressure_setpoint",
                                                    continuous[self.pressure_indices[-1]], 0.0, 20.0), 0.0, 20.0))
        water = self._temperature(obs, self.water_sensor, "supply_water_temperature_sensor", 340.0)
        open_valves = float(valves.sum())
        coef = 0.0
        if open_valves > 1e-9 and pressure > 1e-9:
            self._supply_ref = float(np.dot(valves, self.history[-1]["supply"]) / open_valves)
            gap = max(water - self._supply_ref, 5.0)
            coef = PUMP_W * open_valves * pressure ** 1.5 * gap ** 3
        steps = self.water_rise_steps if coef > self._pump_coef else self.water_decay_steps
        self._pump_coef += (coef - self._pump_coef) / steps
        idle = wu_target <= 310.0 + 1e-3  # Wu's idle branch: no heat demand for two hours
        # The episode start counts as recent heat: the plant's history is unknown, and on the shipped
        # tree the first recharge of a heating month costs more than 72 h of standing loss.
        since_heat = self.step_count - max(self.last_heating_step, 0)
        if self._water_peak_step is not None and self.step_count - self._water_peak_step > self.water_peak_hold_steps:
            self._water_peak = None
        if idle:
            if self.water_idle_hold and since_heat <= self.water_idle_hold_steps:
                hold = self.water_floor if self._water_peak is None else max(self.water_floor, self._water_peak)
                if self.water_rise_guard_K > 0.0:
                    hold = min(hold, max(water + self.water_rise_guard_K, 310.0))
                continuous[index] = self._native_action("supply_water_temperature_setpoint", hold, 310.0, 350.0)
            return
        weight_e, weight_g = self._energy_weights()
        gap_star = (3.0 * weight_e * self._pump_coef / max(weight_g * TANK_UA_W_PER_K, 1e-9)) ** 0.25
        floor = self.water_floor
        if self.water_offpeak_floor > 0.0:
            # The floor is morning capacity (the load-driven target lags the first call by its 24-h
            # rise); for the rest of the day it only adds standing loss.
            since_start = (self._hour(obs) - self.start_hour) % 24.0
            morning = since_start <= self.water_morning_after_hours or since_start >= 24.0 - self.water_morning_before_hours
            if not (morning and self._comfort_day(obs, False)):
                floor = min(floor, self.water_offpeak_floor)
        target = min(max(self._supply_ref + gap_star, floor), self.water_ceiling)
        if self.water_peak_hold_steps > 0:
            # Track the reset's own target (not the tank-following guard below, which would ratchet).
            if self._water_peak is None or target >= self._water_peak:
                self._water_peak, self._water_peak_step = target, self.step_count
            target = max(target, self._water_peak)
        target = min(max(target, water - 0.5), 350.0)
        if self.water_rise_guard_K > 0.0:
            # The other side of the B1 guard: the burner raises the tank by at most 2.5 K per step on
            # both trees, so a setpoint further above the tank changes nothing physically; on the
            # shipped tree the gap is billed as flow x (setpoint - return) while the coils draw at
            # the tank's own temperature.
            target = min(target, max(water + self.water_rise_guard_K, 310.0))
        continuous[index] = self._native_action("supply_water_temperature_setpoint", target, 310.0, 350.0)
