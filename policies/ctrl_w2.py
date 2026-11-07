"""CtrlW2 -- CtrlW with an adaptive, per-zone preheat readiness horizon.

Why
---
CtrlW extended Wu's preheat ramp from the 1 h his `comfort_mode_soon` flag
allows to a fixed 2 h. The measured trade-off on dev30 (shipped, 8641 steps)
is a shallow optimum: the arrival-hour comfort loss falls from -55.4 to -14.3
for ~21 units of extra energy+carbon, and going further keeps buying comfort
at a price the reward does not pay back:

    scenario (own episode)     1 h        2 h        3 h        4 h
    train_1/fairbanks_1 h06   -8.780     -2.138     -0.672     -0.179
    train_1/fairbanks_1 score 0.046467   0.045910   0.046127   0.046386
    train_7/fairbanks_7 h06   -3.721     -0.461     -0.064     -0.006
    train_7/fairbanks_7 score 0.008836   0.008670   0.008868   0.009032

Every extra hour still removes most of the remaining hour-06 comfort loss, but
costs slightly more energy+carbon than the comfort it buys (2 h -> 3 h is
+0.47% on fairbanks_1, +2.3% on fairbanks_7).  The hours are also spent
uniformly on all zones, while only a few zones are actually short at 06:00 --
so the ramp pays for the whole building to fix a handful of rooms.

Diagnosis of what is left (scratch/AdaptivePreheat/, 14-day traces, shipped)
---------------------------------------------------------------------------
Per-step, per-zone logs of CtrlW on four scenarios (2 h horizon) say the
residual morning loss is neither a Monday effect nor a cold-day effect, and
almost none of it is a *readiness* deficit:

* fairbanks_1 -- 4 of 39 zones (VAV_20/29/30/32, one AHU group) carry 89% of
  the hour-06 shortfall (15.5 occupancy-weighted K over 15 days, 8.7 per rated
  day); fairbanks_8 spreads it over 17 of 30 zones at 0.05-0.11 each.
* Those zones sit at the 282 K standby floor all night, then climb 282 -> 288.5
  in the *first 30 minutes* of the ramp (13 K/h, so the plant is nowhere near
  rate-limited), and from then on track 1.3-1.8 K below the ramp band, arriving
  at 06:00 at 292.3 K under a 294.12 K band and still 1 K short at 07:00.
* That is the signature of a proportional offset, not of a short horizon: the
  AHU search inverts `heat_need = (band_low - T - drift) / theta[:, 5]` with
  the frozen prior `theta[:, 5] = 0.31-0.49` K per kW-step, while the response
  identified online here is `g = 0.023` K per kW-step -- the prior is ~10-20x
  too fast, so the loop asks for a fraction of the power it needs and settles.
* Day structure: comfort loss by weekday is Mon -0.105 vs -0.072 other days on
  fairbanks_1 and Mon -0.009 vs -0.012 on fairbanks_8 (no Monday effect), and
  the correlation with the 00:00-03:00 standby energy (a cold proxy) is +0.09
  and -0.50 (no cold-day concentration). The weekend float does show up, but
  in *energy*: Monday 03:00-06:00 energy is 2.6-2.9x the other weekdays in
  phoenix_5 (1.78 vs 0.70) and singapore_3 (0.50 vs 0.19), 1.4x on fairbanks_8.
* Warm stations have no hour-06 comfort loss at all (phoenix_5 0.000 every day,
  singapore_3 0.013 over 15 days); their morning cost is 0.6-0.9 reward units
  per day of energy.

So the horizon is *not* the binding constraint in the cold scenarios: the
identified model predicts 1.6-1.8 h of lead for a 12 K gap (8-18 K/h under
full conditioning, which the plant demonstrably reaches) and CtrlW already
budgets 2 h. What CtrlW2 can still harvest is the difference per zone and per
day -- trimming the horizon where the model says the plant needs less -- and
`preheat_comp_gain` exposes the lever that actually attacks the measured 1.5 K
tracking offset (raise the band by the measured lag, clipped to the comfort
low). That part is measured in the table below.

Mechanism
---------
Wu's controller carries a per-zone thermal model that *is* identified online,
in two pieces:

* the frozen free-drift prior `theta[:, :5]` (geometry-derived UA/area terms),
  corrected every step by the online residual `residual`, and
* the response `theta[:, 5]` (K per kW per 5-min step) used to invert the
  implicit heat equation in `_choose_ahu`.

Only the residual is adapted in the published code, and the response prior is
badly scaled for these buildings: with `theta[:, 5] = 0.31-0.49` the implicit
equation prices a 20 kW zone heat injection at tens of K per 5-minute step,
while the measured plant response is 0.1-0.7 K per step.  CtrlW2 therefore fits
the *same* model structure online, per zone, as a two-parameter regression of
the one-step transition

    dT  =  d + g * x,        x = (conductance * (supply - T) + reheat) / 1000

with `x` in kW per zone -- exactly the quantities Wu already records in
`history` for `_learn`, no new observation is needed.  `d` (K/step) is the
identified drift including the residual, `g` (K per kW-step) the identified
response.  The fit is a recursive least squares with forgetting factor
`preheat_forget`, clipped to physical ranges, started from a deliberately wide
prior so a single preheat episode identifies it.

Each night, for every zone, CtrlW2 runs that identified model forward under
*full* conditioning at the current outside temperature (maximum zone airflow
from the observed `supply_air_flowrate_setpoint`, supply air at the 305 K
setpoint ceiling, reheat capped by the valve law and the 308.15 K discharge
limit) until the zone reaches the comfort band.  The number of steps taken is
the predicted lead time.  A decaying maximum of the *measured* per-step rise
while the zone was heated is kept as an independent upper bound on the rate,
so the plan can only ever start earlier than the identified model alone
suggests.

The per-zone band during the pre-arrival window is then Wu's own linear blend
from the eco band to the comfort band, but with zone `z`'s ramp spanning its

    H_z = clip(preheat_safety * lead_z + preheat_slack_hours,
               preheat_min_hours, preheat_max_hours)

instead of the single fixed `early_preheat_hours`.  The blend is written as
`progress = 0.5 * (H - remaining) / (H - 1)`, which is exactly Wu's
`1 - remaining / precondition_hours` when `H = 2` h and passes the baton to
Wu's own 1 h ramp at `progress = 0.5` for any `H`, so the ramp stays continuous.
The ramp only exceeds the fixed `preheat_fallback_hours` horizon when it is
predicted to leave the zone more than `preheat_gate_k` K short
(`gap - rate * fallback > gate_k`); otherwise the zone is held to
`min(predicted, fallback)`, so the identified plan may also *shorten* the ramp
where it says the plant needs less -- that is the lever the measurements below
say pays most.
Before the response is identified (fewer than `preheat_min_samples` RLS
updates, i.e. roughly the first preheat episode) every zone falls back to
`preheat_fallback_hours`, so the first days are plain CtrlW.

The horizon is latched monotonically non-decreasing within a day (the latch
starts empty and is cleared whenever the pre-arrival window is clearly not open,
`remaining > preheat_max_hours`), so a noisy estimate can never retract heat
that was already paid for.

Measured: full 8641-step episodes, sims/shipped, components hook
--------------------------------------------------------------
The 12 dev30 cold/warm scenarios (all six fairbanks, all six phoenix), CtrlW
from `results/ctrw_all`; every CtrlW2 run is a full 8641-step episode on the
same scenario, sim and step count.  Three variants were measured:

    v2    extend-only   min 2.0  max 3.0  gate 2.0  safety 1.0   slack 0.25
    trim  trim-only     min 1.0  max 2.0  gate 1.0  safety 1.25  slack 0.25
    v4    trim+extend   min 1.0  max 3.0  gate 2.0  safety 1.0   slack 0.25

    scenario (dev30)         CtrlW   v2 d%  trim d%  v4 d% | h06 CtrlW   v4 | 03-06en CtrlW     v4
    train_1/fairbanks_1    0.045730   -0.09    -0.08  -0.21 |   -2.189 -2.323 |      -34.51 -34.65
    train_2/fairbanks_2    0.012412   -0.15    -0.33  -0.46 |   -0.514 -0.612 |      -13.77 -13.62
    train_4/fairbanks_4    0.001606   +0.03    -0.48  -0.28 |   -0.289 -0.299 |       -1.75  -1.73
    train_5/fairbanks_5    0.001553   -0.00    +0.00  +0.00 |   -0.086 -0.086 |       -0.79  -0.78
    train_7/fairbanks_7    0.008450   +0.07    -0.69  -0.36 |   -0.469 -0.583 |       -8.11  -7.97
    train_8/fairbanks_8    0.024166   -0.01    -0.33  -0.26 |   -0.336 -0.428 |      -22.71 -22.39
    train_1/phoenix_5      0.014240   +0.00    +0.00  +0.00 |    0.000  0.000 |      -20.90 -20.90
    train_2/phoenix_6      0.007199   -0.00    -0.00  -0.00 |   -0.000 -0.000 |      -11.57 -11.57
    train_4/phoenix_8      0.001644   -0.01    +0.00  +0.37 |   -0.221 -0.235 |       -1.62  -1.60
    train_5/phoenix_1      0.002350   -0.00    -0.00  -0.26 |   -0.891 -0.946 |       -2.19  -2.13
    train_7/phoenix_3      0.005454   +0.00    +0.00  +0.00 |   -0.000 -0.000 |       -2.23  -2.23
    train_8/phoenix_4      0.009606   -0.00    -0.00  -0.00 |    0.000  0.000 |      -16.16 -16.16

    mean score 0.011201: v2 0.011196 (-0.04%), trim 0.011182 (-0.16%),
    v4 0.011180 (-0.18%); v4 better on 8 of 12, worst +0.37%, none > 3% worse.
    sums over the 12: comfort -6.21 -> v2 -5.75 / trim -7.05 / v4 -6.64
    (v4 -6.9%), energy -502.86 -> v4 -501.96 (+0.2%), carbon -652.36 -> -650.69
    (+0.3%); hour-06 comfort -4.996 -> -5.511, 03-06 energy -136.31 -> -135.74.

Reading of the three:

* Extending the ramp (v2) is what the diagnosis predicts: the model only lengthens
  the horizon on the zones the identified plan says will be short, buys +7.4% of
  the remaining arrival comfort and costs no energy at all (the plant substitutes
  early heat for later heat almost unit for unit). 0.04% of score.
* Trimming it (trim) is the *more valuable* lever, and the opposite of what the
  static 1 h/2 h/3 h sweep suggested: fairbanks improves on all five scenarios
  (-0.08 ... -0.69%) because the zones whose predicted lead is under 2 h no
  longer sit 2 h in a ramp band they cannot follow -- hour-06 comfort is 0.79
  worse but the 03-06 energy (-2.1) and carbon are cheaper. Phoenix is
  bit-identical, which corrects the earlier claim that its early ramp is worth
  0.2-3.7%: that number came from the *fixed* 1 h horizon, which flattens the
  ramp curve, while a 1 h horizon here only skips the pre-ramp (Wu's own 1 h
  ramp still runs) and the warm zones are above the band either way.
* v4 does both and keeps most of trim's energy/carbon gain for half of trim's
  comfort loss, so it is the variant shipped as the PARAMS default, and it is
  the one run on the full dev30 (see below). v4 vs trim is a 0.02% tie in score
  at a better comfort/energy trade.

Not promoted: `preheat_comp_gain = 1.0` with `preheat_comp_max = 1.5` (raise the
band by the measured tracking lag) -- 700 steps of fairbanks_1 gave -14.9735 ->
-14.9832 (-0.07%) with hour-06 comfort unchanged (-0.0378 -> -0.0376), i.e.
offsetting the band does not close the plant's proportional offset. Default 0.0.

Reproducibility: identical params on the same machine are bit-identical
(`results/ctrw_a` == `results/ctrw_c` on fairbanks_1); the same run on the Mac
and on the x86 PC differs by 3.6e-6 relative in total reward
(`results/ctw2_trim` vs `results/ctw2_trim_check`, 12 of 12 scenarios agree to
6 decimals of the mean score), three orders below the deltas above.

Full dev30 (30 scenarios, shipped, 8641 steps, mean over scenarios of |mean
per-step reward|): CtrlW 0.007369419 -> CtrlW2/v4 0.007357548 (-0.161%), better
on 14 of 30, worse on 8 (worst train_1/denver_7 +0.66%), identical on 8;
comfort -15.40 (-7.15%), energy -966.75 (+0.14%), carbon -925.15 (+0.30%),
hour-06 comfort -9.99 -> -11.06.  Records: `results/ctw2_v4` (the 12
fairbanks/phoenix scenarios, PC) + `results/ctw2_v4_dev30` (the 18
atlanta/denver/singapore scenarios, Mac, `--fast`), scored together by
`scratch/AdaptivePreheat/dev30.py`; the split is immaterial at the 3.6e-6
relative machine-to-machine reproducibility measured above.

Knobs (all in PARAMS, `preheat_adaptive_enabled=False` reproduces CtrlW
bit-exactly)
--------------------------------------------------------------------------
    preheat_safety          1.0    multiply the predicted lead
    preheat_slack_hours     0.25   add after the safety factor
    preheat_min_hours       1.0    floor (1 h == Wu's own ramp, no early heat)
    preheat_max_hours       3.0    ceiling on the horizon
    preheat_fallback_hours  2.0    fixed horizon used until identified / ungated
    preheat_gate_k          2.0    only extend where the fallback leaves > this
    preheat_min_samples     40     RLS updates before the model is trusted
    preheat_forget          0.995  RLS forgetting factor
    preheat_drift_slope     1.0    scale on the outside-coupling slope of d(T)
    preheat_target_margin   0.0    K above the comfort low to aim for
    preheat_supply_ceiling  305.0  assumed full-conditioning supply temperature
    preheat_update_steps    6      recompute the plan every 30 minutes

Nothing here reads a building id, a weather file, the occupancy seed or the
future; every quantity comes from the observation, the public metadata, or
Wu's own online state.
"""

from __future__ import annotations

import numpy as np

try:  # packaged notebook: the embedded modules are top-level names
    from ctrl_w import CtrlW
except ImportError:  # workspace: policies/ is a package
    from .ctrl_w import CtrlW

STEPS_PER_HOUR = 12.0


class CtrlW2(CtrlW):
    """CtrlW + a per-zone, model-predicted preheat readiness horizon."""

    name = "ctrl_w2"

    PARAMS = {
        **CtrlW.PARAMS,
        "preheat_adaptive_enabled": True,
        "preheat_safety": 1.0,
        "preheat_slack_hours": 0.25,
        "preheat_min_hours": 1.0,
        "preheat_max_hours": 3.0,
        "preheat_fallback_hours": 2.0,
        "preheat_gate_k": 2.0,
        "preheat_min_samples": 40,
        "preheat_forget": 0.995,
        "preheat_drift_slope": 1.0,
        "preheat_target_margin": 0.0,
        "preheat_supply_ceiling": 305.0,
        "preheat_update_steps": 6,
        "preheat_debug": False,
        "preheat_comp_gain": 0.0,
        "preheat_comp_max": 1.5,
    }

    def __init__(self, building_meta_data, action_spec=None):
        params = dict(self.PARAMS)
        self.adaptive_enabled = bool(params["preheat_adaptive_enabled"])
        self.safety = max(0.0, float(params["preheat_safety"]))
        self.slack_hours = max(0.0, float(params["preheat_slack_hours"]))
        self.min_hours = max(0.0, float(params["preheat_min_hours"]))
        self.max_hours = max(self.min_hours, float(params["preheat_max_hours"]))
        self.fallback_hours = float(np.clip(params["preheat_fallback_hours"],
                                            self.min_hours, self.max_hours))
        self.gate_k = float(params["preheat_gate_k"])
        self.min_samples = max(1, int(params["preheat_min_samples"]))
        self.forget = float(np.clip(params["preheat_forget"], 0.90, 0.999999))
        self.drift_slope = float(params["preheat_drift_slope"])
        self.target_margin = float(params["preheat_target_margin"])
        self.supply_ceiling = float(params["preheat_supply_ceiling"])
        self.update_steps = max(1, int(params["preheat_update_steps"]))
        self.debug = bool(params["preheat_debug"])
        self.comp_gain = float(np.clip(params["preheat_comp_gain"], 0.0, 1.0))
        self.comp_max = max(0.0, float(params["preheat_comp_max"]))
        super().__init__(building_meta_data, action_spec)

    # ------------------------------------------------------------------ state
    def reset(self):
        super().reset()
        n = self.n_zones
        # RLS on dT = d + g * x  (d in K/step, g in K per kW-step).
        self._fit = np.column_stack((np.zeros(n), np.full(n, 0.02)))
        self._cov = np.tile(np.array([[4.0, 0.0], [0.0, 0.25]]), (n, 1, 1))
        self._fit_updates = 0
        self._fit_temp = None          # temperature at which d was identified
        self._measured_rise = np.zeros(n)
        # The latch starts empty: only an identified plan (or the explicit
        # fallback below) may raise it, so a zone the model says needs less than
        # the fallback really does get a shorter ramp.
        self._horizon = np.zeros(n)
        self._horizon_step = -10 ** 9
        self._lag = np.zeros(n)
        self._band_low_asked = None
        self._ramping = False
        self.debug_log = []

    # -------------------------------------------------- online identification
    def _identify(self, applied, old, observed):
        """One RLS update of the Wu model structure, per zone.

        `applied` is the action Wu's `history` records for the transition that
        produced `observed` (conductance W/K, supply and discharge
        temperatures, delivered reheat W), `old` the previous observation.
        """
        conductance = np.asarray(applied["conductance"], dtype=float)
        supply = np.asarray(applied["supply"], dtype=float)
        reheat = np.asarray(applied["reheat"], dtype=float)
        y = np.clip(observed - old, -15.0, 15.0)
        x = (conductance * (supply - observed) + reheat) / 1000.0  # kW

        phi = np.column_stack((np.ones(self.n_zones), x))
        cov = self._cov
        scaled = np.einsum("nij,nj->ni", cov, phi)
        denom = self.forget + np.einsum("ni,ni->n", phi, scaled)
        gain = scaled / np.maximum(denom, 1e-9)[:, None]
        error = y - np.einsum("ni,ni->n", phi, self._fit)
        fit = self._fit + gain * error[:, None]
        cov = (cov - gain[:, :, None] * scaled[:, None, :]) / self.forget

        # Physical bounds keep a single wild sample from wrecking the plan.
        fit[:, 0] = np.clip(fit[:, 0], -2.0, 2.0)
        fit[:, 1] = np.clip(fit[:, 1], 1e-4, 0.5)
        cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))  # keep it symmetric
        diag = np.clip(np.einsum("nii->ni", cov), 1e-6, 1e3)
        cov[:, 0, 0] = diag[:, 0]
        cov[:, 1, 1] = diag[:, 1]
        self._fit, self._cov = fit, cov
        self._fit_updates += 1
        self._fit_temp = observed.copy()

        # Independent bound on the achievable ramp: a slowly decaying maximum
        # of the observed rise while the zone was being heated hard.
        heated = x > 3.0
        self._measured_rise = np.where(
            heated,
            np.maximum(self._measured_rise * 0.998, np.maximum(y, 0.0)),
            self._measured_rise * 0.998,
        )

    def _learn(self, observed):
        delay = int(self.parameters["observation_delay"])
        if self.previous_observed is not None and len(self.history) > delay:
            applied = list(self.history)[-(delay + 1)]
            self._identify(applied, self.previous_observed, observed)
        super()._learn(observed)

    # ------------------------------------------------------- readiness horizon
    def _conditioning_limits(self, obs):
        """Full-conditioning conductance (kW/K) and reheat ceiling (kW), per zone."""
        flow = np.full(self.n_zones, 2.0)
        valid = self.zone_flow_obs >= 0
        if np.any(valid):
            reading = obs[self.zone_flow_obs[valid]].astype(float)
            flow[valid] = np.where(np.isfinite(reading) & (reading > 0.0), reading, 2.0)
        flow = np.clip(flow, 0.01, 100.0)
        conductance = flow * self.AIR_CAPACITY / 1000.0
        water = self._temperature(obs, self.water_sensor,
                                  "supply_water_temperature_sensor", 340.0)
        supply = self.supply_ceiling
        valve = self.WATER_VALVE_CAPACITY * np.sqrt(20.0) * np.maximum(water - supply, 0.0)
        discharge = conductance * np.maximum(self.MAX_DISCHARGE_K - supply, 0.0)
        return conductance, np.minimum(valve / 1000.0, discharge)

    def _lead_rate(self, temperature, obs):
        """Predicted (lead hours, K/h rate) per zone under full conditioning.

        Returns None when the response has not been identified yet.
        """
        target = self.comfort_band[0] + self.target_margin
        gap = np.maximum(target - temperature, 0.0)
        if self._fit_updates < self.min_samples:
            return None
        drift = self._fit[:, 0].copy()
        gain = self._fit[:, 1].copy()
        if self._fit_temp is not None and self.drift_slope:
            # d was identified at one operating temperature; the outside
            # coupling in Wu's prior gives its slope in T (theta[:, 1] is
            # already the K/step sensitivity per unit (outside - T)/20).
            drift = drift + self.drift_slope * self.theta[:, 1] * (
                self._fit_temp - temperature) / 20.0
        conductance, reheat = self._conditioning_limits(obs)
        supply = self.supply_ceiling
        steps = np.zeros(self.n_zones)
        done = gap <= 1e-6
        current = temperature.copy()
        horizon = int(round(self.max_hours * STEPS_PER_HOUR)) + 1
        for k in range(horizon):
            if done.all():
                break
            heat = conductance * (supply - current) + reheat
            current = current + drift + gain * heat
            steps = np.where(done, steps, k + 1.0)
            done = done | (current >= target)
        lead = np.where(done, steps / STEPS_PER_HOUR, self.max_hours)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(gap > 1e-6, gap / np.maximum(lead, 1.0 / STEPS_PER_HOUR), 0.0)
        measured = self._measured_rise * STEPS_PER_HOUR
        rate = np.where(measured > 1e-3, np.minimum(rate, measured), rate)
        return lead, rate

    def _readiness(self, temperature, obs, remaining):
        """Per-zone preheat horizon in hours, latched for the day."""
        if self.step_count - self._horizon_step >= self.update_steps:
            self._horizon_step = self.step_count
            plan = self._lead_rate(temperature, obs)
            if plan is None:
                self._horizon[:] = self.fallback_hours
            else:
                lead, rate = plan
                target = self.comfort_band[0] + self.target_margin
                gap = np.maximum(target - temperature, 0.0)
                wanted = np.clip(self.safety * lead + self.slack_hours,
                                 self.min_hours, self.max_hours)
                short = gap - rate * self.fallback_hours
                # Only exceed the proven fallback horizon where the fallback is
                # predicted to leave the zone badly short; shortening is always
                # allowed (the plant does not need the time in warm climates).
                wanted = np.where(short > self.gate_k, wanted,
                                  np.minimum(wanted, self.fallback_hours))
                self._horizon = np.maximum(self._horizon, wanted)
                if self.debug:
                    self.debug_log.append(dict(
                        step=int(self.step_count), remaining=float(remaining),
                        horizon_min=float(self._horizon.min()),
                        horizon_mean=float(self._horizon.mean()),
                        horizon_max=float(self._horizon.max()),
                        lead_min=float(lead.min()), lead_max=float(lead.max()),
                        rate_mean=float(rate.mean()), rate_max=float(rate.max()),
                        gap_mean=float(gap.mean()), gap_max=float(gap.max()),
                        gated=int((short > self.gate_k).sum()),
                        drift_mean=float(self._fit[:, 0].mean()),
                        gain_mean=float(self._fit[:, 1].mean()),
                        updates=int(self._fit_updates),
                    ))
        return self._horizon

    # ------------------------------------------------------------------- band
    def _band(self, obs, temperature):
        low, high, now = super()._band(obs, temperature)
        was_ramping = self._ramping  # only the pre-arrival ramp measures the lag
        self._ramping = False
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
        remaining = (self.start_hour - self._hour(obs)) % 24.0
        if remaining > self.max_hours:  # window not open: forget yesterday
            self._horizon[:] = 0.0
            self._horizon_step = -10 ** 9
        soon = (bool(obs[self.comfort_soon_index] > 0.5)
                if self.comfort_soon_index is not None else False)
        if soon:  # Wu's own ramp is already running
            return low, high, now
        if self.holiday_guard_enabled and not self._comfort_day(obs, now):
            return low, high, now
        if not self.adaptive_enabled:
            # Bit-exact CtrlW: one fixed horizon and Wu's own progress formula.
            fixed = min(self.early_preheat_hours,
                        max(1.0, float(self.parameters["precondition_hours"])))
            if fixed <= 0.0 or remaining > fixed:
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
            recovery = max(1.0, float(self.parameters["recovery_steps"]))
            low = np.minimum(low, temperature + np.maximum((low - temperature) / recovery, 0.0))
            high = np.maximum(high, temperature - np.maximum((temperature - high) / recovery, 0.0))
            return low, high, now
        # Measured steady-state lag of the zones behind the ramp band: the
        # band is raised by that much so the *plant* arrives on time.
        if (self._band_low_asked is not None and was_ramping
                and self.comp_gain > 0.0):
            lag = np.clip(self._band_low_asked - temperature, 0.0, 3.0)
            self._lag += 0.05 * (lag - self._lag)
        horizon = self._readiness(temperature, obs, remaining)
        if remaining > float(horizon.max()) or remaining < 1.0:
            return low, high, now
        # Wu's blend, generalised: 0.5 at one hour left for any horizon, which
        # is exactly his 1 - remaining / precondition_hours when H = 2 h.
        progress = 0.5 * (horizon - remaining) / np.maximum(horizon - 1.0, 0.1)
        progress = np.clip(progress, 0.0, 0.5)
        if not np.any(progress > 0.0):
            return low, high, now
        low = (self.eco_band[0] + progress * (self.comfort_band[0] - self.eco_band[0]))
        high = (self.eco_band[1] + progress * (self.comfort_band[1] - self.eco_band[1]))
        margin = float(self.parameters["comfort_margin"])
        low = low + margin
        if self.comp_gain > 0.0:
            low = np.minimum(low + self.comp_gain * np.clip(self._lag, 0.0, self.comp_max),
                             self.comfort_band[0] + margin)
        high = high - margin
        recovery = max(1.0, float(self.parameters["recovery_steps"]))
        low = np.minimum(low, temperature + np.maximum((low - temperature) / recovery, 0.0))
        high = np.maximum(high, temperature - np.maximum((temperature - high) / recovery, 0.0))
        self._band_low_asked = low
        self._ramping = True
        return low, high, now
