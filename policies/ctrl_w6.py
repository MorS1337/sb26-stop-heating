"""CtrlW6 -- the cooling-season supervisory layer, and the measurement that it
should be left off.

CtrlW5's 13.8 % came from two things that only exist in heating: the plant was
*running for nothing* overnight (Wu's 282 K standby floor held the coldest zone
of every AHU group at 8.9 C and kept the group's boiler loop on all night), and
the coil's *efficiency depended on the delivery temperature*
(`gas/Q = 1 + 0.3(T_zone - T_out)/(T_dis - T_zone)`, so the recovery is 40 %
cheaper delivered in one burst at the 308.15 K cap).

Neither holds in cooling. Measured on `building_train_1/phoenix_5` (7 days,
36.4 C mean outside) and `building_train_1/singapore_8`:

* the plant is already off in every unoccupied hour and all weekend -- chiller
  energy in 19:00-04:00 is **0.5 %** of the bill and on weekends **0.0 %**, and
  the hottest zone of the week (31.2 C) never reaches the 33.9 C standby
  ceiling, so raising `standby_maximum` from 307 to 320 K is **bit-identical**.
  The only unoccupied block that costs anything is the deliberate 04:00-06:00
  pre-arrival ramp (9.1 %), and that is what makes the occupied comfort loss
  exactly zero;
* the cooling coil has a **constant COP of 3.5** and no capacity limit
  (`chiller = flow*rho*cp*(T_mixed - T_supply)/3.5`), so unlike heating there is
  no efficiency gain from a colder delivery -- only the *total airflow* matters,
  and it is already minimal: the supply-air setpoint is at the 285 K floor of
  the action space on **100 %** (Phoenix) / 97.5 % (Singapore) / 91.8 %
  (Atlanta_5) of live AHU-steps, and recirculation is at the 0.70 device cap
  (30 % outside air) on 97-100 % of them.

So every cooling knob below is **inert at its default** and the class is
bit-identical to CtrlW5-278:

    CtrlW6.PARAMS  ==  CtrlW5.PARAMS | {"standby_minimum": 278.0}

Measured: `-24.56561696150714` on `building_train_1/phoenix_5` and
`-86.68101...` on `building_train_1/fairbanks_1`, 2016 steps, identical to
`policies.ctrl_w5:CtrlW5` with `standby_minimum = 278.0`.

The knobs are kept because each one is the reproduction handle for a measured
refutation; all the numbers are in `docs/cool_gap.md` §4.

Knobs (all off by default; a *negative* delta means worse, the totals are
negative, and every one of these loses):

`cool_margin_delta_K` -- raise the occupied band's upper edge by this much,
    eating into `comfort_margin`.  The reward charges nothing inside the band, so
    the top 0.12 K looks like free cooling headroom.  It is not: raising it by
    the full 0.12 K costs **5.02 reward units on Phoenix_5** (comfort 0.000 ->
    -5.467, energy +0.37) because the occupied zones then sit an average of
    0.017 K above the reward's own 297 K edge, and a kelvin of occupied
    deviation is worth 298 units a week against 0.62 units of cooling.  The
    margin is exactly the controller's tracking error, not slack.

`precool_hours` / `precool_ceiling_offset_K` -- widen the pre-arrival window to
    `precool_hours` before `morning_start_hour` and drive its ceiling to
    `comfort_band[0] + precool_ceiling_offset_K`, i.e. cool the mass to the
    bottom of the comfort band while electricity is nominally cheapest.  Loses
    3.9-6.9 units on Phoenix_5 and 2.8-4.2 on Singapore_8 at every setting
    (2/3/4 h at two ceilings), and costs the heating anchors too.  The reason is
    in the reward: its electricity term is *price-normalised*
    (`elec.cost(p)/elec.cost(max_electricity_rate)` at the same hour), so the
    time-of-use price cancels and the whole-week marginal weight of an
    electrical watt varies by only **5.7 %**; moving a week's entire cooling to
    the cheapest hour is worth 0.91 units on Phoenix_5, and the building stores
    only ~146 kWh/K against a 1.7 MWh_th daily load.

`flush_K` / `flush_min_gap_K` / `flush_floor_offset_K` -- a night flush while
    the building is unoccupied and the outside air is at least
    `flush_min_gap_K` colder than the mean zone temperature.  `flush_free`
    selects the physical version: 100 % outside air at the static-pressure floor
    with the supply setpoint above the mixed-air temperature, so the coil never
    runs and the air handler becomes a fan moving cool night air.  Bit-identical
    where the night air is not colder (Phoenix, Singapore); loses 0.64 units on
    Atlanta_5, 0.15 on Fairbanks_1 and 0.08 on Atlanta_8 in the mechanical
    version, and loses in the free version too -- the stored coolth is paid back
    by the extra envelope gain of a colder building for the whole next day.

`standby_maximum` (from Wu) and `release_hour` (from CtrlW) are the two knobs
the brief names for the unoccupied side.  `standby_maximum` 307 -> 320 is
bit-identical; `release_hour` 12 -> 6 makes all three cooling anchors worse by
0.36-0.37 and Atlanta_8 worse by 0.74.

Submission rules: numpy only, no `smart_buildings` import, no building or
weather id, no future information, and at the defaults this file is a strict
subclass that changes no number.
"""

from __future__ import annotations

import math

import numpy as np

try:  # packaged notebook: the embedded modules are top-level names
    from ctrl_w5 import CtrlW5
except ImportError:  # workspace: policies/ is a package
    from .ctrl_w5 import CtrlW5


class CtrlW6(CtrlW5):
    """CtrlW5-278 plus an opt-in cooling-season layer that measures out to zero."""

    name = "ctrl_w6"

    PARAMS = {
        **CtrlW5.PARAMS,
        "standby_minimum": 278.0,
        # --- cooling side: inert at 0.0 / False ------------------------------
        "cool_margin_delta_K": 0.0,
        "precool_hours": 0.0,
        "precool_ceiling_offset_K": 0.0,
        "flush_K": 0.0,
        "flush_min_gap_K": 2.0,
        "flush_floor_offset_K": 0.0,
        "flush_free": False,
    }

    def __init__(self, building_meta_data, action_spec=None):
        params = dict(self.PARAMS)
        self.cool_margin_delta = max(0.0, float(params["cool_margin_delta_K"]))
        self.precool_hours = max(0.0, float(params["precool_hours"]))
        self.precool_ceiling_offset = float(params["precool_ceiling_offset_K"])
        self.flush_K = max(0.0, float(params["flush_K"]))
        self.flush_min_gap = max(0.0, float(params["flush_min_gap_K"]))
        self.flush_floor_offset = float(params["flush_floor_offset_K"])
        self.flush_free = bool(params["flush_free"])
        self._flush_on = False
        super().__init__(building_meta_data, action_spec)

    # ------------------------------------------------------------- mechanisms
    def _band(self, obs, temperature):
        low, high, now = super()._band(obs, temperature)
        self._flush_on = False
        if now:
            if self.cool_margin_delta > 0.0:
                # The edge after `super()._band` is
                # `min(sensor, comfort_high) - comfort_margin`, so adding back
                # at most `comfort_margin` can never leave the schedule band.
                cap = min(self.comfort_band[1],
                          float(self.parameters["comfort_margin"]))
                high = high + min(self.cool_margin_delta, cap)
            return low, high, now
        if self.precool_hours > 0.0:
            remaining = (self.start_hour - self._hour(obs)) % 24.0
            if 0.0 < remaining <= self.precool_hours:
                ceiling = self.comfort_band[0] + self.precool_ceiling_offset
                high = np.minimum(high, ceiling + np.zeros(self.n_zones))
        if self.flush_K > 0.0:
            mean_zone = float(np.average(temperature, weights=self.zone_area))
            outside = self._outside(obs)
            floor = self.comfort_band[0] + self.flush_floor_offset
            if (outside is not None and mean_zone - outside >= self.flush_min_gap
                    and mean_zone > floor):
                self._flush_on = True
                if not self.flush_free:
                    high = np.maximum(np.minimum(high, temperature - self.flush_K),
                                      floor + np.zeros(self.n_zones))
        return low, high, now

    def _choose_ahu(self, group, temperature, free, low, high, max_flow,
                    outside, water):
        if self.flush_free and self._flush_on:
            # 100 % outside air at the static-pressure floor, with the supply
            # setpoint above the mixed-air temperature so the coil does not run:
            # the air handler becomes a fan that moves cool night air, which is
            # what a night flush is.
            ratio = math.sqrt(self.MIN_PRESSURE / self.DESIGN_PRESSURE)
            flow = max_flow[group] * ratio
            return {"run": 1, "pressure": float(self.MIN_PRESSURE),
                    "setpoint": float(outside) + 1.0, "recirc": 0.0,
                    "damper": np.ones(group.size),
                    "conductance": flow * self.AIR_CAPACITY,
                    "supply": np.full(group.size, float(outside)),
                    "reheat": np.zeros(group.size)}
        return super()._choose_ahu(group, temperature, free, low, high, max_flow,
                                   outside, water)

    # ---------------------------------------------------------------- helpers
    def _outside(self, obs) -> float | None:
        """Mean outside-air temperature over the air handlers, from the obs."""
        values = [self._temperature(obs, index, "outside_air_temperature_sensor",
                                    float("nan"))
                  for index in self.outside_obs if index is not None]
        values = [v for v in values if np.isfinite(v)]
        return float(np.mean(values)) if values else None
