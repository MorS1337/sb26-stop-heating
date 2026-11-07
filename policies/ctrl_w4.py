"""CtrlW4 -- CtrlW2 (adaptive preheat horizon) plus a fixed pre-dawn night lift.

Why
---
`docs/voi_oracle.md` §4-6 measured what a 12 h outside-temperature forecast is
worth to CtrlW and found that almost none of the gain is information: a *fixed*
2 K lift of the night setback floor -- no forecast, no occupancy, no oracle of
any kind -- reproduces the forecast variant's −0.50% on the 10-scenario VoI
subset (−0.51% for the forecast-driven lift, +0.01% apart), and it does so by
reading only the clock, the schedule predicate and the plant state that the
observation already exposes.  The forecast variant's own `_night_lift` is that
mechanism; this module ports it onto CtrlW2 so the two *legal* levers -- the
adaptive preheat horizon (W1b, −0.16% on dev30) and the pre-dawn lift -- can be
stacked.

Mechanism
---------
Whenever CtrlW2's band call is in the *night* mode (`now` false: no comfort
window open, so the zone is being held at Wu's standby floor), the floor is
lifted by

    lift = night_lift_K * night_lift_share

for the zones that satisfy all of

* `0 < hours_until(start_hour, hour) <= night_lift_lead_hours` -- inside the
  pre-dawn window (4 h by default, so 02:00-06:00 for a 06:00 comfort start),
* the comfort-day predicate `_comfort_day(obs, now)` holds -- no lift is spent
  on a weekend or a declared holiday, exactly as CtrlW's own guard decides it,
* the zone is sitting at the standby floor,
  `low <= standby_minimum + night_lift_floor_K` -- which is why the mechanism is
  inert on the cooling-dominated stations (their zones are above the floor and
  the band they get is not the standby band),

and the lifted floor is capped at `comfort_band[0] - comfort_margin` and passed
through the same one-step recovery clamp CtrlW applies to every band
(`low = min(low, T + (low - T) / recovery_steps)`), so the lift can never ask
for more than a single step's worth of heat above the current zone temperature.

Rationale (from the VoI study): a deep setback before a cold morning saves one
night of gas and then pays it back twice -- a larger reheat ramp inside the
window where the gas/carbon price is highest, and a colder building at the first
arrival.  Lifting the floor at 02:00-06:00 is the cheapest heat of the day
(`elec_hour_blend`/`GAS_RATIO` put the 02:00 gas price at 0.4734 weighted units
against 0.4447 at 15:00) and it is *charged at the dawn price* against a smaller
later ramp.

Measured (full 8641-step episodes, sims/shipped, components hook)
------------------------------------------------------------------
18 dev30 scenarios measured (the 12 fairbanks+phoenix of the CtrlW2 study plus
atlanta_2, atlanta_8, denver_8, denver_6, singapore_8, singapore_7), scored as
mean over scenarios of |mean per-step reward|:

    subset                     CtrlW    CtrlW2 v4    CtrlW4
    12 fairbanks+phoenix      0.011201   0.011180    0.011146  (-0.49% / -0.30% vs v4)
    10 VoI subset             0.011920   0.011896    0.011856  (-0.54% / -0.34% vs v4)
    18 measured               0.009112   0.009094    0.009067  (-0.50% / -0.30% vs v4)

Better on 5/12 and 4/10, worst scenario +1.34% (train_4/phoenix_8), None above
the 3% guard.  dev30 estimate with the 12 unmeasured scenarios left at their
CtrlW2 v4 values: 0.0073415 (-0.22% vs v4, -0.38% vs CtrlW).  The lift's own
contribution reproduces the VoI study: `results/voi_night_fixed` (the same fixed
lift on the CtrlW base) scores 0.011860 on the VoI subset against CtrlW4's
0.011856, i.e. the two levers are additive to 0.03% of that subset.

The trade-off is explicit: over the 18 scenarios comfort regret goes -8.90 ->
-13.50 (CtrlW2 v4 -> CtrlW4) and hour-06 comfort -7.05 -> -11.29, while total
energy is 0.6% and carbon 1.0% cheaper -- the lift buys cheap 02:00-06:00 heat at
a lower marginal price than the ramp heat it displaces, and the arrival hour pays
for it.  Warm stations are inert when their zones are not on the standby floor
(phoenix_5/6/3/4 and both singapore runs are bit-identical to CtrlW2), but two
warm scenarios do fire and lose 0.7-1.3%.

Exactness (2 scenarios x 576 steps): with `night_lift_enabled = false` this class
is bit-identical to CtrlW2, and with the lift on it is bit-identical to
`OracleFcstCtrlW`'s fixed-share `_night_lift` stacked on CtrlW2
(`scratch/AdaptivePreheat/stack_lift.py`) -- total reward, components and by_hour.
See `docs/adaptive_preheat.md`, section "CtrlW4: the fixed 2 K pre-dawn night
lift stacked on CtrlW2", for the per-scenario tables and the go/no-go.

Knobs (all in PARAMS)
---------------------
    night_lift_enabled      True   master switch (False == CtrlW2, bit-exact)
    night_lift_K            2.0    lift of the night setback floor, K
    night_lift_share        1.0    scale on the lift (the VoI fixed share)
    night_lift_lead_hours   4.0    how long before comfort start the lift runs
    night_lift_floor_K      1.0    a zone counts as "at the floor" within this

Nothing here reads a building id, a weather file, the occupancy seed, the
forecast or the future: only the observation, the public metadata and CtrlW2's
own online state.
"""

from __future__ import annotations

import numpy as np

try:  # packaged notebook: the embedded modules are top-level names
    from ctrl_w2 import CtrlW2
except ImportError:  # workspace: policies/ is a package
    from .ctrl_w2 import CtrlW2


def _hours_until(target_hour: float, hour: float) -> float:
    """Hours from ``hour`` forward to the next occurrence of ``target_hour``."""
    return (target_hour - hour) % 24.0


class CtrlW4(CtrlW2):
    """CtrlW2 + a fixed, clock-gated pre-dawn lift of the night setback floor."""

    name = "ctrl_w4"

    PARAMS = {
        **CtrlW2.PARAMS,
        "night_lift_enabled": True,
        "night_lift_K": 2.0,
        "night_lift_share": 1.0,
        "night_lift_lead_hours": 4.0,
        "night_lift_floor_K": 1.0,
    }

    def __init__(self, building_meta_data, action_spec=None):
        params = dict(self.PARAMS)
        self.night_lift_enabled = bool(params["night_lift_enabled"])
        self.night_lift_K = max(0.0, float(params["night_lift_K"]))
        self.night_lift_share = min(max(float(params["night_lift_share"]), 0.0), 1.0)
        self.night_lift_lead_hours = max(0.0, float(params["night_lift_lead_hours"]))
        self.night_lift_floor_K = max(0.0, float(params["night_lift_floor_K"]))
        super().__init__(building_meta_data, action_spec)

    # ------------------------------------------------------------- mechanisms
    def _night_lift(self, obs, low, temperature):
        """Raise the standby floor of every zone that is sitting on it."""
        if not self.night_lift_enabled or self.night_lift_K <= 0.0:
            return low
        share = self.night_lift_share
        if share <= 0.0:
            return low
        remaining = _hours_until(self.start_hour, self._hour(obs))
        if remaining > self.night_lift_lead_hours or remaining <= 0.0:
            return low
        if not self._comfort_day(obs, False):
            return low  # no comfort window coming (weekend / holiday)
        margin = float(self.parameters["comfort_margin"])
        ceiling = self.comfort_band[0] - margin
        at_floor = low <= float(self.parameters["standby_minimum"]) + self.night_lift_floor_K
        target = np.minimum(low + self.night_lift_K * share, ceiling)
        low = np.where(at_floor, np.maximum(low, target), low)
        recovery = max(1.0, self.parameters["recovery_steps"])
        return np.minimum(
            low, temperature + np.maximum((low - temperature) / recovery, 0.0)
        )

    # ------------------------------------------------------------------- band
    def _band(self, obs, temperature):
        low, high, now = super()._band(obs, temperature)
        if not now:
            low = self._night_lift(obs, low, temperature)
        return low, high, now
