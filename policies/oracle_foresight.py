"""OracleForesightCtrlW -- CtrlW plus *perfect* per-zone occupancy foresight.

Analysis-only upper bound on the value of knowing the day's occupancy schedule.
``sbx/evalrun --oracle foresight --oracle-file <recorded schedule>`` hands the
policy a dense ``(steps x zones)`` matrix of the reference run's per-zone
``average_occupancy`` -- recorded from reward infos, so the occupancy model is
never queried twice and the trajectory is untouched. Because occupancy is
exogenous (fixed seed 17321, forum B7; verified identical across two different
policies on the same episode), that matrix is the *realised future* of this
episode: the policy knows, for every zone, today's first arrival and last
departure before the day starts.

Bands, per zone, replacing CtrlW's building-wide schedule:

* a zone with no arrival today stays at the standby band all day;
* a zone with arrivals gets the comfort band from its own first arrival to its
  own last departure, with Wu's eco -> comfort ramp over the
  ``foresight_preheat_lead_hours`` window before the arrival;
* after the last departure the zone drops to standby for the rest of the day.

This is what a controller could do with a perfect calendar/sensor fusion; it is
**never submitted** (the observation space exposes neither zonal occupancy nor
a schedule) and it is the strongest form of the occupancy information measured
in ``docs/voi_oracle.md``.

Safety: if the plan says nobody is present while the public ``num_occupants``
count says otherwise, the comfort band is restored for the whole building
(``foresight_safety``), so a misaligned recording degrades to CtrlW instead of
to an unconditioned building.
"""

from __future__ import annotations

import numpy as np

from .ctrl_w import CtrlW
from .oracle_util import clock

try:  # packaged notebook
    from oracle_util import clock as _clock  # noqa: F401
except ImportError:  # workspace package
    pass


STEPS_PER_HOUR = 12
# Longest day in steps: a 24 h day is 288; the episodes contain DST days that
# are 12 steps longer/shorter, so the day window is cut at the plan length.
MAX_DAY_STEPS = 300


class OracleForesightCtrlW(CtrlW):
    """Per-zone bands from a recorded, perfect occupancy schedule."""

    name = "oracle_foresight_ctrl_w"

    PARAMS = {
        **CtrlW.PARAMS,
        "foresight_preheat_lead_hours": 2.0,
        "foresight_occupied_threshold": 0.0,
        "foresight_safety": True,
        "foresight_standby_after_departure": True,
    }

    def __init__(self, building_meta_data, action_spec=None):
        params = dict(self.PARAMS)
        self.foresight_lead_steps = max(
            0, int(round(float(params["foresight_preheat_lead_hours"]) * STEPS_PER_HOUR))
        )
        self.foresight_threshold = float(params["foresight_occupied_threshold"])
        self.foresight_safety = bool(params["foresight_safety"])
        self.foresight_standby_after_departure = bool(
            params["foresight_standby_after_departure"]
        )
        super().__init__(building_meta_data, action_spec)

    # ------------------------------------------------------------------ state
    def reset(self):
        super().reset()
        self._plan = None
        self._plan_step = None
        self._last_clock_hour = None
        self._day_start = 0
        self._day_start_hour = None
        self.foresight_fallbacks = 0
        self.foresight_steps = 0

    # ----------------------------------------------------------------- oracle
    def observe_oracle(self, info):
        plan = info.get("foresight_occupancy") if info else None
        if plan is None:
            return
        self._plan = np.asarray(plan, dtype=float)
        step = int(info.get("foresight_step") or 0)
        self._plan_step = step
        _, hour = clock(info.get("timestamp"))
        if self._last_clock_hour is None:
            self._day_start_hour = hour  # episode starts mid-day
        elif hour < self._last_clock_hour - 12.0:
            self._day_start = step  # local midnight
            self._day_start_hour = hour
        self._last_clock_hour = hour
        self.foresight_steps += 1

    # ------------------------------------------------------------------ masks
    def _plan_masks(self, obs):
        """(on, soon, progress) per zone for the current step, or None."""
        plan, step = self._plan, self._plan_step
        if plan is None or step is None or step >= plan.shape[0]:
            return None
        # The day ends at the next local midnight: the hours left from the step
        # the day started on (an episode can start mid-day).
        start_hour = 0.0 if self._day_start_hour is None else float(self._day_start_hour)
        day_steps = int(round((24.0 - start_hour) * STEPS_PER_HOUR))
        day_steps = max(1, min(MAX_DAY_STEPS, day_steps))
        stop = min(plan.shape[0], self._day_start + day_steps)
        window = plan[self._day_start:stop] > self.foresight_threshold
        if window.shape[0] == 0:
            return None
        occupied = window.any(axis=0)
        first = np.where(occupied, window.argmax(axis=0), -1)
        last = np.where(
            occupied, window.shape[0] - 1 - window[::-1].argmax(axis=0), -1
        )
        now = step - self._day_start
        on = occupied & (now >= first) & (now <= last)
        soon = occupied & (now >= first - self.foresight_lead_steps) & (now < first)
        lead = max(1, self.foresight_lead_steps)
        progress = np.clip(1.0 - (first - now) / float(lead), 0.0, 1.0)
        return on, soon, progress

    # ------------------------------------------------------------------- band
    def _band(self, obs, temperature):
        masks = self._plan_masks(obs)
        if masks is None:
            return super()._band(obs, temperature)
        on, soon, progress = masks
        if self.foresight_safety and self._occupied(obs) and not on.any():
            # The plan and the public count disagree: fall back to CtrlW.
            self.foresight_fallbacks += 1
            return super()._band(obs, temperature)

        margin = float(self.parameters["comfort_margin"])
        comfort_low = float(self.comfort_band[0]) + margin
        comfort_high = float(self.comfort_band[1]) - margin
        eco_low = float(self.eco_band[0]) + margin
        eco_high = float(self.eco_band[1]) - margin
        ramp_low = eco_low + progress * (comfort_low - eco_low)
        ramp_high = eco_high + progress * (comfort_high - eco_high)
        standby_low = np.full(self.n_zones, float(self.parameters["standby_minimum"]))
        standby_high = np.full(self.n_zones, float(self.parameters["standby_maximum"]))

        if self.foresight_standby_after_departure:
            low = np.where(on, comfort_low, np.where(soon, ramp_low, standby_low))
            high = np.where(on, comfort_high, np.where(soon, ramp_high, standby_high))
        else:
            low = np.where(on | soon, np.maximum(comfort_low, ramp_low), standby_low)
            high = np.where(on | soon, np.maximum(comfort_high, ramp_high), standby_high)

        # Never request the whole recovery in one five-minute step (same clamp
        # as Wu's own band).
        recovery = max(1.0, float(self.parameters["recovery_steps"]))
        low = np.minimum(low, temperature + np.maximum((low - temperature) / recovery, 0.0))
        high = np.maximum(
            high, temperature - np.maximum((temperature - high) / recovery, 0.0)
        )
        low = np.nan_to_num(low, nan=float(self.parameters["standby_minimum"]))
        high = np.nan_to_num(high, nan=float(self.parameters["standby_maximum"]))
        base_now = bool(
            obs[self.comfort_now_index] > 0.5
        ) if self.comfort_now_index is not None else True
        return low, high, base_now
