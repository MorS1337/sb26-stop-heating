"""OracleFcstCtrlW -- CtrlW plus an outside-air-temperature forecast.

``sbx.evalrun --oracle forecast`` hands the policy the outside air temperature in
K for the next 1..12 hours, read from the episode's own weather controller at the
current simulated clock. A submission cannot see it (only the current outside
temperature reaches the observation), so this variant measures what a short
numerical weather forecast would be worth to a transferable controller.

Two mechanisms, both keyed on the forecast temperature in the arrival window
(comfort start +/- 1 h):

``fcst_preheat_enabled``
    Sets the eco -> comfort preheat horizon from how much lift the building will
    need at arrival: ``demand = max(comfort_low - T_coldest, T_warmest -
    comfort_high, 0)`` K, mapped linearly from ``fcst_preheat_hours_min`` at
    demand 0 to ``fcst_preheat_hours_max`` at ``fcst_demand_full_K``.
    ``precondition_hours`` is temporarily set to the same value so Wu's
    ``1 - remaining/duration`` ramp stays continuous -- the controller's own
    continuity argument, extended to a per-step duration.

    The default ceiling is 2 h, not 3 h, because the dev30 horizon scan is
    already flat-topped at 2 h on both cold and warm stations: per scenario,
    ``h2`` beats Wu's 1 h and 3 h/4 h on 25 of 30 scenarios, Fairbanks
    included. A forecast can therefore only *shorten* the ramp towards the mild
    end without giving back a measured gain -- which is exactly what the sweep
    in ``docs/voi_oracle.md`` tests.

``fcst_night_enabled``
    Lifts the night setback floor by up to ``fcst_night_lift_K`` during the
    ``fcst_night_lead_hours`` before comfort start, scaled by the same demand.
    Rationale: a deep setback before a very cold morning buys one night of
    savings and then pays for it twice -- a large reheat ramp inside the window
    where the gas/carbon price is highest, and a cold building at the first
    arrival. Measuring it is the point.

``fcst_night_fixed_share``
    Ablation knob (default ``None``): a constant lift share that ignores the
    forecast, so the lift runs on the controller's own clock. ``1.0`` with
    ``fcst_preheat_enabled = false`` is a **legal** CtrlW variant -- it reads no
    oracle at all -- worth -0.5% on the 10-scenario subset in
    ``docs/voi_oracle.md``.

Never submit: it consumes a signal the competition does not provide.
"""

from __future__ import annotations

import numpy as np

from .ctrl_w import CtrlW
from .oracle_util import clock, hours_until

try:  # packaged notebook
    from oracle_util import clock as _clock, hours_until as _hours_until  # noqa: F401
except ImportError:  # workspace package
    pass


class OracleFcstCtrlW(CtrlW):
    """Forecast-driven preheat horizon and night setback."""

    name = "oracle_fcst_ctrl_w"

    PARAMS = {
        **CtrlW.PARAMS,
        "fcst_preheat_enabled": True,
        "fcst_preheat_hours_min": 1.0,
        "fcst_preheat_hours_max": 2.0,
        # Arrival-window temperature deficit at which the horizon saturates.
        "fcst_demand_full_K": 30.0,
        "fcst_night_enabled": True,
        "fcst_night_lead_hours": 4.0,
        "fcst_night_lift_K": 2.0,
        # Ablation: a *fixed* lift share (0..1) that needs no forecast at all.
        # None keeps the forecast-driven share. 1.0 = the same 2 K lift on every
        # cold morning regardless of the forecast, i.e. a legal controller.
        "fcst_night_fixed_share": None,
    }

    def __init__(self, building_meta_data, action_spec=None):
        params = dict(self.PARAMS)
        self.fcst_preheat_enabled = bool(params["fcst_preheat_enabled"])
        self.fcst_preheat_hours_min = max(0.0, float(params["fcst_preheat_hours_min"]))
        self.fcst_preheat_hours_max = max(
            self.fcst_preheat_hours_min, float(params["fcst_preheat_hours_max"])
        )
        self.fcst_demand_full_K = max(1e-6, float(params["fcst_demand_full_K"]))
        self.fcst_night_enabled = bool(params["fcst_night_enabled"])
        self.fcst_night_lead_hours = max(0.0, float(params["fcst_night_lead_hours"]))
        self.fcst_night_lift_K = max(0.0, float(params["fcst_night_lift_K"]))
        fixed_share = params.get("fcst_night_fixed_share")
        self.fcst_night_fixed_share = (
            None if fixed_share is None else min(max(float(fixed_share), 0.0), 1.0)
        )
        super().__init__(building_meta_data, action_spec)
        # CtrlW's own numbers, restored after every band call.
        self._base_early_preheat_hours = float(self.early_preheat_hours)
        self._base_precondition_hours = float(self.parameters["precondition_hours"])

    # ------------------------------------------------------------------ state
    def reset(self):
        super().reset()
        self._fcst = None
        self._fcst_hour = None
        self._fcst_weekday = None
        self._fcst_stamp = None
        self._fcst_demand = None

    # ----------------------------------------------------------------- oracle
    def observe_oracle(self, info):
        """Consumes ``forecast_oat`` (K, next 1..H hours) and the clock."""
        if not info:
            return
        forecast = info.get("forecast_oat")
        if forecast is None:
            return
        forecast = np.asarray(forecast, dtype=float).reshape(-1)
        if forecast.size == 0:
            return
        self._fcst = forecast
        self._fcst_weekday, self._fcst_hour = clock(info.get("timestamp"))
        self._fcst_stamp = info.get("timestamp")
        self._fcst_demand = self._arrival_demand()

    # ------------------------------------------------------------- mechanisms
    def _arrival_demand(self):
        """Temperature the ramp must bridge at the coming comfort start, in K.

        Takes the coldest and warmest forecast hour around the arrival
        (computed from the newest forecast every step, so the value is causal),
        and measures the larger of the heating and the cooling gap to the
        public comfort band.
        """
        forecast, hour = self._fcst, self._fcst_hour
        if forecast is None or hour is None:
            return None
        usable = forecast[np.isfinite(forecast)]
        if usable.size == 0:
            return None
        remaining = hours_until(self.start_hour, hour)
        # Forecast index 0 is +1 h, so the arrival hour is index round(remaining)-1.
        centre = int(round(remaining)) - 1
        centre = min(max(centre, 0), forecast.size - 1)
        window = forecast[max(centre - 1, 0): min(centre + 1, forecast.size - 1) + 1]
        window = window[np.isfinite(window)]
        if window.size == 0:
            return None
        heating = self.comfort_band[0] - float(window.min())
        cooling = float(window.max()) - self.comfort_band[1]
        return max(heating, cooling, 0.0)

    def _preheat_hours(self):
        if not self.fcst_preheat_enabled or self._fcst_demand is None:
            return None
        share = min(max(self._fcst_demand / self.fcst_demand_full_K, 0.0), 1.0)
        return self.fcst_preheat_hours_min + share * (
            self.fcst_preheat_hours_max - self.fcst_preheat_hours_min
        )

    def _night_lift(self, obs, low, temperature):
        if not self.fcst_night_enabled or self.fcst_night_lift_K <= 0.0:
            return low
        if self.fcst_night_fixed_share is None:
            if self._fcst_hour is None or self._fcst_demand is None:
                return low
            hour = self._fcst_hour
            share = min(max(self._fcst_demand / self.fcst_demand_full_K, 0.0), 1.0)
        else:
            # Ablation without the oracle: the clock comes from the observation.
            hour = self._hour(obs)
            share = self.fcst_night_fixed_share
        remaining = hours_until(self.start_hour, hour)
        if remaining > self.fcst_night_lead_hours or remaining <= 0.0:
            return low
        if not self._comfort_day(obs, False):
            return low  # no comfort window coming (weekend / holiday)
        if share <= 0.0:
            return low
        margin = float(self.parameters["comfort_margin"])
        ceiling = self.comfort_band[0] - margin
        at_floor = low <= float(self.parameters["standby_minimum"]) + 1.0
        target = np.minimum(low + self.fcst_night_lift_K * share, ceiling)
        low = np.where(at_floor, np.maximum(low, target), low)
        recovery = max(1.0, self.parameters["recovery_steps"])
        return np.minimum(
            low, temperature + np.maximum((low - temperature) / recovery, 0.0)
        )

    # ------------------------------------------------------------------- band
    def _band(self, obs, temperature):
        hours = self._preheat_hours()
        if hours is None:
            # Horizon mechanism off: CtrlW's own band, but the night lift still
            # has to run (it is an independent mechanism).
            low, high, now = super()._band(obs, temperature)
        else:
            window = max(1.0, float(hours))
            # Extend both the window and Wu's ramp duration by the same amount,
            # so the eco -> comfort morph still lands exactly on the comfort band
            # at the arrival hour (see CtrlW's continuity argument).
            self.early_preheat_hours = window
            self.parameters["precondition_hours"] = window
            try:
                low, high, now = super()._band(obs, temperature)
            finally:
                self.early_preheat_hours = self._base_early_preheat_hours
                self.parameters["precondition_hours"] = self._base_precondition_hours
        if not now:
            low = self._night_lift(obs, low, temperature)
        return low, high, now
