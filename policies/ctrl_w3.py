"""CtrlW3 -- CtrlW with two band levers, measured against the LP bound.

`scratch/OptGap/lp.py` puts the reward loss of CtrlW 1.5-2x above the LP
optimum on the same window, and the gap is almost entirely gas: the LP heats to
the bottom of the band and nothing else, while CtrlW (a) keeps `comfort_margin`
inside the band, and (b) holds a night band the reward does not charge for,
because comfort regret is occupancy weighted and nobody is in the building
outside the work window.

Two switches, both off by default, so CtrlW3 with no flags is bit-identical to
CtrlW:

  band_edge     hold the exact band bottom: subtract `comfort_margin` from the
                low edge the base produced, so the AHU search targets the
                schedule's own heating setpoint instead of 0.12 K above it.
  float_night   outside the comfort window, and only when the preheat ramp is
                not already running, replace Wu's standby band (282..307 K)
                with (`float_floor`..307 K).  The floor is the knob: it has to
                be low enough to save night losses and high enough that the
                morning ramp can recover inside the reheat capacity, which is
                what the LP's own capacity constraints say.  Wu's one-step
                recovery clamp still applies.

Nothing else changes: the AHU search, the residual ensemble, the fuel weights
and the occupancy release are CtrlW's.
"""

from __future__ import annotations

import numpy as np

try:  # packaged notebook: the embedded modules are top-level names
    from ctrl_w import CtrlW
except ImportError:  # workspace: policies/ is a package
    from .ctrl_w import CtrlW


class CtrlW3(CtrlW):
    """CtrlW plus an exact band edge and a tunable free-floating night."""

    name = "ctrl_w3"

    PARAMS = dict(CtrlW.PARAMS)
    PARAMS.update({
        "band_edge": True,
        "float_night": True,
        "float_floor": 285.0,
    })

    def __init__(self, building_meta_data, action_spec=None):
        super().__init__(building_meta_data, action_spec)
        params = dict(self.PARAMS)
        self.band_edge = bool(params["band_edge"])
        self.float_night = bool(params["float_night"])
        self.float_floor = float(params["float_floor"])

    def _band(self, obs, temperature):
        low, high, now = super()._band(obs, temperature)
        if not (self.band_edge or self.float_night):
            return low, high, now
        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)
        if self.band_edge:
            low = low - float(self.parameters["comfort_margin"])
        if self.float_night and not now:
            standby = float(self.parameters["standby_minimum"])
            if np.all(low <= standby + 0.01):  # Wu's standby band, not the ramp
                low = np.full(self.n_zones, self.float_floor)
                high = np.full(self.n_zones, float(self.parameters["standby_maximum"]))
                recovery = max(1.0, float(self.parameters["recovery_steps"]))
                low = np.minimum(low, temperature + np.maximum((low - temperature) / recovery, 0.0))
                high = np.maximum(high, temperature - np.maximum((temperature - high) / recovery, 0.0))
        return low, high, now


class CtrlW3Edge(CtrlW3):
    """Exact band edge only."""

    name = "ctrl_w3_edge"
    PARAMS = dict(CtrlW3.PARAMS)
    PARAMS.update({"float_night": False})


class CtrlW3Float(CtrlW3):
    """Free-floating night only."""

    name = "ctrl_w3_float"
    PARAMS = dict(CtrlW3.PARAMS)
    PARAMS.update({"band_edge": False})
