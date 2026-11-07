"""OracleOccCtrlW -- CtrlW plus per-zone occupancy, an analysis-only upper bound.

The submission-legal ``CtrlW`` can only see the building-total ``num_occupants``
field, so its ``(a)`` release has to wait until the *whole* building is empty:
one person in one zone keeps every zone at the comfort band. The reward does not
work that way -- productivity regret is weighted by each zone's own
``average_occupancy`` -- so conditioning an empty zone is pure energy cost.

This variant receives ``zone_occupancy`` from ``sbx.evalrun --oracle occupancy``:
the per-zone ``average_occupancy`` of the **last completed** interval, i.e. a
realistic 5-minute-late zonal occupancy sensor (no extra model query, so the
episode's arrival draw is untouched). After ``occ_release_hour`` (noon, when
this occupancy model's arrival window has closed and a departure is final for
the day) a zone that has been empty for ``occ_release_empty_steps`` consecutive
readings, having held someone earlier the same day, is dropped to Wu's standby
band until the next comfort-mode start. Occupied zones keep the comfort band.

``occ_morning_enabled`` additionally holds zones that have not yet been occupied
today at a relaxed band. It is off by default because the first arrival of the
day pays full regret in that zone: the reward's mean is over present occupants,
so one person arriving in a cold zone costs as much as a full floor.

Never submit: it consumes a signal the competition does not provide.
"""

from __future__ import annotations

import numpy as np

from .ctrl_w import CtrlW
from .oracle_util import clock

try:  # packaged notebook
    from oracle_util import clock as _clock  # noqa: F401
except ImportError:  # workspace package
    _clock = clock


class OracleOccCtrlW(CtrlW):
    """Per-zone release on a 5-minute-late zonal occupancy channel."""

    name = "oracle_occ_ctrl_w"

    PARAMS = {
        **CtrlW.PARAMS,
        "occ_release_enabled": True,
        # Same hour as CtrlW's building-wide release: after it, this occupancy
        # model's arrival window (06-12) is closed, so an empty zone stays empty.
        "occ_release_hour": 12.0,
        "occ_release_empty_steps": 3,
        # A zone counts as occupied above this average over the interval; the
        # model assigns 2 occupants per zone, so anything non-zero is someone.
        "occ_occupied_threshold": 0.0,
        # Override Wu's 282 K night floor for released zones (None keeps it).
        "occ_release_low": None,
        "occ_morning_enabled": False,
        "occ_morning_relax_K": 1.0,
    }

    def __init__(self, building_meta_data, action_spec=None):
        params = dict(self.PARAMS)
        self.occ_release_enabled = bool(params["occ_release_enabled"])
        self.occ_release_hour = float(params["occ_release_hour"])
        self.occ_release_empty_steps = max(1, int(params["occ_release_empty_steps"]))
        self.occ_occupied_threshold = float(params["occ_occupied_threshold"])
        release_low = params["occ_release_low"]
        self.occ_release_low = None if release_low is None else float(release_low)
        self.occ_morning_enabled = bool(params["occ_morning_enabled"])
        self.occ_morning_relax_K = max(0.0, float(params["occ_morning_relax_K"]))
        super().__init__(building_meta_data, action_spec)

    # ------------------------------------------------------------------ state
    def reset(self):
        super().reset()
        self._occ_seen = np.zeros(self.n_zones, dtype=bool)
        self._occ_empty = np.zeros(self.n_zones, dtype=int)
        self._occ_last = np.zeros(self.n_zones)
        self._occ_hour = None
        self._occ_weekday = None
        self._occ_stamp = None

    # ----------------------------------------------------------------- oracle
    def observe_oracle(self, info):
        """Consumes ``zone_occupancy`` (last completed interval, zone-ordered)."""
        occupancy = info.get("zone_occupancy") if info else None
        if occupancy is None:
            return
        occupancy = np.asarray(occupancy, dtype=float)
        if occupancy.size != self.n_zones:
            raise ValueError(
                f"oracle occupancy has {occupancy.size} zones; policy has {self.n_zones}"
            )
        weekday, hour = clock(info.get("timestamp"))
        # Day rollover: a new day re-arms every zone (nobody has arrived yet).
        if self._occ_hour is not None and hour < self._occ_hour - 12.0:
            self._occ_seen[:] = False
            self._occ_empty[:] = 0
        self._occ_hour = hour
        self._occ_weekday = weekday
        self._occ_stamp = info.get("timestamp")
        self._occ_last = occupancy
        occupied = occupancy > self.occ_occupied_threshold
        self._occ_seen |= occupied
        self._occ_empty = np.where(occupied, 0, self._occ_empty + 1)

    # ------------------------------------------------------------- mechanisms
    def _released_zones(self):
        """Mask of zones released to standby, or ``None`` when none qualify."""
        if not self.occ_release_enabled or self._occ_hour is None:
            return None
        if self._occ_hour < self.occ_release_hour:
            return None  # arrivals are still possible; keep the comfort band
        if self._occ_weekday is not None and self._occ_weekday >= 5:
            return None
        mask = self._occ_seen & (self._occ_empty >= self.occ_release_empty_steps)
        return mask if mask.any() else None

    def _standby_band(self, temperature):
        low, high = super()._standby_band(temperature)
        floor = self.occ_release_low
        if floor is None:
            return low, high
        low = np.maximum(low, float(floor))
        recovery = max(1.0, self.parameters["recovery_steps"])
        low = np.minimum(low, temperature + np.maximum((low - temperature) / recovery, 0.0))
        return low, high

    def _band(self, obs, temperature):
        low, high, now = super()._band(obs, temperature)
        if not now:
            return low, high, now
        mask = self._released_zones()
        if mask is not None:
            released_low, released_high = self._standby_band(temperature)
            low = np.where(mask, released_low, low)
            high = np.where(mask, released_high, high)
        if self.occ_morning_enabled and self.occ_morning_relax_K > 0.0:
            # Zones nobody has entered yet today: relax the low edge a little.
            pending = (~self._occ_seen)
            lift = np.where(pending, self.occ_morning_relax_K, 0.0)
            low = low - lift
        return low, high, now
