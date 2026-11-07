"""CtrlW7 -- frost protection by bursts instead of a held floor. Measured: it loses.

Not shipped. This is the reproduction handle for a refutation reported in the paper's
frost section.

CtrlW5-278 holds a 278 K night floor. The coldest zone of an air-handler group sits on
it and takes a little heat every step, which costs 6.4 of the 7-day anchor's units
(-80.31 released, -86.68 at 278 K). The idea tested here: release the floor as CtrlW5
does (262 K) and give each zone a frost latch instead.
- The latch switches on when the zone falls below `frost_floor_K`.
- It switches off once the zone is back at `frost_floor_K + frost_band_K`.
- While it is on, the zone's lower band is raised to that upper threshold.
The recovery would then be one burst at the discharge cap, and the handler would stay
off in between.

Measured on `building_train_1/fairbanks_1`, shipped tree, 2016 steps, on the Mac,
with `scratch/ControlGap/frost.py` (records in `results/frostburst_anchor/`):

    frost_band_K   total     vs held floor   coldest room
    held (W5-278)  -86.681        --           +4.66 C
    0              -87.004      -0.32          +2.19 C
    1              -87.970      -1.29          +1.05 C
    2              -89.109      -2.43          +1.79 C
    3              -90.061      -3.38          +2.54 C
    4              -90.996      -4.32          +3.27 C

Every band is both dearer and colder than holding the floor. Anticipating the floor
(CtrlW5-278 heats on the predicted next temperature) beats recovering from it.
`frost_band_K = 0` latches only while a zone is below the floor: a reactive copy of the
held floor.

Numpy only, no building or weather identifiers. The latch reads the same lagged zone
temperatures the base controller already uses.
"""

from __future__ import annotations

import numpy as np

try:  # packaged notebook: the embedded modules are top-level names
    from ctrl_w5 import CtrlW5
except ImportError:  # workspace: policies/ is a package
    from .ctrl_w5 import CtrlW5


class CtrlW7(CtrlW5):
    """CtrlW5 with a per-zone frost latch (burst recovery) instead of a held floor."""

    name = "ctrl_w7"

    PARAMS = {
        **CtrlW5.PARAMS,
        "frost_floor_K": 278.0,
        "frost_band_K": 3.0,
    }

    def __init__(self, building_meta_data, action_spec=None):
        params = dict(self.PARAMS)
        self.frost_floor_K = float(params["frost_floor_K"])
        self.frost_band_K = max(0.0, float(params["frost_band_K"]))
        super().__init__(building_meta_data, action_spec)
        self._frost_latched = np.zeros(self.n_zones, dtype=bool)

    def _band(self, obs, temperature):
        low, high, now = super()._band(obs, temperature)
        temperature = np.asarray(temperature, dtype=float)
        target = self.frost_floor_K + self.frost_band_K
        # In comfort hours the band already sits far above the target, so the
        # latch can run in every mode without touching occupied control.
        self._frost_latched = (
            self._frost_latched | (temperature < self.frost_floor_K)
        ) & (temperature < target)
        low = np.where(self._frost_latched, np.maximum(low, target), low)
        return low, high, now
