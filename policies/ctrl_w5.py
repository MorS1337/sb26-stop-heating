"""CtrlW5 -- CtrlW4 with a released night standby floor and a cooler hot-water loop.

Two defaults change; every other knob is CtrlW4's.

1. `standby_minimum` 282.0 -> 262.0 K.  Wu's 282 K standby floor is inherited
   from his notebook and is what keeps the *coldest zone of each AHU group* at
   8.9 C all night, which holds that group's plant running for a building that
   is empty (the challenge's occupancy model puts arrivals at 06:00-12:00 and
   departures at 13:00-19:00).  Releasing the floor lets the group sit off: on
   the 7-day anchor that is 5.4 of the 12.8 reward units recovered, and it also
   hands the morning ramp a colder mass, which the plant heats *more efficiently*
   -- `gas/Q_net = 1 + 0.3(T_zone - T_out)/(T_dis - T_zone)` falls from ~2.0 at
   294 K to ~1.2 at 270 K because the mandatory 30 % outside-air share becomes a
   smaller fraction of the coil lift at the 308.15 K discharge cap.  The building
   mean stays at ~287 K overnight (14 C, against the schedule's own 289 K night
   setpoint); it is the *coldest* zones and the group plant that change.
   `docs/control_gap.md` §4.1 sweeps the floor: the gain saturates at 262 K
   because below that the floor never binds and the building free-floats.

2. `hot_water_target` 345 -> 336.32 K.  The tank's standing loss is 1.2 of the
   anchor's 93 reward units and the coil is limited by the 308.15 K discharge
   cap, not by the water side, so a cooler loop gives up little reheat capacity
   and pays 0.75 units less.  Measured alone: `results/cg_w4_hw336`.

Measured (shipped simulator, `--cenv fused`, building_train_1/fairbanks_1,
2016 steps): **-80.306** against CtrlW -93.123 and CtrlW4 -93.127, i.e. +13.7 %
and +13.8 %.  Comfort -1.12 against CtrlW's -0.54: the release costs a little
arrival-hour comfort and buys the energy back 13 times over.

Submission rules: numpy only, no `smart_buildings` import, no building or
weather id, no future information; CtrlW5 is CtrlW4 with two numbers changed, so
every constraint that holds for the packaged notebook entry holds here.
`CtrlW5.PARAMS | {"standby_minimum": 282.0, "hot_water_target": 345.0}`
reproduces CtrlW4 bit-exactly.

The wider CMA-ES optimum of the same family (16 knobs, 150 evaluations,
`docs/control_gap.md` §4) reaches -80.298 but spends comfort to do it
(-1.35); this two-knob point matches it with a third of the comfort cost.  The
dev30 subsets are in `docs/control_gap.md` §6.
"""

from __future__ import annotations

try:  # packaged notebook: the embedded modules are top-level names
    from ctrl_w4 import CtrlW4
except ImportError:  # workspace: policies/ is a package
    from .ctrl_w4 import CtrlW4


class CtrlW5(CtrlW4):
    """CtrlW4 with the night standby floor released and a cooler hot-water loop."""

    name = "ctrl_w5"

    PARAMS = {
        **CtrlW4.PARAMS,
        "standby_minimum": 262.0,
        "hot_water_target": 336.32,
    }
