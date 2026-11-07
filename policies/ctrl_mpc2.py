"""CtrlMPC2 -- the twin planner with the three defects of version 1 fixed.

`CtrlMPC` (version 1, `results/optg_mpc2_anchor` = -100.20 against CtrlW's
-93.12) is decomposed in `docs/control_gap.md`: it loses 7.1 reward units,
and the mechanism is not the plan's horizon, it is the *price of heat inside the
plan*.  Three things are wrong with the version-1 LP, all of them now switches
on `CtrlMPC` (defaults reproduce version 1 bit-exactly):

1. `heat_price_mode='phys'` -- version 1 prices one watt of zone heat at one
   watt of gas (`cost[QI] = cg`).  The plant cannot do that: the coil must lift
   the outside-air share of the supply stream as well, so at the 308.15 K
   discharge cap one watt into the zone costs

      1 + (T_zone - T_mix) / (T_dis - T_zone)   ~ 2.0 W of gas

   (the measured episode ratio is 1.975, `scratch/ControlGap/eff.py`).  Version 1
   therefore thinks night pre-heating is half price and buys capacity margin all
   night: it holds ~291.5 K from 19:00, spending 2.4 MWh of the week's 27 MWh
   that way, and earns back only ~0.3 MWh in the ramp.
2. `tcold_mode='zone'` -- version 1's coil capacity per m3/s of flow is
   `RHO_CP*(T_dis - T_sup)`, but the *zone* only receives
   `RHO_CP*(T_dis - T_zone)`: the same error makes the plan believe the plant
   can deliver twice the heat for a given airflow, so it under-plans airflow and
   the mapped episode runs at 19 C discharge at night (measured, 0.82x the
   at-cap flow).  With the physical cold side the plan asks for the airflow the
   discharge cap actually implies.
3. `plan_margin` / `night_cap` -- the plan sits exactly on the comfort floor and
   the mapping hands it to CtrlW as a hard floor, so the occupied-hour tracking
   error triples the comfort term (-0.54 -> -2.85).  A margin above the planned
   floor and a ceiling on the planned night temperature remove that.

`lp_steps`/`lp_dt` are unchanged at 8 h / 15 min: a 24 h horizon with
persistence weather was measured separately (`docs/control_gap.md`).

Offline experiment only: imports scipy/HiGHS and loads a precomputed 79-node
model of one building.  Never a submission candidate.
"""

from __future__ import annotations

try:  # packaged notebook: embedded modules are top-level names
    from ctrl_mpc import CtrlMPC
except ImportError:  # workspace: policies/ is a package
    from .ctrl_mpc import CtrlMPC


class CtrlMPC2(CtrlMPC):
    """CtrlMPC with a physically priced reheat coil and a tracking margin."""

    name = "ctrl_mpc2"

    PARAMS = {
        **CtrlMPC.PARAMS,
        "tcold_mode": "zone",
        "heat_price_mode": "phys",
        "plan_margin": 0.15,
        "night_floor": 282.0,
        "night_cap": 284.0,
    }
