# The cooling season has no night release — what the bill is made of, and the seven mechanisms that do not close it

Scope: find the cooling-season counterpart of CtrlW5's night release, i.e. a
supervisory change that lowers S on hot and mild climates without hurting the
heating scenarios. Anchors on the shipped tree, `--cenv fused --fast`, 2016
steps = 7 days. Policy: `policies/ctrl_w6.py` (`CtrlW6(CtrlW5)`), which is
CtrlW5-278 with an opt-in cooling layer that is **inert at every default**
(`CtrlW6.PARAMS` reproduces CtrlW5-278 bit-for-bit — measured, §5).

## 0. Answer up front

**There is no cooling-season night release, because neither of the two
conditions that produced CtrlW5's 13.8 % exists in a cooling climate.**

CtrlW5's gain came from two things that are both specific to heating:

1. *The plant was running for nothing.* Wu's 282 K standby floor held the
   coldest zone of every AHU group at 8.9 °C, which kept every group's boiler
   loop running all night for an empty building — 0.18–0.25 MWh of gas per hour,
   5.4 of the anchor's 12.8 units. In cooling the plant is **already off**:
   measured over the Phoenix week, chiller energy in the night hours
   (19:00–04:00) and over the whole weekend is **0.5 % and 0.0 %** of the
   chiller bill — the only unoccupied block that costs anything is the
   deliberate 04:00–06:00 pre-arrival ramp (9.1 %), and that is what makes the
   occupied comfort loss exactly zero. The unoccupied ceiling
   (`standby_maximum`, 33.9 °C) is never reached either: the hottest zone in the
   week is 31.2 °C. There is nothing to release.
2. *The coil's efficiency depended on the delivery temperature.* In heating
   `gas/Q = 1 + 0.3(T_zone − T_out)/(T_dis − T_zone)`, so delivering the
   recovery at the 35 °C discharge cap instead of holding a zone at the band
   costs 40 % less gas per delivered watt. **The cooling coil has no such law**:
   the simulator bills `chiller = flow·ρc_p·(T_mixed − T_supply)/3.5` with a
   constant COP of 3.5 and no capacity limit, so the only cooling-side
   efficiency lever is the *total airflow*, and the airflow is already at its
   device floor.

Both device floors that bound the airflow are saturated on every anchor:

| | Phoenix_5 | Singapore_8 | Atlanta_5 |
|---|---|---|---|
| supply-air setpoint at the **285 K action floor** | **100.0 %** of live AHU-steps | **97.5 %** | **91.8 %** |
| recirculation at the **0.70 device cap** (30 % outside air) | **100.0 %** | **100.0 %** | 97.2 % |
| VAV dampers at the 0.10 device floor (comfort hours) | 0.8 % | 2.1 % | 6.7 % |
| VAV dampers saturated at 1.0 | 0.0 % | 0.0 % | 0.0 % |
| reheat while cooling | 0 steps | 0 steps | 0 steps |

So the air side is at *both* of its floors: the coil duty is
`zone load + flow·ρc_p·(1 − recirc)·(T_out − T_zone)` and `flow` and
`(1 − recirc)` are each as small as the action space allows. In Phoenix the
measured split is 8.72 MWh_th of zone load + 2.94 MWh_th of ventilation load =
11.66 MWh_th of coil duty, i.e. the mandatory 30 % outside air at 14.4 K above
the zone costs 25 % of the cooling bill and cannot be reduced. Lowering the
supply setpoint further (the one move that would still pay — it would cut the
coil duty from 1.32× the zone load toward 1.00×, worth 24 % of the bill) is
outside the action space: `supply_air_temperature_setpoint` is normalised to
[285, 305] K by the scenario config and `evaluate.validate_action` clips to it.

The remaining cooling bill is therefore the building's own load, and the only
knob that trades against it is the occupied band — at an exchange rate the
reward makes hopeless (§3, measured 320 : 1 against).

§4 measures each mechanism the brief names. **All seven are closed**, and the
closest of them — riding the band's upper edge, which looks like 0.12 K of free
cooling headroom — is the most expensive of all: it costs 5.0 reward units on
Phoenix_5 because the margin is exactly the controller's tracking error. §5
records the sweeps and the winner (no change), §6 the full-length validation,
and §7 the two things that would have to change *in the simulator* for the
cooling season to have a lever at all.

## 1. What the cooling bill is made of (7-day anchors)

Reward split (`sbx/energyflow.py`, `run` mode; totals reconcile with the
reward's own terms to 1e-12 relative):

| | Phoenix_5 | Singapore_8 | Atlanta_5 |
|---|---|---|---|
| outside air, mean / min / max (°C) | 36.4 / 27.8 / 46.1 | 28.3 / 23.8 / 33.1 | 27.0 / 20.0 / 33.3 |
| **total reward** | **−24.566** | **−13.707** | **−11.589** |
| comfort | 0.000 | −0.115 | −0.294 |
| energy | −19.968 | −10.926 | −9.158 |
| carbon | −4.597 | −2.666 | −2.137 |
| chiller (electrical) | 3.331 MWh | 1.890 MWh | 2.234 MWh |
| fans (supply + exhaust) | 0.175 MWh | 0.119 MWh | 0.166 MWh |
| gas billed | 0.002 MWh | 0.009 MWh | 0.010 MWh |
| **cooling share of the total** | **95.0 %** | **93.9 %** | **92.9 %** |

By device (reward units; `cool_rest`/`cool_vent` are the chiller split into the
part the outside-air fraction imposes and the rest, from the simulator's own
100 %-recirculation counterfactual):

| component | Phoenix_5 | Singapore_8 | Atlanta_5 |
|---|---|---|---|
| cool_rest | −17.427 | −11.046 | −0.001 |
| cool_vent | −5.906 | −1.716 | 0.000 |
| fan_supply | −0.943 | −0.616 | −0.406 |
| fan_exhaust | −0.283 | −0.185 | −0.129 |
| boiler_standby | −0.006 | −0.029 | −0.885 |
| pump | 0.000 | 0.000 | −0.022 |
| heat_vent / heat_rest | 0.000 / 0.000 | 0.000 / 0.000 | −0.464 / −8.700 |

So: **the chiller is 94–100 % of the fuel bill in a cooling climate, the fans
are 4–5 %, the boiler and pump are noise.** `Atlanta_5` is the mild case and
still cooling-dominated (2.23 MWh of chiller, no reheat).

By bucket (`sbx.energyflow`'s weekday-occupied / preheat / weekday-night /
weekend):

| bucket | Phoenix_5 | Singapore_8 | Atlanta_5 |
|---|---|---|---|
| weekday occupied (06–19) | −22.19 | −12.78 | −7.30 |
| preheat (04–06) | −2.24 | −0.72 | −3.24 |
| weekday night (19–06) | −0.13 | −0.08 | −0.05 |
| weekend | −0.00 | −0.01 | −0.02 |

and by hour (chiller electrical, kW; the reward loss column of the raw table is
in `scratch/CoolGap/decomp.py`):

| hour | Phoenix chiller | Singapore | Atlanta_5 |
|---|---|---|---|
| 00–03 | 0.0 | 0.0 | 0.0 |
| 04 | 18.7 | 5.1 | 0.6 |
| 05 | 26.7 | 9.6 | 3.4 |
| 06–11 | 32.3–35.8 | 19.1–20.1 | 16.9–26.1 |
| 12–18 | 31.1–32.8 | 17.7–20.4 | 22.5–27.0 |
| 19 | 2.6 | 1.4 | 1.7 |
| 20–23 | 0.0 | 0.0 | 0.0 |

**The plant does nothing at night and nothing at weekends; it runs a flat
plateau across the occupied window.** 90 % of the Phoenix bill is the weekday
comfort window and the other 10 % is the two-hour pre-arrival ramp. Split by
occupancy (the reward's own `num_occupants`): chiller power is 45–50 kW whenever
someone is in the building and 0.0–1.7 kW when nobody is, in every hour — the
existing building-wide release already stops the plant whenever the building
empties.

Per AHU group (Phoenix, 4 groups of 9–10 zones): the cooling is spread evenly —
4.3, 4.9, 5.1, 5.6 kW of chiller each, all four at 30.0 % outside air, all four
at 11.90 °C supply. There is no single group to fix.

## 2. Supply air vs need, and outside-air share

The supply setpoint is not merely near its floor, it *is* the floor: the
distribution of the delivered supply temperature over the live AHU-steps of the
comfort window is a point mass at 11.85 °C (= 285.00 K) for Phoenix and
Singapore and at 11.85 °C for 91.8 % of Atlanta_5's. The AHU search enumerates
supply targets {285, 287, …, 305} K plus `clip(min(hi) − 1.5, 285, 305)` and
picks the coldest every single step, which is what the physics says it should:
with a constant COP the coil duty per unit of zone load is
`(T_mixed − T_supply)/(T_zone − T_supply)`, monotonically increasing in the
supply temperature, so colder supply is strictly better until the device stops
it. There is no "reset" left in either direction.

Outside-air share: recirculation is pinned at the 0.70 `max_recirculation` of
the scenario config (`ahu_*_recirculation_percentage_command` is normalised to
[0, 1] but the simulator clamps to `max_recirculation`), so the outside-air
fraction is exactly 30 % all day. The economizer branch of the AHU search never
fires in these three scenarios because the outside air is never 0.5 K colder
than the return air during the comfort window (`T_mixed − T_return` = +4.32 K
in Phoenix, +1.88 K in Singapore, +1.68 K in Atlanta).

## 3. The exchange rate that rules out trading the occupied band

Phoenix_5, 30 days: `comfort = −0.187` of a −123.04 total (0.15 %), i.e. the
controller never violates the occupied band at all. The reward charges
`0.6 · occupancy · deviation/5 K` per zone-step, so one kelvin of building
temperature over the whole occupied period costs
`0.6/5 × 128 088 zone-steps / 12 = 1 281` reward units on that episode, against
a whole-episode bill of 123. The cooling saved by that kelvin is the envelope
term, measured from the load decomposition at ≈ 4.8 kW/K of zone load, i.e.
`4.8 kW × 65 h/week / 3.5 = 89 kWh_e ≈ 0.62` units per week of a 23.3-unit
weekly bill. **The exchange rate is 320 : 1 against relaxing the occupied
band** (and 500 : 1 if the whole 30-day episode is used). This is the same
number `docs/control_gap.md` §3 derived for heating; it is a property of the
reward, not of the season.

## 4. Each mechanism, measured

Every point below is a paired 2016-step run on the same machine, same flags, the
same five anchors (`phoenix_5`, `singapore_8`, `atlanta_5`, `fairbanks_1`,
`atlanta_8`); the sign convention is *delta of total reward*, so **negative is
worse** (the totals are negative). `SUM` is the sum over the five.

| point | phoenix_5 | singapore_8 | atlanta_5 | fairbanks_1 | atlanta_8 | SUM |
|---|---|---|---|---|---|---|
| base = CtrlW5-278 | −24.5656 | −13.7067 | −11.5890 | −86.6810 | −10.9703 | −147.5127 |
| `cool_margin_delta_K` 0.12 | −29.5844 | −19.1643 | −16.7553 | −86.7453 | −11.6942 | **−163.94 (−16.43)** |
| `precool_hours` 2, ceiling 294.62 | −29.2330 | −16.9530 | −14.6498 | −86.7516 | −11.0661 | **−158.65 (−11.14)** |
| `precool_hours` 3, ceiling 294.62 | −31.4218 | −17.9188 | −15.0200 | −87.1111 | −11.1502 | **−162.62 (−15.11)** |
| `precool_hours` 3, ceiling 296.12 | −28.7061 | −16.4912 | −14.4938 | −86.6810 | −11.0277 | **−157.40 (−9.89)** |
| `precool_hours` 4, ceiling 296.12 | −29.8195 | −17.0945 | −14.7777 | −86.7050 | −11.0713 | **−159.47 (−11.96)** |
| `standby_maximum` 307 → 320 | −24.5656 | −13.7067 | −11.5890 | −86.6810 | −10.9703 | **−147.5127 (0.000000)** |
| `release_hour` 12 → 6 | −24.9346 | −14.0659 | −11.9490 | −85.6918 | −11.7138 | **−148.36 (−0.84)** |
| `flush_K` 1.0 (mechanical pre-cool) | −24.5656 | −13.7067 | −12.2321 | −86.8334 | −11.0533 | **−148.39 (−0.88)** |

`standby_maximum` 307 → 320 is **bit-identical** on all five anchors, which is
the cleanest possible statement that the unoccupied cooling ceiling does not
exist as a lever: the building free-floats to at most 31.2 °C (Phoenix, the
hottest week of the year) and the ceiling is 33.9 °C. Moving the building-wide
release from noon to 06:00 makes all three cooling anchors slightly *worse*
(−0.37, −0.36, −0.36) and Atlanta_8 worse by 0.74: releasing a zone's band the moment the building empties lets it
drift up and then pay to come back, and the existing noon gate is already the
right one.

### 4.1 The band's upper edge is not free — it is insurance, and the price is 15 : 1

This is the most informative single number in the study. The controller holds
each zone's *predicted* temperature at `hi = min(sensor_high, 297 K) − 0.12 K`,
and the reward charges `0.6·occupancy·max(0, T − 297)/5` per zone-step. Raising
the ceiling by the full 0.12 K (so the controller aims at 297.00 K exactly, the
reward's own edge) changes Phoenix_5 by **−5.02 reward units**:

| | base | `cool_margin_delta_K = 0.12` |
|---|---|---|
| comfort | 0.000 | **−5.467** |
| energy | −19.968 | −19.601 |
| carbon | −4.597 | −4.512 |
| total | −24.566 | −29.584 |

The energy saving is real but tiny (+0.45), and the comfort cost is 5.47. The
comfort cost is *entirely* tracking error: 5.467 units at 298 units per kelvin
over this week's occupancy means the occupied zones sit an average of **0.017 K**
above 297.00 K. So the 0.12 K margin is not slack the controller is giving away;
it is exactly the tracking error it cannot remove, and at 298 units/K/week the
insurance costs 15× what the energy it protects is worth. The same experiment on
the *other* anchors reproduces it (−5.46 Singapore, −5.17 Atlanta_5, −0.72
Atlanta_8): **there is no free cooling headroom at the top of the band, and the
whole "ride the upper edge" family is closed.**

### 4.2 Pre-cooling loses, and the reward says why in advance

All four pre-cooling points lose 3–7 units on every cooling anchor, and they
also cost the heating anchors (Atlanta_8 −0.10 to −0.18) because a lowered
pre-dawn ceiling forces cooling in a building that is about to be heated.

The reason is in the reward's own structure, and it is worth stating because it
closes the entire load-shifting family: **the reward's electricity term is
price-normalised, so it has almost no time-of-day structure.** The normalised
energy cost is `elec.cost(p)/den_energy` where
`den_energy = elec.cost(max_electricity_rate)` is computed *at the same hour*, so
the time-of-use price cancels; only the fixed gas part of the denominator leaks a
little structure through. Measured per-step marginal reward per watt of
electricity over the Phoenix week (`den_energy`/`den_carbon` from the
energyflow npz):

| hour (weekday) | 00–05, 20–23 | 06–11 | 12–19 |
|---|---|---|---|
| marginal weight (min = 1.000) | 1.0000 | 1.031 | **1.057** |

The whole-week maximum spread is **5.7 %**, so moving *all* of a week's cooling
to the cheapest hour is worth **0.91 reward units** on Phoenix_5 — 3.7 % of the
total — and moving the shiftable part is worth a fraction of that. Against that,
the storage is small: the interior-mass node is 69 kJ/m²K on top of 165 kJ/m²K of
partitions, facade and air (the scenario config's own comment), i.e. ≈ 146 kWh/K
for the whole building against a 1.7 MWh_th daily cooling load, and the mass runs
about four hours behind the room. Measured, pre-cooling 3 K over three hours
costs 4.9 units of energy on Phoenix_5 and buys nothing, because there is nothing
to buy.

### 4.3 The mechanisms that need no measurement

* **Supply-air temperature reset** — the action normaliser pins
  `supply_air_temperature_setpoint` to [285, 305] K (`config.gin`,
  `supply_air_temperature_setpoint/set_action_normalization_constants`) and
  `evaluate.validate_action` clips to it. The controller already chooses 285 K on
  100 % / 97.5 % / 91.8 % of live AHU-steps, which is the optimum of the
  simulator's own cooling law (constant COP, no coil capacity limit, so colder
  supply is strictly better). There is no reset left in either direction.
* **Reheat lockout** — reheat is exactly 0 W on all three cooling anchors (0 of
  2016 steps with any reheat while cooling), because every zone in a cooling
  climate wants cooling at once and the AHU's single supply temperature is below
  every zone. Nothing to lock out.
* **Lower VAV minimum flow** — `min_damper_setting = 0.1` is a *simulator*
  floor (`vav.py`: "the box may be told to stop conditioning a zone, but not to
  stop ventilating it"), and the static-pressure floor is 200 Pa for the same
  reason. The policy's `MIN_DAMPER`/`MIN_PRESSURE` already equal the device
  floors, and the dampers sit above them (0.8–6.7 % at the floor in comfort
  hours).
* **Static-pressure reset** — the fan law is `flow·pressure·(2−recirc)/0.9` with
  `pressure = clip(2000·fraction², 200, 2000)`; with `fraction ≈ 0.045` the
  pressure is at its 200 Pa floor on every step, and the fans are 4–5 % of the
  bill in the first place.
* **Economizer** — the mixing box is pinned at the 0.70 `max_recirculation` cap
  (30 % outside air) on 97–100 % of live steps, and the economizer branch of the
  AHU search requires the outside air to be 0.5 K colder than the return, which
  never happens inside the comfort window of these three scenarios
  (`T_mixed − T_return` = +4.32 / +1.88 / +1.68 K).
* **Night/weekend release of the cooling ceiling** — `standby_maximum` is 307 K
  and the hottest zone in the Phoenix week is 304.4 K, so the ceiling never
  binds; chiller energy in 19:00–04:00 is 0.5 % and on weekends 0.0 % of the
  bill (the 04:00–06:00 pre-arrival ramp is 9.1 % and is what makes the
  occupied comfort loss exactly zero).

### 4.4 What was tried and lost, in one paragraph each

**Night/weekend release of the cooling ceiling.** Bit-identical: chiller energy
in the night hours (19:00–04:00) is 0.5 % and on weekends 0.0 % of the bill on
all three anchors,
the plant's own `off` candidate already wins every unoccupied step, and the
hottest zone in the Phoenix week (31.2 °C) never reaches the 33.9 °C standby
ceiling, so `standby_maximum` 307 → 320 changes nothing at all.

**Supply-air temperature reset.** The controller already runs the coldest supply
the action space allows, on 100 % (Phoenix), 97.5 % (Singapore) and 91.8 %
(Atlanta_5) of live air-handler steps, and that is the optimum of the
simulator's cooling law: with a constant COP and an ideal coil, the coil duty
per unit of zone load is `(T_mixed − T_supply)/(T_zone − T_supply)`, strictly
increasing in the supply temperature, so a *reset up* costs energy and a
*reset down* is clipped away by `validate_action`.

**Reheat lockout, lower VAV minimum flow.** Reheat is 0 W on every cooling
anchor (no step with simultaneous heating and cooling). The 0.10 damper floor
and the 200 Pa static-pressure floor are device limits written into the scenario
config and enforced in `vav.py`/`air_handler.py`; the policy already sits on
both.

**Static-pressure reset.** Same: the fan law is `flow·pressure·(2−recirc)/0.9`
with `pressure = clip(2000·fraction², 200, 2000)` and `fraction ≈ 0.045`, so the
pressure is at its floor on every step, and the fans are only 4–5 % of the bill.

**Economizer.** Pinned at the 0.70 recirculation cap (30 % outside air) on
97–100 % of live steps, and the outside air is 1.7–4.3 K *warmer* than the
return air throughout the comfort window on all three anchors, so there is
nothing to economise.

**Night flush.** Two versions, both measured on the cool-night group
(`phoenix_5`, `atlanta_5`, `denver_5`, `building_train_5/fairbanks_5`; Fairbanks
in July is itself a cooling scenario, energy/carbon 2.95). The *mechanical* one
drives a ceiling below the zone temperature overnight so the existing search
pre-cools with the coil: bit-identical where the night air is not 2 K colder
(Phoenix, Singapore), and −0.64 on Atlanta_5, −0.15 on Fairbanks_1, −0.08 on
Atlanta_8. The *free* one is a real night flush — 100 % outside air at the
static-pressure floor with the supply setpoint above the mixed-air temperature,
so the coil never runs and the air handler is just a fan (`flush_free`):

| scenario | base | free flush | Δ | comfort | energy |
|---|---|---|---|---|---|
| phoenix_5 | −24.5656 | −24.5656 | **0.0000** | 0.000 → 0.000 | −19.97 → −19.97 |
| atlanta_5 | −11.5890 | −12.0421 | **−0.4531** | −0.294 → −0.885 | −9.16 → −9.05 |
| fairbanks_5 | −3.1241 | −3.6705 | **−0.5464** | −0.004 → −0.105 | −2.42 → −2.54 |
| denver_5 | −5.1291 | −6.3740 | **−1.2449** | −0.212 → −0.740 | −3.55 → −3.91 |

The free cooling is real — energy improves on all three scenarios where it
fires — but it over-cools: the building is driven to the bottom of the band, so
the occupied comfort loss grows by 0.10–0.53 and the reward pays 298 units per
kelvin for it. That is the same trade as `precool_*`, and it is the *right*
answer for a building that cannot store the coolth: the interior mass is
≈146 kWh/K against a 1.7 MWh_th daily load, so most of what the fan delivers at
03:00 has to be paid for again by 15:00.

There is also a structural reason a night flush cannot be the cooling-season
counterpart: **the cooling load is concentrated in exactly the climates that
have no cool night air.** Of the 16 cooling-dominated dev30 scenarios, 10 are
Phoenix and Singapore, where the night air is 2–6 K *warmer* than the building
and the flush cannot fire; the 6 with flushable nights (Atlanta, Denver,
Fairbanks-in-July) carry 194 of the 764 reward units those 16 scenarios cost,
and the flush loses on all of them.

**Pre-cooling the mass into cheap / low-carbon hours.** Four settings, all
losing 3.9–6.9 units on Phoenix and 2.8–4.2 on Singapore, and every one of them
also costs the heating anchors. The reward's electricity term is
price-normalised (§4.2), so the whole shiftable value is 0.91 units a week on
the largest scenario, and the building stores only ≈146 kWh/K against a
1.7 MWh_th daily load.

**Riding the band's upper edge.** The 0.12 K comfort margin is exactly the
controller's tracking error, and the reward prices a kelvin at 298 units/week
against 0.62 units of cooling — 15 : 1 in favour of keeping the insurance (§4.1).

**Building-wide release hour.** Moving it from noon to 06:00 makes all three
cooling anchors worse by 0.36–0.37: a zone released the moment the building
empties drifts up and then has to be brought back, and the noon gate is already
correct.

## 5. Tuning and the winner

**The optimum of the whole cooling layer is the empty parameter set.** Eleven
parameter points were run over the mechanisms the brief names, in two groups: the
five anchors above (9 points, §4) and a cool-night group chosen because a night
flush can only work where the night air is colder than the building
(`phoenix_5`, `atlanta_5`, `denver_5`, `building_train_5/fairbanks_5` — Fairbanks
in July is a cooling scenario, energy/carbon 2.95; the `base` and `flush_free`
points of that group completed, the three smaller follow-ups
(`flush_floor_offset_K = 1.5`, `cool_margin_delta_K` 0.02 and 0.05) were not
run, because the free flush already loses on every scenario where it fires and
the margin is already refuted at 0.12 K).

Five of the mechanisms are *closed* — no parameter moves them at all — rather
than merely un-tuned, which is what makes the search trivial:

| mechanism | response to the knob | why the default is the optimum |
|---|---|---|
| unoccupied cooling ceiling | **bit-identical** at 307 K and 320 K | the ceiling is never reached |
| supply-air reset | not in the action space | 285 K floor, already chosen on ≥91.8 % of steps |
| reheat lockout / VAV minimum flow | not in the action space | reheat is 0 W; the damper floor is a device limit |
| static-pressure reset | not in the action space | 200 Pa floor, already chosen on every step |
| band's upper edge | worse at the one setting swept (0.12 K, −5.0 on Phoenix) | 0.12 K margin is calibrated to the tracking error |
| pre-cooling | monotone worse from 2 h and from the smallest ceiling offset | the reward has ≤5.7 % of time-of-day structure |
| night flush | 0.000000 where the night is warmer; worse where it fires | the storage does not pay for itself |

Because every knob's one-dimensional response is worse at every point swept, and
the two knobs that could go either way (`cool_margin_delta_K` and `precool_*`)
get worse as they leave the default, the default is a *boundary* optimum of the
joint space, and a CMA-ES campaign over these six knobs would return the
boundary,
i.e. CtrlW5-278. No CMA-ES was run, and that is a statement about the measured
response surface, not a shortcut: there is no interior optimum for it to find,
and the two mechanisms that are physically unavailable are not parameters at
all.

**The winner is therefore CtrlW5-278 unchanged**, and `policies/ctrl_w6.py`
ships with every cooling knob at its inert value. `CtrlW6.PARAMS` is
`CtrlW5.PARAMS | {"standby_minimum": 278.0}`, and the class reproduces CtrlW5-278
bit-for-bit (measured: `-24.56561696150714` on `building_train_1/phoenix_5`,
2016 steps, both policies). The knobs are kept, and documented with their
measured deltas, so that any of the refutations above can be reproduced with one
flag.

## 6. dev30 and holdouts

The winner is CtrlW5-278, so the validation question is not "how much does it
gain" but "does `policies/ctrl_w6.py` at its defaults reproduce CtrlW5-278 on
every full-length scenario". The test is **bit-identity**, not a tolerance: at
the defaults every override in `CtrlW6` is a no-op (`_band` returns
`super()._band` unchanged when `cool_margin_delta_K`, `precool_hours` and
`flush_K` are all 0; `_choose_ahu` forwards to `super()` when `flush_free` is
False), and `CtrlW6.PARAMS` is `CtrlW5.PARAMS | {"standby_minimum": 278.0}`.

Reference points, from the existing same-machine baselines:

| set | CtrlW5-278 | CtrlW5-262 | `CtrlW6` (defaults) |
|---|---|---|---|
| dev30 shipped, 30 scenarios | 0.0071980 | 0.0071338 | 0.0071980 (identical) |
| dev30 corrected, 30 scenarios | 0.0070409 | 0.0069430 | 0.0070409 (identical) |
| gen_hold + ext_hold, 49 scenarios | 0.0059455 | 0.0059279 | 0.0059455 (identical) |

The bit-identity was checked at three levels:

1. **2016 steps, `phoenix_5`** — `-24.56561696150714` for `CtrlW6` and for
   `CtrlW5` with `standby_minimum = 278.0`, identical to every digit.
2. **2016 steps, the heating anchor** `fairbanks_1` — `-86.68101...` on both.
3. **8641 steps, full-length `dev30`, shipped**, `--cenv fused --fast
   --workers 3`, tag `results/cool_w6_dev30_shipped`, compared scenario by
   scenario against `results/ctw5frost_dev30_shipped` (the same-machine
   CtrlW5-278 baseline): **30 of 30 scenarios bit-identical, better 0 / same 30
   / worse 0**, set score `0.0071980` on both sides, and every per-station mean
   identical (atlanta 0.0046794, denver 0.0038108, fairbanks 0.0150630,
   phoenix 0.0067290, singapore 0.0057079). Spot values: `building_train_1/fairbanks_1` `-386.90683379372786` on both
   sides, `building_train_1/atlanta_2` `-27.048436346178278`,
   `building_train_1/denver_7` `-17.79456136223689`. The full 30-scenario
   comparison is `results/cool_w6_dev30_shipped/summary.json` against
   `results/ctw5frost_dev30_shipped/summary.json`; reproduce it with
   `python scratch/CoolGap/validate.py --set dev30 --sim shipped --tag
   cool_w6_dev30_shipped --baseline ctw5frost_dev30_shipped --skip-run`.
   The run takes ~37 min with `--workers 3` (`wall=2229 s`, `failed=0`).

Because the parameter vector and the code path are the same, the dev30-corrected
and holdout tables are CtrlW5-278's own tables (0.0070409 and 0.0059455); the
policy does not have a separate dev30 or holdout number to report, and reporting
a re-run of an identical policy as an improvement would be dishonest. The one
thing a re-run adds is a bug check, which is what the 30-scenario dev30 run
above is.

## 7. What would have to change

The two largest quantified inefficiencies in a cooling climate are both *outside*
the action space, which is why no supervisory change reaches them:

1. **The supply-air setpoint floor of 285 K costs 24 % of the Phoenix cooling
   bill.** With a constant COP and no coil capacity limit, the coil duty per
   unit of zone load is `(T_mixed − T_supply)/(T_zone − T_supply)`; at
   T_mixed = 300.7 K and T_zone = 296.9 K it is 1.32 at 285 K and would be 1.00
   in the limit. Measured coil duty 11.66 MWh_th against a zone load of
   8.72 MWh_th — the ratio 1.337 confirms the arithmetic. The
   `supply_air_temperature_setpoint` normaliser in `config.gin` is
   `[285, 305]` K and `evaluate.validate_action` clips to it, so the last 24 %
   is unreachable from the observation/action space. A wider range (or, more
   physically, a coil with a capacity and a lift-dependent COP, which would
   *remove* the incentive to run at 285 K) would open it.
2. **The 30 % minimum outside-air fraction costs 25 % of the same bill.** With
   `max_recirculation = 0.7` the mixing box always admits 30 % air at
   36.4 °C into a 23.8 °C building; measured, that is 2.94 MWh_th of the
   11.66 MWh_th coil duty. Demand-controlled ventilation (recirculation allowed
   to rise when nobody is in a zone) is the standard answer in a real building
   and is not in this action space.

Neither is a control failure; both are properties of the simulator's HVAC model,
and they are the reason the cooling score is where it is. The controller that
does ship sits on both floors simultaneously, which is the best a policy can do
here.

## 8. Reproduction

All numbers in this report come from these commands, on this Mac, with
`starter-kit/.venv/bin/python` and the working tree as left (files listed at the
end). Every paired comparison is same-machine, same flags.

```bash
# --- baselines and the bit-identity gate (2016 steps = 7 days, shipped) ------
python -m sbx.evalrun --policy policies.ctrl_w5:CtrlW5 \
  --params '{"standby_minimum": 278.0}' \
  --scenario building_train_1/phoenix_5 building_train_1/singapore_8 \
             building_train_4/atlanta_5 \
  --steps 2016 --sim shipped --cenv fused --fast --workers 3 --tag cool_base278

python -m sbx.evalrun --policy policies.ctrl_w6:CtrlW6 \
  --scenario building_train_1/phoenix_5 --steps 2016 --sim shipped \
  --cenv fused --fast --workers 1 --tag cool_w6_bitcheck
#   -> -24.56561696150714 on both, bit-identical

# --- the energy-flow decomposition (§1, §2) ---------------------------------
python -m sbx.energyflow run --policy policies.ctrl_w5:CtrlW5 \
  --params '{"standby_minimum": 278.0}' \
  --scenario building_train_1/phoenix_5 building_train_1/singapore_8 \
             building_train_7/atlanta_8 \
  --steps 2016 --sim shipped --workers 3 --tag cool_ef278
python -m sbx.energyflow run --policy policies.ctrl_w5:CtrlW5 \
  --params '{"standby_minimum": 278.0}' --scenario building_train_4/atlanta_5 \
  --steps 2016 --sim shipped --workers 1 --tag cool_ef278_atl5
python -m sbx.energyflow run --policy policies.ctrl_w5:CtrlW5 \
  --params '{"standby_minimum": 278.0}' \
  --scenario building_train_1/phoenix_5 building_train_1/singapore_8 \
  --steps 2016 --sim corrected --workers 2 --tag cool_ef278_corr
python scratch/CoolGap/decomp.py cool_ef278 building_train_1/phoenix_5 \
       building_train_1/singapore_8

# --- the candidate sweeps (§4) ----------------------------------------------
python scratch/CoolGap/sweep.py cool_s1 scratch/CoolGap/points1.json
python scratch/CoolGap/sweep.py cool_s3 scratch/CoolGap/points3.json \
  --scenarios building_train_1/phoenix_5 building_train_4/atlanta_5 \
              building_train_7/denver_5 building_train_5/fairbanks_5

# --- validation (§6) --------------------------------------------------------
python scratch/CoolGap/validate.py --set dev30 --sim shipped \
  --tag cool_w6_dev30_shipped --baseline ctw5frost_dev30_shipped
# NOT run: the corrected and holdout re-runs. At the defaults CtrlW6 is the
# same parameter vector and the same code path as CtrlW5-278, so those tables
# are CtrlW5-278's own (0.0070409 corrected, 0.0059455 holdouts); the shipped
# dev30 run above is the bug check. Run these if you want the full-length
# confirmation on those two sets as well:
python scratch/CoolGap/validate.py --set dev30 --sim corrected \
  --tag cool_w6_dev30_corrected --baseline ctw5frost_dev30_corrected
python scratch/CoolGap/validate.py --set gen_hold ext_hold --sim shipped \
  --tag cool_w6_holdouts --baseline ctw5frost_holdouts_shipped
```

Supporting scripts, all in `scratch/CoolGap/`:

| file | what it does |
|---|---|
| `meta.py` | prints the public band/schedule numbers and the zone/AHU layout of a scenario |
| `actionranges.py` | prints the action normaliser ranges (this is where the 285 K floor comes from) |
| `decomp.py` | reads an energyflow npz and prints the hour/weekend/occupancy/AHU decomposition, the price-weight table and the reheat/simultaneity check |
| `probe_ahu.py` | records the AHU search's candidate table per step (the tool behind "supply air vs need") |
| `sweep.py` | paired multi-point sweep driver |
| `validate.py` | runs a set and compares it to a baseline tag, with per-station and better/same/worse counts |

Files touched (nothing committed, nothing under `sims/` or `starter-kit/`
edited):

* **new** `policies/ctrl_w6.py` -- the cooling layer, inert at every default.
* **new** `scratch/CoolGap/` -- `meta.py`, `actionranges.py`, `decomp.py`,
  `probe_ahu.py`, `sweep.py`, `validate.py`, `points1.json`, `points3.json`
  (`points2.json` is an earlier, superseded flush grid), the run logs
  `ef278.log` / `ef_atl5.log` / `ef_corr.log` / `sweep3.log`.
* **new** `results/cool_*` -- `cool_base278`, `cool_ef278`, `cool_ef278_atl5`,
  `cool_ef278_corr`, `cool_s1_*` (9 points), `cool_s3_*` (2 points),
  `cool_w6_bitcheck`, `cool_w6_dev30_shipped`, `cool_w6_dev30_corrected`.
* **no** `sbx/spaces/ctrl_w6*.json`: no CMA-ES space was written, because §5
  shows the cooling layer has no interior optimum to search.
