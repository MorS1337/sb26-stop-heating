# What closes the 12 % anchor gap — a measured answer

Anchor: `building_train_1/fairbanks_1`, shipped simulator, default start
(2023-01-01 06:00), first 2016 steps = 7 days, mean outside −26.3 °C.
Every run uses `--cenv fused` (the bit-identical C step); runs marked "fast" also
use `--fast` (also bit-identical — the CtrlW total is
`−93.12290938248043` in all three configurations).

Scripts in `scratch/ControlGap/`: `eff.py` (per-VAV air-side reconstruction from
a capture), `tune_anchor.py` + `sbx/spaces/ctrl_w4_anchor.json` (CMA-ES),
`apply_best.py`, `dev30_check.py`, `planner_ablate.py`. Captures:
`ctrlw_anchor2016.npz`, `w5c_anchor2016.npz`, `mpc_anchor2016.npz`.
Policies: **`policies/ctrl_w5.py`** (the tuned controller, submission-safe) and
`policies/ctrl_mpc2.py` (the offline planner experiment).

## 0. Answer up front

**The gap closes, and neither information nor prediction closes it: it closes by
changing *when the plant is allowed to heat*, so that the recovery is delivered
in one burst at the plant's most efficient operating point instead of being
spread across the night and the day.** Measured on the anchor, 2016 steps:

| | anchor total | Δ vs CtrlW | Δ % |
|---|---|---|---|
| CtrlW | −93.123 | — | — |
| CtrlW4 | −93.127 | −0.004 | −0.00 % |
| CtrlMPC v1 (8 h LP on the 79-node twin) | −100.200 | −7.078 | −7.6 % |
| **CtrlW5 (CtrlW with the night floor released, two numbers changed)** | **−80.306** | **+12.817** | **+13.8 %** |
| public MOBI-EVERGi (for scale) | −81.9 | +11.2 | +12.1 % |
| public Houssem twin+LP (for scale) | −85.4 | +7.7 | +8.3 % |

Three mechanisms, all measured on the same episode (component split from
`decomp.py`, hour split from the runs' own `by_hour`, air side from `eff.py`):

1. **Night heating stops.** CtrlW holds the *coldest zone of each AHU group* at
   its 282 K standby floor from 19:00 to 04:00, and one cold zone holds the whole
   group's AHU on: 0.18–0.25 MWh of gas per hour, 5.4 reward units a week, for a
   building that is empty (the challenge's occupancy model puts arrivals at
   06:00–12:00). Released, the night coil duty is **30 kW → 0 kW** and the
   building's *mean* temperature barely moves (287.9 → 287.4 K, i.e. 14 °C,
   still 2 K from the schedule's own 289 K night setpoint): it is the coldest
   zones and the group plant that change, not the building.
2. **The recovery is one burst at the discharge cap.** Heat delivered into a
   colder building is cheaper per delivered watt by the plant's own law
   `gas/Q = 1 + 0.3(T_zone − T_out)/(T_dis − T_zone)` — 2.0 at 294 K, 1.2 at
   270 K — because the mandatory 30 % outside-air share is a smaller fraction of
   the coil lift when the air only has to be lifted to the zone, not held above
   it. CtrlW5's discharge sits at 35.00 °C (p10 34.39 °C, mean cap headroom
   **1.1 K** vs CtrlW's 2.3 K) and its gas per net zone watt is **2.061 against a
   cap-limited 2.045** — 0.7 % of air-side slack, against CtrlW's 1.2 %.
3. **The stored heat pays for the day.** Charging the mass at 04:00–06:00 (the
   cheapest hours: 88 kg CO₂/MWh, 16 c/kWh) and letting it drift through the band
   costs nothing in comfort — the band's upper edge is free — and removes much of
   the continuous daytime coil duty: hours 12–18 save 3.1 units between them.

Net: gas 27.19 → 23.27 MWh (−14.4 %), coil heat 26.72 → 22.97 (−14.0 %),
ventilation-attributable load 14.79 → 12.50 (−15.5 %), comfort −0.54 → −1.12.

What this is *not*: perfect per-zone occupancy foresight is worth +2.6 units here
(`docs/voi_oracle.md` §9.3), a fifth of the effect; and the 79-node LP, which
does look 8 h ahead, moves the *opposite* way — it buys night capacity margin and
loses 8.1 units at night (§5). It is a **supervisory sequencing** effect: which
hours the plant is allowed to run, and at what zone temperature the recovery heat
is delivered. That is why it ships as two numbers in `PARAMS` and needs no model,
no forecast and no occupancy channel.

### Why it was not found earlier, and what it costs

CtrlW's 282 K standby floor is Wu's own default, inherited by the whole family;
the schedule's night setpoint is 289 K, so nobody had tested *releasing* the
floor. Every earlier night experiment moved it the other way — `float_night = 285 K`
cost 6.9 units, and 278 K went through a band-clamp path that bought cooling in a
heating climate (`docs/opt_gap.md` §6). §4.1 now sweeps the floor: the gain
increases monotonically as it falls and saturates at 262 K, where it stops
binding and the building free-floats.

Fragility, stated because it is the deciding risk: the mechanism relies on the
building being empty overnight (true by construction in this challenge) and on
the recovery fitting inside the comfort window in every scenario. Where the plant
cannot recover a colder mass in time, the bill comes back as comfort at 94 reward
units per kelvin (§3). `dev30_fairbanks` + `dev30_phoenix` at 8641 steps are the
test — §6.

## 1. Headline table (anchor, 2016 steps)

| controller | anchor total | Δ vs CtrlW | Δ vs CtrlW4 | mechanism of the difference |
|---|---|---|---|---|
| **CtrlW** (`results/cg_ctrlw_anchor`) | **−93.123** | — | +0.004 | Wu's controller + our layer: band +0.12 K inside the schedule, adaptive 2–3 h preheat, reward-consistent fuel weights |
| **CtrlW4** (`results/cg_ctrlw4_anchor`) | **−93.127** | −0.004 | — | + fixed 2 K pre-dawn night lift. **Inert on the anchor** (0.004 units; on the 30-day sets it is −0.22 %) |
| **CtrlMPC v1** (`results/optg_mpc2_anchor`) | **−100.200** | −7.078 | −7.073 | receding-horizon LP on the 79-node twin, 8 h/15 min, no occupancy, plan → band floor. Day plan **+3.3 better**, night **−8.1**, comfort **−2.3** (§5) |
| **CtrlMPC2** (`results/cg_mpc2_night`) | **−94.146** | −1.023 | −1.019 | the three LP defects fixed (physical heat price, physical coil cold side, plan margin, night ceiling): recovers **+6.05** of the 7.08 and still does not find the release |
| **CtrlW5** (`results/cg_w5_final`, `policies/ctrl_w5.py`) | **−80.306** | **+12.817** | **+12.821** | night floor released + cooler hot-water loop: gas −14.4 %, comfort −0.6 (§0, §4) |
| CtrlW4 + released night floor only | −81.056 | +12.067 | +12.071 | the single knob responsible for 12.1 of the 12.8 |
| CtrlW4 + the 150-eval CMA bound (`standby_minimum` at the space's floor) | −83.109 | +10.014 | +10.018 | the same lever, stopped at the search-space bound |
| perfect per-zone occupancy foresight (`docs/voi_oracle.md` §9.3) | −90.523 | +2.600 | +2.604 | information only; not available to a submission |
| public MOBI-EVERGi / Houssem twin+LP | −81.9 / −85.4 | +11.2 / +7.7 | | for scale only |

## 2. The air side is not the gap (measured per VAV-step)

`eff.py` reconstructs the operating point from a capture:
`T_dis = T_sup + reheat/g`, `Q_net = reheat + g(T_sup − T_zone)`, and
`gas/Q_net = 1 + (T_zone − T_mix)/(T_dis − T_zone)` exactly.

| | CtrlW | CtrlMPC v1 | CtrlW5 |
|---|---|---|---|
| gas/Q_net, episode | 1.975 | 2.037 | 2.061 |
| gas/Q_net at the 308.15 K cap, same Q_net | 1.949 | 1.946 | 2.045 |
| discharge temperature, median | 35.00 °C | 34.96 °C | 35.00 °C |
| discharge, p10 / mean cap headroom | 25.4 °C / 2.3 K | 11.8 °C / 5.9 K | **34.4 °C / 1.1 K** |
| attainable saving if every VAV ran at the cap flow | 0.34 MWh = 1.15 units (1.2 %) | 1.27 MWh = 4.29 units | 0.18 MWh = 0.58 units (0.7 %) |
| flow used / flow the cap implies | 0.89× | 0.82× | 0.94× |

The ×2.24 multiplier `docs/opt_gap.md` quotes is the *counterfactual* split
(vent-attributable share of the coil), not the marginal cost; the marginal cost a
controller can move is this number, and CtrlW is already within 1.3 % of its
floor. **The air side has ~1 unit of slack and CtrlW5 takes most of it.** The
planner, by contrast, runs 18 % more airflow than the cap implies and pays 3 %
more gas per delivered watt, because its LP models the coil's cold side as the
mixed-air supply temperature instead of the zone (§5).

By hour (reward units, from each run's own `by_hour`):

| | 00–04 | 04–06 | 06–12 | 12–19 | 19–24 |
|---|---|---|---|---|---|
| CtrlW | −3.365 | −14.450 | −41.626 | −30.291 | −3.390 |
| CtrlMPC v1 | −3.603 | −13.427 | −40.411 | −31.508 | −11.249 |
| CtrlW5 | −0.030 | −11.778 | −40.968 | −27.177 | −0.351 |
| Δ CtrlW5 − CtrlW | **+3.34** | **+2.67** | +0.66 | **+3.11** | **+3.04** |

## 3. The comfort exchange rate that rules out band-shaving

The reward's productivity term is `max_prod·(1 − shortfall)·occupancy·dt` with
`shortfall = deviation/5 K`, occupancy-weighted and **never capped**: a uniform
1 K shortfall over the week's 780 occupied zone-hours costs
0.6 × (1/5) × 780 = **94 reward units**. One kelvin of building temperature is
`UA_eff 1.7 kW/K × 168 h = 0.29 MWh` of net heat, i.e. 0.57 MWh of gas =
**1.9 units**. The exchange rate is 50 : 1 against letting the *occupied* band
go, which is why

* CtrlW holds +0.41 K above the band bottom (p10 +0.11, p90 +1.71): 0.8–1.7
  units of heat, bought as insurance against a tracking shortfall that would cost
  10–90× more;
* shaving the band loses, measured: `comfort_margin 0.20` −0.44, `band_edge`
  −1.53 on the anchor and −4.3 % on the six fairbanks scenarios
  (`docs/opt_gap.md` §6);
* CtrlW5 keeps that behaviour — it pays 0.6 units more comfort, not less.

What §0 exploits instead is the *unoccupied* band, where the same reward charges
nothing. That is the line between sequencing the plant and trading comfort, and
it is where the 12.8 units live.

## 4. Tuning bound of the heuristic family

CMA-ES (`cma` 4.5.0, offline) over CtrlW4's 16 supervisory knobs
(`sbx/spaces/ctrl_w4_anchor.json`: comfort margin, preheat horizon/safety/slack,
recovery, comfort and gas prices, hot-water target, standby band, response scale,
night-lift gain and lead), anchor only, 2016 steps, 2 simulator processes,
`--cenv fused --fast`, one fresh process per evaluation (`tune_anchor.py`).
Campaign, all on the anchor:

1. a 6-point smoke test at sigma 0.18 from the defaults;
2. a 34-point run at sigma 0.18 whose driver minimised the (negative) total
   reward instead of maximising it — the gradient direction was wrong, so it
   drifted away from the optimum, but every point is a valid anchor measurement
   and its best (−90.241) was re-run independently (`results/cg_ctrlw5_rnd`);
3. **150-evaluation CMA-ES at sigma 0.05 seeded from that verified point**, with
   the default point injected in every population (deduplicated, so it costs one
   evaluation), so the reported bound can never be worse than the hand-set
   controller.

| run | anchor total | Δ vs CtrlW4 | Δ % |
|---|---|---|---|
| CtrlW4 (defaults) | −93.127 | — | — |
| best of 40 random points (sigma 0.18) | −90.241 | +2.886 | +3.10 % |
| **CMA-ES best, 150 evaluations** | **−83.109** | **+10.018** | **+10.76 %** |
| + the floor sweep below (search bound released) | −80.298 | +12.829 | +13.78 % |

CMA went from 1 to 10 units in 15 generations and was still improving when the
budget ran out. Its optimum sits on the *lower bound of `standby_minimum`*
(274.006 in a [274, 289] range), which is why the bound is not the end of the
story.

### 4.1 The optimum is "stop heating at night": the floor sweep

Holding the other tuned knobs fixed and moving only `standby_minimum`
(`results/cg_sweep_sb*`):

| `standby_minimum` (K) | °C | anchor total | comfort |
|---|---|---|---|
| 282.0 (CtrlW / CtrlW4 default) | 8.9 | −93.13 | −0.54 |
| 274.0 | 0.9 | −83.11 | −1.35 |
| 272.0 | −1.2 | −82.01 | −1.35 |
| 270.0 | −3.2 | −81.19 | −1.35 |
| 268.0 | −5.2 | −80.71 | −1.35 |
| 265.0 | −8.2 | −80.46 | −1.35 |
| **262.0 and below (identical to the last digit)** | ≤ −11 | **−80.30** | −1.35 |

The curve saturates at 262 K because below that the floor never binds: the
building simply free-floats. So the optimum is not "a deep setback", it is **no
overnight heating at all**, and the 12.8 units are the difference between heating
an empty building and not. The comfort term is *flat* at −1.35 K across the whole
sweep: none of this is bought from the occupied band. Secondary sweeps at the
262 K floor (night-lift lead 2/3/4.5/6 h, preheat horizon 1.5/2.25/3/4 h) move the
total by ≤0.02 units — once the plant is off, the recovery is already
capacity-saturated and its timing stops mattering.

### 4.2 The two knobs that ship

The CMA optimum spends comfort (−1.35) on other knobs too. Isolating the
mechanism (`results/cg_w4_*`):

| point | anchor total | comfort |
|---|---|---|
| CtrlW4 | −93.127 | −0.54 |
| `standby_minimum = 262` only | −81.056 | −1.04 |
| + `night_lift_enabled = false` | −81.236 | −0.79 |
| + `hot_water_target = 336.32` | **−80.306** | **−1.12** |
| 16-knob CMA optimum | −80.298 | −1.35 |

So `policies/ctrl_w5.py` = CtrlW4 with **`standby_minimum` 282 → 262 K and
`hot_water_target` 345 → 336.32 K**, everything else at CtrlW4's defaults
(numpy only; restoring those two numbers reproduces CtrlW4 bit-exactly). The
cooler hot-water loop is worth 0.75 units because the tank's standing loss is
1.2 of the anchor's 93 and the coil is limited by the 308.15 K discharge cap, not
by the water side.

### 4.3 Component decomposition of the gain

| term | CtrlW | CtrlW5 | Δ |
|---|---|---|---|
| heat_rest (reheat net of the outside-air share) | −67.858 | −59.104 | **+8.75** |
| heat_vent (outside-air share) | −21.202 | −16.786 | **+4.42** |
| fan_supply | −1.366 | −1.161 | +0.21 |
| fan_exhaust | −0.410 | −0.348 | +0.06 |
| pump | −0.549 | −0.772 | −0.22 |
| boiler_standby | −1.197 | −1.015 | +0.18 |
| comfort | −0.541 | −1.119 | −0.58 |
| **total** | **−93.123** | **−80.306** | **+12.82** |

Volumes: gas billed 27.19 → 23.27 MWh, coil heat 26.72 → 22.97, ventilation load
14.79 → 12.50 MWh, fan 0.27 → 0.21. `analyze.py` on the two captures: night
(00–04) mean building temperature 287.9 → 287.4 K, night coil duty 30 → 0 kW.

## 5. The planner: diagnosis and fix

### 5.1 What version 1 does (measured)

`results/optg_mpc2_anchor` = −100.200 vs CtrlW −93.123, split by the recorder's
own buckets:

| bucket | CtrlW | CtrlMPC v1 | Δ |
|---|---|---|---|
| weekday occupied (06–19) | −66.18 | −63.77 | **+2.41** |
| preheat (04–06) | −12.96 | −12.05 | **+0.91** |
| weekday night (19–06) | −1.22 | −9.33 | **−8.11** |
| weekend | −12.22 | −12.21 | +0.02 |

**The planner's daytime plan is better than CtrlW and its night plan is much
worse** — the same mechanism as §0 with the opposite sign: it *buys* night
capacity margin (holds ~291.5 K from 19:00, +2.4 MWh of the week's gas) instead
of selling it. Three defects, all in the LP, none of them the horizon (the plan is
already re-solved hourly over 8 h):

1. **Heat is priced at half its true cost.** `cost[QI] = cg` charges one watt of
   zone heat as one watt of gas; the plant pays ~2 W (measured, §2), because the
   coil must also lift the mandatory outside-air share to the discharge cap.
2. **The coil's cold side is the wrong temperature.** The LP's capacity per m³/s
   is `RHO_CP·(T_dis − T_sup)`; the zone receives `RHO_CP·(T_dis − T_zone)`, 2×
   less at these temperatures. The plan under-plans airflow, and the mapped
   episode runs a 19 °C discharge at night (p10 11.8 °C, §2).
3. **The plan is a floor with no margin.** The LP holds the band bottom as a hard
   constraint (294.00 K exactly) and the mapping hands that to CtrlW's VAV layer
   as the band floor, so the occupied-hour tracking error triples the comfort term
   (−0.54 → −2.85).

All three are now switches on `CtrlMPC` (defaults reproduce version 1
bit-exactly); `policies/ctrl_mpc2.py` turns them on
(`heat_price_mode='phys'`, `tcold_mode='zone'`, `plan_margin=0.15`,
`night_floor=282`, `night_cap=284`).

### 5.2 Variants measured on the anchor

| variant | anchor total | Δ vs v1 | comfort | what it shows |
|---|---|---|---|---|
| v1 | −100.200 | — | −2.846 | night margin bought at half price; band floor with no margin |
| + physical heat price only (`cg_mpc_price`) | −105.370 | −5.17 | −5.326 | alone it is *worse*: the plan becomes more expensive per watt but is still unrealisable through the airflow model, so the mapping collapses on comfort |
| + physical heat price + physical coil cold side (`cg_mpc_zone`) | −104.109 | −3.91 | −5.233 | same: fixing the two prices without the margin makes the plan track badly |
| **= MPC2**: both + 0.15 K plan margin + night ceiling 284 K (`cg_mpc2_night`) | **−94.146** | **+6.05** | −0.857 | the night loss is gone (19–24 h: −7.86 → −0.08 units), the day gain goes too (06–12 h: +1.21 → −0.42), comfort is restored |
| MPC2 + night floor released to 262 K, ceiling off (`cg_mpc2_float`) | −98.528 | +1.67 | −0.520 | the LP does not discover the release even when the constraint allows it: it buys heat it cannot use |

MPC2 lands 1.0 unit behind CtrlW and 13.8 behind CtrlW5. The three defects were
worth 6.05 of the 7.08 deficit — the diagnosis is confirmed — but with correct
prices the LP's economics collapse onto "hold the band exactly with a hard
constraint", which is worse than CtrlW's own margin and never considers turning
the plant off. **A planner that predicts better cannot find this; only a policy
that is allowed to stop heating can.**

## 6. dev30 confirmation

## 6. dev30 confirmation

`dev30_fairbanks` + `dev30_phoenix`, 8641 steps (30 days), shipped simulator,
`--cenv fused --fast`, 2 processes, CtrlW4 as the reference (same flags:
`results/cg_dev30_ctrlw4`) and CtrlW5 (`results/cg_dev30_w5b`).
`building_train_4/phoenix_8` fails for **both** policies with the same
pre-existing weather-range error (`Timestamp not in range ... after the latest
timestamp`), so the comparison is 11 scenarios.

| scenario | CtrlW4 | CtrlW5 | Δ | Δ % |
|---|---|---|---|---|
| train_1/fairbanks_1 | −393.16 | −383.72 | **+9.44** | −2.40 % |
| train_2/fairbanks_2 | −106.05 | −93.58 | **+12.47** | **−11.76 %** |
| train_4/fairbanks_4 | −14.03 | −13.25 | +0.77 | −5.52 % |
| train_5/fairbanks_5 | −13.35 | −13.04 | +0.30 | −2.28 % |
| train_7/fairbanks_7 | −72.29 | −70.70 | +1.60 | −2.21 % |
| train_8/fairbanks_8 | −206.65 | −189.54 | **+17.11** | **−8.28 %** |
| train_1/phoenix_5 | −123.04 | −123.04 | 0.00 | 0.00 % |
| train_2/phoenix_6 | −62.20 | −62.20 | 0.00 | 0.00 % |
| train_5/phoenix_1 | −20.45 | −19.87 | +0.58 | −2.84 % |
| train_7/phoenix_3 | −47.13 | −47.13 | 0.00 | 0.00 % |
| train_8/phoenix_4 | −83.00 | −83.00 | 0.00 | 0.00 % |
| **subset score (mean of \|per-step reward\|, 11 scenarios)** | **0.0120079** | **0.0115632** | | **−3.70 %** |

Where the change lands: over the 6 fairbanks scenarios energy+carbon improves by
**55.5** reward units and comfort costs **13.9** (a 4 : 1 trade, against the
50 : 1 the *occupied* band would charge — this is the unoccupied band); over the
5 phoenix scenarios comfort is bit-identical (−1.81) and energy+carbon improves
0.6.

**Better on 7 of 11, identical to the last digit on 4, worse on none.** All six
fairbanks scenarios improve (2.2–11.8 %; subset score −5.18 %); the phoenix scenarios are inert to four
decimals wherever the night floor never binds and the boiler is not used, and
`phoenix_1` improves 2.8 % from the cooler hot-water loop. The two defaults are
therefore safe to ship as they stand: they cannot be worse than CtrlW4 on any
scenario where the release does not fire, because they are *literally the same
numbers* there.

Two honest qualifications. First, the 7-day anchor overstates the effect: the
same controller gains **2.4 %** on the 30-day `fairbanks_1` episode (+9.44 on
−393.16) against 13.8 % on the first week, which is the coldest and the one the
tuning was done on; the population of 30-day fairbanks deltas is +2.2 to +11.8 %.
Second, the mechanism is cold-climate. On the warm set it is inert, not harmful,
but a controller that had to carry the whole hidden set would want the release
gated on a climate observable; here the release is naturally self-gating (it only
acts when a zone sits on the standby floor, which in cooling climates it does
not).

**Submittable? Yes — it already is.** `policies/ctrl_w5.py` is CtrlW4 with two
numbers in `PARAMS`: numpy only, no `smart_buildings` import, no building or
weather id, no future information, well inside the 80 ms/step budget (the same
code path as the packaged entry). The offline pieces of this study (CMA-ES,
HiGHS twin, captures) are not part of it and are not needed to run it.

## 7. What kind of control closes the gap

**Sequencing, not information and not prediction.** The 12 % on this anchor is
bought by deciding *when* the plant heats and *at what zone temperature* the
recovery heat is delivered: release the night (an empty building has no comfort
requirement in this reward, and one cold zone per AHU group was holding every
group's plant on all night), then deliver the whole recovery in one burst at the
308.15 K discharge cap into the now-colder mass, where the mandatory 30 %
outside-air share costs a smaller fraction of the coil lift, and let that charge
drift through the band during the day. Measured: −80.306 against CtrlW's −93.123
(+13.8 %), gas −14.4 %, comfort −0.6, better than both public entries we are
chasing (−81.9, −85.4) — and it needs no model, no forecast and no occupancy
channel, so it ships as two numbers in `PARAMS` (`policies/ctrl_w5.py`).

The measurements that *exclude* the other explanations are on the same page: the
air side is within 1.2 % of its cap-limited floor (§2); perfect per-zone
occupancy foresight is worth +2.6 units, a fifth of the effect
(`docs/voi_oracle.md` §9.3); and an 8-hour LP planner on a 79-node twin of the
same building, with the same reward weights, is 7.6 % worse because it prices
night heat at half its cost and buys the margin this controller sells (§5). What
generalises is the principle — *let the plant idle when the building is empty and
put the recovery where the coil is efficient* — and §6 is the evidence on 30-day
episodes.

## 8. Frost protection: exposure, cost, and what to ship

CtrlW5's release lets whole AHU groups idle, and a building that idles in
Fairbanks does not stay above freezing on its own: the release gains 12.8 units
on the anchor by letting the coldest room reach −8.6 °C. The hosts encouraged a
frost-protection limit and asked what it costs, so both are measured here.
Recorder: `scratch/ControlGap/frost.py` — it steps the episode exactly as
`sbx.evalrun` does and records only `building.get_zone_average_temps()` after each
step (no monkeypatching, no extra reward-info call, no clock read). The printed
total is the proof it does not perturb the episode: it reproduces the evalrun
total to four decimals in every configuration where both exist.

### 8.1 Exposure

Anchor (2016 steps). `h-any` = hours in which *any* room is below the threshold;
`zone-h` = room-hours below it (39 rooms × 168 h).

| policy | total reward | coldest room | min T | h-any < 5 °C | h-any < 0 °C | zone-h < 0 °C |
|---|---|---|---|---|---|---|
| CtrlW | −93.1229 | room_20 | **+8.55 °C** | 0.00 | 0.00 | 0 |
| CtrlW5 (floor 262 K) | −80.3056 | room_25 | **−8.58 °C** | 26.42 | 19.08 | 111 |
| CtrlW4, floor 278 K | −86.6810 | room_1 | +4.66 °C | 26.42 | 0.00 | 0 |
| CtrlW4, floor 280 K | −89.5751 | room_20 | +6.53 °C | 0.00 | 0.00 | 0 |

`dev30_fairbanks`, 8641 steps (30 days) — the same recorder, the same policies:

| scenario | CtrlW total | CtrlW min | CtrlW5 total | CtrlW5 min | CtrlW5 h-any <5 / <0 °C | CtrlW4 @278 K total | @278 K min |
|---|---|---|---|---|---|---|---|
| fairbanks_1 | −395.15 | +8.53 | −383.72 | **−11.21** | 136.4 / 89.9 | −386.91 | +4.66 |
| fairbanks_2 | −107.26 | +8.61 | −93.58 | **−5.71** | 181.8 / 87.6 | −99.35 | +4.56 |
| fairbanks_4 | −13.88 | +10.74 | −13.25 | +10.88 | 0 / 0 | (inert: floor never binds) | — |
| fairbanks_5 | −13.42 | +16.82 | −13.04 | +16.82 | 0 / 0 | (inert) | — |
| fairbanks_7 | −73.01 | +8.79 | −70.70 | +7.53 | 0 / 0 | (inert) | — |
| fairbanks_8 | −208.82 | +8.58 | −189.54 | **−11.00** | 177.9 / 97.4 | −197.70 | +4.68 |
| **set score** | **0.0156529** | | **0.0147329** | | | est. **0.015128** (−3.4 % vs CtrlW) | |

So the exposure is real but concentrated: on the three cold scenarios
(`fairbanks_1/2/8`) CtrlW5 freezes rooms for ~90 h each and reaches −11 °C;
the three mild ones never leave the band. CtrlW never comes near 5 °C anywhere.

### 8.2 Cost of a frost floor, anchor

`standby_minimum` swept with CtrlW5's other knobs (hot-water 336.32 K), 2016
steps, same recorder:

| floor (K) | °C | total | Δ vs floor 262 = CtrlW5 | min room T | h-any <0 °C |
|---|---|---|---|---|---|
| 262 (CtrlW5) | −11 | −80.306 | — | −8.58 | 19.1 |
| 274 | +0.9 | −83.090 | −2.78 | +0.63 | 0 |
| 276 | +2.9 | −84.638 | −4.33 | +2.66 | 0 |
| 278 | +4.9 | −86.681 | −6.37 | +4.66 | 0 |
| 280 | +6.9 | −89.575 | −9.27 | +6.53 | 0 |
| 282 (CtrlW4 default) | +8.9 | −93.127 | −12.82 | +8.55 | 0 |

The curve is steep at the top and flat at the bottom: giving up the last two
kelvin of protection (282 → 280 K) buys 3.5 units, and 280 → 278 buys another
2.9 at the price of letting the coldest room dip to 4.66 °C. **Zero freezing
everywhere is reached at 278 K**; **"no room below 5 °C" needs 280 K.**

On the three cold 30-day scenarios the same floor gives min 4.56/4.66/4.68 °C and
**zero room-hours below freezing**, at a set cost of ~3.4 % of CtrlW's score
(0.015653 → 0.015128, with the three mild scenarios counted unchanged because the
floor never binds there).

### 8.3 Recommendation

* **Paper's main controller: floor 278 K** (`CtrlW5.PARAMS["standby_minimum"] =
  278`). It keeps a stateable guarantee — measured minimum +4.6 °C across all
  cold scenarios, zero room-hours below 0 °C, against CtrlW's +7.5…+8.6 °C — and
  still beats CtrlW by 6.9 % on the anchor (−86.68 vs −93.12) and ~3.4 % on
  `dev30_fairbanks`. That is the configuration to describe in the paper, with the
  cost curve above quoted verbatim: the frost limit costs 6.4 of the 12.8 units
  the unconstrained release finds, and the remaining 6.4 is the paper's claim.
* **Submit 278 K**, not 262 K. The unconstrained release is a leaderboard
  artefact of the challenge's occupancy model (no arrivals between 19:00 and
  06:00); with any night occupancy, or any hidden scenario colder than
  `fairbanks_8`, it is unbounded risk for at most 6 units.
* **Do not go below 278 K** unless the crate is explicitly frost-tolerant: 274 K
  leaves rooms at +0.63 °C on the anchor, which is inside the pipe-freezing
  margin of the plant it is meant to protect.
* If a 5 °C floor is the criterion (rather than 0 °C), 280 K is the honest
  choice; it still gains 3.8 % over CtrlW on the anchor and needs no new code,
  only the one number.

Flag for whoever compares dev30 numbers across flag sets: on 8641-step episodes
the plain-shipped CtrlW total and the `--cenv fused --fast` total differ by ~2
units on `fairbanks_1` (−395.15 vs −393.16), while the 2016-step anchor is
identical to the last digit in all three configurations and CtrlW5's 30-day total
is identical either way (−383.72). The frost table above is on the plain-shipped
numbers (it reproduces `docs/opt_gap.md` §6 exactly); the CtrlW4-vs-CtrlW5
dev30 deltas in §6 were measured within one flag set, so the comparison stands,
but absolute 30-day totals should not be mixed between the two.

**Correction (orchestrator, 2026-11-06, measured):** there is no flag effect.
CtrlW on `building_train_1/fairbanks_1`, 8641 steps, shipped, gives
−395.1538459855292 in all four configurations — plain, `--cenv fused`, `--fast`,
and `--cenv fused --fast` (`results/ctrw_all`, `scratch/orch/x_f1_*`). The
−393.16 above is the **CtrlW4** baseline of the §6 table, not CtrlW, so the
"~2 units" is CtrlW4 vs CtrlW (different policies), and absolute totals *can* be
compared across flag sets.
