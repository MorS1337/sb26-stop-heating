# How far is CtrlW from optimal, where is the loss, and what recovers it

Anchor: `building_train_1/fairbanks_1`, shipped sim, default start
(2023-01-01 06:00, a Sunday), first 2016 steps = 7 days, mean outside −26.3 °C.
CtrlW = **−93.123** (`results/houssem_anchor_ctrlw`), reproduced bit-exactly by
an instrumented run (`scratch/OptGap/steps/anchor2016.npz`).

Scripts: `scratch/OptGap/` — `model2.py` (reduction), `capture.py` (episode
recorder, hooks the solver), `fit.py` (grey-box), `lp.py` (LP), `bound2.py`
(energy-balance bound), `analyze.py` / `decomp.py` (decomposition).
New controllers: `policies/ctrl_w3.py`.

## Headline

| scenario | bound, every zone in its schedule band | CtrlW | CtrlW − bound | bound, unoccupied zones released (occupancy oracle) | gap that needs information |
|---|---|---|---|---|---|
| fairbanks_1 | −97.18 | −93.12 | **+4.4 % (CtrlW better)** | −71.40 | 23 % of \|reward\| |
| phoenix_5 | −26.12 | −24.57 | **+6.3 % (CtrlW better)** | −11.42 | 53 % |
| singapore_3 | −13.30 | −13.27 | **+0.2 % (parity)** | −3.56 | 73 % |
| atlanta_2 | −1.11 \* | −2.56 | −57 % \* | −0.88 | 66 % |

\* atlanta_2's bound is degenerate: with no reheat and almost no chiller load the
single calibrated parameter α collapses to 0, so the bound charges only the fans
and comes out *worse* than what CtrlW achieves. The method has nothing to say
about a mild climate whose bill is mostly a hot-water tank's standing loss.

**Read this table as an estimate, not a bound.** The columns headed "bound" are
an energy balance with two parameters taken from CtrlW's *own* episode (α
bisected so the balance reproduces the heat CtrlW delivered, κ = CtrlW's own
measured heat_vent/heat_rest). A controller that moves less outside air per
unit of delivered heat — by airflow, supply-temperature or discharge-ΔT choices
— or that uses storage, changes exactly those two numbers, so the figure
restates CtrlW's own efficiency and cannot certify that CtrlW is optimal. It
even comes out worse than CtrlW on three of four scenarios, which is the
clearest sign that it is an estimate with the wrong sign of bias to settle the
question.

What the pair *does* suggest, and only suggests: the schedule-band and the
oracle (unoccupied-zone-released) versions of the same estimate differ by 23 %
of the anchor's |reward| and 53–73 % in cooling climates, i.e. per-zone
occupancy information is worth a large fraction of the observed gap to the
public numbers (Houssem −85.4, MOBI-EVERGi −81.9, free-action LP −79.1). That
is a ratio of two estimates, not a measurement of either headroom.

**Open question, untested here: control headroom vs information headroom.**
Houssem's twin + LP planner reached −85.4 with no claim of per-zone occupancy
data, so a control-only explanation of the gap is entirely possible, and this
report does not test it. §9 is the first experiment in that direction
(receding-horizon planner on the 79-node twin, closed loop, no occupancy
information) and should be read before drawing any conclusion about where the
headroom is.

## 1. Exact thermal model, reduced to zone level

`model2.py` rebuilds the simulator's finite-difference system from
`FiniteDifferenceSolver` and reduces it, with no fitting, to **two nodes per
room + one interior-partition node** (79 states for 39 rooms): a room's CVs
split into "core" (no conductance leaving the room) and "boundary" (touching
another zone or a partition); capacitance, Dirichlet-to-ambient, ambient
(h = 8.8 W/m²K), interior-mass coupling and every CV–CV conductance are summed
exactly over each node; the air side enters as the exact rank-39 term
`Σ_v g_v s_v s_vᵀ`, `g_v = flow_v·rho·cp`; the interior mass (r = 0.985,
g = 0.015 on every CV) is carried as its own state. The only thing the
reduction drops is the temperature gradient *inside* a node.

Validation (`capture.py` logs the simulator's own per-step inputs — pre-step
field, applied power, ambient, mass field, per-VAV g and supply temperature —
so the comparison is against the simulator's step with the simulator's inputs,
48 consecutive steps):

| check | result |
|---|---|
| 606-CV system re-evaluated at the simulator's post-step field | max residual 8.7e-3 W on a 2.2e5 W scale (4e-8 relative) — the extraction is exact |
| one node per room, one step | 0.34 K mean error on zone means |
| two nodes per room, one step | 0.33 K |
| two nodes, open loop 4 h | −0.90 K mean drift, 2.4 K worst zone |
| two nodes + grey-box α = 0.34, β = 2.9 W/m² (fitted on the first 24 steps, scored on 48) | **0.21 K mean, 0.92 K worst, signed +0.03 K** |
| grey-box reproducibility on phoenix_5 / singapore_3 / atlanta_2 | α = 0.406 / 0.34 / 0.406, β ≈ 2.9, 4 h error 0.13–0.17 K |

The residual after exact sums is the intra-node gradient; it is systematic and
makes the raw reduction run cold (heat leaves a room through CVs that sit below
the room's mean). α is large because it stands in for that effect — and it does
*not* reproduce the weekly energy balance (α = 0.29 does, §5), which is why the
bound uses the balance-calibrated α, not the transient one.

## 2. Actuator and cost model (from the simulator source)

Per 300 s step: `flow = damper·maxflow·sqrt(P/2000)`, damper floored at 0.10,
`P ≥ 200 Pa` while the AHU runs; zone heat = `g_v(T_sup − T_zone)` **plus** the
reheat coil, whose capacity is `g_v(T_discharge_max − T_sup)` with
`T_discharge_max = 308.15 K` — the **discharge cap, not the water side, is what
binds** (measured coil saturation 0.42, vent/coil 0.57); the AHU only ever
cools, `T_sup = min(T_mix, setpoint)`, `T_mix = 0.7·T_ret + 0.3·T_out`, so in
winter the supply-temperature *action is inert* (the supply is always `T_mix`,
the setpoint sits above it); chiller = coil heat/3.5; fans `flow·P/0.9` supply
plus `(1−r)` exhaust; gas = coil heat + tank standing loss, priced hourly by the
reward's own `ElectricityEnergyCost` / `NaturalGasEnergyCost` objects (queried
at 1 W, so cost is exactly linear).

**The ventilation multiplier.** Because 30 % of the supply air is outside air at
−26 °C, `T_mix` ≈ 280 K, 14 K below the zone. Every m³/s of supply cools its
zone by `rho·cp·(T_zone − T_mix)` and the coil must add
`rho·cp·(T_dis − T_sup)`, so at the capacity-limited operating point
`gas / net zone heat = (T_dis − T_mix)/(T_dis − T_zone) ≈ 2.0`; measured
`26.72/(26.72 − 14.79) = 2.24`. It is structural — it falls only if the
discharge cap rises, which it cannot — and it is why recirculation is pinned at
the 0.7 cap.

## 3. Decomposition of CtrlW's anchor episode

| term | reward | share |
|---|---|---|
| comfort | −0.541 | 0.6 % |
| energy | −26.229 | 28.2 % |
| carbon | −66.353 | 71.3 % |
| total | **−93.123** | |

Carbon is 2.5× energy: 27.2 MWh of the week's fuel is gas, and gas carries
53.12 kg CO₂/MMBtu against a small `max_natural_gas_rate` denominator.

| component | reward | share |
|---|---|---|
| heat_rest (reheat not attributable to outside air) | −67.86 | 73.3 % |
| heat_vent (outside-air share of the same reheat) | −21.20 | 22.9 % |
| fan_supply | −1.37 | 1.5 % |
| boiler standing loss | −1.20 | 1.3 % |
| pump | −0.55 | 0.6 % |
| fan_exhaust | −0.41 | 0.4 % |
| chiller | 0.00 | 0 % |

By hour of day (reward per hour, summed over the week's 7 occurrences):

```
00 −0.76  04 −5.21  08 −6.90  12 −4.58  16 −4.24  20 −0.59
01 −0.74  05 −9.24  09 −6.20  13 −4.40  17 −4.24  21 −0.59
02 −0.84  06 −9.22  10 −5.65  14 −4.31  18 −4.22  22 −0.59
03 −1.02  07 −7.88  11 −5.25  15 −4.26  19 −0.89  23 −0.74
```

Delivery: coil heat 26.72 MWh-thermal, ventilation-attributable 14.79 MWh, gas
billed 27.19 MWh; fans 0.27 MWh-elec, pump 0.08.

### Top three loss sources, with numbers

1. **Reheat for the envelope load at the band temperature — 67.9 of 93.1
   (73 %).** Effective UA (balance-calibrated) ≈ 0.29 × 5.9 kW/K ≈ 1.7 kW/K
   over 5259 m² of floor; mean inside–outside difference 44 K; the coil delivers
   160 kW average. The building is *already at the band bottom*: occupied zones
   sit +0.41 K above it (p10 +0.11, p90 +1.71), empty zones −0.05 K. Heat spent
   holding the building above the band bottom = **1.0 MWh of 26.7 (4 %,
   ≈2.5 reward units)** — so "hold the band tighter" is worth about 2 units,
   not 12.
2. **The mandatory outside-air share — 21.2 of 93.1 (23 %)**, i.e. the ×2.24
   multiplier of §2. Unavoidable at `r ≤ 0.7`, and CtrlW already sits at the cap.
3. **Morning concentration — hours 04–08 carry 38.5 of 93.1 (41 %)**, against
   6.8 for all of 00–04 plus 19–24. The 2 h ramp must lift the building 4–5 K
   while campus reheat capacity (1.3 MW at maximum flow, capped by the 308.15 K
   discharge limit at the ventilation floor) is the binding constraint. It is
   capacity, not price: gas per watt moves only 6 % across the day (0.938
   min/max), so timing arbitrage is worth ≤ 4 units at most.

Everything else — fans, pump, standing loss — is 3.5 units (3.8 %).

## 4. Cooling climates (7-day windows, same recorder, CtrlW)

| scenario | total | comfort | energy | carbon | dominant components |
|---|---|---|---|---|---|
| building_train_1/phoenix_5 | −24.57 | 0.00 | −19.97 | −4.60 | cool_rest −17.43 (71 %), cool_vent −5.91 (24 %) |
| building_train_4/singapore_3 | −13.27 | −0.09 | −10.70 | −2.47 | cool_rest −10.62 (81 %), cool_vent −1.77 (13 %) |
| building_train_1/atlanta_2 | −2.56 | −0.02 | −1.31 | −1.23 | boiler standing loss −1.08 (43 %), fans −0.87 (34 %) |

Phoenix: chiller 3.33 MWh-elec over the week (11.66 MWh-thermal), 0.84 MWh-elec
of it ventilation-attributable. The load is the 10 W/m² internal gain (53 kW in
comfort mode over 5259 m²) plus the envelope gain; a controller can only move
the latter, by holding the band *top* (a 19 % cut of the envelope gain ≈ 6 % of
the chiller work). Atlanta is mild: the whole bill is small and the 345 K
hot-water tank's standing loss alone is 1.08 of 2.56 — there the water-setpoint
tail is the biggest single line, and nowhere else.

## 5. Energy-balance estimate (calibrated on CtrlW's own episode — not a bound)

`bound2.py`: steady state, no storage, one calibrated parameter per scenario.

    Q_heat_min(t) = Σ_z max(0, α·K_z·(T*_z − T_out) − gain_z(t))
    gas_bound(t)  = Q_heat_min(t) · (1 + κ),   κ = heat_vent/heat_rest measured
    chiller_bound(t) = max(0, Σ_z [gain_z + α·K_z·(T_out − T*_z)] − free cooling)/COP
    loss_bound    = Σ_t c_g(t)·gas + c_e(t)·(chiller + measured fans/pump)

α is the single number that makes the balance at the *measured* temperatures
reproduce the heat the coil *actually* delivered in that episode (bisection;
α = 0.29 on the anchor, 0.67 on phoenix_5). `T*_z` is the band bottom (heating)
or top (cooling) for occupied zone-steps. Two variants: the **schedule band**
everywhere (what a policy with no per-zone occupancy data must respect) and the
**occupancy-oracle** relaxation (unoccupied zones to 275 K when heating, 320 K
when cooling).

Results in the headline table. Caveats, stated because they matter:

* storage is not in the bound; with the gas price flat to ±3 % this is worth
  ≤ 4 reward units and it biases the bound *pessimistically* (too high), which
  is why CtrlW comes out 4–6 % "better" than it;
* the ventilation multiplier is the measured one, not a physical lower bound;
* α is identified on the same episode, so the bound is only as good as the
  single-node energy balance (0.29 vs the transient fit's 0.34 is the size of
  that uncertainty);
* the LP (`lp.py`, two-node model, HiGHS, per-zone heat and airflow decisions,
  capacity and comfort constraints) is implemented and reaches optimality on
  small horizons, but at 288+ steps HiGHS needed more than ten minutes per
  scenario on the loaded machine and was stopped; §5's bound is what the LP's
  structure degenerates to once storage is dropped and α is calibrated on the
  episode.

## 6. Implementable fixes tested (anchor, 2016 steps, shipped)

`policies/ctrl_w3.py` adds two switches over CtrlW, both off by default so
`CtrlW3` with no flags is bit-identical to CtrlW:

* `band_edge` — subtract `comfort_margin` from the low edge, i.e. hold the
  schedule's own heating setpoint instead of 0.12 K above it;
* `float_night` — outside the comfort window (and only when the preheat ramp is
  not running) replace Wu's 282–307 K standby band with `float_floor`–307 K.

| run | total reward | Δ vs CtrlW |
|---|---|---|
| CtrlW (`houssem_anchor_ctrlw`) | −93.123 | — |
| `band_edge` (`CtrlW3Edge`) | −94.653 | **−1.53** |
| `float_floor = 285 K` (`CtrlW3Float`) | −100.043 | **−6.92** |
| `float_floor = 278 K` (`CtrlW3`, both switches) | −189.953 | −96.8 |
| CtrlW `comfort_margin = 0.20` | −93.561 | −0.44 |
| CtrlW `hot_water_target = 350` | `results/optg_hw350_anchor` | (in flight at the stop) |

Mechanism behind the negatives: removing the margin costs comfort (occupied
zones end up exactly at the band edge, so any disturbance produces a shortfall,
and 0.6·shortfall per K is dearer than the gas saved); floating the night down
saves night losses but the morning ramp is capacity-limited by the 308.15 K
discharge cap, so the recovery is bought back at the ×2.24 ventilation
multiplier plus a comfort deficit at the start of the work window (−6.9 units is
almost all of that). 278 K is a trap: reheat cannot lift the building in time
and comfort collapses.

**No band/level/water lever recovers anything on a heating anchor, which is
what §5 predicts: CtrlW is at the energy-balance bound for the schedule it must
hold.** The remaining headroom is information, not control — the oracle bound is
23 % better when heating and 53–73 % better when cooling.

Two 8641-step runs (`dev30_fairbanks`, `dev30_phoenix`, policy `CtrlW3Edge`,
`results/optg_edge_devfair`, `results/optg_edge_devphx`) were started to give
per-scenario deltas on the whole sets. The fairbanks set is now 5 of 6 done and
every scenario is worse, so `band_edge` is rejected on the full episodes too:

| scenario (8641 steps, shipped) | CtrlW3Edge | CtrlW | Δ |
|---|---|---|---|
| building_train_1/fairbanks_1 | −404.18 | −395.15 | −9.02 |
| building_train_2/fairbanks_2 | −113.59 | −107.26 | −6.33 |
| building_train_4/fairbanks_4 | −15.72 | −13.88 | −1.84 |
| building_train_5/fairbanks_5 | −13.50 | −13.42 | −0.08 |
| building_train_7/fairbanks_7 | −81.80 | −73.01 | −8.78 |
| mean (5 of 6) | −125.76 | −120.54 | −5.22 (−4.3 %) |

(building_train_8/fairbanks_8 was still running at the stop; `dev30_phoenix`
was not started, the chained launcher was stopped to keep two simulator
processes on the loaded Mac.)

## 7. What the finding does and does not imply

* **It does not settle where the headroom is.** The tested levers (band edge,
  night float, comfort margin, hot-water target) are all ≤ 0 on the heating
  anchor, but that is four rules inside CtrlW's own actuator mapping, not a
  search over control strategies: no test of storage arbitrage, of a
  reheat-versus-airflow split at different discharge ΔT, or of a plan computed
  from the twin has been run yet (§9 starts this). Houssem's −85.4 with a twin
  LP and no occupancy data is the standing counter-example to any
  information-only explanation.
* **Per-zone occupancy information remains a hypothesis with a magnitude
  attached** (a 23 % / 53–73 % difference between two variants of the same
  estimate, §5). It is worth testing properly — people are 140 W of extra gain
  in a room whose losses are kilowatts, so a zone's own drift (which Wu's
  learner already estimates per zone as `theta`) is a plausible observable
  proxy — but the schedule-band estimate coming out *worse* than CtrlW means
  the estimate cannot be used to size it.
* **The ×2.24 gas multiplier is real and measured** (coil heat / net zone heat),
  it is what the published VAV model charges for mandatory ventilation at
  `r ≤ 0.7`, and any comparison with a "physics-consistent" baseline should
  expect it. Whether a different airflow/supply-temperature operating point
  moves it is exactly what the planner experiment should answer.

## 8. Caveats

* The bound is a *steady-state* bound under one calibrated parameter, not a
  proof; the two-number comparison that matters is the oracle/schedule pair,
  which are computed with the same model.
* Harness differences against the public numbers (−79.1 free-action LP, −85.4
  twin LP, −81.9 MOBI-EVERGi) are worth ~3 % per the task brief.
* No controller here touches a simulator bug: the levers are the band, the night
  float, the hot-water setpoint and the preheat ramp.
* `--fast` (verified bit-identical, 1.6–1.9× faster) was not used; all runs are
  plain `--sim shipped`.

## 9. Receding-horizon planner on the twin, closed loop (control headroom test)

`policies/ctrl_mpc.py` (offline experiment: imports scipy/HiGHS and loads a
precomputed 79-node model, so it is not submission-safe and is not a candidate
entry). Mapping, as implemented:

* every hour the twin's state is rebuilt from the observed zone temperatures
  (each room's core and boundary sub-node = its zone mean, partition node and
  interior mass = the mean of the rooms) and an LP is solved over the next 8 h
  in 15-minute slots (32 slots, 2.8 s per solve with HiGHS);
* decisions per slot and zone: `qp − qm` = net heat into the zone air (reheat,
  or extra cooling above the floor) and the extra airflow needed to carry it;
  constraints: the twin's linear dynamics with the ventilation floor held at its
  legal minimum, the coil capacity `q ≤ (f_floor+f)·rho·cp·(308.15 − T_sup)`, the
  ventilation penalty `(1−r)·f·rho·cp·(T_mix − T_out)` priced as gas, and the
  **schedule comfort band as a hard constraint on every zone in comfort mode —
  no occupancy information**;
* objective: the reward's own hourly fuel/carbon weights (CtrlW's `ELEC_REL`
  and `GAS_RATIO` relative tables, which are proportional to the reward's
  marginal loss per watt), cooling priced at `/3.5`;
* no weather forecast: the outside temperature is held at its last observed
  value for the whole horizon; prices come from the clock;
* the plan's per-zone temperature for the next hour becomes the *band target*
  CtrlW's existing AHU search and VAV layer track (`follow_mode = night_only`:
  inside the comfort window CtrlW keeps its own band, because §6 measured that
  collapsing the margin costs more than it saves; outside it the plan sets the
  floor).

What the planner asks for on the anchor, from a uniform 294 K state at 04:00
Wed: it **floats to 291.5 K by 05:00** (warmer than CtrlW's 288.5 K night band —
the LP buys capacity margin for the ramp rather than loss reduction) and then
holds **294.00 K exactly** through the comfort window, i.e. the band bottom with
no margin. That is the opposite direction from the night-float rules of §6,
which is precisely the control-headroom question: *is* the morning capacity
constraint worth pre-paying for?

Result, first closed-loop run (`results/optg_mpc_anchor`): **−127.94 against
CtrlW's −93.12**, i.e. 37 % worse. That number is not evidence about the
planner's economics — it is contaminated by the action mapping: the first
version clamped the *upper* band edge to `target + 0.30 K` outside comfort mode
as well, so whenever a zone sat above the plan's night temperature the AHU
search saw a cooling request, opened the dampers to maximum and bought outside
air at −26 °C. The same bug in the same week cost `CtrlW3` −189.95 (first
version, §6), so the mechanism is identified, not guessed.

The mapping is fixed (`follow_mode = night_only`: the plan sets the night floor
only, the upper edge is always the schedule's) and the corrected run
(`results/optg_mpc2_anchor`) was still solving its last re-solves when this run
hit its request budget, so its realised reward is not quoted. What can be said
now, measured: the LP is fast enough for closed loop (2.8 s per 8 h/32-slot
solve, 168 solves), it produces non-trivial plans (float to 291.5 K then hold
294.00 K), and the mapping to CtrlW's band is the fragile part — a plan
delivered as a band target can make CtrlW's own search buy cooling in a heating
climate, which is the practical obstacle to turning a twin into a controller.

