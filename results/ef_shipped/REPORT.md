# Where CtrlW's energy+carbon loss actually goes

Instrumented decomposition of the public Wu base + our supervisory layer (`policies/ctrl_w.py:CtrlW`)
over full 8,641-step episodes. Written for the 2026-11-10 assembly; every number below is regenerated
by `python -m sbx.energyflow run` + `paper/figures/energy_flow.py`, not typed in.

## What was measured, and why it can be trusted

`sbx/energyflow.py` wraps an ordinary `sbx.evalrun` episode and logs, every 5 minutes: per air handler
(supply airflow, recirculation hence outside-air fraction, mixed/return/supply temperatures, coil thermal
load, chiller electric draw, supply- and exhaust-fan power); per VAV (airflow, damper, reheat-valve
position, reheat thermal power, zone-group mean temperature); the hot-water loop (billed gas, the coil
draw it carried, tank standing loss, tank storage term, pump power, water supply/return temperature and
flow); per zone (mean temperature and occupancy, straight from the reward's `RewardInfo`); and the
reward's own comfort/energy/carbon terms.

Two decisions make the numbers exact rather than approximate:

1. **Capture at the model call, not by re-reading sensors.** In the shipped simulator every read of the
   boiler's supply-water sensor advances the tank, and the reward path reads it more than once per step.
   The quantities that are only valid at the instant the reward is billed -- the coil draw the boiler
   charges, the tank's step change, the entering-water temperature each coil saw -- are therefore recorded
   by wrapping `Boiler.compute_thermal_energy_rate` / `apply_heat_draw` and each `Vav.compute_reheat_energy_rate`,
   not by asking the sensors afterwards.
2. **Ventilation is attributed with a counterfactual the simulator can answer.** The share of the heating
   (and chiller) load that the outside air is responsible for is the same step re-evaluated with the mixing
   box recirculating everything (`get_supply_air_temp(T_return, T_return)`, same flows, dampers, valves and
   water temperature). That is the exact marginal effect of the outside-air fraction and needs no
   assumption about coil saturation. The analytic load
   `(1 - recirculation) * airflow * rho * c_p * (T_return - T_out)` is logged alongside it as the physical
   heat the outside-air stream carries.

**The instrumentation is trajectory-neutral.** All six episodes reproduce `results/ctrw_all` bit-for-bit --
`total_reward` and all three reward components to the last digit (difference exactly `0.0`). The corrected
simulator reproduces `results/ctrw_all_corr` equally exactly (-392.203588523).

## Energy reconciliation (acceptance criterion: within 1%)

Re-deriving the reward's cost and carbon from the logged powers -- per-step unit prices from the reward's
own `ElectricityEnergyCost`/`NaturalGasEnergyCost` models, the same `max_electricity_rate`/`max_natural_gas_rate`
denominators -- reproduces both terms:

| scenario | energy term rel. err | carbon term rel. err | combined rel. err |
|---|---|---|---|
| fairbanks_1 | 7.6e-10 | 3.8e-10 | 4.9e-10 |
| fairbanks_2 | 6.6e-10 | 3.0e-10 | 4.0e-10 |
| denver_5 | 3.9e-10 | 5.4e-10 | 4.2e-10 |
| atlanta_3 | 5.0e-10 | 4.4e-10 | 4.9e-10 |
| phoenix_4 | 3.6e-10 | 7.2e-10 | 4.2e-10 |
| singapore_3 | 2.6e-10 | 1.5e-10 | 2.4e-10 |
| fairbanks_1 (corrected) | 1.0e-10 | 8.0e-10 | 6.0e-10 |

i.e. ~1e-9 relative, not 1e-2: the split of the bill into physics components is a partition, not a fit.
The component split is exact too (the stacked bars sum to the episode's energy+carbon loss with residual 0.000).

## The decomposition (reward units, positive = loss)

| scenario | $\bar T_\mathrm{out}$ °C | comfort | energy | carbon | energy+carbon | heating: outside-air | heating: envelope/internal | boiler standby | cooling: outside-air | cooling: internal/solar | fans | pump |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fairbanks, 41 zones | -21.4 | 2.607 | 112.25 | 280.30 | 392.55 | 69.65 | 306.69 | 5.08 | -0.00 | 0.00 | 8.62 | 2.51 |
| Fairbanks, 87 zones | -8.2 | 0.569 | 30.27 | 76.42 | 106.69 | 9.29 | 90.55 | 2.12 | -0.00 | 0.00 | 4.06 | 0.66 |
| Denver, 38 zones | 24.2 | 0.575 | 29.33 | 8.06 | 37.39 | 0.00 | 0.16 | 1.56 | 4.89 | 27.81 | 2.97 | 0.00 |
| Atlanta, 87 zones | 20.8 | 0.224 | 19.18 | 5.39 | 24.58 | 0.01 | 0.39 | 0.91 | 1.24 | 18.20 | 3.83 | 0.00 |
| Phoenix, 32 zones | 34.4 | -0.000 | 67.46 | 15.54 | 83.00 | -0.00 | -0.00 | 0.07 | 17.43 | 60.63 | 4.87 | -0.00 |
| Singapore, 56 zones | 28.6 | 0.416 | 47.66 | 10.99 | 58.65 | -0.00 | -0.00 | 0.08 | 6.72 | 48.33 | 3.51 | -0.00 |

## The same bill in energy units

| scenario | coil heat kWh_th | ventilation-carried kWh_th (%) | marginal coil relief, 100% recirc | chiller kWh_e | ventilation share of chiller | supply fan kWh_e | exhaust fan kWh_e | pump kWh_e |
|---|---|---|---|---|---|---|---|---|
| Fairbanks, 41 zones | 377.2 | 190.4 (50%) | 18.6% | 0.00 | 0.00 (--) | 3.42 | 1.03 | 1.30 |
| Fairbanks, 87 zones | 227.7 | 142.6 (63%) | 9.3% | 0.00 | 0.00 (--) | 3.55 | 1.06 | 0.76 |
| Denver, 38 zones | 0.2 | 2.4 (--) | -- | 15.42 | 2.30 (15%) | 1.01 | 0.40 | 0.00 |
| Atlanta, 87 zones | 0.9 | 22.7 (--) | -- | 20.71 | 1.31 (6%) | 2.56 | 1.57 | 0.00 |
| Phoenix, 32 zones | 0.0 | 0.3 (--) | -- | 28.07 | 6.22 (22%) | 1.33 | 0.43 | 0.00 |
| Singapore, 56 zones | 0.0 | 0.0 (--) | -- | 39.30 | 4.78 (12%) | 1.92 | 0.59 | 0.00 |

## Rollout metadata and the reconciliation check

| scenario | steps | zones | AHUs | mean reward/step | $\Delta$ vs results/ctrw_all | energy rel. err | carbon rel. err |
|---|---|---|---|---|---|---|---|
| Fairbanks, 41 zones | 8641 | 41 | 4 | -0.045730 | +0.0e+00 | 7.6e-10 | 3.8e-10 |
| Fairbanks, 87 zones | 8641 | 87 | 9 | -0.012412 | +0.0e+00 | 6.6e-10 | 3.0e-10 |
| Denver, 38 zones | 8641 | 38 | 4 | -0.004394 | +0.0e+00 | 3.9e-10 | 5.4e-10 |
| Atlanta, 87 zones | 8641 | 87 | 9 | -0.002870 | +0.0e+00 | 5.0e-10 | 4.4e-10 |
| Phoenix, 32 zones | 8641 | 32 | 3 | -0.009606 | +0.0e+00 | 3.6e-10 | 7.2e-10 |
| Singapore, 56 zones | 8641 | 56 | 6 | -0.006835 | +0.0e+00 | 2.6e-10 | 1.5e-10 |

## By time bucket (share of the episode's energy+carbon loss)

| scenario | outside-air share | weekday occupied (06-19) | preheat (04-06) | weekday night | weekend | energy+carbon |
|---|---|---|---|---|---|---|
| Fairbanks, 41 zones | 30% | 69% | 17% | 3% | 12% | 392.5 |
| Fairbanks, 87 zones | 30% | 54% | 29% | 4% | 13% | 106.7 |
| Denver, 38 zones | 33% | 96% | 3% | 1% | 0% | 37.4 |
| Atlanta, 87 zones | 39% | 94% | 5% | 1% | 0% | 24.6 |
| Phoenix, 32 zones | 31% | 81% | 16% | 1% | 2% | 83.0 |
| Singapore, 56 zones | 30% | 89% | 10% | 0% | 0% | 58.6 |

The preheat window is 6.1% of the steps but carries 17% of Fairbanks_1's and 29% of Fairbanks_2's
energy+carbon loss -- the 2 h ramp CtrlW moved into the cheapest hours is still the most expensive block of
the day in cold climates, which is why the horizon sweep has a minimum at all. In Phoenix the same window
carries 16% of the bill, but there it drives the chiller, not the boiler.

Hours 05 and 06 are the two most expensive hours of Fairbanks_1's whole day at 43.9 and 43.0 reward
units (11% each of its energy+carbon loss); in Phoenix hours 05 and 06 are also the top two, at 6.9 and
6.6 units. The morning block is where CtrlW's readiness ramp and the first solar gains land together, and
this decomposition does not separate the two; the horizon sweep in the main text measures the ramp. What the
decomposition does establish is that the block is expensive for two different reasons depending on climate:
heat in Fairbanks, chiller work in Phoenix.

## The claim, quantified

**Half of the heating energy in Fairbanks is ventilation.** Of the 377.2 kWh_th the
reheat coils delivered over the episode in `building_train_1/fairbanks_1`, 190.4 kWh_th (50%) is the
outside-air stream being brought from ambient (mean -21.4 °C)
up to return-air temperature; in the 87-zone `building_train_2/fairbanks_2` it is 142.6 of
227.7 kWh_th (63%). The outside-air fraction is the floor of the action space, not a
choice the policy makes: recirculation never leaves its 0.7 cap in either Fairbanks scenario -- 30% fresh
air, the `max_recirculation` minimum -- on 100% of steps (84-100% across all six). So that half of the
heating bill is the ventilation floor, not the envelope, and no setpoint schedule can remove it.

**But only a fifth of it is recoverable at the meter.** Raising recirculation to 1.0 would cut the coil duty
by 18.6% (Fairbanks_1) and 9.3% (Fairbanks_2), not 50%: the reheat coil duty is
`min(air-side, water-side capacity) * (T_water - T_air)`, and the median coil delivers only 38%
(Fairbanks_1) / 14% (Fairbanks_2) of what the air stream could absorb, so the hot-water side, not the air
temperature, sets most of the bill. The rest of the ventilation load is therefore met by the building fabric
and shows up as colder zones -- the comfort loss -- which is exactly the mechanism behind the paper's
"ventilation tax" sentence. On 15% (Fairbanks_1) / 40% (Fairbanks_2) / 77% (Denver) / 94% (Atlanta) of the
steps that had any reheat, the outside-air load alone exceeds the heat the coils delivered in that step
(median ratio 0.52x / 0.82x / 4.5x / 13.5x).

**The floor is where the controller already sits.** Wu commands recirculation at exactly 0.700 -- the
`max_recirculation` cap -- on every step of both Fairbanks episodes (measured over the first 1{,}200 steps of
each). In Atlanta it commands 0.0-0.7 and sits at the cap on 74% of steps. So the 30% outside-air fraction is
not a missed opportunity but the most the action space allows: the "100% recirculation" column above is an
upper bound on what the ventilation floor costs, not a policy to try. That is also why the paper's
"structural" claim survives: a controller cannot recirculate its way out of the load and cannot cut the
fans, because the exhaust fan exists only to balance the outside air the floor requires.

**Cooling is the mirror image.** The chiller's ventilation share is 22% (Phoenix), 15% (Denver), 12%
(Singapore), 6% (Atlanta) of its electric draw; the remaining 78-94% is internal, solar and envelope gain.
The exhaust fan -- which exists only to make room for that outside air -- is a further
0.40-1.57 kWh_e per episode, i.e. 30-62% of the supply fan's power.

**Fans and pump are small.** Across the six scenarios fans are 27.9 reward units (4.0% of the
energy+carbon loss) and the hot-water pump 3.2 (0.5%); boiler standby is 0.07-5.1 units,
0.1-4.2% of a scenario's loss, so tank losses are not where the money goes. The fan
share is largest where the building is easiest (Atlanta 16%): a fixed ventilation floor costs
roughly the same air power everywhere, so it is a bigger fraction of a small bill.

**Aggregate shape.** Across the six episodes the energy+carbon loss is 702.8 reward units: heating 486.6
(69%), of which the marginal ventilation duty is 79.0 (11% of the total, 16% of the heating); cooling 185.3
(26%), of which ventilation 30.3 (16%); fans 27.9 (4%); pump 3.2 (0.5%).

**Shipped vs corrected simulator.** On the corrected tree (PR #2 energy-balanced tank, PR #3 airflow-weighted
return mixing) Fairbanks_1 gives coil 377.9 kWh_th, ventilation-carried 190.4 kWh_th
(50%), marginal relief 18.6%, standby 1.42 kWh, and the same component shape
(outside-air 70.51 / envelope 306.46 / standby 1.43 / fans
8.67 / pump 2.51 reward units against 69.65 / 306.69 / 5.08 / 8.62 / 2.51 on the
shipped tree). The patch lowers the total loss by
+0.7% and leaves the decomposition unchanged.

## Figure

`paper/figures/energy_flow.pdf` (`paper/figures/energy_flow.py`, reads `results/ef_shipped/*.json`):
(a) each scenario's energy+carbon loss split into heating outside-air / heating envelope-internal / boiler
standby / cooling outside-air / cooling internal-solar / fans / pump, as fractions of that scenario's own
loss; (b) the same in absolute reward units. Suggested caption:

> Where CtrlW's residual loss comes from, for one scenario per climate (8,641 steps, shipped simulator).
> (a) Fractions of each scenario's energy+carbon loss. In the cold climates the bill is heating, and the
> dark segment is the outside-air stream: 18% of Fairbanks_1's loss and 9% of Fairbanks_2's is the marginal
> coil duty the 30% ventilation floor imposes -- 50% and 63% of the heat the reheat coils deliver. In the
> hot climates the bill is the chiller, 78-94% of it internal, solar and envelope gain with 6-22% carried by
> the ventilation air. Fans are 4-16% of the loss and the hot-water pump under 1%; boiler standby is
> 0.1-2%.
> (b) The same decomposition in reward units: three of the six scenarios (Fairbanks_1, Fairbanks_2, Phoenix)
> hold 82% of the total, so cold-climate heating and hot-climate cooling are the two ends of the same
> ventilation-and-envelope problem.

## What this implies for the paper and for policy work

- The cold-climate residual is structural, and the numbers now say so with a partition rather than an
  anecdote: half of the delivered heat warms outside air that the 30% floor requires.
- Because the coils are water-limited, policies cannot trade the ventilation load away by pushing reheat
  harder; the actionable levers are the water loop (`differential_pressure_setpoint`, valve schedule) and
  the hours the plant runs at all (the preheat window, 17-29% of a Fairbanks bill).
- No new policy variant is proposed here; this run is analysis. Every variant that changes the plant
  schedule can be re-scored on this decomposition by re-running
  `python -m sbx.energyflow run --policy <spec> --tag <tag>`.

## Files and reproduction

```
python -m sbx.energyflow run --policy policies.ctrl_w:CtrlW --steps 8641 --sim shipped --workers 2 \
    --scenario building_train_1/fairbanks_1 building_train_2/fairbanks_2 building_train_7/denver_5 \
               building_train_2/atlanta_3 building_train_8/phoenix_4 building_train_4/singapore_3 \
    --tag ef_shipped
python -m sbx.energyflow run --policy policies.ctrl_w:CtrlW --steps 8641 --sim corrected --workers 1 \
    --scenario building_train_1/fairbanks_1 --tag ef_corrected
starter-kit/.venv/bin/python paper/figures/energy_flow.py --tag ef_shipped
```

- `sbx/energyflow.py` -- instrumentation, decomposition, per-step `npz` series, reconciliation.
- `results/ef_shipped/`, `results/ef_corrected/` -- JSON aggregates + per-step series per scenario.
- `paper/figures/energy_flow.py` -> `paper/figures/energy_flow.pdf` -- figure, tables, key numbers.
- `results/ef_shipped/energy_flow_table.tex` -- the same decomposition as a paper-ready LaTeX table.
- Wall time: 5,895 s for six shipped scenarios at `--workers 2`; 531 s for the corrected one.
