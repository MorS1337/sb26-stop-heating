# Value of oracle information for CtrlW: zonal occupancy vs a short weather forecast

Analysis-only study (2026-11-05/06). Baseline: `policies.ctrl_w.py:CtrlW` = the public Wu adaptive-physics controller + our supervisory layer, 2 h preheat, dev30/shipped score 0.0073694. The occupancy and forecast channels are **never submitted**: they read signals the competition does not expose. One configuration found along the way -- a fixed 2 K pre-dawn lift that needs no oracle -- is legal, and is called out in sections 4-6.

## Measurement setup

- 10-scenario subset of dev30, 2 per climate station, 30–39 zones, 3–9 AHUs: `building_train_1/atlanta_2`, `building_train_7/atlanta_8`, `building_train_2/denver_8`, `building_train_8/denver_6`, `building_train_1/fairbanks_1`, `building_train_8/fairbanks_8`, `building_train_1/phoenix_5`, `building_train_7/phoenix_3`, `building_train_1/singapore_8`, `building_train_8/singapore_7`.
- Full 8641-step episodes, `--sim shipped`, `--components hook`, `--occupancy-seed` left at the config default (17321).
- Subset baseline (CtrlW, `results/ctrw_all`): **0.0119199**. The subset over-represents Fairbanks (the worst station), so its level is not the dev30 level (0.0073694); only deltas are meaningful.
- Hosts: `voi_occ_dev10_pc`, `voi_fcst_dev10_pc` ran on a second machine (Ryzen 5 5600X under WSL2, 4 workers); `ctrw_all`, `voi_occ_mild`, `voi_occ_morning`, `voi_fcst_night_lift`, `voi_night_fixed` and the no-op control on the Apple M3 (2 workers). **Cross-host noise floor**: the same forecast policy run on both machines (`voi_fcst_dev10` vs `voi_fcst_dev10_pc`, same params) differs by up to 6.2e-6 relative per scenario (`building_train_7/phoenix_3`) and 4e-9 on the 10-scenario mean, because a handful of discrete damper decisions flip under different float arithmetic. Any oracle delta below ~1e-5 of the total reward is therefore not separable from host drift, and every ablation below is baselined against a run from the same host.

## 1. The runner channel (`sbx/evalrun.py --oracle`)

`--oracle occupancy|forecast|occupancy,forecast` (+`--oracle-horizon`, default 12 h) calls `policy.observe_oracle(info)` before every `get_action`:

- `zone_occupancy` — per-zone `average_occupancy` of the **last completed** interval, read from the `RewardInfo` the step's own reward computation already built (`_StepRecorder.last_info`). No second occupancy-model query: a second query would advance the shared `RandomState` and move every later arrival (forum digest B3). It is therefore a realistic 5-min-late zonal sensor.
- `forecast_oat` — outside air temperature in K for the next 1..H hours from the episode's own `ReplayWeatherController`, i.e. the same interpolation that produced the ambient used by the step, evaluated at future clocks.
- Zone order follows `policy.zone_ids`, mapped from `room_<n>` reward-info keys by numeric suffix (the `room_x -> VAV_x` correspondence `wu_core` already assumes).

### Non-perturbation proof

A policy that receives the payload and drops it (`policies/oracle_noop.py:OracleNoopCtrlW`) must reproduce a normal run exactly. 2 scenarios x 576 steps, shipped sim, hook components (`results/np_proof_plain` vs `results/np_proof_oracle`):

| scenario | arms | total reward (plain) | total reward (oracle) | identical |
|---|---|---|---|---|
| `building_train_1/fairbanks_1` | 576 | `-6.9648686744622` | `-6.9648686744622` | yes (bit-identical) |
| `building_train_8/singapore_7` | 576 | `-2.8305741771856` | `-2.8305741771856` | yes (bit-identical) |

Beyond the totals, `sum_comfort` / `sum_energy` / `sum_carbon`, every `by_hour` array and every comfort statistic are bitwise equal in both scenarios (only the metadata and timing fields differ), so the channel perturbs neither the occupancy draw nor the simulator clock.
### Policy-level no-op control

`OracleFcstCtrlW` with *both* mechanisms switched off (`fcst_preheat_enabled=false`, `fcst_night_enabled=false`) receives the 12 h forecast, computes the arrival demand from it and then does nothing with it. Full 8641-step episodes against CtrlW on the same host (`results/voi_fcst_noop` vs `results/ctrw_all`):

| scenario | CtrlW total reward | no-op policy total reward | identical |
|---|---|---|---|
| `building_train_1/fairbanks_1` | `-395.1538459855292` | `-395.1538459855292` | yes (bit-identical) |
| `building_train_7/phoenix_3` | `-47.1307097942185` | `-47.1307097942185` | yes (bit-identical) |

This check exists because the first night-lift ablation was configured with `fcst_preheat_enabled=false` alone, and in the original `_band` that returned before `_night_lift` was reached: the run (`results/voi_fcst_night`, on the PC) was a *semantic* no-op whose per-step deltas (<=1.1e-6) are pure cross-host drift. `policies/oracle_forecast.py` now runs the night lift independently of the horizon mechanism; the ablation below is the rerun, on the baseline's own host.
### Forecast alignment

One 400-step Atlanta episode (`scratch/VoIOracle/align_check.py`): the observed AHU outside-air sensor equals the weather controller at the simulator's own timestamp, and the forecast at +h h equals the sensor observed h h later, for every h tested:
| lag | max abs difference | pairs |
|---|---|---|
| 0 (sensor vs clock) | 1.42e-05 K | 400 |
| +1 h | 1.42e-05 K | 388 |
| +4 h | 1.42e-05 K | 352 |
| +12 h | 1.42e-05 K | 256 |

1.4e-5 K is the float32 quantisation of the observation, i.e. the forecast is the realised weather exactly -- it is perfect foresight of outside temperature, not a noisy NWP product. Any gain measured below is therefore an **upper bound** on what a real forecast is worth.

## 2. OracleOccCtrlW — per-zone occupancy sensing

`policies/oracle_occupancy.py:OracleOccCtrlW(CtrlW)`. CtrlW's own release can only fire when the *whole* building is empty (`num_occupants == 0` after 12:00): one person anywhere keeps every zone at the comfort band. The reward weights productivity regret by each zone's own `average_occupancy`, so conditioning an empty zone is pure energy cost. With the zonal channel, a zone that has been empty for `occ_release_empty_steps = 3` (15 min) after having held someone earlier that day is dropped to Wu's standby band (282–307 K) until the next comfort-mode start; zones with anyone in them keep the comfort band, and after noon the drop is final because this occupancy model's arrival window has closed (`ZoneOccupant` can only re-enter while `hour <= latest_arrival_hour = 12`).

Mechanism check on `building_train_1/atlanta_2` (3 days, `scratch/VoIOracle/occ_debug.py`) — released zones out of 39, mean/max per hour: 13:00 0.9/2, 14:00 6.2/12, 15:00 13.2/14, 16:00 17.5/19, 17:00 22.8/27, 18:00 27.5/29, 19:00+ 37.7→39/39, i.e. the release tracks each zone's own departures instead of waiting for the building to empty; weekends stay at 0.

Two more knobs were swept on the same subset. `occ_release_low` raises the floor the released zones are held at (282 K Wu standby vs 289 K), i.e. how much of the afternoon saving to take in energy; `occ_morning_enabled` relaxes the band by `occ_morning_relax_K = 1 K` in zones nobody has entered yet that day, which is the asymmetric bet on the arrival window: the model gives every zone a 06-12 h arrival window, so a zone that stays empty until 11:00 was conditioned for nothing. It is off by default because the reward averages comfort over *present* occupants: the first arrival in a relaxed zone pays the full shortfall, so one person is enough to wipe out the saving in that zone.

## 3. Value of information — 10-scenario subset, full 8641-step episodes

| scenario | zones | CtrlW | occ (standby 282 K) | occ (standby 289 K) | occ + morning relax | forecast (both mechanisms) | forecast night lift only | fixed 2 K night lift (no oracle) |
|---|---|---|---|---|---|---|---|---|
| `building_train_1/atlanta_2` | 39 | -0.0032093 | -0.0030766 | -0.0030766 | -0.0033564 | -0.0032255 | -0.0032323 | -0.0032323 |
| `building_train_7/atlanta_8` | 36 | -0.0070463 | -0.0067075 | -0.0067107 | -0.0073243 | -0.0069937 | -0.0069671 | -0.0069781 |
| `building_train_2/denver_8` | 85 | -0.0043695 | -0.0041783 | -0.0041812 | -0.0045465 | -0.0043147 | -0.0042967 | -0.0042919 |
| `building_train_8/denver_6` | 30 | -0.0032313 | -0.0029772 | -0.0029772 | -0.0029931 | -0.0032544 | -0.0032317 | -0.0032317 |
| `building_train_1/fairbanks_1` | 39 | -0.0457301 | -0.0456687 | -0.0455030 | -0.0460306 | -0.0455070 | -0.0455049 | -0.0455049 |
| `building_train_8/fairbanks_8` | 30 | -0.0241661 | -0.0236726 | -0.0237283 | -0.0242512 | -0.0239095 | -0.0239062 | -0.0239121 |
| `building_train_1/phoenix_5` | 39 | -0.0142396 | -0.0137319 | -0.0137319 | -0.0137319 | -0.0142413 | -0.0142396 | -0.0142396 |
| `building_train_7/phoenix_3` | 36 | -0.0054543 | -0.0049132 | -0.0049131 | -0.0049131 | -0.0054679 | -0.0054543 | -0.0054543 |
| `building_train_1/singapore_8` | 39 | -0.0064105 | -0.0063003 | -0.0063003 | -0.0063003 | -0.0063761 | -0.0064105 | -0.0064105 |
| `building_train_8/singapore_7` | 30 | -0.0053421 | -0.0052292 | -0.0052292 | -0.0052292 | -0.0053126 | -0.0053421 | -0.0053421 |

| variant | subset score | delta | delta % | better on | sum comfort | sum energy | sum carbon |
|---|---|---|---|---|---|---|---|
| CtrlW | 0.0119199 | — | — | — | -5.331 | -449.972 | -574.696 |
| occ (standby 282 K) | 0.0116456 | -0.0002744 | -2.30% | 10/10 | -5.493 | -437.672 | -563.128 |
| occ (standby 289 K) | 0.0116352 | -0.0002847 | -2.39% | 10/10 | -5.239 | -437.307 | -562.848 |
| occ + morning relax | 0.0118677 | -0.0000523 | -0.44% | 5/10 | -39.636 | -433.128 | -552.720 |
| forecast (both mechanisms) | 0.0118603 | -0.0000596 | -0.50% | 6/10 | -9.362 | -446.902 | -568.581 |
| forecast night lift only | 0.0118585 | -0.0000614 | -0.51% | 4/10 | -7.004 | -447.739 | -569.953 |
| fixed 2 K night lift (no oracle) | 0.0118597 | -0.0000602 | -0.50% | 4/10 | -7.003 | -447.774 | -570.024 |

Comfort is negative regret, energy and carbon are negative costs, so *less negative is better*: a **negative energy/carbon delta is extra consumption**, a **positive comfort delta is extra regret**. All three columns are sums over the 10 scenarios.

### Where the occupancy gain comes from

Sum over the 10 scenarios, variant minus CtrlW, by local hour:

| window | comfort | energy | carbon | reading |
|---|---|---|---|---|
| 00–03 | +0.000 | -1.296 | -2.965 | released mass is still cold at dawn: extra reheat |
| 04–05 | +0.000 | -6.724 | -6.382 | preheat ramp has more lift to do |
| 06–09 | -0.561 | -9.533 | -11.290 | arrival: the comfortable zones come back |
| 10–12 | +0.027 | -3.423 | -3.861 | cold mass still being recovered |
| 13–19 | +0.372 | +33.813 | +37.144 | the release window itself: empty zones at standby |
| 20–23 | +0.000 | -0.537 | -1.077 | both variants in standby |

Per-scenario deltas (total reward is negative; a positive delta is an improvement; comfort is negative regret):

| scenario | Δ total reward | Δ comfort | Δ energy | Δ carbon | Δ comfort 06–09 |
|---|---|---|---|---|---|
| `building_train_1/atlanta_2` | +1.146 | +0.057 | +0.620 | +0.469 | +0.004 |
| `building_train_7/atlanta_8` | +2.927 | -0.003 | +0.777 | +2.154 | -0.002 |
| `building_train_2/denver_8` | +1.652 | +0.012 | +0.450 | +1.190 | +0.011 |
| `building_train_8/denver_6` | +2.195 | +0.200 | +1.603 | +0.392 | +0.005 |
| `building_train_1/fairbanks_1` | +0.530 | -0.605 | -0.636 | +1.771 | -0.602 |
| `building_train_8/fairbanks_8` | +4.265 | +0.019 | +0.682 | +3.563 | +0.021 |
| `building_train_1/phoenix_5` | +4.387 | +0.060 | +3.522 | +0.805 | +0.000 |
| `building_train_7/phoenix_3` | +4.676 | +0.057 | +3.758 | +0.861 | +0.001 |
| `building_train_1/singapore_8` | +0.952 | +0.026 | +0.748 | +0.177 | +0.000 |
| `building_train_8/singapore_7` | +0.976 | +0.013 | +0.776 | +0.187 | +0.000 |

The only scenario with a material comfort loss is `building_train_1/fairbanks_1` (-0.605, essentially all of it in 06–09 h): the zone that was released the previous afternoon is cold when its first occupant arrives, and one arrival is enough because the reward averages over present occupants only. Every other scenario gains a little comfort (less overheating in zones that sit near the standby edge).

Per-station view (mean per scenario, same sign convention):

| station | scenarios | Δ total reward | Δ comfort | Δ energy | Δ carbon |
|---|---|---|---|---|---|
| atlanta | 2 | +2.037 | +0.027 | +0.698 | +1.311 |
| denver | 2 | +1.924 | +0.106 | +1.027 | +0.791 |
| fairbanks | 2 | +2.397 | -0.293 | +0.023 | +2.667 |
| phoenix | 2 | +4.531 | +0.059 | +3.640 | +0.833 |
| singapore | 2 | +0.964 | +0.020 | +0.762 | +0.182 |

### Where the forecast gain comes from

| window | combined: comfort / energy / carbon | night lift (forecast): comfort / energy / carbon | fixed 2 K lift (no forecast): comfort / energy / carbon |
|---|---|---|---|
| 02–03 | +0.000 / -3.452 / -8.813 | +0.000 / -3.451 / -8.812 | +0.000 / -3.596 / -9.162 |
| 04–05 | +0.000 / +15.419 / +19.401 | +0.000 / +5.593 / +13.814 | +0.000 / +5.678 / +14.038 |
| 06 | -4.195 / -2.314 / -1.334 | -1.699 / +0.002 / -0.180 | -1.699 / +0.010 / -0.165 |
| 07–18 | +0.164 / -6.595 / -3.194 | +0.026 / +0.071 / -0.122 | +0.027 / +0.085 / -0.093 |
| 19–23 | +0.000 / +0.006 / +0.040 | +0.000 / +0.011 / +0.027 | +0.000 / +0.016 / +0.039 |

Sign convention as above: positive energy = less energy used, positive comfort = more regret. The pre-dawn lift (02–03 h) *spends* 3.5 energy units and *saves* 8.8 carbon units (the price model charges a unit of gas at 0.4734 weighted units at 02:00 vs 0.4447 at 15:00, `ELEC_REL * GAS_RATIO` in `policies/ctrl_w.py`), and the ramp then needs less heat (04–05 h).

### Decomposing the forecast: information vs a fixed pre-dawn lift

| mechanism | subset score | delta | delta % | better on | lift scaled by |
|---|---|---|---|---|---|
| horizon + night lift (combined) | 0.0118603 | -0.0000596 | -0.50% | 6/10 | forecast demand |
| night lift only | 0.0118585 | -0.0000614 | -0.51% | 4/10 | forecast demand |
| fixed 2 K lift, **no forecast** (legal) | 0.0118597 | -0.0000602 | -0.50% | 4/10 | always |
| horizon mechanism alone (combined − night lift) | 0.0119216 | +0.0000017 | +0.01% | — | forecast demand |

The difference between the fixed lift and the forecast-driven lift is the value of the *information* as opposed to the mechanism: **+0.0000012 score (+0.010%)**. That is 300x the subset noise floor but the same order as the per-scenario cross-host drift, and it is the wrong sign for the forecast: the demand-scaled lift is worth ~nothing over a constant one. Note *why* the fixed lift does not misfire on the warm stations: the mechanism is gated on the zone already sitting at the standby floor (`low <= standby_minimum + 1 K`) and on a comfort day coming, not on the forecast, so Phoenix and Singapore are bit-identical to CtrlW in both runs. The information the forecast supplies is therefore worth 0.01% -- it is a redundant way of expressing a rule the controller can state itself.

## 4. Decomposition of the two information sources

| source | mechanism | subset score | subset delta | delta % |
|---|---|---|---|---|
| zonal occupancy | release empty zones to standby (floor 282 K) | 0.0116456 | -0.0002744 | -2.30% |
| zonal occupancy | release empty zones to standby (floor 289 K) | 0.0116352 | -0.0002847 | -2.39% |
| zonal occupancy | release empty zones to standby (floor 282 K) + 1 K morning relax on never-occupied zones | 0.0118677 | -0.0000523 | -0.44% |
| weather forecast | preheat horizon 1-2 h from the arrival-window demand + night lift | 0.0118603 | -0.0000596 | -0.50% |
| weather forecast | night lift only (preheat horizon fixed at 2 h) | 0.0118585 | -0.0000614 | -0.51% |
| *(no information)* | fixed 2 K pre-dawn lift, no oracle at all | 0.0118597 | -0.0000602 | -0.50% |

Ratio of the best occupancy variant to the best forecast variant: 4.6x in score terms.

### Why a forecast can only shorten the ramp: the dev30 horizon sweep

CtrlW's preheat horizon is a scalar that was tuned once; the dev30 sweep (`results/refwu_dev30_shipped` = Wu's 1 h, `ctrw_d2`/`ctrw_d3`/`ctrw_d4` = 2/3/4 h, 30 scenarios each) is flat-topped at 2 h:

| horizon | dev30 score |
|---|---|
| 1 h (Wu) | 0.0075040 |
| 2 h | 0.0074263 |
| 3 h | 0.0074906 |
| 4 h | 0.0075614 |

Per scenario, 2 h beats the other three horizons on 14/30 (<=), and strictly beats 1 h on 23/30, 3 h on 21/30, 4 h on 27/30. The response is flat around the optimum, so a forecast can only move the horizon towards 1 h on mild mornings -- the one direction the sweep says is a wash -- which is why the horizon mechanism measures +0.01% below.

## 5. Interpretation

- **Zonal occupancy is the information that pays.** Releasing each empty zone to standby the moment its own occupants have gone is worth -2.30% on the subset and improves **all 10 of 10** scenarios (Δ total reward +0.53 … +4.68). Net energy+carbon recovered is +23.9 reward units for 0.16 of comfort regret (comfort sum -5.331 → -5.493), and that regret is concentrated in one building (`building_train_1/fairbanks_1`). It is also the *deployable* half: real buildings have zonal occupancy (CO2, badge, PIR) even though this observation space only exposes a building total. The channel does not replay the occupancy model from its fixed seed (forum B7); it reads the realised occupancy with one interval of lag, which is exactly what such a sensor delivers.
- **The reward is per-zone, the observation is per-building.** Comfort regret is weighted by each zone's own `average_occupancy`, so an empty zone is nearly free to drift -- yet a legal CtrlW cannot tell which zones are empty, and its release must wait for `num_occupants == 0`. That mismatch, not model error, is the gap the occupancy channel closes.
- **About two thirds of the afternoon saving is paid back elsewhere in the day.** The release window 13–19 h saves +33.8 energy units on the subset, while 00–12 h spends 21.0 extra: the zone mass that was allowed to go cold has to be brought back, and it lands in the dawn preheat and the arrival ramp, not only at 06:00. A milder release floor does not fix it either (see the 289 K row): it barely moves the subset score and shifts the trade around inside the two Fairbanks runs. The trade is a property of the building mass, not of the release setpoint -- but it is a *profitable* trade, because the recovery happens where the marginal price of heat is lower than the standby saving.
- **The forecast's headline -0.5% is really a legal fixed rule.** Perfect 12 h foresight, wired to both mechanisms, is worth -0.50% on the subset (6/10 scenarios) -- an order of magnitude less than zonal occupancy, and almost none of it is information. Decomposed, the night setback lift carries all of it (-0.51%) and the arrival-demand preheat horizon contributes +0.015% -- the horizon is flat-topped at 2 h on dev30, so there is nothing to win there. The lift itself is not really a forecast mechanism: a *fixed* 2 K lift that reads no forecast at all scores -0.50% (4/10), i.e. +0.01% relative to the forecast-driven lift. The forecast's own contribution is negligible because the lift is gated by the *plant* state, not by the weather: it only fires where a zone already sits at the standby floor before the comfort window, which never happens on Singapore/Phoenix (both runs bit-identical to CtrlW) and saturates on Fairbanks, where `demand` is far above the 30 K scale. A noisy NWP product would therefore cost nothing here; an exact one buys nothing.
- **Relaxing the unfilled morning band is the wrong bet.** Stacking `occ_morning_relax_K = 1 K` on the release variant moves the score by -0.44% -- comparable to the forecast, a fifth of the release -- while comfort regret explodes from -5.33 to -39.64 (-34.3): the energy side is happy (energy+carbon +38.8) but the reward now pays 34.3 units of productivity regret for it, and only 5/10 scenarios improve (Δ total reward -2.60 … +4.68). The asymmetry documented in `oracle_occupancy.py` is real: a zone that nobody has entered yet may stay empty all day (free to relax), or may receive its first occupant at 06:30 (full shortfall at full weight). With the model's 06-12 h arrival window the second case is common enough that the bet loses. This is the clearest negative result in the study: *knowing who is present* is valuable, *guessing about who has not arrived yet* is not.
- **The one actionable result is legal.** The pre-dawn lift that the forecast variant switches on is worth about half a percent on its own and does not need the forecast: a fixed 2 K lift, gated only on `hour`, the comfort-day predicate and the zone sitting at the standby floor, reproduces it (-0.50% vs -0.51%, +0.010% apart). That is a CtrlW variant the observation space allows, and it should go to dev30 next to the occupancy study. The caveat is the comfort bill: the lift costs 1.67 reward units of regret (vs 0.16 for the zonal release), it does nothing on the cooling-dominated stations (Phoenix/Singapore runs are bit-identical to CtrlW), and half of the subset gain is the two Fairbanks scenarios, so the subset number is not a dev30 prediction.
- **For the paper**: quoting the occupancy channel as the value of sensing (an upper bound on a zonal-sensor retrofit: -2.39% score with comfort regret -0.162 on the subset), the forecast channel as an upper bound that collapses onto a fixed rule, and the fixed lift as the deployable part of it frames why the competition's observation space, not the controller, is the limiting factor for a transferable controller.

## 6. What to run next (orchestrator)

Which subset variants deserve a full dev30 run, in priority order:

| variant | full dev30? | why |
|---|---|---|
| occ (standby 282 K) (-2.30%, 10/10) | **yes, shipped + corrected** | the only source with a large, uniform effect; it is the paper's VOI number |
| occ + morning relax (-0.44%, 5/10) | no | the 1 K morning relax buys -0.44% of score but costs 34.3 reward units of comfort regret (5/10) -- the first-arrival effect dominates |
| occ (standby 289 K) (-2.39%, 10/10) | no | the 289 K floor sits within 1e-5 of the 282 K variant on the subset |
| forecast (both mechanisms) (-0.50%, 6/10) | one run, for the paper's decomposition | confirms that the -0.5% survives dev30, and gives the upper bound for an exact forecast |
| forecast night lift only (-0.51%, 4/10) | no | the fixed-lift control below shows the forecast adds nothing over the mechanism |
| fixed 2 K night lift (no oracle) (-0.50%, 4/10) | **yes — legal candidate for the submission** | -0.50% on the subset (4/10) with the oracle channel switched off entirely; needs dev30 + corrected before it goes in |

The occupancy and forecast channels are analysis-only and belong in the paper. The fixed night lift is **not**: it reads no oracle, so it is a legal CtrlW variant and should be validated before the next submission:

```bash
# value of zonal occupancy sensing, full dev30 (paper Table: VOI)
starter-kit/.venv/bin/python -m sbx.evalrun \
    --policy policies.oracle_occupancy:OracleOccCtrlW --oracle occupancy \
    --set dev30 --steps 8641 --sim shipped --workers 6 --components hook \
    --tag voi_occ_dev30_shipped

# same on the corrected sim (paper reports both)
... --sim corrected --tag voi_occ_dev30_corrected

# value of a perfect 12 h weather forecast, full dev30
starter-kit/.venv/bin/python -m sbx.evalrun \
    --policy policies.oracle_forecast:OracleFcstCtrlW --oracle forecast \
    --set dev30 --steps 8641 --sim shipped --workers 6 --components hook \
    --tag voi_fcst_dev30_shipped

# LEGAL: fixed 2 K pre-dawn lift, no oracle at all (candidate for submission)
starter-kit/.venv/bin/python -m sbx.evalrun \
    --policy policies.oracle_forecast:OracleFcstCtrlW \
    --params '{"fcst_preheat_enabled": false, "fcst_night_fixed_share": 1.0}' \
    --set dev30 --steps 8641 --sim shipped --workers 6 --components hook \
    --tag nightlift_dev30_shipped
```

Do **not** submit a run that uses `--oracle`: the observation space provides neither per-zone occupancy nor a forecast. The fixed-lift configuration above is the one exception -- it is the same class with the oracle mechanisms switched off, and its summary records `oracle: []`.

## 7. Data provenance

Every number above comes from one of these directories (all `--sim shipped`, 8641 steps, `--components hook`). `ctrw_all` is the full dev30 baseline; every subset table uses only the 10 scenarios listed under *Measurement setup*.

| tag | results dir | policy | params | oracle | host | scenarios |
|---|---|---|---|---|---|---|
| `ctrw_all` | `results/ctrw_all/` | `policies.ctrl_w:CtrlW` | — | — | Mac | 30 |
| `np_proof_plain` | `results/np_proof_plain/` | `policies.ctrl_w:CtrlW` | — | — | Mac | 2 |
| `np_proof_oracle` | `results/np_proof_oracle/` | `policies.oracle_noop:OracleNoopCtrlW` | — | occupancy,forecast | Mac | 2 |
| `voi_fcst_noop` | `results/voi_fcst_noop/` | `policies.oracle_forecast:OracleFcstCtrlW` | fcst_night_enabled=False, fcst_preheat_enabled=False | forecast | Mac | 2 |
| `voi_fcst_dev10` | `results/voi_fcst_dev10/` | `policies.oracle_forecast:OracleFcstCtrlW` | — | forecast | Mac | 10 |
| `voi_occ_dev10_pc` | `results/voi_occ_dev10_pc/` | `policies.oracle_occupancy:OracleOccCtrlW` | — | occupancy | PC | 10 |
| `voi_occ_mild` | `results/voi_occ_mild/` | `policies.oracle_occupancy:OracleOccCtrlW` | occ_release_low=289.0 | occupancy | Mac | 10 |
| `voi_occ_morning` | `results/voi_occ_morning/` | `policies.oracle_occupancy:OracleOccCtrlW` | occ_morning_enabled=True | occupancy | Mac | 10 |
| `voi_fcst_dev10_pc` | `results/voi_fcst_dev10_pc/` | `policies.oracle_forecast:OracleFcstCtrlW` | — | forecast | PC | 10 |
| `voi_fcst_night_lift` | `results/voi_fcst_night_lift/` | `policies.oracle_forecast:OracleFcstCtrlW` | fcst_preheat_enabled=False | forecast | Mac | 10 |
| `voi_night_fixed` | `results/voi_night_fixed/` | `policies.oracle_forecast:OracleFcstCtrlW` | fcst_night_fixed_share=1.0, fcst_preheat_enabled=False | — | Mac | 10 |

## 8. Files

- `sbx/evalrun.py` — `--oracle` / `--oracle-horizon`, `run(oracle=...)`, `_OracleChannel`; `_StepRecorder.last_info` keeps the step's `RewardInfo`. Additive and opt-in: default `oracle=None` reproduces every earlier run bit-for-bit (proved above).
- `sbx/README.md` — the flag documented next to `--workers`.
- `policies/oracle_occupancy.py:OracleOccCtrlW`, `policies/oracle_forecast:OracleFcstCtrlW`, `policies/oracle_noop.py:OracleNoopCtrlW`, `policies/oracle_util.py`.
- Analysis scripts: `scratch/VoIOracle/{align_check,occ_debug,report_table,hour_delta,compose_report}.py` and `scratch/VoIOracle/queue_after.sh`.
- This report is generated, not typed: `starter-kit/.venv/bin/python scratch/VoIOracle/compose_report.py voi_occ_dev10_pc voi_occ_mild voi_occ_morning voi_fcst_dev10_pc voi_fcst_night_lift voi_night_fixed > docs/voi_oracle.md`.


---

# 9. Appendix — perfect occupancy foresight: is the public anchor gap information?

Anchor: `building_train_1/fairbanks_1`, default start, **2016 steps** (7 days, Fairbanks winter), `--sim shipped`, `--components hook`, `--fast` (bit-identical to plain runs). CtrlW total reward **-93.1229** (the orchestrator's −93.12). Reference points on the same episode: OptGap's receding-horizon LP on a 79-node twin −100.20, MOBI-EVERGi's public score −81.9, Houssem's twin+LP −85.4, a free-action LP bound −79.1.

## 9.1 The channel: recorded schedules, played back

`--oracle foresight --oracle-file <dir|json>` hands the policy a dense `(steps x zones)` matrix of a **reference run's** per-zone `average_occupancy`, recorded with `--oracle-record <dir>` (sparse JSON: per step, the zone indices with anyone present; 2016 steps x 39 zones is 70 KB). Two properties make it a valid oracle:

1. **Occupancy is exogenous.** The same episode recorded with two completely different controllers -- CtrlW and `baseline:all_off`, total rewards −93.12 vs −3427.76 -- gives sparsely identical schedules (2016 rows, 39 zones, 16813 occupied zone-steps, `results/foi_anchor_ref` vs `results/foi_anchor_ref2`; the sparse schedules themselves are in `scratch/VoIOracle/foresight_ref{,2}`). The occupancy draw does not depend on the actions.
2. **The channel does not perturb.** `policies/oracle_noop.py:OracleNoopCtrlW` receives the foresight payload and drops it: total reward `-93.1229093824804` vs CtrlW `-93.1229093824804` -- bit-identical (`results/foi_anchor_noop`). The payload is read from the reward info the step already computed, so no second occupancy-model query can advance the shared `RandomState` (forum B3).

## 9.2 The policy

`policies/oracle_foresight.py:OracleForesightCtrlW(CtrlW)` replaces CtrlW's building-wide schedule with per-zone windows computed from the recorded day: a zone with no arrival today is held at the standby band all day; a zone with arrivals gets the comfort band from its own first arrival to its own last departure, with Wu's eco -> comfort ramp over `foresight_preheat_lead_hours` before that arrival; after the last departure it returns to standby. If the plan and the public `num_occupants` disagree, the comfort band is restored for the whole building (`foresight_safety`), so a misaligned recording degrades to CtrlW rather than to an unconditioned building.

## 9.3 Anchor: perfect foresight, best case

| preheat lead | total reward | Δ vs CtrlW | Δ % | Δ comfort | Δ energy | Δ carbon |
|---|---|---|---|---|---|---|
| CtrlW | -93.1229 | — | — | — | — | — |
| 2 h | -92.6800 | +0.4429 | +0.48% | -1.7257 | +0.3145 | +1.8542 |
| 3 h | -91.1868 | +1.9361 | +2.08% | -0.2261 | +0.3347 | +1.8275 |
| 4 h | -90.6468 | +2.4761 | +2.66% | +0.2565 | +0.3876 | +1.8320 |
| 5 h | -90.5225 | +2.6004 | +2.79% | +0.4025 | +0.4247 | +1.7732 |
| 6 h | -90.5541 | +2.5688 | +2.76% | +0.4840 | +0.4243 | +1.6606 |
| 8 h | -90.9027 | +2.2202 | +2.38% | +0.5131 | +0.3310 | +1.3761 |

Best case: **5 h lead**, -90.5225 (+2.79%): perfect per-zone occupancy foresight plus per-zone windows buys +2.60 reward on the anchor. The public −81.9 is 11.22 reward *better* than CtrlW -- **4.3x** larger than what information alone gives on the same 2016 steps.

The hour split explains the ceiling. On this heating-dominated week the foresight policy mostly trades energy between hours: it spends more in the pre-dawn ramp (04-05 h) and the evening (16-18 h) and saves during the occupied midday (07-13 h), buying comfort with energy. It cannot create a saving that the plant does not have, because a Fairbanks zone that is empty still needs its mass kept above the freeze/recovery floor and the AHU still runs for the occupied ones.

The day window is cut at the next local midnight (`_day_start_hour` is set from the clock at the wrap, so an episode that starts at 06:00 has an 18 h first day); re-running leads 2 h and 5 h with that cut gives bit-identical totals (`results/foi_anchor_check2/5`), because the occupancy model only places arrivals in 06:00-12:00, so a window that spills an hour into the next day contains no arrival to mis-schedule.

## 9.4 Subset: the same oracle over 10 scenarios, 8641 steps

| | subset score | delta | delta % | sum comfort | sum energy | sum carbon |
|---|---|---|---|---|---|---|
| CtrlW (`--fast`) | 0.0119199 | — | — | -5.331 | -449.972 | -574.696 |
| perfect foresight | 0.0112001 | -0.0007198 | -6.04% | -4.387 | -423.412 | -540.001 |

| scenario | CtrlW | perfect foresight | Δ total reward | Δ comfort |
|---|---|---|---|---|
| `building_train_1/atlanta_2` | -0.0032093 | -0.0029318 | +2.398 | +0.379 |
| `building_train_7/atlanta_8` | -0.0070463 | -0.0058282 | +10.526 | -1.372 |
| `building_train_2/denver_8` | -0.0043695 | -0.0037266 | +5.555 | +0.242 |
| `building_train_8/denver_6` | -0.0032313 | -0.0029041 | +2.828 | +0.120 |
| `building_train_1/fairbanks_1` | -0.0457301 | -0.0440257 | +14.728 | +1.974 |
| `building_train_8/fairbanks_8` | -0.0241661 | -0.0229249 | +10.725 | +0.241 |
| `building_train_1/phoenix_5` | -0.0142396 | -0.0137018 | +4.646 | -0.520 |
| `building_train_7/phoenix_3` | -0.0054543 | -0.0046704 | +6.774 | +0.087 |
| `building_train_1/singapore_8` | -0.0064105 | -0.0060569 | +3.055 | -0.219 |
| `building_train_8/singapore_7` | -0.0053421 | -0.0052304 | +0.965 | +0.013 |

Perfect foresight wins on 10/10 scenarios and moves the subset score by -6.04%, against −2.30% for the 5-minute-late zonal sensor (section 3) and −0.50% for the exact weather forecast (section 4): the schedule is worth 2.6x the sensor, so the 5-minute lag plus the absence of a calendar costs 3.7 points of score on this subset. It is still single-digit percent, which is what matters for the anchor question below.

## 9.5 Answer to the question

**No: perfect occupancy foresight does not explain the public anchor gap.** Held against the same controller, the strongest form of the occupancy information -- the realised per-zone schedule of the actual episode, with no sensor lag at all -- is worth +2.79% on the anchor (5 h lead, the best of six) and −6.04% on the 10-scenario subset. The public MOBI-EVERGi run is 12.1% better than CtrlW on the same 2016 steps (−81.9), while an LP on a 79-node twin with no information at all is 7.6% *worse* than CtrlW (−100.20). The spread between those two runs -- same task, one with no plant information advantage at all -- is 7.0x the whole value of perfect occupancy foresight, in both directions. The gap is controller quality -- optimisation horizon, plant-model fidelity, action space -- not information.

Two caveats worth keeping in the paper. First, a *reactive band-shaping* policy cannot use foresight the way an MPC does: the LP can plan a whole trajectory of AHU supply temperatures, so information enters as a constraint on a much larger action space. The measurement here is the value of information *to this controller*, which is the honest quantity for a transferable claim. Second, the anchor is Fairbanks in winter, where the marginal cost of the planted mass dominates and an empty zone is nearly free to be cold anyway; the subset's Phoenix/Singapore scenarios are where zonal information actually pays, and even there the perfect-schedule upper bound stays single-digit percent.

Files: `policies/oracle_foresight.py`, `sbx/evalrun.py` (`--oracle foresight`, `--oracle-file`, `--oracle-record`), results `foi_anchor_ref`, `foi_anchor_ref2`, `foi_anchor_noop`, `foi_anchor`, `foi_anchor_lead{3,4,5,6,8}`, `foi_ref10`, `foi_foresight10`; schedules in `scratch/VoIOracle/foresight_ref*`.

