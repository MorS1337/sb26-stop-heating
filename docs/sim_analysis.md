# Simulator deep dive — NeurIPS 2026 Smart Buildings (Kaggle)

2026-11-05. Machine: Apple M3, 8 cores, 16 GB, macOS 25.5.
Env: `starter-kit/.venv` (uv, CPython 3.12.8 — 3.13 does **not** work, numba 0.60 has no 3.13 wheel),
numpy 2.0.2, numba 0.60.0, scipy 1.16.3, protobuf 5.29.5.
Scripts: `scratch/SimDeepDive/{bench,prof,parallel_scale,solver_anatomy,dump_meta,determinism,zone_uniformity,obs_probe,occ_probe}.py`.

**Measurement caveat.** Four other heavy jobs ran throughout (load average 11–12 on 8 cores), so *wall clock* is
inflated 1.3–5×. Everything below reports **process CPU seconds** first; those are stable except for a
±25% cache/bandwidth spread on the largest building (`building_train_2` measured 0.102 s and 0.133 s/step
in two runs). Reproduce any row with `bench.py <building> <weather> <steps>`.

---

## 1. Smoke test

```
$ .venv/bin/python -m smart_buildings.run_environment --building building_train_1 --weather fairbanks_1
=== Action spec ===   (92 continuous, 5 discrete; 39 VAVs, 4 AHUs)
=== Building metadata ===  discrete 5, continuous 92, AHUs 4, zone temp sensors 39, obs size 352, floors 3
Ran 10 step(s) in 4.3s (0.43 s/step)
```
Passes. The printed 0.43 s/step covers only the first ten steps, which include the ~2.9 s numba JIT of the
convection kernel on step 1, under contention. Steady state is 0.040–0.133 s/step (§2).

---

## 2. Speed

### 2.1 Per building — single process, zero action, 10-step warmup, 200 measured steps

| building | floors | K (AHUs) | V (zones) | obs dim | cont/discrete actions | CV grid (cells) | **CPU s/step** | steps/s (CPU) | 8640-step episode | 30-scenario sweep |
|---|---|---|---|---|---|---|---|---|---|---|
| train_8 (smallest) | 2 | 3 | 30 | 283 | 71 / 4 | 386×320 = 123,520 | **0.040** | 25.0 | 5.8 min | 2.9 CPU-h |
| train_7 | 2 | 4 | 36 | 334 | 86 / 5 | 458×279 = 127,782 | **0.049** | 20.3 | 7.1 min | 3.5 CPU-h |
| train_1 | 3 | 4 | 39 | 352 | 92 / 5 | 606×339 = 205,434 | **0.069** | 14.5 | 9.9 min | 5.0 CPU-h |
| train_5 | 4 | 8 | 79 | 652 | 184 / 9 | 858×342 = 293,436 | **0.091** | 11.0 | 13.1 min | 6.6 CPU-h |
| train_4 | 3 | 6 | 54 | 472 | 128 / 7 | 664×432 = 286,848 | **0.094** | 10.6 | 13.6 min | 6.8 CPU-h |
| train_2 (largest) | 2 | 9 | 85 | 703 | 199 / 10 | 624×422 = 263,328 | **0.102–0.133** | 7.5–9.8 | 14.7–19.1 min | 7.4–9.6 CPU-h |

Mean ≈ **0.077 CPU s/step**; a mixed 30-scenario sweep ≈ **5.4–5.7 CPU-hours**. Cost tracks grid cells
(0.32–0.55 µs/CV·step) and zone count (reward/observation bookkeeping), *not* the action count.
Cross-check at 300 measured steps (the two headline buildings): train_8 **0.0427 CPU s/step**, train_2
**0.1024 CPU s/step** — consistent with the table, within the ±25% contention spread.

Environment **construction** (`load_hybrid_environment`: gin parse → floor plan/zone map → conductance
assembly → `splu`): **4.6 s CPU** (train_8), 5.2 (train_7), 7.7 (train_1), 10.1 (train_4), 10.4
(train_5), 9.8–10.7 (train_2). `reset()` 0.06–0.19 s. First `step()` in a fresh process: ~1.5–3 s of JIT.

**Extrapolations (one core):** one 8640-step episode = **5.8–19 CPU-min**; a full 30-scenario evaluation
= **2.9–9.6 CPU-h, ≈5.5 CPU-h** for the shipped mix. Single-core throughput 7.5–25 steps/s.
For reference, the policy is irrelevant to the budget: `pid.get_action` ≈ 0.2 ms/step (≤0.5% of a step).
(The source comments price the air-side Woodbury update as "the difference between a few hundred
milliseconds a step and a second triangular solve" — in this build the whole step is 40–133 ms, so
that comment reflects a much larger/older configuration.)

### 2.2 Profile — cProfile on `building_train_2`, 60 steps

17.93 s profiled (0.299 s/step, ~3× CPU inflation from the profiler; real 0.102 s/step).

**Sorted by cumulative:**

| s | % | function |
|---|---|---|
| 17.93 | 100% | `environment.py:1650 step` |
| 11.57 | 64.5% | `simulator_building.py:285 wait_time` → `execute_step_sim` |
| 11.55 | 64.4% | `simulator_flexible_floor_plan.py:280 execute_step_sim` (physics) |
| 7.90 | 44.1% | `simulator.py:101 finite_differences_timestep` |
| 7.87 | 43.9% | `finite_difference_solver.py:968 solve` |
| **6.96** | **38.8%** | `SuperLU.solve` — 120 calls / 60 steps = **2 triangular solves per step** |
| 3.82 | 21.3% | `finite_difference_solver.py:1027 _add_air_side` (Woodbury for VAV airflow) |
| 3.69 | 20.6% | `environment.py:1499 _get_reward` → `reward_info` |
| 2.67 | 14.9% | `_get_zone_reward_infos` (85 zones) |
| 2.31 | 12.9% | `building.py:1083 get_zone_average_temps` — **3× per step** (sim, obs, reward) |
| 2.10 | 11.7% | `environment.py:1295 _get_observation` |
| 1.85 | 10.3% | `simulator_flexible_floor_plan.py:626 _get_zone_reward_info` (85 calls/step, proto + occupancy) |
| 1.15 | 6.4% | `simulator_building.py:171 request_observations` (protobuf) |
| 0.85 | 4.7% | occupancy `average_zone_occupancy` (of which `_to_local_time` 0.63, `peek` 0.76) |
| 0.77 | 4.3% | `apply_convection` → numba `_convect_room` 0.725 |
| 0.60 | 3.3% | `setpoint_schedule.is_comfort_mode` (25,680 calls — timezone conversion per zone per step) |

**Sorted by tottime (self):** `SuperLU.solve` 6.96 (38.8%) · `building.py:1069 get_zone_temp_stats` 1.10
(6.1%; **15,300 calls / 60 steps = 3 × 85 × 60**, each a fancy-index gather over a zone's cells) ·
numba `_convect_room` 0.73 (4.0%) · `apply_thermal_power_zone` 0.69 (3.9%; 10,200 calls) ·
`_get_zone_reward_info` 0.57 · `randomized_arrival_departure_occupancy.py:72 _to_local_time` 0.53 ·
`execute_step_sim` 0.32 · `is_weekend` 0.31 · `pandas_to_proto_timestamp` 0.16 (39,480 calls) ·
`CopyFrom` 0.12 (91,320 calls) · repeated-field `append` 0.14 (64,200 calls).

**Budget: ~60% sparse linear algebra** (one implicit solve + the air-side Woodbury pair), **~20% reward
bookkeeping**, **~12% observation bookkeeping**, **4% convection**, **~4% protobuf/pandas**, **<1% the
reward arithmetic itself**. Physics ≈65%, scaffolding ≈35%. Two of the three full-grid zone-mean passes
per step (obs and reward) are pure duplication.

**Grid / integration.** CV = 20 cm × 20 cm × 300 cm (`control_volume_cm`, `floor_height_cm` in every
`config.gin`). train_8: 123,520 CVs, 81,388 solved (65.9%; 42,132 exterior CVs pinned to ambient and
eliminated), `nnz`(conduction) = 405,068, sparsity 0.0061%, 30 VAV air-side terms. train_2: 263,328 CVs,
~85 air-side terms. **One backward-Euler solve per 300 s step** — no sub-stepping, no iteration/convergence
loop: `splu`(MMD_AT_PLUS_A) is **factorized once** and reused while the convection coefficient is constant
(`finite_difference_solver.py:998-1004`; every shipped weather controller returns a fixed `h`), and the
VAV airflow enters as a rank-k Woodbury update (`:1027`) rather than a refactorization. Interior-mass nodes
are eliminated analytically into a diagonal + forcing term (`:790-812`).

Isolated timing (`solver_anatomy.py`, train_8, n=81,388): **`splu` = 0.32 s once per scenario; one
triangular solve = 0.0085 s** ⇒ 2 solves = 0.017 s of the 0.040 s step. So the solve is ~42% and the
Python/protobuf around it is the rest, contrary to what the source comments assume.

### 2.3 Process parallelism

Same scenario (`train_8/fairbanks_8`), 150 steps/worker, `multiprocessing.Pool`:

| workers | aggregate steps/s | speedup | CPU s/step per worker |
|---|---|---|---|
| 1 | 8.2 | 1.00× | 0.0367 |
| 2 | 12.95 | 1.58× | 0.0410 |
| 4 | 19.3 | 2.35× | 0.0427 |
| 8 | 19.9 | 2.43× | 0.0436 |

The 8-worker row is capped by the *peers*, not by the code (the pool got only 5.6 worker-seconds per wall
second). Two conclusions that survive the noise: per-worker CPU cost rises only **19%** from 1→8 workers,
so there is no lock or shared state — workers are independent processes; and therefore on an idle 8-core
M3 expect **6.0–6.8×** from 8 workers, bounded by memory bandwidth (~1.5 GB RSS/worker at train_2 size).
There is **no thread-level parallelism** to exploit: the SuperLU solve is single-threaded, and the numba
kernel is `njit` without `parallel`.

---

## 3. Physics and reward model

### 3.1 Thermal field
* 2-D grid of CVs per floor, 4-connected (`building.py:994 _calculate_neighbors`).
* `simulator_flexible_floor_plan.py:280 execute_step_sim` is the whole step, in order:
  1. zone mean temperatures — `building.py:1083 get_zone_average_temps` → `:1069 get_zone_temp_stats`
     (mean over the zone's cells; `_zone_indices` caches the index arrays, `:1025`);
  2. per-VAV thermostat → damper + reheat valve (`vav.py:454 update_settings`, floored at
     `min_damper_setting = 0.1`);
  3. AHU mixed/supply air temperature (`air_handler.py:257 get_mixed_air_temp`:
     `T_mix = r·T_return + (1−r)·T_ambient`; `:271 get_supply_air_temp`: `T_supply = min(T_mix, setpoint)`
     while running — **the AHU can cool but never heat**; heating is the reheat coils only);
  4. per-VAV reheat (`vav.py:375 compute_reheat_energy_rate` = `ε·ṁ_air·c_p·(T_water − T_supply)`,
     `ε = min(1, C_water/C_air)` further capped by `max_discharge_air_temperature`, `vav.py:316`);
  5. zone heat = reheat share + internal gains (`internal_gain_w_per_m2 = 10`, ×0.2 outside comfort mode)
     + occupants (2/zone × `occupant_sensible_heat_w`) → `apply_thermal_power_zone`
     (`building.py:1097`, spread uniformly over the zone's cells, *assigns* `input_q` per zone);
  6. FD solve (`simulator.py:119`), then the *air-side* heat is re-evaluated at the end-of-step temperature
     and folded back into `input_q` (`:504-514`) — this is what makes damper/airflow backward-Euler stable;
  7. `apply_convection` (`:516`) — the stochastic CV swap;
  8. boiler return-water temperature from the reheat energy balance (`:567`).
* Materials: interior air (k=50 "well mixed"), interior wall, exterior wall (UA + fabric capacity — the
  per-building difficulty knob), plus an interior-mass node ~4 h behind the air. Values exist only in
  `config.gin`/`config.yaml`, **not** in the metadata.
* **Convection is stochastic**: `stochastic_convection_simulator.py:27 _convect_room` (numba) samples CVs
  with p=0.1 and swaps temperatures with a partner within `distance = 2` CVs, sequential Fisher-Yates over
  the pairs; `seed = 5` from `config.gin:106`.
* Divergence guard: field clipped to [150, 500] K with a warning (`execute_step_sim`, ~`:530`).

### 3.2 Energy, cost, carbon
Electrical energy rate (W), summed at `base_setpoint_energy_carbon_reward.py:132`:
AHU supply fan `flow·Δp/η` (`air_handler.py:366`, η=0.9) + exhaust fan `flow·(1−recirc)·Δp/η` (:393) +
AHU cooling = **thermal/COP** with COP 3.5 (`:347`; thermal = `ṁ·c_p·(T_supply − T_mix)`, `:325`) +
HWS pump (`hot_water_system.py:290`; 6 m head, η=0.98). Every shipped config uses a boiler (`BLR`), so
the heat-pump path is dead code here.

Natural-gas rate (W) = boiler load (`boiler.py:200`) =
`c_w·ρ_w·flow·(T_supply_water − T_return_water)` + tank dissipation + tank heat storage. The return-water
temperature is *derived* from the reheat heat actually delivered (`simulator_flexible_floor_plan.py:567`)
and clamped between supply and coil-entering air temperature. **This is the fixed side of the "free reheat"
forum bug (PR #2)**: the old code averaged VAV *supply-air* temperatures, which under-billed gas ~3.5× and
let a policy heat the building for ~1/700th of the gas bill with a tiny static pressure.

Exact price/carbon inputs:
* electricity 16 ¢/kWh off-peak, 18 ¢ 06–11, 20 ¢ 12–18 on weekdays, flat 16 ¢ weekends
  (`electricity_energy_cost.py:50-88`, converted to $/W/s at `:129-134`); carbon 87.8–104.7 kg CO₂/MWh by
  hour (`:17-42`, → kg/W/s at `:124`); both charge `|energy_rate|` (`:162`, `:187`), so cooling counts.
* gas `$6.69–9.02 per 1000 ft³` by month (`natural_gas_energy_cost.py:15-28`), carbon `GAS_CO2 = 53.12`
  kg/MMBtu ⇒ 0.181 kg CO₂/kWh (`utils/constants.py:33,41`).

### 3.3 Reward
`setpoint_energy_carbon_regret.py:185 compute_reward`:

```
r = [ w_p·N_regret − w_e·Ê − w_c·Ĉ ] / (w_p + w_e + w_c)              # :333-345
w_p, w_e, w_c = 0.6, 0.2, 0.2                                          # every config.gin
N_regret = (actual_prod − min_prod)/(max_prod − min_prod) − 1         # :217   (=0 if occupancy 0, :221)
Ê = (elec_cost + gas_cost) / (cost@max_electricity_rate, cost@max_gas_rate)   # :295-306
Ĉ = (elec_carbon + gas_carbon) / (carbon@same maxima)                  # :309-316
```
* Per-zone productivity (`base_setpoint_energy_carbon_reward.py:106-120`): `d = max(0, T − T_cool_sp,
  T_heat_sp − T)`, `w = productivity_deviation_ref = 5 K`, `shortfall = d/w` for `d ≤ w` and
  `1 + ((d−w)/w)²` beyond — **linear then quadratic and unbounded**, so one runaway zone dominates.
  `productivity = 300 $/person-hour × (1 − shortfall) × occupancy × Δt/3600`, summed over zones, each
  weighted by **its own occupancy, not its area**.
* Occupancy per zone from `RandomizedArrivalDepartureOccupancy` (2 occupants per zone; arrivals 06–12,
  departures 13–19, `p_arrival = 1/36` per 5-min tick; **weekends and holidays force AWAY**, so occupancy
  is 0 and the comfort term vanishes entirely on Sat/Sun — measured).
* Normalization = `c·K` (`energy_scale.py:43-44`): `max_electricity_rate = 80,000·K`,
  `max_natural_gas_rate = 205,000·K` W. **Not** clipped: draw above the denominator makes `Ê, Ĉ > 1` and
  `r < −1`. Verified for every scenario by `check_scales.py`.
* Schedule (`setpoint_schedule.py:77 is_comfort_mode`): comfort (294, 297) K weekdays 06:00–19:00 local,
  eco (289, 298) K otherwise; all 30 shipped configs are identical, no holidays.
* Scale check: 39 zones × 2 occupants ⇒ the per-step comfort numerator is ≈23,400 $/h, so the public-LB #1
  score of 0.00470 means ≈**0.8% of full comfort loss**, i.e. near-perfect comfort *and* near-zero energy.

---

## 4. What the policy can see

`evaluate.py:420 create_building_meta_data` — identical in the Kaggle gateway, which imports it rather than
reimplementing it:

| key | value |
|---|---|
| `n_discrete_action`, `n_continuous_action` | `K+1`, `3K+2+2V` |
| `temp_indices`, `pressure_indices`, `water_indices`, `recirculation_indices` | continuous slots by role, AHU order (`evaluate.py:301 get_action_indices`) |
| `zone_action_indices` | dict VAV → {damper, reheat} slots |
| `zone_temp_indices` | obs slots of the zone temperatures |
| `observation_field_names` | 283 (train_8) … 703 (train_2) |
| `floor_plan`, `zone_map` | H×W float64, **3-class labels only** (0 interior space, 1 interior wall, 2 exterior space) |
| `ahu_zone_assignments` | dict ahu_id → [room_1…] (`evaluate.py:344`) |
| `num_floors`, `morning_start_hour`, `evening_start_hour`, `comfort_temp_window`, `eco_temp_window`, `time_zone`, `holidays` | schedule, fully known a priori |
| `observation_normalization_metadata`, `action_normalization_metadata` | mean/var per field; native min/max per action |

Observation per step is exactly **`6V + 15K + 58`** (283 = 180+45+7+24+24+3 for train_8, 703 = 510+135+… for
train_2): 6 fields per VAV (damper %, reheat %, supply-air flow setpoint, zone heating setpoint sensor, zone
cooling setpoint sensor, zone air temperature) + 15 per AHU (`ahu_N_*`: supply-air temperature and
setpoint, static pressure and setpoint, supply/outside air flows, supply/discharge fan speeds, supply/exhaust
fan run commands, recirculation, cooling request count, outside-air temperature) + 7 HWS (water temperature
and setpoint, differential pressure and setpoint, run status, supervisor command, heating request count) +
24 hour-of-day and 24 day-of-week Fourier features (`num_hod_features = num_dow_features = 12` ⇒ 12 sin + 12
cos each) + `comfort_mode_now`, `comfort_mode_soon` (comfort mode **60 min ahead**) + `num_occupants`.
Verified numerically: zone temperatures are **normalized** (≈ −0.29), while
zone heating/cooling setpoint sensors (289.0/298.0), outside-air temperature (273.5) and supply-water
temperature (330.0) are **raw kelvin** — so the band the reward charges against is directly readable.

**Exposed / derivable**
* Everything about the schedule, the AHU↔zone map, and the per-zone actions and temperatures. Every
  setpoint the reward measures is in the observation.
* **The exact room segmentation is recoverable.** `zone_map` is the simulator's own input
  (`building.py:836-860` → `building_utils.construct_building_data_types` → 4-connected components of
  `zone_map == 0`). Running `scipy.ndimage.label(zone_map == 0)` reproduces `room_dict` exactly — verified:
  30 components for train_8, sizes 589/2510/5076 cells = 23.6/100.4/203.0 m², matching the simulator's
  zones. So room areas, the room-adjacency graph, per-room exterior-wall length (a UA proxy) and the
  room↔AHU grouping are all available to the policy. (In the shipped scenarios `floor_plan` and `zone_map`
  are the same label grid with only 40 differing cells out of 123,520.)
* `comfort_mode_soon` leaks the next hour of the *schedule* (not the weather).
* The geometry fingerprint (grid shape + K + V) identifies the scenario.
* `num_occupants` (building total, raw count after the `occupancy_normalization_constant` typo — the
  attribute the observation reads is never set, so it is effectively unnormalized) is a good proxy for the
  reward's own `total_occupancy`: measured within 1–2 people of 60 during the working day.

**Not exposed**
* Material and plant parameters: exterior/interior conductivity, density, heat capacity, the interior-mass
  node, `internal_gain_w_per_m2`, CV size, floor height, fan efficiency, COP, boiler rates, water setpoint
  range, occupancy seed, weather station/file. They live in `config.gin`/`config.yaml` only.
* **Per-zone occupancy.** Only the building total is observable; the comfort term weights every zone by
  its own hidden occupancy. This is the single most important unobservable.
* **Weather forecast.** The simulator replays the whole hourly CSV, but the policy only ever sees the
  *current* outside-air temperature.
* Room-level power/energy, the temperature field (only zone means), zone min/max temps, `input_q`, the
  interior-mass temperature, per-VAV actual airflow (only the setpoint and the AHU total).

---

## 5. Shipped baselines

| policy | file | logic |
|---|---|---|
| `bang_bang` | `baselines/bang_bang.py` | per AHU: mean zone temp vs band → full heat (T_sp +1, Δp +1, damper 1, reheat 1) / full cool (T_sp −1, Δp +1, damper 1, reheat 0) / off (damper & reheat at minimum). Recirculation pinned at the cap. Water on iff any AHU heating; pump iff any active. |
| `fixed_setpoint` | `baselines/fixed_setpoint.py` | `pid` with the eco band replaced by the comfort band — isolates the value of the night setback. |
| `pid` | `baselines/pid.py` | per AHU PID (kp .5, ki .02, kd .1, ±1 clamp) on the *mean* zone error → supply-air temp setpoint, fan Δp = 2|u|−1, idle inside the band; per-VAV damper ∝ |error|/2 K; **binary** reheat valve. |
| `pid_plus.py` chain | `baselines/pid_plus.py` | `PIDWorstZonePolicy:67` (AHU loop on the *worst* zone) → `PIDEconomizerPolicy:266` (+ proportional economizer, `_recirculation:302`, with a fan-vs-chiller break-even in Pa) → `PIDProportionalReheatPolicy:353` (+ modulating reheat valve, full open at 2 K) → **`PIDWaterResetPolicy:403`** (+ hot-water reset = max(mean shortfall/4 K, q25 of per-AHU effort), `_water_fraction:545`). |

`baseline_returns.json`, `score_leaderboard.py` metric = mean over scenarios of |total_reward| / 8640
(30 scenarios × 8640 steps):

| policy | score | mean total return |
|---|---|---|
| **pid_water** | **0.05414** | −467.8 |
| fixed_setpoint | 0.06592 | −569.5 |
| pid | 0.06736 | −582.0 |
| bang_bang | 0.22520 | −1945.7 |
| zero action, 200 steps (my measurement, train_8/train_2) | ≈0.13–0.15 | – |
| *public LB #1, 2026-11-04* | **0.00470** | – |
| *public LB #20* | ≈0.00594 | – |

Per weather (`pid_water`): atlanta 0.030, denver 0.037, singapore 0.047, fairbanks 0.078, phoenix 0.078 —
i.e. deep cold and peak cooling are where hand-written controllers lose.
**The shipped baselines are 11.5× worse than the public leaderboard.** What the top of the LB implies,
arithmetically: with comfort in band at all times `N_regret = 0`, so `r = −0.2(Ê+Ĉ)`, and a score of
0.00470 means `Ê+Ĉ ≈ 0.0235` — i.e. ≈7.5 kW of *average* draw against the 320 kW + 820 kW denominators of a
4-AHU building, or proportionally less in gas. At the ventilation floor (damper 0.1, static pressure 200 Pa)
a 39-zone building draws ~1.7 kW of fan power alone, so that figure is reachable only if reheat is nearly
free in the cold scenarios. Either the leaders found a genuinely cheap way to hold every zone in band, or
they lean on the known billing seams (PR #2/#3) — worth knowing before we spend days trying to match them.

---

## 6. Speed-up feasibility

| option | expected gain | effort | risk |
|---|---|---|---|
| **(a) process parallelism** | **6.0–6.8×** on 8 idle cores (measured 2.35× on 4 *loaded* cores; per-worker CPU +19% at 8) | ~30 LOC + runner, half a day | none; independent processes, **1.04 GB** peak RSS each on train_2 (measured `maximum resident set size`) |
| **(b) vectorize / numba the scaffolding** | **1.3–1.5×** single core | 150–250 LOC, 1 day | low. Targets: the 3 redundant full-grid zone-mean passes per step (12.9%), `get_zone_temp_stats`' 15,300 fancy-index gathers (12.5%), the 10,200 `apply_thermal_power_zone` calls (3.9%), 25,680 `is_comfort_mode`/timezone conversions (3.3%), and the `reward_info` protobuf round-trip that is immediately re-read (≈6%). One `np.bincount` pass + caching the per-step zone means covers most of it. |
| **(c) reduced-order thermal model** | removes ~60% of the step → **2.0–2.5×** single core | 300–500 LOC, 2–4 days | medium. The step is *exactly linear* given `h`: `T⁺ = A⁻¹(C/dt·T + q + d)`, inputs are zone-uniform power, ambient (a Dirichlet diagonal term) and rank-k supply-air terms, and the only outputs the reward reads are zone means. Projecting onto a Krylov/POD basis `V` of the reachable set and precomputing `A⁻¹V` once per scenario (m≈150–250 solves, 1.5–2.5 s) turns every step into one small dense matvec: **~0.1 ms instead of 17 ms of triangular solves**, with a per-scenario measurable projection residual. |
| **(d) batched GPU re-implementation (PyTorch/JAX)** | 100–1000× throughput, hundreds of envs, differentiable | 1,500–2,500 LOC, 5–10 days | high, and not in 8 days at the fidelity the LB needs (boiler tank dynamics, coil effectiveness caps, stochastic convection, reward). Also our GPU is a Radeon RX 9060 XT on a Windows box — ROCm is not available there, so it would be PyTorch-DirectML, far slower than CUDA for these tiny kernels and with no sparse LU. |
| **(e) learned surrogate (fit an RC/linear model on logged rollouts)** | 50–100×/step once fitted, and differentiable | 300–600 LOC + data, 2–3 days | medium-high. The reward is dominated by *worst-zone* excursions, which a mean-fit surrogate smooths away. Usable inside a planner, never for producing the score. |

**A naive "one node per zone" lumped model is not accurate enough** — measured on train_8 over 120 steps:
intra-zone temperature std is **1.04–1.52 K** (per-zone range 5.2–7.0 K) while the *inter*-zone std is only
**0.18–0.78 K** (range 0.7–2.9 K). The room air is not isothermal and the internal gradient shape is
non-stationary, so a lumped model must keep at least air + interior-mass nodes per zone and be fitted, or
be a projection of the exact operator (option (c)) rather than a hand-built RC network.

**Recommendation: (a) + (b) now, (c) as the training accelerator, skip (d)/(e) for this deadline.**
8 workers × 1.4× faster step takes a 30-scenario sweep from ~5.5 h to **~35–45 min wall**; with (c) to
**~15–20 min**. That is the difference between one experiment a day and twenty.

**Fidelity discipline.** The winning margin is 0.0047 vs 0.0047-adjacent; a 0.5 K error in one zone mean
moves the comfort term by 0.5/5 × 0.6 ≈ 6% of scale, which is 5× the whole margin. So any approximated
simulator may only be used *inside* a planner whose output is re-scored by the real simulator — never to
produce the number that goes on the leaderboard.

---

## 7. Determinism

`determinism.py`, train_8/fairbanks_8, 60 steps, zero action:
* two independently constructed environments: `max|Δobs₀| = 0`, `max|Δreward| = 1.49e-8`, mean reward
  identical to 8 decimals, occupancy sequence identical.
* `reset()` twice inside one env: same (1.49e-8).

The simulator is deterministic to float round-off; the 1.5e-8 is accumulation-order noise, not RNG.
All randomness is seeded: `StochasticConvectionSimulator.seed = 5` in every `config.gin:106` (seeds both
CPython `random` and numba's `np.random`), and `RandomizedArrivalDepartureOccupancy(seed=17321)` by default,
not overridden anywhere. Two caveats:
1. **`reset()` resets neither RNG.** Neither class implements `reset()`, so a second episode in one `env`
   object continues the first episode's streams. The gateway builds a fresh env per scenario, so scoring is
   unaffected — but locally, "run 2 episodes and average" is not what you get.
2. The occupancy Markov chain is advanced **twice per 5-minute step**: the observation's `num_occupants`
   queries the *trailing* window `[t−5 min, t]` and the reward queries the *forward* window `[t, t+5 min]`,
   and the memo cache is keyed on the interval, so both miss (`randomized_arrival_departure_occupancy.py:254`).
   Deterministic, but the effective arrival/departure rate is ~2× nominal; measured ramp is 0 → 60 people
   between 06:00 and 12:00 and 60 → ~2 between 13:00 and 16:00.

---

## 8. Kaggle package vs GitHub package

`diff -r comp_data/smart_buildings starter-kit/smart_buildings`: **22 files differ, +/−146 lines.**

| file | change | class |
|---|---|---|
| `smart_control/utils/scenario_generator.py:1711` | the tracker's occupancy series now reuses `reward_response.total_occupancy` instead of calling `average_zone_occupancy` with **VAV** ids | **behavioural — tracker only** |
| `smart_control/simulator/constants.py:97` | `VIDEO_PATH_ROOT` `/cns/...` → `"videos/"` | behavioural, diagnostics only |
| `smart_control/simulator/building_utils.py` | `save_images_to_cns_for_debugging` → `save_debug_images`, dir `debug_images/` | cosmetic (debug path) |
| `smart_control/utils/constants.py:57` | `REPO_DIRPATH` no longer honours bazel's `TEST_SRCDIR` | cosmetic |
| `baselines/pid_plus.py` (4 hunks) | docstring numbers/scenario names removed | cosmetic |
| 16 other files | `TODO(user)`→`TODO`, `# pylint: disable=…` removal, docstring wording | cosmetic |

**Nothing that changes the physics, the reward, the action/observation interface, or the metadata.** The one
behavioural hunk is in the *metrics tracker* (`SimulationTracker`), which the Kaggle gateway never
instantiates — but it did matter locally: those spurious `VAV_*` occupancy queries consumed RNG draws and so
perturbed the simulated trajectory whenever the tracker was on. That is why the GitHub HEAD "phantom
occupancy" fix changes local `evaluate.py` numbers and not the KB gateway's.

Non-Python: `comp_data` adds `evaluation_scenarios/building_train_1/` (one sample) and five **evaluation**
weather CSVs — `atlanta_may, denver_apr, fairbanks_jan, minneapolis_jan, phoenix_jul` — while the starter kit
ships 30 `training_scenarios` on `{atlanta,denver,fairbanks,phoenix,singapore}_<n>`. The hidden set is 6
unseen buildings × those 5 station-months.

### Which version runs, and how (`comp_data/kaggle_evaluation/`)
* Simulator and policy run in **separate containers** joined by the Hearth relay (standard gymnasium
  protocol, one `process_turn` endpoint).
* The simulator is the **Kaggle copy** of `smart_buildings`, loaded with
  `smart_buildings.evaluate.load_hybrid_environment` under `cwd = simulator_root`, `env.metrics_path = None`
  (`gateway.py:308-340`).
* Scenario discovery = `smart_buildings/evaluation_scenarios/*/*/config.gin` only (`:290-303`); the training
  scenarios are never used for scoring. No cap on scenario count and **no wall-clock check inside the
  container** (the rerun allowance is enforced outside it).
* Metadata is built once per scenario by the same `create_building_meta_data` and delivered in the gym `info`
  dict **on the first turn only** (`gateway.py:186-212`); a **fresh `Policy` object is constructed per
  scenario** (`smart_buildings_inference_server.py:102-110`), so no state can carry across scenarios.
* Timeouts: **15 min** for a scenario's first turn, **60 s** per later turn (`:90-91`).
* Episode length comes from the scenario's own `config.gin` (8640 steps); the only truncation is the host's
  `HEARTH_HOST_SMART_BUILDINGS_MAX_STEPS` smoke-test variable.
* Output is one row per scenario = mean per-step reward; the run aborts as an invalid submission on a
  `ValueError` from `env.step` or on `ACTION_REJECTION_REWARD = −inf`, so the policy must never emit NaN,
  out of range, or a command the building refuses.
* The gateway knowingly mirrors the starter kit's `_occupancy_normalization_constant` typo, so
  `num_occupants` is unnormalized raw count in both places.

---

## 9. Implications for our strategy

**Leaderboard**
1. The tax is the sweep, not the policy: 5.5 CPU-h per 30-scenario evaluation. Option (a) alone
   (~30 LOC, half a day) takes it to ~1 h wall, (b) to ~40 min. Do that first — it gates everything else.
2. Honest baselines are 0.054; the LB leader is 0.0047. A 10× gap will not be closed by PID tuning, so the
   strategy has to be structural: per-zone, per-AHU optimizing control using the setpoint sensors (which
   *are* observable, raw kelvin) and the room geometry (which is recoverable from `zone_map`, §4), instead of
   a plant-level scalar loop. The plant is slow (τ ≈ hours) and the state is 30–85 zones, so a
   model-based receding-horizon controller fits comfortably inside the 60 s/turn budget.
3. Do **not** build on the known billing seams (gas under-billing PR #2, return-air area weighting PR #3,
   frozen pipes). The host has said final verification runs on a corrected simulator, and PRs #2/#3 are
   already open — that is exactly the class of trick a fix deletes. Comfort is 0.6 of the reward and its
   formula has no seam worth exploiting, so spend the 8 days on being robustly comfortable at low energy.
4. Two cheap, transferable wins are already in the baselines: drive the AHU loop off the **worst** zone
   (`pid_worst`) and reset the hot-water temperature on demand (`pid_water`). Those two are most of the
   0.068 → 0.054 gain; the remaining value is in the per-zone actuators and in not over-ventilating.
5. Free-money structure, and the sharpest quantitative finding in this report: **every shipped episode starts
   on a Sunday at 06:00** (`start_timestamp` = 2023-01-01, -02-12, -05-07, -06-18, -09-10, -10-22, all
   Sundays), so a 30-day episode contains **9 weekend days out of 30 (30%)**, and on those days occupancy is
   identically 0 (`is_work_day` forces every occupant AWAY, `randomized_arrival_departure_occupancy.py:155`),
   which makes the comfort term **exactly 0** by construction (`setpoint_energy_carbon_regret.py:221`).
   Measured on the Sunday start: `total_occupancy = 0.00`, `normalized_productivity_regret = 0.0000`,
   `Ê = 0.233`, `Ĉ = 0.390` ⇒ `r = −0.1246` for the zero action, i.e. **on 30% of the steps the reward is
   nothing but the fuel bill**. The optimal weekend action is "plant off, let the fabric float" (subject only
   to not freezing / not diverging); every shipped baseline keeps conditioning at the weekend eco band.

**Paper**
6. The defensible novelty in 8 days is the **information structure**, not a faster simulator: (i) the
   objective is a per-zone, occupancy-weighted, *unbounded* worst-zone comfort charge, while per-zone
   occupancy is unobservable (only the building total is in `obs`) and the per-zone chain is queried on two
   different intervals per step, so the effective arrival/departure rate is ~2× nominal [INFERENCE from the
   cache key + measured ramp 0→60 between 06:00 and 12:00] — a POMDP whose hidden state is exactly what
   weights the objective; (ii) the room
   geometry, areas, adjacency and AHU grouping are fully recoverable from the metadata by replaying the
   simulator's own connected-components, while every material/plant parameter is hidden. That is a clean,
   honest framing: **online system identification + control under partial observability**, with a measurable
   ablation (fitted model + MPC vs model-free vs rule-based) and the required real leaderboard score.
7. Quantify it with the measurements here: 0.040–0.133 CPU s/step, 2.9–9.6 CPU-h per 30-scenario sweep on
   one M3 core, ~60% of it a single sparse LU solve on a 123k–263k-CV grid. That is the concrete reason RL
   is impractical in this environment and a structured controller is the right answer — which is the
   argument we want the paper to make.
8. Do not promise a JAX/PyTorch simulator rewrite as a contribution: 5–10 days, no sparse-LU equivalent, no
   ROCm on our GPU, and its fidelity cannot be validated against the hidden set before 2026-11-13.
