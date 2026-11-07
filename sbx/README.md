# sbx — scenario runner for the Smart Buildings competition

Everything here runs with the starter-kit venv: `starter-kit/.venv/bin/python`
(CPython 3.12; 3.13 breaks numba). Run commands from the workspace root.

## Simulator trees

Two full copies of the simulator, outside `starter-kit/` so it stays pristine:

| tree | contents |
| --- | --- |
| `sims/shipped` | starter-kit HEAD `95cea5b` — what the Kaggle notebook/gateway runs |
| `sims/corrected` | the same + PR #2 (energy-balanced hot-water tank) + PR #3 (airflow-weighted return-air mixing) |

```bash
starter-kit/.venv/bin/python -m sbx.simload shipped      # prints the verified smart_buildings.__file__
starter-kit/.venv/bin/python -m sbx.simload corrected
```

`sbx.simload.use_sim("shipped"|"corrected")` must be called **before** anything
imports `smart_buildings`: it removes the starter-kit editable meta-path finder
(which otherwise shadows `sys.path`), puts the tree root first, and `chdir`s
there because every `config.gin` names its data files relative to the tree root.
One process = one tree.

## Run a policy

```bash
# the 30 shipped scenarios, 30-day episodes, 6 processes
starter-kit/.venv/bin/python -m sbx.evalrun \
    --policy baseline:pid_water --set dev30 --steps 8640 --sim shipped \
    --workers 6 --tag pid_water_shipped

# one station, a policy from policies/, with PARAMS overrides
starter-kit/.venv/bin/python -m sbx.evalrun \
    --policy policies.ctrl_v1:CtrlV1 --params '{"kp": 0.4}' \
    --set dev30_fairbanks --sim corrected --tag ctrl_v1_fbk

# explicit scenarios: ids, config.gin paths or scenario dirs
starter-kit/.venv/bin/python -m sbx.evalrun --policy baseline:pid \
    --scenario building_train_1/atlanta_2 \
    --configs /abs/path/to/building_train_2/denver_8 --steps 8640 --workers 2
```

* `--policy`: `baseline:<key>` (`pid_water`, `pid`, `bang_bang`, `fixed_setpoint`,
  `all_off`, taken from the selected tree), `module:Class`, or a bare baseline key.
* `--set`: `dev30`, `dev30_<station>`, `dev30_<strip>`, `gen_train`, `gen_hold`,
  `ext_hold`, `gen_all` (see `sbx/scenarios.py`).
* `--components parity` (default) repeats `evaluate.py`'s extra per-step
  reward-info call so totals match `smart_buildings.evaluate` bit-for-bit;
  `--components hook` harvests the response from the step's own reward
  computation instead: identical totals, ~7% cheaper (measured).
* `--start-offset-days N` shifts every episode's start date by N days.
* `--occupancy-seed N` binds `RandomizedArrivalDepartureOccupancy.seed`
  (default 17321) before the environment is built, for robustness tests against
  a different hidden occupancy draw; it is recorded in every result JSON and in
  `summary.json`. Omitting it leaves the config untouched.
* `--workers`: one process per scenario (`maxtasksperchild=1`), so a scenario
  never inherits another's numba/gin state. `--workers 1` runs in-process.
* `--fast`: install `sbx/fastpatch.py` in each worker (see below). Off by
  default; a run without it is unchanged, and the Kaggle/official path never
  imports it.
* `--cenv bookkeeping,observation` (opt-in, additive to `--fast`/`--oracle`):
  installs the C components from `cenv/` in each worker, by monkeypatching the
  simulator classes before the environment is built (see below). `--cenv all`
  enables every component whose shadow gate has passed in this checkout;
  naming a component enables it regardless, which is what a gate run does. The
  list is recorded in every result JSON and in `summary.json`. Off by default:
  with no flag `cenv` is never imported.
* `--oracle occupancy|forecast|occupancy,forecast` (analysis only, `--oracle-horizon`
  sets the forecast length, default 12 h): before each `get_action`, a policy that
  defines `observe_oracle(info)` receives `zone_occupancy` — the per-zone
  `average_occupancy` of the **last completed** interval, read from the reward
  info the step already built (a realistic 5-min-late zonal sensor; a second
  occupancy-model query would advance the shared `RandomState` and move every
  later arrival, forum B3) — and `forecast_oat` — outside air temperature in K
  for the next 1..H hours from the episode's own `ReplayWeatherController`, so it
  is aligned with the simulated clock by construction. Zone order follows
  `policy.zone_ids`, mapped from the reward info's `room_<n>` keys by numeric
  suffix. The flag is inert for policies without the hook, and a policy that
  receives the payload and ignores it reproduces a normal run bit-for-bit
  (`policies/oracle_noop.py:OracleNoopCtrlW`). Analysis-only: never submitted.
* `--oracle foresight --oracle-file <dir|json>` (analysis only): plays back a
  **recorded** per-zone occupancy schedule and hands the policy
  `foresight_occupancy` (dense `steps x zones` matrix of a reference run's
  `average_occupancy`) plus `foresight_step` (its row index, so no offset is
  needed). Record one with `--oracle-record <dir>` together with
  `--oracle occupancy` on any policy -- the schedule is read from the reward
  info, so it costs no extra occupancy-model query. Occupancy is exogenous: the
  same episode recorded under CtrlW and under `baseline:all_off` gives sparsely
  identical schedules (`docs/voi_oracle.md` section 9), which is what makes
  the playback a valid oracle. `--oracle-record` writes one sparse JSON per
  scenario (2016 steps x 39 zones = 70 KB) and is inert without `--oracle
  occupancy`. Policies named `oracle_foresight*` are analysis-only: never
  submitted.

Output: `results/<tag>/<building>__<weather>.json` per scenario (total and mean
per-step reward, wall/CPU seconds, comfort/energy/carbon sums overall, by
weekday-occupied / weekday-unoccupied / weekend bucket and by local hour, min
zone temperature in occupied hours, count of steps with an occupied zone >5 K
out of band) plus `results/<tag>/summary.json` with the leaderboard score
(`mean over scenarios of |mean per-step reward|`, lower is better), and
per-weather and per-building means.

## Faster local runs (`--fast`, `sbx/fastpatch.py`)

`--fast` monkeypatches the simulator at runtime, after `sbx.simload.use_sim` and
before the environment is built. No file under `sims/` or `starter-kit/` is
edited, official/Kaggle scoring is unaffected (it runs the unpatched tree), and
`--fast` off imports nothing.

```bash
starter-kit/.venv/bin/python -m sbx.evalrun --policy policies.ctrl_w:CtrlW \
    --scenario building_train_2/atlanta_3 --steps 8640 --sim shipped --fast \
    --workers 3 --tag ctrlw_fast

# tune locally with the same patches
starter-kit/.venv/bin/python -m sbx.tune --policy policies.ctrl_w:CtrlW \
    --tag ctrlw_tune --fast
```

What it patches (`sbx/fastpatch.py` has the full rationale):

| patch point | what changes |
|---|---|
| `FiniteDifferenceSolver._add_air_side` | keeps the dense Woodbury response `A^-1 P` (built from the same blocked solves that already build `_sensed_response`; `k x n` float64, 164 MB on `building_train_2`) and replaces the per-step `A^-1 (P q)` triangular solve with a dense matvec |
| `FiniteDifferenceSolver.solve` | integer gathers into preallocated buffers instead of boolean-mask gathers plus four n-vector temporaries per step |
| `FloorPlanBasedBuilding.get_zone_average_temps` | memoized on the identity of `building.temp` (the solve installs a fresh array; `apply_convection` is wrapped to drop the memo) and computed as one flat gather plus one `np.mean` per zone, instead of 3-4 fancy-index passes with min/max that nobody reads |
| `SetpointSchedule.is_comfort_mode` / `get_temperature_window` | per-instance timestamp memo; `_localize_or_convert_timezone`, `is_weekend` and occupancy `ZoneOccupant._to_local_time` use one memo shared by every zone, since they depend only on the timestamp and the time zone |
| `SimulatorFlexibleGeometries._get_zone_reward_infos` / `_get_air_handler_reward_infos` | static per-zone VAV shares, room areas and setpoint windows hoisted out of the loops; the per-zone infos of a step are reused by the parity reward call (they are a pure function of the temperature field and timestamp) |
| `FloorPlanBasedBuilding.apply_thermal_power_zone` | same assignment through a precomputed flat index vector (3x cheaper) |

Exactness: `policies.ctrl_w:CtrlW`, 8641-step episodes, `--components parity`,
unpatched vs `--fast`, both simulator trees (full campaign: per-scenario JSON
records in `scratch/FastExact/records`, log `scratch/FastExact/campaign_final2.log`):

| tree | scenario | episode total (unpatched = `--fast`) | \|Δ total\| | max \|ΔT\| control volumes | max \|ΔT\| zone means |
|---|---|---|---|---|---|
| shipped | `building_train_8/fairbanks_8` | -208.819351931546 | 0 | 2.27e-13 K | 5.7e-14 K |
| shipped | `building_train_1/atlanta_2` | -27.731165579042 | 0 | 2.27e-13 K | 0 |
| shipped | `building_train_2/phoenix_6` | -62.203300826744 | 0 | 2.27e-13 K | 1.1e-13 K |
| corrected | `building_train_2/atlanta_3` | -24.115765109600 | 0 | 2.84e-13 K | 5.7e-14 K |

The per-step reward is bit-identical at every one of the 8641 steps on all four
legs. Against the project's exactness standard (≤1e-12 relative per temperature
component along a closed-loop episode, ≤1e-9 relative on the episode total) the
total is exact (relative difference 0) and the worst **component-relative**
deviation over the whole trajectory — not only at the end — is 1.0e-15 for
shipped `train_8` (1000 steps, every step sampled) and 1.2e-15 for corrected
`atlanta_3` (800 steps, every 8th sampled): ~1000x inside the bound, and it does
not grow with step count (max \|ΔT\| sits at 1.7e-13 K from step 400 to 1000).
The unpatched legs reproduce the pre-`fastpatch` baseline `results/ctrw_all`
bit-for-bit (8641-step totals identical to the last bit on all three shipped
scenarios), so `--fast` off is unchanged.

Speed, per-step CPU, single process, `policies.ctrl_w:CtrlW`, 300 steps with
the first 40 dropped. Conditions interleaved (plain → `--fast` → `--cenv all` →
`--fast --cenv all`, 5 repeats) so machine load hits all of them equally; this
machine's load average was 7-18 on 8 cores while it ran (several agents), which
inflates the absolute milliseconds ~2x, so compare ratios within a row:

| scenario | plain | `--fast` | `--cenv all` | `--fast --cenv all` | `--fast` vs plain | `--fast --cenv` vs `--cenv` |
|---|---|---|---|---|---|---|
| shipped `building_train_8/fairbanks_8` | 29.38 ms | 17.85 ms | 18.58 ms | 15.96 ms | 1.65x | 1.16x |
| shipped `building_train_2/phoenix_6` | 77.82 ms | 49.23 ms | 56.65 ms | 56.85 ms | 1.58x | 1.00x |
| corrected `building_train_2/atlanta_3` (largest) | 91.70 ms | 58.28 ms | 79.00 ms | 59.92 ms | 1.57x | 1.32x |

Quiet machine (5 alternating runs per condition, taken before the other agents
started): `building_train_8/fairbanks_8` 13.22 → 7.82 ms (1.69x),
`building_train_1/atlanta_2` 25.13 → 13.04 ms (1.93x),
`building_train_2/atlanta_3` 42.39 → 24.64 ms (1.72x).

What limits it: on the largest building 9.1 ms of the remaining 24.6 ms (quiet
machine) is the single sparse triangular solve per step that no patch can remove
(the air-side solve it replaces cost 9.3 ms; the dense matvec is 2.7 ms). The
rest is numba convection, the occupancy Markov chains (they consume the shared
`RandomState`, so they cannot be skipped) and protobuf assembly — what `--cenv`
ports to C.

With `--cenv` (`--fast --cenv all`: `evalrun` installs `--fast` first, then the
C components): the two are compatible and the combination is bit-identical to
`--cenv all` alone (max CV temperature difference 0.0 K over a 400-step CtrlW
episode on `building_train_8/fairbanks_8`, identical total). `cenv`'s `thermal`
component replaces `FiniteDifferenceSolver.solve` wholesale, air side included,
so `--fast` stands down from the solve and the Woodbury machinery is never built
(no dense `k x n` response, no extra memory); the schedule/zone-mean/reward
patches still apply and are what the remaining gain comes from. Measured in the
interleaved table above: `building_train_8/fairbanks_8` 18.58 → 15.96 ms/step
(1.16x) on top of `--cenv all`, 29.38 → 15.96 ms/step (1.84x) against plain
Python; on the two `building_train_2` legs the gain over `--cenv all` alone is
1.00x-1.32x.
Cost of the patch: peak RSS +170 MB on `building_train_2` (measured on a 60-step
`phoenix_6` run: plain 1076 MB → `--fast` 1246 MB; the `k×n` dense response is
164 MB) — that is per worker, so `--workers 6` on the largest buildings needs
~1 GB more. Under `--fast --cenv all` the peak RSS equals `--cenv all` alone
(1442 vs 1449 MB), because the dense response is never built in that
combination. `fastpatch.apply(..., dense_air_side=False)` restores the shipped
per-step triangular solve (all other patches stay) for memory-constrained runs.

## C components (`--cenv`, `cenv/`)

`--cenv` replaces individual simulator functions with C, one component at a
time, and only after that component has been proved to reproduce the Python
result. Installation is a runtime monkeypatch: no file under `sims/` or
`starter-kit/` is touched, so official/Kaggle scoring (which runs the unpatched
tree) is unaffected, and with no flag nothing from `cenv/` is imported.

```bash
# gate one component against the Python original, on a live episode
starter-kit/.venv/bin/python -m cenv.verify.shadow --component bookkeeping \
    --scenario building_train_8/fairbanks_8 --steps 200 --policy policies.ctrl_w:CtrlW

# per-function cost, python vs C, on the same state
starter-kit/.venv/bin/python -m cenv.verify.micro --scenario building_train_8/fairbanks_8

# interleaved A/B of per-step CPU (one process, blocks alternate)
starter-kit/.venv/bin/python -m cenv.verify.ab --scenario building_train_8/fairbanks_8 \
    --steps 100 --repeats 3 --config "" --config bookkeeping,observation

# whole-episode gate: trace both sides, compare per step
starter-kit/.venv/bin/python -m cenv.verify.gate trace --scenario building_train_8/fairbanks_8 \
    --steps 8641 --cenv "" --out /tmp/g/t8_py.npz
starter-kit/.venv/bin/python -m cenv.verify.gate trace --scenario building_train_8/fairbanks_8 \
    --steps 8641 --cenv all --out /tmp/g/t8_c.npz
starter-kit/.venv/bin/python -m cenv.verify.gate compare /tmp/g/t8_py.npz /tmp/g/t8_c.npz
```

Layout: `cenv/src/*.c` + `cenv/include/*.h` (one `.c` per component, all globbed
into `cenv/_build/libcenv.{dylib,so}` by `cenv/build.py`, which rebuilds when any
source, header or flag changes), `cenv/bindings.py` (ctypes loader),
`cenv/<component>.py` with `install(sim_root)` and `shadow_pairs(shadow, sim_root)`.
Flags are `-O3 -ffp-contract=off`, never `-ffast-math`, so a C sum can be made to
match numpy's summation order exactly (`cenv_pairwise_sum` is numpy's pairwise
algorithm: bit-identical to `np.add.reduce` for every zone-sized array).

| component | patch points | measured cost, `building_train_8` |
|---|---|---|
| `bookkeeping` | `FloorPlanBasedBuilding.get_zone_temp_stats` / `get_zone_average_temps`; `SetpointEnergyCarbonRegretFunction.compute_reward`, its productivity / rate helpers, and `ElectricityEnergyCost` / `NaturalGasEnergyCost` `cost`/`carbon` | one zone's stats 27.7 -> 17.1 us; a whole-building mean pass (30 zones) 1117 -> 196 us, 3.2 passes per step; `compute_reward` 180 -> 88 us |
| `observation` | `Environment._get_observation` (writes the float32 vector straight from the step's `ObservationResponse`; keeps the protobuf path whenever a metrics writer or histogram reducer is configured) | 4.02 -> 2.77 ms |

Exactness: the shadow gate reports **every** compared quantity bit-identical over
150-400 step episodes (per-zone min/max/mean, all reward-response fields, the
whole observation vector), and whole CtrlW episodes of 8641 steps reproduce the
official run exactly -- identical total reward, identical per-step reward, all
259,230 zone-temperature values bit-identical (see `cenv/verify/gate.py`).
Per-step CPU, single process, CtrlW: `building_train_8/fairbanks_8`
31.55 -> 28.82 ms (1.09x) over a full episode. What remains in Python on this
path is the `RewardInfo` protobuf construction (30-85 zone messages per step,
~4.8 ms of a 31 ms step on `building_train_8`) and the sparse solve.

## Scenario pools

`sbx/make_pool.py` generates, reproducibly (all seeds recorded in
`sbx/pools/manifest.json`):

* 24 new buildings at shipped scale (16 `gen_train`, 8 `gen_hold`), zone counts
  28–85 and AHU counts 3–9;
* new weather strips `<station>_g<k>` — same recorded 2023 CSV, replayed from a
  Sunday 06:00 in the 12-day gap before that station's shipped window `k+1`;
* 64 `gen_train` + 32 `gen_hold` + 17 `ext_hold` runnable scenarios, generated in
  `sims/shipped` and mirrored into `sims/corrected`.

```bash
starter-kit/.venv/bin/python -m sbx.make_pool            # add --force to regenerate
```

`ext_hold` pairs one strip from each external climate station contributed by
`sbx/weather_ext.py` with the eight `gen_hold` buildings plus two shipped ones;
install those presets into both trees first:
`starter-kit/.venv/bin/python sbx/weather_ext.py --copy-to sims/<sim>/smart_buildings/weather`.

## Repeating the evaluate.py cross-check

Starter-kit must stay untouched, so the reference run writes its results
elsewhere (`--results-dir`) and skips `.pyc` files. The `--buildings` /
`--weathers` filter intersects the discovered scenarios, so this pair selects
exactly `building_train_1/atlanta_2` and `building_train_8/fairbanks_8`.

```bash
cd starter-kit && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m smart_buildings.evaluate \
    --policies pid_water --buildings building_train_1 building_train_8 \
    --weathers atlanta_2 fairbanks_8 --steps 288 \
    --results-dir /tmp/ref_eval && cd ..

starter-kit/.venv/bin/python -m sbx.evalrun --policy baseline:pid_water \
    --scenario building_train_1/atlanta_2 building_train_8/fairbanks_8 \
    --steps 288 --sim shipped --workers 2 --tag parity
```

Measured: `total_reward` agrees to 0.0 absolute difference in both component
modes; `--sim corrected` moves the pair to `-0.00811 -> 0.00000` and
`-2.58049 -> -2.00275`.
## Parameter tuning (`sbx/tune.py`, CMA-ES)

`policies/ctrl_v1.py` exposes one global `PARAMS` dict shared by every building,
so tuning is a black-box search over that vector. `sbx/tune.py` runs CMA-ES in a
normalized `[0, 1]^d` space, evaluates every candidate on a fixed stratified
subset, and keeps honest bookkeeping of what it tried.

```bash
# 15-knob core space, 8 scenarios of gen_train, 30 generations
starter-kit/.venv/bin/python -m sbx.tune \
    --policy policies.ctrl_v1:CtrlV1 --space sbx/spaces/ctrl_v1_core.json \
    --set gen_train --subset 8 --subset-seed 0 --steps 2880 --sim corrected \
    --popsize 10 --workers 7 --generations 30 --tag tune_v1

# continue where it stopped (--generations is the total, not an increment)
starter-kit/.venv/bin/python -m sbx.tune ...same flags... --resume

# generalization: best.json on the held-out set, standard runner, own tag
starter-kit/.venv/bin/python -m sbx.tune \
    --policy policies.ctrl_v1:CtrlV1 --validate results/tune/tune_v1/best.json \
    --set gen_hold --steps 8640 --sim corrected --workers 7 --tag tune_v1_hold
```

Space files (`sbx/spaces/*.json`): `{"sigma0": 0.25, "params": [...]}` where each
entry is `{name, init, lo, hi, scale}` with `scale` one of `lin` (linear map of
`[lo, hi]`), `log` (geometric, `lo > 0`), `int` (rounded) or `bool` (true at
`x >= 0.5`). `init` may be omitted: `sbx.tune` reads the policy class's *live*
`PARAMS`, so a changed default moves the incumbent automatically (and the gen-0
baseline evaluation) without editing the space.

Per generation it asks λ candidates and submits all λ×S jobs to one pool at
once, one spawn process per (candidate, scenario) with `maxtasksperchild=1`,
reusing `sbx.evalrun`'s worker. Objective = mean over the subset of `|mean
per-step reward|` (the leaderboard metric, lower is better). Generation 0 also
evaluates the incumbent default vector as `idx=-1, kind="baseline"`.

Outputs in `results/tune/<tag>/`: `evals.jsonl` (gen, idx, params, per-scenario
mean reward, objective, wall s), `es.pkl` (pickled CMA state), `state.json`
(space signature, subset ids, flags), `best.json`, `trace.csv` (`gen,best,
best_gen,median,sigma,baseline,wall_s`), `records/` (raw per-scenario JSON).
`--resume` drops the rows of an interrupted generation before redoing it, so no
generation is logged twice.

Robustness: a scenario that raises, dies, or returns a non-finite reward scores
`PENALTY = 1.0` and is logged in the eval row; a wedged job costs one
`--job-timeout` (default 1800 s) and no run dies with it. Ctrl-C saves
`es.pkl`/`state.json`/`trace.csv` and exits 130.

### Running a campaign on Kaggle kernels (`--backend kaggle`)

Each generation's λ×S `(candidate, scenario)` jobs are pushed as one wave of
shard kernels instead of running here: the tasks are split by zone count into
`--shards` kernels (default 5 = the measured concurrent-session limit), each
kernel runs `--shard-workers` processes (default 4 = the vCPUs of a session),
and `--wave-timeout` (default 3600 s) bounds one generation. The task dicts --
and so `record_name`, `params`, `evals.jsonl`, `trace.csv`, `best.json` -- are
identical to the local backend's; each task carries its own `params` and
`sbx.evalrun._worker` applies them as `PolicyCls.PARAMS` exactly as locally.

```bash
starter-kit/.venv/bin/python -m sbx.tune \
    --policy policies.ctrl_w:CtrlW --space sbx/spaces/ctrl_w.json \
    --set gen_train --subset 8 --steps 2880 --sim corrected \
    --popsize 10 --generations 30 --tag tune_w \
    --backend kaggle --shards 5
```

The payload (runner code, `policies/`, patches, scenario/weather bundles) is
published as the private dataset `mors1337/sbx-payload` on the first wave and
its **content hash is pinned in `state.json`**, so later waves and `--resume`
run the shipped code even if `policies/` is edited meanwhile; resuming with a
different hash is refused unless `--repin` is passed. A shard that dies or times
out is re-pushed once, then its jobs score `PENALTY` and the generation is
logged as failed. `--validate` still runs locally (`sbx.evalrun`).

Measured on a 2-generation demo (CtrlW, 18 dims, `gen_train --subset 2`,
288 steps, `--popsize 4 --shards 2 --shard-workers 4`): 10 then 8 tasks in 2
kernels, kernels busy 236 s / 283 s, generation wall 308 s / 1269 s (the second
includes a one-off 16-min stall in `kaggle kernels output`), best objective
0.002432 vs 0.002571 baseline, no failed jobs. Re-evaluating the winner locally
gives the same `total_reward` to 1e-8 relative (comfort bit-identical;
energy/carbon differ in the last digits between arm64 and Kaggle's x86-64).

Cost per scenario-eval, measured here (M3, `--sim corrected`, CtrlV1 defaults):
~20 s of fixed setup per process plus ~1.5 ms of wall time per zone per
5-minute step. Two measured points: 78-zone `building_gen_002/phoenix_g4`,
0.117 s/step between 144 and 288 steps; 44-zone
`building_hold_000/minneapolis_g1`, 0.0712 s/step over 8640 steps (615 s wall).
A 2880-step eval therefore costs ~230 s (49 zones) to ~380 s (84 zones), ~306 s
for the mean of the `--subset 8 --subset-seed 0` gen_train subset: ~80
scenario-evals/hour at `--workers 7` (66-109 depending on building size), and
one generation of λ=10 x S=8 ≈ 58 min. `--components hook` is the default here:
identical `total_reward` to `parity` (checked on 288 steps) and ~1.5% cheaper.

Two caveats from the smoke runs. (1) Scenario strips start on a Sunday 06:00, so
a short horizon with no `--start-offset-days` can contain no occupied steps at
all: every reward is then exactly 0 and the search has no gradient (`--subset 2
--steps 144` on `gen_train` does this). Use `--steps 2880` for real runs, or
`--start-offset-days 1` for cheap weekday-horizon experiments. (2) Subset
selection is deterministic per `--subset-seed` and stratified by climate station
and building size (manifest zone count when known, else floor-plan volume), but
it is drawn from *whichever set you pass* — train on `gen_train`, report on
`gen_hold`/`ext_hold`.
