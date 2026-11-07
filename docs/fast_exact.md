# FastExact — exact pure-Python speedups for local runs (`--fast`)

Deliverables: `sbx/fastpatch.py` (all patches), `--fast` in `sbx/evalrun.py` and
`sbx/tune.py`, `--fast` section in `sbx/README.md`. Nothing under `sims/` or
`starter-kit/` is edited; every patch is a class-attribute monkeypatch installed
right after `sbx.simload.use_sim`, i.e. before the environment exists. `--fast`
off imports nothing from `fastpatch` (verified below by reproducing the
pre-patch baselines bit-for-bit).

## What is patched

1. **Air-side Woodbury response** (`FiniteDifferenceSolver._add_air_side`). The
   VAV coupling was already a Woodbury update on the existing factorization but
   still paid one triangular solve per step to get `A⁻¹(Pq)`. `A⁻¹P` is constant
   per factorization and the same blocked right-hand sides are already solved to
   build `_sensed_response`, so the dense response is cached next to it,
   transposed as `k×n` float64, and the per-step cost becomes `power @ response`
   (2.7 ms/step against 19.5 ms for the triangular solve it replaces on
   `building_train_2`; row-major layout measured 6.4 ms). Cache invalidation is
   exactly the shipped `_sensed_response = None` site, so a new factorization or
   a new coupling rebuilds it.
2. **Zone-average temperatures** (`FloorPlanBasedBuilding.get_zone_average_temps`),
   called 3–4× per step: memoized on the identity of `building.temp` (the solve
   installs a fresh array every step; `apply_convection` is wrapped to drop the
   memo because it mutates in place), each zone mean computed as `np.mean` on its
   gathered values — the shipped values in the shipped order, bit-identical.
3. **Schedule / timezone memoization**: `SetpointSchedule.is_comfort_mode`,
   `get_temperature_window`, `_localize_or_convert_timezone`, `is_weekend` and
   occupancy `ZoneOccupant._to_local_time`, keyed on `(timestamp.value, tz)`;
   per-instance where the answer depends on schedule parameters, one dict shared
   by every zone otherwise (85 zones × N timestamps → N pytz conversions).
4. **Reward-info static data** (`_get_zone_reward_infos`,
   `_get_air_handler_reward_infos`): per-zone `room_areas` dicts, VAV zone lists
   and area shares, setpoint windows hoisted into a per-simulator plan; the
   per-zone infos of a step are reused by the `--components parity` second call
   (pure function of the temperature field and timestamp).
5. **Gather/assembly CPU**: `FiniteDifferenceSolver.solve` uses a cached integer
   index instead of a boolean mask and accumulates in place; `apply_thermal_power_zone`
   writes through a precomputed flat index (3x cheaper), with a fallback for a
   non-C-contiguous `input_q`.

## Exactness verdict (against the project standard)

Standard: ≤1e-12 relative per temperature component along a closed-loop episode,
≤1e-9 relative on the episode total reward.

| tree | scenario | 8641-step total, unpatched = `--fast` | \|Δ total\| | rel | max \|ΔT\| CVs | max \|ΔT\| zone means | verdict |
|---|---|---|---|---|---|---|---|
| shipped | `building_train_8/fairbanks_8` | -208.819351931546 | 0 | 0 | 2.27e-13 K | 5.7e-14 K | pass |
| shipped | `building_train_1/atlanta_2` | -27.731165579042 | 0 | 0 | 2.27e-13 K | 0 | pass |
| shipped | `building_train_2/phoenix_6` | -62.203300826744 | 0 | 0 | 2.27e-13 K | 1.1e-13 K | pass |
| corrected | `building_train_2/atlanta_3` | -24.115765109600 | 0 | 0 | 2.84e-13 K | 5.7e-14 K | pass |

Per-step reward is bit-identical at every step on all four legs (max |Δ reward/step| = 0).

Trajectory check (not just the final field), `scratch/FastExact/trace_pair.py` +
`compare_trace.py`, comparing every control volume at sampled steps:

| tree | scenario | steps / sampling | max \|ΔT\| | max relative per component | bound |
|---|---|---|---|---|---|
| shipped | `train_8/fairbanks_8` | 1000, every step | 2.84e-13 K at step 358 | **1.0e-15** | 1e-12 |
| corrected | `train_2/atlanta_3` | 800, every 8th | 3.41e-13 K at step 736 | **1.2e-15** | 1e-12 |

The deviation saturates instead of growing: max |ΔT| is 0 up to step ~400, then
1.7e-13 K and flat through step 1000 (relative 5.8e-16 → 5.9e-16), which is the
last-bit noise of replacing one triangular solve with a matvec on the same
factors. Margin to the standard is ~1000x.

`--fast` off is unchanged: the unpatched campaign legs reproduce
`results/ctrw_all` (a pre-`fastpatch` CtrlW run) bit-for-bit on all three
shipped scenarios (`total_reward` identical as a Python float, components sums
identical).

`--fast --cenv all` is **bit-identical to `--cenv all`** (max |ΔT| = 0.0 K over a
400-step CtrlW episode, identical total, 4-way check on `train_8`): `cenv`'s
`thermal` component owns `FiniteDifferenceSolver.solve` wholesale, so the air-side
Woodbury patch is inert under that combination (nothing is allocated and the
dense response is never built).

## Speed

Per-step CPU, single process, CtrlW, 300 steps with the first 40 dropped,
conditions interleaved per repeat (plain → `--fast` → `--cenv all` →
`--fast --cenv all`), median of 5. `uptime` load average 7 → 18 on 8 cores while
this ran (other agents), which inflates absolute ms ~2x; ratios are the
load-robust part. Harness: `scratch/FastExact/bench_ab.py`, log `bench_ab.log`.

| scenario | plain | `--fast` | `--cenv all` | `--fast --cenv all` | fast/plain | fast+cenv / cenv |
|---|---|---|---|---|---|---|
| shipped `train_8/fairbanks_8` | 29.38 ms | 17.85 ms | 18.58 ms | 15.96 ms | 1.65x | 1.16x |
| shipped `train_2/phoenix_6` | 77.82 ms | 49.23 ms | 56.65 ms | 56.85 ms | 1.58x | 1.00x |
| corrected `train_2/atlanta_3` (largest) | 91.70 ms | 58.28 ms | 79.00 ms | 59.92 ms | 1.57x | 1.32x |

Same harness on a quiet machine earlier (5 alternating runs per condition,
`bench_final2.log`): `train_8/fairbanks_8` 13.22 → 7.82 ms (1.69x),
`train_1/atlanta_2` 25.13 → 13.04 ms (1.93x), `train_2/atlanta_3`
42.39 → 24.64 ms (1.72x). Episode totals agree in every condition and every
repeat of the 4-way benchmark (300 steps: -2.831134654, -0.674202979,
-0.128751801).

### What limits it

On `building_train_2` the patched step is 24.6 ms (quiet machine) of which
9.1 ms is the single sparse triangular solve per step that no patch can remove;
the air-side solve it replaced cost 9.3 ms and is now a 2.7 ms matvec. The
remainder is numba convection, the occupancy Markov chains (they consume the
shared `RandomState`, so they cannot be reordered or skipped) and protobuf
assembly in the observation/reward paths — exactly what `--cenv` ports to C.
Hence ~1.6-1.9x, not 2x+; the acceptance bar "≥2x on the largest building" is
not met and the measured reason is above, not an unexplored patch.

## Cost and caveats

* Memory: the dense response is `k×n` float64 — 164 MB on `building_train_2`.
  Measured peak RSS on a 60-step `phoenix_6` run: plain 1076 MB, `--fast`
  1246 MB (+170 MB), `--cenv all` 1449 MB, `--fast --cenv all` 1442 MB (no
  extra memory: the dense response is never built when `cenv`'s `thermal` owns
  `solve`). The +170 MB is per worker, so `--workers 6` on the largest buildings
  wants ~1 GB more; `fastpatch.apply(..., dense_air_side=False)` keeps every
  other patch and restores the shipped per-step triangular solve.
* `exact_zone_means=False` is an option (one sequential segment sum, ~1 ms/step
  cheaper) but it is **not** the default and changes temperatures by ~1e-13 K;
  the default (`True`) is the bit-identical `np.mean` per zone.
* Only local runs: `sbx.tune --backend kaggle` refuses `--fast`, and the
  competition/official path never imports `fastpatch`.
* Two workers that share a scenario tree still each build their own cache; there
  is no cross-process sharing.

## How to run

```bash
starter-kit/.venv/bin/python -m sbx.evalrun --policy policies.ctrl_w:CtrlW \
    --set dev30 --sim shipped --fast --workers 6 --tag ctrlw_fast
starter-kit/.venv/bin/python -m sbx.tune --policy policies.ctrl_w:CtrlW --tag t --fast
```

## Reproducing the evidence (scratch, git-ignored)

```bash
# exactness campaign: unpatched vs --fast, 8641 steps, both trees
starter-kit/.venv/bin/python scratch/FastExact/campaign.py --concurrency 2 --steps 8641
#   records/  campaign_final2.log  (skips finished legs; delete a record to redo it)
# trajectory-level exactness (per-step, per-component)
starter-kit/.venv/bin/python scratch/FastExact/trace_pair.py --sim shipped \
    --scenario building_train_8/fairbanks_8 --steps 1000 --every 1 --out /abs/t8_base.npy
starter-kit/.venv/bin/python scratch/FastExact/trace_pair.py ... --fast --out /abs/t8_fast.npy
starter-kit/.venv/bin/python scratch/FastExact/compare_trace.py /abs/t8_base.npy /abs/t8_fast.npy
# interleaved 4-condition timing
starter-kit/.venv/bin/python scratch/FastExact/bench_ab.py --steps 300 --reps 5
```
