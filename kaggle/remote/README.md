# kaggle/remote/ — evaluating `sbx` runs on Kaggle kernels

Private Kaggle CPU kernels are free extra compute: **4 vCPU / 32 GB** each, and
**five sessions run concurrently** for `mors1337` with no queueing (measured, see
"Measured" below). This directory turns one `sbx` evaluation into N shards, one
kernel each, and merges the results back into a normal `results/<tag>/`
directory — same per-scenario JSONs, same `summary.json`, produced by
`sbx.evalrun._summarise`, so scores come from the local code.

```bash
# the 30 shipped scenarios, 3 shards, 30-day episodes
starter-kit/.venv/bin/python -m kaggle.remote.remote_eval \
    --policy policies.ctrl_v1:CtrlV1 --set dev30 --steps 8640 --sim shipped \
    --shards 3 --tag remote_test

# same, and diff the totals against a local run afterwards
... --compare results/<local_tag>

# corrected simulator, a generated pool scenario
... --scenario building_gen_000/fairbanks_g0 --sim corrected --steps 288
```

## Files

| path | role |
|---|---|
| `payload.py` | builds `build/payload/` (runner code + policies + `cenv/` sources + patches + pool + weather) and publishes it as the private dataset `mors1337/sbx-payload`; content-hashed, versioned only when the hash changes |
| `kernel_gen.py` | generates one self-contained `run.py` + `kernel-metadata.json` per shard; the job spec is embedded as a JSON block |
| `patch_apply.py` | pure-Python unified-diff applier (the image has no `git`/`patch` need); reproduces `sims/corrected` byte-for-byte from the mount |
| `remote_eval.py` | driver: sharding, push, poll, download, merge; `submit_jobs` / `collect` API for `sbx/tune.py` |
| `build/` | local scratch (gitignored): `payload/`, `kernels/<slug>/`, `state.json` = published hash + dataset version |

## What a shard kernel does

1. Finds the payload dataset (`/kaggle/input/sbx-payload`, or the newer
   `/kaggle/input/datasets/<owner>/<slug>` layout) and verifies every file
   against the SHA-256 in `payload_info.json`; a *stale* dataset aborts at t+20s
   instead of producing numbers from the wrong code.
2. Copies `smart_buildings` off the **competition mount** into
   `payload/sims/shipped` — remote "shipped" is the competition's own code.
3. Copies that to `payload/sims/corrected` and applies
   `patches/pr2_*.diff` + `pr3_*.diff`. Patch-created files must be
   byte-identical to `sims/corrected`; patched files that differ *outside* the
   hunks are reported (the mount snapshot can lag our pinned `sims/shipped`).
4. Materialises the 24 generated pool buildings (`building_gen_*`,
   `building_hold_*`), the `ext_hold` scenarios that `sbx/weather_ext.py` adds
   under shipped buildings (`building_train_1/lasvegas_3`,
   `building_train_8/london_1`; the mount has no copy of them) and the external
   weather presets into both trees.
5. With `--cenv`/`--gate`, compiles `payload/cenv` in place first
   (`python -m cenv.build`: gcc/clang, `-O3 -ffp-contract=off -fno-fast-math`;
   a failure aborts the shard rather than falling back to Python) and records
   the compiler banner, flags and library size in `env.json → cenv_build`.
6. Runs `python -m sbx.evalrun --scenario <ids> --workers 4` as a subprocess
   (spawn needs a real script; Kaggle runs script kernels through nbconvert).
7. Writes `/kaggle/working/results/<tag>/*.json`, `shard.json`, `env.json`,
   `eval.log`, then deletes the payload + sim trees so the kernel output is a
   handful of files (`--keep-trees` keeps them for debugging).

## The C step (`--cenv fused --fast`)

Off by default: a job without `--cenv` runs the same plain-Python simulator as
before. `--cenv <spec>` ships `cenv/`'s Python, `.c` and `.h` files in the
payload (no `_build/`, so the payload is compiler- and platform-neutral),
compiles `libcenv.so` inside the kernel, and adds `--cenv <spec>` to the
`sbx.evalrun` command line; `--fast` adds `--fast` the same way. The pairing
that matters is `--cenv fused --fast`: the whole `env.step` (sim step,
observation, reward info) in one C call plus `sbx.fastpatch`'s pure-Python
memoisation.

```bash
# dev30, 5 shards, C step, merged into results/<tag>/ exactly like a plain run
starter-kit/.venv/bin/python -m kaggle.remote.remote_eval \
    --policy policies.ctrl_w5:CtrlW5 --params '{"standby_minimum": 278.0}' \
    --set dev30 --steps 8641 --sim shipped --shards 5 \
    --cenv fused --fast --tag remote_dev30_c

# the same job without --cenv/--fast is the plain-Python reference
starter-kit/.venv/bin/python -m kaggle.remote.remote_eval \
    --policy policies.ctrl_w5:CtrlW5 --params '{"standby_minimum": 278.0}' \
    --set dev30 --steps 8641 --sim shipped --shards 5 --tag remote_dev30_py
```

`--gate` replaces the evaluation with the correctness gate: for every scenario
and every sim tree it runs two episodes of the gate policy in independent
processes — pure Python and `--cenv` — and compares the traces with
`cenv/verify/gate.py compare` *inside the kernel* (per-step reward, per-zone
temperatures, per-step CPU). The gate passes when `total_reward` agrees to
`--gate-rel-tol` (default 1e-9) and max `|dT_zone|` to `--gate-temp-rel-tol`
(default 1e-12) times the trace's own temperature scale (max `|T|`), because
`gate.py compare` takes an absolute K tolerance and the criterion is relative. It writes `gate.json` (merged, with `passed`) instead of
`results/`/`summary.json`. `--gate-steps` (default 600), `--gate-sims`
(default `shipped,corrected`) and `--gate-policy` (default
`policies.ctrl_w:CtrlW`) tune it.

```bash
starter-kit/.venv/bin/python -m kaggle.remote.remote_eval --gate --shards 1 \
    --scenario building_train_8/atlanta_1 building_train_2/phoenix_6 \
    --gate-steps 600 --cenv fused --tag remote_gate
```

## Sharding

`--shards N` splits scenarios by **zone count** (parsed from each scenario's
`config.gin` `ahu_*_rooms` lists), largest first, because step cost tracks zones.
Kernels are pushed in waves of at most `--max-concurrency` (default 5 = the
measured limit); the next shard is pushed as soon as a session finishes.

## API (for `sbx/tune.py`)

```python
from kaggle.remote.remote_eval import submit_jobs, collect

handle = submit_jobs([
    {"policy": "policies.ctrl_v1:CtrlV1", "params": {"kp": 0.4},
     "set": "dev30", "steps": 2880, "sim": "corrected", "tag": "tune_0042", "shards": 2},
])
result = collect(handle)             # {"summaries": {tag: summary}, "records": {...}, "shards": [...]}
summary = result["summaries"]["tune_0042"]
```

* Several jobs per handle are supported; each job's records and `summary.json`
  land in `results/<handle tag>/<job tag>/` (a single job writes
  `results/<tag>/` directly).
* `collect("results/<tag>/handle.json")` resumes from a handle on disk without
  re-pushing, and `collect(..., repush=True)` re-runs the shards (`--retry`).
* `--payload-hash <hash>` pins an already published payload instead of
  rebuilding; `summary["remote"]["local_sources_match_payload"]` then reports any
  local `sbx/`/`policies/` file that drifted from the shipped copy (peers edit
  them while runs are in flight).

## The CMA-ES tuner (`sbx.tune --backend kaggle`)

`sbx/tune.py` gains `--backend kaggle` (plus `--shards`, default 5;
`--shard-workers`, default 4; `--wave-timeout`, `--max-concurrency`, `--repin`).
Each generation's λ×S `(candidate, scenario)` tasks become one wave:

```python
handle = submit_tasks(tasks, tag=f"{campaign}_g{gen:03d}", policy=..., steps=...,
                      sim=..., components=..., start_offset_days=...,
                      shards=5, workers=4, job_timeout=1800)
result = collect(handle, allow_partial=True, retries=1)
records = result["task_results"][handle["kernels"][0]["job_tag"]]  # {job_name: record}
```

`tasks` are the tuner's own task dicts (`record_name`, `scenario`, `params`, run
flags); the kernel fills `config_path`/`out_dir` and runs
`evalrun._worker` in a spawn pool (`tools/remote_tasks.py`, shipped in the
payload), so `params` reach the policy as `PolicyCls.PARAMS` exactly like the
local path and every record keeps the tuner's job name. There is no `summary`
for task waves -- a summary across candidates is meaningless -- so
`collect` returns the records keyed by name and skips `_summarise`.

The campaign pins `handle["payload_hash"]` in `results/tune/<tag>/state.json`;
`--resume` rebuilds the payload locally, compares hashes and refuses to continue
if our code moved (override with `--repin`). A shard that dies is re-pushed once
(`retries=1`), then its tasks are missing and the tuner scores `PENALTY`.

Each task gets its **own** scratch directory. `evalrun._shifted_config` caches
the offset-shifted `config.gin` under `out_dir/_gin/<building>__<weather>__off<N>.gin`,
which does not depend on the candidate: candidates of one generation evaluating
the same scenario concurrently read a half-written file and die with
`In call to configurable 'HybridActionEnvironment'` (2 of 18 tasks, measured).
`tools/remote_tasks.py` writes each task into `out_dir/_task_<record stem>/` and
moves the records back, which closes the window; the same hazard still exists in
the *local* backend, which shares one `records/` directory (`sbx/tune.py`'s
`_build_tasks` sets `out_dir=records_dir` for every task).

### Tuner on Kaggle, measured (`tune_remote_demo`, 2026-11-05)

`CtrlW`, `sbx/spaces/ctrl_w.json` (18 dims), `--set gen_train --subset 2
--steps 288 --start-offset-days 1 --popsize 4 --generations 2 --shards 2
--shard-workers 4`:

| generation | tasks | kernels | kernel wall | generation wall |
|---|---|---|---|---|
| 0 (incl. baseline) | 10 | 2 | 236 s | 308 s (first dataset push + 35 s readiness grace included) |
| 1 | 8 | 2 | 283 s | 1269 s |

The gen-1 number is dominated by one transient `kaggle kernels output` stall of
~16 min (the same download re-run took 5 s); the kernels themselves took 283 s
with both sessions `RUNNING` the whole time. 18/18 tasks returned records, no
`PENALTY`, best objective 0.002432 (gen 1) vs 0.002571 baseline, and the local
re-evaluation of the winner on the same two scenarios agrees to **1.0e-8
relative** in `total_reward` (comfort component bit-identical, energy/carbon
1e-8 — see "Validation": the same cross-architecture float64 effect that shows
up at 8640 steps; local-vs-local and remote-vs-remote are both bit-identical).

## Local rehearsal (no Kaggle, no cost)

The generated kernel honours three env vars, so the exact same file runs
against `comp_data/` before anything is pushed:

```bash
SB_REMOTE_WORK=/tmp/kwork \
SBX_PAYLOAD_DATASET_DIR=/tmp/kds/sbx-payload \
SB_COMPETITION_DATA_FOLDER=$PWD/comp_data \
starter-kit/.venv/bin/python kaggle/remote/build/kernels/<slug>/run.py
```

`/tmp/kds/sbx-payload` is a copy of `build/payload/`; keep `pool/scenarios.zip`
there for the zip path, or expand it into `pool/scenarios/` to mimic what Kaggle
stores. This rehearsal caught four bugs that would each have burned kernel runs
(`--scenario` repeated instead of one `nargs="+"` flag, the patch path, the
expanded-archive layout, and pool resolution when the dataset mounts off the
documented path).

## Kaggle behaviour this had to work around

* **Uploaded archives are expanded.** `pool/scenarios.zip` arrives on the mount
  as `pool/scenarios/<building>/<scenario>/...`, so the archive's raw bytes are
  not stable; the payload hashes and verifies the *contents* instead
  (digest for the small `config.gin`/`config.yaml`, size for the 2.7 MB `.npy`
  grids, which are read through symlinks rather than copied).
* **Kernel output listings are paginated at 500 files** and the CLI only reads
  the first page. A `/kaggle/working` holding the payload + two sim trees (~1.3 k
  files) therefore *loses* `shard.json` on download. Hence the end-of-run
  cleanup plus a `--file-pattern` download, plus a driver fallback that rebuilds
  a shard from the per-scenario JSONs if `shard.json` ever goes missing.
* **A new dataset version takes ~a minute to become the mounted one.** "ready"
  from `kaggle datasets status` is not enough: the driver waits a further 35 s
  and, if a shard still reports a stale payload, waits 60 s and re-pushes it.
* **Script kernels run through nbconvert**, so `__main__` has no usable
  `__file__` for `multiprocessing`'s spawn; the evaluation therefore runs as
  `python -m sbx.evalrun` in a child process.
* Each kernel version is a new session; a re-run of the same tag re-pushes the
  same kernel (version 2, 3, ...) rather than creating new ones.

## Measured (2026-11-05, account `mors1337`)

Image: Python 3.13.18, 4 vCPU, 32.9 GB RAM, internet off.

| what | value |
|---|---|
| push → first log line | ~15 s |
| shard setup (payload verify, both sim trees, patches, pool, weather) | **17–26 s** (shipped copy 2.7 s, corrected+patches 0.1 s, pool+weather 4.2 s, payload verify 12–19 s) |
| session wall after the eval | +10 s (nbconvert) |
| concurrency | **5 of 5** kernels `RUNNING` at once, no `PENDING`; a 6th was not tested |
| per-step cost | 30-zone building 0.088 s/step, 39-zone 0.134 s/step, 84-zone pool building 0.26 s/step (wall, one scenario per session) |
| throughput | **4.8 steps/s per Kaggle CPU-second**; 17 280 steps (dev30 × 576) in 291 s wall over 5 shards, 3 578 CPU-s |
| Kaggle vs local CPU | **2.4×** — one vCPU ≈ 0.41 local core, so 5 sessions ≈ 8 local cores |
| dev30 @ 576 steps | remote 291 s wall / 3 578 CPU-s vs local 786 s wall / 1 471 CPU-s on 2 workers |
| full 8640-step episode (30-day) | 39-zone `building_train_1/atlanta_2`: **812 s** eval (0.094 s/step), 801 CPU-s; 30-zone `building_train_8/phoenix_4`: **497 s** eval (0.058 s/step), 487 CPU-s — both on one vCPU of their own session |
| 2 full episodes in parallel | 896 s wall push→collected (two kernels, two sessions) vs 427 s for the same two on 2 local workers |

### The C step, measured (2026-11-07, image Python 3.13.18 / numpy 2.1.3 / numba 0.61.2)

Payload `209362142bc8fe4e5a375cc8af4c1cf4` carries `cenv/` (46 files, 13.6 MB
payload total, no `_build/`). Kaggle has **gcc 13.3.0**; the in-kernel build
takes **4.8 s** and produces `libcenv.so` of 116 728 bytes with
`-O3 -std=c11 -fPIC -ffp-contract=off -fno-fast-math -Wall -Wextra
-Wno-unused-parameter -D_GNU_SOURCE`. Shard setup including that build: 23 s.

| what | value |
|---|---|
| CtrlW5-278, 5 dev30 scenarios × 8641 steps, `--cenv fused --fast` | **2 291.6 CPU-s / 43 205 steps = 53.0 ms/step** |
| same, plain Python | **4 378.8 CPU-s = 101.4 ms/step** |
| speedup | **1.91×** CPU; per scenario 1.84–2.18× (67.6→35.4 ms/step atlanta_8 … 131.0→71.0 singapore_3) |
| wall for both runs | 1 228 s for 4 kernels in 2 waves of 2 (plain wave then C wave) |
| gate shell, `cenv/verify/gate.py`, 600 steps | 30-zone `building_train_8/atlanta_1` 300.6 → 131.0 ms/step (2.29×); 85-zone `building_train_2/phoenix_6` 550.1 → 485.1 ms/step (1.13×) |

## Validation

* **shipped, 2 shards × 2 scenarios, 576 steps** — `results/remote_parity` vs
  `results/local_ref_576`: `identical: true`, `max_abs_delta: 0.0`.
* **dev30, 5 shards, 576 steps (all 30 scenarios)** — `results/remote_dev30_576`
  vs `results/local_dev30_576`: `identical: true`, `max_abs_delta: 0.0`,
  `leaderboard_score` 0.0288583… both.
* **corrected, 1 scenario, 288 steps** — `results/remote_corrected` vs
  `results/local_gen_corrected`: identical (`total_reward`
  -37.701378152589314).
* **8640 steps × 2 scenarios** — `results/remote_8640/{full_atlanta,full_phoenix}`
  vs `results/local_8640`: phoenix_4 bit-identical (-1384.2102924751061);
  atlanta_2 differs by **3.7e-9** absolute (1.7e-11 relative) in `total_reward`,
  and the per-component sums shift by the same order. That is float64 rounding,
  not a harness difference: the trajectories agree for 576 steps exactly, and
  the simulator's only compiled kernel
  (`stochastic_convection_simulator.py`, `@numba.njit`) is code-generated
  per architecture — NumPy/Numba on arm64 (local) and on Kaggle's x86-64 Xeon
  differ in the last bits, which the 30-day thermal integration amplifies.
  Repeating the same 8640-step run on Kaggle (`results/remote_8640_repeat`)
  reproduces the remote number exactly.
* **C step equals plain Python, on Kaggle, same machine** — `--gate` on
  `building_train_8/atlanta_1` (30 zones) and `building_train_2/phoenix_6`
  (85 zones), both trees, 600 steps, CtrlW: `total_reward` is **bit-identical**
  (relative difference 0.0, four for four) and max `|dT_zone|` is 2.8e-13 …
  3.0e-12 K, i.e. **9.6e-16 … 9.7e-15 relative** to the trace's temperature
  scale — five orders inside the 1e-12 limit. `results/remote_gate_cenv/gate.json`.
* **CtrlW5-278, full 8641-step episodes, 5 dev30 scenarios, Kaggle plain vs
  Kaggle `--cenv fused --fast`** — per-scenario `total_reward` identical to the
  last bit in all five (fairbanks_1 −386.906852069, phoenix_5 −123.044051206,
  singapore_3 −59.061950787, atlanta_8 −57.463043976, phoenix_4
  −83.003446054); `leaderboard_score` 0.016421232359491233 both.
  `results/remote_c5/{remote_c5_plain,remote_c5_cenv}`.
* **Kaggle vs this Mac is *not* bit-identical, C step or not** — the same five
  totals differ from `results/ctw5frost_dev30_shipped` (Mac, `cenv fused`) by
  3.3e-8 … 2.6e-6 relative, and a PC (Linux x86-64) run of the CtrlW anchor
  differs from the Mac by 5e-6 relative while *its own* C and plain paths agree
  exactly. BLAS/libm/SIMD round-off between machines, amplified by the closed
  loop and the near-threshold comfort term. So exactness is only meaningful
  against a same-machine reference: use `--gate`, never a Mac baseline.
* The mounted simulator code differs from our pinned `sims/shipped` in 23 of 96
  `.py` files (comments plus the PR #1 tracker fix we carry), and `shard.json`
  records exactly which; corrected (`mount + patches`) still reproduces our
  `sims/corrected` totals bit-for-bit, and shipped reproduces `sims/shipped`
  bit-for-bit for all 30 scenarios measured at 576 steps.

## Limits

* Kernels are **private** and never submitted to the competition; the driver
  never calls `kaggle competitions submit`.
* Five concurrent CPU sessions; more shards queue and the driver fills slots as
  sessions free up. Every session is billed as ~4 CPU-hours per hour of wall
  clock against the account's weekly CPU quota.
* `kaggle kernels output` preserves `/kaggle/working` paths, so
  `results/<tag>/*.json` arrive intact.
* A shard that dies during setup writes `/kaggle/working/error.json`; the driver
  downloads it and puts the traceback and log tail into the error message.
