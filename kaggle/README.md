# kaggle/ — submission pipeline

Everything needed to turn one of our policy modules into a Kaggle code-competition
submission, test it locally, push it, and submit it.

The rerun container has **no internet** and **no import path into this repository**,
so the notebook must carry the policy itself. `build_notebook.py` does that: it
embeds the policy module (and any sibling module it imports from the same
directory) zlib-compressed and base64-encoded — the kernels API rejects a notebook
over ~1 MiB with a bare `400 Bad Request`, and `wu_weights.py` alone is most of
that — verifies the SHA-256 of each decompressed source at import, writes the files to
`/kaggle/working/sbx_policy_modules/`, puts that directory on `sys.path`, and then
runs the inference server exactly like the official starter notebook.

The account allows 5 concurrent batch CPU sessions; a push beyond that fails with
`Maximum batch CPU session count of 5 reached`. Remote-eval queues therefore run at
`--max-concurrency 4`, keeping one session free for submission notebooks.

## Quick start

```bash
# 1. build (metadata + notebook are written into kaggle/build/<slug>/)
starter-kit/.venv/bin/python kaggle/build_notebook.py \
    --policy policies/ctrl_v1.py:CtrlV1 --slug sbx-ctrl-v1
#    extra flags: --param key=value (PARAMS override), --title "...",
#    --smoke-steps N (test builds only: truncate episodes, never submit)

# 2. local pre-flight: runs the generated notebook's code cells against comp_data/
#    (single public scenario, episodes truncated to 30 steps)
starter-kit/.venv/bin/python kaggle/run_local.py \
    kaggle/build/sbx-ctrl-v1/submission.ipynb --max-steps 30

# 3. push (private kernel, CPU, internet off), then poll until COMPLETE
kaggle/push.sh sbx-ctrl-v1
kaggle/status.sh sbx-ctrl-v1
kaggle/logs.sh sbx-ctrl-v1            # log + submission.parquet -> kaggle/build/<slug>/output/

# 4. submit the version that scored well (this is the only step that consumes quota)
kaggle/submit.sh sbx-ctrl-v1 3 "ctrl-v1 occupancy-aware setpoints"
```

## The exact submission command

Verified against `kaggle competitions submit -h` (CLI 2.2.1, `--help` output of
`-k/-v/-f` quoted in `submit.sh`):

```bash
kaggle competitions submit neurips-2026-smart-buildings \
    -k mors1337/<slug> -v <version> -f submission.parquet -m "<message>"
```

* `-k` is the kernel in `owner/slug` form, `-v` the version number (1-based, the
  order `kaggle kernels list -m` reports).
* `-f` for a code competition is the *name of the output file the kernel
  produces*, not a local upload — the gateway writes `submission.parquet` itself
  (`{row_id, reward}`, one row per scenario), so nothing is uploaded.
* The competition slug is pinned inside the script so a stale
  `kaggle config set competition` cannot redirect it.
* Submissions are reserved for real controllers; the test kernel below is never
  submitted.

`kaggle/submit.sh <slug> <version> "<message>"` is the wrapper. It refuses to
submit a notebook whose local build is marked `TEST BUILD` (i.e. built with
`--smoke-steps`), unless `SB_ALLOW_TEST_SUBMIT=1` is set.

## Files

| path | role |
|---|---|
| `build_notebook.py` | policy module -> `kaggle/build/<slug>/{submission.ipynb,kernel-metadata.json}` |
| `run_local.py` | execute a built notebook against `comp_data/` with truncated episodes |
| `placeholder_policy.py` | zero-action reference policy (official `TransferPolicy` behaviour) for pipeline tests |
| `push.sh` / `status.sh` / `logs.sh` | `kaggle kernels push` / `status` / `output` wrappers |
| `remote/` | remote evaluation runner: shards `sbx.evalrun` jobs over several kernels and merges the results (`remote/README.md`) |
| `submit.sh` | `kaggle competitions submit ... -k -v -f submission.parquet` |
| `.pydeps/` | local-only runtime packages for `run_local.py` (never uploaded) |
| `build/<slug>/` | build output; `build/<slug>/output/<version>/` holds downloaded logs and `submission.parquet` |

## What the builder puts in the notebook

1. **Markdown cell** — slug, policy file/class, SHA-256 of every embedded module.
2. **Setup cell** — resolves `COMPETITION_DATA_FOLDER` (real mount path;
   `SB_COMPETITION_DATA_FOLDER` overrides it for local runs only), puts both the
   mount root and `kaggle_evaluation/` on `sys.path`, materialises the embedded
   modules into `/kaggle/working/sbx_policy_modules/` after digest verification,
   imports the policy class, applies any `--param` `PARAMS` overrides.
3. **Wrapper cell** — `SubmissionPolicy(smart_buildings_inference_server.Policy)`
   delegating to the team class. See below.
4. `kernel-metadata.json` — `mors1337/<slug>`, private, CPU, `enable_internet:
   false`, `competition_sources: ["neurips-2026-smart-buildings"]`, no datasets.

### Why the wrapper exists

The gateway invalidates the *entire* submission on a single bad action: an
out-of-range value raises, a refused action (reward `-inf`) aborts the run, and a
NaN setpoint is not even caught by the simulator's `<`/`>` bounds check. So
`SubmissionPolicy`:

* clips both slots into the scenario's own action space (bounds come from
  `action_spec`, never hardcoded) and `nan_to_num`s them;
* forces `np.float32` / `np.int32` and rejects shape drift;
* if the policy raises, prints the traceback once and then holds the mid-range
  action (zeros / all-devices-on, clipped) for the rest of that scenario instead
  of voiding the submission.

A fresh `SubmissionPolicy` is constructed per scenario by the gateway; there is
no `reset()`.

### Constraints the builder enforces

* The policy module must not import `smart_buildings` (not importable from the
  notebook's own environment) — the build fails loudly if it does.
* Sibling modules are discovered by parsing the policy's AST and following
  imports that resolve to a `.py` file in the same directory, transitively.
* `--param key=value` bakes a `PARAMS` override (JSON-parsed) without editing the
  policy file, e.g. `--param comfort_weight=0.7`.

## Local dry run

`run_local.py` executes the notebook's code cells unmodified with:

* `SB_COMPETITION_DATA_FOLDER=<repo>/comp_data/`,
* `HEARTH_HOST_SMART_BUILDINGS_MAX_STEPS=<--max-steps>` (the gateway refuses to
  run untruncated without a rerun flag),
* the editable install of `smart_buildings` removed from `sys.meta_path`, so the
  simulator that loads is the one on the data mount — the same code the
  submission container imports.

Only the public scenario `building_train_1/atlanta_2` is mounted, so a green run
proves wiring, not runtime for the 30 hidden scenarios.

The workspace venv lacks a few packages the gateway needs. They live in
`kaggle/.pydeps` (not part of any build directory, never uploaded):

```bash
uv pip install --target kaggle/.pydeps --python starter-kit/.venv/bin/python \
    grpcio-tools opentelemetry-api pyarrow polars
```

`run_local.py` puts that directory at the front of `sys.path` automatically
(grpcio-tools' protobuf must win over the venv's older copy).

## Test kernel

`kaggle/build/sbx-pack-test/` is built from `kaggle/placeholder_policy.py`
(`ZeroActionPolicy`, behaviourally identical to the official `TransferPolicy`:
mid-range setpoints, all devices on). It exists to prove push + Kaggle-side
execution; it is private and has never been submitted. It is built with
`--smoke-steps 40` (test builds only): the notebook then sets
`HEARTH_HOST_SMART_BUILDINGS_MAX_STEPS=40` before starting the server, and both
the markdown cell and the build output say "TEST BUILD / do not submit". A real
submission is built without the flag, so episodes run to the length their own
`config.gin` specifies.

Measured: local (M3, `--max-steps 40`) mean reward **-0.1147**, Kaggle v4 mean
reward **-0.114684**, identical `submission.parquet` (`row_id` =
`building_train_1/atlanta_2`, f32 reward). Kaggle-side game loop: 40 steps in
**21 s** (~0.5 s/step, after ~20 s of imports/environment load); the same 40
steps take ~106 s in the local venv (~2.6 s/step), so local timings are roughly
5x pessimistic — use Kaggle for any runtime budgeting.

`kaggle/.pydeps` is ~350 MB of local-only wheels; delete it and reinstall with the
command above if it gets in the way.

## Findings: Kaggle notebooks as batch compute

Probes `mors1337/sbx-probe-a` / `sbx-probe-b` (scratch copies in
`scratch/KagglePack/probe_{a,b}/`, tiny CPU kernels, competition attached).

* **Concurrency**: at least **3 private CPU batch sessions run at once** for this
  account — both probes and the test kernel were `RUNNING` simultaneously and all
  three finished. No queueing was observed. Each session: **4 vCPU, ~31 GB RAM**,
  Python 3.13.5 image, internet off.
* **The competition data is mounted in ordinary batch kernels**, not only during
  the rerun: `/kaggle/input/competitions/neurips-2026-smart-buildings/` contains
  `smart_buildings/` (including `evaluation_scenarios/` and all public
  `training_scenarios/`) and `kaggle_evaluation/`. `sys.path.insert(0, <mount>)`
  then `import smart_buildings` works; `smart_buildings.evaluate` imports in ~7 s
  more.
* That runner now exists: `kaggle/remote/` shards an `sbx.evalrun` job over
  several kernels, rebuilds `sims/shipped` from the mount and `sims/corrected`
  from our patches inside each kernel, and merges the results into a normal
  `results/<tag>/` directory (bit-identical totals; measured 17-26 s of setup
  per shard, 4.8 steps/s per Kaggle CPU-second, 5 sessions at once).
* So a kernel can act as remote compute: push a notebook that imports the mount,
  runs `sbx`-style rollouts with `HEARTH_HOST_SMART_BUILDINGS_MAX_STEPS` (or the
  simulator's own evaluate loop), and writes JSON results to `/kaggle/working`,
  which `kaggle kernels output` downloads. Costs: one kernel version per run and
  the daily notebook-run allowance; 4 vCPU per session, 3+ sessions in parallel.
  No full remote runner is built here (out of scope for this pipeline).
* Simulator throughput measured on that image: **~0.5 s/step** for a 39-zone
  building, i.e. a full 8,640-step scenario is roughly 1.2 h of wall clock and 30
  scenarios in one process would exceed the 9 h rerun budget. Worth planning the
  Round-One controller scope around this.

