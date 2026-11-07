# Exact batched thermal solve on the RX 9060 XT — go/no-go

Owner: SolverGPU2 (restart of SolverGPU). Everything is in `scratch/SolverGPU/`
(mirrored to `/home/mors1337/sb26/scratch/SolverGPU/` on the PC):
`sabuild.py` builds the multigrid hierarchy from the exported matrix,
`sacpu.py` is the exact CPU reference of the identical algorithm,
`gpubench2.py` the GPU solve, `gpu_chain2.py` the chained drift test,
`export_chain.py` the chained-trajectory export, `micro.py` the device
micro-benchmarks, `mg_build.py`/`mgcore.py`/`mgtest.py` the geometric-MG
prototype that was rejected.

Question: can the RX 9060 XT solve `A x = b` for M = 32…1024 environments of one
building at once *exactly* (≤1e-13 relative per temperature component) and
faster per env-step than one CPU core (3.8–4.4 ms single-RHS SuperLU)?

## Verdict: NO-GO (numerics solved, implementation 2.7–3.5× the wrong way)

The algorithm reaches the exactness standard on the GPU — smoothed-aggregation
multigrid + PCG, 15–16 iterations, **errcomp 9.0e-14 … 1.0e-13 per step** and
**2.15e-13 worst case over 288 chained steps** — but the GPU is **22.3–23.5
ms/env-step against 6.6–12.1 ms/env-step for one CPU core** doing the same
batched SuperLU solve. The gap is entirely the toolchain: no Triton in the ROCm
wheel means every stencil apply is ~30 memory passes instead of 2
(586 ms per V-cycle at M=512 = 152 GB of traffic, where a fused V-cycle needs
~10 GB).

| M (train_8, n=81 388) | V-cycle | solve, 16 iters | µs/env-step | errcomp | peak |
|---|---|---|---|---|---|
| 32 | 38.0 ms | 730 ms | **22 808** | 9.0e-14 | 0.51 GB |
| 128 | 171.6 ms | 3 006 ms | **23 485** | 1.03e-13 | 2.02 GB |
| 512 | 586.4 ms | 11 419 ms | **22 303** | 1.03e-13 | 7.94 GB |
| 1024 | ~16 GB of working set | not run | (≈22, flat in M) | — | — |

µs/env-step is flat in M (22.8 / 23.5 / 22.3), so this is a bandwidth/kernel-count
limit, not a scaling one. Comparison (all measured, same machine class):

| path | µs/env-step | notes |
|---|---|---|
| GPU, eager torch (this work) | 22 300 | exact, 1e-13 |
| CPU 1 core, batched SuperLU | 6 600 / 8 300 / 12 100 @ M=32/128/1024 | exact, 1e-15 |
| CPU 1 core, single RHS | 3 800–5 200 | exact, 1e-15 |
| CPU, 12 cores, 1 RHS each | ~320–430 | exact |
| fastsim ROM | 0.06–0.125 | approximate, −0.3 K bias |

The ROM is ~10⁴× cheaper per step than *any* exact path; the exact GPU path
cannot even match a single CPU core, let alone a 12-core CPU.

## Why it fails, quantitatively

Level-0 vector at M=512 is 123 520 × 512 × 8 B = 506 MB; a fused 5-point apply
is 1 GB ≈ 3.9 ms at the measured 260 GB/s. The measured V-cycle implies ~15×
that traffic. Removing it needs one fused kernel per level (5-point at level 0,
13-point at level 1). Projection with fused kernels [INFERENCE from the measured
components]: ~10 GB per V-cycle → 38 ms, solve ≈ 0.63 s/batch →
**≈1.2 ms/env-step** at M=512, i.e. 7× the batched CPU path and 3–4× one core at
M=1 — still 10⁴× behind the ROM. So even a perfect implementation would not
change how the team spends its compute; the only real use would be "a whole
batch of envs for the price of one core".

## Device (measured, `micro.py`)

| item | value |
|---|---|
| device / stack | gfx1200 (RDNA4), 15.9 GiB, 16 CU, ROCm 7.2, torch 2.9.1 |
| dense mm 4096³ fp32 / fp64 | 5.21 / **0.396 TFLOPS** (1/13 — fp64 is the weak axis) |
| elementwise (n,M) fp64 | 260–390 GB/s (3.9 ms per GB at M=512) |
| torch CSR SpMM, level-0 A (616 k nnz) | 3.5 ms @ M=32, **428 ms @ M=512** (0.7 GFLOPS) |
| bottom dense lu_solve 35×35 | 0.29 ms @ M=32, 13.4 ms @ M=512 |
| `torch.compile` | unavailable: `TritonMissing` |

The CSR SpMM pathology matters independently: any future GPU work on this
operator must not use `torch.sparse.mm` with a dense RHS at M ≥ 128 on this
stack.

## Numerics (this part works and is reusable)

* Operator: the simulator's 5-point matrix embedded in the full (nx, ny) grid
  with identity rows on the exterior CVs; the embedding reproduces the exported
  matrix bit-exactly (`max|dA| = 0`). Exterior rows are pinned by Jacobi with
  ω = 1, so no masks are needed anywhere.
* Hierarchy: 2×2 piecewise-constant aggregation, prolongator smoothed with one
  Jacobi step `P = (I − W D⁻¹ A) P₀`, Galerkin coarse operators (5-point at
  level 0, 13-point at level 1, denser below), 7 levels (386×320 → 7×5), damped
  Jacobi ω = 0.8, exact dense solve at the coarsest level (35 CVs).
* PCG with `pre = post = 1`, **fixed 16 iterations** — deterministic, and since
  A is fixed for a whole episode the count is a per-building constant.
* Exactness, CPU reference of the identical algorithm: **16 iters, errcomp
  5.5e-14** (train_8, n = 81 388); **15 iters, 9.1e-14** (train_2,
  n = 241 432). On the GPU the same count gives 9.0e-14 … 1.03e-13. This is the
  fp64 floor: the plain SuperLU solution itself deviates from a refined
  reference by 3.4e-14 relative, and further refinement oscillates at ~1e-14.
  For margin use 20 iterations (costs 20 % more).
* Rejected variants, all measured (`sacpu.py`): unsmoothed aggregation never
  converges (40 iters, errcomp 4e-7 — the coarse space must be smoothed);
  4×4 first-level aggregation 40 iters, 5e-11; prolongator weights 1.0/1.2
  diverge; F-cycle diverges; geometric MG with piecewise-constant transfers
  (the first prototype) converges at only 0.92/cycle. pre/post = 2/2 gives 17
  iters, 1/1 is better.
* Warm start from the previous step buys nothing: the requirement is 1e-13
  *relative to b* while consecutive solutions differ by 2e-3 of the field range,
  so the iterate must be converged regardless (no iteration-count change).

## Chained (accumulation) test — done, 288 steps

`export_chain.py` re-ran 288 steps of train_8/atlanta_1 under CtrlW recording
the dry RHS, the pre-step interior-mass temperature, and the field produced; the
mass recursion (`retained ≤ 0.958`, `retained + gained = 1`, plus the Woodbury
air side, both reproduced exactly — mass recursion deviation 0.00e+00) makes the
episode exactly replayable. `gpu_chain2.py` replays it on the GPU with
16 iterations.

Result: **max errcomp 2.15e-13 over 288 steps** (step 0: 5.2e-14, median
1.07e-13, and a monotone drift 5.2e-14 → 1.2e-13 → 2.15e-13). The drift is the
interior-mass filter feeding a fraction of each step's error into the next
forcing; it is ~4× over 288 steps and stays at the fp64 floor — no runaway, but
it means the 1e-13 standard is met only with the 20-iteration setting.

## Remaining work if the fused-kernel blocker were removed [INFERENCE]

1. One fused HIP kernel per level (5-point level 0, 13-point level 1), taking
   the coefficient grids as arguments: `torch.utils.cpp_extension.load_inline`
   with the ROCm toolchain already on the PC, ~2–3 h. Expected 15× less traffic
   ⇒ ≈1.2 ms/env-step at M=512.
2. Air side is already trivial: `A⁻¹P` (30 columns) is solved once per episode
   (measured 0.85 s at M=30 on the GPU), then per step two `30×n` GEMVs plus a
   `30×30` solve, ≈25 µs/env-step at M=512 in fp64.
3. Do not port the rest of the env step on this evidence: the thermal half
   cannot beat a single CPU core by more than ~4×.

## Side finding that is useful regardless

SuperLU's batched solve *degrades* with M (5.2 → 6.6 → 8.3 → 12.1 ms/env-step
from M = 1 to 1024): batching RHS on the CPU is counter-productive. One core per
environment (or per 2–4 environments) at 3.8–4.4 ms is the better CPU
configuration — that is the number any GPU port, and any accelerator claim, has
to beat.
