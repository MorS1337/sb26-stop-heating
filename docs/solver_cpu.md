# Exact CPU solve for the simulator's per-step system

`finite_difference_solver.FiniteDifferenceSolver.solve` solves `A x = b` once per
300 s step, then once more for the VAV Woodbury correction, plus `n_terms` (30–85)
columns once per factorization for the response matrix. `A` is SPD, so the shipped
`splu(A, permc_spec='MMD_AT_PLUS_A')` is an unsymmetric LU of a symmetric matrix.
This workstream replaced it with an LL^T factorization of a symmetrically permuted
`A` plus a batched solve, measured what each ordering/factorization/kernel costs,
and gated the result against the exactness standard.

Everything is opt-in: `cenv/solver.py` + `cenv/src/solver_chol.c` (new files; no
edit to `cenv/thermal.py`/`thermal.c`), switched on with `cenv.solver.install()`,
`sbx/evalrun.py --cenv solver`, or `Factor`/`solve_many` called directly.
No numpy-only restriction applies here (this is the simulator, not the policy).

## Verdict

Two answers, because the two cases have different winners:

**One environment (the LB runs, `evalrun`): parity, in practice 10% the wrong
way — keep SuperLU.** The shipped SuperLU solve is 12–15% faster than the best
Cholesky path on every building (1.2–2.3 ns per triangular update against
2.4–4.0): its two sweeps are as tight as this kernel and, because it pivots, it
needs neither the divisions nor the transpose copy. Measured end to end on a full
2880-step train_8 episode: **32.69 → 36.13 ms CPU per `env.step` (+10.5%)**. There
is no exact win to be had at single-RHS latency; do not switch.

**M environments of one scenario sharing `A` (training rollouts, the fused env):
4–6x per env-step.** One factorization, `solve_many(factor, B)` with `B` an
`(n, M)` block of right-hand sides. At M=16–64 the batched Cholesky costs
1.6–2.0 ms per env-step on train_8 (5.1 ms on train_1) against 7.3/8.6 ms for
one-at-a-time solves and 7.3/8.6 ms for scipy's own 2-D `solve` (SuperLU's
`gstrs` loops over the columns and does not vectorize, so *it* does not scale
with M at all). That is the whole point of replacing the solver: it is not a
single-step win, it is a throughput win.

| method (train_8, n=81,388) | nnz(factor) | ms/solve M=1 | best batched ms/env-step | at M | max per-step rel err | gate |
|---|---|---|---|---|---|---|
| shipped `splu(MMD_AT_PLUS_A)` | 3,540,476 (L+U) | **8.1** | 8.7–10.7 (no scaling) | 64 | 0 (reference) | — |
| LL^T, METIS, CSC sweeps | 1,865,430 | 9.2 | 1.83 (1.86 M=1) | 64 | 3.7e-14 | **PASS** |
| LL^T, METIS, panels | 2,085,778 | 12.6 | **1.58** | 64 | 4.5e-16 | **PASS** |
| LL^T, AMD, CSC sweeps | 2,076,380 | 8.0 | 2.00 | 64 | 3.7e-14 | **PASS** |
| LL^T, AMD, panels | 2,304,564 | 14.5 | 1.73 | 64 | 4.5e-16 | **PASS** |

(`ms/solve M=1` is the median of interleaved A/B rounds on a Mac M3 while the box
was loaded by other jobs (load average 7–8), so absolute values carry ±10%
run-to-run spread; the *sign and size of every ratio* here reproduced across
three measurement sessions and on the PC. Full per-building tables in §2–3,
end-to-end episode in §6, PC numbers in §7.)

Recommendation: **default `Factor(A, method="metis")` + `solve_many` for all
multi-environment stepping, and leave the single-environment path alone.** The
factor should be built per (scenario, h) exactly where `splu` is built today.

---

## 1. The matrix

One scenario per building, `A = solver.matrix(h)` on the shipped tree after three
steps of `CtrlW` (so `h` is the scenario's real convection coefficient). `n` is
the number of *solved* CVs — the exterior (Dirichlet) CVs are already eliminated
before `A` exists.

| building | n | nnz(A) | nnz(shipped L+U) | L+U / A | κ(A) | h | diag range | Jacobi row ratio | components |
|---|---|---|---|---|---|---|---|---|---|
| train_1/atlanta_2 | 148,725 | 740,139 | 6,264,510 | 8.46 | 1511 | 19.1 | 153–753 | 0.9989 | 3 × 49,575 |
| train_2/atlanta_3 | 241,432 | 1,204,340 | 14,834,176 | 12.32 | 1486 | 19.1 | 57–601 | 0.9989 | 2 × 120,716 |
| train_4/atlanta_5 | 206,814 | 1,029,666 | 9,808,266 | 9.53 | 1538 | 19.1 | 45–601 | 0.9989 | 3 × 68,938 |
| train_5/atlanta_6 | 183,952 | 915,576 | 7,864,960 | 8.59 | 1389 | 19.1 | 107–601 | 0.9989 | 4 × 45,988 |
| train_7/atlanta_8 | 92,106 | 458,410 | 4,029,344 | 8.79 | 1441 | 19.1 | 47–601 | 0.9989 | 2 × 46,053 |
| train_8/fairbanks_8 | 81,388 | 405,068 | 3,540,476 | 8.74 | 1413 | 8.8 | 99–601 | 0.9989 | 2 × 40,694 |

What the numbers say:

* **Symmetric, no explicit zeros, no asymmetry** (`asym_nnz = 0`), so LL^T is
  legitimate. `λmax ≈ 1200` on every building, which is just `A_ii + Σ_j|A_ij|`
  (diagonal up to 600, row sums 599) — i.e. the pinning and the air coupling
  dominate, not the storage term — and `λmin ≈ 0.8` (the slow envelope mode of
  the whole 386 × 320 floor plan), hence κ ≈ 1400–1540 —
  small for an ill-conditioned-looking problem, but not small enough for
  iterative methods: the Jacobi contraction bound is **0.9989** (interior air
  conductivity is 50 W/mK, so the air CVs are almost completely coupled), which
  needs ~2500 iterations for 1e-12. Stationary iterations are out, and so is CG
  on a matrix this cheap to factor.
* **2–4 connected components of exactly equal size.** The floors/air blocks are
  decoupled; every ordering routine in `cenv/solver.py` therefore works per
  component, which is also what makes METIS' nested dissection cheap.
* **Fill is 4.4–6.4× the matrix** on the symmetric side, and SuperLU's 8.5–12.3×
  counts both triangles: the shipped LU and a symmetric Cholesky have the *same*
  number of nonzeros to stream per solve (`2 × nnz(L_chol) ≈ nnz(L)+nnz(U)`), so
  a Cholesky wins nothing by itself — only the ordering and the kernel matter.
* Row ratios per material (train_1): conductivity 0.21 → 0.008,
  0.54 → 0.06, 50 → 0.9989; the max degree is 5 (5-point stencil) but the *fill*
  degree is what makes the solves expensive.

## 2. Fill and single-RHS solve by ordering

`cenv_chol_create` factors `P A P^T = L L^T`; `perm` comes from METIS
(`METIS_NodeND`, 32-bit, per component), AMD (SuiteSparse `amd_order`), natural or
RCM. All timings are medians of interleaved rounds against the shipped
`splu(..., 'MMD_AT_PLUS_A')` on the same matrix, Mac M3, load average 7–8.
"ns/upd" is `ms / (2·nnz(L))` for the Cholesky and `ms / (nnz(L)+nnz(U))` for
SuperLU, i.e. nanoseconds per scalar triangular update — the only fair comparison.

| building | method | nnz(factor) | x nnz(A) | factor s | ms/solve | ns/upd | GB/s |
|---|---|---|---|---|---|---|---|
| train_1 | superlu MMD | 6,264,510 | 8.46 | 0.41 | **9.12** | 1.46 | 11.0 |
| train_1 | METIS csc | 3,275,814 | 4.43 | 0.24 | 9.49 | 2.90 | 4.1 |
| train_1 | METIS panels | 3,701,226 | 5.00 | 0.24 | 14.65 | 4.47 | 2.7 |
| train_1 | AMD csc | 3,684,189 | 4.98 | 0.27 | 9.35 | 2.54 | 4.7 |
| train_2 | superlu MMD | 14,834,176 | 12.32 | 0.89 | **17.42** | 1.17 | 13.6 |
| train_2 | METIS csc | 6,466,042 | 5.37 | 0.71 | 20.44 | 3.16 | 3.8 |
| train_2 | AMD csc | 8,315,540 | 6.90 | 1.15 | 27.17 | 3.27 | 3.7 |
| train_4 | superlu MMD | 9,808,266 | 9.53 | 0.90 | **16.95** | 1.73 | 9.3 |
| train_4 | METIS csc | 5,015,499 | 4.87 | 0.59 | 20.91 | 4.17 | 2.9 |
| train_4 | AMD csc | 5,499,969 | 5.34 | 0.66 | 21.92 | 3.98 | 3.0 |
| train_5 | superlu MMD | 7,864,960 | 8.59 | 0.73 | **14.62** | 1.86 | 8.6 |
| train_5 | METIS csc | 4,051,624 | 4.43 | 0.42 | 18.74 | 4.63 | 2.6 |
| train_5 | AMD csc | 4,729,516 | 5.17 | 0.56 | 19.95 | 4.22 | 2.8 |
| train_7 | superlu MMD | 4,029,344 | 8.79 | 0.37 | **8.45** | 2.10 | 7.6 |
| train_7 | METIS csc | 2,047,066 | 4.47 | 0.21 | 10.13 | 4.95 | 2.4 |
| train_7 | AMD csc | 2,337,398 | 5.10 | 0.27 | 10.49 | 4.49 | 2.7 |
| train_8 | superlu MMD | 3,540,476 | 8.74 | 0.34 | **8.10** | 2.29 | 7.0 |
| train_8 | METIS csc | 1,865,430 | 4.61 | 0.20 | 9.18 | 4.92 | 2.4 |
| train_8 | AMD csc | 2,076,380 | 5.13 | 0.23 | 8.02 | 3.86 | 3.1 |

(panel rows for the other buildings are in `scratch/SolverCPU/data/bench.json`;
they are uniformly 1.3–1.7× slower than the CSC sweeps at M=1.)

**Findings.**

* **METIS beats AMD everywhere** (fill 4.4–5.4× against 5.0–6.9×) and beats
  SuperLU's own ordering on the *symmetric* factor (e.g. train_2 6.47M against
  7.42M per triangle), but SuperLU's factor is 5% smaller on train_8 and the
  per-update rate is better, so the single-RHS totals come out at parity or
  10–20% in SuperLU's favour.
* **`splu`'s `perm_c` is not a usable symmetric ordering**: fed to the Cholesky
  as `P`, it gives nnz(L) = 132,182,829 on train_1 (40× METIS) and a 194 s
  factorization. SuperLU pivots, so `perm_c` is a column ordering for a *pivoted*
  LU. `mmd_perm` is kept only as that comparison and is never a fallback.
* **Factorization cost is a non-issue**: 0.2–1.2 s once per (scenario, h) against
  8641 solves per episode, and `h` is constant in every shipped weather
  controller. The shipped SuperLU is in the same range (0.34–0.90 s).
* Factor bytes: the METIS Cholesky is `12·nnz(L)` = 22 MB for train_8, 39 MB for
  train_1 (Lp+Li+Lx), against 57 MB / 100 MB for the SuperLU L+U of the same
  matrices — plus a second copy for the panel path (`nnz_super`, 2.1M for
  train_8) and the transpose copy used by the CSC backward sweep (15 MB). The
  P-core L2 is 16 MB and the PC's L3 is 32 MB, so **nothing fits**: every sweep is
  a DRAM stream, which is exactly why the batched case is the one that pays.
* Effective bandwidth at M=1 is 2.4–4.7 GB/s — nowhere near the 40 GB/s the
  machine can do, because a triangular solve is latency-bound (each update is a
  dependent load-multiply-store), not bandwidth-bound. That is the ceiling a
  single right-hand side runs into, and it is the same ceiling SuperLU hits.

## 3. Batched multi-RHS

`B` is `(n, M)` C-contiguous — the M right-hand sides of one unknown adjacent, so
the inner loops become `for t in 0..M-1` over contiguous doubles (`restrict`ed, so
LLVM vectorizes them; without that the loop stays scalar and the whole thing is
2× slower). The workspace (`Y`, and the panel scratch) is cached inside the
factor, so repeated calls do not allocate: a 64-wide block on train_1 is 76 MB,
and page-faulting that per call costs more than the solve.

**ms per env-step (one solve for one environment, i.e. total/M):**

| building | superlu | METIS csc M=8/16/32/64 | METIS panels M=8/16/32/64 | best | vs superlu | vs M=1 |
|---|---|---|---|---|---|---|
| train_1 | 8.62 | 2.68 / 2.12 / 2.02 / 2.06 | 3.16 / 2.36 / 2.09 / **1.90** | 1.90 @64 | 4.5× | 4.6× |
| train_2 | 19.59 | 6.87 / 5.06 / 5.06 / 5.10 | 7.56 / 5.78 / 4.75 / **4.18** | 4.18 @64 | 4.7× | 5.4× |
| train_4 | 17.85 | 6.48 / 5.68 / 5.73 / 4.98 | 6.93 / 5.73 / 4.92 / **4.36** | 4.36 @64 | 4.1× | 5.7× |
| train_5 | 17.76 | 5.53 / 5.03 / 3.91 / 4.47 | 6.96 / 4.57 / 3.95 / **3.68** | 3.68 @64 | 4.8× | 4.6× |
| train_7 | 9.44 | 3.14 / 2.13 / 2.24 / 2.17 | 3.28 / 2.40 / 2.07 / **1.94** | 1.94 @64 | 4.9× | 4.2× |
| train_8 | 7.29 | 2.66 / 2.03 / 1.97 / 1.83 | 2.71 / 1.95 / 1.81 / **1.58** | 1.58 @64 | 4.6× | 5.3× |

(Every cell is `total/M` from one interleaved loop, so the SuperLU M=1 cell (7.29)
sits a little under its `ms_single` in §2 (8.10) — same measurement, different
rounds. SuperLU barely moves with M: 7.29 → 8.71 at M=64 on train_8 and
8.62 → 10.72 on train_1, i.e. its 2-D `solve` costs *more* per right-hand side as
the block grows.)

**Where the plateau is.** Per sweep the batched kernel reads 12 B of factor per
nonzero and 2·8·M bytes of right-hand side per nonzero. At M=64 that is 1.0 kB
per nonzero, i.e. 1.9 GB per sweep on train_8, and the measured 79 ms/sweep is
≈ 42 GB/s — main-memory bandwidth. So M=32–64 is the wall, not a tuning problem:
the batched solve is memory-bound, `ms/env-step ≈ 16·nnz(L)/bandwidth` =
1.3–2.0 ms for these buildings, and the only way further down is to reduce
bytes per nonzero (see the negative results in §8: widening the panels to make
them dense was tried and does not pay on these lattices). Practically: **M=16
already captures 90% of the win** (train_8: 1.95 ms/env against 1.58 at M=64) —
use M=16–32 and do not pay for more.

Per-step the simulator makes *two* single-RHS solves (forcing, then the Woodbury
correction), so a fused M-environment stepper should expect `2 ×` the numbers
above per env-step, i.e. 3.2 ms/env-step on train_8 at M=32. The response matrix
build (`n_terms` columns in blocks of 16) also goes through the batched path and
is where the batched kernel wins first: 16 columns cost 31 ms on the Cholesky
against 116 ms for SuperLU's 2-D `solve` (its per-column cost either way).

## 4. Exactness gate

Two sides in **separate processes** (`scratch/SolverCPU/exact.py trace` /
`compare`) — one untouched, one with `scipy.sparse.linalg.splu` routed through the
Cholesky, both driving `CtrlW` on `building_train_8/fairbanks_8` for 2880 steps
(10 sim days). The patched side additionally solves every step *again* with the
reference factorization from the same right-hand side, which gives a per-step
per-component error over both the forcing solve and the 16-column response
blocks.

| quantity | value | standard | verdict |
|---|---|---|---|
| per-step max rel error over all solves | **3.7e-14** | ≤1e-12 | pass |
| total reward, 2880 steps (ref / new) | −66.13828237170674 / −66.13828237170674 | ≤1e-9 rel | pass (0.0) |
| per-step reward, max |Δ| | 0.0 | — | pass |
| zone average temperatures, max |Δ| | 7.1e-11 K (2.4e-13 rel) | per-component 1e-12 rel | pass |
| whole temperature field, 96 checkpoints | 3.05e-5 K = one float32 ulp | — | pass (identical to storage precision) |
| policy decisions that flipped | 0 of 2880 | — | pass |
| interior-mass node temperatures | not separately gated (drives the field above) | — | — |

**Pitfall worth recording.** The first design ran both environments in one
process in lockstep and showed 1e-2 K divergence at step 0, growing to 0.3 K —
*independent of the solver*. The occupancy/convection randomness is drawn from a
process-global stream, so a second environment on the same process sees a
different draw than the first. Two sides of any trajectory comparison must be
separate processes (`evalrun` does this; `cenv/verify/gate.py` does this). This is
also why the `--cenv` A/B comparisons of other components must never be built as
"two envs in one process".

Not gated: train_2 (n=241k, 14.8M nonzeros) — the same kernel and the same
ordering code path, but the episode takes ~25 min per side and was not run.

## 5. The opt-in path

New files only: `cenv/solver.py`, `cenv/include/solver_chol.h`,
`cenv/src/solver_chol.c` (the library is built by `cenv/build.py` from
`cenv/src/*.c` with no edit). `cenv/thermal.py` and `cenv/thermal.c` are
untouched.

```python
from cenv.solver import Factor, solve_many, install

f = Factor(A)                  # A = solver.matrix(h); METIS per component,
                               # AMD fallback if libmetis is absent (the PC)
x = f.solve(b)                 # (n,) -> (n,), CSC sweep pair
X = f.solve_many(B)            # (n, M) C-contiguous -> (n, M); kernel picked by M
f.stats()                      # nnz(L), nnz_super, nsuper, maxwidth, flops, bytes

install(method="metis")        # whole-simulator opt-in: routes
                               # scipy.sparse.linalg.splu through it, returns the
                               # cenv {"patched"/"restore"} dict
```

* **Hook for the simulator** (one line, no rewrite): nothing — `install()`
  replaces `scipy.sparse.linalg.splu` and `FiniteDifferenceSolver.solve` keeps
  calling it. The shim answers `.solve`, `.L`, `.U`, `.perm_r`, `.perm_c` exactly
  as `SuperLU` does, so `cenv/thermal.py` (which exports `lu.L`/`lu.U` into its C
  kernel) also keeps working: its recipe ( `y = b[argsort(perm_r)]`, L and U
  sweeps, `out[i] = y[perm_c[i]]` ) reproduces `A⁻¹b` with the shim's factors to
  **9.7e-15** relative.
* **Hook for a fused environment** (what CEnvFused2 needs): keep one `Factor`
  per (scenario, h) where it keeps the `splu` factor today, and replace the
  per-environment solve with `solve_many(B)` where `B` packs M environments'
  right-hand sides. The factor is not thread-safe (one cached workspace per
  factor); several environments in one thread is fine.
* `--cenv solver` works end to end in `sbx/evalrun.py` (verified: 300-step
  train_8 scenario, identical total reward −2.831134653861227 with and without
  it). It is **not** in `--cenv all` (`GATE_PASSED = False`, deliberately): on a
  single environment it is the 10% regression of §6, so it must be named, and it
  only earns its keep through `solve_many`. `--cenv solver,thermal` is consistent
  (see above) but pointless — thermal's C kernel does its own sweeps over the same
  3.7M nonzeros.

## 6. End-to-end

One full 2880-step `CtrlW` episode on `train_8/fairbanks_8`, `env.step` CPU time
measured inside the trace (process CPU, so other processes' load does not inflate
it), both sides run concurrently so they see the same memory contention.

| side | ms CPU per `env.step` | solve time inside it |
|---|---|---|
| shipped (`splu` per step) | **32.69** | 2 × 8.1 ms = 16.2 ms (49%) |
| LL^T (CSC sweeps), no shadow | **36.13** (+10.5%) | 2 × 9.2 ms = 18.4 ms (51%) |

Both sides ran concurrently so they saw the same machine, and both end on
−66.138282372. The +3.4 ms is the Cholesky's 13% slower solve, twice per step,
exactly as §2 predicts.

The per-step system is 2 solves out of 32.7 ms of step work, so **the
single-environment end-to-end effect of this workstream is −10.5% (a regression),
by construction**: the Cholesky is 13% slower per solve and the rest of the step
is untouched. The gain that matters is the batched one in §3 — a
per-environment-step factor of 4–6 at M=16–64 — which does not appear in a
single-environment episode at all.

What the single-env A/B does buy is the *measurement*: solve = 16.2 ms of 32.7 ms
(49%) of a `train_8` step on the shipped path. That is the budget any fused-C
thermal component, or any solver swap, is chasing.

## 7. PC (Ryzen 5600X, WSL2, single-threaded, ≤2 threads)

Same `bench.py`, same matrices, mirrored with `scratch/SolverCPU/pc_bench.sh`
(gcc there needs `-D_GNU_SOURCE` for `cenv/src/hvac.c`'s `M_PI`; the wrapper in
`pc_bench.sh` supplies it without editing the file). No METIS on the PC (no
package install allowed), so AMD is the ordering there.

| building | superlu ms | AMD csc ms (M=1) | AMD panels M=8 | M=32 | M=64 |
|---|---|---|---|---|---|
| train_1 | 22.07 | 19.96 | 6.55 | 4.83 | 4.60 |
| train_2 | 38.84 | 37.87 | 12.83 | 10.04 | 9.97 |
| train_4 | 32.30 | 24.98 | 8.79 | 6.70 | 6.63 |
| train_5 | 24.34 | 22.71 | 8.45 | 5.86 | 5.46 |
| train_7 | 13.75 | 12.21 | 3.91 | 2.97 | 2.71 |
| train_8 | 12.20 | 11.39 | 3.91 | 2.64 | 2.43 |

(full table: `scratch/SolverCPU/data/pc/bench_pc.json`; PC absolute times are
1.2–1.7× the M3's per solve, and the batched plateau at M=32–64 is 2.4–2.6 ms per
env-step against 11–13 ms for one-at-a-time — the same 4–5× ratio, so the
recommendation is machine-independent. `sbx/evalrun.py --cenv` must not be used
on the PC yet: the C library there needs the `-D_GNU_SOURCE` wrapper above.)

## 8. Negative results (so nobody repeats them)

* **Stationary iterations**: Jacobi bound 0.9989 → ~2500 iterations for 1e-12.
  Not exact, not fast. CG would need a preconditioner and land well short of
  exactness; a direct factorization at 0.2–1.2 s once per episode is not the
  bottleneck.
* **SuperLU's `perm_c` as a symmetric ordering**: 40× fill, 194 s to factorize.
* **Supernodal panels for a single right-hand side**: 1.3–1.7× slower than the
  CSC sweeps. Fundamental supernodes here are ~1.3 columns wide (62,262 of them
  for 81,388 unknowns), so a panel has almost nothing to amortize.
* **Gather (dot-product) form of the sweeps**, which would need no transpose copy
  of the factor: 1.6× slower at M=1 and 1.3× at M=8. The scatter form wins because
  each column's stores run *up* the row indices, i.e. sequentially.
* **Relaxed amalgamation** (merge consecutive supernodes into dense w-column
  panels, zero-padded; `CENV_CHOL_RELAX_WIDTH`): doubles the factor
  (nnz_super 3.86M against 2.09M) and the panel sweep then becomes bound by the
  padded factor rather than by the right-hand sides — 1.40 against 1.31 ms per
  right-hand side at M=64, and 1.7× slower at M=1. Left behind the knob, default
  off. Widening would only pay if the supernodes were also *split* (front-limited
  width), which is more machinery than the 5× already in hand justifies.
* **Dense `A⁻¹P`** (already measured by the `cenv/thermal` workstream): only
  viable when `n·k` is small against the factor, which on these buildings is
  never true for the per-step solves.

## 9. Files

| path | what |
|---|---|
| `cenv/solver.py` | `Factor`, `solve_many`, orderings (METIS/AMD/natural/RCM/SuperLU-`perm_c`), the `splu` shim + `install()` |
| `cenv/include/solver_chol.h`, `cenv/src/solver_chol.c` | LL^T factorization (up-looking, Davis), fundamental supernodes + panels, CSC sweep pair, batched solves, workspace cache |
| `scratch/SolverCPU/charac.py` | characterisation of `A` per building → `data/<building>.{json,npz}` |
| `scratch/SolverCPU/bench.py` | interleaved fill/solve/batch benchmark → `data/bench.json` |
| `scratch/SolverCPU/exact.py` | `trace` / `compare`: the closed-loop exactness gate |
| `scratch/SolverCPU/pc_bench.sh`, `report.py` | mirror + run on the PC; markdown tables |

Reproduce:

```
starter-kit/.venv/bin/python scratch/SolverCPU/charac.py <building>/<scenario>
starter-kit/.venv/bin/python scratch/SolverCPU/bench.py \
    --orderings metis,amd --modes 1,0 --M 1,4,8,16,32,64 --reps 5
starter-kit/.venv/bin/python scratch/SolverCPU/exact.py trace \
    --scenario building_train_8/fairbanks_8 --steps 2880 --new ref  --out /tmp/r.npz
starter-kit/.venv/bin/python scratch/SolverCPU/exact.py trace \
    --scenario building_train_8/fairbanks_8 --steps 2880 --new chol --out /tmp/c.npz
starter-kit/.venv/bin/python scratch/SolverCPU/exact.py compare /tmp/r.npz /tmp/c.npz
```

The OpenReview-relevant claim: *the per-step solve can be made exact to 4e-14 per
step and 0 in total reward, and it becomes 4–6× cheaper per environment once
several environments share one factorization* — which is the shape most RL
training loops want and which the shipped `splu` cannot exploit at all.
