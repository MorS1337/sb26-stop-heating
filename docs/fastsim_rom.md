# fastsim — reduced-order surrogate of the Smart Buildings thermal step

Status: reference implementation + fidelity harness in `fastsim/`, measured on
`building_train_8/fairbanks_8` and `building_train_1/atlanta_2`, shipped sim,
CtrlW, first 2880 steps (10 days).

**Verdict (updated after Part II).** GO as a *ranking / selection* accelerator:
the surrogate's reward bias is **policy-independent** and the ordering of five
CtrlW variants is reproduced exactly (Spearman = Kendall = 1.000 on both
buildings, §11), while the absolute reward is off by 2.5–5 %.
NO-GO as a substitute for the real simulator; NO-GO as a drop-in speedup today
(1.19x end-to-end, §12, because the non-thermal half of the step dominates once
the solve is gone); and the 0.1 K over 10 days target is **not** met
(1.0–1.2 K max / 0.2–0.3 K mean, flat from m = 100 to 400) — §5.4 says what the
residual is, and §10 shows that refitting the reduced maps does not remove it.
See §13 for the two go/no-go calls (CMA-ES use, torch/ROCm port).

---

## 1. The exact per-step linear algebra

All line numbers are `sims/shipped/smart_buildings/smart_control/simulator/`.
The step is executed by `SimulatorFlexibleGeometries.execute_step_sim`
(`simulator_flexible_floor_plan.py:280-530`) and the linear algebra by
`FiniteDifferenceSolver` (`finite_difference_solver.py`).

### 1.1 State

* `T` — temperature field, one value per control volume (CV), shape 386x320
  (123,520 CV) for building_train_8, 606x339 (205,434) for building_train_1.
* `T_m` — interior-mass node, one per solved CV that carries mass (the floor
  slab and furnishings). Not exposed by the building; the solver owns it.
* Exterior CVs are **pinned** to ambient and *eliminated* from the unknowns
  (`:754` `self._solved`). n_solved = 81,388 and 148,725.

### 1.2 Matrices, assembled once per building (`:705-830`)

```
A(h) = D(k_rowsum + C/dt + M_c) - K + D(g_amb(h))          [n_solved x n_solved]
K    = conductance between solved CVs (5-point, per-face k*A/d)
C/dt = (floor_height * A_h * A_v * rho * cp) / dt          `_capacitance_rate` :775
d    = total conductance to the pinned CVs                 `_dirichlet_conductance` :764
g_amb= h * exposed face area, exposed CVs only             `ambient_conductance(h)` :867
M_c  = (C_m/dt * G_m) / (C_m/dt + G_m)                     `_mass_coupling` :808
r,g  = C_m/dt / (C_m/dt + G_m), G_m / (C_m/dt + G_m)       `_mass_retained/_gained`
```

The interior mass node is eliminated analytically (`:790-812`); it survives as
one extra diagonal term plus one forcing term. `A` is symmetric positive
definite and constant in time for a constant `h`: every shipped
`ReplayWeatherController` returns the config's `convection_coefficient`
(8.8 W/m²K here, verified constant over all 2880 logged steps), so the
factorization is built once (`:998`, `splu(..., permc_spec='MMD_AT_PLUS_A')`).

### 1.3 One step (`solve` :968-1025)

```
y    = A⁻¹ [ (C/dt)⊙T + q + T_amb (d + g_amb) + M_c⊙T_m ]              (1)
q_a  = (I + G S)⁻¹ G (T_s - Qᵀ y),      S = Qᵀ A⁻¹ P                    (2)
T'   = y + A⁻¹ P q_a                                                   (3)
T_m' = r⊙T_m + g⊙T'                                                    (4)
T'[pinned] = T_amb                                                     (5)
```

* `q` = `building.input_q`, W per CV. **Assigned, not accumulated**
  (`building.py:1123`): a zone's total power is written as `power / n_cv` to
  every CV of the zone, and only zones present in `zone_q_aggregate` are
  written, so a zone that gets no power this step keeps last step's value
  *including the air-side fold-back*. The recorder logs the array as passed and
  reproduces it exactly (max |per-CV − q_zone/n_cv| = 7e-15 W).
* Zone power sources (`execute_step_sim`): reheat `reheat_w * area_z/area_vav`
  (:432), internal gains `10 W/m² * cv_area * fraction` (:446-461), occupant
  sensible heat `70 W * occupancy_z` (:465-481). Coil duty never depends on the
  temperature field; the air stream does, and it is *not* in `q`.
* Air side, rank k = one column per VAV (k = 30 and 39):
  `G_j = flow_rate_demand_j * rho_air * cp_air` (:425), `T_s,j` = the AHU's
  supply temperature (:415). `P` (n x k) is where column j's power lands —
  diffuser CVs of the zones that VAV serves, weighted `n_cv/area_vav`, summing
  to 1 per column. `Qᵀ` (k x n) is the mean that VAV's thermostat reads — the
  same CVs, weights `1/area_vav`, summing to 1 per row. Both are registered
  once by `SimulatorFlexibleGeometries._couple_air_side` (:160-230) and passed
  to `couple_air_side` (:908-966); (2) is the Woodbury solve in `_add_air_side`
  (:1027-1103) and is *stable* at any `dt`: the air stream is implicit.
* After the solve, `air_side_q = G ⊙ (T_s - Qᵀ T')` is folded back into
  `input_q` (:500-513) — this is what `get_zone_thermal_energy_rate` reports.
* `apply_convection()` (:516) then shuffles CVs *within each room*:
  `StochasticConvectionSimulator`, p = 0.1, distance = 2, seed = 5
  (`stochastic_convection_simulator.py:130,229`, numba). It preserves every
  room mean exactly and permutes CVs, so it moves no energy.
* Reward and observations read only *zone means*
  (`building.py:1083-1095`, mean over the zone's CVs) plus HVAC energy.

### 1.4 What the reduction has to reproduce

Only quantities built from zone means: the 30–39 zone temperatures (comfort,
productivity), each VAV's sensing temperature `Qᵀ T` (drives the air side and
the AHU), and the AHU recirculation temperature = area-weighted zone mean
(`:311-322`). Energy: reheat duty is temperature-independent; fan and pump duty
depend on air/water flow setpoints; the AHU heating coil depends on the return
temperature, i.e. on zone means.

---

## 2. Operator extraction (`fastsim/extract.py`)

Nothing is re-derived: every array is read off the live solver instance the real
environment built, so the ROM's linear algebra is the simulator's own.

| | building_train_8 | building_train_1 |
|---|---|---|
| grid / n_solved | 386x320 / 81,388 | 606x339 / 148,725 |
| zones = air terms | 30 | 39 |
| A nnz | 405,068 | 740,139 |
| h (constant) | 8.8 W/m²K | 8.8 W/m²K |
| C/dt values (W/K) | 0.4824 (ext. wall), 95.96, 121.1, 191.9, 383.8 (room air) | 0.4824, 149.4, 187.2, 374.3, 748.6 |
| mass CVs | 77,127 | 141,779 |
| `M_c` / `r` / `g` | 0.191644 / 0.958222 / 0.041778 | 0.196950 / 0.984752 / 0.015248 |
| `d` non-zero, `g_amb` non-zero | 9, 1,852 | 0, 3,444 |
| mass CVs outside every zone | 40 | 72 |

Two structural facts the reduction uses, both verified (`fastsim.rom.describe_ops`):
`M_c`, `r` and `g` are **single scalars** across the whole building, and every
zone CV carries a mass node. So the mass node is exactly a per-room scalar
state (plus one bucket for the 40/72 CVs outside all rooms) — no approximation
is needed there.

**Extraction is validated, not assumed.** `fastsim.fit.exact_replay` replays the
logged inputs through the extracted operators and the real factorization and
reproduces the real rollout's zone means to **1.6e-3 K max over 10 days**
(1 h: 2e-5 K). Every error in §5 is therefore reduction error.

---

## 3. The reduced model (`fastsim/rom.py`)

### 3.1 Projection in the energy inner product

`A⁻¹C` is not symmetric, and a plain Galerkin projection of it onto a truncated
basis produced spurious eigenvalues outside the unit disc: at m = 200 the state
grew by e^1.2 per step and left the physical range within 12 steps (measured,
reproducible). But `S = C^½ A⁻¹ C^½` **is** symmetric with spectrum in (0,1], and
its Galerkin projection inherits that spectrum because Rayleigh quotients of a
symmetric matrix are bounded by its eigenvalues. So the state is taken as
`u = C^½ T` and projected onto an orthonormal `W`:

```
b    = Ac s + Aq q_zone + dv T_amb + Mr u_m                 [m]        (6)
q_a  = (I + G S_a)⁻¹ G (T_s - Qq b)                         [k]        (7)
s'   = b + Pr q_a                                           [m]        (8)
u_m' = r u_m + g (Zm_u s')                                  [ng]       (9)
z    = Zo s'                                                [n_zones] (10)
```

with, all precomputed once per (basis, factorization):

```
Ac = Wᵀ C^½ A⁻¹ C^½ W        Aq = Wᵀ C^½ A⁻¹ Zs        Zs[cv,z] = 1/n_cv(z)
dv = Wᵀ C^½ A⁻¹ (d+g_amb)    Mr = Wᵀ C^½ A⁻¹ (M_c ∘ S_m)   S_m = 0/1 scatter
Pr = Wᵀ C^½ A⁻¹ P            S_a = Qᵀ A⁻¹ P   (exact, k x k)
Qq = Qᵀ C^-½ W               Zo = Z_avg C^-½ W             Zm_u = Z_m C^-½ W
```

`S_a` is formed **exactly** (k extra solves) rather than through the basis: the
air-side loop is stiff, `G/(C/dt)` reaches 3.5 on a room CV so `(I+GS_a)⁻¹`
amplifies by up to 4.5, and a projected `S_a` makes it spuriously
ill-conditioned. `Ac` is symmetric by construction, so the reduced step is
unconditionally stable regardless of `m` (measured ρ(Ac) = 0.9906 / 0.9977).

### 3.2 Basis

`hybrid` = block-Krylov of `S` seeded with the images of the temperature
sources (`A⁻¹ P`, `A⁻¹ Zs`, `A⁻¹(d+g)`, `A⁻¹ 1`, `C^½ Zs`, `C^½ P`) for the first
m/2 columns, then POD of the 720 logged fields for the rest. `krylov` and `pod`
are also available; `pod` from the fields alone is as accurate as `hybrid`, and
Krylov alone is much worse (see §5.4).

### 3.3 Per-step cost

O(m² + m n_zone + m k + k³) dense flops, no sparse solve. m = 400:
≈360 kflop, 12 small numpy kernel launches.

---

## 4. Fidelity harness (`fastsim/record.py`, `fastsim/replay.py`)

`record` wraps `FiniteDifferenceSolver.solve` on the live solver instance and
logs, per step, exactly what the step consumed and produced:

`q_zone` (W, n_zones), `G` and `T_s` (per VAV), `T_amb`, `h`, the zone means
entering the step, the zone means leaving it, the reward, and the full field
every 4 steps (720 frames) for the POD basis. Nothing is recomputed from
policy state; `q_zone` reproduces `input_q` exactly (7e-15 W).

`replay` drives the ROM open-loop from the real initial field with only those
inputs, and reports zone-mean error at 1 h / 1 day / 10 days, per zone, plus
the air-side delivered heat.

---

## 5. Measured fidelity

### 5.1 Zone-mean temperature error (K), fairbanks_8, convection on

| m | 1 h max | 1 day max | 10 d max | 10 d mean | 10 d RMS | 10 d bias | bldg-mean max | air-heat |
|---|---|---|---|---|---|---|---|---|
| 50 | 0.041 | 0.512 | 1.295 | 0.333 | 0.408 | −0.332 | 0.560 | −3.3 % |
| 100 | 0.004 | 0.472 | 1.233 | 0.317 | 0.396 | −0.308 | 0.540 | −3.0 % |
| 200 | 0.002 | 0.450 | 1.168 | 0.315 | 0.398 | −0.303 | 0.541 | −3.0 % |
| 400 | 0.002 | 0.449 | 1.175 | 0.315 | 0.397 | −0.302 | 0.538 | −3.0 % |

### 5.2 Zone-mean temperature error (K), atlanta_2, convection on

| m | 1 h max | 1 day max | 10 d max | 10 d mean | 10 d RMS | 10 d bias | bldg-mean max | air-heat |
|---|---|---|---|---|---|---|---|---|
| 50 | 0.034 | 0.161 | 1.217 | 0.221 | 0.293 | −0.221 | 0.538 | −4.9 % |
| 100 | 0.001 | 0.112 | 1.055 | 0.205 | 0.272 | −0.205 | 0.504 | −4.6 % |
| 200 | 0.001 | 0.107 | 1.025 | 0.203 | 0.269 | −0.203 | 0.501 | −4.5 % |
| 400 | 0.001 | 0.106 | 1.007 | 0.200 | 0.267 | −0.200 | 0.496 | −4.5 % |

`air-heat` = (Σ|q_a| ROM − Σ|q_a| real)/Σ|q_a| real over 10 days, q_a the
per-VAV air-side power; the real gross is 6.16 MWh (fairbanks, 30 VAVs, 10 d)
and the signed totals agree to +3.0 % / +4.5 %. The B7 envelope is flat after
about day 3 — the ROM has a slightly different equilibrium, it is not diverging.

Plot: `docs/fastsim_fidelity.png` (also `fastsim/plots/fidelity.png`), max
and mean error envelope vs simulated day, both buildings, m = 100 and 400
nested — the two m are indistinguishable, and the dotted line is the 0.1 K
target at 10 days.

### 5.3 Stochastic convection is irrelevant at this accuracy

Re-recording both scenarios with `building._convection_simulator = None`:

| | m | 1 h | 1 day | 10 d max | 10 d mean |
|---|---|---|---|---|---|
| fairbanks noconv | 100 | 0.004 | 0.473 | 1.234 | 0.317 |
| fairbanks noconv | 400 | 0.002 | 0.450 | 1.176 | 0.315 |
| atlanta noconv | 100 | 0.001 | 0.112 | 1.056 | 0.205 |
| atlanta noconv | 400 | 0.001 | 0.106 | 1.009 | 0.200 |

Identical to the 3rd decimal in every horizon. Treating convection as identity
(which is exact for zone means, since it is a within-room permutation) is
therefore justified at this level: the 4 % of CPU it costs buys nothing that
survives to the zone mean. A convection-aware model would have to reproduce the
within-room *gradients*, which cost ≥1 K of fidelity elsewhere.

### 5.4 What the residual is (diagnostics)

| probe | result |
|---|---|
| one-step closure error from the **true** state (m=200) | 0.014 K mean, 0.122 K max |
| steady ambient gain, zone mean, m=100/200/400 | 1.0000 K/K, error **0.0000** |
| steady local zone-heat gain, m=100/200/400 | 30.8 / 8.6 / 4.6 % relative error |
| state projected, exact solve every step (m=200) | 0.002 K max over 10 days |
| POD-only basis, m=200 | 1.174 K (hybrid 1.168) |
| Krylov-only basis, m=400 | 3.890 K |
| temperature-norm POD, m=200 | 1.172 K |
| per-zone heat-bias fit (30 params, 1st half), λ=0.25/0.5/1.0 | 1.251 / 1.333 / 1.498 K — *worse* |

Reading: the basis represents any *given* field to 0.04 K and reproduces zone
means to 2e-3 K when the state is exact, so the drift is not an output error.
It is the **out-of-basis part of the state entering the `(C/dt)⊙T` forcing**:
the fields are well represented in the energy norm but the step reads all CVs
through `C/dt`, and the part of `C/dt·T` outside `span(C^½W)` has a small
coherent component that integrates into a −0.3 K (−0.2 K atlanta) cold bias over
10 days. It is flat in m because m = 100–400 already captures the fields (§5.4 also shows the same 1.17 K
from three unrelated bases). The one step that would
remove it is a **refit of the four reduced maps by least squares on exact
transitions** (operator inference), which is not attempted here; the naive
heat-input parameterisation of the residual over-corrects by 3x and is reported
as a negative result.

### 5.5 Closed loop (`fastsim/closedloop.py`, m = 200, 2880 steps, CtrlW)

The real environment with only `solver.solve` replaced: CtrlW, the air handlers,
the VAVs, the hot-water loop, the occupancy model and the reward are the shipped
code, acting on the ROM's field.

| scenario | real total reward | ROM total reward | error |
|---|---|---|---|
| building_train_8/fairbanks_8 | −66.138282 | −67.879926 | −2.63 % |
| building_train_1/atlanta_2 | −9.940480 | −10.433095 | −4.96 % |

Both are *more negative*, as expected from the cold bias (the ROM over-states
comfort regret). Whether that bias is consistent enough to preserve a ranking is
the question §11 answers, on five CtrlW variants: it does.

---

## 6. Speed (Apple M-series, one BLAS thread, machine also running a 7-process
CMA-ES campaign, so absolute numbers are pessimistic and the *ratio* is the
result)

> **Superseded in part by §12.** These timings were taken under a saturated
> machine. On a quiet machine the solve is 13.7 ms/step (49 % of a step) and an
> end-to-end episode gains 1.19x, not the ~40x the thermal ratio suggests. The
> ratio between the reduced step and the solve (14–40x depending on load) and
> the batched projections stand.

| | building_train_8 | building_train_1 |
|---|---|---|
| one SuperLU solve, 1 RHS | 19.9 ms | 27.7 ms |
| forcing vector work per step | 2.1 ms | 2.8 ms |
| **real thermal step (2 solves + forcing)** | **42.0 ms** | **58.2 ms** |
| factorization (once) | 0.74 s | 1.29 s |
| `QᵀA⁻¹P` response (once per factorization, block of k) | 0.62 s | 1.31 s |
| measured whole-env step CPU in the rollout (incl. CtrlW) | 40.3 ms | 63.4 ms |
| ROM step, m = 200, numpy single core | 0.98 ms | 1.43 ms |
| ROM env-steps/s per core, m = 200 | 1,024 | 698 |
| speedup, thermal part | 43x | 41x |

Basis + reduction build (one-off, per building and m, includes the m
right-hand-side solves): 49 s / 104 s / 165 s (fairbanks m = 100/200/400),
108 s / 124 s / 277 s (atlanta). Memory: `W` is
n x m float64 = 260 MB (m = 400) / 476 MB; the reduced matrices are m² = 1.3 MB.

The 0.98 ms is ~20x more than the 149 kflop it performs: it is 12 numpy
dispatch-bound kernels of 10–50 µs. Fusing the four maps into one
`(m + k) x (m_cat)` matmul removes most of it.

**Batched (the reason this exists).** Same matrices, B environments of one
building — one `(B,m) @ (m,m)` per map:

| B | m | GFLOP/step | MB/step | flop/byte |
|---|---|---|---|---|
| 1 | 400 | 0.0004 | 1.7 | 0 |
| 64 | 400 | 0.028 | 1.9 | 15 |
| 1024 | 400 | 0.454 (fbk) / 0.517 (atl) | 5.2 / 5.4 | 87 / 96 |

At B = 1024 and m = 400 the thermal step is ~0.5 GFLOP and ~5 MB per step: on a
ROCm GPU at even 5 TFLOP/s effective that is ~0.1 ms/step for the whole
1024-env batch, i.e. ≥10^4 env-steps/s before the per-env Python work. The real
simulator does 24 env-steps/s per core (42 ms), so the Mac fleet the campaign is
using (~7 processes) is ~170 env-steps/s.

---

## 7. Recommended architecture for the full fast environment

Reuse, do not port:

* **Reuse the extraction path.** `fastsim.extract` is the only thing that must
  ever touch the real simulator: build the env once per building, dump
  `A`, `P`, `Qᵀ`, `C/dt`, `d+g_amb`, `M_c`, zone CSR, then build `W` and the
  reduced maps. Cache them keyed by (building, weather's `h`, dt, m).
* **Reuse the reduced step as the thermal kernel**, batched over environments of
  the same building: `b = Ac@s + Aq@q + dv*Tamb + Mr@u_m` and
  `s = b + Pr@q_a`. All maps shared, all state `(B, m)`. This is 1–2 days of
  PyTorch (CPU now, ROCm later) work. It does **not** get 40x on its own: §12
  measures the solve at 49 % of a step on a quiet machine, so the kernel alone
  buys ~2x and the batching below is what buys the rest.
* **Do not translate the protobuf/sensor scaffolding.** It is 51 % of the step
  (21 % other simulator work + 30 % reward/observation/protobuf, §12)
  and every bit of it is indirection around dicts of readings. Replace with
  plain batched arrays: zone temps `(B,n_zones)`, VAV flows/dampers
  `(B,k)`, AHU setpoints `(B,n_ahu)`, and the reward as tensor ops. The
  `--components parity` double call (each boiler read advances the tank) must
  not be reproduced: use the hook semantics, which are the graded ones.
* **Keep the stochastic convection out.** §5.3 measures it at <1e-3 K effect on
  zone means. It is 4 % of the real CPU and pure cost in the surrogate.
* **Do batch across envs of the same building only.** Different buildings have
  different `W` and different `k`; group them (a dict of batched tensors per
  building) rather than padding.
* **Fidelity gate in CI.** Keep `fastsim.replay` as the regression test: a new
  building/weather must reproduce a recorded rollout to <2 K max at 10 days
  before it is used for tuning.

Effort estimate: batched thermal kernel 1–2 d; batched HVAC / VAV / AHU / boiler
/ hot-water port 3–5 d; reward + occupancy port 2–3 d; fidelity harness and
regression 2 d; ROCm bring-up + tuning 2 d. **~2 weeks** for a training-grade
batched environment, ~3–4 days for a tuning-grade one (replay the scenario's
logged control inputs, no RL observation path).

Where the remaining accuracy would come from, in order of expected payoff:
(1) operator inference — refit `Ac, Aq, dv, Mr` on exact transitions (one
shot, 2880 samples vs 462 unknowns at m = 200, ridge for the rest), validated
on a held-out weather for the same building; (2) put the *forcing-relevant*
part of the state in the basis, i.e. build `W` so that `span(C/dt · W) ` is
captured — a `C/dt`-weighted Krylov rather than a `C^½`-weighted one;
(3) m ≳ 2000 (3.2 GB per building) — not attractive.

---

## 8. Commands

```bash
OP=starter-kit/.venv/bin/python        # run from the workspace root
cd sims/shipped                        # every fastsim entry point chdirs via simload

# 1. extract the operators from the real solver (once per building, ~10 s)
PYTHONPATH=../.. $OP -m fastsim.extract building_train_8/fairbanks_8 building_train_1/atlanta_2

# 2. record an instrumented rollout (2880 steps, ~40-65 ms CPU/step)
PYTHONPATH=../.. $OP -m fastsim.record building_train_8/fairbanks_8 --steps 2880
PYTHONPATH=../.. $OP -m fastsim.record building_train_8/fairbanks_8 --steps 2880 --no-convection

# 3. open-loop fidelity (builds the ROM, replays the logged inputs)
cd ../..
$OP -m fastsim.replay building_train_8/fairbanks_8 --m 50,100,200,400 --kinds hybrid
$OP -m fastsim.replay building_train_1/atlanta_2 --m 400 --err-dir fastsim/plots/err
$OP -m fastsim.plot                      # fastsim/plots/fidelity.png

# 4. exact replay (validates the extraction) and the residual diagnostics
$OP -m fastsim.fit building_train_8/fairbanks_8 --m 200

# 5. closed loop: real env, ROM thermal step, identical reward code
cd sims/shipped && PYTHONPATH=../.. $OP -m fastsim.closedloop building_train_8/fairbanks_8 --m 200 --steps 2880

# 6. costs
cd ../.. && $OP -m fastsim.bench building_train_8 building_train_1 --m 100,400 --envs 64,1024
```

`fastsim/` layout: `scen` (env construction), `extract` (operators),
`record` (instrumented rollout), `rom` (basis + reduced step), `replay`
(fidelity), `fit` (exact replay + residual), `closedloop`, `bench`, `plot`.
Data lands in `fastsim/ops/<building>/`, `fastsim/data/<bld>__<wx>/<conv|noconv>/`,
`fastsim/plots/`. `fastsim/ops` is 4 MB (regenerate in ~10 s); `fastsim/data` is
1.8 GB because each rollout stores 720 full float32 fields for the POD basis
(regenerate with step 2, ~6–12 min per rollout); `fastsim/data` can be deleted
once the basis is rebuilt, and `fastsim.replay` is the only consumer of it.

---

## 9. Open issues

1. **10-day open-loop target missed** (1.0–1.2 K vs 0.1 K). Cause identified
   (§5.4): out-of-basis state in the `C/dt` forcing; not basis size, not
   convection, not the air side. Unblocked by operator inference on the reduced
   maps, which needs a second weather for the same building to validate.
2. **Reward bias is one-sided** (−2.6 % / −5.0 %, always worse) and differs
   between buildings, so absolute scores and cross-scenario comparisons are
   unusable without a per-scenario calibration offset. The *ranking* question is
   settled for two buildings and five variants (§11: Spearman 1.000); it is not
   settled for a full dev30 sweep or for a policy family as different as
   bang-bang vs CtrlW.
3. **Wall-clock savings are not yet end-to-end.** Measured on a quiet machine
   (§12) the solve is 49 % of a step, not 90 %: swapping it out gives 1.19x.
   The other 51 % (HVAC 21 %, reward+observation+protobuf 30 %) is the critical
   path for the port and must be batched over environments, not micro-optimised.
4. Everything here is `sims/shipped`. `sims/corrected` (PR #2/#3) changes the
   water loop, not the thermal operators, so the operators and basis carry over
   unchanged; the reward path differs and would need re-verification.
5. Absolute timings in §6 were taken with a 7-process CMA-ES campaign running;
   re-measure on a quiet machine before quoting them to anyone.

---

# Part II — operator inference, policy ranking, and the real cost of a step

Added after the first round. Three questions: does refitting the reduced maps
fix the 1 K open-loop error; does the surrogate rank policies the way the real
simulator does; and what does a step actually cost once the solve is gone.

## 10. Operator inference does not help (negative result)

`fastsim/opinfer.py` fits the four reduced maps by ridge least squares on exact
transitions of the *fitting* strip. The right-hand side is the exact
pre-air-side solution `b_k = Wᵀ C^½ A⁻¹(C/dt T_k + q_k + T_amb(d+g) + M_c T_m,k)`,
read off the real factorization along the real trajectory; the regressors are
the exact reduced state and the logged inputs; the correction is fitted relative
to the projected maps so the ridge penalty bounds it. The basis is built **only**
from the fitting strip's fields, and validation is a **different weather strip of
the same building**.

Setup: fit on `building_train_8/fairbanks_8` (heating), validate on
`building_train_8/denver_6` (285.9–309.2 K, i.e. cooling — the opposite regime),
m = 200, 2880 transitions, 462 features.

| ridge λ | ‖ΔAc‖/‖Ac‖ | ρ(Ac) | fit strip 1 h / 1 d / 10 d (max K) | denver_6 1 h / 1 d / 10 d (max K) | denver mean |
|---|---|---|---|---|---|
| projected (no fit) | — | 0.9906 | 0.002 / 0.450 / **1.168** | 0.023 / 0.519 / **0.553** | **0.123** |
| 1e-6 | 0.201 | 0.9927 | 1.454 / 15.43 / 103.4 | 1.56 / 28.9 / 34.7 | 5.42 |
| 1e-4 | 0.576 | 1.0085 | 3.89 / 170.6 / 5.2e12 | 4.23 / 871 / 4.3e13 | diverged |
| 1e-2 | 0.375 | 0.9906 | 5.86 / 22.5 / 66.1 | 6.18 / 82.8 / 97.7 | 21.0 |
| 1e-1 | 0.046 | 0.9906 | 0.78 / 2.64 / 3.10 | 0.80 / 5.58 / 6.99 | 1.67 |
| 1 | 0.005 | 0.9906 | 0.076 / 0.338 / 0.942 | 0.092 / 0.986 / 1.137 | 0.225 |
| 10 | 0.0007 | 0.9906 | 0.046 / 0.265 / 0.930 | 0.067 / 0.550 / 0.621 | 0.150 |

Reading, in order of what it establishes:

1. **Weak ridge is a trap.** With λ ≤ 1e-2 the correction is fitted in
   directions the trajectory barely excites (the design matrix is strongly
   collinear: the state spends its time on a handful of slow modes), and it
   either destroys the slow eigenvalues (λ = 1e-4 gives ρ = 1.0085 and the model
   diverges) or wrecks the transient within one hour. 462 features from 2880
   correlated samples is not enough conditioning to identify a 200x200 matrix.
2. **Strong ridge only re-learns the projection badly.** At λ = 10 the
   correction is 0.07 % of `Ac`, and it buys 1.168 → 0.930 K on the strip it was
   fitted on while costing 0.553 → 0.621 K on the holdout. That is
   regime-specific overfitting: the correction encodes a fairbanks-specific
   residual, not a property of the operator.
3. **A delay-embedded variant is worse.** Adding `s_{k-1}` to the regressors
   (`s_{k+1} = A1 s_k + A2 s_{k-1} + ...`, the standard remedy for unmodelled
   fast dynamics) gives 5.6 K error within one hour at every λ tried
   (1e-2…1e2), because the lag introduces a phase error the fit turns into a
   near-unit pole pair.

So the residual is **not** an operator error that data can correct. It is a
non-Markovian effect in the chosen state: the missing information is the part of
the field outside `span(W)`, which is not a function of `s` and therefore cannot
be recovered by a linear map from `s`. This is consistent with §5.4 — a
projected state reproduces zone means to 2e-3 K, so the fields are fine, while
the step's `C/dt·T` forcing still sees the missing part.

The one genuinely good news in the table: the projected model **generalises
across weather better than it fits its own strip** — 0.553 K max / 0.123 K mean
on an unseen cooling strip against 1.168 K / 0.315 K on the strip its basis came
from. A fairbanks-winter basis is not over-specialised.

Recommendation: keep the projection, drop the operator refit. If the 0.1 K
target is required, the next thing to try is a **state** extension (extra latent
variables from a subspace identification, or DMDc on an augmented state built
from the air-side history), not an operator correction.

## 11. Policy ranking: the bias cancels (the decisive result)

`fastsim/ranking.py`. For each building a fixed surrogate (m = 200, basis from
that strip's fields, projected maps) replaces *only* `FiniteDifferenceSolver.solve`
inside the real environment; five CtrlW parameter variants then run for 2880
steps with the real solve and with the ROM solve. The real-solve "default" row
reproduces the recorded rollout total bit-for-bit (−66.138282), which validates
the harness.

| building | variant | real total | ROM total | Δ | Δ % | rank (real / ROM) |
|---|---|---|---|---|---|---|
| train_8 | margin0.0 | −65.888748 | −67.621976 | −1.7332 | 2.63 % | 1 / 1 |
| train_8 | default | −66.138282 | −67.879926 | −1.7416 | 2.63 % | 2 / 2 |
| train_8 | preheat1h | −66.474684 | −68.236446 | −1.7618 | 2.65 % | 3 / 3 |
| train_8 | standby285 | −67.561469 | −69.320010 | −1.7585 | 2.60 % | 4 / 4 |
| train_8 | margin0.4 | −67.740622 | −69.403166 | −1.6625 | 2.45 % | 5 / 5 |
| train_1 | default | −9.940480 | −10.433095 | −0.4926 | 4.96 % | 1 / 1 |
| train_1 | standby285 | −9.940480 | −10.433095 | −0.4926 | 4.96 % | 1 / 1 |
| train_1 | margin0.0 | −10.264238 | −10.747526 | −0.4833 | 4.71 % | 3 / 3 |
| train_1 | preheat1h | −10.311724 | −10.834938 | −0.5232 | 5.07 % | 4 / 4 |
| train_1 | margin0.4 | −10.754050 | −11.222602 | −0.4686 | 4.36 % | 5 / 5 |

* train_8: **Spearman 1.000, Kendall 1.000** (p = 1.4e-24 / 0.017), best and
  worst both correct. Δ = −1.732 ± 0.036 on a real spread of 1.852 — the bias
  is 2.1 % of its own mean and 94 % of the entire policy spread.
* train_1: **Spearman 1.000, Kendall 1.000**, best and worst correct.
  Δ = −0.492 ± 0.018 on a spread of 0.814. (`standby_minimum = 285` is exactly
  a no-op on this warm strip — identical to 6 decimals — so 5 variants give 4
  distinct values and a tie.)
* Pooled over 10 points: Spearman 1.000, Kendall 1.000, worst relative reward
  error 5.07 %.

The absolute bias is large (2.5–5 %) and *as large as the whole spread between
variants*, yet the ordering is exact: the bias is a function of the scenario, not
of the policy. That is the property a tuner needs, and it is exactly what the
cold-bias diagnosis in §5.4 predicted — every policy pays the same ~0.3 K.

Caveat, and it is the important one: the bias **differs between buildings**
(2.6 % vs 4.9 %), so a score that averages over scenarios (dev30, the leaderboard
score) is shifted by the average of those per-scenario biases. Two consequences:
do not read absolute fastsim scores; and if two candidates are within a few
percent of each other on the *mean*, the real simulator has to separate them.

## 12. What a step actually costs once the solve is gone

Timed on the Mac with the campaign stopped, 300 steps of
`building_train_8/fairbanks_8`, one BLAS thread, instrumented (ms/step):

| component | ms/step | share |
|---|---|---|
| thermal solve (two SuperLU triangular solves) | 13.70 | 49 % |
| rest of `execute_step_sim` (HVAC, VAV, zone power, convection) | 6.0 | 21 % |
| reward + observation assembly + protobuf | 8.4 | 30 % |
| CtrlW policy action (in addition, outside `env.step`) | 1.8 | — |
| **total `env.step`** | **28.1** | |

This **corrects §6**: the earlier 40–58 ms/step figures were taken while a
7-process campaign saturated the machine (the same 2880-step episode now costs
36.7 ms/step, and `bench`'s 19.9 ms single solve was measured under that load).
The ratio between the reduced step and the solve is unchanged (0.98 ms vs
13.7 ms, 14×), but the **end-to-end** gain from swapping the solve out is far
smaller than §6 implied:

| wiring (identical 2880-step fairbanks episode, back to back) | wall | CPU | per step |
|---|---|---|---|
| real solve | 118.2 s | 105.8 s | 36.7 ms |
| ROM solve | 99.1 s | 86.5 s | 30.0 ms |

**1.19x.** (The first version of the closed loop measured 30 ms on top of this,
because it rebuilt each zone's CV coordinates from lists of tuples every step;
that is now a precomputed 0/1 map and one sparse matvec. Worth stating because
it is the same trap the full port will hit.)

So for any fast environment the thermal kernel is only half the problem: 15 ms
of Python/protobuf scaffolding remains per step, and it must be **batched over
environments**, not merely reused. That is what turns the 0.1 ms/env-step the
GPU projection (§6) promises into throughput, and it is where the effort goes.

## 13. Go / no-go

**(a) fastsim inside CMA-ES, as it stands.** GO, with conditions.

* Ranking is exact on the evidence we have: Spearman = Kendall = 1.000 on both
  buildings across five variants whose real reward spread is 0.8–1.9, with a
  policy-independent bias of 2.5–5 %. Use it to *select and order candidates*.
* Do **not** feed fastsim numbers into the leaderboard score path or compare
  across scenario sets: the bias is per-scenario (2.6 % vs 4.9 % here) and would
  need a per-scenario calibration offset, measured by running the incumbent
  through both sims.
* Speed is the blocker, not accuracy: 1.19x end-to-end today. A drop-in
  replacement is not worth wiring into the campaign.
* Cheapest useful form for the campaign right now: use fastsim as an *offline
  filter* — run a large candidate set through the ROM-only path (no
  `env.step`, just the ROM + the policy's own logic) to shortlist, then spend
  real-sim CPU on the shortlist. That avoids the protobuf scaffolding entirely.

**(b) torch/ROCm port for January.** GO, but re-scope it.

* The thermal kernel is the easy half and is already specified (§3, §6): maps
  are (m x m) and shared per building, state is (B, m), 0.45–0.52 GFLOP and
  ~5 MB per 1024-env step. 1–2 days.
* The critical path is the **batched non-thermal step**: VAV/AHU/boiler/hot-water
  physics, the reward function, the occupancy model, and the observation
  assembly, all as (B, ...) arrays with no protobuf and no per-device dicts.
  Budget 7–10 days, not 4. On the evidence of §12, a port that keeps the
  existing scaffolding in Python buys ~1.2x and nothing more.
* Fidelity gate before any training run: `fastsim.replay` on a recorded rollout
  (< 2 K max at 10 days, already met: 0.55–1.17 K) plus a ranking check
  (`fastsim.ranking`, Spearman 1.0) on at least two buildings.
* Do not bank on 0.1 K open-loop fidelity: the projection floor is ~1 K at
  10 days, and §10 shows the obvious fixes do not work. For RL the relevant
  property is the reward bias, which is measurable and currently policy-independent.

## 14. Commands added

```bash
OP=starter-kit/.venv/bin/python
# operator inference: fit on one strip, validate on another (fastsim/opinfer.py)
$OP -m fastsim.opinfer --pairs building_train_8/fairbanks_8:building_train_8/denver_6 \
    --m 200 --lam 1e-6,1e-4,1e-2,1e-1,1,10 --out /tmp/oi.json

# policy ranking: real vs ROM, 5 CtrlW variants, Spearman/Kendall
$OP -m fastsim.ranking --buildings building_train_8 --steps 2880 --m 200 \
    --out fastsim/results/ranking_btrain8.json
$OP -m fastsim.ranking --buildings building_train_1 --steps 2880 --m 200 \
    --out fastsim/results/ranking_btrain1.json

# cost attribution / episode timing (real vs ROM, same wiring)
cd sims/shipped && PYTHONPATH=../.. $OP -m fastsim.closedloop \
    building_train_8/fairbanks_8 --steps 2880            # ROM
cd sims/shipped && PYTHONPATH=../.. $OP -m fastsim.closedloop \
    building_train_8/fairbanks_8 --steps 2880 --real-only # real
```

New modules: `opinfer` (ridge refit + the two negative variants),
`ranking` (the ranking harness). New data: `fastsim/data/*__denver_6,7/`
(the two holdout strips, 2880 steps each).
