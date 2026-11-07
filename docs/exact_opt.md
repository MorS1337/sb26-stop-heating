# Exact optimum: a differentiable twin of the simulator, and what it is for

Status: the twin is built and its **forward is bit-exact** (reward 0.0 relative
on both trees, states 7e-13). The **adjoint is not yet at the 1e-6 gate**:
measured 4e-3 to 5.6e-2 relative per action block on a one-step horizon, with
two blocks (supply-air setpoint, hot-water setpoint) already exact to 1e-10.
Consequently **the headroom table below is not filled in yet** — see §5 for the
exact resume commands. Everything else is measured.

## 1. What was built (`diffsim/`)

The twin is a port of `cenv/src/step.c` with a reverse pass, not a model:

* one environment step = the same 19 items in the same order, including the
  boiler's supply-water sensor reads, whose *count and position* are physics on
  the shipped tree;
* the thermal solve reuses the simulator's own system: `A = solver.matrix(h)`
  factorised once (`cenv.solver.Factor`, SPD Cholesky), `R = A⁻¹P`,
  `S = QᵀA⁻¹P`; the air side is the same Woodbury update;
* the convection RNG is reused rather than re-implemented: `cenv_conv_apply`
  run on an *index field* with a copy of the kernel's MT19937 state gives the
  per-step permutation exactly;
* the state is the solved subspace only (148 725 of 205 434 control volumes on
  `building_train_1`; the rest are Dirichlet nodes, never read, no zone
  contains one), so a 2016-step episode's tape is tens of megabytes and the
  backward needs no full-field checkpointing at all;
* the reward is the simulator's own path, including the two things a naive port
  gets wrong: the reward's occupancy is the **Python** occupancy chain
  (`--cenv fused` does not install the stochastic component, and
  `SimulatorFlexibleGeometries` overrides the method cenv patches), and every
  `RewardInfo`/`RewardResponse` numeric field is a **float32** proto field, so
  the episode total only matches bit-for-bit after the same rounding.

Cost on `building_train_1` (n = 148 725, k = 39), one process: forward ≈ 10 ms
per step (6.3 ms of it the sparse solve, 2.4 ms the `R rhs` matvec), backward
≈ 12 ms (one transpose solve + one `Rᵀλ` matvec + the small algebra). **The
adjoint costs ~1.2× the forward**, against 92 × 2016 central differences.

## 2. Forward gate (passes, bit-exact)

`python -m diffsim.gates.forward --sim <tree> --scenario <id> --steps 600
--offset-days 2`, closed loop with CtrlW5-262 actions, 600 steps, comparing the
twin against the fused kernel every step (`results/xopt_gate_*.json`):

| tree | scenario | temp rel | mass rel | reward per step | episode total |
|---|---|---|---|---|---|
| corrected | building_train_1/fairbanks_1 | 7.4e-13 | 6.9e-13 | **0.0** | **0.0** |
| shipped | building_train_1/fairbanks_1 | 7.3e-13 | 6.8e-13 | **0.0** | **0.0** |
| shipped | building_train_8/fairbanks_8 | ≤1e-12 | ≤1e-12 | **0.0** | **0.0** |
| corrected | building_train_8/fairbanks_8 | ≤1e-12 | ≤1e-12 | **0.0** | **0.0** |
| corrected | building_train_1/phoenix_5 (cooling live) | ≤1e-12 | ≤1e-12 | **0.0** | **0.0** |
| shipped | building_train_1/phoenix_5 (cooling live) | ≤1e-12 | ≤1e-12 | **0.0** | **0.0** |

Tank and return-water temperatures are exact too. Offset 2 is used because the
shipped anchor starts on a Sunday, when the plant is off and the reward is 0 for
the first ~30 hours; at offset 0 the comparison is vacuous.

## 3. Gradient gate (does not pass yet)

One-step horizon at start = 300 (Monday 07:00, plant on), per action block,
adjoint vs central differences, ε = 1e-6 × the block's own scale:

| block | central FD | adjoint | relative error |
|---|---|---|---|
| damper | +5.914e-03 | +5.890e-03 | 4.1e-03 |
| valve | +4.141e-04 | +4.143e-04 | 3.8e-04 |
| supply-air setpoint | −8.400e-05 | −8.400e-05 | **3.4e-10** |
| fan static pressure | −3.064e-05 | 0 | kink (§4) |
| recirculation | +1.441e-02 | +1.528e-02 | 5.6e-02 |
| hot-water setpoint | −5.923e-03 | −5.923e-03 | **5.5e-11** |
| pump head | −2.373e-05 | −1.848e-05 | 2.2e-01 |

The two exact blocks are the ones that carry the whole gas/burner chain, and the
thermal/state chain is exact by construction (the forward is), so the error is
confined to the air side and the plant: the reheat effectiveness derivative
(`eff` vs air/water capacity rate and supply temperature), the water-flow share
(`rf`), the fan pressure, and the return-water/coil chain. Signs are right —
this is a small missing term, not a structural error — which is why the
optimiser still descends, but **no optimiser result is reported until the gate
passes**.

## 4. Two artefacts, not bugs

* **fan static pressure**: CtrlW5 holds it at exactly the `min_sp = 200 Pa`
  clamp, so `press = max(fan_sp, min_sp)` has a kink there; central differences
  see the right-hand derivative, the twin's mask returns the left subgradient
  (0). The optimiser will therefore be parameterised in `press ≥ min_sp`
  directly, which removes the kink and the stationary point.
* **shipped boiler**: the reward's gas is
  `cp·ρ·flow·(max(sp, T_ret) − T_ret) + dissipation + Cw·ΔT_tank/Δt` with the
  tank advanced once per step by the *observation* read (the prelude and
  return-water reads have zero duration because the action timestamp equals the
  observation timestamp) — i.e. B1 is reproduced exactly, which is what makes
  the shipped-tree optimiser a valid probe of it.

## 5. Baselines measured (2016 steps, `diffsim.replay` / `capture`, CtrlW5-262)

| scenario | tree | total reward | mean per-step |
|---|---|---|---|
| building_train_1/fairbanks_1 | corrected | −79.339950 | −0.039355 |
| building_train_1/fairbanks_1 | shipped | −80.305560 | −0.039834 |
| building_train_1/phoenix_5 | corrected | −24.566706 | −0.012186 |
| building_train_1/singapore_8 | corrected | −13.680045 | −0.006786 |
| building_train_2/atlanta_3 | corrected | −4.389119 | −0.002177 |
| building_train_7/denver_5 | corrected | −4.615326 | −0.002289 |

(CtrlW5-262 on the corrected anchor is *better* than the shipped one by 0.97
units, as expected: the corrected tree removes B2's free heat.)

## 6. The headroom table (pending the gradient gate)

The optimiser is written and ready (`diffsim.optim`): projected Adam on the
whole 2016 × 92 trajectory inside the action bounds the environment's own
normalisers enforce, warm-started from the CtrlW5-262 trajectory, two
information regimes — `full` (the true reward against the realised occupancy)
and `band` (comfort as a hard constraint on every zone in its schedule band, i.e.
no per-zone occupancy information) — and every reported number comes from
`diffsim.replay`, i.e. the fused kernel and the simulator's own reward, not from
the twin.

Resume, in this order (on the PC, ≤2 heavy processes):

    # 0. gate on the same machine that will optimise (Mac vs PC differ ~5e-6)
    ssh pc "wsl -d Ubuntu -- bash /home/mors1337/sb26/scratch/ExactOpt/pc_job.sh gate"
    # 1. the anchor, both regimes, plus one restart from a different init
    ssh pc "wsl -d Ubuntu -- bash /home/mors1337/sb26/scratch/ExactOpt/pc_job.sh anchor"
    # 2. the warm scenarios (phoenix_5 July, singapore_8, atlanta_3, denver_5)
    ssh pc "wsl -d Ubuntu -- bash /home/mors1337/sb26/scratch/ExactOpt/pc_job.sh warm"
    # 3. the shipped-tree B1/B2 probe
    ssh pc "wsl -d Ubuntu -- bash /home/mors1337/sb26/scratch/ExactOpt/pc_job.sh shipped"


## 6b. Machine cost, measured (this matters for scheduling)

The Windows PC's WSL2 is ~9x slower than the Mac on the sparse solve
(57 ms vs 6.3 ms per substitution) and ~5x on the dense `R rhs` matvec
(11.3 ms vs 2.4 ms), so one 2016-step episode costs there:

| phase | per step | per 2016-step episode |
|---|---|---|
| forward | 149 ms | ~5 min |
| backward | 134 ms | ~4.5 min |
| **one Adam iteration** | 283 ms | **~9.5 min** |

Sixteen iterations (the anchor, both regimes, two processes in parallel) is
~2.5 h; the four warm scenarios plus the shipped probe (10 more regime-runs) is
~7 h at two processes. `scratch/ExactOpt/pclaunch.sh` is the detached launcher
(8 iterations, writes a `.jsonl` history per iteration so a partial run is
readable); pull with

    RS='ssh -o BatchMode=yes -o LogLevel=ERROR'
    rsync -az -e "$RS" --rsync-path="wsl -d Ubuntu -- rsync" \
        pc:/home/mors1337/sb26/results/ results/

Note: on WSL2/glibc the C kernel's teardown corrupts the heap on exit (a
segfault *after* the results are written). `diffsim.optim` and both gates now
end with `os._exit`, and `build()` resets the environment before extracting
(an un-reset environment made that corruption fire earlier). The Mac does not
show it.

## 7. What the optimum should be asked to explain

Once the table exists, the behavioural comparison is scripted from the
trajectory itself: per-hour plant use (the `run` flags and the total flow), the
supply-air temperature and the damper/valve split per zone, which zones are
allowed to float and when (deviation from the band per zone per hour), and when
heat or cooling is bought relative to the electricity price and the carbon rate
(both are in the plan's per-step tables). The cooling prior from
`docs/cool_gap.md` is the check to beat: the chiller is 93–95 % of the bill,
the supply setpoint already sits at its 285 K floor on 92–100 % of live
AHU-steps, recirculation at the 0.70 cap, and the price term makes time-shifting
worth ≤0.91 of 24.6 units on Phoenix — so if the optimum finds cooling headroom,
the mechanism must be named (which actuator, which hours).

## 8. Next step, recommended

Backpropagation through the twin across the generated building pool (a
parametric policy, not a trajectory): the twin is exact, the adjoint is ~1.2× a
forward, and `sbx/scenarios.py` already exposes `gen_train` / `ext_hold`. A
policy with a handful of parameters per (VAV, hour-of-day) — trained through the
twin on `gen_train` and scored on `ext_hold` with the exact simulator — is the
natural next experiment, and it is the only route that turns a per-scenario
open-loop optimum into something submittable. The open-loop table then serves as
the upper bound that says how much of the gap is closed-loop learnability rather
than physics.
