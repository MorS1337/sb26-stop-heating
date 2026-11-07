# `diffsim` — an exact differentiable twin of the Smart Buildings step

The twin reproduces the fused C step (`cenv/src/step.c`) formula for formula and
carries a reverse pass (adjoint) for every continuous quantity, so the gradient
of an episode objective with respect to the *whole* action trajectory costs one
more sparse solve per step — about 1.5× a forward pass.

It is a port, not a model: the same order of operations, the same boiler sensor
reads (whose side effects are physics on the shipped tree), the same
thermostat/damper/valve logic, the same `apply_thermal_power_zone` assignment,
the same Woodbury air side, and the simulator's *own* reward path.

## What it reuses (never modifies)

| piece | source |
|---|---|
| environment construction, gin, scenario paths | `sbx.evalrun._load_environment`, `sbx.scenarios` |
| the exact step, for the gate's reference | `cenv.fused.FusedState` (one C call per step) |
| the thermal system `A`, and its factorisation | `simulator._solver` + `cenv.solver.Factor` (Cholesky, same SPD matrix) |
| air-side operators `P`, `Q` | `solver._applied`, `solver._sensed`; `R = A⁻¹P` and `S = Qᵀ A⁻¹ P` computed once per episode |
| convection permutations | `cenv_conv_apply` run on an *index field* with a copy of the kernel's MT19937 state, so the exact draw stream is reused rather than re-implemented |
| occupancy for the **gains** | a copy of the kernel's C occupancy handle |
| occupancy for the **reward** | the simulator's Python chain (`--cenv fused` does not install the stochastic component, and the flexible-geometry class overrides the method cenv patches, so the reward's occupancy is the Python one) |
| reward arithmetic | `setpoint_energy_carbon_regret.SetpointEnergyCarbonRegretFunction` semantics, incl. the float32 `RewardInfo`/`RewardResponse` protos |

## Working space

The state is the *solved* subspace (`n = 148 725` of `n_full = 205 434` control
volumes on `building_train_1`; the rest are Dirichlet nodes pinned to ambient,
never read — no zone contains one). Carried state: `T`, `q`, interior-mass `M`,
tank temperature, and the hot-water return temperature (which is carried only
while no water flows). Convection is a per-step sparse permutation on the solved
destinations (~12–30 k entries per step), precomputed once.

Per step the twin stores a *small* tape (VAV/AHU/HWS scalars, the `k = 39`
air-side vectors, the zone means) — tens of kilobytes — so a 2016-step episode's
tape fits in a few tens of megabytes.

## Gates

    # forward: the twin's step against the fused kernel, closed loop, both trees
    starter-kit/.venv/bin/python -m diffsim.gates.forward \
        --sim corrected --scenario building_train_1/fairbanks_1 \
        --steps 600 --offset-days 2

    # gradient: the adjoint against central finite differences
    starter-kit/.venv/bin/python -m diffsim.gates.gradient --steps 6 --start 300

Measured (results/xopt_gate_*.json, 600 steps, CtrlW5 actions, offset 2 so the
plant, the chiller and the reward are live):

| tree | scenario | temp | mass | tank | reward total |
|---|---|---|---|---|---|
| corrected | building_train_1/fairbanks_1 | 7.4e-13 | 6.9e-13 | 0 | **0.0 (bit-exact)** |
| shipped | building_train_1/fairbanks_1 | 7.3e-13 | 6.8e-13 | 0 | **0.0** |
| shipped | building_train_8/fairbanks_8 | ≤1e-12 | ≤1e-12 | 0 | **0.0** |
| corrected | building_train_8/fairbanks_8 | ≤1e-12 | ≤1e-12 | 0 | **0.0** |
| corrected | building_train_1/phoenix_5 (cooling) | ≤1e-12 | ≤1e-12 | 0 | **0.0** |
| shipped | building_train_1/phoenix_5 (cooling) | ≤1e-12 | ≤1e-12 | 0 | **0.0** |

(Reward exactness needs the float32 quantisation of the proto fields *and* of
`agent_reward_value`; without it the per-step reward differs by ~1e-7 relative
and the episode total by ~1e-9 — the arithmetic of a float32-derived number.)

Gradient: the *forward* is exact, the adjoint is not yet at 1e-6. Measured on a
one-step horizon, per single action, central differences with ε = 1e-6:
dampers 5.3e-3, valves 7.7e-3, supply-air setpoint 0 (no effect while the
supply is mixed-air limited), recirculation 5.3e-3, hot-water setpoint 5.5e-3,
pump head 2.2e-1 relative. Signs are right and the residual is a small missing
term in the energy chain, not a structural error — the optimiser still descends.
**This is the one acceptance item the twin does not meet**, and the next step is
to localise that term (see `docs/exact_opt.md`).

## Cost

Per step, `building_train_1` (n = 148 725, k = 39), single process:

| phase | ms |
|---|---|
| `A⁻¹ b` (Cholesky substitution through `cenv.solver.Factor`) | 6.3 |
| `R rhs` (dense 148 725 × 39 matvec, forward) or `Rᵀ λ` (adjoint) | 2.4 |
| Adagrad-free device algebra, zone means, convection, `k×k` solve | ~3 |
| **forward** | **~10** |
| **backward** (one solve + one matvec + device VJPs) | **~12** |

so a gradient of a 2016-step episode is ~45 s of CPU: the adjoint costs **~1.2×
the forward**, against 92 × 2016 central differences for the same information.

## Files

* `diffsim/plan.py` — extraction from a built environment (geometry, thermal
  operators, per-step tables, occupancy streams, convection permutations) and
  the on-disk cache under `scratch/ExactOpt/cache`.
* `diffsim/twin.py` — the forward step and its adjoint.
* `diffsim/gates/forward.py`, `diffsim/gates/gradient.py` — the gates.
* `diffsim/replay.py` — drives the fused kernel with an arbitrary command block
  and scores it with the simulator's own reward (the reported numbers).
* `diffsim/optim.py` — projected Adam on the trajectory, the two information
  regimes, and the exact replay of each result.

## Usage

    starter-kit/.venv/bin/python -m diffsim.gates.forward --sim corrected \
        --scenario building_train_1/fairbanks_1 --steps 600 --offset-days 2
    starter-kit/.venv/bin/python -m diffsim.optim scenario --name anchor_corrected \
        --steps 2016 --iters 40 --lr 0.01 --regimes full
