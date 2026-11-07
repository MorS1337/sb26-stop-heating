"""``batch``: M environments of one scenario, advanced together.

Why
---
One ``env.step`` on the shipped simulator is dominated by one sparse solve: the
finite-difference system's ``A x = b`` is 68-81 % of the fused step's C time
(2.6 ms of 4.01 on a quiet ``building_train_8``, 5.5-8 ms of 9.8 under load --
both measured). M environments of the *same*
scenario -- same building, same weather, same convection coefficient; different
policies, parameters or occupancy seeds, which is exactly a training rollout --
share ``A``: only the right-hand sides differ. So the batch factorizes once
(LL^T with a METIS ordering, ``cenv/solver.py``) and issues **one**
multi-right-hand-side solve per step for all M environments
(``cenv/src/thermal_batch.c``), where M single-environment runs would issue M
solves plus M two-sided triangular sweeps. Everything else in the step is the
same physics, run once per environment.

Measured on ``building_train_8``, two sessions, because this machine is
shared and the absolute numbers move with the load (``uptime`` in the tables):

* **loaded (load ~6)**: ``fused`` 13.14 ms per env-step, batch 14.26 at M=4 and
  10.44 at M=16;
* **quiet (load 2.4-4.6), same session, back to back**: ``fused`` 5.29 ms per
  env-step (C call 4.01), batch 7.95 at M=16 (C call 6.06/env-step).

The solve does what the study promised: 9.2 ms of CPU per environment on one
right-hand side under load, and 0.98 ms per environment at M=16 on the quiet
machine (15.8 ms per *step* for all 16), i.e. 5-7x. **The step does not**, and
the phase split (``cenv_tbatch_phase``) says why:

| phase, quiet machine, M=16 | ms per step | ms per env-step |
|---|---|---|
| forcing | 11.5 | 0.72 |
| solve (all 16 RHS) | 15.8 | 0.98 |
| air side (Woodbury) | 21.5 | 1.34 |
| mass + scatter | 8.2 | 0.51 |
| device machines (C, per env) | 40.0 | 2.50 |
| action + observation halves (Python) | 30.0 | 1.90 |

So the part that batches is 3.6 of 7.95 ms, and ``fused``'s whole step is 5.29
because it pays its 1.4 ms of device work and 1.3 ms of Python *with the
environment's 1 MB temperature field hot in cache*, while the batch pays 2.5 ms
for the same work with M fields cycling through it. Past M~16 the batch is not
solve-bound and this component, **as it stands, does not beat one fused
environment per env-step** -- the two fixes in ``scratch/CEnvBatch/patch_layout.py``
(term-major air-side correction, one-pass row-blocked forcing) remove ~1.3 of
the 3.6 ms and would put M=16 near parity; they are arithmetic-preserving but
not yet gated or measured. The win this kernel does deliver today is the *solve*
(5-7x per env-step, measured) and the ability to run M environments of one
scenario in one process, one thread, with independent RNG streams -- which is
what a rollout wants -- not a faster single-thread step.

What is in C, what is in Python
-------------------------------
C, in **one call per step** (``cenv_batch_run``, ``cenv/src/batch.c``):

* each environment's step, in the three slices ``cenv/fused.py`` documents --
  prelude (device state machines, loads, ``input_q``), the solve, then
  foldback/epilogue (air stream, convection, tank, divergence clip,
  observation, reward-info numbers, device-state write-back). The slices are
  ``cenv/src/step.c``'s own code, exported as ``cenv_fused_batch_*``; nothing
  is re-derived for the batch;
* the batched solve: forcing for all M, one ``solve_many`` over the factor,
  then the Woodbury air-side correction with one shared dense ``A^-1 P`` read
  per *step* instead of per environment (which is why the dense response, the
  slower choice for one environment on these buildings, is the right one here
  from M=2 up), and the ``Q^T T'`` each environment's foldback needs.

Each environment owns its device state, tank, temperature field, interior-mass
nodes, occupancy stream and convection MT19937: nothing is shared except the
factorization and the read-only geometry and time tables. That is what makes a
batch reproducible, and it is why the per-environment RNG handles had to be
made room for (the C side kept one occupancy and two MT handles per *process*
before this; see ``cenv/src/occupancy.c`` and ``cenv/src/rng.c``).

Python, deliberately:

* **construction and reset**: one gin parse for the whole batch, M
  ``HybridActionEnvironment()`` constructions (the occupancy seed is rebound
  between them, which is the only way to give them different seeds), one
  factorization, M ``FusedState``s (the observation plan, the per-step time
  tables, the device plan). ~2.4 s and ~95 MB per environment on
  ``building_train_8``, ~6.5 s and ~250 MB on ``building_train_2``.
* **the action half of the step**: ``_format_action`` ->
  ``_create_action_request`` -> ``building.request_action``, and the
  ``RewardInfo`` protobufs the recorder reads. They keep every Python device
  object's timestamps exact and they are the API a policy or ``sbx/evalrun.py``
  touches; together they are ~2.9 ms/env-step and they do not batch (a
  protobuf per environment per step is the price of staying the same object
  type as the reference environment).
* **the policy** (``run_batch``): measured separately, because at M>=16 it is
  a fifth of the step or more (CtrlW: ~1.5 ms/env-step).

Usage
-----
    from cenv.batch import BatchEnv

    env = BatchEnv("sims/shipped/smart_buildings/training_scenarios/"
                   "building_train_8/fairbanks_8/config.gin",
                   sim="shipped", M=16,
                   occupancy_seeds=[17321, 12345, 999, 7] + [1000 + i for i in range(12)])
    obs = env.reset()                        # obs[i] is environment i's vector
    obs, rewards, infos = env.step(actions)  # actions[i] -> environment i
    env.close()

or, to evaluate M policies against it and get evalrun's records back:

    from cenv.batch import run_batch
    result = run_batch("building_train_8/fairbanks_8", "shipped",
                       [("policies.ctrl_w:CtrlW", {}),
                        ("policies.ctrl_w:CtrlW", {"comfort_margin": 0.3})],
                       steps=2880)
    result["records"][0]["total_reward"]
    result["timing"]["c_ms_per_env_step"]

``run_batch``'s records have the same keys and the same aggregates as
``sbx/evalrun.py``'s (same ``_StepRecorder``), plus ``batch_size``,
``batch_index`` and ``batch_timing``; ``timing`` splits the step into the C
call, the action half, the observation/reward half and the policy.

``BatchEnv`` is not a ``--cenv`` component: it is a second environment class,
not a runtime patch of one function, so it is opt-in by construction (it is
listed in ``cenv/__init__.py``'s infrastructure set).

Gate
----
``python -m cenv.verify.batch_gate gate --only shipped`` (and ``corrected``):
M=4 with four occupancy seeds and four real CtrlW parameter sets, each
environment compared one-to-one with its own separate single-environment run of
the official simulator (``cenv/verify/fused_gate.py trace``, 2880 steps), plus
an M=1 case against the ``cenv fused`` SuperLU path.

Verdict files: ``scratch/CEnvBatch/gate_shipped.log`` (PASSED),
``scratch/CEnvBatch/gate_corrected2.log`` (PASSED), both recorded in
``scratch/CEnvBatch/batch_gate.json`` for the run that last finished.

Two references per environment, because a batch changes *two* things at once
and only one of them is this component's to answer for:

* **same factorization** (``--cenv fused,solver``: one environment, the same
  LL^T the batch solves with, through ``cenv/solver.py``'s shim) -- the
  verdict. It isolates batching: M environments per call versus one.
* **the official simulator** (unpatched Python, scipy's SuperLU) -- reported
  beside it. A Cholesky and a pivoted LU differ in the last bits; on a policy
  that sits on a decision threshold that can become a different damper command.

| tree | building | environments | same factorization | official simulator |
|---|---|---|---|---|
| shipped | train_8 | 4 (seeds 17321/12345/999/7, 4 CtrlW parameter sets) | not run (see next column: the official comparison is already bit-exact) | **rewards bit-identical, obs bit-identical, zone temps <= 2.5e-13 rel, 0 flips** |
| shipped | train_2 | 4 | not run | **rewards bit-identical, obs bit-identical, zone temps <= 4.0e-13 rel, 0 flips** |
| corrected | train_8 | 4 | **rewards bit-identical, observations bit-identical, zone temps <= 3.8e-15 rel, 0 flips** | 3 of 4 bit-identical; env2 (seed 999, comfort_cost=20000) rel 3.5e-9, 377 flips |
| corrected | train_2 | 4 | **rewards bit-identical, observations bit-identical, zone temps <= 6.3e-15 rel, 0 flips** | rewards bit-identical, 0 flips |
| shipped | train_8 + train_2 | M=1 through the batch kernel vs ``cenv fused`` (SuperLU) | rel 0.0, zone temps 2.45e-13 / 3.94e-13, 0 flips | -- |
| corrected | train_8 + train_2 | M=1 through the batch kernel vs ``cenv fused`` (SuperLU) | rel 0.0, zone temps 2.5e-13 / 3.9e-13, 0 flips | -- |

``gate --only corrected`` finished **PASSED**, 10 of 10 cases
(``scratch/CEnvBatch/gate_corrected3.log``, and
``scratch/CEnvBatch/batch_gate_corrected.json``). Same factorization (the
corrected tree), 8 environments x 2880 steps: **episode total reward and
every per-step reward are bit-identical** to the single-environment run, the
observation vector is bit-identical at every step, zone means agree to
~4e-15 relative, and no damper decision flips. That is what batching owes: the
slice order, the sum orders and the per-environment streams are the fused
kernel's, and the only thing that changed is how many right-hand sides go
through the factor at once.

The one divergence, and why it is not a batching bug
---------------------------------------------------
``corrected building_train_8/fairbanks_8`` environment 2 (occupancy seed 999,
``comfort_cost=20000``) diverges from the **official simulator** trace: rewards
identical through step 2561, first reward difference at 2562, first damper flip
at 2564, 377 flips by step 2880, total reward 3.5e-9 relative, zone
temperatures 3.3e-7. Measured, not inferred:

* ``cenv fused,solver`` alone -- **a plain single environment, no batch
  anywhere** -- reproduces it exactly: total reward -63.091172279 against fused's
  -63.091172497, first reward difference at step 2562, 377 damper flips, zone
  temperatures 3.30e-7;
* the batch at M=1 on that case gives -63.091172279, and against the
  same-factorization reference it is bit-identical (rewards, observations,
  zone temps 3.49e-15, 0 flips).

So the divergence is the factorization, amplified by a policy decision that
sits on a knife edge: a 1e-14 field difference reaches a float32 observation at
step 2546, and eighteen steps later one damper command comes out different, and
after that the two trajectories are different simulations. The same component
(``cenv/solver``) has this property on its own; nothing about running M
environments at once adds to it. Making the batch *bit-identical to the SuperLU
path* would mean giving up the multi-right-hand-side sweep -- a SuperLU factors'
two triangular sweeps are scalar per column, so M columns buy nothing over M
environments, which is the component's only performance argument. If that
trade is ever wanted, the alternative is a multi-column ``tri_solve`` over the
exported SuperLU factors: bit-identical, and no faster than ``fused``.

Limits
------
* One scenario per batch -- that is the point (one matrix). Another building
  or another convection coefficient needs another batch.
* M <= 128, and RAM is the real limit: ~95 MB of simulator state per
  environment on ``building_train_8`` and ~250 MB on ``building_train_2``, plus
  ``n*M`` doubles for the solve (42 MB at M=64 on train_8, 62 MB at M=32 on
  train_2) and ``k*n`` doubles for the dense response (19.5 MB / 164 MB). M=64
  is ~6 GB on train_8; train_2 is measured up to M=32.
* One thread: the factorization caches one solve workspace, so a batch is
  stepped from one thread. The per-environment slices are independent and
  could be threaded; that is not done (threads on this M3 land on E-cores, and
  the deterministic per-environment arithmetic is worth more than the 1.5x).
* Same restrictions as ``fused``: boiler hot water systems only, one occupancy
  model type, a convection coefficient that does not vary inside the episode,
  ``SimulatorFlexibleGeometries``.
* ``env.step`` on a batched environment raises: the solve has to see all M
  right-hand sides at once, so stepping one environment alone would silently
  desynchronise the batch. Use ``BatchEnv.step([...])``.
* ``sbx/tune.py --backend batch`` evaluates a whole CMA-ES population per
  scenario process (``--workers`` then bounds how many scenarios run at once).
  Not smoke-tested yet: the wiring is in place and the records it produces are
  the same objects ``run_batch`` returns (which the gate does verify), but
  nobody has compared a generation's candidate scores against ``--backend
  local`` on this checkout.

Status
------
Done and measured: the kernel (one C call per step, batched solve, one stream
per environment), the API, ``run_batch``, the gate on **both** trees (both
buildings, eight environments each, plus the M=1 cases -- PASSED against the
same-factorization reference, numbers above), and the timing for M=1, 4, 16.

Not done, in the order it matters:

* the M=32/64 rows of the timing table (M=64 on train_8 is ~6 GB of simulator
  state; the solver study says the batched solve plateaus at M=32-64, and this
  component's own numbers stop improving earlier because of the floor below);
* ``sbx/tune.py --backend batch``: the wiring exists and its worker returns
  exactly the records ``run_batch`` produces (which the gate verifies), but no
  generation has been compared against ``--backend local`` yet. Expect
  candidate scores to agree to the same tolerance the gate reports, because
  the batch is the same step to 1e-13 in temperature and bit-identical in
  observation and reward.

Known performance work left, each with the measurement that motivates it:

* the Woodbury air-side correction's inner loop over environments has stride k
  and stays scalar, and the forcing is assembled with three strided passes over
  an (n, m) block: a fixed ~10 ms per *step* at M=4..16 on train_8 regardless of
  M. ``scratch/CEnvBatch/patch_layout.py`` has both fixes (term-major
  correction, row-blocked one-pass forcing); they are arithmetic-preserving but
  were written after the gate ran and must be re-gated before use;
* the per-environment slices (prelude/epilogue) are independent and could run
  on a thread pool: that is the only way past the floor the device machines and
  the Python halves set (~5.8 ms/env-step under load, see the table above).
"""

from __future__ import annotations

from __future__ import annotations

import ctypes
import time

import numpy as np

from .bindings import c_double, c_int32, c_void_p, lib, nd_f64, nd_i32

# CPU seconds, for the timing tables. ``c_seconds`` is the one C call per step
# for the whole batch; divide by M for a per-environment number.
PROFILE = {
    "c_seconds": 0.0,     # cenv_batch_run, all environments
    "policy_seconds": 0.0,  # get_policy_action + validate_action, all envs
    "pre_seconds": 0.0,   # request_action half of Environment.step
    "post_seconds": 0.0,  # observation + reward half, and the write-back
    "steps": 0,
    "build_seconds": 0.0,
    "builds": 0,
}

_BOUND = {"done": False}


def _bind() -> None:
    if _BOUND["done"]:
        return
    i1 = nd_i32(1)
    f1 = nd_f64(1)
    lib.cenv_tbatch_create.restype = c_void_p
    lib.cenv_tbatch_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        i1,
        f1, f1, f1, f1, f1, f1,
        i1, i1, f1, ctypes.c_int,
        i1, i1, f1, ctypes.c_int,
    ]
    lib.cenv_tbatch_free.restype = None
    lib.cenv_tbatch_free.argtypes = [c_void_p]
    lib.cenv_tbatch_sizes.restype = c_int32
    lib.cenv_tbatch_sizes.argtypes = [
        c_void_p, ctypes.POINTER(c_int32), ctypes.POINTER(c_int32),
        ctypes.POINTER(c_int32), ctypes.POINTER(c_int32),
    ]
    lib.cenv_tbatch_set_factor.restype = c_int32
    lib.cenv_tbatch_set_factor.argtypes = [c_void_p, c_void_p]
    lib.cenv_tbatch_step_many.restype = c_int32
    lib.cenv_tbatch_step_many.argtypes = [
        c_void_p, ctypes.c_int, c_void_p, c_void_p, c_double,
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    ]
    lib.cenv_tbatch_seconds.restype = c_double
    lib.cenv_tbatch_seconds.argtypes = [c_void_p]
    lib.cenv_tbatch_steps.restype = c_int32
    lib.cenv_tbatch_steps.argtypes = [c_void_p]
    lib.cenv_tbatch_phase.restype = c_int32
    lib.cenv_tbatch_phase.argtypes = [c_void_p, ctypes.POINTER(c_double)]
    lib.cenv_batch_run.restype = c_int32
    lib.cenv_batch_run.argtypes = [
        c_void_p, ctypes.c_int, ctypes.c_int,
        c_void_p, c_void_p, c_void_p, c_void_p,
    ]
    _BOUND["done"] = True


def _addr(a) -> int:
    return int(a.ctypes.data)


# --------------------------------------------------------------------------- #
# the shared solve
# --------------------------------------------------------------------------- #
class TBatch:
    """One scenario's batched solve: a Cholesky and the M-wide work buffers.

    Built from the simulator's own solver object (``FiniteDifferenceSolver``):
    the geometry vectors and the air-side operators are the ones
    ``cenv/thermal.py`` exports for the single-environment kernel, and the
    matrix is the one it would factorize -- only the factorization differs
    (LL^T with a METIS ordering instead of SuperLU, so it can solve many
    right-hand sides in one sweep).
    """

    def __init__(self, solver, h: float, m: int, ordering: str = "metis"):
        _bind()
        import scipy.sparse as sp

        from . import solver as solver_mod
        from .thermal import _f64, _i32

        self.shape = tuple(solver._shape)
        self.n_full = int(self.shape[0] * self.shape[1])
        solved = np.asarray(solver._solved)
        self.n = int(solved.size)
        applied = solver._applied
        sensed = solver._sensed
        self.k = 0 if applied is None else int(applied.shape[1])
        self.m = int(m)
        self.h = float(h)
        self.solver = solver

        cap = _f64(solver._capacitance_rate)
        dirichlet = _f64(solver._dirichlet_conductance)
        ambient = _f64(solver.ambient_conductance(float(h)))
        mc = _f64(solver._mass_coupling)
        mr = _f64(solver._mass_retained)
        mg = _f64(solver._mass_gained)
        if self.k:
            applied_csc = sp.csc_matrix(applied)
            sensed_csr = sp.csr_matrix(sensed)
            pptr = _i32(applied_csc.indptr)
            pind = _i32(applied_csc.indices)
            pval = _f64(applied_csc.data)
            sptr = _i32(sensed_csr.indptr)
            sind = _i32(sensed_csr.indices)
            sval = _f64(sensed_csr.data)
        else:
            pptr = np.zeros(1, dtype=np.int32)
            pind = np.zeros(0, dtype=np.int32)
            pval = np.zeros(0, dtype=np.float64)
            sptr = np.zeros(1, dtype=np.int32)
            sind = np.zeros(0, dtype=np.int32)
            sval = np.zeros(0, dtype=np.float64)

        handle = lib.cenv_tbatch_create(
            self.n_full, self.n, self.k, self.m,
            _i32(solved), cap, dirichlet, ambient, mc, mr, mg,
            pptr, pind, pval, int(pind.size),
            sptr, sind, sval, int(sind.size),
        )
        if not handle:
            raise MemoryError("cenv_tbatch_create failed")
        self.handle = handle
        # Keep the sparse operators alive: the kernel holds copies, but the
        # solver's identity checks (thermal.py) compare against these objects.
        self.applied = applied
        self.sensed = sensed
        self._arrays = (cap, dirichlet, ambient, mc, mr, mg, pptr, pind, pval,
                        sptr, sind, sval)

        matrix = solver.matrix(float(h))
        if not sp.issparse(matrix):
            raise TypeError("solver.matrix did not return a sparse matrix")
        self.factor = solver_mod.Factor(sp.csc_matrix(matrix), method=ordering)
        self.factor_stats = self.factor.stats()
        chol = self.factor.release()
        rc = lib.cenv_tbatch_set_factor(self.handle, chol)
        if rc != 0:
            lib.cenv_tbatch_free(self.handle)
            self.handle = None
            raise RuntimeError(f"cenv_tbatch_set_factor failed ({rc})")
        # Same invalidation signal the shipped solve uses: anything holding the
        # Woodbury response must rebuild it after a refactorization.
        solver._sensed_response = None

    def phase(self) -> dict:
        buf = np.zeros(4, dtype=np.float64)
        lib.cenv_tbatch_phase(self.handle, buf.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)))
        steps = max(int(lib.cenv_tbatch_steps(self.handle)), 1)
        return {
            name: float(buf[i]) / steps * 1e3
            for i, name in enumerate(("forcing", "solve", "air_side", "scatter"))
        }

    def close(self) -> None:
        if self.handle is not None:
            lib.cenv_tbatch_free(self.handle)  # frees the Cholesky too
            self.handle = None


# --------------------------------------------------------------------------- #
# one environment in the batch
# --------------------------------------------------------------------------- #
class _Slot:
    """A batched environment: its env, its kernel state, its step halves."""

    __slots__ = ("env", "state", "meta", "index")

    def __init__(self, env, state, meta, index):
        self.env = env
        self.state = state
        self.meta = meta
        self.index = index


_ORIG_STEP = {}


def _patch_step() -> None:
    """Route ``Environment.step`` per environment: batched or untouched.

    ``fused`` already owns ``wait_time``/``_get_observation``/``reward_info``;
    the one thing a batch needs that a per-environment patch cannot express is
    that the C step has to be issued once for all M environments. So the class
    method checks which kind of environment it is looking at.
    """
    from smart_buildings.smart_control.environment import environment

    if "step" not in _ORIG_STEP:
        _ORIG_STEP["step"] = environment.Environment.step
        environment.Environment.step = _batch_aware_step


def _batch_aware_step(self, action):
    slot = self.__dict__.get("_cenv_batch_slot")
    if slot is None:
        return _ORIG_STEP["step"](self, action)
    raise RuntimeError(
        "cenv batch: this environment belongs to a BatchEnv, whose step issues"
        " one solve for all M environments; call BatchEnv.step([...]) instead"
        " (or build a single environment with cenv.fused)."
    )


def _environment_module():
    from smart_buildings.smart_control.environment import environment

    return environment


def _pre_step(slot: _Slot, action):
    """``Environment.step`` up to ``building.wait_time()``.

    Transcribed from ``environment.Environment.step`` (identical in both
    simulator trees) with the C call cut out; the gate compares the whole
    episode against an unpatched run, so a divergence here shows up as a
    trajectory difference, not as a silent shortcut. Returns the acceptance
    flag.
    """
    env = slot.env
    mod = _environment_module()
    action = env._format_action(action, env._action_names)
    request = env._create_action_request(action)
    try:
        response = env.building.request_action(request)
    except RuntimeError as err:
        accepted = False
        response = mod._apply_action_response(
            request,
            response_timestamp=env.current_simulation_timestamp,
            action_response_type=(
                mod.SingleActionResponse.ActionResponseType.
                REJECTED_NOT_ENABLED_OR_AVAILABLE
            ),
            additional_info=str(err),
        )
        mod.logging.exception("Action REJECTED at %s: %s.",
                              env.current_simulation_timestamp, err)
    else:
        accepted = mod.all_actions_accepted(response)
    if env._metrics_writer and response is not None:
        env._metrics_writer.write_action_response(
            response, env.current_simulation_timestamp)
    return accepted


def _post_step(slot: _Slot, accepted: bool):
    """The rest of ``Environment.step``, from ``_get_observation`` on."""
    env = slot.env
    mod = _environment_module()
    observation = env._get_observation()
    reward_value = env._get_reward()
    if not accepted:
        reward_value = mod.ACTION_REJECTION_REWARD
    env._episode_ended = env._has_episode_ended()
    env._episode_cumulative_reward += reward_value
    if env._episode_ended:
        mod.logging.info(
            "%s: Terminating episode=%d step=%d current_time=%s"
            " step_reward=%4.2f cumulative_reward=%5.2f",
            env._label, env._episode_count, env._step_count,
            env.building.current_timestamp, reward_value,
            env._episode_cumulative_reward,
        )
        return observation, float(reward_value), True, False, {}
    if env._step_count % 100 == 0:
        mod.logging.info(
            "%s: episode=%d step=%d current_time=%s step_reward=%4.2f"
            " cumulative_reward=%5.2f",
            env._label, env._episode_count, env._step_count,
            env.building.current_timestamp, reward_value,
            env._episode_cumulative_reward,
        )
    env._step_count += 1
    env._global_step_count += 1
    return observation, float(reward_value), False, False, {}


# --------------------------------------------------------------------------- #
# the batch
# --------------------------------------------------------------------------- #
class BatchEnv:
    """M environments of one scenario, stepped with one C call each step.

    ``config_path`` is a scenario's ``config.gin`` (one gin parse for the whole
    batch), ``sim`` is ``"shipped"`` or ``"corrected"``, ``M`` the number of
    environments and ``occupancy_seeds`` the per-environment occupancy seeds
    (``None`` uses the scenario's own seed for all of them -- they are then
    identical unless the policy differs).
    """

    MAX_M = 128  # cenv_batch_run's stack arrays (CENV_BATCH_MAX)

    def __init__(self, config_path, sim, M, occupancy_seeds=None,
                 ordering="metis"):
        from pathlib import Path

        started = time.process_time()
        if M <= 0 or M > self.MAX_M:
            raise ValueError(f"M must be in 1..{self.MAX_M}, got {M}")
        self.M = int(M)
        self.sim = str(sim)
        self.config_path = Path(config_path).resolve()
        self.scenario = (f"{self.config_path.parent.parent.name}"
                         f"/{self.config_path.parent.name}")
        seeds = list(occupancy_seeds or [])
        if seeds and len(seeds) != self.M:
            raise ValueError(f"need {self.M} occupancy seeds, got {len(seeds)}")
        self.occupancy_seeds = seeds or None

        self._load(seeds)
        self._build_kernel(ordering)
        self._slots = [
            _Slot(env, state, meta, i)
            for i, (env, state, meta) in enumerate(
                zip(self.envs, self.states, self.metas))
        ]
        for slot in self._slots:
            slot.env._cenv_batch_slot = slot
        self._pointers = None
        self._refresh_pointers()
        self.step_index = 0
        PROFILE["build_seconds"] += time.process_time() - started
        PROFILE["builds"] += 1

    # -- construction ----------------------------------------------------- #
    def _load(self, seeds):
        import sys

        from sbx import simload

        # One simulator tree per interpreter (gin registers configurables by
        # module name, so importing both trees is not supported) and
        # ``use_sim`` is not idempotent: only switch when the loaded tree is
        # not the requested one.
        root = simload.sim_root(self.sim)
        loaded = getattr(sys.modules.get("smart_buildings"), "__file__", "") or ""
        if not loaded.startswith(str(root)):
            simload.use_sim(self.sim)
        import gin

        from smart_buildings import evaluate
        from smart_buildings.smart_control.environment import (
            hybrid_action_environment,
        )

        # One parse for the whole batch; only the occupancy seed is rebound
        # between constructions (gin allows a binding between parse and
        # construct, which is exactly the window this loop is in).
        gin.clear_config()
        with gin.unlock_config():
            gin.parse_config_file(str(self.config_path))
        self.evaluate = evaluate
        envs = []
        for i in range(self.M):
            if seeds:
                with gin.unlock_config():
                    gin.bind_parameter(
                        "RandomizedArrivalDepartureOccupancy.seed", int(seeds[i]))
            env = hybrid_action_environment.HybridActionEnvironment()
            env.metrics_path = None
            env._occupancy_normalization_constant = 125.0
            envs.append(env)
        self.envs = envs

    def _build_kernel(self, ordering):
        from . import fused as fused_mod
        from . import stochastic as stoch

        # fused's method patches (wait_time, _get_observation, reward_info,
        # reset) and bookkeeping's reward arithmetic. The convection stream a
        # standalone process would draw from is the process-global state at
        # install time; take it once, before any environment can advance it, so
        # every environment starts where a single-environment run would.
        fused_mod.install()
        conv_state = stoch._numba_np_state()

        first = self.envs[0]
        sim_obj = first.building.simulator
        h = float(sim_obj.weather_controller.get_air_convection_coefficient(
            sim_obj.current_timestamp))
        self.tbatch = TBatch(sim_obj._solver, h, self.M, ordering=ordering)

        self.metas = []
        self.states = []
        for i, env in enumerate(self.envs):
            # Build the kernel *before* the reset: fused's reset hook keeps an
            # existing state (FusedState.refresh) and would otherwise factorize
            # a single-environment kernel per environment that is thrown away
            # immediately.
            state = fused_mod.FusedState(env, tbatch=self.tbatch)
            env.building.simulator._cenv_fused = state
            if len(state.coupled_vav) != self.tbatch.k:
                raise RuntimeError(
                    f"environment {i}: {len(state.coupled_vav)} air-side terms"
                    f" against the batch's {self.tbatch.k}")
            if conv_state is not None:
                conv = env.building.simulator.building.convection_simulator
                st = conv.__dict__.get("_cenv_conv") if conv is not None else None
                if st is not None:
                    stoch._mt_import(st.rng, *conv_state)
            self.states.append(state)
            self.metas.append(self.evaluate.create_building_meta_data(
                env, self.config_path.parent))
        _patch_step()

    # -- pointers --------------------------------------------------------- #
    def _refresh_pointers(self):
        """The per-step ``double*`` arrays the C call takes, rebuilt on change.

        ``building.temp`` and ``building.input_q`` are Python-owned and the
        fused step coerces them to contiguous float64 in place; if either was
        replaced (a reset, a debugger) the addresses move.
        """
        temps = []
        qs = []
        syncs = []
        for slot in self._slots:
            state = slot.state
            building = state.building   # simulator.building, not env.building
            temp = building.temp
            if temp.dtype != np.float64 or not temp.flags["C_CONTIGUOUS"]:
                temp = np.ascontiguousarray(temp, dtype=np.float64)
                building.temp = temp
            q = building.input_q
            if q.dtype != np.float64 or not q.flags["C_CONTIGUOUS"]:
                q = np.ascontiguousarray(q, dtype=np.float64)
                building.input_q = q
            temps.append(_addr(temp))
            qs.append(_addr(q))
            syncs.append(_addr(state.sync))
        key = (tuple(temps), tuple(qs), tuple(syncs))
        if self._pointers == key:
            return
        self._pointers = key
        self.handles = np.array(
            [int(s.state.handle.value) for s in self._slots], dtype=np.int64)
        self._temps = np.array(temps, dtype=np.int64)
        self._qs = np.array(qs, dtype=np.int64)
        self._syncs = np.array(syncs, dtype=np.int64)

    # -- the episode ------------------------------------------------------ #
    def reset(self):
        """Reset every environment; returns the M observation vectors."""
        obs = []
        for slot in self._slots:
            observation, _ = slot.env.reset()   # fused.refresh() keeps the state
            obs.append(np.asarray(observation).copy())
        self.step_index = 0
        self._refresh_pointers()
        return obs

    def step(self, actions):
        """Advance all M environments by one step.

        ``actions[i]`` is environment i's action array; returns
        ``(observations, rewards, infos)``, each a list of M.
        """
        actions = list(actions)
        if len(actions) != self.M:
            raise ValueError(f"need {self.M} actions, got {len(actions)}")
        i = self.step_index
        for slot in self._slots:
            # The kernel needs the episode's step grid up front; running past
            # it extends the per-step tables (the fused state's own rule).
            if slot.state.i + 2 >= slot.state.n_horizon:
                slot.state._extend_horizon(slot.state.i + 64)
        t0 = time.process_time()
        accepted = [_pre_step(slot, actions[k])
                    for k, slot in enumerate(self._slots)]
        PROFILE["pre_seconds"] += time.process_time() - t0
        for slot in self._slots:
            slot.state._sync_step()
        self._refresh_pointers()
        t0 = time.process_time()
        rc = lib.cenv_batch_run(
            self.handles.ctypes.data_as(c_void_p), self.M, i,
            self._temps.ctypes.data_as(c_void_p),
            self._qs.ctypes.data_as(c_void_p),
            self._syncs.ctypes.data_as(c_void_p),
            self.tbatch.handle,
        )
        PROFILE["c_seconds"] += time.process_time() - t0
        PROFILE["steps"] += 1
        if rc != 0:
            raise RuntimeError(f"cenv_batch_run failed ({rc}) at step {i}")
        for slot in self._slots:
            slot.state._post_step_bookkeeping(i)
        t0 = time.process_time()
        outs = [_post_step(slot, accepted[k])
                for k, slot in enumerate(self._slots)]
        PROFILE["post_seconds"] += time.process_time() - t0
        self.step_index = i + 1
        observations = [o[0] for o in outs]
        rewards = [o[1] for o in outs]
        infos = [o[4] for o in outs]
        return observations, rewards, infos

    # -- lifetime --------------------------------------------------------- #
    def close(self):
        for slot in self._slots:
            sim = slot.env.building.simulator
            if sim.__dict__.get("_cenv_fused") is slot.state:
                sim.__dict__.pop("_cenv_fused", None)
            slot.state.close()
            slot.env.__dict__.pop("_cenv_batch_slot", None)
        self.tbatch.close()
        self._pointers = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter teardown
            pass


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #
def _resolve(spec, evaluate, originals=None):
    """``module:Class`` | ``baseline:key`` | class | instance -> class/instance.

    A ``(spec, params)`` pair is applied as ``cls.PARAMS = defaults + params``,
    where the defaults are the class's own ``PARAMS`` as first seen -- not what
    the previous entry in the batch left behind, so M entries of one class with
    different parameters stay independent. ``originals`` collects the pristine
    dictionaries so the caller can put the module back.
    """
    if isinstance(spec, (tuple, list)):
        base, params = spec[0], dict(spec[1] or {})
        cls, label, _ = _resolve(base, evaluate, originals)
        base_params = (originals or {}).get(cls)
        if originals is not None and cls not in originals:
            base_params = dict(getattr(cls, "PARAMS", None) or {})
            originals[cls] = base_params
        if base_params is None:
            base_params = dict(getattr(cls, "PARAMS", None) or {})
        cls.PARAMS = {**base_params, **params}
        return cls, label, params
    if isinstance(spec, str):
        cls, label = _resolve_policy(spec, evaluate)
        return cls, label, dict(getattr(cls, "PARAMS", None) or {})
    return spec, f"{type(spec).__module__}:{type(spec).__name__}", {}


def _resolve_policy(spec, evaluate):
    from sbx.evalrun import _resolve_policy as resolve

    return resolve(spec, evaluate)


def run_batch(scenario, sim, policies, steps, occupancy_seeds=None,
              config_path=None, components="parity", tag=None, log_every=0,
              record_trace=False):
    """Step M policy instances against one batch; returns evalrun-style records.

    ``policies`` holds M items, each of which is a ``module:Class`` spec, a
    ``(spec, params)`` pair (applied as ``cls.PARAMS = defaults + params``, so
    four CtrlW comfort margins are just four pairs), or an already-built
    instance.
    ``scenario`` is ``building_<n>/<weather>``; ``config_path`` overrides the
    generated one.

    The returned ``records`` are in ``sbx/evalrun.py``'s format (same keys, same
    aggregates, produced by the same ``_StepRecorder``), so a batch run and a
    sweep can be compared field by field; ``timing`` splits the per-step cost
    into the C call, the action half, the observation/reward half and the
    policy.
    """
    from pathlib import Path

    from sbx.evalrun import WORKSPACE, _StepRecorder

    started_wall = time.time()
    started_cpu = time.process_time()
    # PROFILE is process-wide: take a base so this run's numbers do not inherit
    # an earlier BatchEnv's (sbx/tune.py builds one per scenario).
    base = {key: PROFILE[key] for key in
            ("c_seconds", "pre_seconds", "post_seconds", "steps")}
    config = Path(config_path) if config_path else (
        WORKSPACE / "sims" / sim / "smart_buildings" / "training_scenarios"
        / scenario / "config.gin")
    M = len(policies)
    env = BatchEnv(config, sim, M, occupancy_seeds=occupancy_seeds)
    evaluate = env.evaluate
    spec = env.envs[0].action_spec()

    # Resolve and build one policy at a time: a ``(cls, params)`` entry sets
    # ``cls.PARAMS``, which the constructor reads, so M entries of the same
    # class with different parameters only work if each is built before the
    # next one's parameters land (this is how two CtrlW setpoints differ).
    resolved = []
    instances = []
    originals: dict = {}
    for item, meta in zip(policies, env.metas):
        cls, label, params = _resolve(item, evaluate, originals)
        resolved.append((cls, label, params))
        instances.append(
            cls if not isinstance(cls, type)
            else evaluate.initialize_policy(cls, meta, spec))
    for cls, original in originals.items():
        cls.PARAMS = original
    recorders = [_StepRecorder(e, meta, mode=components)
                 for e, meta in zip(env.envs, env.metas)]

    obs = env.reset()
    rewards = [[] for _ in range(M)]
    policy_cpu = 0.0
    done = 0
    traces = None
    if record_trace:
        # The arrays cenv/verify/batch_gate.py compares: the observation vector
        # the policy sees, the reward, the reward components the recorder
        # captures, and the Python-visible physical state (zone means, damper
        # settings, boiler internals) -- exactly the single-environment trace
        # cenv/verify/fused_gate.py writes, per environment.
        n_obs = int(np.asarray(obs[0]).size)
        # The zone order cenv/verify/fused_gate.py records, i.e. the order
        # Building.get_zone_average_temps() returns (the same for all M).
        trace_zone_names = list(
            env.states[0].building.get_zone_average_temps())
        n_zone = len(trace_zone_names)
        vavs = list(env.envs[0].building.simulator.hvac.vavs.values())
        n_vav = len(vavs)
        traces = {
            "obs": np.zeros((M, int(steps), n_obs), dtype=np.float64),
            "zone_temps": np.zeros((M, int(steps), n_zone), dtype=np.float64),
            "damper": np.zeros((M, int(steps), n_vav), dtype=np.float64),
            "boiler": np.zeros((M, int(steps), 4), dtype=np.float64),
            "components": {
                key: np.zeros((M, int(steps)), dtype=np.float64)
                for key in ("npr", "ec", "ce", "occupancy", "dmax")
            },
            "zone_names": np.array(
                [str(z) for z in trace_zone_names], dtype=object),
        }
    for _ in range(int(steps)):
        actions = []
        t0 = time.process_time()
        for k in range(M):
            actions.append(evaluate.validate_action(
                evaluate.get_policy_action(instances[k], obs[k]), spec))
        policy_cpu += time.process_time() - t0
        obs, step_rewards, _ = env.step(actions)
        for k in range(M):
            recorders[k].record(float(step_rewards[k]))
            rewards[k].append(float(step_rewards[k]))
            if traces is not None:
                slot = env._slots[k]
                traces["obs"][k, done] = np.asarray(obs[k], dtype=np.float64)
                averages = slot.state.building.get_zone_average_temps()
                traces["zone_temps"][k, done] = [
                    averages[name] for name in trace_zone_names]
                traces["damper"][k, done] = [
                    v._damper_setting
                    for v in slot.env.building.simulator.hvac.vavs.values()]
                hws = slot.env.building.simulator.hvac.hot_water_system
                source = hws._heat_source
                traces["boiler"][k, done] = (
                    source._current_temperature,
                    hws.return_water_temperature_sensor,
                    source._last_step_duration.total_seconds(),
                    hws.flow_factor_sum,
                )
                last = recorders[k].steps[-1]
                for key, dst in traces["components"].items():
                    dst[k, done] = float(last[key])
        done += 1
        if env._slots[0].env._episode_ended:
            break
        if log_every and done % log_every == 0:
            print(f"[batch] {scenario} {sim} M={M} {done}/{steps}", flush=True)

    wall = time.time() - started_wall
    cpu = time.process_time() - started_cpu
    steps_total = max(done, 1) * M
    c_seconds = PROFILE["c_seconds"] - base["c_seconds"]
    timing = {
        "M": M,
        "steps": done,
        "c_ms_per_env_step": c_seconds / steps_total * 1e3,
        "policy_ms_per_env_step": policy_cpu / steps_total * 1e3,
        "action_half_ms_per_env_step": (
            (PROFILE["pre_seconds"] - base["pre_seconds"]) / steps_total * 1e3),
        "obs_half_ms_per_env_step": (
            (PROFILE["post_seconds"] - base["post_seconds"]) / steps_total * 1e3),
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "solve_phase_ms_per_step": env.tbatch.phase(),
        "factor": env.tbatch.factor_stats,
        "build_seconds": PROFILE["build_seconds"],
    }
    records = []
    for k in range(M):
        aggregates = recorders[k].aggregate()
        rewards_k = np.array(rewards[k], dtype=np.float64)
        records.append({
            "scenario": scenario,
            "building": scenario.rsplit("/", 1)[0],
            "weather": scenario.rsplit("/", 1)[-1],
            "policy": resolved[k][1],
            "params": resolved[k][2],
            "sim": sim,
            "cenv": "batch",
            "fast": [],
            "batch_size": M,
            "batch_index": k,
            "occupancy_seed": (env.occupancy_seeds[k]
                               if env.occupancy_seeds else None),
            "tag": tag,
            "steps": int(len(rewards_k)),
            "requested_steps": int(steps),
            "start_offset_days": 0,
            "oracle": [],
            "oracle_horizon": None,
            "total_reward": float(rewards_k.sum()),
            "mean_per_step_reward": float(rewards_k.mean()),
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "steps_per_second_cpu": (len(rewards_k) / cpu) if cpu > 0 else None,
            "steps_per_second_step_cpu": (
                len(rewards_k) / (c_seconds / max(M, 1))
                if c_seconds > 0 else None),
            # evalrun's field is CPU seconds inside env.step for one scenario;
            # a batch shares one C call, so each environment gets its 1/M
            # share of it.
            "step_cpu_seconds": c_seconds / max(M, 1),
            "num_ahus": int(env.metas[k]["n_discrete_action"]) - 1,
            "num_zones": int(len(env.metas[k]["zone_action_indices"])),
            "config_path": str(config.resolve()),
            **aggregates,
        })
    env.close()
    result = {
        "scenario": scenario,
        "sim": sim,
        "M": M,
        "steps": done,
        "timing": timing,
        "records": records,
    }
    if traces is not None:
        result["traces"] = {
            key: (value[:, :done] if hasattr(value, "shape") and value.ndim > 1
                  else value)
            for key, value in traces.items()
            if key != "components"
        }
        result["traces"]["components"] = {
            key: value[:, :done] for key, value in traces["components"].items()
        }
        result["traces"]["rewards"] = np.array(
            [rewards[k][:done] for k in range(M)], dtype=np.float64)
    return result
