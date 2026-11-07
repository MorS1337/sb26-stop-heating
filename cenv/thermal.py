"""Drop-in C replacement for ``FiniteDifferenceSolver.solve``.

``install(root)`` monkeypatches ``FiniteDifferenceSolver.solve`` on whichever
simulator tree is already imported. Nothing under ``sims/`` is edited.

What moves into C
-----------------
Per step, the whole of ``solve`` (finite_difference_solver.py:968) and
``_add_air_side`` (:1027): forcing assembly, the SuperLU triangular solve, the
Woodbury system for the VAV air streams, the interior-mass update and the
scatter back into the full field. Python keeps what happens once per
factorization: assembling ``A(h)`` and factorizing it with scipy exactly as the
simulator does (``splu(..., permc_spec='MMD_AT_PLUS_A')``), then exporting the
factors.

Two things are precomputed per factorization:

* ``S = Q^T A^-1 P`` (k x k) -- exactly what the simulator recomputes; one
  small dense solve per step then follows, as in ``_add_air_side``.
* ``R = A^-1 P`` (n_solved x k) -- what the simulator recomputes *every step*
  as ``factorization.solve(applied @ power)``. Whether that is worth holding
  depends on the building: the step then does one triangular solve plus a dense
  n*k matvec instead of two triangular solves. Both are bandwidth-bound, so the
  handle compares n*k against the factor size (``cenv_thermal_set_factor``) and
  falls back to a second solve of the sparse ``P q`` where the matvec would
  stream more bytes. Measured: building_train_8 (n*k = 2.4M, factors 3.5M)
  takes the dense path, building_train_2 (20.5M against ~9M) the second solve;
  the dense path was 0.26x slower end to end on train_2.

``A`` and its ordering are untouched, so the only numerical difference from the
Python path is the summation order inside the triangular solves (scipy uses
SuperLU's supernodal blocked solve). Measured on recorded rollouts: 5-8e-15
relative on the solved field, 1.9e-12 K absolute.

Usage
-----
    from cenv.thermal import install
    install()                      # fast path
    install(shadow=True)           # run both, gate, return the Python result
"""

from __future__ import annotations

import ctypes
import time

import numpy as np
import scipy.sparse as sp

from .bindings import c_double, c_int32, c_void_p, lib, nd_f64, nd_i32

# Set once cenv/verify/shadow.py reports a passing gate in this checkout; only
# then does `--cenv all` pick this component up. `--cenv thermal` opts in
# regardless.
GATE_PASSED = True

_SOLVER_ATTR = "_cenv_thermal"
_ORIG_ATTR = "_cenv_thermal_orig"

# Seconds spent inside the C step, and the number of steps taken.
PROFILE = {"c_seconds": 0.0, "steps": 0, "build_seconds": 0.0, "builds": 0}
SHADOW = False
# Filled by shadow mode: worst relative/absolute deviation seen so far.
DIFF = {"max_abs": 0.0, "max_rel": 0.0, "steps": 0, "mass_abs": 0.0}

_bindings_done = False


def _bind() -> None:
    global _bindings_done
    if _bindings_done:
        return
    f1 = nd_f64(1)
    i1 = nd_i32(1)
    lib.cenv_thermal_create.restype = c_void_p
    lib.cenv_thermal_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        i1,                       # solved
        f1, f1, f1, f1, f1, f1,   # cap, dir, amb, mc, mr, mg
        i1, i1, f1, ctypes.c_int,  # P's CSC + nnz
        i1, i1, f1, ctypes.c_int,  # Q^T's CSR + nnz
    ]
    lib.cenv_thermal_free.restype = None
    lib.cenv_thermal_free.argtypes = [c_void_p]
    lib.cenv_thermal_set_factor.restype = ctypes.c_int
    lib.cenv_thermal_set_factor.argtypes = [
        c_void_p,
        i1, i1, f1, ctypes.c_int,
        i1, i1, f1, ctypes.c_int,
        i1, i1,
    ]
    lib.cenv_thermal_step.restype = ctypes.c_int
    # The two air-side vectors and the optional power output may be NULL, so
    # they are declared as raw pointers rather than ndpointer (which rejects
    # None on numpy 2).
    lib.cenv_thermal_step.argtypes = [
        c_void_p, f1, f1, c_double, c_void_p, c_void_p, f1, f1, c_void_p,
    ]
    lib.cenv_thermal_response.restype = c_void_p
    lib.cenv_thermal_response.argtypes = [c_void_p]
    lib.cenv_thermal_sensed_response.restype = c_void_p
    lib.cenv_thermal_sensed_response.argtypes = [c_void_p]
    _bindings_done = True


def _i32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32).reshape(-1)


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).reshape(-1)


class State:
    """One C handle per (solver, factorization, air-side coupling)."""

    __slots__ = (
        "handle", "h", "n_full", "n", "k", "applied", "sensed",
        "response", "sresp", "shape",
    )

    def __init__(self, solver, h: float):
        _bind()
        self.shape = solver._shape
        n_full = int(self.shape[0] * self.shape[1])
        solved = np.asarray(solver._solved)
        n = int(solved.size)
        applied = solver._applied
        sensed = solver._sensed
        k = 0 if applied is None else int(applied.shape[1])
        self.k = k

        cap = _f64(solver._capacitance_rate)
        dirichlet = _f64(solver._dirichlet_conductance)
        ambient = _f64(solver.ambient_conductance(h))
        mc = _f64(solver._mass_coupling)
        mr = _f64(solver._mass_retained)
        mg = _f64(solver._mass_gained)
        sol = _i32(solved)

        if k:
            # Both air-side operators stay sparse: the step forms P q from P's
            # nonzeros and a k x n "Q^T y" from Q^T's.
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

        handle = lib.cenv_thermal_create(
            n_full, n, k, sol, cap, dirichlet, ambient, mc, mr, mg,
            pptr, pind, pval, int(pind.size),
            sptr, sind, sval, int(sind.size),
        )
        if not handle:
            raise MemoryError("cenv_thermal_create failed")
        self.handle = handle
        self.n_full = n_full
        self.n = n
        try:
            self._factorize(solver, h)
        except BaseException:
            lib.cenv_thermal_free(handle)
            self.handle = None
            raise
        # Keep the sparse matrices alive so identity checks stay meaningful.
        self.applied = applied
        self.sensed = sensed
        self.h = h

    def _factorize(self, solver, h: float) -> None:
        import scipy.sparse.linalg as spla

        matrix = solver.matrix(h)
        if not sp.issparse(matrix):
            raise TypeError("solver.matrix did not return a sparse matrix")
        lu = spla.splu(sp.csc_matrix(matrix), permc_spec="MMD_AT_PLUS_A")
        L = lu.L.tocsc()
        U = lu.U.tocsc()
        ipr = np.argsort(np.asarray(lu.perm_r)).astype(np.int32)
        pc = np.asarray(lu.perm_c).astype(np.int32)
        rc = lib.cenv_thermal_set_factor(
            self.handle,
            _i32(L.indptr), _i32(L.indices), _f64(L.data), int(L.nnz),
            _i32(U.indptr), _i32(U.indices), _f64(U.data), int(U.nnz),
            _i32(ipr), _i32(pc),
        )
        if rc != 0:
            raise RuntimeError(f"cenv_thermal_set_factor failed: {rc}")
        k = self.k
        if k:
            n = self.n
            resp = lib.cenv_thermal_response(self.handle)
            sresp = lib.cenv_thermal_sensed_response(self.handle)
            self.response = (
                None if not resp else np.ctypeslib.as_array(
                    ctypes.cast(ctypes.c_void_p(resp),
                                ctypes.POINTER(c_double)),
                    shape=(k * n,),
                )
            )
            self.sresp = np.ctypeslib.as_array(
                ctypes.cast(ctypes.c_void_p(sresp), ctypes.POINTER(c_double)),
                shape=(k * k,),
            )
        else:
            self.response = None
            self.sresp = None

    def close(self) -> None:
        if self.handle:
            lib.cenv_thermal_free(self.handle)
            self.handle = None


def _state(solver, h: float) -> State:
    state = getattr(solver, _SOLVER_ATTR, None)
    if (
        state is not None
        and state.h == h
        and state.applied is solver._applied
        and state.sensed is solver._sensed
        and state.n_full == solver._shape[0] * solver._shape[1]
    ):
        return state
    if state is not None:
        state.close()
    t0 = time.perf_counter()
    state = State(solver, h)
    # Same invalidation signal the shipped solve uses: anything holding the
    # Woodbury response must rebuild it after a refactorization.
    solver._sensed_response = None
    PROFILE["build_seconds"] += time.perf_counter() - t0
    PROFILE["builds"] += 1
    setattr(solver, _SOLVER_ATTR, state)
    return state


def _run(solver, temp, input_q, ambient_temperature, convection_coefficient,
         air_side_conductance, supply_temperature, want_power=False):
    state = _state(solver, float(convection_coefficient))
    n_full = state.n_full
    temp_flat = np.asarray(temp, dtype=np.float64).reshape(-1)
    q_flat = np.asarray(input_q, dtype=np.float64).reshape(-1)
    if temp_flat.size != n_full or q_flat.size != n_full:
        raise ValueError(
            f"temp/input_q must hold {n_full} CVs, got "
            f"{temp_flat.size}/{q_flat.size}"
        )
    mass = solver._mass_temp
    if mass.dtype != np.float64 or not mass.flags["C_CONTIGUOUS"]:
        mass = np.ascontiguousarray(mass, dtype=np.float64)
        solver._mass_temp = mass
    out = np.empty(n_full, dtype=np.float64)
    k = state.k
    if air_side_conductance is not None:
        g = _f64(air_side_conductance)
        ts = _f64(supply_temperature)
        g = g.ctypes.data_as(c_void_p)
        ts = ts.ctypes.data_as(c_void_p)
    else:
        g = None
        ts = None
    power = np.zeros(k, dtype=np.float64) if (want_power and k) else None
    power_ptr = power.ctypes.data_as(c_void_p) if power is not None else None
    t0 = time.perf_counter()
    rc = lib.cenv_thermal_step(
        state.handle,
        _f64(temp_flat),
        _f64(q_flat),
        float(ambient_temperature),
        g,
        ts,
        out,
        _f64(mass),
        power_ptr,
    )
    PROFILE["c_seconds"] += time.perf_counter() - t0
    PROFILE["steps"] += 1
    if rc != 0:
        raise RuntimeError(f"cenv_thermal_step failed: {rc}")
    return out.reshape(state.shape), power


def _record_diff(ref: np.ndarray, got: np.ndarray, scale: np.ndarray) -> None:
    diff = np.abs(ref - got)
    denom = np.maximum(np.abs(ref), scale)
    DIFF["max_abs"] = max(DIFF["max_abs"], float(diff.max()) if diff.size else 0.0)
    rel = float((diff / denom).max()) if diff.size else 0.0
    DIFF["max_rel"] = max(DIFF["max_rel"], rel)
    DIFF["steps"] += 1


def _solve(self, temp, input_q, ambient_temperature, convection_coefficient,
           air_side_conductance=None, supply_temperature=None):
    if SHADOW:
        original = getattr(type(self), _ORIG_ATTR)
        mass_before = np.array(self._mass_temp, dtype=np.float64, copy=True)
        ref = original(
            self, temp, input_q, ambient_temperature, convection_coefficient,
            air_side_conductance, supply_temperature,
        )
        ref_mass = np.array(self._mass_temp, dtype=np.float64, copy=True)
        self._mass_temp[...] = mass_before
        got, _ = _run(
            self, temp, input_q, ambient_temperature, convection_coefficient,
            air_side_conductance, supply_temperature,
        )
        mass_got = np.array(self._mass_temp, dtype=np.float64, copy=True)
        self._mass_temp[...] = ref_mass
        # Scale for the relative measure: the field level, so a CV that sits at
        # ambient cannot turn a 1e-13 K difference into a big relative one.
        scale = np.full(ref.size, float(np.mean(np.abs(ref))), dtype=np.float64)
        _record_diff(ref.reshape(-1), got.reshape(-1), scale)
        DIFF["mass_abs"] = max(
            DIFF["mass_abs"],
            float(np.abs(mass_got - ref_mass).max()) if ref_mass.size else 0.0,
        )
        return ref
    out, _ = _run(
        self, temp, input_q, ambient_temperature, convection_coefficient,
        air_side_conductance, supply_temperature,
    )
    return out


# FastExact's --fast patch checks this marker (and its own) to decide whether
# another component already owns FiniteDifferenceSolver.solve; whichever installs
# last wins and nothing double-computes.
_solve.__cenv__ = "thermal"


def install(sim_root=None, shadow: bool = False) -> dict:
    """Monkeypatches ``FiniteDifferenceSolver.solve``.

    Args:
      sim_root: unused; the tree is selected by ``sbx.simload.use_sim`` before
        ``smart_buildings`` is imported.
      shadow: run the Python path as well, keep its answer, and accumulate the
        deviation in ``cenv.thermal.DIFF``.
    """
    global SHADOW
    from smart_buildings.smart_control.simulator.finite_difference_solver import (  # noqa: PLC0415
        FiniteDifferenceSolver,
    )

    if not hasattr(FiniteDifferenceSolver, _ORIG_ATTR):
        setattr(FiniteDifferenceSolver, _ORIG_ATTR, FiniteDifferenceSolver.solve)
    FiniteDifferenceSolver.solve = _solve
    SHADOW = bool(shadow)
    return {"patched": ["FiniteDifferenceSolver.solve"], "shadow": SHADOW}


def shadow_pairs(shadow, sim_root=None) -> None:
    """Registers the Python-vs-C pair driven by ``python -m cenv.shadow``.

    The comparison is the whole temperature field (``kind="array"``) plus the
    interior-mass state, which is the only thing carried between steps. The
    Python original runs first, so the episode it produces is the official one;
    the mass temperatures it advanced are put back before the C step runs and
    restored after, so the two sides see identical inputs.
    """
    from smart_buildings.smart_control.simulator.finite_difference_solver import (  # noqa: PLC0415
        FiniteDifferenceSolver,
    )

    inner = getattr(FiniteDifferenceSolver, _ORIG_ATTR, None)
    if inner is None:
        inner = FiniteDifferenceSolver.solve

    def snapshotting(self, *args, **kwargs):
        mass = self._mass_temp
        mass_before = np.array(mass, dtype=np.float64, copy=True)
        out = inner(self, *args, **kwargs)
        self._cenv_mass_pair = (mass_before, np.array(mass, dtype=np.float64,
                                                      copy=True))
        return out

    if not hasattr(FiniteDifferenceSolver, "_cenv_snapshot_installed"):
        FiniteDifferenceSolver.solve = snapshotting
        FiniteDifferenceSolver._cenv_snapshot_installed = True

    def c_step(self, temp, input_q, ambient_temperature,
               convection_coefficient, air_side_conductance=None,
               supply_temperature=None):
        mass_before, mass_after = self._cenv_mass_pair
        mass = self._mass_temp
        mass[...] = mass_before
        out, _ = _run(
            self, temp, input_q, ambient_temperature, convection_coefficient,
            air_side_conductance, supply_temperature,
        )
        # Only the CVs that actually carry a node: the rest are a constant 0 on
        # both sides and would make the relative metric divide by zero.
        mask = self._mass_coupling > 0.0
        shadow.compare(
            "FiniteDifferenceSolver.solve.mass_temp", mass_after[mask],
            np.array(mass, dtype=np.float64, copy=True)[mask], "array",
        )
        mass[...] = mass_after
        return out

    shadow.pair(
        FiniteDifferenceSolver, "solve", c_step, kind="array",
        label="FiniteDifferenceSolver.solve",
    )


def uninstall() -> None:
    from smart_buildings.smart_control.simulator.finite_difference_solver import (  # noqa: PLC0415
        FiniteDifferenceSolver,
    )

    if hasattr(FiniteDifferenceSolver, _ORIG_ATTR):
        FiniteDifferenceSolver.solve = getattr(FiniteDifferenceSolver, _ORIG_ATTR)


def reset_profile() -> None:
    for key in PROFILE:
        PROFILE[key] = 0.0 if key.endswith("seconds") else 0
    for key in DIFF:
        DIFF[key] = 0.0 if isinstance(DIFF[key], float) else 0
