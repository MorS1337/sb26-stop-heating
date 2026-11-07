"""Opt-in exact sparse-Cholesky path for the simulator's per-step system.

The timestep matrix ``A`` (``finite_difference_solver.matrix``) is symmetric and
positive definite, so the shipped ``splu(A, permc_spec='MMD_AT_PLUS_A')`` is
doing an unsymmetric LU of a symmetric matrix: twice the factor storage, twice
the entries to stream per solve, and a fill-reducing ordering derived from the
graph of ``A^T A`` rather than of ``A``. ``cenv/src/solver_chol.c`` replaces it
with an LL^T factorization of a symmetrically permuted A plus a supernodal
(blocked) substitution that is also capable of serving many right-hand sides at
once.

Nothing here changes a default. Importing this module has no effect; call
``install()`` to route ``scipy.sparse.linalg.splu`` through the Cholesky for the
factorizations the simulator makes, and ``uninstall()`` to put it back.

Orderings available (see :func:`ordering`): ``metis`` (nested dissection, best
fill), ``amd`` (approximate minimum degree), ``mmd`` (SuperLU's own, i.e. the
shipped ordering), ``rcm`` and ``natural``. ``metis`` and ``amd`` need the
SuiteSparse/METIS shared libraries; they are loaded with ``ctypes`` at first use
and are never a build dependency of ``cenv``.

Usage
-----
    from cenv.solver import Factor, solve_many
    factor = Factor(A)                 # metis by default, mmd fallback
    x = factor.solve(b)                # n-vector
    X = factor.solve_many(B)           # B is (n, m) C-contiguous, X likewise

    import cenv.solver
    cenv.solver.install()              # whole-simulator opt-in (splu shim)
    ...                                # run the environment
    cenv.solver.uninstall()

The ``splu`` shim returns an object that answers ``.solve``, ``.L``, ``.U``,
``.perm_r`` and ``.perm_c`` exactly as ``scipy.sparse.linalg.SuperLU`` does, so
both the pure-Python ``FiniteDifferenceSolver.solve`` and
``cenv.thermal`` (which exports ``lu.L``/``lu.U`` into its C kernel) keep
working unchanged.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as _spla

from .bindings import c_double, c_int32, c_void_p, lib, nd_f64, nd_i32

# Seconds and counts, filled by the shim.
PROFILE = {"factor_seconds": 0.0, "factors": 0, "solve_seconds": 0.0,
           "solves": 0, "factor_nnz": 0}

_MODE_SUPER = 0
_MODE_CSC = 1

_splu_orig = _spla.splu

_bindings_done = False


def _bind() -> None:
    global _bindings_done
    if _bindings_done:
        return
    lib.cenv_chol_create.restype = c_void_p
    lib.cenv_chol_create.argtypes = [c_int32, nd_i32(1), nd_i32(1),
                                     nd_f64(1), nd_i32(1)]
    lib.cenv_chol_free.restype = None
    lib.cenv_chol_free.argtypes = [c_void_p]
    lib.cenv_chol_solve.restype = c_int32
    lib.cenv_chol_solve.argtypes = [c_void_p, nd_f64(1), nd_f64(1), c_int32]
    lib.cenv_chol_solve_many.restype = c_int32
    lib.cenv_chol_solve_many.argtypes = [c_void_p, nd_f64(), nd_f64(),
                                         c_int32, c_int32]
    for name in ("n", "nnz", "nsuper", "maxwidth"):
        fn = getattr(lib, f"cenv_chol_{name}")
        fn.restype = c_int32
        fn.argtypes = [c_void_p]
    lib.cenv_chol_nnz_super.restype = ctypes.c_int64
    lib.cenv_chol_nnz_super.argtypes = [c_void_p]
    lib.cenv_chol_flops.restype = c_double
    lib.cenv_chol_flops.argtypes = [c_void_p]
    for name in ("perm", "colptr", "rowind"):
        fn = getattr(lib, f"cenv_chol_{name}")
        fn.restype = c_void_p
        fn.argtypes = [c_void_p]
    lib.cenv_chol_values.restype = c_void_p
    lib.cenv_chol_values.argtypes = [c_void_p]
    _bindings_done = True


# --------------------------------------------------------------------------- #
# orderings
# --------------------------------------------------------------------------- #
def _lib_path(names: tuple[str, ...], env: str | None = None) -> str | None:
    """Path of a shared library given candidate file names, or None."""
    if env and os.environ.get(env):
        return os.environ[env]
    for name in names:
        found = ctypes.util.find_library(name)
        if found:
            return found
    # The brewed SuiteSparse/METIS live under <prefix>/lib and (keg-only)
    # under <prefix>/opt/<formula>/lib, whose formula names differ from the
    # library names ("amd" is in suite-sparse), so walk them all.
    dirs: list[str] = []
    for base in ("/opt/homebrew", "/usr/local", "/usr"):
        dirs.append(os.path.join(base, "lib"))
        opt = os.path.join(base, "opt")
        if os.path.isdir(opt):
            dirs += [os.path.join(opt, d, "lib") for d in sorted(os.listdir(opt))]
    for name in names:
        for d in dirs:
            for ext in (".dylib", ".so"):
                cand = os.path.join(d, f"lib{name}{ext}")
                if os.path.exists(cand):
                    return cand
    return None


_METIS = None
_AMD = None

# SuiteSparse AMD's ABI: amd_order takes `double Control[AMD_CONTROL]` and
# `double Info[AMD_INFO]`, fixed-size arrays the library *writes into* -- not
# pointers with a length the caller chooses. Both constants are pinned by
# amd.h (5 and 20 in every released SuiteSparse; AMD_DMAX, the last documented
# Info slot, is 13). Sizing Info to 8 here wrote 8 doubles of the *host* heap
# past the numpy buffer on glibc (silent corruption that aborted at process
# teardown); see scratch/CenvHeap/amd_overflow_proof.py.
_AMD_CONTROL = 5
_AMD_INFO = 20


def _metis_lib():
    global _METIS
    if _METIS is None:
        path = _lib_path(("metis",), "CENV_METIS")
        if path is None:
            raise RuntimeError("libmetis not found (brew install metis)")
        _METIS = ctypes.CDLL(path)
        _METIS.METIS_NodeND.restype = ctypes.c_int
        _METIS.METIS_NodeND.argtypes = [ctypes.POINTER(c_int32),
                                        ctypes.POINTER(c_int32),
                                        ctypes.POINTER(c_int32), c_void_p,
                                        c_void_p,
                                        ctypes.POINTER(c_int32),
                                        ctypes.POINTER(c_int32)]
    return _METIS


def _amd_lib():
    global _AMD
    if _AMD is None:
        path = _lib_path(("amd",), "CENV_AMD")
        if path is None:
            raise RuntimeError("libamd not found (brew install suite-sparse)")
        _AMD = ctypes.CDLL(path)
        _AMD.amd_order.restype = ctypes.c_int
        _AMD.amd_order.argtypes = [c_int32, ctypes.POINTER(c_int32),
                                   ctypes.POINTER(c_int32),
                                   ctypes.POINTER(c_int32),
                                   ctypes.POINTER(c_double),
                                   ctypes.POINTER(c_double)]
    return _AMD


def _pattern_graph(A):
    """Adjacency of A's off-diagonal pattern, symmetric, no self-loops."""
    G = sp.csc_matrix(A).copy()
    G = G + G.T
    G.setdiag(0.0)
    G.eliminate_zeros()
    G.data[:] = 1.0
    return G.tocsr()


def metis_perm(A) -> np.ndarray:
    """METIS nested dissection of A's graph, applied per connected component."""
    from scipy.sparse.csgraph import connected_components

    G = _pattern_graph(A)
    n = G.shape[0]
    ncomp, labels = connected_components(G, directed=False)
    lib_m = _metis_lib()
    perm = np.empty(n, dtype=np.int32)
    pos = 0
    for c in range(ncomp):
        idx = np.flatnonzero(labels == c).astype(np.int32)
        nv = ctypes.c_int32(idx.size)
        nvi = int(idx.size)
        sub = G[idx][:, idx].tocsr()
        sub.sort_indices()
        xadj = np.ascontiguousarray(sub.indptr, dtype=np.int32)
        adjncy = np.ascontiguousarray(sub.indices, dtype=np.int32)
        part = np.empty(nvi, dtype=np.int32)
        iperm = np.empty(nvi, dtype=np.int32)
        rc = lib_m.METIS_NodeND(ctypes.byref(nv),
                                xadj.ctypes.data_as(ctypes.POINTER(c_int32)),
                                adjncy.ctypes.data_as(ctypes.POINTER(c_int32)),
                                None, None,
                                part.ctypes.data_as(ctypes.POINTER(c_int32)),
                                iperm.ctypes.data_as(ctypes.POINTER(c_int32)))
        if rc != 1:
            raise RuntimeError(f"METIS_NodeND failed ({rc}) on component {c}")
        perm[pos:pos + nvi] = idx[part]
        pos += nvi
    return perm


def amd_perm(A) -> np.ndarray:
    """AMD on the strictly-upper triangle of A, per connected component."""
    from scipy.sparse.csgraph import connected_components

    G = _pattern_graph(A)
    n = G.shape[0]
    ncomp, labels = connected_components(G, directed=False)
    lib_a = _amd_lib()
    Upper = sp.triu(sp.csc_matrix(A), k=1).tocsc()
    Upper.eliminate_zeros()
    perm = np.empty(n, dtype=np.int32)
    ctrl = np.zeros(_AMD_CONTROL, dtype=np.float64)
    info = np.zeros(_AMD_INFO, dtype=np.float64)
    pos = 0
    for c in range(ncomp):
        idx = np.flatnonzero(labels == c)
        nv = np.int32(idx.size)
        sub = Upper[idx][:, idx].tocsc()
        sub.sort_indices()
        Ap = np.ascontiguousarray(sub.indptr, dtype=np.int32)
        Ai = np.ascontiguousarray(sub.indices, dtype=np.int32)
        P = np.empty(int(nv), dtype=np.int32)
        rc = lib_a.amd_order(nv, Ap.ctypes.data_as(ctypes.POINTER(c_int32)),
                             Ai.ctypes.data_as(ctypes.POINTER(c_int32)),
                             P.ctypes.data_as(ctypes.POINTER(c_int32)),
                             ctrl.ctypes.data_as(ctypes.POINTER(c_double)),
                             info.ctypes.data_as(ctypes.POINTER(c_double)))
        if rc != 0:
            raise RuntimeError(f"amd_order failed ({rc}) on component {c}")
        perm[pos:pos + int(nv)] = idx[P]
        pos += int(nv)
    return perm


def mmd_perm(A) -> np.ndarray:
    """SuperLU's ``perm_c`` (MMD_AT_PLUS_A) -- *not* a symmetric ordering.

    Kept for the ordering study only: SuperLU pivots, so its column permutation
    is not a permutation of the symmetric factor. Measured on train_1 it gives
    nnz(L) = 132,182,829 against 3,275,814 for METIS and takes 194 s to
    factorize. Never use it as ``Factor``'s permutation.
    """
    lu = _splu_orig(sp.csc_matrix(A), permc_spec="MMD_AT_PLUS_A")
    return np.ascontiguousarray(lu.perm_c, dtype=np.int32)


def natural_perm(A) -> np.ndarray:
    return np.arange(sp.csc_matrix(A).shape[0], dtype=np.int32)


def rcm_perm(A) -> np.ndarray:
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    return np.ascontiguousarray(
        reverse_cuthill_mckee(_pattern_graph(A), symmetric_mode=True),
        dtype=np.int32)


_ORDERINGS = {"metis": metis_perm, "amd": amd_perm, "mmd": mmd_perm,
              "natural": natural_perm, "rcm": rcm_perm}


def ordering(A, method: str = "metis") -> np.ndarray:
    """perm[new] = old for the requested fill-reducing ordering."""
    try:
        return _ORDERINGS[method](A)
    except KeyError:
        raise ValueError(f"unknown ordering {method!r}; "
                         f"choose from {sorted(_ORDERINGS)}") from None


# --------------------------------------------------------------------------- #
# factor
# --------------------------------------------------------------------------- #
class Factor:
    """LL^T of a symmetrically permuted A, with supernodal solves."""

    __slots__ = ("n", "perm", "_handle", "A", "ordering", "seconds", "_arrays")

    def __init__(self, A, perm=None, method: str = "metis"):
        _bind()
        A = sp.csc_matrix(A)
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got {A.shape}")
        A.sum_duplicates()
        n = A.shape[0]
        if perm is None:
            try:
                perm = ordering(A, method)
            except Exception:
                # METIS is not on every machine (it is absent on the PC);
                # AMD is a valid symmetric ordering and only a few percent
                # worse here. Never fall back to ``mmd_perm``: SuperLU's
                # ``perm_c`` is a column ordering for its *pivoted*
                # factorization, and used as a symmetric permutation it gave
                # 40x the fill and a 194 s factorization on train_1.
                if method != "metis":
                    raise
                method = "amd"
                perm = ordering(A, "amd")
        self.ordering = method
        self.A = A
        self.n = n
        self.perm = np.ascontiguousarray(perm, dtype=np.int32)
        Ap = np.ascontiguousarray(A.indptr, dtype=np.int32)
        Ai = np.ascontiguousarray(A.indices, dtype=np.int32)
        Ax = np.ascontiguousarray(A.data, dtype=np.float64)
        t0 = time.perf_counter()
        handle = lib.cenv_chol_create(n, Ap, Ai, Ax, self.perm)
        self.seconds = time.perf_counter() - t0
        if not handle:
            raise ValueError("A is not numerically positive definite")
        self._handle = handle
        self._arrays = (Ap, Ai, Ax)

    # -- statistics ------------------------------------------------------- #
    def stats(self) -> dict:
        h = self._handle
        return {
            "n": self.n,
            "nnz": int(lib.cenv_chol_nnz(h)),
            "nnz_super": int(lib.cenv_chol_nnz_super(h)),
            "nsuper": int(lib.cenv_chol_nsuper(h)),
            "maxwidth": int(lib.cenv_chol_maxwidth(h)),
            "flops": float(lib.cenv_chol_flops(h)),
            "bytes": 12 * int(lib.cenv_chol_nnz(h)),
            "bytes_super": 8 * int(lib.cenv_chol_nnz_super(h)),
            "factor_seconds": self.seconds,
            "ordering": self.ordering,
        }

    def l_matrix(self) -> sp.csc_matrix:
        """The CSC lower factor, for inspection and for the LU-style shim."""
        h = self._handle
        n = self.n
        nnz = int(lib.cenv_chol_nnz(h))
        ptr = lambda p, ctype: ctypes.cast(  # noqa: E731
            ctypes.c_void_p(p), ctypes.POINTER(ctype))
        indptr = np.ctypeslib.as_array(ptr(lib.cenv_chol_colptr(h), c_int32),
                                       shape=(n + 1,)).copy()
        indices = np.ctypeslib.as_array(ptr(lib.cenv_chol_rowind(h), c_int32),
                                        shape=(nnz,)).copy()
        data = np.ctypeslib.as_array(ptr(lib.cenv_chol_values(h), c_double),
                                     shape=(nnz,)).copy()
        return sp.csc_matrix((data, indices, indptr), shape=(n, n))

    # -- solves ----------------------------------------------------------- #
    def solve(self, b, mode: int = _MODE_CSC) -> np.ndarray:
        b = np.ascontiguousarray(b, dtype=np.float64).reshape(-1)
        if b.size != self.n:
            raise ValueError(f"b must hold {self.n} entries, got {b.size}")
        x = np.empty(self.n, dtype=np.float64)
        t0 = time.perf_counter()
        rc = lib.cenv_chol_solve(self._handle, b, x, mode)
        PROFILE["solve_seconds"] += time.perf_counter() - t0
        PROFILE["solves"] += 1
        if rc != 0:
            raise RuntimeError(f"cenv_chol_solve failed: {rc}")
        return x

    def release(self):
        """Hand the factorization to another owner, which will free it.

        Used by the batched environment (``cenv/batch.py``): the factor has to
        live inside the C handle that steps M environments, since only that one
        knows when the last solve is done. The Factor is unusable afterwards
        (``close`` and ``__del__`` then do nothing).
        """
        handle = self._handle
        self._handle = None
        return handle

    def solve_many(self, B, mode: int | None = None) -> np.ndarray:
        """Solve A X = B for a block B of shape (n, m), C-contiguous.

        ``mode=None`` picks the kernel by width: the CSC sweeps win up to about
        two dozen right-hand sides (fundamental supernodes on these lattices are
        barely wider than one column, so the panels mostly pay their bookkeeping
        without amortizing anything) and the panels are ~10% ahead from M = 32 up
        -- measured per M in ``scratch/SolverCPU/bench.py``.
        """
        B = np.ascontiguousarray(B, dtype=np.float64)
        if B.ndim != 2 or B.shape[0] != self.n:
            raise ValueError(f"B must be ({self.n}, m), got {B.shape}")
        X = np.empty_like(B)
        if mode is None:
            mode = _MODE_SUPER if B.shape[1] >= 24 else _MODE_CSC
        rc = lib.cenv_chol_solve_many(self._handle, B, X, B.shape[1], mode)
        if rc != 0:
            raise RuntimeError(f"cenv_chol_solve_many failed: {rc}")
        return X

    def close(self) -> None:
        if self._handle:
            lib.cenv_chol_free(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def solve_many(factor: Factor, B, mode: int | None = None) -> np.ndarray:
    """A X = B for a block of right-hand sides: the batched entry point.

    ``B`` is (n, M) C-contiguous -- the M right-hand sides of one unknown
    adjacent, so the triangular sweeps run one sequential pass per sweep no
    matter how many environments share the factorization.
    """
    return factor.solve_many(B, mode=mode)


def factorize(A, method: str = "metis", perm=None) -> Factor:
    return Factor(A, perm=perm, method=method)


# --------------------------------------------------------------------------- #
# scipy.sparse.linalg.splu shim
# --------------------------------------------------------------------------- #
class CholeskyAsLU:
    """What ``scipy.sparse.linalg.splu(A)`` would have returned, from LL^T.

    ``perm_r``/``perm_c`` and the two triangular factors are chosen so that the
    consumer's ``y = b[argsort(perm_r)]; L\\\\y; U\\\\y; x[i] = y[perm_c[i]]``
    reproduces ``A^-1 b``: with ``A_p = A[p][:, p] = L L^T``, the unit lower
    factor is ``L D^-1`` and the upper one ``D L^T`` (``D = diag(L)``), whose
    product is ``A_p``.
    """

    def __init__(self, factor: Factor):
        self._factor = factor
        self.shape = (factor.n, factor.n)
        self.perm_r = np.argsort(factor.perm).astype(np.int32)
        self.perm_c = np.argsort(factor.perm).astype(np.int32)
        self._L = None
        self._U = None
        self._d = None

    def _build(self) -> None:
        if self._L is not None:
            return
        L = self._factor.l_matrix()
        d = L.diagonal().copy()
        self._d = d
        inv = 1.0 / d
        # column scaling by 1/d makes the diagonal unit; row scaling of L^T by
        # d makes L_unit @ U == L @ L^T == A_p
        self._L = sp.csc_matrix(L @ sp.diags(inv))
        self._U = sp.csc_matrix(sp.diags(d) @ L.T.tocsc())

    @property
    def L(self):
        self._build()
        return self._L

    @property
    def U(self):
        self._build()
        return self._U

    def solve(self, b, trans: str = "N"):
        """``A x = b`` for one right-hand side or a block of them.

        The simulator asks for a single column for the timestep forcing and for
        a (n, 16) block when it builds the VAV response matrix, so a 2-D
        argument must go through the batched kernel rather than being flattened.
        """
        if trans not in ("N", None):
            raise NotImplementedError(f"only the forward solve is available: {trans}")
        b = np.asarray(b, dtype=np.float64)
        if b.ndim == 2:
            return self._factor.solve_many(np.ascontiguousarray(b))
        return self._factor.solve(b)


_INSTALLED = False
_ORDER = "metis"

# Set once the closed-loop gate in scratch/SolverCPU/exact.py has passed on the
# shipped tree; ``--cenv all`` only picks up components that say so.
# The gate passes -- 2880 steps x 2 processes on train_8/fairbanks_8, per-step
# shadow 3.7e-14, total reward identical to 15 digits, zone temperatures within
# 7e-11 K, zero action flips (docs/solver_cpu.md) -- but this component is
# deliberately *not* in `--cenv all`: on a single environment it is 10% slower per
# step than the shipped SuperLU, because a Cholesky does not beat a pivoted LU at
# single-right-hand-side latency. Its win is multi-environment throughput
# (solve_many: 4-6x per env-step at M=16..64), which needs explicit use, so opt in
# by name: `--cenv solver` or cenv.solver.install().
GATE_PASSED = False


def _splu_shim(A, *args, **kwargs):
    if not _INSTALLED:
        return _splu_orig(A, *args, **kwargs)
    A = sp.csc_matrix(A)
    if A.shape[0] != A.shape[1] or A.shape[0] < 1024:
        return _splu_orig(A, *args, **kwargs)
    t0 = time.perf_counter()
    factor = Factor(A, method=_ORDER)
    PROFILE["factor_seconds"] += time.perf_counter() - t0
    PROFILE["factors"] += 1
    PROFILE["factor_nnz"] = factor.stats()["nnz"]
    return CholeskyAsLU(factor)


def install(sim_root=None, method: str = "metis", **_) -> dict:
    """Route the simulator's ``splu`` calls through the Cholesky. Opt-in.

    Matches the ``cenv`` component protocol (``cenv/__init__.install``) so
    ``sbx/evalrun.py --cenv solver`` and ``--cenv solver,bookkeeping`` work and
    the A/B is a switch rather than a code edit. ``sim_root`` is unused: the
    simulator tree is already selected by ``sbx.simload.use_sim``.
    """
    global _INSTALLED, _ORDER
    _ORDER = method
    _INSTALLED = True
    _spla.splu = _splu_shim
    return {"patched": ["scipy.sparse.linalg.splu"],
            "restore": [(_spla, "splu", _splu_orig)]}


def uninstall() -> dict:
    """Put scipy's SuperLU back; returns ``cenv``-style restore information."""
    global _INSTALLED
    _INSTALLED = False
    _spla.splu = _splu_orig
    return {"patched": [], "restore": [(_spla, "splu", _splu_orig)]}
