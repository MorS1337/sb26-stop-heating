"""AMD ABI gate: the Control/Info buffers `amd_perm` hands to libamd.

    starter-kit/.venv/bin/python -m cenv.verify.amd_abi

SuiteSparse AMD's ``amd_order`` takes ``double Control[AMD_CONTROL]`` and
``double Info[AMD_INFO]`` -- fixed-size arrays the library *writes into*, not
pointers with a caller-chosen length. ``AMD_CONTROL`` is 5 and ``AMD_INFO`` is
20 in every released SuiteSparse; ``amd_order`` writes Info[0..AMD_DMAX] with
AMD_DMAX = 13, so an Info buffer of 8 doubles (the bug this gate exists for)
puts several doubles of the host heap past the numpy allocation. On glibc that
is silent corruption: nothing aborts until an unrelated free at process
teardown ("corrupted size vs. prev_size" / a bare SIGSEGV at interpreter exit).
It also cannot be caught by AddressSanitizer, because the store happens inside
libamd.so, which is not instrumented -- hence the canary below.

Nothing here depends on a simulator, so the gate is a second.

The canary is the whole test: ``Info`` points at the start of a buffer of
``size + slack`` doubles filled with NaN. amd_order writes from index 0 upward,
so any write past ``size`` lands in the NaN tail and is visible afterwards.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

SLACK = 8  # doubles of NaN canary behind every buffer


def _overflow(lib, n, Ap, Ai, P, ctrl_size, info_size):
    """amd_order on a fixed pattern; returns the number of canary doubles hit."""
    info = np.full(info_size + SLACK, np.nan, dtype=np.float64)
    ctrl = np.zeros(ctrl_size, dtype=np.float64)
    p_int = ctypes.POINTER(ctypes.c_int32)
    p_dbl = ctypes.POINTER(ctypes.c_double)
    rc = lib.amd_order(
        n,
        Ap.ctypes.data_as(p_int),
        Ai.ctypes.data_as(p_int),
        P.ctypes.data_as(p_int),
        ctrl.ctypes.data_as(p_dbl),
        info.ctypes.data_as(p_dbl),
    )
    if rc != 0:
        raise RuntimeError(f"amd_order failed ({rc})")
    if not np.array_equal(np.sort(P), np.arange(n, dtype=np.int32)):
        raise RuntimeError("amd_order did not return a permutation")
    return int(np.count_nonzero(np.isfinite(info[info_size:])))


def main() -> int:
    from cenv import solver

    print(f"libamd: {solver._lib_path(('amd',), 'CENV_AMD')} "
          f"(find_library -> {ctypes.util.find_library('amd')})")
    print(f"solver._AMD_CONTROL = {solver._AMD_CONTROL}, "
          f"solver._AMD_INFO = {solver._AMD_INFO}")

    # the modules' own sizes must be the ABI's
    if solver._AMD_CONTROL < 5 or solver._AMD_INFO < 20:
        print("FAIL AMD_CONTROL/AMD_INFO smaller than the amd.h ABI (5/20)")
        return 1

    try:
        lib = solver._amd_lib()
    except RuntimeError as exc:
        print(f"SKIP libamd not present ({exc}); amd_perm is unreachable here")
        return 0

    import scipy.sparse as sp

    n = 400
    A = sp.diags([-1.0, 3.0, -1.0], [-1, 0, 1], shape=(n, n)).tocsc()
    A.sort_indices()
    Ap = np.ascontiguousarray(A.indptr, dtype=np.int32)
    Ai = np.ascontiguousarray(A.indices, dtype=np.int32)
    P = np.empty(n, dtype=np.int32)

    # 1. the sizes amd_perm uses: no write may reach the canary
    blown = _overflow(lib, n, Ap, Ai, P,
                      solver._AMD_CONTROL, solver._AMD_INFO)
    print(f"amd_perm's buffers: {blown} canary doubles touched")
    if blown:
        print("FAIL amd_order wrote past the Info buffer")
        return 1

    # 2. the pre-fix size (8), to prove the canary sees this class of bug
    small = _overflow(lib, n, Ap, Ai, P, 8, 8)
    print(f"regression probe at Info=8: {small} canary doubles touched "
          f"(expected > 0)")
    if not small:
        print("FAIL the canary cannot see an undersized Info buffer; "
              "this gate is not testing what it claims")
        return 1

    print("AMD ABI GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
