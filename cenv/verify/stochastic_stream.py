"""Stream proof for cenv/src/rng.c.

Checks, with 10**6 draws each, that `cenv_mt_*` is bit-identical to

  * numpy's legacy ``RandomState`` (``random_sample`` and the raw uint32
    stream), which is what ``RandomizedArrivalDepartureOccupancy`` consumes
    through ``random_state.rand()``;
  * numba's ``np.random`` inside ``@njit`` (``random()`` and ``randint(n)``),
    which is what ``stochastic_convection_simulator._convect_room`` consumes
    after ``_seed_kernel_rng(seed)``.

run:  starter-kit/.venv/bin/python -m cenv.verify.stochastic_stream
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np
from numba import njit

from cenv.bindings import bind, i64, lib  # noqa: F401  (import builds the library)
from cenv import bindings as _b

N = 1_000_000
SEEDS = (5, 17321, 0, 1, 12345, 0xFFFFFFFF)

_u32 = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS")


def _bind() -> None:
    global _create, _destroy, _fill_double, _fill_uint32, _fill_randint
    _create = bind("cenv_mt_create", ctypes.c_int64, [ctypes.c_uint32])
    _destroy = bind("cenv_mt_destroy", None, [ctypes.c_int64])
    _fill_double = bind(
        "cenv_mt_fill_double", ctypes.c_int32, [ctypes.c_int64, _b.f64, ctypes.c_int64]
    )
    _fill_uint32 = bind(
        "cenv_mt_fill_uint32", ctypes.c_int32, [ctypes.c_int64, _u32, ctypes.c_int64]
    )
    _fill_randint = bind(
        "cenv_mt_fill_randint",
        ctypes.c_int32,
        [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, i64],
    )


_bind()
_b.on_reload(_bind)

lib.cenv_version.restype = ctypes.c_char_p


def c_doubles(seed: int, n: int) -> np.ndarray:
    h = _create(seed)
    out = np.empty(n, dtype=np.float64)
    assert _fill_double(h, out, n) == 0
    _destroy(h)
    return out


def c_uint32(seed: int, n: int) -> np.ndarray:
    h = _create(seed)
    out = np.empty(n, dtype=np.uint32)
    assert _fill_uint32(h, out, n) == 0
    _destroy(h)
    return out


def c_randint(seed: int, max_incl: int, n: int) -> np.ndarray:
    h = _create(seed)
    out = np.empty(n, dtype=np.int64)
    assert _fill_randint(h, max_incl, n, out) == 0
    _destroy(h)
    return out


@njit(cache=False)
def nb_doubles(seed, n):
    np.random.seed(seed)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.random.random()
    return out


@njit(cache=False)
def nb_randint(seed, max_incl, n):
    np.random.seed(seed)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = np.random.randint(max_incl + 1)
    return out


def _cmp(label: str, a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        print(f"  FAIL {label}: shape {a.shape} != {b.shape}")
        return False
    bad = np.flatnonzero(a != b)
    if bad.size == 0:
        print(f"  ok   {label}: {a.size} values bit-identical")
        return True
    print(
        f"  FAIL {label}: {bad.size} of {a.size} differ, first at {bad[0]}: "
        f"{a[bad[0]]!r} != {b[bad[0]]!r}"
    )
    return False


def main() -> int:
    ok = True
    print(f"cenv.version = {lib.cenv_version().decode()}")
    print(f"numpy {np.__version__}, N = {N:,} draws per stream")

    for seed in SEEDS:
        print(f"seed {seed}:")
        ok &= _cmp(
            "RandomState.random_sample",
            c_doubles(seed, N),
            np.random.RandomState(seed).random_sample(N),
        )
        ok &= _cmp(
            "RandomState raw uint32",
            c_uint32(seed, N),
            np.random.RandomState(seed).randint(0, 2**32, size=N, dtype=np.uint32),
        )

    for seed in (5, 17321):
        print(f"seed {seed} (numba njit):")
        ok &= _cmp("np.random.random", c_doubles(seed, N), nb_doubles(seed, N))
        for max_incl in (0, 1, 2, 3, 4, 17, 100, 5075, 123456):
            ok &= _cmp(
                f"np.random.randint({max_incl + 1})",
                c_randint(seed, max_incl, N),
                nb_randint(seed, max_incl, N),
            )

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
