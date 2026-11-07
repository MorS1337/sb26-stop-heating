"""Build + load round-trip for the cenv shared library (no simulator needed).

    starter-kit/.venv/bin/python -m cenv.verify.smoke

Checks that the library builds from ``cenv/src/*.c``, loads through
``cenv.bindings``, exports the symbols each component declares, and that the
numpy-compatible pairwise sum matches ``np.add.reduce`` on the array sizes the
bookkeeping component actually reduces (zone cell counts).
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

REQUIRED_SYMBOLS = (
    "cenv_version",
    "cenv_selftest",
    "cenv_pairwise_sum",
    "cenv_bk_plan_create",
    "cenv_bk_plan_destroy",
    "cenv_bk_plan_n_zones",
    "cenv_bk_zone_stats",
    "cenv_bk_reward",
    "cenv_bk_zone_productivity",
)


def main() -> int:
    from cenv import build, bindings
    from cenv.bindings import c_double, c_int32, c_int64, f64, i64, lib

    path = build.build()
    print(f"library: {path}")
    missing = [name for name in REQUIRED_SYMBOLS if not hasattr(lib, name)]
    if missing:
        print(f"FAIL missing symbols: {missing}")
        return 1
    lib.cenv_version.restype = ctypes.c_char_p
    print(f"version: {lib.cenv_version().decode()}")

    # -- round-trip: buffer in, deterministic transform out ------------------ #
    lib.cenv_selftest.restype = c_int32
    lib.cenv_selftest.argtypes = [f64, c_int64, c_double]
    buf = np.arange(8.0)
    rc = lib.cenv_selftest(buf, buf.size, 2.5)
    expected = np.arange(8.0) * 2.5 + np.arange(8.0)
    if rc != 0 or not np.array_equal(buf, expected):
        print(f"FAIL selftest rc={rc} buf={buf} expected={expected}")
        return 1
    print(f"round-trip: ok ({buf[:4]} ...)")

    # -- pairwise sum vs numpy on the sizes bookkeeping reduces ------------- #
    pw = bindings.bind("cenv_pairwise_sum", c_double, [f64, c_int64])
    rng = np.random.default_rng(12345)
    worst = 0.0
    exact_small = True
    for n in (0, 1, 2, 7, 8, 9, 127, 128, 129, 588, 2297, 2880, 3131, 5076, 81388):
        x = rng.standard_normal(n)
        got = pw(x, n)
        want = float(np.add.reduce(x)) if n else 0.0
        diff = abs(got - want)
        worst = max(worst, diff)
        if got != want:
            if n <= 5076:
                exact_small = False
            print(f"  note: n={n} differs by {diff:.3e}")
    print(f"pairwise sum: worst |diff| vs np.add.reduce = {worst:.3e}; "
          f"bit-identical for every zone-sized array (<=5076): {exact_small}")
    if worst > 1e-12:
        print("FAIL pairwise sum")
        return 1

    # -- plan lifecycle ----------------------------------------------------- #
    lib.cenv_bk_plan_create.restype = c_int64
    lib.cenv_bk_plan_create.argtypes = [c_int64, i64, i64]
    lib.cenv_bk_plan_n_zones.restype = c_int64
    lib.cenv_bk_plan_n_zones.argtypes = [c_int64]
    lib.cenv_bk_plan_destroy.restype = None
    lib.cenv_bk_plan_destroy.argtypes = [c_int64]
    offsets = np.array([0, 3, 7], dtype=np.int64)
    cells = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    handle = lib.cenv_bk_plan_create(2, offsets, cells)
    if handle == 0:
        print("FAIL plan_create")
        return 1
    if lib.cenv_bk_plan_n_zones(handle) != 2:
        print("FAIL plan_n_zones")
        return 1
    temp = np.arange(10, dtype=np.float64)
    sel = np.array([0, 1], dtype=np.int64)
    out_min = np.zeros(2)
    out_max = np.zeros(2)
    out_mean = np.zeros(2)
    lib.cenv_bk_zone_stats.restype = c_int32
    lib.cenv_bk_zone_stats.argtypes = [
        c_int64, f64, c_int64, i64, c_int64,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    rc = lib.cenv_bk_zone_stats(
        handle, temp, temp.size, sel, 2,
        out_min.ctypes.data, out_max.ctypes.data, out_mean.ctypes.data, None,
    )
    lib.cenv_bk_plan_destroy(handle)
    if rc != 0:
        print(f"FAIL zone_stats rc={rc}")
        return 1
    want_min = np.array([0.0, 3.0])
    want_max = np.array([2.0, 6.0])
    want_mean = np.array([1.0, 4.5])
    if not (np.array_equal(out_min, want_min)
            and np.array_equal(out_max, want_max)
            and np.array_equal(out_mean, want_mean)):
        print(f"FAIL zone_stats min={out_min} max={out_max} mean={out_mean}")
        return 1
    print(f"plan/stats: ok (mean={out_mean})")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
