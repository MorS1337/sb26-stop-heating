"""ctypes loading of the cenv shared library.

Usage from a component module::

    from cenv.bindings import lib, f64, i32, i64, bind
    fn = bind("cenv_bk_zone_stats", ctypes.c_int32, [i64, f64, i32, f64, f64, f64])

``f64``/``i32``/``i64``/``u8`` are ``numpy.ctypeslib.ndpointer`` factories (a
contiguous array of that dtype); ``nd_f64(1)`` pins one dimension. Scalars are
plain ``ctypes.c_*`` objects. Importing this module triggers the (idempotent)
build, so a stale or missing library is recompiled on first import.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from . import build as _build

path: Path = _build.build()
lib: ctypes.CDLL = ctypes.CDLL(str(path))

# -- ndpointer factories ---------------------------------------------------- #
_ndp = np.ctypeslib.ndpointer


def nd_f64(ndim: int | None = None, flags: str = "C_CONTIGUOUS"):
    return _ndp(dtype=np.float64, ndim=ndim, flags=flags)


def nd_f32(ndim: int | None = None, flags: str = "C_CONTIGUOUS"):
    return _ndp(dtype=np.float32, ndim=ndim, flags=flags)


def nd_i32(ndim: int | None = None, flags: str = "C_CONTIGUOUS"):
    return _ndp(dtype=np.int32, ndim=ndim, flags=flags)


def nd_i64(ndim: int | None = None, flags: str = "C_CONTIGUOUS"):
    return _ndp(dtype=np.int64, ndim=ndim, flags=flags)


def nd_u8(ndim: int | None = None, flags: str = "C_CONTIGUOUS"):
    return _ndp(dtype=np.uint8, ndim=ndim, flags=flags)


f64 = nd_f64()
f32 = nd_f32()
i32 = nd_i32()
i64 = nd_i64()
u8 = nd_u8()

c_double = ctypes.c_double
c_float = ctypes.c_float
c_int = ctypes.c_int
c_int32 = ctypes.c_int32
c_int64 = ctypes.c_int64
c_void_p = ctypes.c_void_p


def bind(name: str, restype, argtypes):
    """Declare one exported symbol and return the ctypes callable."""
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


def rebuild(force: bool = True) -> Path:
    """Recompile and reload the library in place (development helper)."""
    global lib, path
    path = _build.build(force=force)
    lib = ctypes.CDLL(str(path))
    _bind_all()
    return path


def _bind_all() -> None:
    # Components bind their own symbols at import (see the module docstring);
    # this hook re-runs the built-in ones after a rebuild.
    for fn in list(_BINDINGS):
        fn()


_BINDINGS: list = []


def on_reload(fn):
    """Register a zero-argument callback to (re)declare argtypes after reload."""
    _BINDINGS.append(fn)
    return fn
