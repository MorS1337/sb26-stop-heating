"""C ports of the simulator's stochastic components, installed by monkeypatch.

Four drop-in replacements, each reproducing its Python original bit for bit
(verified by ``cenv/verify/stochastic_gate.py``):

* ``MT19937``   -- numpy's legacy ``RandomState`` and numba's ``np.random``, one
                   stream for both (``cenv/src/rng.c``);
* ``convection``-- ``StochasticConvectionSimulator.apply_convection`` and the
                   numba kernel ``_convect_room`` it calls;
* ``occupancy`` -- ``RandomizedArrivalDepartureOccupancy.average_zone_occupancy``
                   and the ``ZoneOccupant`` Markov chain it drives, including the
                   per-interval memo cache the simulator's call pattern relies
                   on;
* ``weather``   -- ``ReplayWeatherController.read_weather_csv``,
                   ``_get_interpolated_value``, ``get_current_temp`` and
                   ``get_current_humidity`` (CSV read + np.interp).

``install(sim_root)`` checks the working tree, then patches. ``CENV_SHADOW`` (or
``shadow=``) puts any subset into shadow mode: the Python original still
produces the value the simulator sees, the C port runs alongside it on its own
generator state, and every disagreement lands in ``GATE``. Shadow mode exists
for the gate; it is slower than plain Python.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time as _time
from pathlib import Path

import numpy as np

from .bindings import bind, f64, i32, i64, u8, on_reload

COMPONENT = "stochastic"
SUBCOMPONENTS = ("convection", "occupancy", "weather")

_c_double_p = ctypes.POINTER(ctypes.c_double)
_cpu = _time.process_time


# --------------------------------------------------------------------------- #
# C surface
# --------------------------------------------------------------------------- #
class _OccConfig(ctypes.Structure):
    _fields_ = [
        ("zone_assignment", ctypes.c_int32),
        ("earliest_arrival_hour", ctypes.c_int32),
        ("latest_arrival_hour", ctypes.c_int32),
        ("earliest_departure_hour", ctypes.c_int32),
        ("latest_departure_hour", ctypes.c_int32),
        ("p_arrival", ctypes.c_double),
        ("p_departure", ctypes.c_double),
        ("step_ns", ctypes.c_int64),
        ("origin_ns", ctypes.c_int64),
        ("n_grid", ctypes.c_int64),
        ("local_hour", ctypes.POINTER(ctypes.c_int32)),
        ("work_day", ctypes.POINTER(ctypes.c_ubyte)),
        ("seed", ctypes.c_uint32),
    ]


def _bind_all() -> None:
    g = globals()
    g["_mt_create"] = bind("cenv_mt_create", ctypes.c_int64, [ctypes.c_uint32])
    g["_mt_destroy"] = bind("cenv_mt_destroy", None, [ctypes.c_int64])
    g["_mt_set_state"] = bind(
        "cenv_mt_set_state", ctypes.c_int32, [ctypes.c_int64, u8, ctypes.c_int32]
    )
    g["_mt_get_state"] = bind(
        "cenv_mt_get_state",
        ctypes.c_int32,
        [ctypes.c_int64, u8, ctypes.POINTER(ctypes.c_int32)],
    )
    g["_mt_fill_double"] = bind(
        "cenv_mt_fill_double", ctypes.c_int32, [ctypes.c_int64, f64, ctypes.c_int64]
    )
    g["_mt_fill_uint32"] = bind(
        "cenv_mt_fill_uint32", ctypes.c_int32, [ctypes.c_int64, u8, ctypes.c_int64]
    )
    g["_mt_fill_randint"] = bind(
        "cenv_mt_fill_randint",
        ctypes.c_int32,
        [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, i64],
    )
    g["_conv_room_create"] = bind(
        "cenv_conv_room_create",
        ctypes.c_int64,
        [i64, i64, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32],
    )
    g["_conv_room_destroy"] = bind(
        "cenv_conv_room_destroy", None, [ctypes.c_int64]
    )
    g["_conv_room_size"] = bind(
        "cenv_conv_room_size", ctypes.c_int64, [ctypes.c_int64]
    )
    g["_conv_room_nnz"] = bind(
        "cenv_conv_room_nnz", ctypes.c_int64, [ctypes.c_int64]
    )
    g["_conv_room_csr"] = bind(
        "cenv_conv_room_csr", ctypes.c_int32, [ctypes.c_int64, i64, i64]
    )
    g["_conv_apply"] = bind(
        "cenv_conv_apply",
        ctypes.c_int32,
        [i64, ctypes.c_int64, ctypes.c_int64, f64, ctypes.c_int64,
         ctypes.c_int64, ctypes.c_double],
    )
    g["_occ_create"] = bind(
        "cenv_occ_create", ctypes.c_int64, [ctypes.POINTER(_OccConfig)]
    )
    g["_occ_destroy"] = bind("cenv_occ_destroy", None, [ctypes.c_int64])
    g["_occ_rng"] = bind("cenv_occ_rng", ctypes.c_int64, [ctypes.c_int64])
    g["_occ_draws"] = bind("cenv_occ_draws", ctypes.c_int64, [ctypes.c_int64])
    g["_occ_set_grid"] = bind(
        "cenv_occ_set_grid",
        ctypes.c_int32,
        [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, i32, u8],
    )
    g["_occ_average"] = bind(
        "cenv_occ_average",
        ctypes.c_int32,
        [ctypes.c_int64, ctypes.c_int32, ctypes.c_int64, ctypes.c_int64,
         _c_double_p],
    )
    g["_wx_load"] = bind("cenv_weather_load_csv", ctypes.c_int64, [ctypes.c_char_p])
    g["_wx_create"] = bind(
        "cenv_weather_create", ctypes.c_int64, [f64, f64, f64, ctypes.c_int64]
    )
    g["_wx_destroy"] = bind("cenv_weather_destroy", None, [ctypes.c_int64])
    g["_wx_query"] = bind(
        "cenv_weather_query",
        ctypes.c_int32,
        [ctypes.c_int64, ctypes.c_int64, ctypes.c_int32, _c_double_p],
    )
    g["_wx_copy"] = bind(
        "cenv_weather_copy", ctypes.c_int32, [ctypes.c_int64, f64, f64, f64]
    )
    g["_wx_bounds"] = bind(
        "cenv_weather_bounds",
        ctypes.c_int32,
        [ctypes.c_int64, _c_double_p, _c_double_p, ctypes.POINTER(ctypes.c_int64)],
    )
    g["_f2k"] = bind(
        "cenv_weather_fahrenheit_to_kelvin", ctypes.c_double, [ctypes.c_double]
    )


_bind_all()
on_reload(_bind_all)


# --------------------------------------------------------------------------- #
# gate bookkeeping
# --------------------------------------------------------------------------- #
class Gate:
    """Per-subcomponent agreement counters for shadow mode."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stats: dict[str, dict] = {}
        self.times: dict[str, list[float]] = {}

    def _stat(self, name: str) -> dict:
        st = self.stats.get(name)
        if st is None:
            st = self.stats[name] = {
                "calls": 0,
                "mismatches": 0,
                "max_abs_diff": 0.0,
                "first": None,
            }
        return st

    def ok(self, name: str) -> None:
        self._stat(name)["calls"] += 1

    def bad(self, name: str, detail=None, diff: float = 0.0) -> None:
        st = self._stat(name)
        st["calls"] += 1
        st["mismatches"] += 1
        if diff > st["max_abs_diff"]:
            st["max_abs_diff"] = float(diff)
        if st["first"] is None:
            st["first"] = detail

    def spend(self, name: str, c_seconds: float, py_seconds: float) -> None:
        """Accumulates the two paths' CPU seconds (shadow mode only)."""
        slot = self.times.get(name)
        if slot is None:
            slot = self.times[name] = [0.0, 0.0]
        slot[0] += c_seconds
        slot[1] += py_seconds

    def report(self) -> dict:
        out = {}
        for name, st in sorted(self.stats.items()):
            entry = dict(st, passed=st["mismatches"] == 0 and st["calls"] > 0)
            slot = self.times.get(name)
            if slot is not None:
                entry["cpu_c"] = round(slot[0], 4)
                entry["cpu_py"] = round(slot[1], 4)
            out[name] = entry
        return out

    def passed(self) -> bool:
        report = self.report()
        return bool(report) and all(v["passed"] for v in report.values())


GATE = Gate()

# Live per-model occupancy states, so the gate can report how many distinct
# zone ids the simulator actually used (two namespaces -- room names for the
# thermal path, coordinates for the reward -- would show up here as 2V).
_OCC_STATES: list = []


def _parse_shadow(shadow) -> frozenset[str]:
    if shadow is None:
        shadow = os.environ.get("CENV_SHADOW", "")
    if isinstance(shadow, str):
        items = [s.strip() for s in shadow.replace(",", " ").split() if s.strip()]
    else:
        items = [str(s) for s in shadow]
    if "all" in items:
        return frozenset(SUBCOMPONENTS)
    unknown = set(items) - set(SUBCOMPONENTS)
    if unknown:
        raise ValueError(f"unknown cenv.stochastic shadow set(s): {sorted(unknown)}")
    return frozenset(items)


# --------------------------------------------------------------------------- #
# MT19937 state exchange
# --------------------------------------------------------------------------- #
def _numpy_mt_state(rs) -> tuple[np.ndarray, int]:
    """(624 words, block index) of a numpy legacy RandomState."""
    name, keys, pos, *_ = rs.get_state()
    if name != "MT19937":
        raise RuntimeError(f"random_state is {name}, not MT19937")
    return np.ascontiguousarray(keys, dtype=np.uint32), int(pos)


def _mt_import(handle: int, words, index: int) -> None:
    words = np.ascontiguousarray(words, dtype=np.uint32)
    if _mt_set_state(handle, words.view(np.uint8), int(index)) != 0:
        raise RuntimeError("cenv_mt_set_state failed")


def _mt_export(handle: int) -> tuple[np.ndarray, int]:
    words = np.zeros(624, dtype=np.uint32)
    index = ctypes.c_int32()
    if _mt_get_state(handle, words.view(np.uint8), ctypes.byref(index)) != 0:
        raise RuntimeError("cenv_mt_get_state failed")
    return words, int(index.value)


def _numba_np_state() -> tuple[np.ndarray, int] | None:
    """numba's njit np.random state, or None when it cannot be read."""
    try:
        from numba import _helperlib

        index, words = _helperlib.rnd_get_state(
            _helperlib.rnd_get_np_state_ptr()
        )
        return np.ascontiguousarray(words, dtype=np.uint32), int(index)
    except Exception:  # pragma: no cover - exotic numba build
        return None


def _gin_seed(name: str):
    try:
        import gin

        return gin.query_parameter(name)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 1. convection
# --------------------------------------------------------------------------- #
class _ConvState:
    __slots__ = ("rng", "p", "distance", "rooms", "rooms_np", "coords",
                 "room_dict_id", "shape")

    def __init__(self) -> None:
        self.rng = 0
        self.p = 0.0
        self.distance = 2
        self.rooms: list[int] = []
        self.rooms_np = np.zeros(0, dtype=np.int64)
        self.coords: dict[str, list] = {}
        self.room_dict_id = 0
        self.shape = (0, 0)


def _conv_build(sim, room_dict, shape) -> _ConvState:
    st = _ConvState()
    st.p = float(sim.p)
    st.distance = int(sim.distance)
    st.shape = (int(shape[0]), int(shape[1]))
    st.room_dict_id = id(room_dict)
    unlimited = 1 if st.distance == -1 else 0
    for key, coords in room_dict.items():
        if key == "exterior_space" or key == "interior_wall":
            continue
        if not len(coords):
            continue
        arr = np.ascontiguousarray(np.asarray(coords, dtype=np.int64))
        handle = _conv_room_create(
            np.ascontiguousarray(arr[:, 0]), np.ascontiguousarray(arr[:, 1]),
            arr.shape[0], st.distance, unlimited,
        )
        if handle <= 0:
            raise RuntimeError(f"cenv_conv_room_create failed for room {key!r}")
        st.rooms.append(handle)
        st.coords[key] = coords
    st.rooms_np = np.ascontiguousarray(st.rooms, dtype=np.int64)
    seed = getattr(sim, "_cenv_seed", None)
    if seed is None:
        seed = _gin_seed("StochasticConvectionSimulator.seed")
    st.rng = _mt_create(int(seed) if seed is not None else 0)
    if st.rng <= 0:
        raise RuntimeError("cenv_mt_create failed for the convection generator")
    state = _numba_np_state()
    if state is not None:
        # Continue the stream the njit kernel is at right now, whatever the
        # install order was.
        _mt_import(st.rng, *state)
    return st


def _conv_state(sim, room_dict, shape) -> _ConvState:
    st = sim.__dict__.get("_cenv_conv")
    if (
        st is None
        or st.room_dict_id != id(room_dict)
        or st.shape != tuple(shape)
        or st.distance != int(sim.distance)
    ):
        if st is not None:
            _conv_dispose(st)
        st = _conv_build(sim, room_dict, shape)
        sim._cenv_conv = st
    return st


def _conv_dispose(st: _ConvState) -> None:
    for handle in st.rooms:
        _conv_room_destroy(handle)
    st.rooms = []
    st.rooms_np = np.zeros(0, dtype=np.int64)
    if st.rng:
        _mt_destroy(st.rng)
        st.rng = 0


def _install_convection(mod, shadow: bool) -> None:
    # Only the method is patched, never __init__: gin inspects the constructor
    # of a @gin.configurable class (and deep-copies its kwargs), and a replaced
    # __init__ breaks that. The generator state is lifted from numba's own
    # np.random state at first use instead, so wrapping __init__ would buy
    # nothing anyway.
    cls = mod.StochasticConvectionSimulator
    original_apply = cls.apply_convection

    def apply_convection(self, room_dict, temp):
        p = self._p
        distance = self._distance
        if p == 0 or distance == 0:
            return original_apply(self, room_dict, temp)
        if distance == -1 and p == 1:
            # CPython's random.shuffle over a list -- a different generator, and
            # no shipped config takes this branch.
            return original_apply(self, room_dict, temp)
        if (
            not isinstance(temp, np.ndarray)
            or temp.dtype != np.float64
            or temp.ndim != 2
            or not temp.flags["C_CONTIGUOUS"]
        ):
            return original_apply(self, room_dict, temp)

        st = _conv_state(self, room_dict, temp.shape)
        if not shadow:
            rc = _conv_apply(
                st.rooms_np, len(st.rooms), st.rng, temp, temp.shape[0],
                temp.shape[1], p,
            )
            if rc != 0:
                raise RuntimeError(f"cenv_conv_apply failed ({rc})")
            return None

        reference = temp.copy()
        t0 = _cpu()
        original_apply(self, room_dict, reference)
        t_py = _cpu() - t0
        t0 = _cpu()
        _conv_apply(
            st.rooms_np, len(st.rooms), st.rng, temp, temp.shape[0],
            temp.shape[1], p,
        )
        GATE.spend("convection", _cpu() - t0, t_py)
        if np.array_equal(reference, temp):
            GATE.ok("convection")
        else:
            diff = np.flatnonzero(reference.ravel() != temp.ravel())
            GATE.bad(
                "convection",
                f"cells={int(diff.size)} first={int(diff[0]) if diff.size else -1}",
                float(np.max(np.abs(reference.ravel()[diff] - temp.ravel()[diff]))),
            )
        return None

    original_candidates = getattr(cls, "candidates_for", None)

    def candidates_for(self, room_key, max_dist, cv):
        """The C room keeps the CSR; this rebuilds the numpy one on demand."""
        if original_candidates is None:
            raise AttributeError("candidates_for")
        st = self.__dict__.get("_cenv_conv")
        cache_key = (room_key, max_dist, max_dist == -1)
        if st is not None and cache_key not in self._room_arrays_cache:
            coords = st.coords.get(room_key)
            if coords is not None:
                self._room_arrays(room_key, coords, max_dist, max_dist == -1)
        return original_candidates(self, room_key, max_dist, cv)

    apply_convection._cenv_installed = True
    apply_convection._cenv_shadow = shadow
    cls.apply_convection = apply_convection
    if original_candidates is not None:
        cls.candidates_for = candidates_for


# --------------------------------------------------------------------------- #
# 2. occupancy
# --------------------------------------------------------------------------- #
class _OccState:
    """Per-model C state: generator position, zone indices, local-time grid."""

    __slots__ = ("handle", "zones", "origin_ns", "n_grid", "step_ns", "tz")

    GRID_BLOCK = 288 * 16  # 16 days per (re)build, extended on demand

    def __init__(self, model) -> None:
        self.zones: dict[str, int] = {}
        self.origin_ns = 0
        self.n_grid = 0
        self.step_ns = int(model.step_size.value)
        self.tz = model.time_zone

        placeholder_hour = np.zeros(1, dtype=np.int32)
        placeholder_work = np.zeros(1, dtype=np.uint8)
        cfg = _OccConfig()
        cfg.zone_assignment = int(model._zone_assignment)
        cfg.earliest_arrival_hour = int(model._earliest_expected_arrival_hour)
        cfg.latest_arrival_hour = int(model._latest_expected_arrival_hour)
        cfg.earliest_departure_hour = int(model._earliest_expected_departure_hour)
        cfg.latest_departure_hour = int(model._latest_expected_departure_hour)
        cfg.p_arrival = _event_probability(
            model, model._earliest_expected_arrival_hour,
            model._latest_expected_arrival_hour,
        )
        cfg.p_departure = _event_probability(
            model, model._earliest_expected_departure_hour,
            model._latest_expected_departure_hour,
        )
        cfg.step_ns = self.step_ns
        cfg.origin_ns = 0
        cfg.n_grid = 1
        cfg.local_hour = placeholder_hour.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int32)
        )
        cfg.work_day = placeholder_work.ctypes.data_as(
            ctypes.POINTER(ctypes.c_ubyte)
        )
        cfg.seed = 0
        self.handle = _occ_create(ctypes.byref(cfg))
        if self.handle <= 0:
            raise RuntimeError("cenv_occ_create failed")
        _OCC_STATES.append(self)

        # Start exactly where the model's own RandomState is, so the chain is
        # the same even if rows were drawn before install.
        state = model.__dict__.get("_random_state")
        if state is None:
            seed = _gin_seed("RandomizedArrivalDepartureOccupancy.seed")
            _mt_import(self.rng, *_numpy_mt_state(np.random.RandomState(
                int(seed) if seed is not None else 0)))
        else:
            _mt_import(self.rng, *_numpy_mt_state(state))

    @property
    def rng(self) -> int:
        return int(_occ_rng(self.handle))

    # -- local-time grid ---------------------------------------------------- #
    def _make_grid(self, origin_ns: int, n: int):
        import pandas as pd

        from smart_buildings.smart_control.utils import conversion_utils

        idx = pd.date_range(
            start=pd.Timestamp(int(origin_ns)),
            periods=int(n),
            freq=pd.Timedelta(self.step_ns),
        )
        if idx.tz is None:
            # ZoneOccupant._to_local_time: naive -> localize, then read .hour
            local = idx.tz_localize(
                self.tz, nonexistent="shift_forward", ambiguous=True
            )
        else:
            local = idx.tz_convert(self.tz)
        hours = np.ascontiguousarray(local.hour.to_numpy(), dtype=np.int32)
        dates = local.tz_localize(None).normalize()
        uniq, inverse = np.unique(dates.asi8, return_inverse=True)
        work_unique = np.array(
            [1 if conversion_utils.is_work_day(pd.Timestamp(int(u))) else 0
             for u in uniq],
            dtype=np.uint8,
        )
        work = np.ascontiguousarray(work_unique[inverse], dtype=np.uint8)
        return hours, work

    def _set_grid(self, hours, work) -> None:
        rc = _occ_set_grid(self.handle, self.origin_ns, hours.size, hours, work)
        if rc != 0:
            raise RuntimeError(f"cenv_occ_set_grid failed ({rc})")

    def ensure(self, start_ns: int) -> int:
        """Grid index of ``start_ns``, (re)building or extending the table."""
        step = self.step_ns
        if not self.n_grid:
            self.origin_ns = start_ns
            self.n_grid = self.GRID_BLOCK
            self._set_grid(*self._make_grid(self.origin_ns, self.n_grid))
            return 0
        delta = start_ns - self.origin_ns
        if delta % step == 0:
            idx = delta // step
            if 0 <= idx < self.n_grid:
                return int(idx)
            if idx >= self.n_grid:
                want = int(idx) + self.GRID_BLOCK
                self._set_grid(*self._make_grid(self.origin_ns, want))
                self.n_grid = want
                return int(idx)
        # A query before the current origin: a fresh episode or a rewound clock.
        self.origin_ns = start_ns
        want = max(self.GRID_BLOCK, self.n_grid)
        self._set_grid(*self._make_grid(self.origin_ns, want))
        self.n_grid = want
        return 0


def _event_probability(model, start_hour: int, end_hour: int) -> float:
    """ZoneOccupant._get_event_probability, evaluated exactly as Python does."""
    import pandas as pd

    window = pd.Timedelta(end_hour - start_hour, unit="hour")
    n_halfway = window / model.step_size / 2.0
    return 1.0 / n_halfway


def _install_occupancy(mod, shadow: bool) -> None:
    # Method-level patch only (see _install_convection): the C chain is seeded
    # from the model's own RandomState on first use.
    cls = mod.RandomizedArrivalDepartureOccupancy
    original = cls.average_zone_occupancy

    def average_zone_occupancy(self, zone_id, start_time, end_time):
        st = self.__dict__.get("_cenv_occ")
        if st is None:
            st = _OccState(self)
            self._cenv_occ = st
        zone = st.zones.get(zone_id)
        if zone is None:
            zone = len(st.zones)
            st.zones[zone_id] = zone
        start_ns = int(start_time.value)
        end_ns = int(end_time.value)
        st.ensure(start_ns)
        value = ctypes.c_double()
        t0 = _cpu()
        rc = _occ_average(st.handle, zone, start_ns, end_ns, ctypes.byref(value))
        t_c = _cpu() - t0
        if rc != 0:
            raise RuntimeError(f"cenv_occ_average failed ({rc}) for {zone_id}")
        if not shadow:
            return value.value
        t0 = _cpu()
        reference = original(self, zone_id, start_time, end_time)
        GATE.spend("occupancy", t_c, _cpu() - t0)
        if reference == value.value:
            GATE.ok("occupancy")
        else:
            GATE.bad(
                "occupancy",
                f"zone={zone_id} start={start_ns} end={end_ns} "
                f"c={value.value!r} py={reference!r}",
                abs(reference - value.value),
            )
        return reference

    average_zone_occupancy._cenv_installed = True
    average_zone_occupancy._cenv_shadow = shadow
    cls.average_zone_occupancy = average_zone_occupancy


# --------------------------------------------------------------------------- #
# 3. weather
# --------------------------------------------------------------------------- #
_WX_TABLES: dict[str, int] = {}


def _wx_table_for_path(path: str) -> int:
    """One C table per CSV file per process (each controller re-reads it)."""
    handle = _WX_TABLES.get(path)
    if handle:
        return handle
    handle = int(_wx_load(path.encode()))
    if handle <= 0:
        raise RuntimeError(f"cenv_weather_load_csv failed for {path!r}")
    _WX_TABLES[path] = handle
    return handle


def _wx_table_for(controller) -> int:
    handle = int(controller.__dict__.get("_cenv_weather") or 0)
    if handle:
        return handle
    df = controller.weather_df
    handle = int(getattr(df, "attrs", {}).get("cenv_weather_handle") or 0)
    if not handle:
        handle = _wx_table_for_path(str(controller.local_weather_path))
        lo = ctypes.c_double()
        hi = ctypes.c_double()
        n = ctypes.c_int64()
        _wx_bounds(handle, ctypes.byref(lo), ctypes.byref(hi), ctypes.byref(n))
        if n.value != int(df.shape[0]):
            raise RuntimeError(
                f"cenv weather: parsed {n.value} rows, pandas has {df.shape[0]}"
            )
    controller._cenv_weather = handle
    return handle


def _wx_value(controller, timestamp, which: int) -> float:
    out = ctypes.c_double()
    rc = _wx_query(_wx_table_for(controller), int(timestamp.value), which,
                   ctypes.byref(out))
    if rc == 1:
        raise ValueError(
            f"Timestamp not in range. Timestamp {timestamp} is outside the "
            f"weather table."
        )
    if rc != 0:
        raise RuntimeError(f"cenv_weather_query failed ({rc})")
    return out.value


def _install_weather(mod, shadow: bool) -> None:
    cls = mod.ReplayWeatherController
    original_read = cls.read_weather_csv
    original_interp = cls._get_interpolated_value
    original_temp = cls.get_current_temp
    original_humidity = cls.get_current_humidity

    def read_weather_csv(self, csv_filepath):
        import pandas as pd

        handle = _wx_table_for_path(str(csv_filepath))
        lo = ctypes.c_double()
        hi = ctypes.c_double()
        n = ctypes.c_int64()
        _wx_bounds(handle, ctypes.byref(lo), ctypes.byref(hi), ctypes.byref(n))
        xp = np.empty(n.value, dtype=np.float64)
        tf = np.empty(n.value, dtype=np.float64)
        hu = np.empty(n.value, dtype=np.float64)
        if _wx_copy(handle, xp, tf, hu) != 0:
            raise RuntimeError("cenv_weather_copy failed")
        df = pd.DataFrame(
            {
                "Time": pd.to_datetime(xp.astype(np.int64), unit="s", utc=True),
                "Humidity": hu,
                "TempF": tf,
            }
        )
        df.index = xp
        df.index.name = "SecondsSinceEpoch"

        if shadow:
            reference = original_read(self, csv_filepath)
            same_index = np.array_equal(
                np.asarray(reference.index, dtype=np.float64), xp
            )
            same_time = np.array_equal(
                reference["Time"].to_numpy(), df["Time"].to_numpy()
            )
            same_t = np.array_equal(
                reference["TempF"].to_numpy(dtype=np.float64), tf
            )
            same_h = np.array_equal(
                reference["Humidity"].to_numpy(dtype=np.float64), hu
            )
            if same_index and same_time and same_t and same_h:
                GATE.ok("weather.read_weather_csv")
            else:
                GATE.bad(
                    "weather.read_weather_csv",
                    f"index={same_index} time={same_time} tempf={same_t} "
                    f"humidity={same_h}",
                )

        df.attrs["cenv_weather_handle"] = handle
        self._cenv_weather = handle
        return df

    def _get_interpolated_value(self, timestamp, values):
        name = getattr(values, "name", None)
        if name == self.humidity_column:
            which = 1
        elif name == "TempF":
            which = 0
        else:
            return original_interp(self, timestamp, values)
        t0 = _cpu()
        got = _wx_value(self, timestamp, which)
        t_c = _cpu() - t0
        if not shadow:
            return got
        t0 = _cpu()
        reference = original_interp(self, timestamp, values)
        GATE.spend("weather.interp", t_c, _cpu() - t0)
        if reference == got:
            GATE.ok("weather.interp")
        else:
            GATE.bad(
                "weather.interp",
                f"t={timestamp} which={which} c={got!r} py={reference!r}",
                abs(reference - got),
            )
        return reference

    def get_current_temp(self, timestamp):
        got = _f2k(_wx_value(self, timestamp, 0))
        if not shadow:
            return got
        reference = original_temp(self, timestamp)
        if reference == got:
            GATE.ok("weather.get_current_temp")
        else:
            GATE.bad(
                "weather.get_current_temp",
                f"t={timestamp} c={got!r} py={reference!r}",
                abs(reference - got),
            )
        return reference

    def get_current_humidity(self, timestamp):
        got = _wx_value(self, timestamp, 1)
        if not shadow:
            return got
        reference = original_humidity(self, timestamp)
        if reference == got:
            GATE.ok("weather.get_current_humidity")
        else:
            GATE.bad(
                "weather.get_current_humidity",
                f"t={timestamp} c={got!r} py={reference!r}",
                abs(reference - got),
            )
        return reference

    for fn in (read_weather_csv, _get_interpolated_value, get_current_temp,
               get_current_humidity):
        fn._cenv_installed = True
        fn._cenv_shadow = shadow
    cls.read_weather_csv = read_weather_csv
    cls._get_interpolated_value = _get_interpolated_value
    cls.get_current_temp = get_current_temp
    cls.get_current_humidity = get_current_humidity


# --------------------------------------------------------------------------- #
# install
# --------------------------------------------------------------------------- #
_EXPECTED = {
    "stochastic_convection_simulator": (
        "StochasticConvectionSimulator",
        ("apply_convection", "_room_arrays", "candidates_for"),
    ),
    "randomized_arrival_departure_occupancy": (
        "RandomizedArrivalDepartureOccupancy",
        ("average_zone_occupancy",),
    ),
    "weather_controller": (
        "ReplayWeatherController",
        ("read_weather_csv", "_get_interpolated_value", "get_current_temp",
         "get_current_humidity"),
    ),
}


def _check_tree(root: Path) -> None:
    from smart_buildings.smart_control.simulator import (
        randomized_arrival_departure_occupancy as occ_mod,
        stochastic_convection_simulator as conv_mod,
        weather_controller as wx_mod,
    )

    for mod in (conv_mod, occ_mod, wx_mod):
        path = Path(mod.__file__).resolve()
        if root not in path.parents:
            raise RuntimeError(
                f"cenv.stochastic was built against {root}, but {path} is loaded"
            )
        class_name, methods = _EXPECTED[path.stem]
        cls = getattr(mod, class_name)
        for name in methods:
            if not hasattr(cls, name):
                raise RuntimeError(
                    f"unsupported simulator: {cls.__name__}.{name} is missing"
                )
    if not hasattr(conv_mod, "_convect_room"):
        raise RuntimeError("unsupported simulator: no numba _convect_room kernel")


def install(sim_root=None, shadow=None, verbose: bool = False) -> dict:
    """Monkeypatch the stochastic components of the loaded simulator.

    Args:
      sim_root: the simulator tree root (``sims/shipped`` or ``sims/corrected``);
        defaults to the tree ``smart_buildings`` was imported from.
      shadow: subcomponents to run in shadow mode, e.g. ``"convection,occupancy"``;
        defaults to ``$CENV_SHADOW``.
      verbose: print the patch points to stderr.

    Returns:
      ``{"component": "stochastic", "shadow": [...], "patched": {...}}``.
    """
    from smart_buildings.smart_control.simulator import (
        randomized_arrival_departure_occupancy as occ_mod,
        stochastic_convection_simulator as conv_mod,
        weather_controller as wx_mod,
    )

    if sim_root is None:
        mod = sys.modules.get("smart_buildings")
        if mod is None or not getattr(mod, "__file__", None):
            raise RuntimeError(
                "import smart_buildings (sbx.simload.use_sim) before installing cenv"
            )
        sim_root = Path(mod.__file__).resolve().parent.parent
    root = Path(sim_root).resolve()
    _check_tree(root)

    wanted = _parse_shadow(shadow)
    patched: dict[str, bool] = {}

    # A component can only be gated by the install that patches it: silently
    # installing a second time in a different mode would look like a passing
    # gate that never ran.
    for name, cls, method in (
        ("convection", conv_mod.StochasticConvectionSimulator, "apply_convection"),
        ("occupancy", occ_mod.RandomizedArrivalDepartureOccupancy,
         "average_zone_occupancy"),
        ("weather", wx_mod.ReplayWeatherController, "get_current_temp"),
    ):
        installed = getattr(getattr(cls, method), "_cenv_installed", False)
        shadowed = getattr(getattr(cls, method), "_cenv_shadow", False)
        if installed and (name in wanted) != shadowed:
            raise RuntimeError(
                f"cenv.stochastic.{name} is already installed "
                f"({'shadow' if shadowed else 'active'}); re-install in the same "
                f"mode is a no-op, but switching modes needs a fresh process"
            )

    if not getattr(conv_mod.StochasticConvectionSimulator.apply_convection,
                   "_cenv_installed", False):
        _install_convection(conv_mod, "convection" in wanted)
        patched["convection"] = True
    if not getattr(
        occ_mod.RandomizedArrivalDepartureOccupancy.average_zone_occupancy,
        "_cenv_installed", False,
    ):
        _install_occupancy(occ_mod, "occupancy" in wanted)
        patched["occupancy"] = True
    if not getattr(wx_mod.ReplayWeatherController.get_current_temp,
                   "_cenv_installed", False):
        _install_weather(wx_mod, "weather" in wanted)
        patched["weather"] = True

    if wanted and not getattr(install, "_cenv_atexit", False):
        import atexit

        def _report() -> None:
            if GATE.stats:
                payload = {
                    "components": GATE.report(),
                    "occupancy_zones": [len(st.zones) for st in _OCC_STATES],
                }
                print(
                    "cenv-gate: " + json.dumps(payload, sort_keys=True),
                    file=sys.stderr,
                    flush=True,
                )

        atexit.register(_report)
        install._cenv_atexit = True

    if verbose:
        print(
            f"cenv.stochastic: patched {sorted(patched)} "
            f"shadow={sorted(wanted)} root={root}",
            file=sys.stderr,
        )
    return {"component": COMPONENT, "shadow": sorted(wanted), "patched": patched}
