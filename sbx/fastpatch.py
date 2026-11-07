"""Opt-in runtime accelerator for the shipped Smart Buildings simulator.

``--fast`` (``sbx.evalrun`` / ``sbx.tune``) calls :func:`apply` immediately after
``sbx.simload.use_sim``, i.e. before the environment is constructed. Nothing
under ``sims/`` or ``starter-kit/`` is edited: every patch is a monkeypatch of a
class attribute on the already-imported modules, and the official notebook path
(evaluate.py, unpatched) never imports this module.

Usage::

    from sbx import simload, fastpatch
    simload.use_sim("shipped")
    fastpatch.apply(simload.sim_root("shipped"))   # idempotent

    python -m sbx.evalrun --policy policies.ctrl_w:CtrlW --scenario \\
        building_train_2/atlanta_3 --steps 8640 --sim shipped --fast

What is patched, and why it does not change the numbers
-------------------------------------------------------
``FiniteDifferenceSolver._add_air_side``
    The VAV air-side coupling is already a Woodbury update on the existing
    factorization, but it still pays one triangular solve per step to evaluate
    ``A^{-1} (P q)``. ``A^{-1} P`` is constant for a given factorization, and the
    same blocked right-hand sides are already being solved to build
    ``_sensed_response``, so the dense response is kept alongside it, stored
    transposed as k x n float64 (164 MB on the largest shipped building), and
    the per-step solve becomes ``power @ response``: a stream of k contiguous
    n-vectors, which measured 3.3 ms/step against 6.4 ms for the row-major
    layout and 19.5 ms for the triangular solve it replaces. ``_sensed_response`` itself is still computed with
    the shipped expression on the shipped blocks, so the k x k system and its
    solution are bit-identical; only the final ``A^{-1} P q`` changes, by ~1e-16
    relative. The cache is rebuilt whenever ``_sensed_response`` is None, which
    is exactly where the shipped code invalidates it (new factorization in
    ``solve``, new coupling in ``couple_air_side``).

``FloorPlanBasedBuilding.get_zone_average_temps``
    Called three or four times per step (VAV update, thermal step, reward --
    twice in ``--components parity``), each time re-gathering every zone's CVs
    with fancy indexing and computing min/max/mean where only the mean is used.
    The patched method memoizes on the identity of ``building.temp`` (the solve
    assigns a fresh array every step, and ``apply_convection`` is wrapped to
    drop the memo because it mutates in place) and computes every zone mean in
    one flat gather plus one ``np.mean`` per zone, which is the same values in
    the same order as the shipped per-zone call, so the results are bit-identical.

``SetpointSchedule.is_comfort_mode`` / ``get_temperature_window`` /
``_localize_or_convert_timezone`` / ``is_weekend`` and
``ZoneOccupant._to_local_time``
    Pure functions of a timestamp, called a few hundred times per step for the
    handful of distinct timestamps a step contains, each doing a pytz
    conversion (and, for ``is_weekend``, building a day-name string). Memoized
    on ``(timestamp.value, timestamp.tz)``: per instance where the answer also
    depends on schedule parameters, and in one dict shared by every zone where
    it depends only on the time zone -- all 85 zones of a building share a zone,
    so the shared form turns 85 identical conversions per timestamp into one.
    Identical inputs, identical returns.

``SimulatorFlexibleGeometries._get_zone_reward_infos`` /
``_get_air_handler_reward_infos``
    Both rebuild static data inside their per-zone / per-VAV loops: a fresh
    ``room_areas`` dict (per zone!), VAV zone lists and VAV area shares, plus a
    setpoint window lookup per zone. Those are hoisted into a per-simulator plan
    (keyed on the identity of hvac and building). The dynamic per-step sums are
    kept in the shipped order, so every value is bit-identical, and the reward
    protos are still rebuilt once per call, because the parity reward call is
    *not* numerically neutral (boiler sensor advance) and must still run. The
    per-zone infos themselves are a pure function of the temperature field and
    the timestamp, so the second call of a step reuses the first call's map
    instead of rebuilding 85 protos.

``FiniteDifferenceSolver.solve``
    The forcing vector is gathered with a cached integer index instead of a
    boolean mask (652 us vs 325 us for the same selection on the largest
    building) and accumulated in place, saving four n-vector temporaries per
    step. Same arithmetic in the same order.

``FloorPlanBasedBuilding.apply_thermal_power_zone``
    Same assignment through a precomputed flat index vector instead of a 2-D
    fancy index (``input_q.ravel()[flat] = power / n``), 3x cheaper, same
    values. Falls back to the shipped body when ``input_q`` is not C-contiguous.

``FloorPlanBasedBuilding.apply_convection``
    Wrapper that keeps the zone-temperature memo honest (see above).

Exactness, measured
-------------------
``policies.ctrl_w:CtrlW``, 8641 steps, ``--components parity``, unpatched vs
``--fast``, same process configuration (records in
``scratch/FastExact/records``)::

    scenario                      |d total|  max |d reward/step|  max |d T|
    shipped  train_8/fairbanks_8        0               0        2.3e-13 K
    shipped  train_1/atlanta_2          0               0        2.3e-13 K
    shipped  train_2/phoenix_6          0               0        2.3e-13 K
    corrected train_2/atlanta_3         0               0        2.8e-13 K

The totals agree to every digit, not merely to 1e-9; the only difference is
the last bits of the temperature field, which is what replacing one triangular
solve with a matvec on the same factors costs.

That table is the final field of an 8641-step closed-loop episode. Sampled over
the whole trajectory instead, the worst component-relative deviation is 1.0e-15
(shipped ``train_8/fairbanks_8``, 1000 steps, every step sampled) and 1.2e-15
(corrected ``train_2/atlanta_3``, 800 steps, every 8th sampled), against the
project's 1e-12 standard, and it does not grow with step count (max |dT| stays
at 1.7e-13 K from step 400 to step 1000). The unpatched legs also reproduce the
pre-patch baseline ``results/ctrw_all`` bit-for-bit on all three shipped
scenarios, which is the evidence that ``--fast`` off changes nothing.

Speed, measured (single process, CtrlW, 300 steps x 5 alternating runs, median
of steps 40+; per-step CPU, this machine). Quiet machine::

    building_train_8/fairbanks_8   13.22 ->  7.82 ms   1.69x
    building_train_1/atlanta_2     25.13 -> 13.04 ms   1.93x
    building_train_2/atlanta_3     42.39 -> 24.64 ms   1.72x   (largest)

and with the ``--cenv`` combination in the same interleaved A/B, while the
machine was loaded (7-18 on 8 cores: the absolute milliseconds are ~2x
inflated, the ratios are not)::

    scenario (tree)                    plain    fast   cenv  f+cenv   fast/plain
    train_8/fairbanks_8 (shipped)     29.38   17.85   18.58   15.96      1.65x
    train_2/phoenix_6   (shipped)     77.82   49.23   56.65   56.85      1.58x
    train_2/atlanta_3   (corrected)   91.70   58.28   79.00   59.92      1.57x

``--fast --cenv all`` is bit-identical to ``--cenv all`` alone (max |dT| = 0.0 K
over 400 steps) because ``cenv``'s ``thermal`` component replaces ``solve``
wholesale — the air-side machinery here is then never built and never runs, only
the schedule/zone-mean/reward patches contribute (1.00x-1.32x over ``--cenv``
alone, 1.37x-1.84x over plain Python).

Cost: the dense response is k*n float64, 164 MB on ``building_train_2``
(measured peak RSS +170 MB per worker: plain 1076 MB vs 1246 MB on a 60-step
``phoenix_6`` run), so a run with many workers on the largest buildings wants
the memory; ``apply(..., dense_air_side=False)`` restores the shipped solve.
Under ``--fast --cenv all`` the dense response is never built (``cenv``'s
``thermal`` owns ``solve``), so peak RSS equals ``--cenv all`` alone.

What limits it: on ``building_train_2`` the patched step is 24.6 ms, of which
9.1 ms is the one sparse triangular solve that no patch can remove (the air-side
solve it replaces cost 9.3 ms and is the dense matvec now, 2.7 ms), so a step is
already close to the cost of the physics alone. The remainder is numba
convection, the occupancy Markov chains (which consume the shared
``RandomState`` and so cannot be skipped) and protobuf assembly in the
observation and reward paths -- the parts ``--cenv`` ports to C.

``--fast`` off imports nothing here and is byte-identical to the previous
behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["apply", "installed"]

_INSTALLED: list[str] = []
_MISSING = object()
# Pure timestamp->value memos shared by every instance of a class (all zones of
# a building use the same time zone, so per-instance caches would each miss).
_GLOBAL_MEMO: dict[str, dict] = {}
_GLOBAL_MEMO_CAP = 8192
# A dense Woodbury response is k*n float64 (164 MB on the largest shipped
# building: 241k solved CVs x 85 VAVs). Past this cap the shipped triangular
# solve is used instead -- still correct, just slower -- so a pathological
# floor plan cannot turn into a memory hog.
DENSE_RESPONSE_MAX_BYTES = 512_000_000


def installed() -> tuple[str, ...]:
    """Names of the patches installed by the most recent :func:`apply`."""
    return tuple(_INSTALLED)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mark(fn, name: str):
    fn.__fastpatch__ = name
    return fn


def _patched(fn) -> bool:
    return bool(getattr(fn, "__fastpatch__", False))


def _plain(fn):
    """The undecorated function behind a (possibly patched) class attribute."""
    return getattr(fn, "__fastpatch_orig__", fn)


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _stamp_key(timestamp):
    """Hashable identity of a ``pd.Timestamp`` that cannot alias another one.

    ``value`` alone would conflate a naive and a tz-aware stamp for the same
    wall clock, and the two take different branches in every timezone helper
    here; ``tz`` alone obviously does not identify the instant.
    """
    return (timestamp.value, timestamp.tz)


# --------------------------------------------------------------------------- #
# (a) air side: one dense response instead of a second triangular solve
# --------------------------------------------------------------------------- #
def _patch_air_side(solver_mod, dense: bool) -> list[str]:
    cls = solver_mod.FiniteDifferenceSolver
    current = cls._add_air_side
    if _patched(current):
        return []
    original = current
    block = int(getattr(solver_mod, "_RESPONSE_BLOCK", 64))
    max_bytes = DENSE_RESPONSE_MAX_BYTES
    installed = ["air_side"]

    def _add_air_side(
        self, factorization, uncoupled, conductance, supply_temperature
    ):
        if self._applied is None:
            raise ValueError(
                "solve was given air-side conductances but no term was"
                " registered with couple_air_side."
            )
        if supply_temperature is None:
            raise ValueError(
                "solve needs a supply temperature for every air-side"
                " conductance."
            )
        response = getattr(self, "_fastpatch_dense_response", _MISSING)
        if self._sensed_response is None or response is _MISSING:
            n_terms = self._applied.shape[1]
            n_rows = self._applied.shape[0]
            keep = dense and n_rows * n_terms * 8 <= max_bytes
            # Stored transposed (k, n): a step then evaluates ``power @ R``,
            # whose inner loop walks one contiguous n-vector per VAV. The
            # row-major (n, k) layout makes dgemv use its transposed path and
            # measured 6.4 ms against 3.3 ms per step on the largest building.
            response = (
                np.empty((n_terms, n_rows), dtype=np.float64) if keep else None
            )
            sensed_response = np.empty((n_terms, n_terms), dtype=np.float64)
            for start in range(0, n_terms, block):
                stop = min(start + block, n_terms)
                solved_block = factorization.solve(
                    self._applied[:, start:stop].toarray()
                )
                # The shipped expression, evaluated on the shipped blocks, so
                # _sensed_response keeps its exact value.
                sensed_response[:, start:stop] = self._sensed @ solved_block
                if keep:
                    response[start:stop] = solved_block.T
            self._sensed_response = sensed_response
            self._fastpatch_dense_response = response
        conductance = np.asarray(conductance, dtype=np.float64).reshape(-1)
        supply_temperature = np.asarray(
            supply_temperature, dtype=np.float64
        ).reshape(-1)
        power = np.linalg.solve(
            np.eye(len(conductance))
            + conductance[:, np.newaxis] * self._sensed_response,
            conductance * (supply_temperature - self._sensed @ uncoupled),
        )
        if response is None:
            return original(
                self, factorization, uncoupled, conductance, supply_temperature
            )
        return uncoupled + power @ response

    _mark(_add_air_side, "air_side")
    _add_air_side.__fastpatch_orig__ = original
    cls._add_air_side = _add_air_side
    return installed


# --------------------------------------------------------------------------- #
# (b) zone averages: memoized per temperature field, one vectorized pass
# --------------------------------------------------------------------------- #
def _zone_plan(building, constants_mod):
    """Static per-zone indexing for ``get_zone_average_temps``."""
    rooms = building.room_dict
    names = [
        zone for zone in rooms if not constants_mod.is_non_physical_space(zone)
    ]
    ncols = building.temp.shape[1]
    flat = []
    for zone in names:
        rows, cols = building._zone_indices(zone)
        flat.append(rows.astype(np.int64) * ncols + cols.astype(np.int64))
    counts = np.array([f.size for f in flat], dtype=np.int64)
    if counts.size == 0 or counts.min() == 0:
        # A zero-cell zone: keep the shipped path for it (mean of nothing).
        return names, None, None, None
    flat_all = np.concatenate(flat) if flat else np.empty(0, dtype=np.int64)
    starts = np.zeros(len(flat), dtype=np.int64)
    if len(flat) > 1:
        starts[1:] = np.cumsum(counts)[:-1]
    stops = starts + counts
    return names, flat_all, starts, stops


def _patch_zone_averages(
    building_mod, constants_mod, exact: bool
) -> list[str]:
    cls = building_mod.FloorPlanBasedBuilding
    current = cls.get_zone_average_temps
    if _patched(current):
        return []
    original = current

    def get_zone_average_temps(self):
        temp = self.temp
        cache = getattr(self, "_fastpatch_zone_cache", None)
        if cache is not None and cache[0] is temp:
            return dict(cache[1])
        plan = getattr(self, "_fastpatch_zone_plan", None)
        if plan is None or plan[0] is not self.room_dict:
            plan = (self.room_dict,) + _zone_plan(self, constants_mod)
            self._fastpatch_zone_plan = plan
        _, names, flat_all, starts, stops = plan
        if flat_all is None:
            result = original(self)
        else:
            values = temp.ravel()[flat_all]
            if exact:
                means = np.array(
                    [values[s:e].mean() for s, e in zip(starts, stops)],
                    dtype=np.float64,
                )
            else:
                means = np.add.reduceat(values, starts) / (
                    stops - starts
                ).astype(np.float64)
            result = dict(zip(names, means.tolist()))
        self._fastpatch_zone_cache = (temp, result)
        return dict(result)

    _mark(get_zone_average_temps, "zone_averages")
    get_zone_average_temps.__fastpatch_orig__ = original
    cls.get_zone_average_temps = get_zone_average_temps

    # apply_convection mutates the field in place, so it must drop the memo.
    conv = getattr(cls, "apply_convection", None)
    if conv is not None and not _patched(conv):
        conv_original = conv

        def apply_convection(self):
            conv_original(self)
            self._fastpatch_zone_cache = None

        _mark(apply_convection, "zone_averages_cache_drop")
        apply_convection.__fastpatch_orig__ = conv_original
        cls.apply_convection = apply_convection
    return ["zone_averages"]


# --------------------------------------------------------------------------- #
# (c) timestamp memoization
# --------------------------------------------------------------------------- #
def _memoize_single(fn, name: str, cap: int = 8):
    """Per-object memo keyed on ``(timestamp.value, timestamp.tz)``.

    A one-entry cache is not enough: within a step these helpers are called with
    a handful of timestamps in rotation (the current one, ``+1 step``, ``+60
    min``, ``-5 min``, and the thermostat's previous stamp), so an alternating
    pattern would miss every time. ``cap`` bounds the dict per object.
    """
    original = _plain(fn)
    attr = f"_fastpatch_memo_{name}"

    def memoized(self, timestamp):
        cache = getattr(self, attr, None)
        if cache is None:
            cache = {}
            setattr(self, attr, cache)
        key = _stamp_key(timestamp)
        value = cache.get(key, _MISSING)
        if value is _MISSING:
            value = original(self, timestamp)
            if len(cache) >= cap:
                cache.clear()
            cache[key] = value
        return value

    _mark(memoized, name)
    memoized.__fastpatch_orig__ = original
    return memoized


def _memoize_shared(fn, name: str):
    """Timestamp memo shared by every instance, keyed on its ``_time_zone``.

    ``_localize_or_convert_timezone``, ``is_weekend`` and ``_to_local_time``
    depend on the timestamp and on the object's time zone and nothing else, and
    every zone of a building carries the same zone. A per-instance cache still
    misses once per (schedule, timestamp) -- 85 schedules is 85 misses for one
    value -- while this one has two or three entries per step. Only pure
    functions may use it; the name namespaces the dict so two functions that
    share a time zone cannot read each other's answers.
    """
    original = _plain(fn)
    store = _GLOBAL_MEMO.setdefault(name, {})

    def memoized(self, timestamp):
        key = (getattr(self, "_time_zone"), timestamp.value, timestamp.tz)
        value = store.get(key, _MISSING)
        if value is _MISSING:
            value = original(self, timestamp)
            if len(store) >= _GLOBAL_MEMO_CAP:
                store.clear()
            store[key] = value
        return value

    _mark(memoized, name)
    memoized.__fastpatch_orig__ = original
    return memoized


def _patch_schedules(schedule_mod, shared: bool = True) -> list[str]:
    cls = schedule_mod.SetpointSchedule
    changed = False
    for name in (
        "is_comfort_mode",
        "get_temperature_window",
        "_localize_or_convert_timezone",
    ):
        current = getattr(cls, name)
        if _patched(current):
            continue
        changed = True
        if shared and name == "_localize_or_convert_timezone":
            setattr(cls, name, _memoize_shared(current, f"schedule{name}"))
        else:
            # These also read morning/evening hours and holidays, so the key
            # would have to carry them; a per-instance cache is exact and the
            # expensive helpers underneath are shared.
            setattr(cls, name, _memoize_single(current, name))
    if shared and not _patched(cls.is_weekend):
        cls.is_weekend = _memoize_shared(cls.is_weekend, "schedule.is_weekend")
        changed = True
    return ["schedule_memo"] if changed else []


def _patch_occupancy_timezone(occ_mod, shared: bool = True) -> list[str]:
    cls = getattr(occ_mod, "ZoneOccupant", None)
    if cls is None:
        return []
    current = getattr(cls, "_to_local_time", None)
    if current is None or _patched(current):
        return []
    if shared:
        namespace = occ_mod.__name__.rsplit(".", 1)[-1]
        cls._to_local_time = _memoize_shared(
            current, f"{namespace}._to_local_time"
        )
    else:
        cls._to_local_time = _memoize_single(current, "_to_local_time")
    return ["occupancy_timezone_memo"]


# --------------------------------------------------------------------------- #
# (d) reward info assembly
# --------------------------------------------------------------------------- #
def _reward_plan(sim):
    """Static (zones, air handler) plan for the two reward-info builders."""
    plan = getattr(sim, "_fastpatch_reward_plan", None)
    if plan is not None and plan[0] is sim._hvac and plan[1] is sim.building:
        return plan[2]

    hvac = sim._hvac
    building = sim.building
    room_areas = {z: len(coords) for z, coords in building.room_dict.items()}
    zone_temps = building.get_zone_average_temps()

    zones = []
    for zone in zone_temps:
        vav_ids = hvac.get_vav_ids_for_zone(zone)
        vavs = [hvac.vavs[v_id] for v_id in vav_ids]
        if not vavs:
            zones.append((zone, None, None))
            continue
        schedule = vavs[0].thermostat.get_setpoint_schedule()
        shares = []
        for vav in vavs:
            serviced = hvac.get_zones_for_vav(vav.device_id())
            total_area = sum(room_areas[z] for z in serviced)
            shares.append((vav, room_areas[zone] / total_area))
        zones.append((zone, schedule, shares))

    air_handler = hvac.air_handler
    vav_areas = []
    for v_id, vav in hvac.vavs.items():
        assigned = [z for z in hvac.get_zones_for_vav(v_id) if z in zone_temps]
        if not assigned:
            continue
        v_area = sum(room_areas[z] for z in assigned)
        vav_areas.append((vav.air_handler.device_id(), assigned, v_area))

    data = {
        "room_areas": room_areas,
        "zones": zones,
        "vav_areas": vav_areas,
        "air_handler_id": air_handler.device_id(),
    }
    sim._fastpatch_reward_plan = (hvac, building, data)
    return data


def _patch_reward_infos(flex_mod, air_handler_mod) -> list[str]:
    cls = flex_mod.SimulatorFlexibleGeometries
    reward_cls = flex_mod.RewardInfo

    changed = False
    current = cls._get_zone_reward_infos
    if not _patched(current):
        changed = True
        original_zones = current
        warned: set[str] = set()

        def _get_zone_reward_infos(self, occupancy_function):
            # The per-zone infos are a pure function of (temperature field,
            # timestamp): no VAV, thermostat or occupancy state can change
            # between two calls within one step, and both reward_info() calls
            # of a step happen after that step's solve. So the second call (the
            # parity one) reuses the first call's infos instead of rebuilding
            # 85 protos. Keyed on the identity of the temperature field, which
            # the solver replaces every step, plus the timestamp.
            temp = self.building.temp
            stamp = _stamp_key(self._current_timestamp)
            cached = getattr(self, "_fastpatch_zone_info_cache", None)
            if cached is not None and cached[0] is temp and cached[1] == stamp:
                return cached[2]
            plan = _reward_plan(self)
            zone_temps = self.building.get_zone_average_temps()
            timestamp = self._current_timestamp
            end = timestamp + flex_mod.pd.Timedelta(self._time_step_sec, unit="s")
            infos = {}
            for zone, schedule, shares in plan["zones"]:
                if schedule is None:
                    if zone not in warned:
                        warned.add(zone)
                        flex_mod.logging.warning(
                            "Zone %s has no servicing VAVs.", zone
                        )
                    infos[zone] = reward_cls.ZoneRewardInfo()
                    continue
                heating, cooling = schedule.get_temperature_window(timestamp)
                flow_setpoint = 0.0
                flow_actual = 0.0
                for vav, share in shares:
                    flow_setpoint += vav.max_air_flow_rate * share
                    flow_actual += vav.flow_rate_demand * share
                occupancy = occupancy_function.average_zone_occupancy(
                    zone, timestamp, end
                )
                infos[zone] = reward_cls.ZoneRewardInfo(
                    heating_setpoint_temperature=heating,
                    cooling_setpoint_temperature=cooling,
                    zone_air_temperature=zone_temps[zone],
                    air_flow_rate_setpoint=flow_setpoint,
                    air_flow_rate=flow_actual,
                    average_occupancy=occupancy,
                )
            self._fastpatch_zone_info_cache = (temp, stamp, infos)
            return infos

        _mark(_get_zone_reward_infos, "zone_reward_infos")
        _get_zone_reward_infos.__fastpatch_orig__ = original_zones
        cls._get_zone_reward_infos = _get_zone_reward_infos

    current = cls._get_air_handler_reward_infos
    if not _patched(current):
        changed = True
        original_ahu = current
        collections = flex_mod.collections

        def _get_air_handler_reward_infos(self):
            plan = _reward_plan(self)
            hvac = self._hvac
            air_handler = hvac.air_handler
            blower = air_handler.compute_intake_fan_energy_rate() + (
                air_handler.compute_exhaust_fan_energy_rate()
            )
            ambient = self._weather_controller.get_current_temp(
                self._current_timestamp
            )
            zone_temps = self.building.get_zone_average_temps()
            mixing = getattr(self, "_recirculation_temps", None)
            if mixing is not None:
                # The corrected tree replaced the area-weighted mixing with an
                # airflow-weighted one that reads the dampers set this step.
                # That formula is the tree's business, not this patch's: call
                # it, exactly as the shipped body would have.
                recirculation = mixing(zone_temps)
            else:
                room_areas = plan["room_areas"]
                temp_weighted = collections.defaultdict(float)
                area_total = collections.defaultdict(float)
                for ahu_id, assigned, v_area in plan["vav_areas"]:
                    if v_area <= 0:
                        continue
                    weighted = 0.0
                    for zone in assigned:
                        weighted += zone_temps[zone] * room_areas[zone]
                    sensing = weighted / v_area
                    temp_weighted[ahu_id] += sensing * v_area
                    area_total[ahu_id] += v_area

                if isinstance(air_handler, air_handler_mod.AirHandlerSystem):
                    ahus = air_handler.ahus
                else:
                    ahus = [air_handler]

                recirculation = {}
                for ahu in ahus:
                    ahu_id = ahu.device_id()
                    if area_total.get(ahu_id, 0) > 0:
                        recirculation[ahu_id] = (
                            temp_weighted[ahu_id] / area_total[ahu_id]
                        )
                    else:
                        recirculation[ahu_id] = self.building.temp.mean()

            if isinstance(air_handler, air_handler_mod.AirHandlerSystem):
                recirculation_input = recirculation
            else:
                recirculation_input = recirculation[air_handler.device_id()]

            cooling = air_handler.compute_cooling_electrical_energy_rate(
                recirculation_input, ambient
            )
            return {
                plan["air_handler_id"]: reward_cls.AirHandlerRewardInfo(
                    blower_electrical_energy_rate=blower,
                    air_conditioning_electrical_energy_rate=cooling,
                )
            }

        _mark(_get_air_handler_reward_infos, "air_handler_reward_infos")
        _get_air_handler_reward_infos.__fastpatch_orig__ = original_ahu
        cls._get_air_handler_reward_infos = _get_air_handler_reward_infos
    return ["reward_infos"] if changed else []


# --------------------------------------------------------------------------- #
# (e) timestep assembly: integer gathers into preallocated buffers
# --------------------------------------------------------------------------- #
def _patch_solver_solve(solver_mod) -> list[str]:
    """Same arithmetic, without the boolean-mask gathers and temporaries.

    ``solve`` assembles the forcing vector from the solved half of the
    temperature and power fields. ``field[mask]`` costs a full mask scan on
    every call (measured 652 us against 325 us for the same selection with a
    cached integer index vector), and the four temporaries of n float64 are
    avoidable. The values and their order are unchanged.
    """
    cls = solver_mod.FiniteDifferenceSolver
    current = cls.solve
    if _patched(current):
        return []
    if getattr(current, "__cenv__", None):
        # A C replacement (cenv.install("thermal")) is already in place and
        # owns the whole step, air side included. Wrapping it with a Python
        # reimplementation would only make it slower; stand down. evalrun
        # installs --cenv after --fast, so in practice this is the accidental
        # order case.
        return []
    original = current
    splu = solver_mod.scipy.sparse.linalg.splu

    def solve(
        self,
        temp,
        input_q,
        ambient_temperature,
        convection_coefficient,
        air_side_conductance=None,
        supply_temperature=None,
    ):
        factorization = self._factorization
        if (
            factorization is None
            or convection_coefficient != self._factorized_at
        ):
            factorization = splu(
                self.matrix(convection_coefficient), permc_spec="MMD_AT_PLUS_A"
            )
            self._factorization = factorization
            self._factorized_at = convection_coefficient
            self._sensed_response = None

        buffers = getattr(self, "_fastpatch_solve_buffers", None)
        if buffers is None or buffers[0] is not self._solved:
            # ``_solved`` is the index vector of the solved CVs (it is used both
            # as ``field[_solved]`` and as ``mask[_solved] = False`` in the
            # shipped code); accept a boolean mask too, since both index forms
            # select the same positions in the same order.
            index = self._solved
            if index.dtype == np.bool_:
                index = np.flatnonzero(index)
            index = np.ascontiguousarray(index, dtype=np.intp)
            scratch = np.empty(index.size, dtype=np.float64)
            buffers = (self._solved, index, scratch)
            self._fastpatch_solve_buffers = buffers
        _, index, scratch = buffers

        exposure = self.ambient_conductance(convection_coefficient)
        temp_flat = np.asarray(temp, dtype=np.float64).reshape(-1)
        q_flat = np.asarray(input_q, dtype=np.float64).reshape(-1)
        forcing = temp_flat[index]
        forcing *= self._capacitance_rate
        forcing += q_flat[index]
        forcing += ambient_temperature * (
            self._dirichlet_conductance + exposure
        )
        forcing += self._mass_coupling * self._mass_temp

        solved = factorization.solve(forcing)
        if air_side_conductance is not None:
            solved = self._add_air_side(
                factorization, solved, air_side_conductance, supply_temperature
            )
        self._mass_temp *= self._mass_retained
        np.multiply(self._mass_gained, solved, out=scratch)
        self._mass_temp += scratch

        updated = np.full(
            self._shape[0] * self._shape[1], ambient_temperature
        )
        updated[index] = solved
        return updated.reshape(self._shape)

    _mark(solve, "solver_solve")
    solve.__fastpatch_orig__ = original
    cls.solve = solve
    return ["solver_solve"]


# --------------------------------------------------------------------------- #
# (f) thermal power: flat index assignment
# --------------------------------------------------------------------------- #
def _patch_thermal_power(building_mod) -> list[str]:
    cls = building_mod.FloorPlanBasedBuilding
    current = cls.apply_thermal_power_zone
    if _patched(current):
        return []
    original = _plain(current)

    def apply_thermal_power_zone(self, zone_name, power):
        input_q = self.input_q
        view = getattr(self, "_fastpatch_input_q", None)
        if view is None or view[0] is not input_q:
            usable = (
                input_q.flags.c_contiguous
                and input_q.shape == self.temp.shape
            )
            view = (input_q, input_q.ravel() if usable else None)
            self._fastpatch_input_q = view
        flat_q = view[1]
        if flat_q is None:
            return original(self, zone_name, power)
        ncols = self.temp.shape[1]
        flat = getattr(self, "_fastpatch_flat_indices", None)
        if flat is None:
            flat = {}
            self._fastpatch_flat_indices = flat
        indices = flat.get(zone_name)
        if indices is None:
            rows, columns = self._zone_indices(zone_name)
            indices = rows.astype(np.int64) * ncols + columns.astype(np.int64)
            flat[zone_name] = indices
        if indices.size == 0:
            return None
        flat_q[indices] = power / float(indices.size)
        return None

    _mark(apply_thermal_power_zone, "thermal_power_flat")
    apply_thermal_power_zone.__fastpatch_orig__ = original
    cls.apply_thermal_power_zone = apply_thermal_power_zone
    return ["thermal_power_flat"]


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def apply(
    sim_module_root=None,
    *,
    dense_air_side: bool = True,
    exact_zone_means: bool = True,
    solver_solve: bool = True,
    shared_memos: bool = True,
) -> tuple[str, ...]:
    """Installs every patch on the simulator already imported by ``use_sim``.

    Args:
      sim_module_root: the tree passed to ``sbx.simload.use_sim``. Checked
        against ``smart_buildings.__file__`` so a patch can never be applied to
        a different copy than the one the run uses; ``None`` skips the check.
      dense_air_side: keep the n x k dense Woodbury response (see module
        docstring). ``False`` restores the per-step triangular solve.
      exact_zone_means: compute each zone mean with ``np.mean`` on that zone's
        gathered values (bit-identical to the shipped code) instead of a
        single sequential segment sum (1 ms/step cheaper, ~1e-13 K different).

    Returns:
      The names of the patches installed by this call; an already-patched
      module yields an empty tuple.
    """
    import smart_buildings  # noqa: PLC0415

    package = Path(smart_buildings.__file__).resolve()
    root = Path(sim_module_root).resolve() if sim_module_root is not None else None
    if root is not None and not _under(package, root):
        raise ValueError(
            f"fastpatch.apply: smart_buildings was imported from {package},"
            f" which is not inside {root}; call sbx.simload.use_sim() first."
        )

    from smart_buildings.smart_control.simulator import (  # noqa: PLC0415
        air_handler as air_handler_mod,
        building as building_mod,
        constants as constants_mod,
        finite_difference_solver as solver_mod,
        randomized_arrival_departure_occupancy as occupancy_mod,
        setpoint_schedule as schedule_mod,
        simulator_flexible_floor_plan as flex_mod,
    )

    names: list[str] = []
    names += _patch_air_side(solver_mod, dense_air_side)
    names += _patch_zone_averages(building_mod, constants_mod, exact_zone_means)
    names += _patch_schedules(schedule_mod, shared_memos)
    names += _patch_occupancy_timezone(occupancy_mod, shared_memos)
    try:
        from smart_buildings.smart_control.simulator import (  # noqa: PLC0415
            stochastic_occupancy as stochastic_mod,
        )
    except ImportError:
        pass
    else:
        names += _patch_occupancy_timezone(stochastic_mod, shared_memos)
    names += _patch_reward_infos(flex_mod, air_handler_mod)
    if solver_solve:
        names += _patch_solver_solve(solver_mod)
    names += _patch_thermal_power(building_mod)

    _INSTALLED.clear()
    _INSTALLED.extend(names)
    return tuple(names)
