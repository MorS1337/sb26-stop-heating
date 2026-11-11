"""Parallel scenario runner for the Smart Buildings simulator.

Runs one policy over a list of scenarios, one operating-system process per
scenario, and writes a JSON record per scenario plus a summary with the
leaderboard score.

    python -m sbx.evalrun --policy baseline:pid_water --set dev30 \
        --steps 8640 --sim shipped --workers 6 --tag pid_water_shipped

    python -m sbx.evalrun --policy policies.ctrl_v1:CtrlV1 --set dev30_fairbanks \
        --params '{"kp": 0.4}' --sim corrected --tag ctrl_v1_fairbanks

The policy is specified as

* ``baseline:<key>``   -- one of the starter-kit reference policies
                          (``pid_water``, ``pid``, ``bang_bang``, ``fixed_setpoint``,
                          ``all_off``), imported from the *selected* simulator copy;
* ``module:Class``     -- anything importable from the workspace, e.g.
                          ``policies.ctrl_v1:CtrlV1``;
* ``<key>``            -- a bare baseline key is accepted too.

Scenarios come from :mod:`sbx.scenarios` (``--set dev30``, ``--set dev30_fairbanks``,
``--set gen_train``, ...), from explicit ids (``--scenario building_train_1/atlanta_2``)
or from ``config.gin`` paths / scenario directories (``--configs <path> ...``).

Parity with ``smart_buildings.evaluate``
---------------------------------------
The environment is built exactly like ``evaluate.load_hybrid_environment`` and
the metadata via the same ``evaluate.create_building_meta_data`` the Kaggle
gateway calls, with the same two attribute overrides ``evaluate`` applies
(``env.metrics_path = None``, ``env._occupancy_normalization_constant = 125.0``).
No ``SimulationTracker`` is constructed.

``evaluate.py``'s tracker asks the environment for the reward a *second* time
each step (``env.get_reward_info_and_response()``), after the step's own reward
computation. In the shipped simulator every boiler sensor read advances the tank
(``Boiler._set_current_temperature``), so that second call is not numerically
neutral: it changes the trajectory. ``--components parity`` (the default)
therefore makes the same second call, and reproduces ``evaluate.py`` totals
bit-for-bit. ``--components hook`` captures the response from the step's own
reward computation instead, which is what the Kaggle gateway does; measured
totals are identical and it saves ~7% CPU (0.5 s of 7.6 s CPU for a 288-step
``pid_water`` episode).

Per-scenario output (``results/<tag>/<building>__<weather>.json``)
-----------------------------------------------------------------
``total_reward``, ``mean_per_step_reward``, ``steps``, ``wall_seconds``,
``cpu_seconds``, and per-step-aggregated reward components -- comfort
(``w_p * normalized_productivity_regret``), energy (``-w_e * normalized_energy_cost``),
carbon (``-w_c * normalized_carbon_emission``) -- summed overall, split by
``weekday_occupied`` / ``weekday_unoccupied`` / ``weekend``, and binned by local
hour of day. Plus the coldest zone temperature seen during weekday occupied
hours and the number of steps with any occupied zone more than
``productivity_deviation_ref`` (5 K) out of its comfort band.

``results/<tag>/summary.json`` carries the leaderboard score -- the mean over
scenarios of ``|mean per-step reward|``, lower is better -- plus per-weather and
per-building means.

``--fast``
----------
``--fast`` installs :mod:`sbx.fastpatch` in each worker right after
``sbx.simload.use_sim``: the same simulator, monkeypatched at runtime (dense
air-side Woodbury response instead of a second triangular solve per step, a
per-step memo for zone-average temperatures, timestamp memos for the setpoint
schedule and occupancy timezone conversions, hoisted reward-info loops, flat
index assignment for zone thermal power). No file under ``sims/`` or
``starter-kit/`` is touched and the official scoring path never imports it.
Results are unchanged: over 8641-step CtrlW episodes on both trees the total
reward matches to every digit, and the largest final CV temperature difference
is 2.8e-13 K. ``--fast`` is off by default, and a run without it is unchanged.
Measured per-step CPU with ``policies.ctrl_w:CtrlW`` (single process, median of
5 alternating runs): building_train_2/atlanta_3 42.4 -> 24.6 ms (1.72x, the
largest shipped building), building_train_1/atlanta_2 25.1 -> 13.0 ms (1.93x),
building_train_8/fairbanks_8 13.2 -> 7.8 ms (1.69x). See ``sbx/fastpatch.py``
for what limits it -- one sparse solve per step is irreducible, and the
remainder is numba convection, occupancy chains and protobuf, which ``--cenv``
attacks.

``--cenv``
----------
``--cenv bookkeeping,observation`` installs the C components from ``cenv/`` in
each worker, right after ``--fast`` and before the environment is built, by
monkeypatching the simulator classes (no file under ``sims/`` or
``starter-kit/`` is edited, so official scoring is unaffected). Each component
reproduces every number the Python original produces -- the shadow gate
(``python -m cenv.verify.shadow``) compares both implementations on the same
inputs inside a live episode, and ``cenv/verify/gate.py`` compares whole
episodes step by step. ``all`` enables every component whose gate has passed in
this checkout; naming one opts in explicitly. Off by default and additive to
``--fast``/``--oracle``.

``--oracle``
------------
``--oracle occupancy`` / ``forecast`` / ``occupancy,forecast`` is an
analysis-only channel: before each ``get_action`` a policy that defines
``observe_oracle(info)`` is handed a dict with ``zone_occupancy`` (per-zone
average occupancy of the last completed interval, i.e. a 5-minute-late zonal
sensor, read from the reward info the step already computed) and
``forecast_oat`` (outside air temperature in K for the next 1..``--oracle-horizon``
hours, from the episode's own weather controller). ``--oracle foresight
--oracle-file <dir>`` instead plays back a per-zone occupancy schedule recorded
with ``--oracle-record <dir>`` (one sparse JSON per scenario, written from the
same reward infos), so a policy can be handed the realised future of the
episode. Neither source re-queries the occupancy model or touches the clock, so
a policy that ignores the payload reproduces a normal run bit-for-bit
(``policies/oracle_noop.py``). Variants built on it are named ``oracle_*`` and
are never submitted.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

# One BLAS thread per process, set before numpy loads (spawned workers re-import
# this module). A step is a sparse LU solve plus small dense ops, so threaded
# OpenBLAS only oversubscribes the cores when scenarios run in parallel: measured
# 566 CPU-s for a 70 s single-worker run on a 12-thread Ryzen.
for _blas_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from sbx import scenarios as scenario_sets  # noqa: E402
from sbx import simload  # noqa: E402

BUCKETS = ("weekday_occupied", "weekday_unoccupied", "weekend")
HOURS = 24
COMPONENT_SUM_KEYS = ("reward", "comfort", "energy", "carbon")
DEFAULT_STEPS = 8640
DEFAULT_WORKERS = 6
# Analysis-only channels (see _OracleChannel): per-zone occupancy of the last
# completed interval, and the outside-air-temperature forecast.
ORACLE_SOURCES = ("occupancy", "forecast", "foresight")
DEFAULT_ORACLE_HORIZON = 12


# --------------------------------------------------------------------------- #
# per-scenario recording
# --------------------------------------------------------------------------- #
class _StepRecorder:
    """Per-step reward components and comfort statistics for one episode.

    ``mode='parity'`` asks the environment for the reward info after the step,
    exactly like ``evaluate.py``'s tracker does (see the module docstring).
    ``mode='hook'`` wraps ``reward_function.compute_reward`` and captures the
    response from the step's own reward computation, which costs nothing extra.
    Both feed :meth:`_capture`, so the aggregate is identical in shape.
    """

    def __init__(self, env, meta: dict[str, Any], mode: str = "parity"):
        if mode not in ("parity", "hook"):
            raise ValueError(f"unknown components mode {mode!r}")
        self.env = env
        self.mode = mode
        self.morning = int(meta["morning_start_hour"])
        self.evening = int(meta["evening_start_hour"])
        self.weights: dict[str, float] | None = None
        self.steps: list[dict[str, float]] = []
        self._step_index = 0
        # Kept for the analysis-only oracle channel: the RewardInfo of the most
        # recent step. Storing the reference neither copies nor re-queries the
        # occupancy model, so a run that reads it stays bit-identical.
        self.last_info = None
        self.last_timestamp: str | None = None
        self.first_local_timestamp: str | None = None
        self.last_local_timestamp: str | None = None
        if mode == "hook":
            fn = env.reward_function
            original = fn.compute_reward

            def compute_reward(reward_info):
                response = original(reward_info)
                self._capture(reward_info, response)
                return response

            fn.compute_reward = compute_reward

    # -- capture ----------------------------------------------------------- #
    def _capture(self, info, response) -> None:
        # Naive simulation time is already local wall-clock (tz_localize only
        # attaches a zone, and raises on DST gaps/overlaps), so use it as is;
        # convert only if the simulator ever hands out a tz-aware stamp.
        ts = self.env.current_simulation_timestamp
        if ts.tz is not None:
            ts = ts.tz_convert(self.env.time_zone)
        stamp = str(ts)
        if self.first_local_timestamp is None:
            self.first_local_timestamp = stamp
        self.last_local_timestamp = stamp
        self.last_info = info
        self.last_timestamp = stamp
        self.weights = {
            "productivity": float(response.productivity_weight),
            "energy": float(response.energy_cost_weight),
            "carbon": float(response.carbon_emission_weight),
        }

        t_min_all = np.inf
        t_min_occupied = np.inf
        worst_deviation = 0.0
        for zone in info.zone_reward_infos.values():
            temp = float(zone.zone_air_temperature)
            if temp < t_min_all:
                t_min_all = temp
            deviation = max(
                0.0,
                temp - float(zone.cooling_setpoint_temperature),
                float(zone.heating_setpoint_temperature) - temp,
            )
            if deviation > worst_deviation:
                worst_deviation = deviation
            if float(zone.average_occupancy) > 0.0 and temp < t_min_occupied:
                t_min_occupied = temp

        self.steps.append(
            {
                "hour": float(ts.hour),
                "weekday": float(ts.weekday()),
                "npr": float(response.normalized_productivity_regret),
                "ec": float(response.normalized_energy_cost),
                "ce": float(response.normalized_carbon_emission),
                "occupancy": float(response.total_occupancy),
                "dmax": worst_deviation,
                "tmin_all": float(t_min_all),
                "tmin_occupied": float(t_min_occupied),
            }
        )

    def bucket_of(self, step: dict[str, float]) -> str:
        if step["weekday"] >= 5.0:
            return "weekend"
        hour = int(step["hour"])
        if self.morning < self.evening:
            in_window = self.morning <= hour < self.evening
        else:  # overnight window
            in_window = hour >= self.morning or hour < self.evening
        return "weekday_occupied" if in_window else "weekday_unoccupied"

    # -- per-step entry point ---------------------------------------------- #
    def record(self, reward: float) -> None:
        if self.mode == "parity":
            info, response = self.env.get_reward_info_and_response()
            self._capture(info, response)
        if len(self.steps) <= self._step_index:
            raise RuntimeError("reward response was never captured for this step")
        # A step must add exactly one capture. If the reward function ran more
        # than once, keep the last response and drop the earlier ones.
        while len(self.steps) > self._step_index + 1:
            self.steps.pop(self._step_index)
        self.steps[-1]["reward"] = float(reward)
        self._step_index += 1

    # -- aggregate --------------------------------------------------------- #
    def aggregate(self) -> dict[str, Any]:
        if not self.steps:
            raise RuntimeError("no steps recorded")
        weights = self.weights or {"productivity": 0.6, "energy": 0.2, "carbon": 0.2}
        steps = self.steps
        comfort = np.array([weights["productivity"] * s["npr"] for s in steps])
        energy = np.array([-weights["energy"] * s["ec"] for s in steps])
        carbon = np.array([-weights["carbon"] * s["ce"] for s in steps])
        rewards = np.array([s["reward"] for s in steps])

        buckets: dict[str, Any] = {
            name: {
                "steps": 0,
                **{f"sum_{k}": 0.0 for k in COMPONENT_SUM_KEYS},
            }
            for name in BUCKETS
        }
        per_hour = {
            "steps": [0] * HOURS,
            **{f"sum_{k}": [0.0] * HOURS for k in COMPONENT_SUM_KEYS},
        }
        bracket = np.array([self.bucket_of(s) for s in steps], dtype=object)
        hours = np.array([int(s["hour"]) for s in steps], dtype=int)

        for i, step in enumerate(steps):
            slot = buckets[self.bucket_of(step)]
            slot["steps"] += 1
            slot["sum_reward"] += float(rewards[i])
            slot["sum_comfort"] += float(comfort[i])
            slot["sum_energy"] += float(energy[i])
            slot["sum_carbon"] += float(carbon[i])
            hour = hours[i]
            per_hour["steps"][hour] += 1
            per_hour["sum_reward"][hour] += float(rewards[i])
            per_hour["sum_comfort"][hour] += float(comfort[i])
            per_hour["sum_energy"][hour] += float(energy[i])
            per_hour["sum_carbon"][hour] += float(carbon[i])

        occupied = bracket == "weekday_occupied"
        tmin_all = np.array([s["tmin_all"] for s in steps])
        tmin_occ = np.array([s["tmin_occupied"] for s in steps])
        dmax = np.array([s["dmax"] for s in steps])
        finite_occ = np.isfinite(tmin_occ)

        n_steps = len(steps)
        return {
            "components": {
                "weights": weights,
                "sum_reward_from_components": float(
                    comfort.sum() + energy.sum() + carbon.sum()
                ),
                "sum_comfort": float(comfort.sum()),
                "sum_energy": float(energy.sum()),
                "sum_carbon": float(carbon.sum()),
                "per_step_comfort": float(comfort.mean()),
                "per_step_energy": float(energy.mean()),
                "per_step_carbon": float(carbon.mean()),
            },
            "by_bucket": buckets,
            "by_hour": per_hour,
            "comfort": {
                "steps": n_steps,
                "mean_occupancy": float(np.mean([s["occupancy"] for s in steps])),
                "sum_occupancy": float(np.sum([s["occupancy"] for s in steps])),
                "min_zone_temp_occupied_hours_all_zones": (
                    float(tmin_all[occupied].min()) if occupied.any() else None
                ),
                "min_zone_temp_occupied_hours_occupied_zones": (
                    float(tmin_occ[occupied & finite_occ].min())
                    if (occupied & finite_occ).any()
                    else None
                ),
                "steps_any_occupied_zone_over_5K_out_of_band": int((dmax > 5.0).sum()),
                # Every step, occupied or not: the frost exposure of the episode.
                "min_zone_temp_all_steps": float(tmin_all.min()),
                "max_zone_deviation_K": float(dmax.max()),
                "n_steps_weekday_occupied": int(occupied.sum()),
            },
            "first_local_timestamp": self.first_local_timestamp,
            "last_local_timestamp": self.last_local_timestamp,
        }


# --------------------------------------------------------------------------- #
# analysis-only oracle channel
# --------------------------------------------------------------------------- #
def parse_oracle(value: Any) -> tuple[str, ...]:
    """``None`` | ``"occupancy"`` | ``"occupancy,forecast"`` | sequence -> tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    else:
        items = [str(part).strip() for part in value]
    items = [item for item in items if item]
    unknown = sorted(set(items) - set(ORACLE_SOURCES))
    if unknown:
        raise ValueError(
            f"unknown oracle source(s) {unknown}; choose from {list(ORACLE_SOURCES)}"
        )
    return tuple(dict.fromkeys(items))


def _zone_sort_key(name: str) -> tuple:
    """Natural sort key, identical to the one ``policies/wu_core.py`` uses."""
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(name)))


def _tail_number(name: str) -> int | None:
    match = re.search(r"(\d+)$", str(name))
    return int(match.group(1)) if match else None


def _weather_controller(env):
    """The episode's own weather controller, or ``None`` if it is unreachable.

    Reusing the simulator's controller is what makes the forecast exactly
    aligned with the clock: the same ``get_current_temp`` that produced the
    ambient for step ``t`` answers for ``t + h``. The call is a pure
    interpolation over an immutable CSV frame -- no RNG, no state -- so it
    cannot perturb the trajectory.
    """
    for path in (
        ("building", "simulator", "_weather_controller"),
        ("building", "_weather_controller"),
        ("_weather_controller",),
    ):
        node = env
        for attribute in path:
            node = getattr(node, attribute, None)
            if node is None:
                break
        if node is not None and hasattr(node, "get_current_temp"):
            return node
    return None


class _OracleChannel:
    """Analysis-only information a policy may ask for through ``observe_oracle``.

    Two sources, both unavailable to a submission:

    ``occupancy``
        Per-zone occupancy of the **last completed** 5-minute interval, taken
        from the ``RewardInfo`` the step's own reward computation already built
        (``zone_reward_infos[*].average_occupancy``). Reading it adds no query
        to the occupancy model; a second query would advance the shared
        ``RandomState`` and shift every later arrival (forum digest B3).

    ``forecast``
        Outside air temperature in K for the next 1..``horizon`` hours, read
        from the episode's own ``ReplayWeatherController`` at the current
        simulated clock.

    The payload is ordered like the policy's own zone list (``policy.zone_ids``
    when present, else the natural sort of the zone actuator ids), mapped from
    the reward info's ``room_<n>`` keys by their numeric suffix -- the same
    ``room_x -> VAV_x`` correspondence ``wu_core`` relies on.
    """

    def __init__(
        self,
        env,
        meta: dict[str, Any],
        policy,
        sources,
        horizon: int,
        scenario: str | None = None,
        oracle_file: str | None = None,
        record_dir: str | None = None,
    ):
        self.sources = parse_oracle(sources)
        self.horizon = max(1, int(horizon))
        self.policy_zones = self._policy_zone_names(meta, policy)
        self.zone_keys: list[str] | None = None
        self.weather = _weather_controller(env) if "forecast" in self.sources else None
        self.last_forecast: np.ndarray | None = None
        self.last_zone_occupancy: np.ndarray | None = None
        self.scenario = scenario
        self.step = 0
        self.record_dir = record_dir
        self.recorded: list[list[int]] = []
        self.foresight: np.ndarray | None = None
        if "foresight" in self.sources:
            self.foresight = self._load_foresight(oracle_file)

    @staticmethod
    def _policy_zone_names(meta: dict[str, Any], policy) -> list[str]:
        names = getattr(policy, "zone_ids", None)
        if names is None:
            names = list(meta["zone_action_indices"])
            names = sorted(names, key=_zone_sort_key)
        else:
            names = [str(name) for name in names]
        if not names:
            raise ValueError("policy exposes no zone ids for the oracle mapping")
        return names

    def _bind_zone_keys(self, info) -> list[str]:
        """Maps the policy's zone order onto the reward info's zone keys."""
        info_keys = [str(key) for key in info.zone_reward_infos]
        by_tail: dict[int, str] = {}
        for key in info_keys:
            tail = _tail_number(key)
            if tail is not None:
                by_tail.setdefault(tail, key)
        keys: list[str | None] = []
        for name in self.policy_zones:
            tail = _tail_number(name)
            keys.append(by_tail.get(tail) if tail is not None else None)
        missing = [n for n, k in zip(self.policy_zones, keys) if k is None]
        if missing:
            raise ValueError(
                "oracle occupancy: no reward-info zone matches policy zone(s) "
                f"{missing[:5]} (reward-info keys start {info_keys[:5]})"
            )
        resolved = [str(key) for key in keys]
        self.zone_keys = resolved
        return resolved

    def _load_foresight(self, oracle_file: str | None) -> np.ndarray:
        """Dense (steps x zones) occupancy of a reference run, zone-ordered."""
        if not oracle_file:
            raise ValueError("--oracle foresight needs --oracle-file <dir|file>")
        path = Path(oracle_file)
        if path.is_dir():
            if self.scenario is None:
                raise ValueError("--oracle foresight needs the scenario id")
            path = path / (self.scenario.replace("/", "__") + ".json")
        if not path.exists():
            raise FileNotFoundError(f"foresight schedule {path} does not exist")
        blob = json.loads(path.read_text())
        names = [str(n) for n in blob["zones"]]
        by_tail: dict[int, int] = {}
        for column, name in enumerate(names):
            tail = _tail_number(name)
            if tail is not None:
                by_tail.setdefault(tail, column)
        columns = []
        for name in self.policy_zones:
            tail = _tail_number(name)
            column = by_tail.get(tail) if tail is not None else None
            if column is None:
                raise ValueError(
                    f"foresight schedule has no zone matching policy zone {name!r}"
                )
            columns.append(column)
        n_steps = int(blob["n_steps"])
        dense = np.zeros((n_steps, self.n_policy_zones()), dtype=float)
        for step, listed in enumerate(blob["occupied"]):
            for entry in listed:
                if isinstance(entry, (list, tuple)):
                    zone, value = int(entry[0]), float(entry[1])
                else:
                    zone, value = int(entry), 1.0
                dense[step, zone] = value
        return dense[:, columns]

    def n_policy_zones(self) -> int:
        return len(self.policy_zones)

    def finish(self) -> None:
        """Writes the recorded per-zone occupancy (analysis-only)."""
        if not self.record_dir:
            return
        out = Path(self.record_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = (self.scenario or "scenario").replace("/", "__")
        (out / (name + ".json")).write_text(
            json.dumps(
                {
                    "scenario": self.scenario,
                    "zones": list(self.policy_zones),
                    "n_steps": len(self.recorded),
                    "occupied": self.recorded,
                }
            )
        )

    def _forecast(self, timestamp) -> np.ndarray:
        out = np.full(self.horizon, np.nan)
        weather = self.weather
        if weather is None:
            return out
        import datetime  # local: only the oracle path needs it

        # The simulator clock is naive local wall clock and so is the CSV
        # reading (the controller localizes naive stamps to UTC, which per the
        # forum's B6 answer attaches the zone without shifting the hour). The
        # frame's own bounds are tz-aware, so compare wall clock to wall clock.
        def naive(value):
            tz = getattr(value, "tz", None)
            return value.tz_localize(None) if tz is not None else value

        max_time = getattr(weather, "max_time", None)
        min_time = getattr(weather, "min_time", None)
        max_time = naive(max_time) if max_time is not None else None
        min_time = naive(min_time) if min_time is not None else None
        for hour in range(1, self.horizon + 1):
            moment = timestamp + datetime.timedelta(hours=hour)
            # The CSV ends before the last simulated day of some episodes: hold
            # the final reading rather than raising out of the environment.
            if max_time is not None and moment > max_time:
                moment = max_time
            elif min_time is not None and moment < min_time:
                moment = min_time
            out[hour - 1] = float(weather.get_current_temp(moment))
        return out

    def info(self, last_info, timestamp) -> dict[str, Any]:
        """The payload for one step; ``last_info`` is the previous step's."""
        payload: dict[str, Any] = {
            "sources": self.sources,
            "timestamp": str(timestamp),
            "zone_occupancy": None,
            "forecast_oat": None,
        }
        if "occupancy" in self.sources and last_info is not None:
            keys = self.zone_keys
            if keys is None:
                keys = self._bind_zone_keys(last_info)
            occupancy = np.array(
                [
                    float(last_info.zone_reward_infos[key].average_occupancy)
                    if key in last_info.zone_reward_infos
                    else np.nan
                    for key in keys
                ],
                dtype=float,
            )
            self.last_zone_occupancy = occupancy
            payload["zone_occupancy"] = occupancy
        if "forecast" in self.sources:
            self.last_forecast = self._forecast(timestamp)
            payload["forecast_oat"] = self.last_forecast
        if "foresight" in self.sources:
            payload["foresight_occupancy"] = self.foresight
            payload["foresight_step"] = self.step
        if self.record_dir:
            # Row index == step index, so a playback needs no offset. Step 0 has
            # no completed interval yet, hence an empty row.
            occupancy = self.last_zone_occupancy
            self.recorded.append(
                []
                if occupancy is None
                else [
                    int(i) for i in np.flatnonzero(np.nan_to_num(occupancy) > 0.0)
                ]
            )
        self.step += 1
        return payload



def _resolve_policy(spec: str, evaluate_module):
    """``baseline:<key>`` | ``module:Class`` | bare baseline key -> (class, label)."""
    if spec.startswith("baseline:"):
        key = spec.split(":", 1)[1]
        if key not in evaluate_module.POLICY_MAP:
            raise KeyError(
                f"unknown baseline {key!r}; available: {sorted(evaluate_module.POLICY_MAP)}"
            )
        return evaluate_module.POLICY_MAP[key], f"baseline:{key}"
    if ":" in spec:
        module_name, class_name = spec.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name), spec
    if spec in evaluate_module.POLICY_MAP:
        return evaluate_module.POLICY_MAP[spec], f"baseline:{spec}"
    raise ValueError(
        f"cannot resolve policy {spec!r}: use 'baseline:<key>', 'module:Class', "
        f"or one of {sorted(evaluate_module.POLICY_MAP)}"
    )


def _normalise_cenv(value: Any) -> list[str]:
    """``--cenv bookkeeping,thermal`` | ``all`` | ``None`` -> component list.

    Names are validated here, in the parent, so a typo fails before any worker
    starts. ``all`` is expanded inside the worker (it needs the tree's
    gate-passed flags), so it is carried through as the single name ``all``.
    """
    if not value:
        return []
    items = [item.strip() for item in str(value).replace(";", ",").split(",")]
    items = [item for item in items if item and item != "none"]
    if not items:
        return []
    from cenv import COMPONENTS

    unknown = [item for item in items if item != "all" and item not in COMPONENTS]
    if unknown:
        raise ValueError(
            f"unknown cenv component(s) {unknown}; known: {list(COMPONENTS)} or 'all'"
        )
    return list(dict.fromkeys(items))


def _load_environment(evaluate_module, config_path: Path, bindings: dict[str, Any]):
    """``evaluate.load_hybrid_environment`` plus gin parameter overrides.

    With no bindings this *is* that function. With bindings the same three steps
    are performed by hand -- clear, parse, construct -- because gin only allows
    a binding to be set between parsing and construction, and no ``config.gin``
    exposes a hook for that.

    ``RandomizedArrivalDepartureOccupancy.seed`` is the only key used today: the
    occupancy model's constructor default is 17321 and no generated config binds
    it, so an unscoped binding reaches every scenario. Changing it changes which
    zones are occupied when, which the reward weights by, so it is a legitimate
    robustness knob -- and a hardcoding temptation the competition bans.
    """
    if not bindings:
        return evaluate_module.load_hybrid_environment(config_path)

    import gin  # imported from the selected sim tree via sys.path
    from smart_buildings.smart_control.environment import (
        hybrid_action_environment,
    )

    gin.clear_config()
    with gin.unlock_config():
        gin.parse_config_file(str(config_path))
        for key, value in bindings.items():
            gin.bind_parameter(key, value)
    return hybrid_action_environment.HybridActionEnvironment()


def _shifted_config(config_path: Path, offset_days: int, out_dir: Path) -> Path:
    """Rewrites ``start_timestamp`` in a config.gin by ``offset_days`` days.

    The generated configs pin one start date each, and the same strip cannot be
    reused at a different start otherwise. Only that one line is changed; the
    data paths inside stay relative to the tree root, so the copy can live
    anywhere.
    """
    import pandas as pd  # only needed for this path

    text = config_path.read_text()
    match = re.search(r"^start_timestamp\s*=\s*'([^']*)'", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"{config_path} has no start_timestamp line to shift")
    start = pd.Timestamp(match.group(1)) + pd.Timedelta(days=int(offset_days))
    replacement = f"start_timestamp = '{start.strftime('%Y-%m-%d %H:%M')}'"
    text = text[: match.start()] + replacement + text[match.end() :]
    dest_dir = out_dir / "_gin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (
        f"{config_path.parent.parent.name}__{config_path.parent.name}"
        f"__off{int(offset_days)}.gin"
    )
    # Concurrent workers may shift the same scenario: write a private temp file
    # and rename it into place so no reader ever sees a partially written gin.
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, dest)
    return dest


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def _run_scenario(task: dict[str, Any]) -> dict[str, Any]:
    """Runs one scenario in the current process and writes its JSON record."""
    started_wall = time.time()
    started_cpu = time.process_time()

    simload.use_sim(task["sim"])
    from smart_buildings import evaluate  # noqa: PLC0415

    # --fast: pure-Python runtime patches from sbx/fastpatch.py, installed
    # before anything builds an environment or a recorder (and before --cenv,
    # whose C replacements then wrap or replace them). Off by default: with no
    # flag this module is never imported and the run is unchanged.
    fast_report = None
    if task.get("fast"):
        from sbx import fastpatch  # noqa: PLC0415

        fast_report = fastpatch.apply(
            simload.sim_root(task["sim"]),
            dense_air_side=task.get("fast_dense_air_side", True),
            exact_zone_means=task.get("fast_exact_zone_means", True),
        )

    # --cenv: install the C components before anything builds an environment or
    # a recorder, so class-level patches are what the env (and the hook-mode
    # recorder wrapper) binds to. Additive to --fast/--oracle: nothing here
    # touches another component's patch points.
    cenv_report = None
    if task.get("cenv"):
        from cenv import install as _cenv_install  # noqa: PLC0415

        cenv_report = _cenv_install(
            task["cenv"],
            sim_root=simload.sim_root(task["sim"]),
            verbose=False,
        )

    policy_cls, policy_label = _resolve_policy(task["policy"], evaluate)

    config_path = Path(task["config_path"])
    scenario_dir = config_path.parent
    if task["start_offset_days"]:
        config_path = _shifted_config(
            config_path, task["start_offset_days"], Path(task["out_dir"])
        )

    bindings: dict[str, Any] = {}
    if task.get("occupancy_seed") is not None:
        bindings["RandomizedArrivalDepartureOccupancy.seed"] = int(
            task["occupancy_seed"]
        )

    env = _load_environment(evaluate, config_path, bindings)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0

    meta = evaluate.create_building_meta_data(env, scenario_dir)
    action_spec = env.action_spec()

    if task["params"]:
        policy_cls.PARAMS = {
            **(getattr(policy_cls, "PARAMS", None) or {}),
            **task["params"],
        }

    policy = evaluate.initialize_policy(policy_cls, meta, action_spec)
    recorder = _StepRecorder(env, meta, mode=task["components"])

    oracle = None
    observe_oracle = getattr(policy, "observe_oracle", None)
    oracle_sources = parse_oracle(task.get("oracle"))
    if oracle_sources:
        oracle = _OracleChannel(
            env,
            meta,
            policy,
            oracle_sources,
            int(task.get("oracle_horizon") or DEFAULT_ORACLE_HORIZON),
            scenario=task["scenario"],
            oracle_file=task.get("oracle_file"),
            record_dir=task.get("oracle_record"),
        )
        if not callable(observe_oracle):
            print(
                f"[evalrun] {task['scenario']}: --oracle {','.join(oracle_sources)} "
                f"given but {type(policy).__name__} defines no observe_oracle; "
                "the sources are enabled but unused",
                flush=True,
            )

    obs, _ = env.reset()

    step_cpu = 0.0
    policy_cpu = 0.0
    rewards: list[float] = []
    for _ in range(int(task["steps"])):
        if oracle is not None:
            # info() is also what feeds --oracle-record, so it runs even for a
            # policy that has no observe_oracle (the payload is then dropped).
            oracle_info = oracle.info(
                recorder.last_info, env.current_simulation_timestamp
            )
            if callable(observe_oracle):
                observe_oracle(oracle_info)
        t0 = time.process_time()
        action = evaluate.validate_action(
            evaluate.get_policy_action(policy, obs), action_spec
        )
        policy_cpu += time.process_time() - t0
        t0 = time.process_time()
        obs, reward, terminated, truncated, _ = env.step(action)
        step_cpu += time.process_time() - t0
        recorder.record(float(reward))
        rewards.append(float(reward))
        if terminated or truncated:
            break

    if oracle is not None:
        oracle.finish()

    aggregates = recorder.aggregate()
    reward_array = np.array(rewards, dtype=np.float64)
    wall = time.time() - started_wall
    cpu = time.process_time() - started_cpu

    record = {
        "scenario": task["scenario"],
        "building": task["scenario"].rsplit("/", 1)[0],
        "weather": task["scenario"].rsplit("/", 1)[-1],
        "policy": policy_label,
        "params": task["params"] or {},
        "sim": task["sim"],
        "cenv": None if not task.get("cenv") else str(task["cenv"]),
        "fast": list(fast_report or ()),
        "tag": task["tag"],
        "steps": int(len(rewards)),
        "requested_steps": int(task["steps"]),
        "start_offset_days": int(task["start_offset_days"]),
        "occupancy_seed": task.get("occupancy_seed"),
        "oracle": list(oracle_sources),
        "oracle_horizon": (
            int(task.get("oracle_horizon") or DEFAULT_ORACLE_HORIZON)
            if oracle_sources
            else None
        ),
        "total_reward": float(np.sum(reward_array)),
        "mean_per_step_reward": float(np.mean(reward_array)),
        "wall_seconds": float(wall),
        "cpu_seconds": float(cpu),
        "step_cpu_seconds": float(step_cpu),
        "policy_cpu_seconds": float(policy_cpu),
        "steps_per_second_cpu": float(len(rewards) / cpu) if cpu > 0 else None,
        "steps_per_second_step_cpu": (
            float(len(rewards) / step_cpu) if step_cpu > 0 else None
        ),
        "num_ahus": int(meta["n_discrete_action"]) - 1,
        "num_zones": int(len(meta["zone_action_indices"])),
        "config_path": str(Path(task["config_path"]).resolve()),
        **aggregates,
    }

    out_path = Path(task["out_dir"]) / task["record_name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=False))
    record["record_path"] = str(out_path)
    return record


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    try:
        return _run_scenario(task)
    except Exception:  # noqa: BLE001 - reported, not swallowed
        return {
            "scenario": task["scenario"],
            "sim": task["sim"],
            "config_path": task["config_path"],
            "record_name": task["record_name"],
            "error": traceback.format_exc(),
        }


# --------------------------------------------------------------------------- #
# parent side
# --------------------------------------------------------------------------- #
def _auto_tag(policy: str) -> str:
    slug = policy.split(":", 1)[-1].replace(".", "_")
    return f"{slug}_{time.strftime('%Y%m%d_%H%M%S')}"


def _expand_scenarios(items: Iterable[Any], sim: str) -> list[tuple[str, Path]]:
    paths = scenario_sets.config_paths(list(items), sim)
    pairs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for path in paths:
        scenario = scenario_sets.scenario_id(path)
        if scenario in seen:
            continue
        seen.add(scenario)
        pairs.append((scenario, path))
    return pairs


def run(
    policy: str = "policies.ctrl_v1:CtrlV1",
    params: dict[str, Any] | None = None,
    scenarios: Sequence[Any] | None = None,
    steps: int = DEFAULT_STEPS,
    sim: str = "shipped",
    workers: int = DEFAULT_WORKERS,
    out: str | Path | None = None,
    start_offset_days: int = 0,
    tag: str | None = None,
    components: str = "parity",
    occupancy_seed: int | None = None,
    oracle: str | Sequence[str] | None = None,
    oracle_horizon: int = DEFAULT_ORACLE_HORIZON,
    oracle_file: str | None = None,
    oracle_record: str | None = None,
    cenv: str | None = None,
    fast: bool = False,
) -> dict[str, Any]:
    """Runs ``policy`` over ``scenarios`` and returns the summary dict.

    Args:
      policy: ``baseline:<key>``, ``module:Class`` or a bare baseline key.
      params: optional ``PARAMS`` overrides for a tunable policy class.
      scenarios: set names (``dev30``, ``gen_train``, ...), scenario ids
        (``building/weather``), ``config.gin`` paths or scenario directories.
        Defaults to ``['dev30']``.
      steps: episode length in 5-minute steps (8640 = 30 days).
      sim: ``shipped`` or ``corrected``.
      workers: number of concurrent scenario processes (one per scenario).
      out: results directory; defaults to ``results/<tag>``.
      start_offset_days: shifts every episode's start time by this many days.
      tag: results tag, also recorded in each JSON.
      components: ``parity`` (reproduce ``evaluate.py`` exactly) or ``hook``.
      occupancy_seed: overrides ``RandomizedArrivalDepartureOccupancy.seed``
        (default 17321) via a gin binding, for robustness tests against a
        different hidden occupancy draw. ``None`` leaves the config alone.
      oracle: analysis-only information handed to the policy's
        ``observe_oracle`` each step before ``get_action`` -- ``"occupancy"``
        (per-zone occupancy of the last completed interval), ``"forecast"``
        (outside-air-temperature forecast) or ``"occupancy,forecast"``.
        ``None`` leaves the run untouched; policies that never read the payload
        reproduce a normal run bit-for-bit.
      oracle_horizon: forecast horizon in hours (default 12).
      oracle_file: path (dir or json) of a recorded per-zone occupancy
        schedule, read by the ``foresight`` source.
      oracle_record: directory to write the current run's per-zone occupancy
        schedule to (needs the ``occupancy`` source); tiny, sparse JSON.
      cenv: opt-in C components from ``cenv/`` -- a comma-separated list of
        module names (``bookkeeping``, ``observation``, ``thermal``, ``hvac``,
        ...) or ``all`` for every component whose shadow gate has passed.
        Additive to ``--fast``/``--oracle``; ``None`` runs pure Python.
      fast: install :mod:`sbx.fastpatch` before the environment is built.
        ``False`` (the default) runs the unpatched simulator, bit-for-bit.
    """
    if sim not in simload.SIM_NAMES:
        raise ValueError(f"unknown sim {sim!r}; choose one of {simload.SIM_NAMES}")
    if components not in ("parity", "hook"):
        raise ValueError(f"unknown components mode {components!r}")

    cenv_requested = _normalise_cenv(cenv)

    oracle_sources = parse_oracle(oracle)
    items = ["dev30"] if scenarios is None else list(scenarios)
    pairs = _expand_scenarios(items, sim)
    if not pairs:
        raise ValueError("no scenarios selected")

    tag = tag or _auto_tag(policy)
    out_dir = Path(out) if out is not None else WORKSPACE / "results" / tag
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "sim": sim,
            "scenario": scenario,
            "config_path": str(path.resolve()),
            "record_name": scenario.replace("/", "__") + ".json",
            "out_dir": str(out_dir),
            "policy": policy,
            "params": dict(params or {}),
            "steps": int(steps),
            "start_offset_days": int(start_offset_days),
            "occupancy_seed": (
                None if occupancy_seed is None else int(occupancy_seed)
            ),
            "components": components,
            "cenv": cenv_requested,
            "fast": bool(fast),
            "tag": tag,
            "oracle": list(oracle_sources),
            "oracle_horizon": (
                int(oracle_horizon) if oracle_sources else None
            ),
            "oracle_file": (
                str(Path(oracle_file).resolve()) if oracle_file else None
            ),
            "oracle_record": (
                str(Path(oracle_record).resolve()) if oracle_record else None
            ),
        }
        for scenario, path in pairs
    ]

    print(
        f"[evalrun] tag={tag} sim={sim} policy={policy} scenarios={len(tasks)} "
        f"steps={steps} workers={workers} out={out_dir}"
        + (f" oracle={','.join(oracle_sources)}" if oracle_sources else "")
        + (f" cenv={','.join(cenv_requested)}" if cenv_requested else "")
        + (" fast=on" if fast else ""),
        flush=True,
    )

    results: list[dict[str, Any]] = []
    started = time.time()
    if workers <= 1:
        for task in tasks:
            result = _worker(task)
            results.append(result)
            _print_progress(result, len(results), len(tasks))
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=workers, maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(_worker, tasks, chunksize=1):
                results.append(result)
                _print_progress(result, len(results), len(tasks))

    summary = _summarise(
        results,
        tag=tag,
        policy=policy,
        params=params or {},
        sim=sim,
        steps=int(steps),
        start_offset_days=int(start_offset_days),
        components=components,
        occupancy_seed=occupancy_seed,
        oracle=oracle_sources,
        oracle_horizon=int(oracle_horizon) if oracle_sources else None,
        oracle_file=oracle_file,
        oracle_record=oracle_record,
        out_dir=out_dir,
        wall_seconds=time.time() - started,
        cenv=cenv_requested,
        fast=bool(fast),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[evalrun] score={summary['leaderboard_score']:.6f} "
        f"scenarios={summary['n_scenarios']} failed={summary['n_failed']} "
        f"wall={summary['timing']['wall_seconds_total']:.0f}s "
        f"-> {out_dir / 'summary.json'}",
        flush=True,
    )
    return summary


def _print_progress(result: dict[str, Any], done: int, total: int) -> None:
    if "error" in result:
        first = result["error"].strip().splitlines()[-1]
        print(f"[evalrun] {done}/{total} FAILED {result['scenario']}: {first}", flush=True)
    else:
        print(
            f"[evalrun] {done}/{total} {result['scenario']:<38} "
            f"total={result['total_reward']:12.4f} "
            f"mean={result['mean_per_step_reward']:.6f} "
            f"steps={result['steps']} cpu={result['cpu_seconds']:.1f}s",
            flush=True,
        )


def _summarise(
    results: list[dict[str, Any]],
    tag: str,
    policy: str,
    params: dict[str, Any],
    sim: str,
    steps: int,
    start_offset_days: int,
    components: str,
    occupancy_seed: int | None,
    oracle: tuple[str, ...],
    oracle_horizon: int | None,
    out_dir: Path,
    wall_seconds: float,
    cenv: Sequence[str] = (),
    fast: bool = False,
    oracle_file: str | None = None,
    oracle_record: str | None = None,
) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    if not ok:
        raise RuntimeError(
            "every scenario failed; first error:\n" + (failed[0]["error"] if failed else "")
        )

    scores = np.array([abs(r["mean_per_step_reward"]) for r in ok])
    totals = np.array([r["total_reward"] for r in ok])

    def _group(key: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in ok:
            groups.setdefault(r[key], []).append(r)
        return {
            name: {
                "n_scenarios": len(group),
                "leaderboard_score": float(
                    np.mean([abs(g["mean_per_step_reward"]) for g in group])
                ),
                "mean_total_reward": float(np.mean([g["total_reward"] for g in group])),
                "mean_per_step_reward": float(
                    np.mean([g["mean_per_step_reward"] for g in group])
                ),
            }
            for name, group in sorted(groups.items())
        }

    cpu_total = float(sum(r["cpu_seconds"] for r in ok))
    wall_total = float(sum(r["wall_seconds"] for r in ok))
    total_steps = int(sum(r["steps"] for r in ok))

    summary = {
        "tag": tag,
        "policy": policy,
        "policy_labels": sorted({r["policy"] for r in ok}),
        "params": params,
        "sim": sim,
        "steps": steps,
        "start_offset_days": start_offset_days,
        "occupancy_seed": (
            None if occupancy_seed is None else int(occupancy_seed)
        ),
        "oracle": list(oracle),
        "oracle_horizon": oracle_horizon,
        "oracle_file": oracle_file,
        "oracle_record": oracle_record,
        "components_mode": components,
        "cenv_components": list(cenv),
        "fast": bool(fast),
        "out_dir": str(out_dir),
        # Leaderboard metric: mean over scenarios of |mean per-step reward|.
        "leaderboard_score": float(scores.mean()),
        "mean_abs_total_reward_per_step": float(scores.mean()),
        "mean_total_reward": float(totals.mean()),
        "mean_per_step_reward": float(np.mean([r["mean_per_step_reward"] for r in ok])),
        "n_scenarios": len(ok),
        "n_failed": len(failed),
        "sum_comfort_all": float(sum(r["components"]["sum_comfort"] for r in ok)),
        "sum_energy_all": float(sum(r["components"]["sum_energy"] for r in ok)),
        "sum_carbon_all": float(sum(r["components"]["sum_carbon"] for r in ok)),
        "by_weather": _group("weather"),
        "by_building": _group("building"),
        "timing": {
            "wall_seconds_total": float(wall_seconds),
            "cpu_seconds_total": cpu_total,
            "wall_seconds_sum_scenarios": wall_total,
            "mean_cpu_seconds_per_scenario": cpu_total / len(ok),
            "steps_per_second_cpu": (total_steps / cpu_total) if cpu_total else None,
            "steps_per_second_step_cpu": (
                total_steps
                / max(1e-9, float(sum(r["step_cpu_seconds"] for r in ok)))
            ),
        },
        "scenarios": [
            {
                "scenario": r["scenario"],
                "steps": r["steps"],
                "total_reward": r["total_reward"],
                "mean_per_step_reward": r["mean_per_step_reward"],
                "cpu_seconds": r["cpu_seconds"],
                "wall_seconds": r["wall_seconds"],
                "steps_per_second_cpu": r["steps_per_second_cpu"],
            }
            for r in sorted(ok, key=lambda r: r["scenario"])
        ],
        "failures": [
            {
                "scenario": r["scenario"],
                "config_path": r["config_path"],
                "error": r["error"],
            }
            for r in failed
        ],
    }
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a control policy over scenarios, one process per scenario.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--policy",
        default="policies.ctrl_v1:CtrlV1",
        help=(
            "'baseline:<key>' (pid_water, pid, bang_bang, fixed_setpoint, all_off), "
            "'module:Class', or a bare baseline key"
        ),
    )
    parser.add_argument(
        "--params",
        default=None,
        help="JSON dict of policy PARAMS overrides, or a path to a JSON file",
    )
    parser.add_argument(
        "--set",
        dest="sets",
        nargs="+",
        default=None,
        help="scenario set names: dev30, dev30_<station>, dev30_<strip>, gen_train, gen_hold, gen_all",
    )
    parser.add_argument(
        "--scenario",
        dest="scenario_ids",
        nargs="+",
        default=None,
        help="scenario ids, e.g. building_train_1/atlanta_2",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="config.gin paths or scenario directories",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--sim", default="shipped", choices=list(simload.SIM_NAMES))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", default=None, help="results dir (default results/<tag>)")
    parser.add_argument("--start-offset-days", type=int, default=0)
    parser.add_argument(
        "--occupancy-seed",
        type=int,
        default=None,
        help=(
            "override RandomizedArrivalDepartureOccupancy.seed (default 17321) "
            "for robustness tests; omitted leaves the config alone"
        ),
    )
    parser.add_argument(
        "--components",
        default="parity",
        choices=["parity", "hook"],
        help="parity reproduces evaluate.py's extra reward-info call per step",
    )
    parser.add_argument(
        "--oracle",
        default=None,
        help=(
            "analysis-only channel handed to a policy's observe_oracle each "
            "step: 'occupancy', 'forecast' or 'occupancy,forecast'. Policies "
            "that ignore it reproduce a normal run bit-for-bit"
        ),
    )
    parser.add_argument(
        "--cenv",
        default=None,
        help=(
            "opt-in C components installed by monkeypatch at runtime: a "
            "comma-separated list of bookkeeping, thermal, stochastic, or "
            "'all' for every component whose gate has passed. Additive to "
            "--components/--oracle/--fast; omitted means pure Python"
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "install sbx.fastpatch (pure-Python runtime patches: dense air-side "
            "Woodbury response, per-step zone-average memo, timestamp memos, "
            "hoisted reward-info loops). Same numbers to float round-off; "
            "off by default, and additive to --cenv/--oracle/--components"
        ),
    )
    parser.add_argument(
        "--oracle-file",
        default=None,
        help=(
            "path (directory or json file) of a recorded per-zone occupancy "
            "schedule, used by the 'foresight' oracle source"
        ),
    )
    parser.add_argument(
        "--oracle-record",
        default=None,
        help=(
            "directory to write this run's per-zone occupancy schedule to "
            "(needs --oracle occupancy); sparse json, used by --oracle foresight"
        ),
    )
    parser.add_argument(
        "--oracle-horizon",
        type=int,
        default=DEFAULT_ORACLE_HORIZON,
        help="forecast horizon in hours (default %(default)s)",
    )
    return parser


def _load_params(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    # Inline JSON first: Path(value).exists() raises OSError (name too long) on
    # macOS for a JSON string over 255 characters.
    if value.lstrip().startswith("{"):
        return json.loads(value)
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    selections = [
        args.sets or [],
        args.scenario_ids or [],
        args.configs or [],
    ]
    items: list[Any] = [item for group in selections for item in group]
    if not items:
        items = ["dev30"]

    summary = run(
        policy=args.policy,
        params=_load_params(args.params),
        scenarios=items,
        steps=args.steps,
        sim=args.sim,
        workers=args.workers,
        out=args.out,
        start_offset_days=args.start_offset_days,
        tag=args.tag,
        occupancy_seed=args.occupancy_seed,
        components=args.components,
        cenv=args.cenv,
        fast=args.fast,
        oracle=args.oracle,
        oracle_horizon=args.oracle_horizon,
        oracle_file=args.oracle_file,
        oracle_record=args.oracle_record,
    )
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
