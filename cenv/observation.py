"""``observation``: the observation vector assembled without protobuf.

``HybridActionEnvironment._get_observation`` builds an ``ObservationResponse``
(232 ``SingleObservationResponse`` messages for the shipped buildings), then a
*second* normalised copy, then a dict of 232 numpy scalars, then a float32 array
indexed by ``self.field_names``. Every one of those intermediate objects exists
only to be indexed once.

This component keeps the one part that is not a pure function of state -- the
device sensors, whose reads are stateful (the boiler's tank advances on every
read of its temperature sensor) -- and replaces the pure part. From the raw
``ObservationResponse`` it writes the float32 vector directly, normalising each
measurement with the same per-*measurement-name* constants, instead of

  1. copying the whole response into a second proto and normalising that copy,
  2. building a dict of 232 numpy scalars keyed by field name,
  3. indexing that dict once per field into a fresh array.

The 48 hour-of-day / day-of-week features are still produced by
``regression_building_utils.expand_time_features`` on the same radian input, and
the three auxiliary fields keep their exact formulas, so the array is
bit-identical to the shipped one -- and because the device reads are literally
the original code, the trajectory cannot move.

If a metrics writer or a histogram reducer is configured (``metrics_path`` set),
the replacement delegates to the original method untouched.
"""

from __future__ import annotations

import numpy as np

MARKER = "__cenv__"
GATE_PASSED = True

_COMFORT_NOW = "comfort_mode_now"
_COMFORT_SOON = "comfort_mode_soon"
_NUM_OCCUPANTS = "num_occupants"


def _env_mod():
    from smart_buildings.smart_control.environment import environment

    return environment


class _ObsPlan:
    """Field slots of one environment: device, time feature, or auxiliary."""

    __slots__ = (
        "env",
        "names",
        "n",
        "name_index",
        "device_slots",
        "time_slots",
        "comfort_now",
        "comfort_soon",
        "num_occupants",
        "num_hod",
        "num_dow",
        "hod_label",
        "dow_label",
        "n_device",
    )

    def __init__(self, env, env_mod):
        self.env = env
        self.names = list(env.field_names)
        self.n = len(self.names)
        self.name_index = {name: i for i, name in enumerate(self.names)}
        normalizer = env._observation_normalizer
        # `_id_map.inv` gives the (device_id, measurement) pair behind every
        # device-derived field name, which is exactly the key the observation
        # response is indexed by. Device objects are not needed: the raw values
        # come from the response `_get_observation_response` already built.
        inv = env._id_map.inv
        device_slots = []
        time_slots = []
        self.n_device = 0
        for i, name in enumerate(self.names):
            if name in inv:
                device_id, measurement_name = inv[name]
                constants = normalizer._get_normalization_constants(
                    measurement_name
                )
                variance = constants.sample_variance
                device_slots.append(
                    (
                        i,
                        constants.sample_mean,
                        float(np.sqrt(variance)) if variance > 0.0 else 0.0,
                        (device_id, measurement_name),
                    )
                )
                continue
            if name in (_COMFORT_NOW, _COMFORT_SOON, _NUM_OCCUPANTS):
                continue
            time_slots.append(name)
        self.device_slots = device_slots
        self.n_device = len(device_slots)
        self.time_slots = time_slots
        self.comfort_now = self.name_index.get(_COMFORT_NOW, -1)
        self.comfort_soon = self.name_index.get(_COMFORT_SOON, -1)
        self.num_occupants = self.name_index.get(_NUM_OCCUPANTS, -1)
        self.num_hod = int(env._num_hod_features)
        self.num_dow = int(env._num_dow_features)
        self.hod_label = env_mod.HOD_LABEL
        self.dow_label = env_mod.DOW_LABEL
        env._cenv_obs_plan = self


def _plan(env, env_mod) -> _ObsPlan:
    plan = env.__dict__.get("_cenv_obs_plan")
    if plan is None or plan.env is not env or plan.n != len(env.field_names):
        plan = _ObsPlan(env, env_mod)
    return plan


def _source_from_response(response):
    """``(device_id, measurement) -> raw value`` for the valid observations."""
    source = {}
    for single in response.single_observation_responses:
        request = single.single_observation_request
        if single.observation_valid:
            source[(request.device_id, request.measurement_name)] = (
                single.continuous_value
            )
    return source


def _assemble(env, env_mod, plan, source):
    """Build the observation array from one step's raw measurements.

    ``source`` maps ``(device_id, measurement)`` to the value the building
    reported -- already gap-filled by ``replace_missing_observations_past``
    inside ``_get_observation_response``, so every field is present. The vector
    is identical to the shipped assembly.
    """
    out = np.empty(plan.n, dtype=np.float32)
    ts = env.current_simulation_timestamp

    # -- device measurements ------------------------------------------------ #
    # Same values, same normalisation (per measurement name), same rounding to
    # float32 as `np.array(continuous_value, dtype=np.float32)`.
    for idx, mean, std, key in plan.device_slots:
        value = source[key]
        out[idx] = (value - mean) / std if std > 0.0 else 0.0

    # -- time features ------------------------------------------------------ #
    if plan.time_slots:
        from smart_buildings.smart_control.utils import (
            conversion_utils,
            regression_building_utils,
        )

        index = plan.name_index
        hod_rad = conversion_utils.get_radian_time(
            ts, conversion_utils.TimeIntervalEnum.HOUR_OF_DAY
        )
        for key, value in regression_building_utils.expand_time_features(
            plan.num_hod, hod_rad, plan.hod_label
        ).items():
            out[index[f"{key[0]}_{key[1]}"]] = value
        dow_rad = conversion_utils.get_radian_time(
            ts, conversion_utils.TimeIntervalEnum.DAY_OF_WEEK
        )
        for key, value in regression_building_utils.expand_time_features(
            plan.num_dow, dow_rad, plan.dow_label
        ).items():
            out[index[f"{key[0]}_{key[1]}"]] = value

    # -- schedule and occupancy --------------------------------------------- #
    building = env.building
    if plan.comfort_now >= 0:
        out[plan.comfort_now] = building.is_comfort_mode(ts)
    if plan.comfort_soon >= 0:
        out[plan.comfort_soon] = building.is_comfort_mode(
            ts + env_mod.pd.Timedelta(60, unit="minute")
        )
    if plan.num_occupants >= 0:
        norm = env.occupancy_normalization_constant
        out[plan.num_occupants] = (building.num_occupants - norm) / (norm + 1)

    if np.isnan(out).any():
        names = [env.field_names[i] for i in np.flatnonzero(np.isnan(out))]
        print(f"Observation vector contains Nans at {', '.join(names)}.")
    if np.isinf(out).any():
        names = [env.field_names[i] for i in np.flatnonzero(np.isinf(out))]
        print(f"Observation vector contains Infs at {', '.join(names)}.")
    return out


def install(sim_root) -> dict:
    env_mod = _env_mod()
    cls = env_mod.Environment
    original = cls._get_observation

    def _get_observation(self):
        # Keep the protobuf path wherever the env API needs an actual response:
        # a metrics writer (evaluate.py) or a histogram reducer consumes it, and
        # a silent bypass would change their output.
        if (
            self._metrics_writer is not None
            or self._observation_histogram_reducer is not None
            or not hasattr(self._observation_normalizer, "_get_normalization_constants")
        ):
            return original(self)
        # The response build stays: it is the env API's observation object and it
        # owns the only stateful reads in this method. Everything after it is a
        # pure function of that response.
        response = self._get_observation_response()
        return _assemble(self, env_mod, _plan(self, env_mod),
                         _source_from_response(response))

    _get_observation.__cenv__ = MARKER
    cls._get_observation = _get_observation
    return {
        "patched": ["Environment._get_observation"],
        "originals": {"_get_observation": original},
        "restore": [(cls, "_get_observation", original)],
    }


# --------------------------------------------------------------------------- #
# shadow verification
# --------------------------------------------------------------------------- #
def shadow_pairs(shadow, sim_root):
    """Compare the C-free fast assembly against the shipped one, bit for bit."""
    env_mod = _env_mod()
    shadow.pair(
        env_mod.Environment,
        "_get_observation",
        lambda self: _measure(self, env_mod),
        kind="array",
        label="Environment._get_observation",
    )


def _measure(env, env_mod):
    """Rebuild from the response the original just built (no device access)."""
    response = env._last_observation_response
    return _assemble(env, env_mod, _plan(env, env_mod),
                     _source_from_response(response))
