"""``bookkeeping``: per-zone temperature statistics and the reward arithmetic.

Two drop-in replacements, installed by monkeypatching the simulator classes:

``Building.get_zone_temp_stats`` / ``Building.get_zone_average_temps``
    One CSR pass over precomputed zone labels per call instead of a numpy
    fancy-index gather plus three reductions per zone. The pairwise sum is
    numpy's own summation order, so the means are bit-identical; min/max are
    exactly ``np.min`` / ``np.max``.

``SetpointEnergyCarbonRegretFunction.compute_reward`` (plus the productivity /
cost / carbon helpers it is built from)
    The arithmetic moves to C: productivity per zone, the plant rate sums, and
    the electricity / gas cost and carbon tables. The Python side still owns the
    ``RewardInfo`` / ``RewardResponse`` protobufs, because the environment API
    hands those objects to callers.

Nothing here changes behaviour: every replacement is gated by
``cenv/verify/shadow.py`` against the Python original before ``--cenv`` is used
for a scored run.
"""

from __future__ import annotations

import ctypes

import numpy as np

from . import bindings
from .bindings import c_double, c_int32, c_int64, c_void_p, f64, i64, lib

MARKER = "__cenv__"
# Set once `cenv/verify/gate.py --component bookkeeping` passes here: `--cenv all`
# only enables components that carry this flag (explicit `--cenv name` opts in
# regardless).
GATE_PASSED = True
_REWARD_OUT_LEN = 18  # CENV_RW_LEN in cenv/include/cenv.h


# --------------------------------------------------------------------------- #
# ctypes prototypes
# --------------------------------------------------------------------------- #
def _bind() -> None:
    for fn in (
        ("cenv_bk_plan_create", c_int64, [c_int64, i64, i64]),
        ("cenv_bk_plan_destroy", None, [c_int64]),
        ("cenv_bk_plan_n_zones", c_int64, [c_int64]),
        # Optional outputs are `c_void_p`: None means "skip that output", and an
        # ndpointer argtype would reject None and silently verify the wrong size.
        (
            "cenv_bk_zone_stats",
            c_int32,
            [c_int64, f64, c_int64, i64, c_int64, c_void_p, c_void_p,
             c_void_p, c_void_p],
        ),
        ("cenv_bk_zone_productivity", c_double,
         [c_double] * 7),
    ):
        bindings.bind(*fn)


_bind()
bindings.on_reload(_bind)


# --------------------------------------------------------------------------- #
# zone statistics
# --------------------------------------------------------------------------- #
class _ZonePlan:
    """CSR of flat CV indices per zone, shared by every call on one building."""

    __slots__ = (
        "building",
        "handle",
        "names",
        "index",
        "grid_n",
        "phys_sel",
        "phys_names",
        "sel1",
        "buf_min",
        "buf_max",
        "buf_mean",
        "means",
        "n_zones",
    )

    def __init__(self, building, constants):
        room_dict = building.room_dict
        self.building = building
        self.names = list(room_dict)
        self.index = {name: i for i, name in enumerate(self.names)}
        temp = building.temp
        if not temp.flags["C_CONTIGUOUS"]:
            # The plan stores flat C-order indices; an F-ordered field would
            # silently gather the wrong cells, so refuse instead.
            raise RuntimeError(
                "cenv bookkeeping needs a C-contiguous temperature field"
            )
        shape = temp.shape
        self.grid_n = int(temp.size)
        offsets = np.zeros(len(self.names) + 1, dtype=np.int64)
        cells = []
        ny = shape[1]
        for i, name in enumerate(self.names):
            coords = np.asarray(room_dict[name], dtype=np.int64).reshape(-1, 2)
            flat = coords[:, 0] * ny + coords[:, 1]
            cells.append(flat)
            offsets[i + 1] = offsets[i] + flat.size
        cells = (
            np.concatenate(cells) if cells else np.zeros(0, dtype=np.int64)
        ).astype(np.int64, copy=False)
        self.handle = int(
            lib.cenv_bk_plan_create(len(self.names), offsets, cells)
        )
        if self.handle == 0:
            raise RuntimeError("cenv_bk_plan_create failed")
        self.n_zones = len(self.names)
        # `get_zone_average_temps` skips the non-physical members of room_dict.
        self.phys_sel = np.array(
            [i for i, n in enumerate(self.names) if not constants.is_non_physical_space(n)],
            dtype=np.int64,
        )
        self.phys_names = [self.names[i] for i in self.phys_sel]
        self.sel1 = np.zeros(1, dtype=np.int64)
        self.buf_min = np.zeros(1, dtype=np.float64)
        self.buf_max = np.zeros(1, dtype=np.float64)
        self.buf_mean = np.zeros(1, dtype=np.float64)
        self.means = np.zeros(self.phys_sel.size, dtype=np.float64)
        # The plan is tied to the instance: numpy's zone labels never change, and
        # `temp.size` catches a rebuilt building (a new instance gets a new plan).
        building._cenv_zone_plan = self

    def sel_for(self, zone: int) -> int:
        self.sel1[0] = zone
        return 1


def _plan(building, constants) -> _ZonePlan:
    plan = building.__dict__.get("_cenv_zone_plan")
    if plan is None or plan.grid_n != int(building.temp.size):
        plan = _ZonePlan(building, constants)
    return plan


def install_zone_stats(sim_root) -> dict:
    from smart_buildings.smart_control.simulator import building as building_mod
    from smart_buildings.smart_control.simulator import constants

    cls = building_mod.FloorPlanBasedBuilding
    # Whatever is there now (possibly a peer's cache wrapper) is the reference
    # the shadow harness compares against; we replace the attribute itself.
    original_stats = cls.get_zone_temp_stats
    original_avg = cls.get_zone_average_temps
    stats_fn = lib.cenv_bk_zone_stats

    def get_zone_temp_stats(self, zone_name):
        plan = _plan(self, constants)
        zone = plan.index.get(zone_name)
        if zone is None:
            raise ValueError("Zone name is not present in room_dict.")
        n = plan.sel_for(zone)
        temp = self.temp
        rc = stats_fn(
            plan.handle, temp, temp.size, plan.sel1, n,
            plan.buf_min.ctypes.data, plan.buf_max.ctypes.data,
            plan.buf_mean.ctypes.data, None,
        )
        if rc != 0:
            raise RuntimeError(f"cenv_bk_zone_stats failed ({rc})")
        return plan.buf_min[0], plan.buf_max[0], plan.buf_mean[0]

    def get_zone_average_temps(self):
        plan = _plan(self, constants)
        temp = self.temp
        rc = stats_fn(
            plan.handle, temp, temp.size, plan.phys_sel, plan.phys_sel.size,
            None, None, plan.means.ctypes.data, None,
        )
        if rc != 0:
            raise RuntimeError(f"cenv_bk_zone_stats failed ({rc})")
        return dict(zip(plan.phys_names, plan.means.tolist()))

    get_zone_temp_stats.__cenv__ = MARKER
    get_zone_average_temps.__cenv__ = MARKER
    cls.get_zone_temp_stats = get_zone_temp_stats
    cls.get_zone_average_temps = get_zone_average_temps
    return {
        "patched": [
            "FloorPlanBasedBuilding.get_zone_temp_stats",
            "FloorPlanBasedBuilding.get_zone_average_temps",
        ],
        "originals": {"stats": original_stats, "average": original_avg},
        "restore": [
            (cls, "get_zone_temp_stats", original_stats),
            (cls, "get_zone_average_temps", original_avg),
        ],
        "plan_class": _ZonePlan,
    }


# --------------------------------------------------------------------------- #
# reward
# --------------------------------------------------------------------------- #
class _RewardParams(ctypes.Structure):
    _fields_ = [
        ("delta_time_sec", c_double),
        ("max_productivity_personhour_usd", c_double),
        ("min_productivity_personhour_usd", c_double),
        ("productivity_deviation_ref", c_double),
        ("productivity_weight", c_double),
        ("energy_cost_weight", c_double),
        ("carbon_emission_weight", c_double),
        ("max_electricity_rate", c_double),
        ("max_natural_gas_rate", c_double),
        ("electricity_price", c_double),
        ("electricity_carbon_rate", c_double),
        ("gas_price", c_double),
        ("gas_carbon_rate", c_double),
    ]


class _ZoneInputs(ctypes.Structure):
    _fields_ = [
        ("heating_setpoint", f64),
        ("cooling_setpoint", f64),
        ("zone_temp", f64),
        ("occupancy", f64),
    ]


class _PlantInputs(ctypes.Structure):
    _fields_ = [
        ("ahu_blower_rate", f64),
        ("ahu_cooling_rate", f64),
        ("n_ahu", c_int64),
        ("boiler_pump_rate", f64),
        ("boiler_gas_rate", f64),
        ("n_boiler", c_int64),
        ("hp_pump_rate", f64),
        ("hp_electric_heat_rate", f64),
        ("n_heat_pump", c_int64),
    ]


lib.cenv_bk_reward.restype = c_int32
lib.cenv_bk_reward.argtypes = [
    c_int64,
    ctypes.POINTER(_ZoneInputs),
    ctypes.POINTER(_PlantInputs),
    ctypes.POINTER(_RewardParams),
    f64,
]

_CENV_RW_AGENT_REWARD_VALUE = 0


class _RewardBuffers:
    """Reused per-instance buffers: no allocation on the per-step path."""

    __slots__ = ("zones", "n_zones", "plant", "params", "out", "arrays",
                 "n_ahu", "n_boiler", "n_hp")

    def __init__(self):
        self.zones = _ZoneInputs()
        self.n_zones = 0
        self.plant = _PlantInputs()
        self.n_ahu = 0
        self.n_boiler = 0
        self.n_hp = 0
        self.params = _RewardParams()
        self.out = np.zeros(_REWARD_OUT_LEN, dtype=np.float64)
        self.arrays = None

    def ensure(self, n_zones, n_ahu, n_boiler, n_hp):
        if self.n_zones != n_zones or self.arrays is None:
            self.arrays = {
                "heating_setpoint": np.zeros(n_zones, dtype=np.float64),
                "cooling_setpoint": np.zeros(n_zones, dtype=np.float64),
                "zone_temp": np.zeros(n_zones, dtype=np.float64),
                "occupancy": np.zeros(n_zones, dtype=np.float64),
            }
            self.n_zones = n_zones
        for name, n in (
            ("ahu_blower_rate", n_ahu),
            ("ahu_cooling_rate", n_ahu),
            ("boiler_pump_rate", n_boiler),
            ("boiler_gas_rate", n_boiler),
            ("hp_pump_rate", n_hp),
            ("hp_electric_heat_rate", n_hp),
        ):
            if name not in self.arrays or self.arrays[name].size != n:
                self.arrays[name] = np.zeros(n, dtype=np.float64)
        self.n_ahu = n_ahu
        self.n_boiler = n_boiler
        self.n_hp = n_hp


def _energy_rates(cost, start_time):
    """``ElectricityEnergyCost`` / ``NaturalGasEnergyCost`` inputs for one step.

    Returns (price, carbon_rate) exactly as those classes index their tables:
    the electricity price depends on local hour and work-day, its carbon rate on
    local hour, the gas price on month; the gas carbon rate is a constant.
    """
    if hasattr(cost, "_weekday_energy_prices"):
        from smart_buildings.smart_control.utils import conversion_utils

        hour = start_time.hour
        if conversion_utils.is_work_day(start_time):
            price = float(cost._weekday_energy_prices[hour])
        else:
            price = float(cost._weekend_energy_prices[hour])
        carbon = float(cost._carbon_emission_rates[hour])
        return price, carbon
    price = float(cost._month_gas_price[start_time.month - 1])
    return price, float(cost._carbon_rate)


def _compute_reward_c(reward_function, reward_info):
    """C-backed body of ``SetpointEnergyCarbonRegretFunction.compute_reward``."""
    from smart_buildings.smart_control.proto import smart_control_reward_pb2
    from smart_buildings.smart_control.utils import conversion_utils

    start_time = conversion_utils.proto_to_pandas_timestamp(
        reward_info.start_timestamp
    )
    end_time = conversion_utils.proto_to_pandas_timestamp(reward_info.end_timestamp)
    delta_time_sec = (end_time - start_time).total_seconds()

    zone_infos = reward_info.zone_reward_infos
    n_zones = len(zone_infos)
    ahu_infos = reward_info.air_handler_reward_infos
    boiler_infos = reward_info.boiler_reward_infos
    hp_infos = reward_info.heat_pump_reward_infos
    n_ahu, n_boiler, n_hp = len(ahu_infos), len(boiler_infos), len(hp_infos)

    buf = reward_function.__dict__.get("_cenv_reward_buffers")
    if buf is None:
        buf = _RewardBuffers()
        reward_function._cenv_reward_buffers = buf
    buf.ensure(n_zones, n_ahu, n_boiler, n_hp)
    a = buf.arrays
    hs, cs, zt, oc = (
        a["heating_setpoint"],
        a["cooling_setpoint"],
        a["zone_temp"],
        a["occupancy"],
    )
    # Field-by-field rather than one comprehension per array: one pass over the
    # map, four writes, no intermediate Python lists.
    i = 0
    for info in zone_infos.values():
        hs[i] = info.heating_setpoint_temperature
        cs[i] = info.cooling_setpoint_temperature
        zt[i] = info.zone_air_temperature
        oc[i] = info.average_occupancy
        i += 1
    blower, cooling = a["ahu_blower_rate"], a["ahu_cooling_rate"]
    for j, info in enumerate(ahu_infos.values()):
        blower[j] = info.blower_electrical_energy_rate
        cooling[j] = info.air_conditioning_electrical_energy_rate
    pump, gas = a["boiler_pump_rate"], a["boiler_gas_rate"]
    for j, info in enumerate(boiler_infos.values()):
        pump[j] = info.pump_electrical_energy_rate
        gas[j] = info.natural_gas_heating_energy_rate
    hp_pump, hp_heat = a["hp_pump_rate"], a["hp_electric_heat_rate"]
    for j, info in enumerate(hp_infos.values()):
        hp_pump[j] = info.pump_electrical_energy_rate
        hp_heat[j] = info.electricity_heating_energy_rate

    elec_price, elec_carbon = _energy_rates(
        reward_function._electricity_energy_cost, start_time
    )
    gas_price, gas_carbon = _energy_rates(
        reward_function._natural_gas_energy_cost, start_time
    )

    params = buf.params
    params.delta_time_sec = delta_time_sec
    params.max_productivity_personhour_usd = (
        reward_function._max_productivity_personhour_usd
    )
    params.min_productivity_personhour_usd = (
        reward_function._min_productivity_personhour_usd
    )
    params.productivity_deviation_ref = reward_function._productivity_deviation_ref
    params.productivity_weight = reward_function._productivity_weight
    params.energy_cost_weight = reward_function._energy_cost_weight
    params.carbon_emission_weight = reward_function._carbon_emission_weight
    params.max_electricity_rate = reward_function._max_electricity_rate
    params.max_natural_gas_rate = reward_function._max_natural_gas_rate
    params.electricity_price = elec_price
    params.electricity_carbon_rate = elec_carbon
    params.gas_price = gas_price
    params.gas_carbon_rate = gas_carbon

    zones = buf.zones
    zones.heating_setpoint = hs.ctypes.data
    zones.cooling_setpoint = cs.ctypes.data
    zones.zone_temp = zt.ctypes.data
    zones.occupancy = oc.ctypes.data
    plant = buf.plant
    plant.ahu_blower_rate = blower.ctypes.data
    plant.ahu_cooling_rate = cooling.ctypes.data
    plant.n_ahu = n_ahu
    plant.boiler_pump_rate = pump.ctypes.data
    plant.boiler_gas_rate = gas.ctypes.data
    plant.n_boiler = n_boiler
    plant.hp_pump_rate = hp_pump.ctypes.data
    plant.hp_electric_heat_rate = hp_heat.ctypes.data
    plant.n_heat_pump = n_hp

    rc = lib.cenv_bk_reward(
        n_zones,
        ctypes.byref(zones),
        ctypes.byref(plant),
        ctypes.byref(params),
        buf.out,
    )
    if rc != 0:
        raise RuntimeError(f"cenv_bk_reward failed ({rc})")
    out = buf.out

    response = smart_control_reward_pb2.RewardResponse()
    response.productivity_reward = float(out[1])
    response.natural_gas_energy_cost = float(out[2])
    response.electricity_energy_cost = float(out[3])
    response.carbon_emitted = float(out[4])
    response.productivity_weight = reward_function._productivity_weight
    response.energy_cost_weight = reward_function._energy_cost_weight
    response.carbon_emission_weight = reward_function._carbon_emission_weight
    response.person_productivity = (
        reward_function._max_productivity_personhour_usd
    )
    response.total_occupancy = float(out[9])
    response.reward_scale = 1.0
    response.reward_shift = 0.0
    response.productivity_regret = float(out[5])
    response.normalized_productivity_regret = float(out[6])
    response.normalized_energy_cost = float(out[7])
    response.normalized_carbon_emission = float(out[8])
    response.start_timestamp.CopyFrom(reward_info.start_timestamp)
    response.end_timestamp.CopyFrom(reward_info.end_timestamp)
    response.agent_reward_value = float(out[0])
    return response


def install_reward(sim_root) -> dict:
    from smart_buildings.smart_control.models import base_energy_cost
    from smart_buildings.smart_control.reward import (
        base_setpoint_energy_carbon_reward as base_reward,
    )
    from smart_buildings.smart_control.reward import (
        electricity_energy_cost as elec_mod,
    )
    from smart_buildings.smart_control.reward import (
        natural_gas_energy_cost as gas_mod,
    )
    from smart_buildings.smart_control.reward import (
        setpoint_energy_carbon_regret as regret_mod,
    )

    regret_cls = regret_mod.SetpointEnergyCarbonRegretFunction
    base_cls = base_reward.BaseSetpointEnergyCarbonRewardFunction

    original_compute = regret_cls.compute_reward
    original_zone_prod = base_cls._sum_zone_productivities
    original_elec = base_cls._sum_electricity_energy_rate
    original_gas = base_cls._sum_natural_gas_energy_rate

    def compute_reward(self, reward_info):
        return _compute_reward_c(self, reward_info)

    def _sum_zone_productivities(self, energy_reward_info):
        # Kept for callers other than compute_reward: same numbers, C arithmetic.
        # `compute_reward` itself never reaches these helpers; they exist so a
        # caller of the base class sees the same C-backed behaviour.
        return _zone_productivities_c(self, energy_reward_info)

    def _sum_electricity_energy_rate(self, energy_reward_info):
        rate = 0.0
        for info in energy_reward_info.air_handler_reward_infos.values():
            rate += info.blower_electrical_energy_rate + abs(
                info.air_conditioning_electrical_energy_rate
            )
        for info in energy_reward_info.boiler_reward_infos.values():
            rate += info.pump_electrical_energy_rate
        for info in energy_reward_info.heat_pump_reward_infos.values():
            rate += abs(info.pump_electrical_energy_rate)
            rate += abs(info.electricity_heating_energy_rate)
        return rate

    def _sum_natural_gas_energy_rate(self, energy_reward_info):
        rate = 0.0
        for info in energy_reward_info.boiler_reward_infos.values():
            rate += info.natural_gas_heating_energy_rate
        return rate

    # The cost / carbon tables themselves. `compute_reward` no longer calls
    # these (it hands the four per-step rates to C), but they are what the rest
    # of the tree prices energy with, so they are ported too and the port is
    # gated in the shadow run (100% bit-identical over the gated episodes).
    elec_cls = elec_mod.ElectricityEnergyCost
    gas_cls = gas_mod.NaturalGasEnergyCost
    originals = {
        "elec_cost": elec_cls.cost,
        "elec_carbon": elec_cls.carbon,
        "gas_cost": gas_cls.cost,
        "gas_carbon": gas_cls.carbon,
    }
    patched_costs = {
        (elec_cls, "cost"): (elec_cls.cost, _elec_cost_c),
        (elec_cls, "carbon"): (elec_cls.carbon, _elec_carbon_c),
        (gas_cls, "cost"): (gas_cls.cost, _gas_cost_c),
        (gas_cls, "carbon"): (gas_cls.carbon, _gas_carbon_c),
    }

    for fn in (compute_reward, _sum_zone_productivities,
               _sum_electricity_energy_rate, _sum_natural_gas_energy_rate):
        fn.__cenv__ = MARKER
    regret_cls.compute_reward = compute_reward
    base_cls._sum_zone_productivities = _sum_zone_productivities
    base_cls._sum_electricity_energy_rate = _sum_electricity_energy_rate
    base_cls._sum_natural_gas_energy_rate = _sum_natural_gas_energy_rate
    for (cls, name), (before, after) in patched_costs.items():
        after.__cenv__ = MARKER
        setattr(cls, name, after)
    return {
        "patched": [
            "SetpointEnergyCarbonRegretFunction.compute_reward",
            "BaseSetpointEnergyCarbonRewardFunction._sum_zone_productivities",
            "BaseSetpointEnergyCarbonRewardFunction._sum_electricity_energy_rate",
            "BaseSetpointEnergyCarbonRewardFunction._sum_natural_gas_energy_rate",
            "ElectricityEnergyCost.cost",
            "ElectricityEnergyCost.carbon",
            "NaturalGasEnergyCost.cost",
            "NaturalGasEnergyCost.carbon",
        ],
        "restore": [
            (regret_cls, "compute_reward", original_compute),
            (base_cls, "_sum_zone_productivities", original_zone_prod),
            (base_cls, "_sum_electricity_energy_rate", original_elec),
            (base_cls, "_sum_natural_gas_energy_rate", original_gas),
            *[(cls, name, before) for (cls, name), (before, _) in patched_costs.items()],
        ],
        "originals": {
            "compute_reward": original_compute,
            "sum_zone_productivities": original_zone_prod,
            "sum_electricity": original_elec,
            "sum_gas": original_gas,
            **originals,
        },
    }


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
_INSTALLED: dict = {}


def install(sim_root) -> dict:
    """Install the bookkeeping replacements (idempotent: re-patches in place)."""
    _INSTALLED.clear()
    _INSTALLED["zone_stats"] = install_zone_stats(sim_root)
    _INSTALLED["reward"] = install_reward(sim_root)
    _INSTALLED["patched"] = _INSTALLED["zone_stats"]["patched"] + _INSTALLED[
        "reward"
    ]["patched"]
    return _INSTALLED


# --------------------------------------------------------------------------- #
# shadow verification (cenv.shadow drives this)
# --------------------------------------------------------------------------- #
def _elec_cost_c(cost, start_time, end_time, energy_rate):
    price, _ = _energy_rates(cost, start_time)
    dt = (end_time - start_time).total_seconds()
    return price * abs(energy_rate) * dt


def _elec_carbon_c(cost, start_time, end_time, energy_rate):
    _, carbon = _energy_rates(cost, start_time)
    dt = (end_time - start_time).total_seconds()
    return carbon * abs(energy_rate) * dt


def _gas_cost_c(cost, start_time, end_time, energy_rate):
    rate = 0.0 if energy_rate < 0.0 else energy_rate
    price, _ = _energy_rates(cost, start_time)
    return price * (rate * (end_time - start_time).total_seconds())


def _gas_carbon_c(cost, start_time, end_time, energy_rate):
    rate = 0.0 if energy_rate < 0.0 else energy_rate
    _, carbon = _energy_rates(cost, start_time)
    return carbon * (rate * (end_time - start_time).total_seconds())


def shadow_pairs(shadow, sim_root):
    """Register this component's Python-vs-C pairs on the loaded simulator."""
    from smart_buildings.smart_control.reward import (
        base_setpoint_energy_carbon_reward as base_reward,
    )
    from smart_buildings.smart_control.reward import (
        electricity_energy_cost as elec_mod,
    )
    from smart_buildings.smart_control.reward import (
        natural_gas_energy_cost as gas_mod,
    )
    from smart_buildings.smart_control.reward import (
        setpoint_energy_carbon_regret as regret_mod,
    )
    from smart_buildings.smart_control.simulator import building as building_mod

    cls = building_mod.FloorPlanBasedBuilding
    shadow.pair(cls, "get_zone_temp_stats", _stats_c, kind="tuple")
    shadow.pair(cls, "get_zone_average_temps", _average_c, kind="dict")

    regret_cls = regret_mod.SetpointEnergyCarbonRegretFunction
    base_cls = base_reward.BaseSetpointEnergyCarbonRewardFunction
    shadow.pair(regret_cls, "compute_reward",
                lambda self, info: _compute_reward_c(self, info), kind="proto")
    shadow.pair(base_cls, "_sum_zone_productivities", _zone_productivities_c,
                kind="tuple")
    shadow.pair(base_cls, "_sum_electricity_energy_rate",
                _electricity_rate_c, kind="float")
    shadow.pair(base_cls, "_sum_natural_gas_energy_rate", _gas_rate_c,
                kind="float")
    for name, fn in (
        ("cost", _elec_cost_c),
        ("carbon", _elec_carbon_c),
    ):
        shadow.pair(elec_mod.ElectricityEnergyCost, name, fn, kind="float",
                    label=f"ElectricityEnergyCost.{name}")
    for name, fn in (
        ("cost", _gas_cost_c),
        ("carbon", _gas_carbon_c),
    ):
        shadow.pair(gas_mod.NaturalGasEnergyCost, name, fn, kind="float",
                    label=f"NaturalGasEnergyCost.{name}")


def _stats_c(building, zone_name):
    """C version of ``get_zone_temp_stats`` (shadow: original runs too)."""
    from smart_buildings.smart_control.simulator import constants

    plan = _plan(building, constants)
    zone = plan.index.get(zone_name)
    if zone is None:
        raise ValueError("Zone name is not present in room_dict.")
    n = plan.sel_for(zone)
    temp = building.temp
    rc = lib.cenv_bk_zone_stats(
        plan.handle, temp, temp.size, plan.sel1, n,
        plan.buf_min.ctypes.data, plan.buf_max.ctypes.data,
        plan.buf_mean.ctypes.data, None,
    )
    if rc != 0:
        raise RuntimeError(f"cenv_bk_zone_stats failed ({rc})")
    return plan.buf_min[0], plan.buf_max[0], plan.buf_mean[0]


def _average_c(building):
    """C version of ``get_zone_average_temps``."""
    from smart_buildings.smart_control.simulator import constants

    plan = _plan(building, constants)
    temp = building.temp
    rc = lib.cenv_bk_zone_stats(
        plan.handle, temp, temp.size, plan.phys_sel, plan.phys_sel.size,
        None, None, plan.means.ctypes.data, None,
    )
    if rc != 0:
        raise RuntimeError(f"cenv_bk_zone_stats failed ({rc})")
    return dict(zip(plan.phys_names, plan.means.tolist()))


def _zone_productivities_c(reward_function, energy_reward_info):
    from smart_buildings.smart_control.utils import conversion_utils

    start = conversion_utils.proto_to_pandas_timestamp(
        energy_reward_info.start_timestamp
    )
    end = conversion_utils.proto_to_pandas_timestamp(
        energy_reward_info.end_timestamp
    )
    delta = (end - start).total_seconds()
    productivity = 0.0
    occupancy = 0.0
    for info in energy_reward_info.zone_reward_infos.values():
        occupancy += info.average_occupancy
        productivity += lib.cenv_bk_zone_productivity(
            info.heating_setpoint_temperature,
            info.cooling_setpoint_temperature,
            info.zone_air_temperature,
            delta,
            info.average_occupancy,
            reward_function._max_productivity_personhour_usd,
            reward_function._productivity_deviation_ref,
        )
    return productivity, occupancy


def _electricity_rate_c(reward_function, energy_reward_info):
    rate = 0.0
    for info in energy_reward_info.air_handler_reward_infos.values():
        rate += info.blower_electrical_energy_rate + abs(
            info.air_conditioning_electrical_energy_rate
        )
    for info in energy_reward_info.boiler_reward_infos.values():
        rate += info.pump_electrical_energy_rate
    for info in energy_reward_info.heat_pump_reward_infos.values():
        rate += abs(info.pump_electrical_energy_rate)
        rate += abs(info.electricity_heating_energy_rate)
    return rate


def _gas_rate_c(reward_function, energy_reward_info):
    rate = 0.0
    for info in energy_reward_info.boiler_reward_infos.values():
        rate += info.natural_gas_heating_energy_rate
    return rate

