"""Named scenario sets, resolved to ``config.gin`` paths inside a simulator tree.

A scenario is one (building, weather strip) pair; it lives at

    sims/<sim>/smart_buildings/training_scenarios/<building>/<weather>/config.gin

and is named ``<building>/<weather>`` everywhere in this workspace.

Sets
----
``dev30``            the 30 shipped pairs, read from ``training_set.yaml`` in the
                     selected tree (so it cannot drift from the competition set).
``dev30_<station>``  the same set filtered to one weather station, e.g.
                     ``dev30_fairbanks``; ``dev30_<weather>`` (e.g. ``dev30_phoenix_6``)
                     filters to a single strip; several strips may be separated
                     by commas: ``dev30_phoenix_6,singapore_1``.
``gen_train``        64 generated scenarios: 16 generated buildings x 4 strips
                     drawn from the five training stations at new start dates.
``gen_hold``         32 generated scenarios: 8 other generated buildings x strips
                     from minneapolis (unused by the shipped set) and from the
                     five training stations at unseen start dates.
``ext_hold``         one scenario per *external* climate station (17 stations
                     contributed by ``sbx/weather_ext.py``), round-robin over the
                     eight ``gen_hold`` buildings plus two shipped ones. Needs
                     those presets installed in both trees; ``make_pool`` skips
                     the set with a warning if they are absent.
``gen_all``          ``gen_train`` + ``gen_hold`` + ``ext_hold``.
``<path>``           a ``config.gin`` path, a scenario directory, or a list of
                     either, is passed through.

The generated sets come from ``sbx/pools/manifest.json``, written by
``python -m sbx.make_pool``.

Usage::

    from sbx import scenarios
    ids = scenarios.list_ids("dev30", sim="corrected")       # ['building_train_1/atlanta_2', ...]
    paths = scenarios.config_paths("dev30_fairbanks", sim="shipped")
"""

from __future__ import annotations

import json
from pathlib import Path

from sbx.simload import WORKSPACE, sim_root

TRAIN_STATIONS = ("fairbanks", "denver", "atlanta", "phoenix", "singapore")
HOLD_STATION = "minneapolis"
ALL_STATIONS = TRAIN_STATIONS + (HOLD_STATION,)

MANIFEST_PATH = WORKSPACE / "sbx" / "pools" / "manifest.json"

SCENARIOS_SUBDIR = Path("smart_buildings") / "training_scenarios"


# --------------------------------------------------------------------------- #
# scenario ids -> paths
# --------------------------------------------------------------------------- #
def training_set(sim: str) -> list[str]:
    """The shipped pairs, in ``training_set.yaml`` order."""
    import yaml

    path = sim_root(sim) / "smart_buildings" / "training_set.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return [f"{s['building']}/{s['weather']}" for s in data["scenarios"]]


def _manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} is missing; run `python -m sbx.make_pool` first"
        )
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _filter(ids: list[str], spec: str) -> list[str]:
    """Filters ids by station (``fairbanks``) or strip (``phoenix_6``)."""
    wanted = [w.strip() for w in spec.split(",") if w.strip()]
    if not wanted:
        return ids
    out: list[str] = []
    for scenario in ids:
        weather = scenario.split("/", 1)[1]
        station = weather.rsplit("_", 1)[0]
        for w in wanted:
            if weather == w or (w in ALL_STATIONS and station == w):
                out.append(scenario)
                break
    return out


def list_ids(name: str, sim: str = "shipped") -> list[str]:
    """Expands a set name to scenario ids (``building/weather``)."""
    if name == "dev30":
        return training_set(sim)
    if name.startswith("dev30_"):
        return _filter(training_set(sim), name[len("dev30_"):])
    sets = _manifest()["sets"]
    if name in sets:  # gen_train, gen_hold, ext_hold, ...
        return list(sets[name])
    if name == "gen_all":
        return [s for values in sets.values() for s in values]
    raise KeyError(
        f"unknown scenario set {name!r}; use dev30, dev30_<station>, gen_train, "
        f"gen_hold, ext_hold, gen_all ({sorted(sets)}), or pass config paths directly"
    )


def config_path(sim: str, scenario: str) -> Path:
    """``building/weather`` -> the config.gin path inside the tree."""
    path = sim_root(sim) / SCENARIOS_SUBDIR / scenario / "config.gin"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; the {sim} tree has no scenario {scenario!r}"
        )
    return path


def _is_set_name(name: str) -> bool:
    """True for a set this module resolves (dev30*, gen_all, manifest sets)."""
    if name == "dev30" or name.startswith("dev30_"):
        return True
    if name == "gen_all":
        return True
    try:
        return name in _manifest()["sets"]
    except FileNotFoundError:
        return False


def config_paths(name_or_paths, sim: str = "shipped") -> list[Path]:
    """Resolves set names, scenario ids, dirs and CONFIG paths to gin paths."""
    if isinstance(name_or_paths, (str, Path)):
        items = [name_or_paths]
    else:
        items = list(name_or_paths)

    out: list[Path] = []
    for item in items:
        path = Path(item)
        if isinstance(item, str) and not path.exists() and _is_set_name(item):
            out.extend(config_path(sim, s) for s in list_ids(item, sim))
        elif path.exists() and path.is_dir():
            out.append(path / "config.gin")
        elif path.exists():
            out.append(path)
        elif "/" in str(item):
            out.append(config_path(sim, str(item)))
        else:
            raise FileNotFoundError(f"cannot resolve scenario {item!r}")
    for path in out:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing")
    return out


def scenario_id(config_path_: Path) -> str:
    """``.../training_scenarios/<building>/<weather>/config.gin`` -> id."""
    config_path_ = Path(config_path_)
    if config_path_.name == "config.gin":
        return f"{config_path_.parent.parent.name}/{config_path_.parent.name}"
    return config_path_.stem


def set_names() -> list[str]:
    """Every named set this module can resolve (for CLI help/validation)."""
    names = ["dev30", "gen_all"]
    names += [f"dev30_{s}" for s in ALL_STATIONS]
    if MANIFEST_PATH.exists():
        names += list(_manifest()["sets"])
    return names
