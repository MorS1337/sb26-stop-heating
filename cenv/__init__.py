"""cenv -- C ports of the Smart Buildings simulator's per-step work.

Each component is a drop-in replacement for a specific Python function, applied
at runtime by monkeypatching the class it lives on (never by editing
``sims/`` or ``starter-kit/``). A component is switched on by
``sbx/evalrun.py --cenv <name>`` and is only trusted after its shadow gate
(``cenv/verify/shadow.py``) reports agreement with the Python original.

Components:

* ``bookkeeping`` -- zone temperature statistics (``Building.get_zone_temp_stats``,
  ``Building.get_zone_average_temps``) and the reward arithmetic
  (``SetpointEnergyCarbonRegretFunction.compute_reward`` and the productivity /
  cost / carbon pieces it calls).
* ``observation`` -- ``Environment._get_observation`` assembled directly into the
  float32 vector (no ObservationResponse -> normalised copy -> dict -> array
  round-trip).
* ``fused`` -- the whole ``env.step`` as one C call (sim step, observation and
  the reward-info numbers), reusing the components above instead of
  re-deriving anything. Supersedes ``observation`` and makes ``thermal``,
  ``hvac``, ``bookkeeping`` and ``stochastic`` unnecessary (it calls their C
  entry points directly), but installs ``bookkeeping`` for the reward
  arithmetic. See ``cenv/fused.py`` for the step map, the limits and the gate.
* ``thermal``, ``stochastic`` -- peers; they register their own ``cenv/src/*.c``
  and ``cenv/<name>.py`` and need no edit here.

``batch`` is not a component: ``cenv/batch.py`` is a second environment class
(M environments of one scenario, one C call and one batched solve per step), so
it is opted into by constructing a ``BatchEnv`` rather than by ``--cenv``.

A component module sets ``GATE_PASSED = True`` once its shadow gate passes in
this checkout. ``--cenv all`` enables exactly those; naming a component
explicitly opts in regardless, which is what a gate run does.
"""

from __future__ import annotations

import importlib
from importlib.util import find_spec

# Modules under cenv/ that are infrastructure, not installable components.
# ``batch`` is a second environment class (cenv/batch.py) rather than a runtime
# patch of one function, so it is opted into by constructing it, not by
# ``--cenv batch``.
_INFRA = frozenset({"__init__", "build", "bindings", "shadow", "batch"})

_INSTALLED: dict = {}

__all__ = ["COMPONENTS", "component_modules", "install", "uninstall", "available"]


def _sim_root(env=None):
    """The simulator tree root, derived from an imported module."""
    import sys

    mod = sys.modules.get("smart_buildings")
    if mod is None or not getattr(mod, "__file__", None):
        raise RuntimeError(
            "import smart_buildings (sbx.simload.use_sim) before installing cenv"
        )
    from pathlib import Path

    return Path(mod.__file__).resolve().parent.parent


def component_modules() -> list[str]:
    """Every installable component module in cenv/, discovered on disk.

    Discovery rather than a hardcoded tuple, so a peer adding
    ``cenv/<name>.py`` (with ``install(sim_root)`` and ``shadow_pairs``) is
    picked up by ``--cenv <name>`` and ``--cenv all`` with no edit here. Uses
    file names only and never imports, so the parent process can validate a
    ``--cenv`` list before any worker has selected a simulator tree.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent
    return sorted(
        p.stem for p in root.glob("*.py") if p.stem not in _INFRA
    )


COMPONENTS = tuple(component_modules())


def available(all_passed_only: bool = False) -> list[str]:
    """Component names present in this checkout (optionally gate-passed only)."""
    names = []
    for name in component_modules():
        if find_spec(f"cenv.{name}") is None:
            continue
        if all_passed_only:
            module = importlib.import_module(f"cenv.{name}")
            if not getattr(module, "GATE_PASSED", False):
                continue
        names.append(name)
    return names


def uninstall(components) -> list[str]:
    """Restore the Python originals for installed components (A/B measurement).

    Returns the list of restored attribute names. Safe to call when nothing is
    installed.
    """
    if isinstance(components, str):
        components = [c.strip() for c in components.split(",") if c.strip()]
    restored = []
    for name in components:
        entry = _INSTALLED.get(name)
        if not entry:
            continue
        for target, attr, original in entry.get("restore", []):
            setattr(target, attr, original)
            restored.append(f"{getattr(target, '__name__', target)}.{attr}")
        _INSTALLED.pop(name, None)
    return restored


def install(components, sim_root=None, verbose: bool = False) -> dict:
    """Install the named components (``all`` = every gate-passed one).

    Returns a summary dict ``{component: install-result}``. Each component's
    ``install(sim_root)`` is idempotent and raises when its simulator version
    does not match the one it was written against.
    """
    if isinstance(components, str):
        components = [c.strip() for c in components.split(",") if c.strip()]
    names: list[str] = []
    for name in components:
        if name == "all":
            present = available()
            passed = available(all_passed_only=True)
            skipped = [c for c in present if c not in passed]
            if skipped and verbose:
                print(f"cenv: --cenv all skips gate-unproven {', '.join(skipped)}")
            names.extend(passed)
        elif name == "none":
            continue
        else:
            names.append(name)
    root = sim_root if sim_root is not None else _sim_root()
    done = {}
    for name in dict.fromkeys(names):
        if name in _INSTALLED:
            done[name] = _INSTALLED[name]
            continue
        module = importlib.import_module(f"cenv.{name}")
        result = module.install(root)
        _INSTALLED[name] = result
        done[name] = result
        if verbose:
            print(f"cenv: installed {name}: {result.get('patched')}")
    return done


def live() -> list[str]:
    """Components currently installed (patched, not merely available)."""
    return list(_INSTALLED)
