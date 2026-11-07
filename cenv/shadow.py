"""Shadow verification: run the C implementation beside the Python original.

A *shadow run* is an ordinary episode in which every registered replacement is
executed **twice** on the same inputs -- once as the Python original, once in C
-- and the two results are compared. The value the environment receives is
always the original's, so the shadow run's trajectory is the official one; the
C code is a passenger and cannot move the episode.

Registering a component
-----------------------
A component module exposes ``shadow_pairs(shadow, sim_root)`` which calls

    shadow.pair(target, "method_name", c_fn, kind="float", label="...")

``c_fn`` receives exactly the arguments the Python method received (including
``self``), so the C side sees the same inputs; it may also close over whatever
state the component already prepared. Supported ``kind`` values:

``float``     one float (abs/rel difference)
``tuple``     a tuple/list of floats (compared elementwise)
``array``     a numpy array (compared elementwise)
``dict``      a mapping of label -> float
``proto``     a protobuf message (compares the named float fields)
``exact``     anything with ``==`` (counts mismatches only)

Command line
------------
    python -m cenv.shadow --component bookkeeping --scenario building_train_8/fairbanks_8 \
        --steps 200 --policy policies.ctrl_w:CtrlW

prints, per registered pair, the call count, the max absolute and relative
difference, the first mismatching step and whether the tolerance passed.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]

ABS_TOL = 0.0        # a component gate is relative-only by default ...
REL_TOL = 1e-12      # ... at the ticket's tolerance
EXACT_REL = 0.0      # rel_tol that means "bit-identical"


class Metric:
    """Accumulated difference statistics for one compared quantity."""

    __slots__ = ("label", "kind", "calls", "n_bad", "max_abs", "max_rel",
                 "max_abs_step", "first_bad_step", "first_bad_detail",
                 "exact_calls", "errors", "container")

    def __init__(self, label: str, kind: str):
        # A container (tuple/dict/proto) owns no value of its own: it reports its
        # children's comparisons, so its own status only tracks calls/errors.
        self.container = kind in ("tuple", "dict", "proto")
        self.label = label
        self.kind = kind
        self.calls = 0
        self.n_bad = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.max_abs_step = -1
        self.first_bad_step = -1
        self.first_bad_detail = ""
        self.exact_calls = 0
        self.errors = 0

    def add(self, step, py, c, abs_tol=ABS_TOL, rel_tol=REL_TOL) -> None:
        self.calls += 1
        if py is None or c is None:
            return
        diff = float(c) - float(py)
        adiff = abs(diff)
        if diff == 0.0:
            self.exact_calls += 1
            return
        scale = max(abs(float(py)), abs(float(c)))
        rdiff = adiff / scale if scale > 0.0 else math.inf
        if adiff > self.max_abs:
            self.max_abs = adiff
            self.max_abs_step = step
        if rdiff > self.max_rel:
            self.max_rel = rdiff
        ok = adiff == 0.0 or rdiff <= rel_tol
        if not ok and (abs_tol == 0.0 or adiff > abs_tol):
            self.n_bad += 1
            if self.first_bad_step < 0:
                self.first_bad_step = step
                self.first_bad_detail = f"py={float(py)!r} c={float(c)!r}"

    def as_dict(self) -> dict:
        # A pair that never ran is inert, not proven: report it as skipped so a
        # silent 0-call registration cannot masquerade as a passing gate.
        passed = (self.calls > 0 and self.n_bad == 0 and self.errors == 0)
        return {
            "label": self.label,
            "kind": self.kind,
            "calls": self.calls,
            "n_bad": self.n_bad,
            "errors": self.errors,
            "exact_fraction": (self.exact_calls / self.calls) if self.calls else 0.0,
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "max_abs_step": self.max_abs_step,
            "first_bad_step": self.first_bad_step,
            "first_bad_detail": self.first_bad_detail,
            "passed": passed,
            "skipped": self.calls == 0,
        }


class Shadow:
    """Registry of Python-vs-C pairs plus their accumulated differences."""

    def __init__(self, component: str, abs_tol: float = ABS_TOL,
                 rel_tol: float = REL_TOL, verbose: bool = False):
        self.component = component
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.verbose = verbose
        self.step = 0
        self.metrics: dict[str, Metric] = {}
        self.pairs: list[tuple] = []

    # -- registration ------------------------------------------------------ #
    def pair(self, target, name: str, c_fn, kind: str = "float",
             label: str | None = None, shadow_attr: str | None = None):
        """Wrap ``target.name`` so the C function runs beside the original."""
        original = getattr(target, name)
        label = label or f"{_owning_name(target)}.{name}"
        metric = self.metrics.setdefault(label, Metric(label, kind))
        shadow = self

        def replacement(*args, **kwargs):
            py = original(*args, **kwargs)
            try:
                c = c_fn(*args, **kwargs)
            except Exception:  # a C-side crash must not kill the shadow run
                metric.errors += 1
                if shadow.verbose or metric.errors == 1:
                    print(f"[shadow] {label}: C side raised\n"
                          + traceback.format_exc(), file=sys.stderr)
                return py
            shadow.compare(label, py, c, kind)
            return py

        setattr(target, name, replacement)
        self.pairs.append((label, target, name, original))
        return original

    # -- comparison -------------------------------------------------------- #
    def compare(self, label: str, py, c, kind: str = "float") -> None:
        metric = self.metrics.setdefault(label, Metric(label, kind))
        step = self.step
        if kind == "float":
            metric.add(step, py, c, self.abs_tol, self.rel_tol)
        elif kind == "tuple":
            metric.calls += 1
            metric.exact_calls += int(tuple(map(float, py)) == tuple(map(float, c)))
            for i, (a, b) in enumerate(zip(py, c)):
                sub = f"{label}[{i}]"
                self.metrics.setdefault(sub, Metric(sub, "float")).add(
                    step, a, b, self.abs_tol, self.rel_tol
                )
        elif kind == "array":
            a = np.asarray(py, dtype=np.float64).ravel()
            b = np.asarray(c, dtype=np.float64).ravel()
            if a.shape != b.shape:
                metric.errors += 1
                metric.first_bad_step = step if metric.first_bad_step < 0 else metric.first_bad_step
                metric.first_bad_detail = f"shape {a.shape} vs {b.shape}"
                return
            if np.array_equal(a, b):
                metric.calls += 1
                metric.exact_calls += 1
                return
            d = np.abs(a - b)
            scale = np.maximum(np.abs(a), np.abs(b))
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(scale > 0.0, d / scale, np.inf)
            # Attribute the worst element to the step, then fold in the summary.
            i = int(np.argmax(d))
            metric.add(step, a[i], b[i], self.abs_tol, self.rel_tol)
            bad = (d > self.abs_tol) & (rel > self.rel_tol) & (d != 0.0)
            metric.n_bad += int(bad.sum())
            metric.max_abs = max(metric.max_abs, float(d.max()))
            metric.max_rel = max(metric.max_rel, float(np.nanmax(rel)))
        elif kind == "dict":
            metric.calls += 1
            metric.exact_calls += int(dict(py) == dict(c))
            keys = set(py) | set(c)
            for key in sorted(keys):
                sub = f"{label}.{key}"
                self.metrics.setdefault(sub, Metric(sub, "float")).add(
                    step, py.get(key), c.get(key), self.abs_tol, self.rel_tol
                )
        elif kind == "proto":
            # py/c are protobuf messages; compare every float/double field.
            metric.calls += 1
            for field in py.DESCRIPTOR.fields:
                if field.type not in (1, 2):  # DOUBLE, FLOAT
                    continue
                sub = f"{label}.{field.name}"
                self.metrics.setdefault(sub, Metric(sub, "float")).add(
                    step, getattr(py, field.name), getattr(c, field.name),
                    self.abs_tol, self.rel_tol,
                )
        elif kind == "exact":
            metric.calls += 1
            if py == c:
                metric.exact_calls += 1
            else:
                metric.n_bad += 1
                if metric.first_bad_step < 0:
                    metric.first_bad_step = step
                    metric.first_bad_detail = f"{py!r} != {c!r}"
        else:
            raise ValueError(f"unknown comparison kind {kind!r}")

    # -- reporting --------------------------------------------------------- #
    def report(self, print_table: bool = True) -> dict:
        rows = [m.as_dict() for m in self.metrics.values()]
        passed = all(r["passed"] for r in rows)
        if print_table:
            width = max((len(r["label"]) for r in rows), default=10)
            print(f"[shadow {self.component}] {len(rows)} compared quantities, "
                  f"rel_tol={self.rel_tol:g} abs_tol={self.abs_tol:g}")
            print(f"{'quantity'.ljust(width)}  calls   exact%     max_abs     "
                  f"max_rel   first_bad  status")
            for r in rows:
                status = ("SKIP" if r["skipped"]
                          else ("PASS" if r["passed"] else "FAIL"))
                print(f"{r['label'].ljust(width)}  {r['calls']:6d}  "
                      f"{100 * r['exact_fraction']:6.1f}  {r['max_abs']:11.3e}  "
                      f"{r['max_rel']:9.2e}  {r['first_bad_step']:9d}  {status}")
                if not r["passed"]:
                    print(f"    {r['label']}: {r['n_bad']} bad, first at step "
                          f"{r['first_bad_step']}: {r['first_bad_detail']}")
            print(f"[shadow {self.component}] {'PASS' if passed else 'FAIL'}")
        return {"component": self.component, "passed": passed, "rows": rows}


def _owning_name(target) -> str:
    if isinstance(target, type):
        return target.__name__
    return type(target).__name__


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(sim: str, scenario: str, steps: int, policy_spec: str,
        components: list[str], params: dict | None = None,
        abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL,
        print_table: bool = True, verbose: bool = False) -> dict:
    """Run one episode in shadow mode for the named components.

    Returns ``{"scenario":..., "components": {name: report}}``.
    The episode itself is the official one: the C code never feeds back.
    """
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    from cenv.verify.probe import build_env
    from sbx.evalrun import _resolve_policy

    env, evaluate = build_env(sim, scenario)
    sim_root = WORKSPACE / "sims" / sim

    shadows: dict[str, Shadow] = {}
    import importlib

    for name in components:
        module = importlib.import_module(f"cenv.{name}")
        hooks = getattr(module, "shadow_pairs", None)
        if hooks is None:
            raise SystemExit(f"cenv.{name} defines no shadow_pairs(shadow, sim_root)")
        shadow = Shadow(name, abs_tol=abs_tol, rel_tol=rel_tol, verbose=verbose)
        hooks(shadow, sim_root)
        shadows[name] = shadow

    # Policy: the same resolver sbx.evalrun uses.
    policy_cls, _label = _resolve_policy(policy_spec, evaluate)
    if params:
        policy_cls.PARAMS = {**(getattr(policy_cls, "PARAMS", None) or {}), **params}

    meta = evaluate.create_building_meta_data(
        env, (sim_root / "smart_buildings" / "training_scenarios" / scenario)
    )
    action_spec = env.action_spec()
    policy = evaluate.initialize_policy(policy_cls, meta, action_spec)

    obs, _ = env.reset()
    for i in range(int(steps)):
        for shadow in shadows.values():
            shadow.step = i
        action = evaluate.validate_action(
            evaluate.get_policy_action(policy, obs), action_spec
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    reports = {name: s.report(print_table=print_table) for name, s in shadows.items()}
    return {"scenario": scenario, "steps": int(steps), "components": reports}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--component", action="append", default=None,
                    help="component module under cenv/ (repeatable); default bookkeeping")
    ap.add_argument("--scenario", default="building_train_8/fairbanks_8")
    ap.add_argument("--sim", default="shipped")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--policy", default="policies.ctrl_w:CtrlW")
    ap.add_argument("--params", default=None, help="JSON policy PARAMS overrides")
    ap.add_argument("--abs-tol", type=float, default=ABS_TOL)
    ap.add_argument("--rel-tol", type=float, default=REL_TOL)
    ap.add_argument("--json", default=None, help="write the report here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    import json

    params = json.loads(args.params) if args.params else None
    components = args.component or ["bookkeeping"]
    result = run(args.sim, args.scenario, args.steps, args.policy, components,
                 params=params, abs_tol=args.abs_tol, rel_tol=args.rel_tol,
                 verbose=args.verbose)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    ok = all(r["passed"] for r in result["components"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
