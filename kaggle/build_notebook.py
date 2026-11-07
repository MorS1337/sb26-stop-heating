#!/usr/bin/env python3
"""Build a Kaggle submission notebook from one of our policy modules.

The Kaggle rerun has no internet, no attachments and no import path into this
repository, so everything the policy needs must live in the notebook itself.
This script generates a self-contained .ipynb that

  1. mounts the competition's own ``kaggle_evaluation`` package from the data
     mount,
  2. materialises ``<policy>.py`` plus every sibling module it imports (from the
     same directory) into ``/kaggle/working`` and puts that directory on
     ``sys.path`` -- the sources are embedded byte-for-byte and verified against
     their SHA-256, so the notebook runs exactly the code that was tested here,
  3. wraps the policy in ``SubmissionPolicy``, a
     ``smart_buildings_inference_server.Policy`` that clips every action into the
     scenario's own action space, replaces NaN/inf, fixes dtypes and shapes, and
     falls back to a safe mid-range action if the policy ever raises (one
     exception would otherwise void the whole submission), and
  4. runs the inference server exactly like the official starter notebook.

Usage:
    python kaggle/build_notebook.py \
        --policy policies/ctrl_v1.py:CtrlV1 \
        --slug sbx-ctrl-v1 \
        --out kaggle/build/sbx-ctrl-v1

The generated directory contains ``submission.ipynb`` and
``kernel-metadata.json`` (private, CPU, internet off, competition attached) and
is what ``kaggle/push.sh`` uploads.

Notes
-----
* ``--param k=v`` bakes an override into ``PolicyImpl.PARAMS`` without editing
  the policy file (values are parsed as JSON, falling back to a raw string).
* The policy and its siblings must not import ``smart_buildings``: it is not
  importable from the notebook container's own environment. The builder refuses
  to generate a notebook that does.
* ``--smoke-steps N`` is for pipeline tests only: it truncates every episode
  through the gateway's host smoke-test variable so a Kaggle-side test finishes
  in minutes. The markdown cell records it and ``submit.sh`` refuses to submit a
  build that contains it.
* Local testing: the notebook reads ``SB_COMPETITION_DATA_FOLDER`` when set, so
  ``kaggle/run_local.py`` can point it at ``comp_data/``. On Kaggle the variable
  is absent and the notebook finds the mount itself: Kaggle has mounted the same
  competition at ``/kaggle/input/competitions/<slug>/`` and at ``/kaggle/input/<slug>/``.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import sys
import textwrap
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPETITION_DATA_FOLDER = "/kaggle/input/competitions/neurips-2026-smart-buildings/"
COMPETITION_SLUG = "neurips-2026-smart-buildings"
KAGGLE_USER = "mors1337"

# Modules the notebook container provides. Anything else the policy imports from
# a sibling file gets embedded too; anything else is a build error.
FORBIDDEN_RUNTIME_IMPORTS = ("smart_buildings",)


class BuildError(Exception):
    """A policy cannot be packaged as-is."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_policy_spec(spec: str) -> tuple[Path, str | None]:
    """Split ``path/to/mod.py:ClassName`` and resolve the path."""
    path_part, _, class_name = spec.partition(":")
    path = Path(path_part)
    if not path.is_absolute():
        candidate = REPO_ROOT / path
        path = candidate if candidate.exists() else path
    if not path.is_file():
        raise BuildError(f"policy file not found: {path_part!r} (looked in {path})")
    return path.resolve(), (class_name or None)


def _read_imports(tree: ast.AST) -> set[str]:
    """Top-level module names imported anywhere in the module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (``from . import x``) are resolved against the
            # policy's own directory by the collector, not here.
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def _resolve_class(module_path: Path, tree: ast.Module, class_name: str | None) -> str:
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    if class_name:
        if class_name not in classes:
            raise BuildError(f"{module_path.name} defines no class {class_name!r}; found {classes}")
        return class_name

    # No class given: accept exactly one class that exposes get_action.
    candidates = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(item, ast.FunctionDef) and item.name == "get_action"
            for item in node.body
        )
    ]
    if len(candidates) != 1:
        raise BuildError(
            f"cannot pick a policy class from {module_path.name}: candidates={candidates}; pass path.py:ClassName"
        )
    return candidates[0]


def collect_modules(module_path: Path) -> dict[str, bytes]:
    """Return ``{module_name: source_bytes}`` for the policy and its siblings.

    Only modules that resolve to a ``.py`` file next to the policy are embedded:
    numpy/scipy/json come from the container. Transitive sibling imports (a
    helper importing another helper) are followed.
    """
    package_dir = module_path.parent
    collected: dict[str, bytes] = {}

    def visit(path: Path) -> None:
        name = path.stem
        if name in collected:
            return
        if name == "__init__":
            raise BuildError(f"package-style policy layouts are not supported: {path}")
        source = path.read_bytes()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            raise BuildError(f"{path} does not parse: {exc}") from exc

        imported = _read_imports(tree)
        for forbidden in FORBIDDEN_RUNTIME_IMPORTS:
            if forbidden in imported:
                raise BuildError(
                    f"{path.name} imports {forbidden!r}; the Kaggle notebook has no such module. "
                    "Keep the policy self-contained (numpy/scipy only)."
                )
        collected[name] = source

        for candidate in sorted(imported):
            sibling = package_dir / f"{candidate}.py"
            if sibling.is_file() and candidate != name:
                visit(sibling)

    visit(module_path)
    return collected


def _parse_params(raw: list[str]) -> dict:
    params: dict = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise BuildError(f"--param expects KEY=VALUE, got {item!r}")
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value
    return params


def _module_cell(policy_module: str, policy_class: str, sources: dict[str, bytes], params: dict) -> str:
    """Code cell that mounts the embedded modules and imports the policy class."""
    # zlib before base64: the kernels API rejects a notebook over ~1 MiB with a bare
    # "400 Bad Request" (sbx-ctrl-w4 at 1,091,744 bytes; sbx-ctrl-w at 1,043,678 passed),
    # and wu_weights.py alone is most of that. The digest is still of the raw source.
    embedded = {
        name: base64.b64encode(zlib.compress(source, 9)).decode("ascii")
        for name, source in sources.items()
    }
    digests = {name: _sha256(source) for name, source in sources.items()}
    return textwrap.dedent(
        '''\
        """Set up the competition imports and the team policy modules.

        Generated by kaggle/build_notebook.py; edit the source policy, not this cell.
        """
        import base64
        import zlib
        import hashlib
        import importlib
        import os
        import sys
        from pathlib import Path

        # The competition mount. Kaggle has used two layouts for the same kernel:
        # /kaggle/input/competitions/<slug>/ (sbx-ctrl-w, 2026-11-05) and
        # /kaggle/input/<slug>/ (sbx-ctrl-w4 v1/v2, 2026-11-06), so try both, then any
        # /kaggle/input/**/kaggle_evaluation of this competition, waiting up to 300 s
        # for a late mount. SB_COMPETITION_DATA_FOLDER (set only by
        # kaggle/run_local.py) points the same notebook at a local copy of the data.
        import time as _time
        _override = os.environ.get("SB_COMPETITION_DATA_FOLDER")
        _candidates = [_override] if _override else [
            %(competition_data_folder)r,
            %(competition_data_folder_alt)r,
        ]
        _candidates = [c if c.endswith("/") else c + "/" for c in _candidates]
        COMPETITION_DATA_FOLDER = None
        for _attempt in range(30):
            COMPETITION_DATA_FOLDER = next(
                (c for c in _candidates if Path(c + "kaggle_evaluation").is_dir()), None
            )
            if COMPETITION_DATA_FOLDER is None and not _override and Path("/kaggle/input").is_dir():
                _found = sorted(
                    p for p in list(Path("/kaggle/input").glob("*/kaggle_evaluation"))
                    + list(Path("/kaggle/input").glob("*/*/kaggle_evaluation"))
                    if %(competition_slug)r in str(p)
                )
                if _found:
                    COMPETITION_DATA_FOLDER = str(_found[0].parent) + "/"
            if COMPETITION_DATA_FOLDER is not None:
                break
            _time.sleep(10)
        if COMPETITION_DATA_FOLDER is None:
            _listing = sorted(
                str(p) for p in Path("/kaggle/input").glob("*/*") if p.is_dir()
            )[:40] if Path("/kaggle/input").is_dir() else []
            raise RuntimeError(
                f"kaggle_evaluation not found under {_candidates} after 300 s; "
                f"/kaggle/input holds {_listing}"
            )
        print("Competition data folder:", COMPETITION_DATA_FOLDER, flush=True)

        # The inference server package lives on the data mount, not in the image.
        # Both the mount root and the package directory go on the path: the root
        # makes ``kaggle_evaluation`` itself importable, and the package directory
        # is where the driver embeds ``remote_game_drivers``.
        EVALUATION_FOLDER = COMPETITION_DATA_FOLDER + "kaggle_evaluation"
        for _path in (EVALUATION_FOLDER, COMPETITION_DATA_FOLDER):
            if _path not in sys.path:
                sys.path.insert(0, _path)

        # Policy modules, embedded verbatim and verified before use. On Kaggle the
        # working directory is /kaggle/working (writable); locally it is a temp dir.
        MODULE_SOURCES = %(sources)r
        MODULE_SHA256 = %(digests)r
        MODULE_DIR = Path(
            os.environ.get("SB_MODULE_DIR")
            or ("/kaggle/working/sbx_policy_modules" if Path("/kaggle/working").is_dir() else "")
            or (Path.cwd() / "sbx_policy_modules")
        )
        MODULE_DIR.mkdir(parents=True, exist_ok=True)
        for _name, _payload in MODULE_SOURCES.items():
            _source = zlib.decompress(base64.b64decode(_payload)).decode("utf-8")
            _digest = hashlib.sha256(_source.encode("utf-8")).hexdigest()
            if _digest != MODULE_SHA256[_name]:
                raise RuntimeError(f"embedded source of {_name} does not match its digest")
            (MODULE_DIR / f"{_name}.py").write_text(_source, encoding="utf-8")
        if str(MODULE_DIR) not in sys.path:
            sys.path.insert(0, str(MODULE_DIR))
        print("Policy modules:", sorted(MODULE_SOURCES), "->", MODULE_DIR, flush=True)

        _policy_module = importlib.import_module(%(policy_module)r)
        PolicyImpl = getattr(_policy_module, %(policy_class)r)
        if not callable(getattr(PolicyImpl, "get_action", None)):
            raise RuntimeError(f"{%(policy_class)r} has no get_action method")
        print("Policy class:", PolicyImpl.__module__ + "." + PolicyImpl.__qualname__, flush=True)

        # Optional parameter overrides baked in at build time (empty by default).
        POLICY_PARAMS_OVERRIDES = %(params)r
        if POLICY_PARAMS_OVERRIDES:
            PolicyImpl.PARAMS = {**getattr(PolicyImpl, "PARAMS", {}), **POLICY_PARAMS_OVERRIDES}
            print("Policy PARAMS overrides:", POLICY_PARAMS_OVERRIDES, flush=True)
        '''
    ) % {
        "competition_data_folder": DEFAULT_COMPETITION_DATA_FOLDER,
        "competition_data_folder_alt": f"/kaggle/input/{COMPETITION_SLUG}/",
        "competition_slug": COMPETITION_SLUG,
        "sources": embedded,
        "digests": digests,
        "policy_module": policy_module,
        "policy_class": policy_class,
        "params": params,
    }


def _wrapper_cell(smoke_steps: int | None = None) -> str:
    """Code cell that wraps the policy and runs the inference server.

    ``smoke_steps`` is only ever set for pipeline test builds: it truncates every
    episode through the gateway's host smoke-test variable so the kernel finishes
    in minutes instead of hours.
    """
    cell = textwrap.dedent(
        '''\
        """Serve the team policy to the competition gateway.

        One SubmissionPolicy is built per scenario by the gateway (a fresh instance
        arrives with each scenario's metadata). The wrapper exists only to make an
        invalid action impossible: the gateway rejects out-of-range values, and a
        NaN setpoint is not caught by the simulator's bounds check at all, either of
        which invalidates the whole submission rather than one step.
        """
        import os
        import traceback

        import numpy as np
        from kaggle_evaluation import smart_buildings_inference_server

        __SMOKE_BLOCK__


        def _to_float32(values, shape, low, high, name):
            """Coerce a continuous slot into a bounded float32 array of the expected shape."""
            array = np.asarray(values, dtype=np.float64)
            if array.shape != shape:
                raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(array, low, high).astype(np.float32)


        def _to_int32(values, shape, low, high, name):
            """Coerce a discrete slot into a bounded int32 array of the expected shape."""
            array = np.asarray(values, dtype=np.float64)
            if array.shape != shape:
                raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(np.rint(array), low, high).astype(np.int32)


        class SubmissionPolicy(smart_buildings_inference_server.Policy):
            """Delegates to the team policy and enforces the action contract."""

            def _init_spec(self, action_spec):
                continuous = action_spec["continuous_action"]
                discrete = action_spec["discrete_action"]
                self._c_shape = tuple(continuous.shape)
                self._c_low = np.asarray(continuous.low, dtype=np.float64)
                self._c_high = np.asarray(continuous.high, dtype=np.float64)
                self._d_shape = tuple(discrete.shape)
                self._d_low = np.asarray(discrete.low, dtype=np.float64)
                self._d_high = np.asarray(discrete.high, dtype=np.float64)
                # Mid-range everywhere, every device on: the reference policy, and a
                # valid action for any building because it is clipped into the spec.
                self._safe_action = {
                    "continuous_action": _to_float32(
                        np.zeros(self._c_shape), self._c_shape, self._c_low, self._c_high, "continuous_action"
                    ),
                    "discrete_action": _to_int32(
                        np.ones(self._d_shape), self._d_shape, self._d_low, self._d_high, "discrete_action"
                    ),
                }

            def __init__(self, building_meta_data, action_spec):
                super().__init__(building_meta_data, action_spec)
                self._init_spec(action_spec)
                self._failures = 0
                self._impl = PolicyImpl(building_meta_data, action_spec)

            def get_action(self, obs):
                try:
                    action = self._impl.get_action(obs)
                    return {
                        "continuous_action": _to_float32(
                            action["continuous_action"], self._c_shape, self._c_low, self._c_high, "continuous_action"
                        ),
                        "discrete_action": _to_int32(
                            action["discrete_action"], self._d_shape, self._d_low, self._d_high, "discrete_action"
                        ),
                    }
                except Exception:
                    self._failures += 1
                    if self._failures == 1:
                        # Loud once, then quiet: a traceback per 60 s turn would drown
                        # the log, and the fallback keeps the run scoreable.
                        traceback.print_exc()
                        print(
                            "[submission] policy raised; holding the mid-range action for the rest of this scenario",
                            flush=True,
                        )
                    return {
                        "continuous_action": self._safe_action["continuous_action"].copy(),
                        "discrete_action": self._safe_action["discrete_action"].copy(),
                    }


        smart_buildings_inference_server.SmartBuildingsInferenceServer(SubmissionPolicy).run(
            competition_data_folder=COMPETITION_DATA_FOLDER,
        )
        '''
    )
    if smoke_steps:
        # Only a test build carries this text, so the absence of "TEST BUILD" in a
        # built notebook is what kaggle/submit.sh checks before submitting.
        block = (
            "# --- TEST BUILD ONLY -------------------------------------------------\n"
            "# Truncates every episode through the gateway's host smoke-test variable so a\n"
            "# pipeline test finishes in minutes. Never submit a build that has this.\n"
            f"_SMOKE_STEPS = {smoke_steps}\n"
            "if _SMOKE_STEPS:\n"
            '    os.environ.setdefault("HEARTH_HOST_SMART_BUILDINGS_MAX_STEPS", str(_SMOKE_STEPS))\n'
            '    print(f"[submission] TEST BUILD: episodes truncated to {_SMOKE_STEPS} steps", flush=True)\n'
            "# --- end test build --------------------------------------------------"
        )
    else:
        block = (
            "# Nothing to configure: a real submission runs each scenario to the length\n"
            "# its own config.gin specifies."
        )
    return cell.replace("__SMOKE_BLOCK__", block)


def _markdown_cell(
    policy_path: Path, policy_class: str, sources: dict[str, bytes], slug: str, smoke_steps: int | None = None
) -> str:
    lines = [
        f"# {slug}",
        "",
        "Generated by `kaggle/build_notebook.py`. Do not edit in the Kaggle UI -- rebuild instead.",
        "",
        f"- policy: `{policy_path.name}` -> class `{policy_class}`",
        f"- embedded modules: {', '.join(f'`{name}.py`' for name in sorted(sources))}",
        "- runtime: CPU only, internet off; actions are clipped into each scenario's action space",
    ]
    if smoke_steps:
        lines.append(f"- **TEST BUILD**: episodes truncated to {smoke_steps} steps; never submit this version")
    lines += [
        "",
        "```",
    ]
    for name in sorted(sources):
        lines.append(f"{_sha256(sources[name])}  {name}.py")
    lines += ["```", ""]
    return "\n".join(lines)


def build(
    policy_spec: str,
    slug: str,
    out_dir: Path,
    params: dict,
    title: str | None,
    smoke_steps: int | None = None,
) -> Path:
    policy_path, class_name = _parse_policy_spec(policy_spec)
    tree = ast.parse(policy_path.read_bytes(), filename=str(policy_path))
    resolved_class = _resolve_class(policy_path, tree, class_name)
    sources = collect_modules(policy_path)

    notebook_cells = [
        {
            "cell_type": "markdown",
            "id": "overview",
            "metadata": {},
            "source": _markdown_cell(policy_path, resolved_class, sources, slug, smoke_steps).splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "id": "mount-policy",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _module_cell(policy_path.stem, resolved_class, sources, params).splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "id": "serve-policy",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _wrapper_cell(smoke_steps).splitlines(keepends=True),
        },
    ]
    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": f"{KAGGLE_USER}/{slug}",
        "title": title or slug,
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [COMPETITION_SLUG],
        "model_sources": [],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "submission.ipynb").write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="path/to/module.py:ClassName (relative to the repo root)")
    parser.add_argument("--slug", required=True, help="Kaggle kernel slug, e.g. sbx-ctrl-v1")
    parser.add_argument("--out", default=None, help="output directory (default: kaggle/build/<slug>)")
    parser.add_argument("--title", default=None, help="kernel title (default: the slug)")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="policy PARAMS override")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=None,
        help=(
            "TEST BUILDS ONLY: truncate every episode to this many steps so a Kaggle-side smoke test "
            "finishes in minutes. Never build a submitted version with this set."
        ),
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out) if args.out else REPO_ROOT / "kaggle" / "build" / args.slug
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    try:
        build(args.policy, args.slug, out_dir, _parse_params(args.param), args.title, args.smoke_steps)
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    if args.smoke_steps:
        print(f"TEST BUILD: episodes truncated to {args.smoke_steps} steps -- do not submit this version")
    print(f"wrote {out_dir / 'submission.ipynb'}")
    print(f"wrote {out_dir / 'kernel-metadata.json'}")
    print(f"next: kaggle/push.sh {args.slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
