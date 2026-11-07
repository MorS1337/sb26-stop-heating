"""CMA-ES tuning of a controller's global ``PARAMS`` over a building distribution.

The controller (``policies/ctrl_v1.py:CtrlV1``) exposes one flat ``PARAMS`` dict
that is shared by every building, so tuning is a black-box problem over that
vector only.  This module searches it with CMA-ES in a normalized ``[0, 1]^d``
space (one dimension per entry of a JSON space file), evaluates every candidate
on a fixed stratified subset of scenarios, and writes everything it learns under
``results/tune/<tag>/``.  Each (candidate, scenario) pair is one spawn process
with ``maxtasksperchild=1``, reusing :mod:`sbx.evalrun`'s worker, so a scenario
never inherits numba/gin state from another one.

Commands (workspace root, starter-kit venv)
------------------------------------------
Tune the 15-knob core space on a stratified 8-scenario subset of ``gen_train``::

    starter-kit/.venv/bin/python -m sbx.tune \\
        --policy policies.ctrl_v1:CtrlV1 --space sbx/spaces/ctrl_v1_core.json \\
        --set gen_train --subset 8 --subset-seed 0 --steps 2880 --sim corrected \\
        --popsize 10 --workers 7 --generations 30 --tag tune_v1

Continue the same run where it stopped (``--generations`` is the *total*)::

    starter-kit/.venv/bin/python -m sbx.tune ...same flags... --resume

Measure generalization of the winner on the held-out set, standard runner::

    starter-kit/.venv/bin/python -m sbx.tune \\
        --policy policies.ctrl_v1:CtrlV1 --validate results/tune/tune_v1/best.json \\
        --set gen_hold --steps 8640 --sim corrected --workers 7 --tag tune_v1_holdout

The same campaign on Kaggle kernels (no local CPU; needs `kaggle/remote/`) --
each generation is one wave of shard kernels, and the payload hash is pinned in
``state.json`` so every generation runs the shipped code::

    starter-kit/.venv/bin/python -m sbx.tune \\
        --policy policies.ctrl_v1:CtrlV1 --space sbx/spaces/ctrl_v1_core.json \\
        --set gen_train --subset 8 --steps 2880 --sim corrected \\
        --popsize 10 --generations 30 --tag tune_v1 \\
        --backend kaggle --shards 5 --shard-workers 4

``--workers`` stays the local pool size (backend ``local``); ``--shards`` is how
many kernels a generation uses (5 sessions run concurrently), ``--shard-workers``
the processes inside each kernel, and ``--wave-timeout`` bounds one generation.
A shard that dies or times out is re-pushed once and then its jobs score
``PENALTY``; ``--resume`` works as for the local backend, and refuses to continue
if ``sbx/``/``policies/`` moved since the campaign pinned its payload (pass
``--repin`` to publish the new code).

Smoke test (minutes)::

    starter-kit/.venv/bin/python -m sbx.tune --set gen_train --subset 2 \\
        --steps 144 --popsize 4 --generations 2 --workers 2 --tag tune_demo

Note that a 144-step horizon (12 h) starts on the strips' Sunday 06:00 and is
therefore unoccupied: every reward is exactly 0 and CMA sees no gradient.  Add
``--start-offset-days 1`` for a weekday horizon, or use ``--steps 2880`` (30
days) for real runs::

    starter-kit/.venv/bin/python -m sbx.tune --set gen_train --subset 2 \\
        --steps 144 --start-offset-days 1 --popsize 4 --generations 2 \\
        --workers 2 --tag tune_demo_wd

Outputs under ``results/tune/<tag>/``
------------------------------------
``evals.jsonl``  one line per evaluated candidate: gen, idx, kind, decoded
                 params, normalized x, per-scenario mean per-step reward and
                 score, objective, wall seconds.  Appended; on ``--resume`` the
                 rows of the interrupted (incomplete) generation are dropped so
                 a generation never appears twice.
``es.pkl``       pickled ``cma.CMAEvolutionStrategy`` (its own rng included), so
                 ``--resume`` re-asks exactly the next population.
``state.json``   run identity: space signature, resolved space, set, subset ids,
                 steps/sim/components, seed, popsize, generations done.
``best.json``    best params + objective so far (plus the gen-0 baseline
                 reference), consumable by ``--validate``.
``trace.csv``    gen, cumulative best, best gen, median fitness, sigma, baseline
                 objective (gen 0), generation wall seconds.
``records/``     the raw per-scenario JSON records of every evaluation.

Failure policy: a scenario that raises, dies, times out (``--job-timeout``) or
returns a non-finite reward scores ``PENALTY`` (1.0, larger than any real
leaderboard score we have seen) and is logged in the eval row.  Nothing a single
scenario does can kill the run.  Ctrl-C saves ``es.pkl``/``state.json``/
``trace.csv`` and exits 130, leaving the run resumable.

Objective (lower is better) = mean over the fixed subset of
``|mean per-step reward|``, i.e. exactly the leaderboard metric restricted to
the subset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import pickle
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from sbx import evalrun  # noqa: E402
from sbx import scenarios as scenario_sets  # noqa: E402

TUNE_ROOT = WORKSPACE / "results" / "tune"
SCALES = ("lin", "log", "int", "bool")
DEFAULT_SIGMA0 = 0.25  # fraction of a decade... of the [0, 1] range
PENALTY = 1.0
TRACE_HEADER = ("gen", "best", "best_gen", "median", "sigma", "baseline", "wall_s")
SIZE_BUCKETS = 3


# --------------------------------------------------------------------------- #
# parameter space
# --------------------------------------------------------------------------- #
def _encode(entry: dict[str, Any], value: Any) -> float:
    """Value -> position in [0, 1] for one space entry."""
    scale, lo, hi = entry["scale"], entry["lo"], entry["hi"]
    if scale == "bool":
        return 1.0 if value else 0.0
    v = min(max(float(value), lo), hi)
    if scale == "log":
        return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (v - lo) / (hi - lo)


def _decode(entry: dict[str, Any], value: float) -> Any:
    """Position in [0, 1] -> value (ints rounded, bools thresholded at 0.5)."""
    x = min(1.0, max(0.0, float(value)))
    scale, lo, hi = entry["scale"], entry["lo"], entry["hi"]
    if scale == "bool":
        return bool(x >= 0.5)
    if scale == "log":
        raw = math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo)))
    else:
        raw = lo + x * (hi - lo)
    if scale == "int":
        return int(round(raw))
    return round(float(raw), 6)


class Space:
    """A policy-PARAMS subset with a normalized ``[0, 1]^dim`` encoding."""

    def __init__(self, entries: list[dict[str, Any]], sigma0: float = DEFAULT_SIGMA0):
        self.entries = entries
        self.names = [e["name"] for e in entries]
        self.dim = len(entries)
        self.sigma0 = float(sigma0)

    @property
    def x0(self) -> np.ndarray:
        """Normalized incumbent: the policy's live defaults for these names."""
        return np.array([_encode(e, e["init"]) for e in self.entries], dtype=float)

    def decode(self, x: Sequence[float]) -> dict[str, Any]:
        values = np.asarray(x, dtype=float).ravel()
        if values.size != self.dim:
            raise ValueError(f"expected {self.dim} coordinates, got {values.size}")
        return {e["name"]: _decode(e, values[i]) for i, e in enumerate(self.entries)}

    def signature(self) -> str:
        payload = json.dumps(
            [[e["name"], e["init"], e["lo"], e["hi"], e["scale"]] for e in self.entries]
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def describe(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self.entries]


def load_space(path: str | Path, defaults: dict[str, Any]) -> Space:
    """Reads a space JSON file; ``init`` falls back to the policy's live default."""
    raw = json.loads(Path(path).read_text())
    sigma0 = DEFAULT_SIGMA0
    if isinstance(raw, dict):
        sigma0 = float(raw.get("sigma0", DEFAULT_SIGMA0))
        items = raw.get("params") or raw.get("space") or []
    else:
        items = raw
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a non-empty list of parameter entries")

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        name = str(item["name"])
        if name in seen:
            raise ValueError(f"{path}: duplicate parameter {name!r}")
        seen.add(name)
        if name not in defaults:
            raise KeyError(
                f"{path}: {name!r} is not a PARAMS key of the policy "
                f"(have {len(defaults)} keys, e.g. {sorted(defaults)[:6]})"
            )
        scale = str(item.get("scale", "lin"))
        if scale not in SCALES:
            raise ValueError(f"{path}: {name}: unknown scale {scale!r}")
        lo, hi = float(item["lo"]), float(item["hi"])
        if not hi > lo:
            raise ValueError(f"{path}: {name}: hi ({hi}) must exceed lo ({lo})")
        if scale == "log" and lo <= 0.0:
            raise ValueError(f"{path}: {name}: log scale needs lo > 0, got {lo}")

        init = item.get("init", defaults[name])
        if init is None:
            raise ValueError(f"{path}: {name} has no init and the policy default is None")
        if scale == "bool":
            init = bool(init)
        else:
            init = float(init)
            if not lo <= init <= hi:
                clamped = min(max(init, lo), hi)
                print(
                    f"[tune] WARNING {name}: default {init} outside [{lo}, {hi}], "
                    f"clamping init to {clamped}; widen the range",
                    file=sys.stderr,
                    flush=True,
                )
                init = clamped
        entries.append(
            {"name": name, "lo": lo, "hi": hi, "scale": scale, "init": init}
        )
    return Space(entries, sigma0=sigma0)


# --------------------------------------------------------------------------- #
# stratified subset
# --------------------------------------------------------------------------- #
def _station_of(weather: str, strips: dict[str, str]) -> str:
    """``fairbanks_g2`` / ``atlanta_2`` / ``anchorage_1`` -> climate station."""
    if weather in strips:
        return strips[weather]
    return re.sub(r"_\d+$", "", re.sub(r"_g\d+$", "", weather)) or weather


def _floor_volume(config_path: Path) -> float:
    """Fallback size metric: floor-plan volume (width x height x floors)."""
    import yaml  # cheap, and only needed when the manifest lacks the building

    path = Path(config_path)
    yaml_path = path.with_name("config.yaml")
    if yaml_path.exists():
        path = yaml_path
    config = yaml.safe_load(path.read_text())
    plan = config.get("floor_plan") or {}
    return float(plan.get("width", 0.0)) * float(plan.get("height", 0.0)) * float(
        plan.get("num_floors", 0.0)
    )


def _manifest_index() -> tuple[dict[str, str], dict[str, int]]:
    """(strip name -> station, building name -> zone count) from the pool manifest."""
    path = scenario_sets.MANIFEST_PATH
    if not path.exists():
        return {}, {}
    manifest = json.loads(path.read_text())
    strips = {s["name"]: s["station"] for s in manifest.get("strips", [])}
    zones = {
        b["name"]: int(b["num_zones"])
        for b in manifest.get("buildings", [])
        if b.get("num_zones") is not None
    }
    return strips, zones


def stratified_subset(
    ids: Sequence[str], sim: str, count: int, seed: int
) -> tuple[list[str], dict[str, Any]]:
    """Deterministic subset, stratified by climate station and building size.

    Rounds of round-robin over (station, size-tertile) strata, so a small subset
    still spans climates and building sizes.  Size is the manifest zone count
    when every building in the set is known there, else the floor-plan volume;
    either way it is split into tertiles of *this* set.
    """
    ids = list(dict.fromkeys(ids))
    if count <= 0 or count >= len(ids):
        return ids, {"strata": {}, "size_metric": "none", "count": len(ids)}

    strips, zones = _manifest_index()
    known = [zones.get(sid.split("/", 1)[0]) for sid in ids]
    if all(z is not None for z in known):
        metric = dict(zip(ids, known))
        metric_name = "num_zones"
    else:
        metric = {
            sid: _floor_volume(scenario_sets.config_path(sim, sid)) for sid in ids
        }
        metric_name = "floor_volume"

    values = sorted(metric.values())
    cuts = np.percentile(values, [100.0 / SIZE_BUCKETS * k for k in range(1, SIZE_BUCKETS)])
    buckets = {
        sid: int(np.searchsorted(cuts, metric[sid], side="right")) for sid in ids
    }

    groups: dict[tuple[str, int], list[str]] = {}
    for sid in sorted(ids):
        building, weather = sid.split("/", 1)
        groups.setdefault((_station_of(weather, strips), buckets[sid]), []).append(sid)

    rng = random.Random(f"{seed}|{sim}")
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])

    picked: list[str] = []
    cursor = {key: 0 for key in keys}
    while len(picked) < count:
        progressed = False
        for key in keys:
            index = cursor[key]
            if index < len(groups[key]):
                picked.append(groups[key][index])
                cursor[key] = index + 1
                progressed = True
                if len(picked) >= count:
                    break
        if not progressed:
            break
    strata = {f"{k[0]}|size{k[1]}": len(v) for k, v in sorted(groups.items())}
    return picked, {"strata": strata, "size_metric": metric_name, "count": len(picked)}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _job_name(gen: int, idx: int, scenario: str) -> str:
    tag = "b" if idx < 0 else f"{idx:03d}"
    return f"g{gen:03d}_c{tag}__{scenario.replace('/', '__')}.json"


def _crash(job: str, task: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "scenario": task["scenario"],
        "sim": task["sim"],
        "config_path": task["config_path"],
        "record_name": job,
        "error": message,
    }


def _result_job_name(result: dict[str, Any]) -> str | None:
    if result.get("record_path"):
        return Path(result["record_path"]).name
    return result.get("record_name")


def _evaluate_jobs(
    tasks: list[dict[str, Any]],
    workers: int,
    job_timeout: float,
    worker: Any = None,
) -> dict[str, dict[str, Any]]:
    """Runs every task once (one spawn process each) and returns job -> result.

    A hard-crashed child leaves its task unresolved; ``job_timeout`` bounds the
    wait for the next result so a wedged scenario costs a penalty instead of the
    whole run.  KeyboardInterrupt is propagated (the caller saves and exits).
    ``worker`` overrides the per-scenario entry point (fault-injection tests).
    """
    run_one = worker or evalrun._worker  # noqa: SLF001 - shared evalrun worker
    results: dict[str, dict[str, Any]] = {}
    pending = {task["record_name"]: task for task in tasks}
    if not pending:
        return results

    if workers <= 1:
        for job, task in list(pending.items()):
            try:
                results[job] = run_one(task)
            except Exception:  # noqa: BLE001 - a broken eval is a penalty
                results[job] = _crash(job, task, traceback.format_exc())
        return results

    context = mp.get_context("spawn")
    pool = context.Pool(processes=min(workers, len(pending)), maxtasksperchild=1)
    try:
        iterator = pool.imap_unordered(
            run_one, list(pending.values()), chunksize=1
        )
        while pending:
            try:
                if hasattr(iterator, "next"):  # IMapUnorderedIterator.next(timeout)
                    result = iterator.next(job_timeout)
                else:  # pragma: no cover - defensive
                    result = next(iterator)
            except mp.TimeoutError:
                for job, task in pending.items():
                    results[job] = _crash(
                        job, task, f"job timed out after {job_timeout:.0f}s"
                    )
                pending.clear()
                break
            except StopIteration:
                for job, task in pending.items():
                    results[job] = _crash(job, task, "worker pool ended without a result")
                pending.clear()
                break
            except Exception:  # noqa: BLE001 - pool died; penalize what is left
                detail = traceback.format_exc()
                for job, task in pending.items():
                    results[job] = _crash(job, task, f"pool failure:\n{detail}")
                pending.clear()
                break
            job = _result_job_name(result)
            if job is None:  # pragma: no cover - defensive
                continue
            results[job] = result
            pending.pop(job, None)
    finally:
        pool.terminate()
        pool.join()
    return results


def _batch_groups(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Regroups (candidate, scenario) tasks into one task per scenario.

    The local backend runs one process per (candidate, scenario); a
    ``--backend batch`` run evaluates a whole generation's candidates against
    one scenario in *one* process, so the population becomes the M of
    ``cenv.batch`` and the processes are scenarios (that is the parallelism
    left: one matrix per scenario, M right-hand sides per step).
    """
    groups: dict[tuple, dict[str, Any]] = {}
    for task in tasks:
        key = (
            task["sim"],
            task["scenario"],
            task["config_path"],
            int(task["steps"]),
            int(task["start_offset_days"]),
            task["components"],
            task["policy"],
            task["tag"],
        )
        group = groups.get(key)
        if group is None:
            group = {
                "sim": task["sim"],
                "scenario": task["scenario"],
                "config_path": task["config_path"],
                "steps": int(task["steps"]),
                "start_offset_days": int(task["start_offset_days"]),
                "components": task["components"],
                "policy": task["policy"],
                "tag": task["tag"],
                "out_dir": task["out_dir"],
                "record_names": [],
                "params": [],
            }
            groups[key] = group
        group["record_names"].append(task["record_name"])
        group["params"].append(dict(task["params"]))
    return list(groups.values())


def batch_worker(group: dict[str, Any]) -> dict[str, Any]:
    """One scenario, the whole population: a single ``cenv.batch`` run.

    Returns ``{record_name: record}`` so the caller can score exactly as it
    scores the local backend's per-job results. A failure marks every record of
    the group, which is what ``_score_candidate`` turns into a penalty.
    """
    try:
        from cenv.batch import run_batch  # noqa: PLC0415 - optional component

        config_path = Path(group["config_path"])
        out_dir = Path(group["out_dir"])
        if group["start_offset_days"]:
            config_path = evalrun._shifted_config(  # noqa: SLF001
                config_path, int(group["start_offset_days"]), out_dir
            )
        result = run_batch(
            group["scenario"],
            group["sim"],
            [(group["policy"], params) for params in group["params"]],
            int(group["steps"]),
            config_path=config_path,
            components=group["components"],
            tag=group["tag"],
        )
        records = {}
        for name, record in zip(group["record_names"], result["records"]):
            record = dict(record)
            record["record_name"] = name
            record["batch_timing"] = result["timing"]
            records[name] = record
            path = out_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2, default=str))
        return records
    except Exception:  # noqa: BLE001 - a broken scenario is a penalty
        detail = traceback.format_exc()
        return {
            name: {
                "scenario": group["scenario"],
                "sim": group["sim"],
                "record_name": name,
                "error": detail,
            }
            for name in group["record_names"]
        }


def _evaluate_jobs_batch(
    tasks: list[dict[str, Any]], workers: int, job_timeout: float
) -> dict[str, dict[str, Any]]:
    """Runs one batch process per scenario and flattens the records.

    ``workers`` bounds how many scenarios run at once (each batch process is
    one environment set, so this is the machine's core budget, exactly as
    ``--workers`` is for the local backend).
    """
    groups = _batch_groups(tasks)
    results: dict[str, dict[str, Any]] = {}
    if not groups:
        return results
    if workers <= 1 or len(groups) == 1:
        for group in groups:
            results.update(batch_worker(group))
        return results
    context = mp.get_context("spawn")
    pool = context.Pool(processes=min(workers, len(groups)), maxtasksperchild=1)
    try:
        iterator = pool.imap_unordered(batch_worker, groups, chunksize=1)
        for _ in range(len(groups)):
            try:
                if hasattr(iterator, "next"):
                    records = iterator.next(job_timeout)
                else:  # pragma: no cover - defensive
                    records = next(iterator)
            except mp.TimeoutError:
                break
            results.update(records)
    finally:
        pool.terminate()
        pool.join()
    return results


def _check_payload_pin(args: argparse.Namespace, state: dict[str, Any]) -> None:
    """Refuses a campaign whose pinned payload no longer matches the sources.

    The kernels run whatever the pinned dataset contains; if ``sbx/`` or
    ``policies/`` moved since the first wave, generations would silently mix
    code. ``--repin`` publishes a new version and clears the pin (the campaign
    restarts its payload lineage, which the state records).
    """
    from kaggle.remote import payload as payload_mod  # noqa: PLC0415

    pinned = state.get("payload_hash")
    if not pinned:
        return
    current = payload_mod.build_payload()["hash"]
    if current == pinned:
        print(f"[tune] payload pinned at {pinned} (sources unchanged)", flush=True)
        return
    if not args.repin:
        raise SystemExit(
            f"--resume: payload hash changed since this campaign started "
            f"({pinned} -> {current}); the kernels would run different code. "
            "Pass --repin to publish the new payload (previous generations keep "
            "their own pin in evals.jsonl/state history) or start a new --tag"
        )
    print(
        f"[tune] --repin: payload {pinned} -> {current}; publishing a new version",
        flush=True,
    )
    state["payload_history"] = list(state.get("payload_history") or []) + [pinned]
    state["payload_hash"] = None


def _evaluate_jobs_remote(
    tasks: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    state: dict[str, Any],
    gen: int,
    run_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Runs one generation's tasks on Kaggle; returns ``job_name -> result``.

    The task dicts are byte-identical to the local backend's (same
    ``record_name``, same ``params``), so ``_score_candidate`` cannot tell the
    two apart: ``sbx.evalrun._worker`` inside the kernel applies
    ``PolicyCls.PARAMS`` exactly as a local spawn process would.

    One generation is one wave: all tasks are split by zone count into
    ``--shards`` kernels, each running ``--shard-workers`` processes, and pushed
    together under ``--max-concurrency``. The payload hash is pinned in
    ``state.json`` on the first wave so later waves -- and ``--resume`` -- run
    the same code even if ``policies/`` changed meanwhile.
    """
    from kaggle.remote import remote_eval  # noqa: PLC0415 - local backend must not need it

    wave_tag = f"{args.tag}_g{gen:03d}"
    wave_dir = run_dir / "remote" / wave_tag
    records: dict[str, dict[str, Any]] = {}
    try:
        handle = remote_eval.submit_tasks(
            tasks,
            tag=wave_tag,
            policy=args.policy,
            steps=int(args.steps),
            sim=args.sim,
            components=args.components,
            start_offset_days=int(args.start_offset_days),
            shards=int(args.shards),
            workers=int(args.shard_workers),
            job_timeout=float(args.job_timeout),
            out=wave_dir,
            payload_hash=state.get("payload_hash"),
            max_concurrency=int(args.max_concurrency),
            cenv=args.cenv,
        )
        state["payload_hash"] = handle["payload_hash"]
        state["dataset_version"] = handle.get("dataset_version")
        if state.get("payload_hash") and not state.get("payload_pinned_utc"):
            state["payload_pinned_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        result = remote_eval.collect(
            handle,
            timeout=float(args.wave_timeout),
            poll_seconds=20.0,
            out=wave_dir,
            allow_partial=True,
            retries=1,
        )
        records = dict(result.get("task_results", {}).get(wave_tag, {}))
        poll = handle.get("poll", {})
        print(
            f"[tune] gen {gen} kaggle: {len(handle['kernels'])} kernel(s), "
            f"{len(records)}/{len(tasks)} records, "
            f"max_concurrent={poll.get('max_concurrent_running')} "
            f"wall={poll.get('wall_seconds')}s",
            flush=True,
        )
    except Exception:  # noqa: BLE001 - a dead wave must not end the campaign
        print(
            f"[tune] gen {gen} kaggle wave FAILED; penalising its "
            f"{len(tasks)} jobs:\n{traceback.format_exc()}",
            flush=True,
        )
        return {}

    # Keep records/ meaningful for both backends.
    for name, record in records.items():
        _write_json(run_dir / "records" / name, record)
    missing = [task["record_name"] for task in tasks if task["record_name"] not in records]
    if missing:
        print(
            f"[tune] gen {gen} kaggle: {len(missing)} job(s) missing after one "
            f"re-push -> penalised (e.g. {missing[0]})",
            flush=True,
        )
    return records


def _build_tasks(
    gen: int,
    candidates: Sequence[dict[str, Any]],
    subset: Sequence[str],
    sim: str,
    steps: int,
    components: str,
    policy: str,
    tag: str,
    out_dir: Path,
    start_offset_days: int,
    fast: bool = False,
    cenv: Sequence[str] = (),
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for candidate in candidates:
        for scenario in subset:
            tasks.append(
                {
                    "sim": sim,
                    "scenario": scenario,
                    "config_path": str(
                        scenario_sets.config_path(sim, scenario).resolve()
                    ),
                    "record_name": _job_name(gen, candidate["idx"], scenario),
                    "out_dir": str(out_dir),
                    "policy": policy,
                    "params": dict(candidate["params"]),
                    "steps": int(steps),
                    "start_offset_days": int(start_offset_days),
                    "components": components,
                    "fast": bool(fast),
                    "cenv": list(cenv) or None,
                    "tag": tag,
                }
            )
    return tasks


def _score_candidate(
    candidate: dict[str, Any],
    results: dict[str, dict[str, Any]],
    subset: Sequence[str],
    gen: int,
    finished_at: dict[str, float],
    started: float,
) -> dict[str, Any]:
    """Turns the raw job results of one candidate into an eval row."""
    per_scenario: dict[str, Any] = {}
    for scenario in subset:
        job = _job_name(gen, candidate["idx"], scenario)
        result = results.get(job)
        if result is None:
            entry = {
                "mean_per_step_reward": None,
                "score": PENALTY,
                "steps": 0,
                "wall_s": None,
                "error": "no result (job lost)",
            }
        elif "error" in result:
            entry = {
                "mean_per_step_reward": None,
                "score": PENALTY,
                "steps": 0,
                "wall_s": None,
                "error": str(result["error"]).strip().splitlines()[-1]
                if str(result["error"]).strip()
                else "error",
            }
        else:
            mean = float(result["mean_per_step_reward"])
            if not math.isfinite(mean):
                entry = {
                    "mean_per_step_reward": None,
                    "score": PENALTY,
                    "steps": int(result.get("steps", 0)),
                    "wall_s": float(result.get("wall_seconds", 0.0)),
                    "error": f"non-finite mean_per_step_reward {mean!r}",
                }
            else:
                entry = {
                    "mean_per_step_reward": mean,
                    "score": abs(mean),
                    "steps": int(result.get("steps", 0)),
                    "wall_s": float(result.get("wall_seconds", 0.0)),
                    "error": None,
                }
        per_scenario[scenario] = entry

    scores = np.array([v["score"] for v in per_scenario.values()], dtype=float)
    objective = float(scores.mean()) if scores.size else PENALTY
    if not math.isfinite(objective):
        objective = PENALTY
    stamps = [
        finished_at[_job_name(gen, candidate["idx"], s)]
        for s in subset
        if _job_name(gen, candidate["idx"], s) in finished_at
    ]
    return {
        "gen": gen,
        "idx": int(candidate["idx"]),
        "kind": candidate["kind"],
        "objective": objective,
        "n_scenarios": len(subset),
        "n_failed": int(sum(1 for v in per_scenario.values() if v["error"])),
        "wall_s": round((max(stamps) - started) if stamps else 0.0, 3),
        "params": candidate["params"],
        "x": [round(float(v), 6) for v in np.asarray(candidate["x"]).ravel()],
        "per_scenario": per_scenario,
    }


# --------------------------------------------------------------------------- #
# run-state io
# --------------------------------------------------------------------------- #
def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def _truncate_jsonl(path: Path, keep_gen: int) -> None:
    """Drops rows of ``gen >= keep_gen`` (the generation that was interrupted)."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # half-written last line
            continue
        if int(row.get("gen", 0)) < keep_gen:
            kept.append(json.dumps(row, default=str))
    path.write_text("\n".join(kept) + ("\n" if kept else ""))


def _write_trace(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    lines = [",".join(TRACE_HEADER)]
    for row in rows:
        cells = []
        for key in TRACE_HEADER:
            value = row.get(key)
            cells.append("" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append(",".join(cells))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        cells = line.split(",")
        row: dict[str, Any] = {}
        for key, cell in zip(TRACE_HEADER, cells):
            if cell == "":
                row[key] = None
            elif key == "gen" or key == "best_gen":
                row[key] = int(float(cell))
            else:
                row[key] = float(cell)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# policy defaults
# --------------------------------------------------------------------------- #
def resolve_policy_defaults(spec: str) -> dict[str, Any]:
    """The policy class's *current* ``PARAMS`` (so space ``init`` tracks edits)."""
    if ":" not in spec:
        raise ValueError(
            f"--policy must be 'module:Class' for tuning (got {spec!r}); "
            "baselines have no PARAMS dict"
        )
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    params = getattr(cls, "PARAMS", None) or getattr(module, "DEFAULTS", None)
    if not params:
        raise ValueError(f"{spec} exposes neither PARAMS nor DEFAULTS")
    return dict(params)


# --------------------------------------------------------------------------- #
# tuning
# --------------------------------------------------------------------------- #
def _tune(args: argparse.Namespace) -> int:
    import cma  # imported lazily so --validate works without pycma

    if args.fast and args.backend == "batch":
        raise SystemExit(
            "--fast patches the Python step; --backend batch is the C kernel "
            "already, so the flag has no meaning there (drop one of the two)"
        )
    if args.cenv and args.backend == "batch":
        raise SystemExit(
            "--cenv installs C components into the Python step; --backend batch is "
            "the C kernel already (drop one of the two)"
        )

    defaults = resolve_policy_defaults(args.policy)
    space = load_space(args.space, defaults)

    set_names = list(args.sets or ["gen_train"])
    set_ids: list[str] = []
    for name in set_names:
        set_ids.extend(scenario_sets.list_ids(name, args.sim))
    set_ids = list(dict.fromkeys(set_ids))
    if not set_ids:
        raise ValueError(f"sets {set_names} resolved to no scenarios")

    run_dir = TUNE_ROOT / args.tag
    es_path = run_dir / "es.pkl"
    state_path = run_dir / "state.json"
    evals_path = run_dir / "evals.jsonl"
    trace_path = run_dir / "trace.csv"
    best_path = run_dir / "best.json"
    records_dir = run_dir / "records"

    resuming = bool(args.resume)
    if resuming:
        for required in (es_path, state_path):
            if not required.exists():
                raise SystemExit(f"--resume: {required} is missing")
        state = json.loads(state_path.read_text())
        if state.get("space_signature") != space.signature():
            raise SystemExit(
                "--resume: the space no longer matches state.json "
                f"({state.get('space_signature')} != {space.signature()}); "
                "use a new --tag for a changed space"
            )
        for key, current in (
            ("policy", args.policy),
            ("sim", args.sim),
            ("steps", int(args.steps)),
            ("components", args.components),
            ("fast", bool(args.fast)),
            ("cenv", args.cenv or None),
            ("seed", int(args.seed)),
            ("start_offset_days", int(args.start_offset_days)),
        ):
            if state.get(key) != current:
                raise SystemExit(
                    f"--resume: {key}={current!r} differs from the saved run "
                    f"({state.get(key)!r}); pass the original value"
                )
        subset = list(state["subset_ids"])
        subset_info = state.get("subset_info", {})
        es = pickle.loads(es_path.read_bytes())
        if int(es.popsize) != int(args.popsize):
            raise SystemExit(
                f"--resume: popsize {args.popsize} differs from the saved run "
                f"({es.popsize})"
            )
        _truncate_jsonl(evals_path, int(es.countiter))
        trace_rows = [r for r in _read_trace(trace_path) if int(r["gen"]) < int(es.countiter)]
        best = json.loads(best_path.read_text()) if best_path.exists() else None
        print(
            f"[tune] resuming tag={args.tag} at generation {es.countiter} "
            f"(sigma={es.sigma:.4g})",
            flush=True,
        )
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            if not args.force:
                raise SystemExit(
                    f"{run_dir} already exists and is not empty; pass --resume to "
                    "continue it or --force to overwrite"
                )
            import shutil

            shutil.rmtree(run_dir)
            print(f"[tune] --force: removed {run_dir}", flush=True)
        subset, subset_info = stratified_subset(
            set_ids, args.sim, int(args.subset), int(args.subset_seed)
        )
        for scenario in subset:
            scenario_sets.config_path(args.sim, scenario)  # fail fast if missing
        es = cma.CMAEvolutionStrategy(
            space.x0.tolist(),
            space.sigma0,
            {
                "popsize": int(args.popsize),
                "bounds": [0.0, 1.0],
                "seed": int(args.seed),
                "verbose": -9,
            },
        )
        state = {
            "version": 1,
            "tag": args.tag,
            "policy": args.policy,
            "space": str(Path(args.space).resolve()),
            "space_signature": space.signature(),
            "space_entries": space.describe(),
            "sigma0": space.sigma0,
            "sets": set_names,
            "subset": int(args.subset),
            "subset_seed": int(args.subset_seed),
            "subset_ids": subset,
            "subset_info": subset_info,
            "steps": int(args.steps),
            "sim": args.sim,
            "components": args.components,
            "fast": bool(args.fast),
            "cenv": args.cenv or None,
            "start_offset_days": int(args.start_offset_days),
            "seed": int(args.seed),
            "popsize": int(es.popsize),
            "job_timeout": float(args.job_timeout),
            "backend": args.backend,
            "backend_shards": int(args.shards) if args.backend == "kaggle" else None,
            "backend_shard_workers": (
                int(args.shard_workers) if args.backend == "kaggle" else None
            ),
            "payload_hash": None,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        trace_rows = []
        best = None
        run_dir.mkdir(parents=True, exist_ok=True)
        records_dir.mkdir(parents=True, exist_ok=True)
        _write_json(state_path, state)
        _write_trace(trace_path, trace_rows)

    run_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[tune] tag={args.tag} policy={args.policy} sim={args.sim} "
        f"steps={args.steps} dim={space.dim} popsize={es.popsize} "
        f"workers={args.workers} subset={len(subset)}/{len(set_ids)} "
        f"metric={subset_info.get('size_metric')}",
        flush=True,
    )
    print(f"[tune] subset: {', '.join(subset)}", flush=True)
    print(
        "[tune] space init: "
        + ", ".join(f"{e['name']}={e['init']}" for e in space.entries),
        flush=True,
    )

    if args.backend == "kaggle":
        _check_payload_pin(args, state)

    started_run = time.time()
    try:
        for gen in range(int(es.countiter), int(args.generations)):
            gen_started = time.time()
            candidates: list[dict[str, Any]] = []
            if gen == 0:
                x_baseline = space.x0
                candidates.append(
                    {
                        "idx": -1,
                        "kind": "baseline",
                        "x": x_baseline,
                        "params": space.decode(x_baseline),
                    }
                )
            solutions = es.ask()
            for idx, solution in enumerate(solutions):
                x = np.asarray(solution, dtype=float)
                candidates.append(
                    {"idx": idx, "kind": "candidate", "x": x, "params": space.decode(x)}
                )

            tasks = _build_tasks(
                gen=gen,
                candidates=candidates,
                subset=subset,
                sim=args.sim,
                steps=args.steps,
                components=args.components,
                policy=args.policy,
                tag=args.tag,
                out_dir=records_dir,
                start_offset_days=args.start_offset_days,
                fast=bool(args.fast),
                cenv=evalrun._normalise_cenv(args.cenv),  # noqa: SLF001 - validated spec
            )
            print(
                f"[tune] gen {gen}: {len(candidates)} candidates x "
                f"{len(subset)} scenarios = {len(tasks)} jobs, "
                f"sigma={es.sigma:.4g}",
                flush=True,
            )
            if args.backend == "kaggle":
                results = _evaluate_jobs_remote(tasks, args, state, gen, run_dir)
            elif args.backend == "batch":
                results = _evaluate_jobs_batch(
                    tasks, int(args.workers), float(args.job_timeout)
                )
            else:
                results = _evaluate_jobs(
                    tasks, int(args.workers), float(args.job_timeout)
                )
            finished_at = {name: time.time() for name in results}
            gen_wall = time.time() - gen_started

            rows = [
                _score_candidate(
                    candidate, results, subset, gen, finished_at, gen_started
                )
                for candidate in candidates
            ]
            for row in rows:
                _append_jsonl(evals_path, row)
                flag = "FAIL" if row["n_failed"] else "ok"
                label = "baseline" if row["kind"] == "baseline" else f"#{row['idx']}"
                print(
                    f"[tune] gen {gen} {label:<8} obj={row['objective']:.6f} {flag} "
                    f"{row['n_failed']}/{row['n_scenarios']} failed "
                    f"wall={row['wall_s']:.0f}s",
                    flush=True,
                )

            population = [r for r in rows if r["kind"] == "candidate"]
            es.tell(
                [np.asarray(c["x"], dtype=float) for c in candidates if c["kind"] == "candidate"],
                [r["objective"] for r in population],
            )

            for row in rows:
                if best is None or row["objective"] < best["objective"]:
                    best = {
                        "tag": args.tag,
                        "gen": row["gen"],
                        "idx": row["idx"],
                        "kind": row["kind"],
                        "objective": row["objective"],
                        "params": row["params"],
                        "x": row["x"],
                        "per_scenario": row["per_scenario"],
                        "n_failed": row["n_failed"],
                        "policy": args.policy,
                        "space": state["space"],
                        "space_signature": state["space_signature"],
                        "subset": subset,
                        "steps": int(args.steps),
                        "sim": args.sim,
                        "components": args.components,
                        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    _write_json(best_path, best)

            baseline_row = next((r for r in rows if r["kind"] == "baseline"), None)
            trace_rows.append(
                {
                    "gen": gen,
                    "best": float(best["objective"]),
                    "best_gen": int(best["gen"]),
                    "median": float(np.median([r["objective"] for r in population])),
                    "sigma": float(es.sigma),
                    "baseline": float(baseline_row["objective"]) if baseline_row else None,
                    "wall_s": float(gen_wall),
                }
            )
            _write_trace(trace_path, trace_rows)
            es_path.write_bytes(pickle.dumps(es, protocol=pickle.HIGHEST_PROTOCOL))
            state["generations_done"] = int(es.countiter)
            state["best_objective"] = float(best["objective"])
            state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            _write_json(state_path, state)

            print(
                f"[tune] gen {gen} done: best={best['objective']:.6f} "
                f"(gen {best['gen']}) median={np.median([r['objective'] for r in population]):.6f} "
                f"sigma={es.sigma:.4g} gen_wall={gen_wall:.0f}s "
                f"elapsed={time.time() - started_run:.0f}s",
                flush=True,
            )
    except KeyboardInterrupt:
        es_path.write_bytes(pickle.dumps(es, protocol=pickle.HIGHEST_PROTOCOL))
        state["generations_done"] = int(es.countiter)
        _write_json(state_path, state)
        _write_trace(trace_path, trace_rows)
        print(
            f"\n[tune] interrupted; saved {es_path.name} at generation "
            f"{es.countiter} (resume with --resume)",
            flush=True,
        )
        return 130

    if best is None:
        print(
            f"[tune] nothing evaluated yet (generations={es.countiter}); "
            f"state in {run_dir}",
            flush=True,
        )
        return 0
    print(
        f"[tune] finished tag={args.tag}: generations={es.countiter} "
        f"best={best['objective']:.6f} from gen {best['gen']} "
        f"-> {best_path}",
        flush=True,
    )
    return 0


# --------------------------------------------------------------------------- #
# validation on another set
# --------------------------------------------------------------------------- #
def _load_param_file(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict) and isinstance(raw.get("params"), dict):
        return dict(raw["params"]), raw
    if isinstance(raw, dict):
        return dict(raw), raw
    raise ValueError(f"{path}: expected a JSON object (or one with a 'params' key)")


def _validate(args: argparse.Namespace) -> int:
    params, source = _load_param_file(args.validate)
    set_names = list(args.sets or ["gen_hold"])
    slug = "_".join(str(n).replace("/", "__") for n in set_names)
    tag = args.tag or Path(args.validate).resolve().parent.name
    out_dir = TUNE_ROOT / tag / f"validate_{slug}"
    runner_tag = f"{tag}_validate_{slug}"
    print(
        f"[tune] validating {args.validate} ({len(params)} params) on "
        f"{set_names} steps={args.steps} sim={args.sim} -> {out_dir}",
        flush=True,
    )
    summary = evalrun.run(
        policy=args.policy,
        params=params,
        scenarios=set_names,
        steps=int(args.steps),
        sim=args.sim,
        workers=int(args.workers),
        out=str(out_dir),
        start_offset_days=int(args.start_offset_days),
        tag=runner_tag,
        components=args.components,
        fast=bool(args.fast),
        cenv=args.cenv,
    )
    payload = {
        "source": str(Path(args.validate).resolve()),
        "source_objective": source.get("objective"),
        "source_subset": source.get("subset"),
        "policy": args.policy,
        "sets": set_names,
        "steps": int(args.steps),
        "sim": args.sim,
        "components": args.components,
        "fast": bool(args.fast),
        "cenv": args.cenv or None,
        "leaderboard_score": summary["leaderboard_score"],
        "n_scenarios": summary["n_scenarios"],
        "n_failed": summary["n_failed"],
        "params": params,
        "out_dir": str(out_dir),
        "validated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_json(out_dir / "validation.json", payload)
    print(
        f"[tune] validation score={summary['leaderboard_score']:.6f} "
        f"(train-subset objective was {source.get('objective')}) -> "
        f"{out_dir / 'validation.json'}",
        flush=True,
    )
    return 0 if summary["n_failed"] == 0 else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CMA-ES tuning of a policy's global PARAMS over scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--policy",
        default="policies.ctrl_v1:CtrlV1",
        help="'module:Class' exposing PARAMS (default policies.ctrl_v1:CtrlV1)",
    )
    parser.add_argument(
        "--space",
        default="sbx/spaces/ctrl_v1_core.json",
        help="param-space JSON (default sbx/spaces/ctrl_v1_core.json)",
    )
    parser.add_argument(
        "--set",
        dest="sets",
        nargs="+",
        default=None,
        help="scenario set names, as for sbx.evalrun "
        "(default gen_train when tuning, gen_hold when validating)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
        help="strategic subset size; 0 = the whole set (default 0)",
    )
    parser.add_argument(
        "--subset-seed", type=int, default=0, help="seed of the subset draw (default 0)"
    )
    parser.add_argument("--steps", type=int, default=2880)
    parser.add_argument("--sim", default="shipped", choices=["shipped", "corrected"])
    parser.add_argument("--components", default="hook", choices=["parity", "hook"])
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "install sbx.fastpatch in every eval (runtime monkeypatches; same "
            "numbers to float round-off, ~1.5x per-step CPU); with --backend "
            "kaggle it runs inside the kernels"
        ),
    )
    parser.add_argument(
        "--cenv",
        default=None,
        help=(
            "opt-in exact C step for every eval, e.g. 'fused' (bit-identical to "
            "the Python step on each machine); with --backend kaggle the kernels "
            "compile cenv/ first. Pair with --fast"
        ),
    )
    parser.add_argument("--start-offset-days", type=int, default=0)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument(
        "--generations", type=int, default=30, help="total generations (incl. resumed)"
    )
    parser.add_argument("--seed", type=int, default=0, help="CMA-ES seed (default 0)")
    parser.add_argument("--workers", type=int, default=7,
                        help="local backend: spawn processes")
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "kaggle", "batch"],
        help=(
            "where the (candidate, scenario) jobs run: 'local' pools spawn "
            "processes here, 'kaggle' pushes one wave of shard kernels (needs "
            "the kaggle/remote tooling and the authenticated CLI), 'batch' runs "
            "one cenv.batch process per scenario with the whole population as "
            "its M environments (--workers is then the number of scenarios at "
            "once; needs the gate-passed cenv batch kernel)"
        ),
    )
    parser.add_argument("--shards", type=int, default=5,
                        help="kaggle backend: kernels per generation (default 5)")
    parser.add_argument("--shard-workers", type=int, default=4,
                        help="kaggle backend: processes inside each kernel "
                             "(4 vCPU per session, default 4)")
    parser.add_argument("--wave-timeout", type=float, default=3600.0,
                        help="kaggle backend: seconds to wait for one generation's "
                             "kernels (default 3600)")
    parser.add_argument("--max-concurrency", type=int, default=5,
                        help="kaggle backend: kernels running at once (default 5)")
    parser.add_argument("--repin", action="store_true",
                        help="kaggle backend: publish a new payload version even "
                             "though state.json pinned a different hash")
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=1800.0,
        help="seconds to wait for the next job result before penalizing the rest",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--resume", action="store_true", help="continue --tag")
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing tag directory"
    )
    parser.add_argument(
        "--validate",
        default=None,
        metavar="PARAMS_JSON",
        help="evaluate a param file (e.g. best.json) on --set and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.validate:
        if not Path(args.validate).exists():
            parser.error(f"--validate: {args.validate} does not exist")
        return _validate(args)
    if not args.tag:
        parser.error("--tag is required when tuning")
    if args.generations < 0:
        parser.error("--generations must be >= 0")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return _tune(args)


if __name__ == "__main__":
    sys.exit(main())
