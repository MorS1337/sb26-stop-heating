"""Runs a list of ``sbx.evalrun._worker`` tasks, used by remote tuning shards.

The tuner's local backend submits one spawn process per (candidate, scenario)
task. A remote shard needs exactly those results, with the same record names and
the same JSON schema, so this script is the remote twin of
``sbx.tune._evaluate_jobs``: it takes the task list, fills the paths that are
only valid on this machine (``config_path`` inside the rebuilt simulator tree,
``out_dir``), and runs ``evalrun._worker`` in a spawn pool with
``maxtasksperchild=1`` -- one process per task, so a scenario never inherits
another's numba/gin state.

It runs as a *subprocess* from the kernel, because Kaggle executes script
kernels through nbconvert and ``multiprocessing``'s spawn start method needs a
real script to re-import.

Contract (both sides of the wire):

* input JSON  ``{"tasks": [...], "out_dir": "...", "workers": N,
  "job_timeout": seconds}`` -- each task carries ``scenario``, ``record_name``,
  ``params``, ``steps``, ``sim``, ``components``, ``start_offset_days``,
  ``policy``, ``tag`` (``config_path``/``out_dir`` are filled here).
* output  ``tasks_status.json`` -- ``{"done": [names], "failed": [...],
  "timed_out": [...]}`` plus one record JSON per task in ``out_dir``.

A task that raises, dies or times out is reported, never fatal: the tuner turns
a missing or errored record into a penalty.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
import traceback
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from sbx import evalrun  # noqa: E402
from sbx import scenarios as scenario_sets  # noqa: E402


def _crash(name: str, task: dict, message: str) -> dict:
    return {
        "scenario": task.get("scenario"),
        "sim": task.get("sim"),
        "config_path": task.get("config_path", ""),
        "record_name": name,
        "error": message,
    }


def run_tasks(tasks: list[dict], out_dir: Path, workers: int, job_timeout: float) -> dict:
    """Runs every task once; returns ``{record_name: result}``.

    Each task gets its own scratch output directory: ``evalrun._shifted_config``
    caches the offset-shifted ``config.gin`` under ``out_dir/_gin`` with a name
    that only depends on the scenario and offset, so two candidates evaluating
    the same scenario concurrently would read a half-written file (a torn gin
    parses as "In call to configurable ..."). Per-task directories remove that
    hazard; the records are moved back to ``out_dir`` afterwards.
    """
    prepared = []
    for task in tasks:
        task = dict(task)
        task["config_path"] = str(
            scenario_sets.config_path(task["sim"], task["scenario"]).resolve()
        )
        stem = Path(task["record_name"]).stem
        task["out_dir"] = str(out_dir / f"_task_{stem}")
        prepared.append(task)

    results: dict[str, dict] = {}
    pending = {task["record_name"]: task for task in prepared}
    if workers <= 1:
        for name, task in list(pending.items()):
            try:
                results[name] = evalrun._worker(task)  # noqa: SLF001 - shared worker
            except Exception:  # noqa: BLE001 - a broken eval is a penalty
                results[name] = _crash(name, task, traceback.format_exc())
        _flatten(out_dir, [task["out_dir"] for task in prepared])
        return results

    context = mp.get_context("spawn")
    pool = context.Pool(processes=min(workers, len(pending)), maxtasksperchild=1)
    try:
        iterator = pool.imap_unordered(
            evalrun._worker, list(pending.values()), chunksize=1  # noqa: SLF001
        )
        while pending:
            try:
                if hasattr(iterator, "next"):  # IMapUnorderedIterator.next(timeout)
                    result = iterator.next(job_timeout)
                else:  # pragma: no cover - defensive
                    result = next(iterator)
            except mp.TimeoutError:
                for name, task in pending.items():
                    results[name] = _crash(
                        name, task, f"job timed out after {job_timeout:.0f}s"
                    )
                pending.clear()
                break
            except StopIteration:
                for name, task in pending.items():
                    results[name] = _crash(name, task, "worker pool ended without a result")
                pending.clear()
                break
            except Exception:  # noqa: BLE001 - pool died; penalize what is left
                detail = traceback.format_exc()
                for name, task in pending.items():
                    results[name] = _crash(name, task, f"pool failure:\n{detail}")
                pending.clear()
                break
            job = result.get("record_name") or (
                Path(result["record_path"]).name if result.get("record_path") else None
            )
            if job is None:  # pragma: no cover - defensive
                continue
            results[job] = result
            pending.pop(job, None)
    finally:
        pool.terminate()
        pool.join()
        _flatten(out_dir, [task["out_dir"] for task in prepared])
    return results


def _flatten(out_dir: Path, task_dirs: list[str]) -> None:
    """Moves every record JSON back into ``out_dir`` and drops the scratch dirs."""
    for directory in task_dirs:
        path = Path(directory)
        if not path.is_dir():
            continue
        for record in path.glob("*.json"):
            target = out_dir / record.name
            if target.exists():
                target.unlink()
            record.replace(target)
        shutil.rmtree(path, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", required=True, help="JSON file with the task spec")
    parser.add_argument("--status", required=True, help="where to write tasks_status.json")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.tasks).read_text())
    tasks = spec["tasks"]
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = run_tasks(
        tasks,
        out_dir=out_dir,
        workers=int(spec.get("workers", 4)),
        job_timeout=float(spec.get("job_timeout", 1800)),
    )
    done = [name for name, r in results.items() if "error" not in r]
    failed = {name: r["error"] for name, r in results.items() if "error" in r}
    timed_out = [name for name, err in failed.items() if "timed out" in err]
    status = {
        "n_tasks": len(tasks),
        "done": sorted(done),
        "failed": {k: str(v).strip().splitlines()[-1] for k, v in failed.items()},
        "timed_out": timed_out,
        "wall_seconds": round(time.time() - started, 2),
        "workers": int(spec.get("workers", 4)),
    }
    Path(args.status).write_text(json.dumps(status, indent=1))
    print(
        f"[tasks] {len(done)}/{len(tasks)} done in {status['wall_seconds']}s "
        f"({len(failed)} failed, {len(timed_out)} timed out)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
