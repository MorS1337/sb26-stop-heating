"""Runs ``sbx`` evaluations on Kaggle kernels and collects them locally.

The local machine is the bottleneck (a 8640-step episode costs 5-17 CPU-minutes,
and a 2880-step tuning evaluation ~5 CPU-minutes on 6 workers). Private Kaggle
CPU batch kernels are free extra compute: 4 vCPU / 31 GB each, and at least five
run concurrently for this account. This driver turns one evaluation into N
shards, one kernel each, and merges the results back into the same directory
layout the local runner produces.

CLI::

    python -m kaggle.remote.remote_eval \\
        --policy policies.ctrl_v1:CtrlV1 --set dev30 --steps 8640 \\
        --sim shipped --shards 3 --tag remote_test

API (for ``sbx/tune.py`` later)::

    handle = submit_jobs([{...}, {...}], shards=2, workers=4)
    results = collect(handle)          # {"summaries": {tag: summary}, ...}

Each shard kernel then, inside the kernel:

1. unzips the private dataset ``mors1337/sbx-payload`` (code + scenarios +
   weather; see :mod:`kaggle.remote.payload`),
2. copies ``smart_buildings`` off the competition mount into
   ``/kaggle/working/sims/shipped`` and builds ``sims/corrected`` from it by
   applying the frozen patches (verified against SHA-256 digests),
3. runs ``python -m sbx.evalrun`` (workers=4) for its scenarios,
4. writes ``results/<tag>/*.json`` and a ``shard.json`` roll-up to
   ``/kaggle/working``.

``collect`` downloads each shard with ``kaggle kernels output``, merges the
records, and writes ``results/<tag>/summary.json`` using the local runner's own
summariser, so scores are computed by exactly the same code as on this machine.

Sharding balances by zone count (parsed from each scenario's ``config.gin``),
because step cost scales with the number of zones, not the number of scenarios.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle.remote import kernel_gen, payload as payload_mod  # noqa: E402
from sbx import scenarios as scenario_sets  # noqa: E402
from sbx.evalrun import DEFAULT_ORACLE_HORIZON, _summarise, parse_oracle  # noqa: E402  (same summary as the local runner)

DEFAULT_WORKERS = 4
DEFAULT_SHARDS = 3
DEFAULT_POLL_SECONDS = 20
DEFAULT_MAX_CONCURRENCY = 5  # measured: 5 of 5 kernels ran at once, no queueing
DEFAULT_TIMEOUT = 6 * 3600.0
TERMINAL = {
    "COMPLETE",
    "ERROR",
    "CANCEL_ACKNOWLEDGED",
    "CANCELLED",
    "CANCEL_REQUESTED",
}


# --------------------------------------------------------------------------- #
# scenario weights / sharding
# --------------------------------------------------------------------------- #
_AHU_ROOMS_RE = re.compile(r"^ahu_\d+_rooms\s*=\s*(\[.*\])\s*$", re.MULTILINE)


def zone_count(scenario: str, sim: str = "shipped") -> int:
    """Number of zones in a scenario, from its ``config.gin`` ``ahu_*_rooms``.

    Step cost tracks zone count (a 84-zone building is ~3x a 30-zone one), and
    the config lists the rooms of every air handler, so counting them is exact
    and needs no simulator import.
    """
    import ast

    try:
        path = scenario_sets.config_path(sim, scenario)
        text = path.read_text()
    except Exception:  # noqa: BLE001 - a missing local scenario just weighs 1
        return 1
    total = 0
    for match in _AHU_ROOMS_RE.finditer(text):
        try:
            total += len(ast.literal_eval(match.group(1)))
        except (ValueError, SyntaxError):
            continue
    return max(1, total)


def shard_scenarios(
    scenarios: Sequence[str], sim: str = "shipped", shards: int = DEFAULT_SHARDS
) -> list[list[str]]:
    """Splits scenario ids into ``shards`` bins, balancing total zone count."""
    shards = max(1, int(shards))
    weights = {scenario: zone_count(scenario, sim) for scenario in scenarios}
    bins: list[list[str]] = [[] for _ in range(shards)]
    loads = [0] * shards
    for scenario in sorted(scenarios, key=lambda s: (-weights[s], s)):
        target = loads.index(min(loads))
        bins[target].append(scenario)
        loads[target] += weights[scenario]
    return [sorted(bin_) for bin_ in bins if bin_]


# --------------------------------------------------------------------------- #
# kaggle CLI
# --------------------------------------------------------------------------- #
class RemoteError(RuntimeError):
    """A kernel or the Kaggle CLI failed."""


_TRANSIENT = (
    "connection",
    "timed out",
    "timeout",
    "502",
    "503",
    "504",
    "temporarily unavailable",
    "remote end closed",
)


def _kaggle(
    args: list[str],
    check: bool = True,
    timeout: float | None = 600,
    attempts: int = 3,
) -> subprocess.CompletedProcess:
    """Runs the CLI, retrying transient network failures with backoff."""
    last: subprocess.CompletedProcess | None = None
    for attempt in range(1, attempts + 1):
        try:
            proc = subprocess.run(
                ["kaggle", *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            if attempt == attempts:
                raise RemoteError(f"kaggle {' '.join(args)} timed out after {timeout}s")
            time.sleep(5 * attempt)
            continue
        text = ((proc.stdout or "") + (proc.stderr or "")).lower()
        transient = any(marker in text for marker in _TRANSIENT)
        if proc.returncode == 0 or not transient or attempt == attempts:
            last = proc
            break
        print(
            f"[remote] kaggle {' '.join(args[:2])} failed transiently, retrying "
            f"({attempt}/{attempts})",
            flush=True,
        )
        time.sleep(5 * attempt)
        last = proc
    assert last is not None
    if check and last.returncode != 0:
        raise RemoteError(
            f"kaggle {' '.join(args)} failed ({last.returncode}):\n"
            f"{last.stdout.strip()}\n{last.stderr.strip()}"
        )
    return last


def kernel_status(kernel_id: str) -> str:
    proc = _kaggle(["kernels", "status", kernel_id], check=False, timeout=120)
    text = (proc.stdout or "") + (proc.stderr or "")
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", text)
    if match:
        return match.group(1)
    if proc.returncode != 0:
        return "UNKNOWN"
    return text.strip().splitlines()[-1] if text.strip() else "UNKNOWN"


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")
    return cleaned or "run"


def kernel_slug(tag: str, shard: int) -> str:
    base = f"sbx-remote-{_sanitize_tag(tag)}-{shard}"
    return base[:60].rstrip("-")


# --------------------------------------------------------------------------- #
# job construction
# --------------------------------------------------------------------------- #
def _resolve_scenarios(job: dict) -> list[str]:
    if job.get("scenarios"):
        items = list(job["scenarios"])
    elif job.get("set"):
        items = [job["set"]]
    else:
        raise ValueError("a job needs 'scenarios' or 'set'")
    sim = job.get("sim", "shipped")
    seen: set[str] = set()
    out: list[str] = []
    for path in scenario_sets.config_paths(items, sim):
        scenario = scenario_sets.scenario_id(path)
        if scenario not in seen:
            seen.add(scenario)
            out.append(scenario)
    return out


def build_shard_jobs(
    jobs: Sequence[dict], payload_hash: str, default_shards: int = DEFAULT_SHARDS
) -> list[dict]:
    """Expands job dicts into per-shard kernel job specs."""
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    specs: list[dict] = []
    for index, job in enumerate(jobs):
        sim = job.get("sim", "shipped")
        scenarios = _resolve_scenarios(job)
        tag = job.get("tag") or f"remote_{index}_{_sanitize_tag(job.get('policy', 'policy'))}"
        shards = int(job.get("shards", default_shards))
        bins = shard_scenarios(scenarios, sim, shards)
        for shard, bin_scenarios in enumerate(bins):
            slug = kernel_slug(tag, shard)
            specs.append(
                kernel_gen.default_job(
                    tag=tag,
                    policy=job.get("policy", "policies.ctrl_v1:CtrlV1"),
                    params=job.get("params") or {},
                    scenarios=bin_scenarios,
                    steps=int(job.get("steps", 8640)),
                    sim=sim,
                    workers=int(job.get("workers", DEFAULT_WORKERS)),
                    payload_hash=payload_hash,
                    components=job.get("components", "parity"),
                    start_offset_days=int(job.get("start_offset_days", 0)),
                    occupancy_seed=job.get("occupancy_seed"),
                    oracle=list(parse_oracle(job.get("oracle"))),
                    oracle_horizon=job.get("oracle_horizon"),
                    shard=shard,
                    n_shards=len(bins),
                    slug=slug,
                    created_utc=created,
                    keep_trees=bool(job.get("keep_trees")),
                    cenv=job.get("cenv"),
                    fast=bool(job.get("fast")),
                    gate=job.get("gate"),
                )
            )
    return specs


def shard_tasks(tasks: Sequence[dict], shards: int = DEFAULT_SHARDS) -> list[list[dict]]:
    """Balances the tuner's (candidate, scenario) tasks by zone count."""
    shards = max(1, int(shards))
    weights = {task["record_name"]: zone_count(task["scenario"], task.get("sim", "shipped"))
               for task in tasks}
    bins: list[list[dict]] = [[] for _ in range(shards)]
    loads = [0] * shards
    for task in sorted(tasks, key=lambda t: (-weights[t["record_name"]], t["record_name"])):
        target = loads.index(min(loads))
        bins[target].append(task)
        loads[target] += weights[task["record_name"]]
    return [bin_ for bin_ in bins if bin_]


def build_task_shard_jobs(
    tasks: Sequence[dict],
    payload_hash: str,
    *,
    tag: str,
    policy: str,
    steps: int,
    sim: str,
    components: str,
    start_offset_days: int,
    shards: int = DEFAULT_SHARDS,
    workers: int = DEFAULT_WORKERS,
    job_timeout: float = 1800.0,
    keep_trees: bool = False,
    cenv: str | None = None,
) -> list[dict]:
    """Expands one generation of tuner tasks into per-shard kernel job specs."""
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bins = shard_tasks(tasks, shards)
    specs: list[dict] = []
    for shard, bin_tasks in enumerate(bins):
        slug = kernel_slug(tag, shard)
        specs.append(
            kernel_gen.default_job(
                tag=tag,
                policy=policy,
                params={},  # each task carries its own params
                scenarios=sorted({task["scenario"] for task in bin_tasks}),
                steps=int(steps),
                sim=sim,
                workers=int(workers),
                payload_hash=payload_hash,
                components=components,
                start_offset_days=int(start_offset_days),
                shard=shard,
                n_shards=len(bins),
                slug=slug,
                created_utc=created,
                keep_trees=keep_trees,
                cenv=cenv,
                tasks=[dict(task) for task in bin_tasks],
                job_timeout=float(job_timeout),
            )
        )
    return specs


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #
def submit_tasks(
    tasks: Sequence[dict],
    *,
    tag: str,
    policy: str,
    steps: int,
    sim: str = "shipped",
    components: str = "hook",
    start_offset_days: int = 0,
    shards: int = DEFAULT_SHARDS,
    workers: int = DEFAULT_WORKERS,
    job_timeout: float = 1800.0,
    out: Path | str | None = None,
    build_dir: Path | str | None = None,
    push_dataset: bool = True,
    push_kernels: bool = True,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    payload_hash: str | None = None,
    keep_trees: bool = False,
    cenv: str | None = None,
) -> dict:
    """Pushes one wave of tuner tasks (``sbx.tune`` backend ``kaggle``).

    ``tasks`` are :func:`sbx.tune._build_tasks`-style dicts: ``record_name``,
    ``scenario``, ``params``, plus the run flags. Each task carries its own
    ``params``, so one kernel evaluates many candidates with ``workers``
    processes.
    """
    hash_, state = _resolve_payload(payload_hash, push_dataset)
    specs = build_task_shard_jobs(
        tasks,
        hash_,
        tag=tag,
        policy=policy,
        steps=steps,
        sim=sim,
        components=components,
        start_offset_days=start_offset_days,
        shards=shards,
        workers=workers,
        job_timeout=job_timeout,
        keep_trees=keep_trees,
        cenv=cenv,
    )
    return _submit_specs(
        specs,
        hash_=hash_,
        state=state,
        out=out,
        build_dir=build_dir,
        push_kernels=push_kernels,
        max_concurrency=max_concurrency,
    )


def submit_jobs(
    jobs: Sequence[dict],
    *,
    shards: int = DEFAULT_SHARDS,
    workers: int = DEFAULT_WORKERS,
    out: Path | str | None = None,
    build_dir: Path | str | None = None,
    push_dataset: bool = True,
    push_kernels: bool = True,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    payload_hash: str | None = None,
) -> dict:
    """Builds the payload, pushes one kernel per shard, returns a handle.

    ``jobs``: dicts with ``policy``, ``params``, ``scenarios`` or ``set``,
    ``steps``, ``sim``, ``components``, ``start_offset_days``, optional
    ``occupancy_seed``/``tag``/``shards``/``workers``.
    """
    hash_, state = _resolve_payload(payload_hash, push_dataset)

    prepared = []
    for job in jobs:
        job = dict(job)
        job.setdefault("workers", workers)
        job.setdefault("shards", shards)
        prepared.append(job)
    specs = build_shard_jobs(prepared, hash_, default_shards=shards)

    return _submit_specs(
        specs,
        hash_=hash_,
        state=state,
        out=out,
        build_dir=build_dir,
        push_kernels=push_kernels,
        max_concurrency=max_concurrency,
    )


def _submit_specs(
    specs: list[dict],
    *,
    hash_: str,
    state: dict,
    out: Path | str | None,
    build_dir: Path | str | None,
    push_kernels: bool,
    max_concurrency: int,
) -> dict:
    """Writes every shard kernel, pushes in waves and returns the handle."""
    first_tag = specs[0]["tag"] if specs else "remote"
    out_dir = Path(out) if out else REPO_ROOT / "results" / first_tag
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    build_root = Path(build_dir) if build_dir else payload_mod.BUILD_DIR / "kernels"

    sources = state.get("sources_by_hash", {}).get(hash_, {})
    handle: dict[str, Any] = {
        "payload_hash": hash_,
        "payload_sources": sources,
        "dataset": payload_mod.DATASET_ID,
        "dataset_version": state.get("dataset_version"),
        "out_dir": str(out_dir),
        "created_utc": specs[0]["created_utc"] if specs else "",
        "kernels": [],
        "max_concurrency": int(max_concurrency),
    }
    for spec in specs:
        slug = spec["kernel_slug"]
        kernel_dir = build_root / slug
        kernel_gen.write_kernel(spec, slug, kernel_dir)
        handle["kernels"].append(
            {
                "slug": slug,
                "id": f"{kernel_gen.KAGGLE_USER}/{slug}",
                "dir": str(kernel_dir),
                "job_tag": spec["tag"],
                "shard": spec["shard"],
                "n_shards": spec["n_shards"],
                "scenarios": spec["scenarios"],
                "job": spec,
                "pushed_utc": "",
                "status": "PENDING",
            }
        )
    print(
        f"[remote] {len(handle['kernels'])} kernel(s) prepared for "
        f"{len({k['job_tag'] for k in handle['kernels']})} job(s), "
        f"payload {hash_}",
        flush=True,
    )
    (out_dir / "handle.json").write_text(json.dumps(handle, indent=1))

    if not push_kernels:
        print("[remote] --no-push: kernels not pushed", flush=True)
        return handle

    _push_wave(handle, 0)
    _save_handle(handle)
    return handle


def _save_handle(handle: dict) -> None:
    """Persists the handle, push stamps included, so a killed driver can resume:
    ``collect(<out_dir>/handle.json)`` tops the queue up instead of re-pushing."""
    (Path(handle["out_dir"]) / "handle.json").write_text(json.dumps(handle, indent=1))


def _resolve_payload(payload_hash: str | None, push_dataset: bool) -> tuple[str, dict]:
    """Selects the payload: reuse the pinned published one, else build/push."""
    state = payload_mod.load_state()
    if payload_hash and payload_hash in state.get("sources_by_hash", {}):
        # Pinned replay: that payload is already published, so do not rebuild
        # (a peer editing a policy must not silently change the run's inputs).
        print(f"[remote] reusing published payload {payload_hash}", flush=True)
        return payload_hash, state
    info = payload_mod.build_payload()
    if payload_hash and payload_hash != info["hash"]:
        raise RemoteError(
            f"payload hash mismatch: requested {payload_hash}, built {info['hash']}"
        )
    state = payload_mod.ensure_dataset(info, push=push_dataset)
    return state.get("payload_hash", info["hash"]), state


def _push_one(kernel: dict) -> None:
    # The account's GPU/TPU quota is reserved for other competitions: CPU only.
    meta = json.loads((Path(kernel["dir"]) / "kernel-metadata.json").read_text())
    if meta.get("enable_gpu") or meta.get("enable_tpu"):
        raise RemoteError(f"refusing to push {kernel['id']}: GPU/TPU enabled (CPU only)")
    proc = _kaggle(["kernels", "push", "-p", kernel["dir"]], check=False, timeout=900)
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and "successfully pushed" not in text.lower():
        raise RemoteError(f"push of {kernel['id']} failed:\n{text.strip()}")
    kernel["pushed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[remote] pushed {kernel['id']} shard {kernel['shard']}", flush=True)


def _push_wave(handle: dict, already_running: int) -> int:
    """Pushes as many unpushed kernels as the concurrency budget allows."""
    budget = max(1, int(handle.get("max_concurrency", DEFAULT_MAX_CONCURRENCY)))
    running = max(0, already_running)
    for kernel in handle["kernels"]:
        if kernel.get("pushed_utc"):
            continue
        if running >= budget:
            break
        _push_one(kernel)
        running += 1
    return running


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
def _poll(handle: dict, timeout: float, poll_seconds: float) -> dict:
    """Waits for every kernel, pushing queued shards as sessions free up."""
    previous = handle.get("poll") or {}
    started = float(previous.get("started_at") or time.time())
    samples: list[dict[str, int]] = list(previous.get("samples") or [])
    max_running = int(previous.get("max_concurrent_running") or 0)
    while True:
        states: dict[str, str] = {}
        for kernel in handle["kernels"]:
            if kernel.get("status") in TERMINAL and kernel.get("done"):
                states[kernel["slug"]] = kernel["status"]
                continue
            status = kernel_status(kernel["id"]) if kernel.get("pushed_utc") else "NOT_PUSHED"
            kernel["status"] = status
            states[kernel["slug"]] = status
            if status in TERMINAL:
                kernel["done"] = True

        counts: dict[str, int] = {}
        for status in states.values():
            counts[status] = counts.get(status, 0) + 1
        running = counts.get("RUNNING", 0)
        max_running = max(max_running, running)
        samples.append({"elapsed": round(time.time() - started, 1), **counts})
        print(
            "[remote] "
            + " ".join(f"{slug}={status}" for slug, status in sorted(states.items())),
            flush=True,
        )

        if all(status in TERMINAL for status in states.values()):
            break
        if time.time() - started > timeout:
            raise RemoteError(
                f"timed out after {timeout:.0f}s; states: {states}"
            )
        # Keep the session budget full: a finished kernel frees a slot. Only the
        # pushed, unfinished kernels hold a session; queued ones do not.
        if any(not k.get("pushed_utc") for k in handle["kernels"]):
            active = sum(
                1
                for k in handle["kernels"]
                if k.get("pushed_utc") and states.get(k["slug"]) not in TERMINAL
            )
            _push_wave(handle, active)
            _save_handle(handle)
        time.sleep(poll_seconds)

    handle["poll"] = {
        "samples": samples,
        "started_at": started,
        "max_concurrent_running": max_running,
        "wall_seconds": round(time.time() - started, 1),
        "queue_observed": any(s.get("PENDING", 0) for s in samples),
    }
    return handle


# Only the result JSONs, the shard roll-up and the log: /kaggle/working also
# holds the payload copy and the two simulator trees (~500 MB), which must not
# be downloaded per shard.
_OUTPUT_PATTERN = r"(^[^/]+\.json$)|(^results/.+\.json$)|(\.log$)"


def _download(kernel: dict, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    proc = _kaggle(
        [
            "kernels",
            "output",
            kernel["id"],
            "-p",
            str(dest),
            "-o",
            "--file-pattern",
            _OUTPUT_PATTERN,
        ],
        check=False,
        timeout=900,
    )
    if proc.returncode != 0:
        print(f"[remote] output download for {kernel['id']} failed:\n{proc.stderr}", flush=True)
    return dest


def _read_shard(dir_: Path) -> dict:
    path = dir_ / "shard.json"
    if not path.exists():
        candidates = sorted(dir_.rglob("shard.json"))
        if candidates:
            path = candidates[0]
        else:
            error = next(iter(sorted(dir_.rglob("error.json"))), None)
            detail = error.read_text() if error else ""
            log = next(iter(sorted(dir_.glob("*.log"))), None)
            snippet = log.read_text()[-4000:] if log else ""
            raise RemoteError(
                f"no shard.json downloaded from {dir_}\n{detail}\n--- log ---\n{snippet}"
            )
    return json.loads(path.read_text())


def _log_tail(dir_: Path, lines: int = 40) -> list[str]:
    logs = sorted(dir_.glob("*.log"))
    if not logs:
        return []
    text = logs[0].read_text(errors="replace")
    try:  # Kaggle logs are JSON-lines with {stream_name,time,data}
        chunks = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                chunks.append(line)
                continue
            chunks.append(str(entry.get("data", "")).rstrip("\n"))
        text = "\n".join(chunks)
    except Exception:  # noqa: BLE001 - a malformed log is still worth showing
        pass
    return text.strip().splitlines()[-lines:]


def collect(
    handle: dict | Path | str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    out: Path | str | None = None,
    repush: bool = False,
    allow_partial: bool = False,
    retries: int = 2,
) -> dict:
    """Polls the kernels, downloads their output, merges and summarises.

    Returns ``{"summaries": {job_tag: summary}, "records": {job_tag: [...]},
    "shards": [...], "handle": handle}``.
    """
    if isinstance(handle, (str, Path)):
        handle = json.loads(Path(handle).read_text())
    out_dir = Path(out or handle["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if repush:
        for kernel in handle["kernels"]:
            kernel["status"] = "PENDING"
            kernel["done"] = False
            _push_one(kernel)
    elif not any(k.get("pushed_utc") for k in handle["kernels"]):
        # A --no-push handle starts its first wave here; a handle whose first
        # wave is already out is topped up by _poll as sessions free up.
        _push_wave(handle, 0)

    _poll(handle, timeout=timeout, poll_seconds=poll_seconds)

    shards_dir = out_dir / "_shards"
    shards: list[dict] = []
    records_by_tag: dict[str, list[dict]] = {}
    attempt = 0
    while True:
        shards, records_by_tag, gates_by_tag, stale = _collect_shards(handle, shards_dir)
        if not stale or attempt >= int(retries):
            break
        # A dataset version takes up to a minute to become the mounted one, so
        # a kernel pushed in that window mounts the previous payload and aborts
        # on the hash check. Re-push those shards once the mount has caught up.
        attempt += 1
        print(
            f"[remote] {len(stale)} shard(s) produced no records "
            f"({', '.join(k['slug'] for k in stale)}); waiting 60s and re-pushing "
            f"(attempt {attempt})",
            flush=True,
        )
        time.sleep(60)
        for kernel in stale:
            kernel["status"] = "PENDING"
            kernel["done"] = False
            _push_one(kernel)
        _poll(handle, timeout=timeout, poll_seconds=poll_seconds)

    summaries: dict[str, dict] = {}
    task_results: dict[str, dict] = {}
    gate_results: dict[str, dict] = {}
    multi = len({k["job_tag"] for k in handle["kernels"]}) > 1
    for kernel in handle["kernels"]:
        tag = kernel["job_tag"]
        if tag in summaries:
            continue
        spec = next(
            (k["job"] for k in handle["kernels"] if k["job_tag"] == tag and k.get("job")),
            None,
        )
        if spec is None:
            raise RemoteError(f"job {tag}: the handle carries no job spec")
        records = list(records_by_tag.get(tag, []))
        if spec.get("tasks"):
            # Task mode: hand back the tuner's records keyed by job name, with
            # no summary (a summary over candidates is meaningless).
            keyed: dict[str, dict] = {}
            for record in records:
                name = _record_key(record)
                if name:
                    keyed[name] = record
            task_results[tag] = keyed
            continue
        if spec.get("gate"):
            # Gate mode: the shards carry plain-vs-C comparisons, not records.
            entries = gates_by_tag.get(tag, [])
            expected = len([k for k in handle["kernels"] if k["job_tag"] == tag])
            if (not entries or len(entries) < expected) and not allow_partial:
                raise RemoteError(
                    f"job {tag}: {len(entries)}/{expected} shard(s) produced a gate "
                    f"comparison"
                )
            merged = {
                "spec": spec["gate"],
                "payload_hash": handle["payload_hash"],
                "shards": entries,
                "results": [item for entry in entries for item in entry["results"]],
            }
            merged["passed"] = bool(merged["results"]) and all(
                item["passed"] for item in merged["results"]
            )
            gate_results[tag] = merged
            continue
        if not records and allow_partial:
            print(f"[remote] {tag}: no records at all; reporting nothing", flush=True)
            summaries[tag] = {}
            continue
        if not records:
            detail = [
                {k: s.get(k) for k in ("slug", "status", "error", "detail")}
                | {"log_tail": (s.get("log_tail") or [])[-12:]}
                for s in shards
                if s.get("error")
            ]
            raise RemoteError(
                f"job {tag}: no shard produced results\n"
                + json.dumps(detail, indent=1)[:6000]
            )
        wall = handle.get("poll", {}).get("wall_seconds", 0.0)
        # Several jobs share one handle: each gets its own results directory,
        # exactly like running the local runner once per job.
        job_out = out_dir / tag if multi else out_dir
        summary = _summarise(
            records,
            tag=tag,
            policy=spec["policy"],
            params=spec["params"],
            sim=spec["sim"],
            steps=spec["steps"],
            start_offset_days=spec["start_offset_days"],
            components=spec["components"],
            occupancy_seed=spec.get("occupancy_seed"),
            oracle=parse_oracle(spec.get("oracle")),
            oracle_horizon=spec.get("oracle_horizon"),
            out_dir=job_out,
            wall_seconds=float(wall),
            cenv=[c.strip() for c in str(spec.get("cenv") or "").split(",") if c.strip()],
            fast=bool(spec.get("fast")),
        )
        summaries[tag] = summary

    # Attach the remote block before writing, so summary.json on disk carries it.
    if summaries:
        _attach_payload_info(handle, shards_dir, summaries)
    for tag, summary in summaries.items():
        if not summary:  # partial with nothing at all
            continue
        job_out = out_dir / tag if multi else out_dir
        job_out.mkdir(parents=True, exist_ok=True)
        _write_job_outputs(job_out, records_by_tag.get(tag, []), summary)

    (out_dir / "handle.json").write_text(json.dumps(handle, indent=1))
    for tag, merged in gate_results.items():
        gate_out = (out_dir / tag if multi else out_dir) / "gate.json"
        gate_out.parent.mkdir(parents=True, exist_ok=True)
        gate_out.write_text(json.dumps(merged, indent=1))
        print(
            f"[remote] {tag}: GATE {'PASS' if merged['passed'] else 'FAIL'} "
            f"({len(merged['results'])} comparison(s)) -> {gate_out}",
            flush=True,
        )
    for tag, summary in summaries.items():
        print(
            f"[remote] {tag}: score={summary['leaderboard_score']:.6f} "
            f"scenarios={summary['n_scenarios']} failed={summary['n_failed']} "
            f"wall={summary['timing']['wall_seconds_total']:.0f}s "
            f"-> {out_dir / tag if multi else out_dir}",
            flush=True,
        )
    return {
        "summaries": summaries,
        "task_results": task_results,
        "gate_results": gate_results,
        "records": records_by_tag,
        "shards": shards,
        "handle": handle,
    }


def _shard_dir(shards_dir: Path, kernel: dict) -> Path:
    """Per-kernel download directory (two jobs can both have shard 0)."""
    return shards_dir / kernel["slug"]


def _collect_shards(
    handle: dict, shards_dir: Path
) -> tuple[list[dict], dict[str, list[dict]], dict[str, list[dict]], list[dict]]:
    """Downloads every shard.

    Returns ``(shard info, records by tag, gate results by tag, stale kernels)``;
    a gate shard carries a comparison instead of records.
    """
    shards: list[dict] = []
    records_by_tag: dict[str, list[dict]] = {}
    gates_by_tag: dict[str, list[dict]] = {}
    stale: list[dict] = []
    for kernel in handle["kernels"]:
        dest = _shard_dir(shards_dir, kernel)
        # Always start clean: a re-run of the same tag must not merge a previous
        # attempt's files (or a previous payload's) into this one.
        shutil.rmtree(dest, ignore_errors=True)
        _download(kernel, dest)
        try:
            shard = _read_shard(dest)
        except RemoteError:
            shard = None
        if shard is None:
            shard = _shard_from_results(dest, kernel, handle)
        if shard is None:
            detail = ""
            error_file = next(iter(sorted(dest.rglob("error.json"))), None)
            if error_file is not None:
                detail = error_file.read_text()
            if "payload hash" in detail or "stale" in detail:
                stale.append(kernel)
            else:
                stale.append(kernel)  # any dead shard is worth one re-push
            shards.append(
                {
                    "slug": kernel["slug"],
                    "status": kernel["status"],
                    "scenarios": kernel["scenarios"],
                    "error": "no shard.json (kernel failed before writing results)",
                    "detail": detail[-2000:],
                    "log_tail": _log_tail(dest),
                }
            )
            continue
        if shard["job"]["payload_hash"] != handle["payload_hash"]:
            raise RemoteError(
                f"shard {kernel['slug']} ran payload {shard['job']['payload_hash']}, "
                f"driver expected {handle['payload_hash']}"
            )
        if not shard["records"] and not shard.get("gate"):
            stale.append(kernel)
        records_by_tag.setdefault(shard["job"]["tag"], []).extend(shard["records"])
        gate = shard.get("gate")
        if gate:
            gates_by_tag.setdefault(shard["job"]["tag"], []).append(
                {"slug": kernel["slug"], **gate}
            )
        shards.append(
            {
                "slug": kernel["slug"],
                "status": kernel["status"],
                "scenarios": shard["job"]["scenarios"],
                "sim": shard["job"]["sim"],
                "env": shard.get("env", {}),
                "eval": shard.get("eval", {}),
                "summary": shard.get("summary"),
                "gate": shard.get("gate"),
                "log_tail": _log_tail(dest),
            }
        )
    return shards, records_by_tag, gates_by_tag, stale


def _record_key(record: dict) -> str | None:
    """The tuner's job name for a downloaded record."""
    if record.get("record_name"):
        return record["record_name"]
    if record.get("record_path"):
        return Path(record["record_path"]).name
    return None


def _shard_from_results(dest: Path, kernel: dict, handle: dict) -> dict | None:
    """Rebuilds a shard roll-up from the downloaded ``results/<tag>/*.json``.

    Kaggle lists kernel output one page at a time (500 files), so a run whose
    ``/kaggle/working`` was large can lose ``shard.json`` while the per-scenario
    JSONs still arrive. Those are enough to merge.
    """
    records = []
    for path in sorted(dest.rglob("results/*/*.json")):
        if path.name == "summary.json":
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        record.setdefault("record_name", path.name)
        records.append(record)
    if not records:
        return None
    payload_hash = None
    env_path = next(iter(sorted(dest.rglob("env.json"))), None)
    if env_path is not None:
        try:
            payload_hash = json.loads(env_path.read_text()).get("payload_hash")
        except json.JSONDecodeError:
            payload_hash = None
    if payload_hash and payload_hash != handle["payload_hash"]:
        raise RemoteError(
            f"shard {kernel['slug']} ran payload {payload_hash}, "
            f"driver expected {handle['payload_hash']}"
        )
    job = dict(kernel.get("job") or {})
    print(
        f"[remote] {kernel['slug']}: shard.json missing, rebuilt from "
        f"{len(records)} result file(s)",
        flush=True,
    )
    return {"job": job, "records": records, "env": {}, "eval": {}, "summary": None}


def _write_job_outputs(out_dir: Path, records: list[dict], summary: dict) -> None:
    """Writes the per-scenario JSONs and summary.json like the local runner."""
    for record in records:
        name = record["scenario"].replace("/", "__") + ".json"
        (out_dir / name).write_text(json.dumps(record, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _attach_payload_info(handle: dict, shards_dir: Path, summaries: dict) -> None:
    """Adds the ``remote`` block to every summary (kernels, concurrency, mount)."""
    for tag, summary in summaries.items():
        mine = [k for k in handle["kernels"] if k["job_tag"] == tag]
        slugs = {k["slug"] for k in mine}
        shard_info = []
        for kernel in mine:
            dest = _shard_dir(shards_dir, kernel)
            shard_path = dest / "shard.json"
            if not shard_path.exists():
                candidates = sorted(dest.rglob("shard.json"))
                shard_path = candidates[0] if candidates else None
            if shard_path is None:
                continue
            shard = json.loads(shard_path.read_text())
            env = shard.get("env", {})
            shard_info.append(
                {
                    "slug": kernel["slug"],
                    "status": kernel["status"],
                    "scenarios": shard["job"]["scenarios"],
                    "steps": shard["job"]["steps"],
                    "sim": shard["job"]["sim"],
                    "cpu_count": env.get("cpu_count"),
                    "python": env.get("python"),
                    "setup_seconds": env.get("setup_seconds"),
                    "total_seconds": env.get("total_seconds"),
                    "tree_timings": env.get("tree_timings"),
                    "eval_seconds": (shard.get("eval") or {}).get("wall_seconds"),
                    "steps_per_second_cpu": (shard.get("summary") or {}).get(
                        "timing", {}
                    ).get("steps_per_second_cpu"),
                    "mean_cpu_seconds_per_scenario": (shard.get("summary") or {}).get(
                        "timing", {}
                    ).get("mean_cpu_seconds_per_scenario"),
                    "mount_vs_local_shipped": env.get("mount_vs_local_shipped"),
                    "versions": env.get("versions"),
                }
            )
        poll = handle.get("poll", {})
        mount_checks = [
            s["mount_vs_local_shipped"]
            for s in shard_info
            if s.get("mount_vs_local_shipped")
        ]
        summary["remote"] = {
            "local_sources_match_payload": payload_mod.local_sources_current(
                handle.get("payload_sources") or {}
            ),
            "dataset": handle.get("dataset"),
            "dataset_version": handle.get("dataset_version"),
            "payload_hash": handle["payload_hash"],
            "kernels": sorted(slugs),
            "n_shards": len(shard_info),
            "shards": shard_info,
            "max_concurrent_running": poll.get("max_concurrent_running"),
            "queue_observed": poll.get("queue_observed"),
            "poll_wall_seconds": poll.get("wall_seconds"),
            "mean_setup_seconds": _mean(
                [s["setup_seconds"] for s in shard_info if s["setup_seconds"] is not None]
            ),
            "mean_eval_seconds": _mean(
                [s["eval_seconds"] for s in shard_info if s["eval_seconds"] is not None]
            ),
            "mount_matches_local_shipped": all(m.get("match") for m in mount_checks)
            if mount_checks
            else None,
            "mount_differing_files": sorted(
                {f for m in mount_checks for f in m.get("differing", [])}
            ),
        }


def _mean(values: Sequence[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------------------- #
# comparison helper (validation)
# --------------------------------------------------------------------------- #
def compare_results(remote_dir: Path | str, local_dir: Path | str) -> dict:
    """Per-scenario ``total_reward`` comparison of two result directories."""
    remote_dir = Path(remote_dir)
    local_dir = Path(local_dir)
    out: list[dict] = []
    for path in sorted(remote_dir.glob("*.json")):
        if path.name in ("summary.json", "handle.json", "comparison.json"):
            continue
        other = local_dir / path.name
        if not other.exists():
            out.append({"scenario": path.stem, "remote": None, "local": None})
            continue
        remote = json.loads(path.read_text())
        local = json.loads(other.read_text())
        out.append(
            {
                "scenario": remote["scenario"],
                "remote_total_reward": remote["total_reward"],
                "local_total_reward": local["total_reward"],
                "delta": remote["total_reward"] - local["total_reward"],
                "steps_remote": remote["steps"],
                "steps_local": local["steps"],
            }
        )
    deltas = [abs(row["delta"]) for row in out if row.get("delta") is not None]
    return {
        "n_compared": len(deltas),
        "identical": bool(deltas) and all(delta == 0.0 for delta in deltas),
        "max_abs_delta": max(deltas) if deltas else None,
        "scenarios": out,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sbx evaluations on Kaggle kernels and collect the results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", default="policies.ctrl_v1:CtrlV1")
    parser.add_argument("--params", default=None, help="JSON object or path to a JSON file")
    parser.add_argument("--set", dest="sets", action="append", nargs="+", default=None,
                        help="scenario set (dev30, gen_train, ...); repeatable")
    parser.add_argument("--scenario", dest="scenario_ids", action="append", nargs="+",
                        default=None,
                        help="scenario ids, e.g. --scenario building_train_1/atlanta_2")
    parser.add_argument("--steps", type=int, default=8640)
    parser.add_argument("--sim", default="shipped", choices=["shipped", "corrected"])
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", default=None, help="results dir (default results/<tag>)")
    parser.add_argument("--components", default="parity", choices=["parity", "hook"])
    parser.add_argument("--cenv", default=None,
                        help="opt-in C step inside the kernel: pass a cenv spec "
                             "(e.g. 'fused') and the kernel compiles payload/cenv "
                             "with gcc, then adds --cenv <spec> to sbx.evalrun. "
                             "Omitted = plain Python. Pair with --fast.")
    parser.add_argument("--fast", action="store_true",
                        help="also pass --fast (sbx.fastpatch) to sbx.evalrun inside "
                             "the kernels; the usual pairing is '--cenv fused --fast'")
    parser.add_argument("--gate", action="store_true",
                        help="gate mode instead of an evaluation: for each scenario and "
                             "sim tree, run cenv/verify/gate.py plain vs --cenv and "
                             "compare the traces (no results/summary is produced)")
    parser.add_argument("--gate-steps", type=int, default=600,
                        help="episode length for --gate (default 600)")
    parser.add_argument("--gate-sims", default="shipped,corrected",
                        help="--gate: comma-separated sim trees (default both)")
    parser.add_argument("--gate-policy", default="policies.ctrl_w:CtrlW",
                        help="--gate: policy both sides run")
    parser.add_argument("--gate-rel-tol", type=float, default=1e-9,
                        help="--gate: allowed relative difference in total_reward")
    parser.add_argument("--gate-temp-rel-tol", dest="gate_temp_rel_tol",
                        type=float, default=1e-12,
                        help="--gate: allowed max |dT_zone| as a fraction of the "
                             "trace's temperature scale (max |T|); converted to "
                             "the absolute K tolerance gate.py takes")
    parser.add_argument("--start-offset-days", type=int, default=0)
    parser.add_argument("--occupancy-seed", type=int, default=None)
    parser.add_argument("--oracle", default=None,
                        help="analysis-only channel, as sbx/evalrun.py --oracle "
                             "(occupancy|forecast|occupancy,forecast)")
    parser.add_argument("--oracle-horizon", type=int, default=DEFAULT_ORACLE_HORIZON)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--no-push", action="store_true",
                        help="build payload + notebooks only, never touch Kaggle")
    parser.add_argument("--no-wait", action="store_true",
                        help="push the kernels and print the handle, do not collect")
    parser.add_argument("--compare", default=None,
                        help="after collecting, compare against this local results dir")
    parser.add_argument("--keep-trees", action="store_true",
                        help="leave payload/ + sims/ in /kaggle/working for debugging "
                             "(makes the output listing ~1.3k files)")
    parser.add_argument("--payload-hash", default=None,
                        help="reuse the already published dataset with this hash")
    parser.add_argument("--retry", action="store_true",
                        help="re-push every shard before collecting (kernels failed)")
    return parser


def _load_params(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
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
    items = [
        item
        for group in (args.sets or []) + (args.scenario_ids or [])
        for item in group
    ] or ["dev30"]
    tag = args.tag or f"remote_{time.strftime('%Y%m%d_%H%M%S')}"
    gate = None
    if args.gate:
        gate = {
            "cenv": args.cenv or "fused",
            "scenarios": items,
            "sims": [s.strip() for s in args.gate_sims.split(",") if s.strip()],
            "steps": int(args.gate_steps),
            "policy": args.gate_policy,
            "params": _load_params(args.params),
            "rel_tol": float(args.gate_rel_tol),
            "temp_rel_tol": float(args.gate_temp_rel_tol),
        }
    job = {
        "policy": args.policy,
        "params": _load_params(args.params),
        "scenarios": items,
        "steps": args.steps,
        "sim": args.sim,
        "components": args.components,
        "start_offset_days": args.start_offset_days,
        "occupancy_seed": args.occupancy_seed,
        "oracle": list(parse_oracle(args.oracle)),
        "oracle_horizon": args.oracle_horizon if parse_oracle(args.oracle) else None,
        "tag": tag,
        "workers": args.workers,
        "shards": args.shards,
        "keep_trees": args.keep_trees,
        "cenv": args.cenv,
        "fast": bool(args.fast),
        "gate": gate,
    }
    out = Path(args.out) if args.out else REPO_ROOT / "results" / tag
    handle = submit_jobs(
        [job],
        shards=args.shards,
        workers=args.workers,
        out=out,
        push_dataset=not args.no_push,
        push_kernels=not args.no_push,
        max_concurrency=args.max_concurrency,
        payload_hash=args.payload_hash,
    )
    print(f"[remote] handle: {json.dumps({k: v for k, v in handle.items() if k != 'kernels'})}")
    for kernel in handle["kernels"]:
        print(f"[remote]   shard {kernel['shard']}: {kernel['id']} {kernel['scenarios']}")
    if args.no_push or args.no_wait:
        print(f"[remote] not waiting; handle at {out / 'handle.json'}")
        return 0

    result = collect(
        handle,
        timeout=args.timeout,
        poll_seconds=args.poll_seconds,
        out=out,
        repush=args.retry,
    )
    if gate:
        merged = result["gate_results"].get(tag) or {}
        for item in merged.get("results", []):
            print(
                f"[remote]   gate {item['scenario']} ({item['sim']}): "
                f"total {item['total_reward'][0]:.9f} vs {item['total_reward'][1]:.9f} "
                f"rel={item['total_reward_rel_diff']:.2e} "
                f"max|dT|={item['max_zone_temp_abs_K']:.3e} K "
                f"ms/step {item['ms_per_step'][0]:.2f} -> {item['ms_per_step'][1]:.2f} "
                f"({item['speedup']:.2f}x) -> "
                f"{'PASS' if item['passed'] else 'FAIL'}"
            )
        (out / "comparison.json").write_text(json.dumps(merged, indent=1))
        return 0 if merged.get("passed") else 1
    summary = result["summaries"][tag]
    if args.compare:
        comparison = compare_results(out, args.compare)
        (out / "comparison.json").write_text(json.dumps(comparison, indent=1))
        print(json.dumps({k: v for k, v in comparison.items() if k != "scenarios"}, indent=1))
        for row in comparison["scenarios"]:
            if row.get("delta"):
                print(
                    f"[remote]   {row['scenario']}: remote {row['remote_total_reward']!r} "
                    f"vs local {row['local_total_reward']!r} (delta {row['delta']})"
                )
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
