"""Builds and publishes the private Kaggle dataset the remote kernels run from.

A kernel already has the competition mount (the simulator *data* plus the
shipped code) and nothing else: no repository, no internet, no import path into
this workspace. This module packages exactly what is missing -- our ``sbx``
runner code, the policies, the frozen simulator patches, the pool scenarios and
the external weather presets -- into one small dataset:

    mors1337/sbx-payload

Nothing about the simulator *data* is shipped: the kernel copies
``smart_buildings`` off the competition mount and only adds the generated pool
scenarios (a 3.5 MB zip for 456 MB of ``.npy``) and the weather presets.

Layout of the payload directory (``kaggle/remote/build/payload``)::

    sbx/{__init__,evalrun,scenarios,simload}.py   our runner, imported remotely
    sbx/pools/manifest.json                       scenario sets the sets resolve to
    policies/*.py                                 controller modules
    patches/pr2_*.diff, pr3_*.diff                frozen simulator patches
    tools/patch_apply.py                          git-less patch applier
    pool/scenarios.zip                            generated pool scenario dirs
    weather/*.yaml + weather/csv/*.csv            external weather presets
    payload_info.json                             hashes the kernel verifies

``payload_info.json``  contains a content ``hash`` (see :func:`payload_hash`),
every file's SHA-256, the SHA-256 of each patched file *before* and *after* the
patches, and a manifest of the local ``sims/shipped`` code. The kernel verifies
the payload hash before running, then verifies the patched files against
``post_patch_sha``, and reports whether the mounted code still matches our local
shipped tree (the mount is a snapshot of its own, so that can differ).

Versioning: the driver compares the freshly built ``hash`` with
``kaggle/remote/build/state.json``. A changed hash means our code, a policy, a
patch, a pool scenario or a weather preset changed, and only then is a new
dataset version pushed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_DIR = Path(__file__).resolve().parent
BUILD_DIR = REMOTE_DIR / "build"
PAYLOAD_DIR = BUILD_DIR / "payload"
STATE_PATH = BUILD_DIR / "state.json"
DATASET_SLUG = "sbx-payload"
DATASET_ID = f"mors1337/{DATASET_SLUG}"
PAYLOAD_INFO = "payload_info.json"
DATASET_METADATA = "dataset-metadata.json"
# Bumped whenever the payload_info schema (or the verification semantics the
# kernel applies) changes, so the content hash -- and therefore the published
# dataset version -- changes with it.
SCHEMA = 3

SBX_FILES = ("__init__.py", "evalrun.py", "scenarios.py", "simload.py", "fastpatch.py")
CENV_ROOT = REPO_ROOT / "cenv"
# Compiled artefacts and caches never travel: the kernel compiles from src/ with
# its own gcc, and a macOS .dylib would be poison on Linux.
CENV_EXCLUDE_DIRS = ("_build", "__pycache__")
CENV_SUFFIXES = (".py", ".c", ".h")
POOL_PREFIXES = ("building_gen_", "building_hold_")
SCENARIOS_ROOT = REPO_ROOT / "sims" / "corrected" / "smart_buildings" / "training_scenarios"
KIT_SCENARIOS_ROOT = REPO_ROOT / "starter-kit" / "smart_buildings" / "training_scenarios"
SIM_TREES = ("shipped", "corrected")


class PayloadError(RuntimeError):
    """The payload cannot be built or published."""


# --------------------------------------------------------------------------- #
# hashing helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_hash(payload_dir: Path | str = PAYLOAD_DIR) -> str:
    """Content hash of a payload tree (everything except its own metadata)."""
    payload_dir = Path(payload_dir)
    entries: list[str] = []
    for path in sorted(payload_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(payload_dir).as_posix()
        if rel in (PAYLOAD_INFO, DATASET_METADATA):
            continue
        entries.append(f"{rel} {sha256_file(path)}")
    blob = "\n".join(entries).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


def _rel_paths(root: Path, pattern: str = "*") -> list[Path]:
    return sorted(p for p in root.rglob(pattern) if p.is_file())


# --------------------------------------------------------------------------- #
# payload sources
# --------------------------------------------------------------------------- #
def _sbx_sources() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name in SBX_FILES:
        path = REPO_ROOT / "sbx" / name
        if not path.exists():
            raise PayloadError(f"missing {path}")
        out[f"sbx/{name}"] = path
    manifest = REPO_ROOT / "sbx" / "pools" / "manifest.json"
    if manifest.exists():
        out["sbx/pools/manifest.json"] = manifest
    return out


def _policy_sources() -> dict[str, Path]:
    root = REPO_ROOT / "policies"
    return {
        f"policies/{p.name}": p
        for p in sorted(root.glob("*.py"))
        if p.is_file()
    }


def _cenv_sources() -> dict[str, Path]:
    """``cenv/`` sources (Python + C + headers) the kernel compiles and imports.

    ``_build/`` and ``__pycache__`` are skipped, so the payload is compiler- and
    platform-neutral: the kernel builds ``libcenv.so`` from these sources with
    its own gcc and the flags pinned in :mod:`cenv.build`.
    """
    if not (CENV_ROOT / "src").is_dir():
        raise PayloadError(f"missing cenv sources: {CENV_ROOT}")
    out: dict[str, Path] = {}
    for path in sorted(CENV_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in CENV_SUFFIXES:
            continue
        rel = path.relative_to(CENV_ROOT)
        if any(part in CENV_EXCLUDE_DIRS for part in rel.parts):
            continue
        out[f"cenv/{rel.as_posix()}"] = path
    if not any(name.startswith("cenv/src/") and name.endswith(".c") for name in out):
        raise PayloadError(f"no .c files under {CENV_ROOT / 'src'}")
    return out


def _patch_sources() -> dict[str, Path]:
    root = REPO_ROOT / "sbx" / "patches"
    patches = sorted(root.glob("*.diff"))
    if len(patches) < 2:
        raise PayloadError(f"expected the two frozen patches in {root}")
    return {f"patches/{p.name}": p for p in patches}


def _weather_sources() -> dict[str, Path]:
    root = REPO_ROOT / "data" / "weather_ext" / "weather"
    if not root.is_dir():
        raise PayloadError(f"missing weather presets: {root}")
    out: dict[str, Path] = {}
    for yaml_path in sorted(root.glob("*.yaml")):
        out[f"weather/{yaml_path.name}"] = yaml_path
    for csv_path in sorted((root / "csv").glob("*.csv")):
        out[f"weather/csv/{csv_path.name}"] = csv_path
    return out


def _pool_scenario_dirs() -> list[Path]:
    """Scenario dirs the competition mount lacks: every generated pool building,
    plus scenarios ``weather_ext`` added under a *shipped* building (ext_hold puts
    two external stations on train_1/train_8), which the mount has no copy of."""
    root = SCENARIOS_ROOT
    if not root.is_dir():
        raise PayloadError(f"missing {root}; run sbx/make_sims.sh + sbx/make_pool.py")
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith(POOL_PREFIXES)]
    if not dirs:
        raise PayloadError(f"no generated pool scenarios under {root}")
    for building in sorted(root.iterdir()):
        if not building.is_dir() or building.name.startswith(POOL_PREFIXES):
            continue
        dirs += [
            scenario
            for scenario in sorted(building.iterdir())
            if (scenario / "config.gin").exists()
            and not (KIT_SCENARIOS_ROOT / building.name / scenario.name).is_dir()
        ]
    return dirs


def _pool_fingerprint(dirs: list[Path]) -> str:
    """Cheap change detector for the (456 MB) pool scenario directories."""
    digest = hashlib.sha256()
    for directory in dirs:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                digest.update(
                    f"{path.relative_to(SCENARIOS_ROOT).as_posix()} {stat.st_size}\n".encode()
                )
    return digest.hexdigest()[:16]


def _write_pool_zip(dirs: list[Path], dest: Path) -> None:
    """Zips the generated scenario dirs deterministically (fixed timestamps)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for directory in dirs:
            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(SCENARIOS_ROOT).as_posix()
                info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, path.read_bytes())
    tmp.replace(dest)


def _tree_code_manifest(tree: str) -> dict[str, str]:
    """SHA-256 of every simulator ``.py`` file in one of our local sim trees."""
    root = REPO_ROOT / "sims" / tree / "smart_buildings"
    if not root.is_dir():
        return {}
    out: dict[str, str] = {}
    for path in _rel_paths(root, "*.py"):
        rel = path.relative_to(REPO_ROOT / "sims" / tree).as_posix()
        if "__pycache__" in rel:
            continue
        out[rel] = sha256_file(path)
    return out


def _patched_paths(patch_paths: dict[str, Path]) -> list[str]:
    sys.path.insert(0, str(REMOTE_DIR))
    from patch_apply import parse_patch  # noqa: PLC0415

    paths: list[str] = []
    for path in patch_paths.values():
        for file_patch in parse_patch(path.read_text()):
            if file_patch.path not in paths:
                paths.append(file_patch.path)
    return paths


def _git_info() -> dict[str, str | bool]:
    def run(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True,
                text=True,
                check=False,
            )
            return out.stdout.strip()
        except OSError:
            return ""

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "--short", "HEAD"),
        "dirty": bool(status),
        "status_sha": hashlib.sha256(status.encode()).hexdigest()[:16] if status else "",
    }


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_payload(force: bool = False) -> dict:
    """Materialises ``kaggle/remote/build/payload`` and returns its info dict."""
    state = load_state()
    payload_dir = PAYLOAD_DIR
    payload_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    files.update(_sbx_sources())
    files.update(_policy_sources())
    files.update(_cenv_sources())
    files.update(_patch_sources())
    files["tools/patch_apply.py"] = REMOTE_DIR / "patch_apply.py"
    files["tools/remote_tasks.py"] = REMOTE_DIR / "taskrun.py"
    files.update(_weather_sources())

    pool_dirs = _pool_scenario_dirs()
    pool_fp = _pool_fingerprint(pool_dirs)
    pool_zip = payload_dir / "pool" / "scenarios.zip"
    rebuild_pool = force or not pool_zip.exists() or state.get("pool_fingerprint") != pool_fp
    if rebuild_pool:
        t0 = time.time()
        _write_pool_zip(pool_dirs, pool_zip)
        print(
            f"[payload] pool scenarios: {len(pool_dirs)} dirs -> "
            f"{pool_zip.stat().st_size / 1e6:.2f} MB in {time.time() - t0:.1f}s",
            flush=True,
        )

    for rel, source in files.items():
        dest = payload_dir / rel
        if (
            not force
            and dest.exists()
            and dest.stat().st_size == source.stat().st_size
            and sha256_file(dest) == sha256_file(source)
        ):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

    # Remove files that no longer belong (e.g. a deleted policy module).
    expected = set(files) | {"pool/scenarios.zip"}
    keep = expected | {PAYLOAD_INFO, DATASET_METADATA}
    for path in sorted(payload_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(payload_dir).as_posix()
        if rel in keep or "__pycache__" in rel or rel.endswith(".tmp"):
            continue
        path.unlink()

    patch_paths = _patch_sources()
    patched = _patched_paths(patch_paths)
    pre: dict[str, str] = {}
    post: dict[str, str] = {}
    for rel in patched:
        shipped = REPO_ROOT / "sims" / "shipped" / rel
        corrected = REPO_ROOT / "sims" / "corrected" / rel
        pre[rel] = sha256_file(shipped) if shipped.exists() else ""
        post[rel] = sha256_file(corrected) if corrected.exists() else ""
    missing = [rel for rel, digest in post.items() if not digest]
    if missing:
        raise PayloadError(
            f"sims/corrected is missing patched files {missing}; run sbx/make_sims.sh"
        )

    # Kaggle repacks uploaded archives, so the pool .zip's raw bytes are not
    # stable. What has to be stable is its *contents*: hash every scenario file
    # separately and let the kernel verify them after extraction.
    pool_files: dict[str, str] = {}
    pool_sizes: dict[str, int] = {}
    for directory in pool_dirs:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                rel = path.relative_to(SCENARIOS_ROOT).as_posix()
                pool_files[rel] = sha256_file(path)
                pool_sizes[rel] = path.stat().st_size

    file_hashes = {
        path.relative_to(payload_dir).as_posix(): sha256_file(path)
        for path in sorted(payload_dir.rglob("*"))
        if path.is_file()
        and path.relative_to(payload_dir).as_posix()
        not in (PAYLOAD_INFO, DATASET_METADATA, "pool/scenarios.zip")
    }
    digest_lines = [f"schema {SCHEMA}"]
    digest_lines += [f"{rel} {digest}" for rel, digest in sorted(file_hashes.items())]
    digest_lines += [
        f"pool/{rel} {digest} {pool_sizes[rel]}"
        for rel, digest in sorted(pool_files.items())
    ]
    content_hash = hashlib.sha256("\n".join(digest_lines).encode()).hexdigest()[:32]

    info = {
        "schema": SCHEMA,
        "hash": content_hash,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_id": DATASET_ID,
        "pool_fingerprint": pool_fp,
        "pool_dirs": [d.relative_to(SCENARIOS_ROOT).as_posix() for d in pool_dirs],
        "pool_zip_bytes": (payload_dir / "pool" / "scenarios.zip").stat().st_size,
        "pool_files": pool_files,
        "pool_sizes": pool_sizes,
        "patches": {name: sha256_file(path) for name, path in patch_paths.items()},
        "pre_patch_sha": pre,
        "post_patch_sha": post,
        "shipped_code_sha": _tree_code_manifest("shipped"),
        "git": _git_info(),
        "files": dict(sorted(file_hashes.items())),
    }
    (payload_dir / PAYLOAD_INFO).write_text(json.dumps(info, indent=1, sort_keys=True))

    state["pool_fingerprint"] = pool_fp
    save_state(state)
    total = sum((payload_dir / rel).stat().st_size for rel in file_hashes)
    print(
        f"[payload] {len(file_hashes)} files, {total / 1e6:.2f} MB "
        f"-> hash {content_hash} ({payload_dir})",
        flush=True,
    )
    return info


def local_sources_current(recorded: dict[str, str]) -> dict:
    """Do the local ``sbx/`` and ``policies/`` files still match what was shipped?

    A run is only comparable with a local reference if the runner code is the
    same; this reports exactly which files drifted (peers edit them).
    """
    differing, missing = [], []
    for rel, digest in sorted(recorded.items()):
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)
        elif sha256_file(path) != digest:
            differing.append(rel)
    return {"match": not differing and not missing, "differing": differing, "missing": missing}


# --------------------------------------------------------------------------- #
# state / publishing
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=1, sort_keys=True))


def _run_kaggle(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["kaggle", *args], capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise PayloadError(
            f"kaggle {' '.join(args)} failed ({proc.returncode}):\n"
            f"{proc.stdout.strip()}\n{proc.stderr.strip()}"
        )
    return proc


def _write_dataset_metadata() -> None:
    metadata = {
        "title": "sbx payload",
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (PAYLOAD_DIR / DATASET_METADATA).write_text(json.dumps(metadata, indent=1))


def wait_until_ready(timeout: float = 300.0, interval: float = 5.0, grace: float = 35.0) -> bool:
    """Waits until Kaggle reports the dataset ``ready``.

    A freshly created dataset/version is not mountable for a minute or so;
    pushing a kernel into that window fails its setup at t+1s.
    """
    started = time.time()
    while time.time() - started < timeout:
        proc = _run_kaggle(["datasets", "status", DATASET_ID], check=False)
        text = (proc.stdout or proc.stderr or "").lower()
        if "ready" in text:
            # "ready" is reported before the new version is the one that new
            # kernels actually mount; a kernel pushed inside that window sees
            # the previous payload and aborts on its hash check.
            time.sleep(grace)
            return True
        time.sleep(interval)
    print(f"[payload] warning: {DATASET_ID} not reported ready within {timeout:.0f}s")
    return False


def dataset_exists() -> bool:
    proc = _run_kaggle(["datasets", "list", "-m", "-s", DATASET_SLUG], check=False)
    if proc.returncode != 0:
        return False
    for line in proc.stdout.splitlines():
        fields = line.split()
        if fields and fields[0] == DATASET_ID:
            return True
    return False


def ensure_dataset(info: dict | None = None, push: bool = True) -> dict:
    """Pushes/updates the dataset when its content hash changed.

    Returns the state dict with ``payload_hash`` and ``dataset_version``.
    """
    info = info or build_payload()
    state = load_state()
    if state.get("payload_hash") == info["hash"] and state.get("dataset_version"):
        print(
            f"[payload] dataset {DATASET_ID} v{state['dataset_version']} is current "
            f"({info['hash']})",
            flush=True,
        )
        return state
    if not push:
        print(f"[payload] --no-push: would publish hash {info['hash']}", flush=True)
        return {**state, "payload_hash": info["hash"], "pushed": False}

    _write_dataset_metadata()
    notes = f"sbx payload {info['hash']} (git {info['git']['commit']})"
    exists = bool(state.get("dataset_version")) or dataset_exists()
    if exists:
        proc = _run_kaggle(
            ["datasets", "version", "-p", str(PAYLOAD_DIR), "-r", "zip", "-q", "-m", notes],
            check=False,
        )
        if proc.returncode != 0 and "already exists" not in (proc.stdout + proc.stderr):
            # A fresh checkout with no state file: fall through to create once.
            if not dataset_exists():
                exists = False
            else:
                raise PayloadError(
                    f"kaggle datasets version failed:\n{proc.stdout}\n{proc.stderr}"
                )
        else:
            print(f"[payload] pushed dataset version: {notes}", flush=True)
    if not exists:
        proc = _run_kaggle(
            ["datasets", "create", "-p", str(PAYLOAD_DIR), "-r", "zip", "-q"], check=False
        )
        if proc.returncode != 0 and "already exists" in (proc.stdout + proc.stderr):
            proc = _run_kaggle(
                [
                    "datasets",
                    "version",
                    "-p",
                    str(PAYLOAD_DIR),
                    "-r",
                    "zip",
                    "-q",
                    "-m",
                    notes,
                ]
            )
        elif proc.returncode != 0:
            raise PayloadError(
                f"kaggle datasets create failed:\n{proc.stdout}\n{proc.stderr}"
            )
        print(f"[payload] created dataset {DATASET_ID}", flush=True)

    wait_until_ready()
    sources = {
        rel: digest
        for rel, digest in info.get("files", {}).items()
        if rel.startswith(("sbx/", "policies/"))
    }
    by_hash = dict(state.get("sources_by_hash") or {})
    by_hash[info["hash"]] = sources
    for stale in list(by_hash)[:-5]:
        del by_hash[stale]

    state = {
        **state,
        "sources_by_hash": by_hash,
        "payload_hash": info["hash"],
        "dataset_version": int(state.get("dataset_version", 0)) + 1,
        "dataset_id": DATASET_ID,
        "pushed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pool_fingerprint": info.get("pool_fingerprint", state.get("pool_fingerprint")),
        "git_commit": info["git"]["commit"],
    }
    save_state(state)
    return state


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild every payload file")
    parser.add_argument("--no-push", action="store_true", help="build only, never touch Kaggle")
    parser.add_argument("--print-hash", action="store_true")
    args = parser.parse_args(argv)

    info = build_payload(force=args.force)
    if args.print_hash:
        print(info["hash"])
    if not args.no_push:
        ensure_dataset(info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
