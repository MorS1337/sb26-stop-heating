"""Build ``cenv/_build/libcenv.{dylib,so}`` from ``cenv/src/*.c``.

Every ``cenv/src/*.c`` (sorted) is compiled into one shared library; every
``cenv/include/*.h`` is on the include path. Adding a component means adding a
``.c`` file and, optionally, its own header -- no edit here.

Flags are pinned to preserve IEEE semantics: ``-O3 -ffp-contract=off`` and never
``-ffast-math``, so a C sum/min/max can be made bit-identical to the numpy one it
replaces (see ``cenv.shadow``).

Rebuild policy: the signature of (compiler, flags, numpy include, every source
and header, size + mtime) is hashed into ``_build/stamp.json``; :func:`build`
recompiles when the hash changes, so a fresh checkout, a peer's new ``.c`` file
or an edited header all rebuild on the next ``import cenv.bindings``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
INC_DIR = HERE / "include"
BUILD_DIR = HERE / "_build"
STAMP = BUILD_DIR / "stamp.json"

IS_DARWIN = sys.platform == "darwin"
LIB_NAME = "libcenv.dylib" if IS_DARWIN else "libcenv.so"
LIB_PATH = BUILD_DIR / LIB_NAME

# -O3 with strict FP: no FMA contraction, no fast-math, no reassociation. The
# simulator's numerics are reproduced bit-for-bit only under these rules.
CFLAGS = (
    "-O3",
    "-std=c11",
    "-fPIC",
    "-ffp-contract=off",
    "-fno-fast-math",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
)
LDFLAGS = ("-shared", "-lm")


def default_cc() -> str:
    """The C compiler: ``$CENV_CC``, else clang on macOS, gcc elsewhere."""
    return os.environ.get("CENV_CC") or ("clang" if IS_DARWIN else "gcc")


def numpy_include() -> str | None:
    """numpy's include directory, or None when numpy is not importable."""
    try:
        import numpy
    except Exception:  # pragma: no cover - numpy is a hard requirement elsewhere
        return None
    return numpy.get_include()


def sources() -> list[Path]:
    return sorted(SRC_DIR.glob("*.c"))


def headers() -> list[Path]:
    return sorted(INC_DIR.glob("*.h"))


def _signature(cc: str, extra_cflags: tuple[str, ...]) -> dict:
    def stat(paths):
        return [
            [
                str(p.relative_to(HERE)),
                p.stat().st_mtime_ns,
                p.stat().st_size,
            ]
            for p in paths
            if p.exists()
        ]

    return {
        "cc": cc,
        "cflags": list(CFLAGS) + list(PLATFORM_CFLAGS) + list(extra_cflags),
        "ldflags": list(LDFLAGS),
        "numpy_include": numpy_include(),
        "sources": stat(sources()),
        "headers": stat(headers()),
        "abi": sys.version_info[:2],
    }


def _digest(sig: dict) -> str:
    blob = json.dumps(sig, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


# glibc hides M_PI (hvac.c) and POSIX clocks (step.c) under a strict -std=c11,
# so a Linux build needs a feature-test macro. It exposes declarations only --
# no codegen change, and the strict-FP rules above are what the numerics
# depend on. macOS needs nothing.
PLATFORM_CFLAGS = () if IS_DARWIN else ("-D_GNU_SOURCE",)


def command(cc: str | None = None, extra_cflags: tuple[str, ...] = ()) -> list[str]:
    cc = cc or default_cc()
    inc = [f"-I{INC_DIR}"]
    npy = numpy_include()
    if npy:
        inc.append(f"-I{npy}")
    return [cc, *CFLAGS, *PLATFORM_CFLAGS, *extra_cflags, *inc,
            *[str(p) for p in sources()], *LDFLAGS, "-o", str(LIB_PATH)]


def build(force: bool = False, cc: str | None = None,
          extra_cflags: tuple[str, ...] = (), verbose: bool | None = None) -> Path:
    """Compile the library if it is missing or older than any input.

    Returns the path to the shared library. Raises ``RuntimeError`` with the
    compiler output on failure.
    """
    cc = cc or default_cc()
    if verbose is None:
        verbose = bool(os.environ.get("CENV_VERBOSE"))
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    sig = _signature(cc, extra_cflags)
    digest = _digest(sig)
    if not force and LIB_PATH.exists() and STAMP.exists():
        try:
            if json.loads(STAMP.read_text())["digest"] == digest:
                return LIB_PATH
        except (json.JSONDecodeError, KeyError):
            pass
    srcs = sources()
    if not srcs:
        raise RuntimeError(f"no C sources in {SRC_DIR}")
    cmd = command(cc, extra_cflags)
    if verbose:
        print("cenv: " + " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "cenv build failed:\n" + " ".join(cmd) + "\n"
            + proc.stdout + proc.stderr
        )
    if proc.stderr.strip() and verbose:
        print(proc.stderr, file=sys.stderr)
    STAMP.write_text(json.dumps({"digest": digest, "command": cmd}, indent=1))
    return LIB_PATH


if __name__ == "__main__":
    print(build(force=True, verbose=True))
