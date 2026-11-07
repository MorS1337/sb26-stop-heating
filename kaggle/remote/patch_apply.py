"""Applies the frozen simulator patches without needing ``git``/``patch``.

The Kaggle image has no internet and no repository, and we do not want to depend
on a ``patch`` binary being present, so the two frozen diffs in
``sbx/patches/`` are applied by this small pure-Python unified-diff applier.

It understands exactly what those two files contain:

* ``diff --git a/<path> b/<path>`` file headers, ``index``/``mode`` lines,
  ``new file mode`` + ``--- /dev/null`` (file creation),
* ``--- a/<path>`` / ``+++ b/<path>`` lines and ``@@ -l[,s] +l[,s] @@`` hunk
  headers with optional trailing context,
* ``-``/``+``/`` `` body lines, and ``\\ No newline at end of file``.

Application is strict: every removed/context line must match the target file.
A hunk that does not match at its declared line is retried in a small window
(``fuzz`` lines) and the applied hunk is reported; anything still unmatched
raises :class:`PatchError` with the expected/actual line pair. The caller
verifies SHA-256 digests of every touched file afterwards, so a silently wrong
application is impossible.

Used by the remote kernel (its source is embedded by
``kaggle.remote.kernel_gen``) and locally by ``kaggle.remote.payload`` to
cross-check the frozen patches against ``sims/corrected``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")


class PatchError(RuntimeError):
    """A patch could not be applied to the target tree."""


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[tuple[str, str]] = field(default_factory=list)  # (op, text)


@dataclass
class FilePatch:
    path: str  # relative to the tree root, e.g. smart_buildings/x/y.py
    hunks: list[Hunk] = field(default_factory=list)
    create: bool = False


def _strip_prefix(raw: str) -> str:
    raw = raw.strip()
    if "\t" in raw:  # --- a/x\t2024-01-01 ... (timestamps)
        raw = raw.split("\t", 1)[0].strip()
    if raw == "/dev/null":
        return "/dev/null"
    for prefix in ("a/", "b/"):
        if raw.startswith(prefix):
            return raw[len(prefix) :]
    return raw


def parse_patch(text: str) -> list[FilePatch]:
    """Parses a multi-file unified diff into :class:`FilePatch` objects."""
    patches: list[FilePatch] = []
    current: FilePatch | None = None
    hunk: Hunk | None = None
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        head = line.rstrip("\r\n")
        if head.startswith("diff --git "):
            current = None
            hunk = None
        elif head.startswith("new file mode"):
            if current is None:
                current = FilePatch(path="")
            current.create = True
        elif head.startswith("--- "):
            orig = _strip_prefix(head[4:])
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                raise PatchError(f"missing '+++ ' line after '--- {orig}'")
            new_path = _strip_prefix(lines[i].rstrip("\r\n")[4:])
            path = new_path if new_path != "/dev/null" else orig
            if path == "/dev/null":
                raise PatchError("file deletion patches are not supported")
            if current is None or current.path != path:
                current = FilePatch(path=path)
                patches.append(current)
            if orig == "/dev/null":
                current.create = True
            hunk = None
        elif head.startswith("@@"):
            if current is None:
                raise PatchError(f"hunk before file header: {head!r}")
            match = _HUNK_RE.match(head)
            if not match:
                raise PatchError(f"unparsable hunk header: {head!r}")
            hunk = Hunk(
                int(match.group(1)),
                int(match.group(2) or "1"),
                int(match.group(3)),
                int(match.group(4) or "1"),
            )
            current.hunks.append(hunk)
        elif hunk is not None and head[:1] in (" ", "+", "-"):
            # Keep the line ending: it is part of the content.
            hunk.lines.append((head[0], line[1:]))
        elif hunk is not None and head.startswith("\\"):
            pass  # "\ No newline at end of file" is handled at assembly time
        elif head.startswith(("index ", "old mode", "new mode", "similarity index", "rename ")):
            pass
        elif head == "":
            pass
        else:
            hunk = None
        i += 1
    for patch in patches:
        if not patch.path:
            raise PatchError("file patch without a path")
    return patches


def _split_keep(text: str) -> list[str]:
    """Splits into lines that keep their trailing newline (like file.readlines)."""
    if text == "":
        return []
    parts = text.splitlines(keepends=True)
    return parts


def _apply_hunk(
    target: list[str], hunk: Hunk, fuzz: int, delta: int = 0
) -> tuple[list[str], int, int, int]:
    """Applies one hunk to ``target``.

    ``delta`` is the line offset accumulated by earlier hunks of the same file:
    hunk headers keep the *original* file's line numbers, so they have to be
    shifted by every insertion/deletion already applied above them.

    Returns ``(lines, old_start, applied_at_1based, delta_change)``.
    """
    # "\ No newline" markers are metadata, not content.
    old = [text for op, text in hunk.lines if op in (" ", "-")]
    new = [text for op, text in hunk.lines if op in (" ", "+")]

    anchor = hunk.old_start - 1 + delta
    positions = [anchor] + [
        anchor + shift for dist in range(1, fuzz + 1) for shift in (dist, -dist)
    ]
    chosen = None
    for pos in positions:
        if pos < 0 or pos > len(target):
            continue
        if _equal(target[pos : pos + len(old)], old):
            chosen = pos
            break
    if chosen is None and old:
        # The second patch is authored against the pristine tree, so its line
        # numbers do not account for the first patch's insertions. Fall back to
        # a whole-file search: the hunk body is long enough to be unique, and
        # the caller verifies SHA-256 of the result, so a wrong placement is
        # impossible to miss.
        matches = [
            pos
            for pos in range(len(target) - len(old) + 1)
            if _equal(target[pos : pos + len(old)], old)
        ]
        if matches:
            chosen = min(matches, key=lambda pos: abs(pos - anchor))
    if chosen is None:
        want = "\n".join(repr(t) for t in old[:4])
        got = "\n".join(repr(t) for t in target[anchor : anchor + 4])
        raise PatchError(
            f"hunk at old line {hunk.old_start} does not match:\n"
            f"  expected:\n{want}\n  found:\n{got}"
        )

    replacement = list(new)
    for idx, (op, _text) in enumerate(hunk.lines):
        if op == "\\" and idx > 0 and hunk.lines[idx - 1][0] in (" ", "+"):
            replacement[-1] = replacement[-1].rstrip("\n")
    out = target[:chosen] + replacement + target[chosen + len(old) :]
    return out, hunk.old_start, chosen + 1, len(replacement) - len(old)


def _equal(a: list[str], b: list[str]) -> bool:
    if len(a) != len(b):
        return False
    return all(x.rstrip("\r\n") == y.rstrip("\r\n") for x, y in zip(a, b))


def apply_file_patch(root: Path, patch: FilePatch, fuzz: int = 3) -> list[str]:
    """Applies one file patch under ``root``; returns a short log of positions."""
    path = Path(root) / patch.path
    if patch.create and not path.exists():
        original = ""
    else:
        if not path.exists():
            raise PatchError(f"{path} is missing (patch expects it)")
        original = path.read_text(encoding="utf-8")

    lines = _split_keep(original)
    log: list[str] = []
    delta = 0
    for hunk in patch.hunks:
        lines, expected, actual, change = _apply_hunk(lines, hunk, fuzz, delta)
        delta += change
        log.append(f"{patch.path}: hunk at old line {expected} applied at {actual}")
    result = "".join(lines)

    # Write atomically, breaking any hard link instead of mutating the source.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".patchtmp")
    tmp.write_text(result, encoding="utf-8")
    os.replace(tmp, path)
    return log


def apply_patches(root: Path | str, patch_texts: list[str], fuzz: int = 3) -> list[str]:
    """Applies every patch text to the tree at ``root``; returns the log."""
    root = Path(root)
    log: list[str] = []
    for text in patch_texts:
        for file_patch in parse_patch(text):
            log.extend(apply_file_patch(root, file_patch, fuzz=fuzz))
    return log
