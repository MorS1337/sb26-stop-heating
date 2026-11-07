"""Workspace package for the Kaggle tooling (submission pipeline + remote eval).

Marking this directory a regular package keeps ``python -m kaggle.remote...``
resolving to *these* modules instead of the Kaggle CLI's own ``kaggle`` package
that may be installed in the active interpreter. Nothing here imports that CLI
(it is invoked as a subprocess).
"""
