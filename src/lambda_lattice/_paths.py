"""Locate the repo's shipped data/figures dirs for an editable install.

Resolution order for every getter:
  1. an explicit environment override (LAMBDA_LATTICE_DATA / _FIGURES / _RESULTS)
  2. the repo root discovered by walking up from this file to the first ancestor
     that contains both ``pyproject.toml`` and a ``data`` directory
  3. the current working directory (last-resort fallback)

This keeps the scorers and figure pipeline runnable straight from a
``pip install -e .`` checkout while still honouring explicit paths passed on the
command line or via env vars.
"""
from __future__ import annotations

import os
import pathlib


def repo_root() -> pathlib.Path | None:
    here = pathlib.Path(__file__).resolve()
    for anc in here.parents:
        if (anc / "pyproject.toml").exists() and (anc / "data").is_dir():
            return anc
    return None


def data_dir() -> pathlib.Path:
    env = os.environ.get("LAMBDA_LATTICE_DATA")
    if env:
        return pathlib.Path(env)
    root = repo_root()
    return (root / "data") if root else (pathlib.Path.cwd() / "data")


def figures_dir() -> pathlib.Path:
    env = os.environ.get("LAMBDA_LATTICE_FIGURES")
    if env:
        return pathlib.Path(env)
    root = repo_root()
    return (root / "figures") if root else (pathlib.Path.cwd() / "figures")


def results_dir() -> pathlib.Path:
    """Where fresh harness runs are written by default."""
    env = os.environ.get("LAMBDA_LATTICE_RESULTS")
    if env:
        return pathlib.Path(env)
    root = repo_root()
    return (root / "results") if root else (pathlib.Path.cwd() / "results")
