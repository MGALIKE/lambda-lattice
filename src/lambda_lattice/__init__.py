"""lambda-lattice — a pre-registered instrument that measures WHERE a language
model sits between similarity-based and rule-based induction, and whether it
applies the Bayesian size principle.

Top-level API (heavy deps are imported lazily so ``import lambda_lattice`` stays
torch-free):

    run_boolean(config)     -> run the Boolean attribute-concept instrument
    run_numbergame(config)  -> run the Tenenbaum number-game instrument
    score(kind, *paths)     -> score result JSONs ("jbias"|"gcm"|"sizep"|"numbers")

The exact reference learners live in ``lambda_lattice.references`` and are
importable/unit-testable without a GPU.
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import references  # torch-free, always safe to import

__all__ = ["__version__", "references", "run_boolean", "run_numbergame", "score"]


def run_boolean(config: dict | None = None, *, reasoning: bool = False) -> dict:
    """Run the Boolean lambda instrument. Requires torch + transformers (and a
    real causal LM). Set ``reasoning=True`` for the generation-based
    thinking-mode variant. See the harness docstrings for config keys."""
    if reasoning:
        from .boolean import reasoning as _r
        return _r.run(config)
    from .boolean import harness as _h
    return _h.run(config)


def run_numbergame(config: dict | None = None) -> dict:
    """Run the number-game lambda instrument. Use ``backend="mock"`` with
    ``models="bayes:none,prox:none"`` for the no-GPU oracle power check."""
    from .numbers import harness as _h
    return _h.run(config)


def score(kind: str, *paths: str):
    """Dispatch to a scorer. kind in {jbias, gcm, sizep, numbers}. Returns the
    scorer's structured output where one exists (numbers), else runs its CLI."""
    kind = kind.lower()
    if kind == "numbers":
        from .scoring import numbers as _n
        return _n.score_files(list(paths))
    if kind == "sizep":
        from .scoring import sizep as _s
        return _s.main(list(paths))
    if kind == "gcm":
        from .scoring import gcm as _g
        return _g.analyze(paths[0]) if paths else _g.analyze()
    if kind == "jbias":
        from .scoring import jbias as _j
        return _j.main(list(paths))
    raise ValueError(f"unknown scorer kind: {kind!r} "
                     "(expected jbias|gcm|sizep|numbers)")
