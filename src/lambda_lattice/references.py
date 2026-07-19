"""references.py — the exact reference learners used by the lambda-lattice
instrument, factored out of the harnesses so they are importable and
unit-testable.

NOTHING in this module has been re-derived or re-tuned. Every function body is
a verbatim extraction of the math that runs inside the pre-registered
harnesses (Boolean: ``boolean/harness.py``; number game: ``numbers/harness.py``),
which now import their reference computations from here. The numbers here are
pre-registered science; do not change them.

Two domains:

Boolean attribute concepts (k binary attributes)
  * ``grid_bits`` / ``bayes_join_score`` / ``nn_join_score`` — the per-trial
    exact comparators (size-principle Bayes over conjunctions+disjunctions of
    literals; 1-NN Hamming).
  * ``additive`` / ``meet`` / ``join`` / ``flat`` / ``nn_step`` and the literal
    ``BAYES_K3`` / ``PREREG_K4`` — the analytic ladder reference lines
    (see the provenance notes on each below).

Tenenbaum's (2000) number game over [1,100]
  * ``RULES_EXT`` (32 math rules) + all 5050 intervals => 5082 hypotheses,
    ``build_hspace`` / ``HSPACE`` (rules share 0.5 uniformly, intervals share
    0.5 under an Erlang length prior).
  * ``bayes_predict`` — exact posterior predictive under strong sampling.
  * ``prox_predict`` — the numeric-proximity (GCM-analog) similarity reference.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

# ===========================================================================
# Boolean attribute concepts — exact per-trial comparators
# (extracted verbatim from boolean/harness.py)
# ===========================================================================


def grid_bits(k: int = 4):
    return list(itertools.product((0, 1), repeat=k))


def bayes_join_score(demos, probes) -> float:
    """Size-principle Bayes over H = conjunctions (size 1-3) and disjunctions
    (size 2-3) of literals on 4 binary attributes. Posterior-predictive P(pos)
    avg over revealers."""
    nattr = len(demos[0][0])
    G = grid_bits(nattr)
    lits = [(i, v) for i in range(nattr) for v in (0, 1)]
    hyps = set()
    for size in range(1, nattr):  # 1..nattr-1 (== (1,2,3) for the original 4 attrs)
        for combo in itertools.combinations(lits, size):
            if len({i for i, _ in combo}) < size:
                continue  # same attribute twice
            hyps.add(frozenset(g for g in G if all(g[i] == v for i, v in combo)))
            if size >= 2:
                hyps.add(frozenset(g for g in G if any(g[i] == v for i, v in combo)))
    hyps = list(hyps)
    post = []
    npos = sum(1 for g, y in demos if y)
    for h in hyps:
        ok = all((g in h) == y for g, y in demos)
        post.append(((1.0 / len(h)) ** npos) if ok and len(h) else 0.0)
    Z = sum(post)
    if Z == 0:
        return float("nan")
    scores = [sum(p for p, h in zip(post, hyps) if g in h) / Z
              for g, kind in probes if kind.startswith("rev")]
    return sum(scores) / len(scores)


def nn_join_score(demos, probes) -> float:
    """1-NN Hamming on all 4 attrs; ties -> mean label."""
    scores = []
    for g, kind in probes:
        if not kind.startswith("rev"):
            continue
        dists = [(sum(a != b for a, b in zip(g, d)), y) for d, y in demos]
        m = min(d for d, _ in dists)
        ys = [y for d, y in dists if d == m]
        scores.append(sum(ys) / len(ys))
    return sum(scores) / len(scores)


# ===========================================================================
# Boolean attribute concepts — analytic ladder reference lines
# (extracted verbatim from the figure pipeline's refs.py)
#
# j = number of the krel relevant attributes matched by a probe (interior
# levels only, j = 1..krel-1). The endpoints j=0 ["neither"] and j=krel
# ["both"] are definitionally ~0/~1 for every learner.
# ===========================================================================


def additive(j, k):
    """Linear count-matched profile (== the observed city-block GCM law)."""
    return j / k


def meet(j, k):
    """Conjunctive / version-space closure: only j=k -> 1."""
    return 0.0


def join(j, k):
    """Disjunctive: only j=0 -> 0."""
    return 1.0


def flat(j, k):
    """Chance / no information."""
    return 0.5


def nn_step(j, k):
    """Idealized 1-NN under Hamming distance to the two anchor concepts
    'both' (distance k-j) and 'neither' (distance j); ties (2j == k) break low."""
    return 0.01 if 2 * j <= k else 0.99


# literal task-given Bayes reference for k=3 (see refs.py provenance note)
BAYES_K3 = {1: 0.005, 2: 0.064}

# k=4 pre-registered headline prediction -- literal, labeled j/k prediction
PREREG_K4 = {1: 0.25, 2: 0.50, 3: 0.75}


def ladder_levels(k):
    return list(range(1, k))


# ===========================================================================
# Tenenbaum's number game — hypothesis space, priors, exact learners
# (extracted verbatim from numbers/harness.py)
# ===========================================================================
NMAX = 100
RULE_SHARE, ERLANG_SIGMA = 0.5, 10.0
PROX_SCALE = 10.0


def _mult(k):
    return frozenset(range(k, NMAX + 1, k))


def _end(d):
    return frozenset(x for x in range(1, NMAX + 1) if x % 10 == d)


def _powers(k):
    s, v = set(), k
    while v <= NMAX:
        s.add(v)
        v *= k
    return frozenset(s)  # k^1, k^2, ... (1 excluded)


PRIMES = frozenset(x for x in range(2, NMAX + 1)
                   if all(x % p for p in range(2, int(x ** 0.5) + 1)))

RULES_EXT: dict[str, frozenset] = {
    "even": _mult(2), "odd": frozenset(range(1, NMAX + 1, 2)),
    "squares": frozenset(i * i for i in range(1, 11)),
    "cubes": frozenset(i ** 3 for i in range(1, 5)),
    "primes": PRIMES,
}
for _k in range(3, 11):
    RULES_EXT[f"mult{_k}"] = _mult(_k)
for _k in range(2, 11):
    RULES_EXT[f"pow{_k}"] = _powers(_k)
for _d in range(10):
    RULES_EXT[f"end{_d}"] = _end(_d)
assert len(RULES_EXT) == 32


def build_hspace():
    """[(name, extension, prior)] — rules share RULE_SHARE uniformly;
    intervals share the rest under an Erlang length prior (sigma=10).
    Extensional duplicates (mult10 == end0) both kept: priors are over
    hypotheses, duplicates just stack prior mass (documented choice)."""
    hyps = [(n, e, RULE_SHARE / len(RULES_EXT)) for n, e in RULES_EXT.items()]
    iv = []
    for a in range(1, NMAX + 1):
        for b in range(a, NMAX + 1):
            ln = b - a + 1
            iv.append((f"[{a},{b}]", frozenset(range(a, b + 1)),
                       ln * math.exp(-ln / ERLANG_SIGMA)))
    z = sum(w for _, _, w in iv)
    hyps += [(n, e, (1 - RULE_SHARE) * w / z) for n, e, w in iv]
    return hyps


HSPACE = build_hspace()


def bayes_predict(demos: list[int], values: list[int]) -> dict[int, float]:
    """Exact posterior predictive P(y in C | D) under strong sampling
    P(D|h) = |h|^-n for consistent h, else 0."""
    n = len(demos)
    ds = set(demos)
    num = {v: 0.0 for v in values}
    Z = 0.0
    for _name, ext, prior in HSPACE:
        if ds <= ext:
            w = prior * (1.0 / len(ext)) ** n
            Z += w
            for v in values:
                if v in ext:
                    num[v] += w
    return {v: num[v] / Z for v in values}


def prox_predict(demos: list[int], y: int) -> float:
    """Graded numeric-proximity similarity (the GCM-analog reference)."""
    return math.exp(-min(abs(y - d) for d in demos) / PROX_SCALE)


__all__ = [
    # boolean per-trial comparators
    "grid_bits", "bayes_join_score", "nn_join_score",
    # boolean analytic ladder references
    "additive", "meet", "join", "flat", "nn_step",
    "BAYES_K3", "PREREG_K4", "ladder_levels",
    # number-game space + priors + learners
    "NMAX", "RULE_SHARE", "ERLANG_SIGMA", "PROX_SCALE",
    "PRIMES", "RULES_EXT", "build_hspace", "HSPACE",
    "bayes_predict", "prox_predict",
    # numpy re-export convenience for downstream imports
    "np",
]
