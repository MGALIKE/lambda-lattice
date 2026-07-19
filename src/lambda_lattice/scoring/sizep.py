"""scoring/sizep.py — Amendment D scorer: is the size principle elicitable
under explicit instruction?

Adapted (packaging only) from ``frontier_lab/jthink_sizep_analyze.py``.
Thresholds are unchanged — pre-registered (REASONING_PREREG.md, Amendment D,
committed BEFORE the run).

  Gates per trial: sanity >= 0.75, parse_rate >= 0.8.
  Compliance gate: arm-A lambda-bar <= 0.45 (else NO verdict).
  Statistics-elicitable verdict: arm-B lambda(n=32) - lambda(n=4) <= -0.10, z >= 2
    (paired by seed, seed-clustered SE).
  Statistics-absent verdict (capacity null): point estimate >= -0.05 AND se <= 0.05.
  Else: intermediate, no verdict.

Usage: python -m lambda_lattice.scoring.sizep <sizep.json> [<meet.json>]
"""
import json
import math
import sys

GATE_SANITY, GATE_PARSE = 0.75, 0.8


def load(fn):
    d = json.load(open(fn))
    key = [k for k in d if not k.startswith("_")][0]
    return d[key]


def gated(trials):
    return [t for t in trials
            if t.get("sanity", 1.0) >= GATE_SANITY and t["parse_rate"] >= GATE_PARSE]


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def arm_a_lambda(trials):
    a = [t for t in trials if t["arm"] == "a"]
    lams = [t["lambda"] for t in a if not math.isnan(t["lambda"])]
    return mean_se(lams), len(a)


def arm_b_by_n(trials):
    out = {}
    for t in trials:
        if t["arm"] == "b" and not math.isnan(t["lambda"]):
            out.setdefault(t["n"], {})[t["seed"]] = t["lambda"]
    return out


def paired_slope(by_n, lo=4, hi=32):
    seeds = sorted(set(by_n.get(lo, {})) & set(by_n.get(hi, {})))
    diffs = [by_n[hi][s] - by_n[lo][s] for s in seeds]
    m, se = mean_se(diffs)
    z = m / se if se and se > 0 else float("nan")
    return m, se, z, len(seeds)


def bayes_by_n(trials):
    out = {}
    for t in trials:
        if t["arm"] == "b" and not math.isnan(t.get("bayes", float("nan"))):
            out.setdefault(t["n"], []).append(t["bayes"])
    return {n: mean_se(v)[0] for n, v in sorted(out.items())}


def report(fn, label):
    raw = load(fn)
    tr = gated(raw["trials"])
    n_all = len(raw["trials"])
    print(f"\n=== {label} ({fn}) — {len(tr)}/{n_all} trials pass gates ===")
    (la, se_a), na = arm_a_lambda(tr)
    print(f"arm-A lambda-bar = {la:.3f} +- {se_a:.3f}  (n={na})")
    by_n = arm_b_by_n(tr)
    for n in sorted(by_n):
        m, se = mean_se(list(by_n[n].values()))
        print(f"  arm-B lambda(n={n:2d}) = {m:.3f} +- {se:.3f}  ({len(by_n[n])} seeds)")
    bb = bayes_by_n(tr)
    if bb:
        print("  Bayes ref by n: " + "  ".join(f"n={n}:{v:.3f}" for n, v in bb.items()))
    m, se, z, ns = paired_slope(by_n)
    print(f"paired slope lambda(32)-lambda(4) = {m:+.4f} +- {se:.4f}  z={z:+.2f}  ({ns} seeds)")
    return la, by_n, (m, se, z)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 1
    la, by_n, (m, se, z) = report(argv[0], "C8 sizep")
    print("\n--- PREREG VERDICT (Amendment D) ---")
    if la > 0.45:
        print(f"COMPLIANCE GATE FAILED (arm-A lambda-bar {la:.3f} > 0.45): NO verdict.")
    elif m <= -0.10 and z <= -2:
        print("STATISTICS-ELICITABLE: instructed size principle produces contraction dynamics.")
    elif m >= -0.05 and se <= 0.05:
        print("STATISTICS ABSENT UNDER EXPLICIT INSTRUCTION (capacity-level null, power gate met).")
    else:
        print("INTERMEDIATE: no verdict per prereg.")

    if len(argv) > 1:
        report(argv[1], "C7 meet (secondary comparison)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
