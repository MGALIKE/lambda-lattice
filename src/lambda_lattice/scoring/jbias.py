"""scoring/jbias.py — evaluate the pre-registered join-bias criteria.

Adapted (paths/packaging only) from ``release/lambda-icl/src/jbias_analyze.py``.
Statistics unchanged.

Reads the jbias_*.json result files (default: the shipped ``data/`` dir; or an
explicit list of paths), prints:
  P1: per model x format x n_demos — mean lambda (learned trials), bootstrap 95%
      CI, Bayes + NN references, counterbalance split (wug-pos vs dax-pos seeds).
  P2: AND vs OR accuracy gap + overcoverage/undercoverage asymmetry, bootstrap CI.

Usage: python -m lambda_lattice.scoring.jbias [file1.json file2.json ...]
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
import random


def _default_files():
    from .._paths import data_dir
    d = data_dir()
    return (sorted(glob.glob(str(d / "jbias_*.json")))
            + sorted(glob.glob(str(d / "echo_join_bias.json"))))


def boot_ci(vals, n=5000, seed=0):
    if not vals:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(s) / len(s))
    means.sort()
    return (sum(vals) / len(vals), means[int(0.025 * n)], means[int(0.975 * n)])


def load(paths=None):
    paths = paths or _default_files()
    data = {}
    for p in paths:
        try:
            d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        for k, v in d.items():
            if not k.startswith("_"):
                data[k] = v
    return data


def main(argv=None):
    data = load(argv or None)
    order = sorted(data, key=lambda k: k)
    print("=" * 100)
    print("P1 — lattice position lambda (0=meet/version-space closure, 1=join). "
          "Bayes(size principle) and 1-NN shown as references.")
    print("=" * 100)
    for name in order:
        p1 = data[name].get("p1", [])
        if not p1:
            continue
        learned = [r for r in p1 if r["sanity"] >= 0.75]
        lam, lo, hi = boot_ci([r["lambda"] for r in learned])
        lam_all = sum(r["lambda"] for r in p1) / len(p1)
        bay = [r["bayes"] for r in p1 if not math.isnan(r["bayes"])]
        nn = [r["nn"] for r in p1]
        cb0 = [r["lambda"] for r in learned if r["seed"] % 2 == 0]
        cb1 = [r["lambda"] for r in learned if r["seed"] % 2 == 1]
        print(f"\n{name}  learned {len(learned)}/{len(p1)}")
        print(f"  lambda = {lam:.3f} [{lo:.3f},{hi:.3f}]  (all trials {lam_all:.3f})   "
              f"Bayes={sum(bay)/max(len(bay),1):.3f}  NN={sum(nn)/len(nn):.3f}")
        if cb0 and cb1:
            print(f"  counterbalance: wug-pos {sum(cb0)/len(cb0):.3f} vs "
                  f"dax-pos {sum(cb1)/len(cb1):.3f}")
        for fmt in ("f1", "f2", "f3"):
            sub = [r["lambda"] for r in learned if r["fmt"] == fmt]
            if sub:
                m, l2, h2 = boot_ci(sub, seed=1)
                print(f"    {fmt}: {m:.3f} [{l2:.3f},{h2:.3f}] (n={len(sub)})", end="")
                for nd in sorted({r['n'] for r in learned}):
                    s2 = [r["lambda"] for r in learned if r["fmt"] == fmt and r["n"] == nd]
                    if s2:
                        print(f"  n{nd}={sum(s2)/len(s2):.2f}", end="")
                print()
        fp = [r["p_neither"] for r in learned]
        fn = [1 - r["p_both"] for r in learned]
        if fp:
            print(f"  calibration: P(pos|neither)={sum(fp)/len(fp):.3f}  "
                  f"P(neg|both)={sum(fn)/len(fn):.3f}")

        det = [r for r in learned if "detail" in r]
        if det:
            for krel in sorted({r.get("krel", 2) for r in det}):
                sub = [r for r in det if r.get("krel", 2) == krel]
                for j in range(1, krel):
                    lv = [r.get(f"lambda{j}") for r in sub if r.get(f"lambda{j}") is not None]
                    if lv:
                        m, lo2, hi2 = boot_ci(lv, seed=4 + j)
                        print(f"  k{krel} level {j}/{krel}: lambda_{j} = {m:.3f} "
                              f"[{lo2:.3f},{hi2:.3f}]")
                for lvl in [f"rev{j}" for j in range(1, krel)]:
                    spreads, commit = [], 0
                    for r in sub:
                        by_pat = {}
                        for k, bits, p in r["detail"]:
                            if k == lvl:
                                by_pat.setdefault(bits, []).append(p)
                        if len(by_pat) >= 2:
                            pats = [sum(v) / len(v) for v in by_pat.values()]
                            spreads.append(max(pats) - min(pats))
                            if max(pats) > 0.7 and min(pats) < 0.3:
                                commit += 1
                    if spreads:
                        print(f"  k{krel} {lvl}: within-level spread "
                              f"mean={sum(spreads)/len(spreads):.3f}, "
                              f"attr-selective trials {commit}/{len(spreads)} "
                              f"= {commit/len(spreads):.2f}")
                for nd in sorted({r["n"] for r in sub}):
                    s2 = [r["lambda"] for r in sub if r["n"] == nd]
                    print(f"    k{krel} n={nd}: lambda={sum(s2)/len(s2):.3f} (n={len(s2)})")

    print("\n" + "=" * 100)
    print("P2 — disambiguated concepts: AND vs OR accuracy (humans: AND easier) "
          "+ error direction")
    print("=" * 100)
    for name in order:
        p2 = data[name].get("p2", [])
        if not p2:
            continue
        row = {}
        for c in ("and", "or"):
            accs = [r["acc"] for r in p2 if r["concept"] == c]
            over = [r["overcov"] for r in p2 if r["concept"] == c]
            under = [r["undercov"] for r in p2 if r["concept"] == c]
            row[c] = (boot_ci(accs, seed=2), sum(over) / len(over), sum(under) / len(under))
        gap = [r["acc"] for r in p2 if r["concept"] == "or"]
        ga = [r["acc"] for r in p2 if r["concept"] == "and"]
        diffs = [o - a for o, a in zip(gap, ga)]
        d, dlo, dhi = boot_ci(diffs, seed=3)
        (am, alo, ahi), aover, aunder = row["and"]
        (om, olo, ohi), oover, ounder = row["or"]
        print(f"\n{name}")
        print(f"  AND acc {am:.3f} [{alo:.3f},{ahi:.3f}]  overcov={aover:.3f} "
              f"undercov={aunder:.3f}")
        print(f"  OR  acc {om:.3f} [{olo:.3f},{ohi:.3f}]  overcov={oover:.3f} "
              f"undercov={ounder:.3f}")
        print(f"  OR-minus-AND acc gap = {d:.3f} [{dlo:.3f},{dhi:.3f}]  "
              f"(join bias predicts >0; humans documented <0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
