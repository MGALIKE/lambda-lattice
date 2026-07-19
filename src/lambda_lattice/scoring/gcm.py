"""scoring/gcm.py — score prereg AMENDMENT 3 (GCM vs prototype).

Adapted (paths/packaging only) from ``release/lambda-icl/src/jbias_gcm_analyze.py``.
Statistics unchanged.

GCM (city-block, exponential similarity) on this design:
    logit P(pos) = c*(#matched - #unmatched relevant) + log(S+irr / S-irr)
where S(+/-)irr = sum over positive/negative demo exemplars of exp(-c * d_irr),
d_irr = Hamming distance on the IRRELEVANT attributes only. A prototype /
pure-additive-in-relevant model predicts NO within-level dependence on the
irrelevant term.

Prereg prediction (GCM): pooled within-rev-level rank correlation between
residual logit(p) and the irrelevant signal s = log(S+irr/S-irr) is POSITIVE,
pooled Stouffer z > 3, in BOTH models. Kill: z < 3 -> prototype wins.

This scorer regenerates trials via the Boolean harness with the 5-attribute
(EJB_NATTRS=5) semantics the run used; it therefore imports the harness (torch).

Usage: python -m lambda_lattice.scoring.gcm [results/jbias_gcm_test.json]
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

os.environ.setdefault("EJB_NATTRS", "5")  # must match the run; before harness import

CLIP = 1e-4


def _default_path():
    from .._paths import data_dir
    return data_dir() / "jbias_gcm_test.json"


def _harness():
    """Import the Boolean harness and guarantee the 5-attribute semantics the
    GCM run used (regeneration must see the 'material' attribute)."""
    from ..boolean import harness as ejb
    if len(ejb.WORD_ATTRS) < 5:
        ejb.WORD_ATTRS = ejb.WORD_ATTRS + [("material", ("wooden", "metal"))]
    return ejb


def logit(p: float) -> float:
    p = min(max(p, CLIP), 1 - CLIP)
    return math.log(p / (1 - p))


def fit_c_b0(rows, krel):
    """Least squares on cell-level mean logits over levels j=0..krel
    (x = 2j - krel), using rev levels + both/neither."""
    import numpy as np
    xs, ys = [], []
    for j in range(0, krel + 1):
        kind = "both" if j == krel else ("neither" if j == 0 else f"rev{j}")
        vals = []
        for r in rows:
            ps = [p for (k, b, p) in r["detail"] if k == kind]
            vals.extend(ps)
        if vals:
            xs.append(2 * j - krel)
            ys.append(logit(float(np.mean(vals))))
    A = np.vstack([xs, np.ones(len(xs))]).T
    (c, b0), *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    return float(c), float(b0)


def analyze(path=None):
    import numpy as np
    from scipy import stats

    ejb = _harness()
    path = pathlib.Path(path) if path else _default_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    models = [k for k in data if k != "_meta"]
    verdicts = {}
    for model in models:
        rows_all = [r for r in data[model]["p1"] if r.get("sanity", 1) >= 0.75]
        strata_z, per_stratum = [], []
        for krel in sorted({r["krel"] for r in rows_all}):
            for fmt in sorted({r["fmt"] for r in rows_all}):
                rows = [r for r in rows_all if r["krel"] == krel and r["fmt"] == fmt]
                if not rows:
                    continue
                c, b0 = fit_c_b0(rows, krel)
                for j in range(1, krel):  # within each revealer level
                    res, sig = [], []
                    for r in rows:
                        _, _, demos, probes = ejb.make_p1_trial(
                            r["fmt"], r["n"], r["seed"], krel=krel)
                        stored = [(k, b, p) for (k, b, p) in r["detail"]
                                  if k == f"rev{j}"]
                        regen = [(g, kind) for g, kind in probes if kind == f"rev{j}"]
                        assert len(stored) == len(regen), "probe count mismatch"
                        for (kind, bits, p), (g, _) in zip(stored, regen):
                            full = "".join(map(str, g))
                            if len(bits) == len(full):
                                assert bits == full, f"bits mismatch seed={r['seed']}"
                            girr = g[krel:]
                            sp = sum(math.exp(-c * sum(a != b_ for a, b_ in zip(girr, d[krel:])))
                                     for d, y in demos if y == 1)
                            sn = sum(math.exp(-c * sum(a != b_ for a, b_ in zip(girr, d[krel:])))
                                     for d, y in demos if y == 0)
                            s = math.log(sp / sn) if sp > 0 and sn > 0 else 0.0
                            res.append(logit(p) - (c * (2 * j - krel) - b0 * 0 + b0))
                            sig.append(s)
                    if len(res) >= 8 and np.std(sig) > 1e-9:
                        rho, _ = stats.spearmanr(sig, res)
                        n = len(res)
                        z = math.atanh(max(min(rho, 0.9999), -0.9999)) * math.sqrt(n - 3)
                        slope = float(np.polyfit(sig, res, 1)[0])
                        strata_z.append(z)
                        per_stratum.append(dict(krel=krel, fmt=fmt, level=j, n=n,
                                                rho=round(float(rho), 3),
                                                slope=round(slope, 3),
                                                c=round(c, 3), z=round(z, 2)))
        pooled = float(np.sum(strata_z) / math.sqrt(len(strata_z))) if strata_z else float("nan")
        verdicts[model] = dict(pooled_z=round(pooled, 2), strata=per_stratum)
        print(f"== {model}: pooled Stouffer z = {pooled:.2f} "
              f"({'GCM (z>3)' if pooled > 3 else 'prototype/additive (z<3)'})")
        for s in per_stratum:
            print(f"   k{s['krel']} {s['fmt']} j={s['level']}: rho={s['rho']:+.3f} "
                  f"slope={s['slope']:+.3f} n={s['n']} z={s['z']:+.2f} (c={s['c']:.2f})")
    return verdicts


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    analyze(argv[0] if argv else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
