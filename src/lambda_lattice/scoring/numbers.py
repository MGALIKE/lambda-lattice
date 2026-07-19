"""scoring/numbers.py — NUMGAME prereg verdict scorer (H-NG1..H-NG4).

Adapted (packaging only) from ``frontier_lab/ng_analyze.py``. Thresholds and
statistics are unchanged — they are pre-registered.

Prereg (frozen before any model run; validated against the mock-oracle power
check BEFORE model data are scored — C7 precedent):

  Gates per trial: sanity >= 0.75, parse_rate >= 0.8.
  Condition exclusion: overall parse < 0.8 OR fewer than 24/36 arm-a trials
    pass gates (generalized: < 2/3 of arm-a trials) — reported descriptively.
  H-NG1 (arm-a Dfit, per-trial fit_bayes - fit_prox, seed-clustered SE):
    RULE-like        iff Dfit >= +0.10 with z >= 2
    SIMILARITY-like  iff Dfit <= -0.10 with z >= 2
    MIXED/graded     otherwise.
  H-NG2 (arm-b paired slope lambda(32)-lambda(1), seed-clustered SE):
    CONTRACTION PRESENT iff slope <= -0.10 with z >= 2
    CONTRACTION ABSENT  iff point estimate >= -0.05 AND se <= 0.05
    else intermediate.
  H-NG3 (secondary): paired think-nothink difference on Dfit and on the H-NG2
    slope, reported with z, no thresholds.
  H-NG4 (instructed runs): compliance gate arm-a lambda-bar <= 0.45, else NO
    verdict; then slope scored with the D/D' criteria (as H-NG2).

Usage: python -m lambda_lattice.scoring.numbers <primary.json> [<nothink.json>] ...
"""
import json
import math
import sys

GATE_SANITY, GATE_PARSE = 0.75, 0.8
COND_PARSE_GATE = 0.8
ARMA_PASS_FRAC = 24.0 / 36.0
T_FIT = 0.10
T_SLOPE, T_NULL_PT, T_NULL_SE = -0.10, -0.05, 0.05
COMPLIANCE_MAX = 0.45
N_LO, N_HI = 1, 32


def num(x):
    return isinstance(x, (int, float)) and x == x


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def load_conditions(fn):
    d = json.load(open(fn, encoding="utf-8"))
    return [(k, d[k]) for k in d if not k.startswith("_")]


def gated(trials):
    return [t for t in trials if t.get("sanity", 1.0) >= GATE_SANITY
            and t["parse_rate"] >= GATE_PARSE]


def by_seed(trials, arm, field, n=None):
    out = {}
    for t in trials:
        if t["arm"] == arm and (n is None or t["n"] == n) and num(t.get(field)):
            out[t["seed"]] = t[field]
    return out


def paired_slope(tr, field, lo=N_LO, hi=N_HI):
    d_lo, d_hi = by_seed(tr, "b", field, lo), by_seed(tr, "b", field, hi)
    seeds = sorted(set(d_lo) & set(d_hi))
    diffs = {s: d_hi[s] - d_lo[s] for s in seeds}
    m, se = mean_se(list(diffs.values()))
    z = m / se if se > 0 else float("nan")
    return m, se, z, diffs


def score_condition(key, cond, fn):
    trials = cond["trials"]
    tr = gated(trials)
    r = {"key": key, "fn": fn, "mode": cond.get("mode", "?"),
         "instruct": cond.get("instruct", "")}

    parse_all = [t["parse_rate"] for t in trials]
    r["overall_parse"] = sum(parse_all) / len(parse_all) if parse_all else float("nan")
    a_all = [t for t in trials if t["arm"] == "a"]
    a_pass = [t for t in tr if t["arm"] == "a"]
    r["n_pass"], r["n_total"] = len(tr), len(trials)
    r["a_pass"], r["a_total"] = len(a_pass), len(a_all)
    r["excluded"] = (r["overall_parse"] < COND_PARSE_GATE
                     or (a_all and len(a_pass) < ARMA_PASS_FRAC * len(a_all)))

    # arm a: Dfit (per-trial, seed-clustered = 1 arm-a trial per seed) + strata
    r["dfit_by_seed"] = {t["seed"]: t["fit_bayes"] - t["fit_prox"]
                         for t in a_pass
                         if num(t.get("fit_bayes")) and num(t.get("fit_prox"))}
    r["dfit"], r["dfit_se"] = mean_se(list(r["dfit_by_seed"].values()))
    r["dfit_z"] = r["dfit"] / r["dfit_se"] if r["dfit_se"] > 0 else float("nan")
    for f in ("lambda", "lambda_in", "lambda_off", "lambda_broad", "out_acc"):
        r["a_" + f] = mean_se([t[f] for t in a_pass if num(t.get(f))])[0]

    # arm b: lambda by n with references, paired slopes
    ns = sorted({t["n"] for t in tr if t["arm"] == "b"})
    r["b_ns"] = ns
    r["b_rows"] = []
    for n in ns:
        lam = [t["lambda"] for t in tr if t["arm"] == "b" and t["n"] == n
               and num(t.get("lambda"))]
        bay = [t["bayes"] for t in tr if t["arm"] == "b" and t["n"] == n
               and num(t.get("bayes"))]
        prx = [t["prox"] for t in tr if t["arm"] == "b" and t["n"] == n
               and num(t.get("prox"))]
        m, se = mean_se(lam)
        r["b_rows"].append((n, m, se, len(lam), mean_se(bay)[0], mean_se(prx)[0]))
    m, se, z, diffs = paired_slope(tr, "lambda")
    r["slope"], r["slope_se"], r["slope_z"], r["slope_by_seed"] = m, se, z, diffs
    r["slope_in"] = paired_slope(tr, "lambda_in")[:3]
    r["slope_bayes"] = paired_slope(tr, "bayes")[:3]
    return r


def print_condition(r):
    print(f"\n=== {r['key']}  (instruct='{r['instruct']}')  [{r['fn']}] ===")
    print(f"gates: {r['n_pass']}/{r['n_total']} trials pass "
          f"(sanity>={GATE_SANITY}, parse>={GATE_PARSE}); "
          f"overall parse = {r['overall_parse']:.3f}; "
          f"arm-a pass = {r['a_pass']}/{r['a_total']} "
          f"(exclusion if < {ARMA_PASS_FRAC * r['a_total']:.0f})")
    if r["excluded"]:
        print("*** CONDITION EXCLUDED per prereg gate — descriptive only, "
              "no verdicts. ***")
    m, se, z = r["dfit"], r["dfit_se"], r["dfit_z"]
    zp = (m - T_FIT) / se if se > 0 else float("nan")
    zm = (m + T_FIT) / se if se > 0 else float("nan")
    print(f"arm-A Dfit = {m:+.3f} +- {se:.3f}  z(vs 0)={z:+.2f}  "
          f"[z vs +{T_FIT:.2f}: {zp:+.2f}; z vs -{T_FIT:.2f}: {zm:+.2f}]  "
          f"({len(r['dfit_by_seed'])} seeds)")
    print(f"arm-A lambda-bar = {r['a_lambda']:.3f}  strata: "
          f"in={r['a_lambda_in']:.3f} off={r['a_lambda_off']:.3f} "
          f"broad={r['a_lambda_broad']:.3f} out={r['a_out_acc']:.3f}")
    for n, lm, lse, k, bay, prx in r["b_rows"]:
        print(f"  arm-B lambda(n={n:2d}) = {lm:.3f} +- {lse:.3f}  "
              f"({k} seeds)   Bayes ref {bay:.3f}   prox ref {prx:.3f}")
    print(f"arm-B paired slope lambda({N_HI})-lambda({N_LO}) = "
          f"{r['slope']:+.3f} +- {r['slope_se']:.3f}  z={r['slope_z']:+.2f}  "
          f"({len(r['slope_by_seed'])} seeds)")
    mi, si, zi = r["slope_in"]
    mb, sb, zb = r["slope_bayes"]
    print(f"  secondary in_far slope = {mi:+.3f} +- {si:.3f}  z={zi:+.2f}   "
          f"(Bayes ref slope on same trials: {mb:+.3f} +- {sb:.3f})")


# ---------------------------------------------------------------------------
# programmatic verdict classifiers (same thresholds as the printed verdicts;
# these RETURN the label so tests / the API can assert on it)
# ---------------------------------------------------------------------------
def classify_ng1(r) -> str:
    """RULE | SIMILARITY | MIXED — the H-NG1 headline classification."""
    m, z = r["dfit"], r["dfit_z"]
    if num(m) and num(z) and m >= T_FIT and z >= 2:
        return "RULE"
    if num(m) and num(z) and m <= -T_FIT and z <= -2:
        return "SIMILARITY"
    return "MIXED"


def classify_slope(r) -> str:
    """PRESENT | ABSENT | INTERMEDIATE — the H-NG2 contraction classification."""
    m, se, z = r["slope"], r["slope_se"], r["slope_z"]
    if num(m) and num(z) and m <= T_SLOPE and z <= -2:
        return "PRESENT"
    if num(m) and num(se) and m >= T_NULL_PT and se <= T_NULL_SE:
        return "ABSENT"
    return "INTERMEDIATE"


def verdict_ng1(r):
    m, z = r["dfit"], r["dfit_z"]
    if m >= T_FIT and z >= 2:
        ok = r["a_lambda_off"] <= 0.25 and r["a_lambda_in"] >= 0.75
        print(f"H-NG1 verdict: RULE-like  (Dfit {m:+.3f} >= +{T_FIT:.2f}, "
              f"z {z:+.2f} >= 2)   consistency check l_off<=0.25 & "
              f"l_in>=0.75: {'PASS' if ok else 'FAIL'} "
              f"(l_off={r['a_lambda_off']:.3f}, l_in={r['a_lambda_in']:.3f})")
    elif m <= -T_FIT and z <= -2:
        ok = r["a_lambda_off"] > r["a_lambda_broad"] > r["a_lambda_in"]
        print(f"H-NG1 verdict: SIMILARITY-like  (Dfit {m:+.3f} <= -{T_FIT:.2f}, "
              f"z {z:+.2f} <= -2)   consistency check l_off>l_broad>l_in: "
              f"{'PASS' if ok else 'FAIL'} (off={r['a_lambda_off']:.3f} > "
              f"broad={r['a_lambda_broad']:.3f} > in={r['a_lambda_in']:.3f})")
    else:
        print(f"H-NG1 verdict: MIXED/graded — no headline verdict "
              f"(Dfit {m:+.3f}, z {z:+.2f}); stratum profile: "
              f"in={r['a_lambda_in']:.3f} off={r['a_lambda_off']:.3f} "
              f"broad={r['a_lambda_broad']:.3f}")


def verdict_slope(r, label="H-NG2"):
    m, se, z = r["slope"], r["slope_se"], r["slope_z"]
    if m <= T_SLOPE and z <= -2:
        print(f"{label} verdict: CONTRACTION PRESENT  (slope {m:+.3f} <= "
              f"{T_SLOPE:.2f}, z {z:+.2f} <= -2)")
    elif m >= T_NULL_PT and se <= T_NULL_SE:
        print(f"{label} verdict: CONTRACTION ABSENT (power-gated null: "
              f"slope {m:+.3f} >= {T_NULL_PT:.2f}, se {se:.3f} <= "
              f"{T_NULL_SE:.2f})")
    else:
        print(f"{label} verdict: intermediate — no verdict "
              f"(slope {m:+.3f} +- {se:.3f}, z {z:+.2f})")


def verdicts(r):
    tag = "  [EXCLUDED — descriptive only]" if r["excluded"] else ""
    if tag:
        print(tag.strip())
    if r["instruct"]:
        la = r["a_lambda"]
        print(f"H-NG4 compliance gate: arm-A lambda-bar = {la:.3f} "
              f"(must be <= {COMPLIANCE_MAX})")
        if not num(la) or la > COMPLIANCE_MAX:
            print("H-NG4 verdict: COMPLIANCE GATE FAILED — NO verdict.")
            return
        verdict_slope(r, f"H-NG4 ({r['instruct']})")
        return
    verdict_ng1(r)
    verdict_slope(r)


def paired_diff(d_on, d_off, label):
    seeds = sorted(set(d_on) & set(d_off))
    m, se = mean_se([d_on[s] - d_off[s] for s in seeds])
    z = m / se if se > 0 else float("nan")
    print(f"  {label}: {m:+.3f} +- {se:.3f}  z={z:+.2f}  ({len(seeds)} seeds)")


def score_files(paths):
    """Programmatic entry: return a list of per-condition scored dicts (each
    carrying dfit/slope stats). Use classify_ng1 / classify_slope on the items."""
    scored = []
    for fn in paths:
        for key, cond in load_conditions(fn):
            scored.append(score_condition(key, cond, fn))
    return scored


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 1
    scored = []
    for fn in argv:
        for key, cond in load_conditions(fn):
            r = score_condition(key, cond, fn)
            print_condition(r)
            verdicts(r)
            scored.append(r)

    # H-NG3: paired think-nothink on Dfit and slope (uninstructed on/off pairs)
    plain = [r for r in scored if not r["instruct"]]
    for r_on in plain:
        if r_on["mode"] != "on":
            continue
        model = r_on["key"].rsplit(":", 1)[0]
        for r_off in plain:
            if r_off["mode"] == "off" and r_off["key"].rsplit(":", 1)[0] == model:
                print(f"\n--- H-NG3 paired toggle (think - nothink), {model} ---")
                if r_on["excluded"] or r_off["excluded"]:
                    print("  (a condition is EXCLUDED — descriptive only)")
                paired_diff(r_on["dfit_by_seed"], r_off["dfit_by_seed"],
                            "Dfit(think) - Dfit(nothink)   [arm a]")
                paired_diff(r_on["slope_by_seed"], r_off["slope_by_seed"],
                            f"slope(think) - slope(nothink) [arm b, "
                            f"l({N_HI})-l({N_LO})]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
