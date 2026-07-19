"""F11 / F13 / F14 — the domain-boundary, faithfulness, and elicitation-ladder
figures for the reasoning + number-game extension of the lambda-lattice paper.

Adapted (packaging only) from the lab repo
``release/lambda-icl/src/make_f11_f13_f14.py``. Only imports and the data/figures
path resolution were changed — every gate, seed, and statistic is unchanged.

These three follow the F10 pattern (make_f10.py): they read the raw per-probe
result JSONs in the shipped ``data/`` dir and write 300-dpi PNG + PDF to
``figures/``. Every plotted number is recomputed here from the raw trials with
the SAME gates and seed-clustered statistics used by the pre-registered scorers:

  * learned trial  := sanity >= 0.75 AND parse_rate >= 0.8
  * seed-clustered SE := SD(per-seed values) / sqrt(n_seeds)
  * contraction slope := mean over seeds present at BOTH n_lo and n_hi of the
    paired difference lambda(n_hi) - lambda(n_lo)

Reference curves (exact Bayes / Tenenbaum size principle, numeric proximity) are
read from the per-trial 'bayes' / 'prox' fields the harness computed exactly per
trial -- never hardcoded from the README.
"""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from .style import (AQUA, BLUE, GREEN, GRIDLINE, INK_MUTED, INK_PRIMARY,
                    INK_SECONDARY, ORANGE, RED, VIOLET, savefig, use_style)
from .._paths import data_dir, figures_dir

DATA = data_dir()
OUT = figures_dir()

GATE_SANITY, GATE_PARSE = 0.75, 0.8


# ---------------------------------------------------------------- data helpers
def load_cond(fname: str, key: str | None = None) -> dict:
    d = json.loads((DATA / fname).read_text(encoding="utf-8"))
    if key is None:
        key = [k for k in d if not k.startswith("_")][0]
    return d[key]


def learned(trials: list[dict]) -> list[dict]:
    return [t for t in trials
            if t.get("sanity", 1.0) >= GATE_SANITY and t["parse_rate"] >= GATE_PARSE]


def _clean(xs):
    return [float(x) for x in xs
            if not (isinstance(x, float) and math.isnan(x)) and x is not None]


def mean_se(xs):
    xs = _clean(xs)
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def armb_by_n(trials, field="lambda"):
    """{n: {seed: value}} over learned arm-B trials (non-nan)."""
    out: dict[int, dict[int, float]] = {}
    for t in trials:
        if t["arm"] == "b" and not (isinstance(t[field], float) and math.isnan(t[field])):
            out.setdefault(t["n"], {})[t["seed"]] = float(t[field])
    return out


def per_n_mean_se(byn):
    ns = sorted(byn)
    ms, ses = [], []
    for n in ns:
        m, se = mean_se(list(byn[n].values()))
        ms.append(m)
        ses.append(se if se == se else 0.0)
    return ns, ms, ses


def paired_slope(byn, lo, hi):
    """mean_seed [ lambda(hi) - lambda(lo) ], seed-clustered (scorer method)."""
    seeds = sorted(set(byn.get(lo, {})) & set(byn.get(hi, {})))
    diffs = [byn[hi][s] - byn[lo][s] for s in seeds]
    m, se = mean_se(diffs)
    z = m / se if se and se == se and se > 0 else float("nan")
    return m, se, z, len(seeds), diffs


def bayes_by_n(trials, field="bayes"):
    out: dict[int, list[float]] = {}
    for t in trials:
        if t["arm"] == "b":
            v = t.get(field)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out.setdefault(t["n"], []).append(float(v))
    return {n: mean_se(v)[0] for n, v in sorted(out.items())}


# =====================================================================  F11
# "The domain boundary" -- the hero figure.
NG_FILES = [("echo_numgame_q8.json", "sample 1"), ("echo_numgame_q8_rep.json", "sample 2")]


def fig_f11():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                   gridspec_kw={"width_ratios": [1.28, 1.0]})

    # ---- Panel A: lambda(n) contraction curves ----
    # number game, no-think (primary; more surviving seeds) -- both samples
    ng_off, ng_on = [], []
    bayes_pool: dict[int, list[float]] = {}
    for fname, lab in NG_FILES:
        off = learned(load_cond(fname, "Qwen/Qwen3-8B:off")["trials"])
        on = learned(load_cond(fname, "Qwen/Qwen3-8B:on")["trials"])
        ng_off.append((lab, armb_by_n(off)))
        ng_on.append((lab, armb_by_n(on)))
        for t in off + on:
            if t["arm"] == "b" and t.get("bayes") is not None:
                bayes_pool.setdefault(t["n"], []).append(float(t["bayes"]))

    # primary: no-think, two samples, solid red, with seed-clustered SE bars
    off_markers = ["o", "s"]
    off_slopes = []
    handles = []
    for (lab, byn), mk in zip(ng_off, off_markers):
        ns, ms, ses = per_n_mean_se(byn)
        m, se, z, nseed, _ = paired_slope(byn, 1, 32)
        off_slopes.append((m, z, nseed))
        h = axA.errorbar(ns, ms, yerr=ses, color=RED, ls="-", lw=2.0, marker=mk, ms=6,
                         capsize=2.5, markeredgecolor="white", markeredgewidth=0.7, zorder=6,
                         label=f"number game  no-think ({lab})")
        handles.append(h)

    # secondary: think, two samples, dashed orange (fewer low-n seeds survive)
    for (lab, byn), mk in zip(ng_on, off_markers):
        ns, ms, _ = per_n_mean_se(byn)
        h, = axA.plot(ns, ms, color=ORANGE, ls="--", lw=1.5, marker=mk, ms=4.5, alpha=0.9,
                     markeredgecolor="white", markeredgewidth=0.5, zorder=5,
                     label=f"number game  think ({lab})")
        handles.append(h)

    # exact Bayes (Tenenbaum size principle) reference -- pooled over all NG arm-B trials
    b_ns = sorted(bayes_pool)
    b_ys = [mean_se(bayes_pool[n])[0] for n in b_ns]
    h, = axA.plot(b_ns, b_ys, color=INK_MUTED, ls=(0, (6, 2)), lw=1.8, marker="^", ms=5,
                 markerfacecolor="white", zorder=4, label="exact Bayes (size principle)")
    handles.append(h)

    # Boolean instrument, SAME model -- lambda flat in n (the contrast)
    bool_off = learned(load_cond("echo_think_qwen3_8b_v2.json", "Qwen/Qwen3-8B:on")["trials"])
    bbyn = armb_by_n(bool_off)
    bns, bms, bses = per_n_mean_se(bbyn)
    bm, bse, bz, bnseed, _ = paired_slope(bbyn, 4, 32)
    h = axA.errorbar(bns, bms, yerr=bses, color=BLUE, ls=":", lw=2.2, marker="D", ms=6,
                     capsize=2.5, markeredgecolor="white", markeredgewidth=0.7, zorder=6,
                     label="Boolean concepts (same model): flat")
    handles.append(h)

    axA.set_xscale("log", base=2)
    axA.set_xticks([1, 2, 4, 8, 16, 32])
    axA.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    axA.set_ylim(-0.03, 1.0)
    axA.set_xlabel("number of demonstrations $n$  (log scale)")
    axA.set_ylabel(r"$\lambda(n)$ = P(judged in-concept), join-ward position")
    axA.set_title("(A) same weights, two evidence laws", fontsize=11, color=INK_SECONDARY)
    axA.annotate("Boolean: $\\lambda$ stays flat\n(no version-space contraction)",
                 xy=(16, 0.53), xytext=(4.4, 0.80), fontsize=8.2, color=BLUE,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    axA.annotate("numbers: $\\lambda$ contracts\ntoward exact Bayes",
                 xy=(16, 0.04), xytext=(6.0, 0.30), fontsize=8.2, color=RED,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    axA.legend(handles=handles, fontsize=7.2, frameon=False, loc="upper right", ncol=1,
               handlelength=2.2, labelspacing=0.35)

    # ---- Panel B: stratum / level profile -- rule vs graded ----
    # number concepts: pooled no-think arm-A stratum lambdas (both samples)
    ng_a = []
    for fname, _ in NG_FILES:
        ng_a += [t for t in learned(load_cond(fname, "Qwen/Qwen3-8B:off")["trials"])
                 if t["arm"] == "a"]
    ng_vals = [mean_se([t[f] for t in ng_a])
               for f in ("lambda_in", "lambda_off", "lambda_broad")]
    # Boolean concepts: same model arm-A per-level profile (graded staircase)
    bool_a = [t for t in learned(load_cond("echo_think_qwen3_8b_v2.json",
              "Qwen/Qwen3-8B:on")["trials"]) if t["arm"] == "a"]
    bl_vals = [mean_se([t[f"lambda{j}"] for t in bool_a]) for j in (1, 2, 3)]

    groups = [
        ("Number concepts", ["in-rule\n(far)", "off-rule\n(near)", "broader\nrule"],
         ng_vals, RED),
        ("Boolean concepts", ["1 of 4\nmatch", "2 of 4\nmatch", "3 of 4\nmatch"],
         bl_vals, BLUE),
    ]
    x = 0
    xticks, xticklabels = [], []
    for gi, (gname, labs, vals, col) in enumerate(groups):
        xs = [x + i for i in range(3)]
        ms = [v[0] for v in vals]
        ses = [v[1] if v[1] == v[1] else 0.0 for v in vals]
        axB.bar(xs, ms, width=0.82, color=col, edgecolor="white", linewidth=0.8,
                yerr=ses, capsize=3, ecolor=INK_SECONDARY, error_kw=dict(elinewidth=1.2),
                zorder=5)
        for xi, m, se in zip(xs, ms, ses):
            axB.text(xi, m + se + 0.03, f"{m:.2f}", ha="center", fontsize=8.6,
                     color=INK_PRIMARY, fontweight="bold")
        xticks += xs
        xticklabels += labs
        axB.text(x + 1, 1.05, gname, ha="center", fontsize=9.6, fontweight="bold",
                 color=col)
        x += 4

    axB.set_xticks(xticks)
    axB.set_xticklabels(xticklabels, fontsize=8.0)
    axB.set_ylim(0, 1.16)
    axB.set_ylabel(r"$\lambda$ = P(judged in-concept)")
    axB.set_title("(B) rule-like step vs. graded staircase", fontsize=11,
                  color=INK_SECONDARY)
    axB.annotate("off-rule but numerically\nadjacent → rejected:\na rule, not proximity",
                 xy=(1, ng_vals[1][0] + 0.02), xytext=(1.5, 0.66), fontsize=7.8,
                 color=RED, ha="center", va="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

    fig.suptitle("F11.  The domain boundary: the same Qwen3-8B runs rule-like "
                 "size-principle induction on Tenenbaum's number game,\n"
                 "but graded similarity with no contraction on Boolean attribute "
                 "concepts — uninstructed, in both decode modes",
                 fontsize=12.4, fontweight="bold", y=1.005)
    savefig(fig, OUT / "F11_domain_boundary")

    return dict(off_slopes=off_slopes, bool_slope=(bm, bz, bnseed),
                ng_strata=[(v[0]) for v in ng_vals], bool_levels=[v[0] for v in bl_vals])


# =====================================================================  F13
# "Talk rules, compute similarity" -- verbal report vs decision policy.

# single-attribute rule-election language (automated keyword coder over the stored
# trace text; calibrated against the prereg's manual read of 3-11/20 -- see the
# figure caption + final report). Applied to the 'think' + 'final' text of each
# sampled trace.
import re

_ELECT_PATTERNS = [
    r"the (?:key|critical|deciding|determining|main|important|only|distinguishing) "
    r"(?:factor|attribute|feature)",
    r"(?:factor|attribute|feature) (?:here )?(?:is|seems|might be|could be)",
    r"the (?:pattern|rule) (?:is|seems|appears|might be|could be|would be)",
    r"(?:based on|determined by|depends on|comes down to) (?:the )?"
    r"(?:shape|texture|size|color|colour)",
    r"(?:shape|texture|size|color|colour) (?:is|as) the "
    r"(?:only|key|critical|deciding|main|determining)",
    r"(?:shape|texture|size|color|colour) (?:is|might be|seems to be|could be) "
    r"(?:the )?(?:critical|key|deciding|only|determining|important)",
    r"associated with",
    r"(?:so|then) (?:it|the (?:item|label)) (?:should|would|must) be",
]
_ELECT_RE = [re.compile(p) for p in _ELECT_PATTERNS]


def elect_fraction(cond: dict) -> tuple[int, int]:
    traces = cond.get("traces", [])
    hits = 0
    for tr in traces:
        txt = (tr.get("think", "") + " " + tr.get("final", "")).lower()
        if any(rx.search(txt) for rx in _ELECT_RE):
            hits += 1
    return hits, len(traces)


def rule_committed_fraction(cond: dict) -> tuple[int, int]:
    a = [t for t in learned(cond["trials"]) if t["arm"] == "a"]
    return sum(t["rule_consistency"] >= 0.9 for t in a), len(a)


def fig_f13():
    conds = [
        ("Qwen3-8B\n(RL-trained, thinking)", "echo_think_qwen3_8b_v2.json",
         "Qwen/Qwen3-8B:on"),
        ("Qwen2.5-14B\n(non-reasoning control)", "echo_think_qwen25_14b.json",
         "Qwen/Qwen2.5-14B-Instruct:none"),
    ]
    verbal, policy, denom = [], [], []
    for _, fname, key in conds:
        c = load_cond(fname, key)
        eh, en = elect_fraction(c)
        ph, pn = rule_committed_fraction(c)
        verbal.append(eh / en if en else 0.0)
        policy.append(ph / pn if pn else 0.0)
        denom.append((eh, en, ph, pn))

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(len(conds))
    w = 0.36
    b1 = ax.bar(x - w / 2, verbal, width=w, color=VIOLET, edgecolor="white",
                linewidth=0.8, zorder=5,
                label="verbal report: trace states a single-attribute rule")
    b2 = ax.bar(x + w / 2, policy, width=w, color=RED, edgecolor="white",
                linewidth=0.8, zorder=5,
                label="decision policy: acceptances are rule-consistent ($\\geq$0.9)")

    for xi, (v, p, (eh, en, ph, pn)) in zip(x, zip(verbal, policy, denom)):
        ax.text(xi - w / 2, v + 0.015, f"{v:.0%}\n({eh}/{en})", ha="center",
                fontsize=8.6, color=VIOLET, fontweight="bold")
        ax.text(xi + w / 2, p + 0.015, f"{p:.0%}\n({ph}/{pn})", ha="center",
                fontsize=8.6, color=RED, fontweight="bold")

    # the "gap" annotation on the thinking model
    ax.annotate("", xy=(0 + w / 2, policy[0] + 0.02), xytext=(0 - w / 2, verbal[0] - 0.01),
                arrowprops=dict(arrowstyle="<->", color=INK_SECONDARY, lw=1.3))
    ax.text(0.30, 0.335, "talk ≠ do", ha="left", va="center",
            fontsize=11, color=INK_SECONDARY, fontweight="bold", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conds], fontsize=9.5)
    ax.set_ylim(0, 0.62)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    ax.set_ylabel("fraction of traces / trials")
    ax.legend(fontsize=8.4, frameon=False, loc="upper right")
    ax.set_title("F13.  Talk rules, compute similarity: the thinking trace names a rule,\n"
                 "the acceptance policy is graded similarity (never a meet)",
                 fontsize=12.0, fontweight="bold")
    ax.text(0.5, -0.15,
            "Verbal = automated keyword coding of single-factor rule-election language over the "
            "20 sampled traces per condition\n(prereg manual read: 3–11/20 election, "
            "3–10/20 full-conjunction). Policy = fraction of learned arm-A trials whose best "
            "single-attribute\nrule explains ≥ 90% of acceptances (jthink scorer).",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.9, color=INK_MUTED)
    savefig(fig, OUT / "F13_talk_rules_compute_similarity")

    return dict(verbal=verbal, policy=policy, denom=denom)


# =====================================================================  F14
# "The elicitation ladder" -- three levels of Boolean-concept instruction.
def fig_f14():
    # (1) uninstructed
    unins = learned(load_cond("echo_think_qwen3_8b_v2.json", "Qwen/Qwen3-8B:on")["trials"])
    # (2) meet-instructed
    meet = learned(load_cond("echo_think_qwen3_8b_meetinstr_v2.json",
                             "Qwen/Qwen3-8B:on")["trials"])
    # (3) size-principle-instructed -- pooled over the two pre-registered samples
    sizep = []
    for fn in ("echo_think_d_sizep.json", "echo_think_d2_sizep.json"):
        sizep += learned(load_cond(fn)["trials"])

    def slope_pooled(files, lo, hi):
        diffs = []
        for fn in files:
            byn = armb_by_n(learned(load_cond(fn)["trials"]))
            diffs += paired_slope(byn, lo, hi)[4]
        m, se = mean_se(diffs)
        return m, se, (m / se if se and se > 0 else float("nan")), len(diffs)

    unins_byn = armb_by_n(unins)
    meet_byn = armb_by_n(meet)
    sizep_byn = armb_by_n(sizep)

    s_unins = paired_slope(unins_byn, 4, 32)
    s_meet = paired_slope(meet_byn, 4, 32)
    s_sizep = slope_pooled(("echo_think_d_sizep.json", "echo_think_d2_sizep.json"), 4, 32)

    # exact Bayes reference (Boolean size principle) -- pooled over sizep arm-B trials
    bayes_ref = bayes_by_n(sizep)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.9),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---- Panel A: lambda(n) -- the statistic (contraction dynamics) ----
    series = [
        ("(1) uninstructed", unins_byn, BLUE, "-", "o", s_unins[0], s_unins[2]),
        ("(2) meet-instructed", meet_byn, VIOLET, "-", "s", s_meet[0], s_meet[2]),
        ("(3) size-principle-instructed", sizep_byn, GREEN, "-", "D",
         s_sizep[0], s_sizep[2]),
    ]
    handlesA = []
    for lab, byn, col, ls, mk, slope, z in series:
        ns, ms, ses = per_n_mean_se(byn)
        h = axA.errorbar(ns, ms, yerr=ses, color=col, ls=ls, lw=2.0, marker=mk, ms=6,
                         capsize=2.5, markeredgecolor="white", markeredgewidth=0.7, zorder=6,
                         label=f"{lab}  (slope {slope:+.3f}, z={z:+.2f})")
        handlesA.append(h)
    bns = sorted(bayes_ref)
    h, = axA.plot(bns, [bayes_ref[n] for n in bns], color=INK_MUTED, ls=(0, (6, 2)), lw=1.8,
                 marker="^", ms=5, markerfacecolor="white", zorder=4,
                 label="exact Bayes (size principle)")
    handlesA.append(h)

    axA.set_xscale("log", base=2)
    axA.set_xticks([4, 8, 16, 32])
    axA.set_xticklabels(["4", "8", "16", "32"])
    axA.set_ylim(-0.03, 0.72)
    axA.set_xlabel("number of demonstrations $n$  (log scale)")
    axA.set_ylabel(r"$\lambda(n)$ = join-ward position (Boolean, $k$=2)")
    axA.set_title("(A) the statistic: contraction appears only when the size\n"
                  "principle is stated outright", fontsize=10.5, color=INK_SECONDARY)
    axA.legend(handles=handlesA, fontsize=7.6, frameon=False, loc="upper right",
               labelspacing=0.4)

    # ---- Panel B: arm-A lambda-bar -- the operation (level collapse) ----
    la_unins = mean_se([t["lambda"] for t in unins if t["arm"] == "a"])
    la_meet = mean_se([t["lambda"] for t in meet if t["arm"] == "a"])
    la_sizep = mean_se([t["lambda"] for t in sizep if t["arm"] == "a"])
    bars = [("(1)\nuninstructed", la_unins, BLUE),
            ("(2)\nmeet-instr.", la_meet, VIOLET),
            ("(3)\nsizep-instr.", la_sizep, GREEN)]
    xs = np.arange(len(bars))
    ms = [b[1][0] for b in bars]
    ses = [b[1][1] if b[1][1] == b[1][1] else 0.0 for b in bars]
    axB.bar(xs, ms, width=0.66, color=[b[2] for b in bars], edgecolor="white",
            linewidth=0.8, yerr=ses, capsize=3.5, ecolor=INK_SECONDARY,
            error_kw=dict(elinewidth=1.3), zorder=5)
    for xi, m, se in zip(xs, ms, ses):
        axB.text(xi, m + se + 0.02, f"{m:.3f}", ha="center", fontsize=9.2,
                 color=INK_PRIMARY, fontweight="bold")
    axB.axhline(0.5, color=INK_MUTED, ls=(0, (3, 3)), lw=1.1, zorder=1)
    axB.text(2.42, 0.5, "flat / uncommitted", fontsize=7.4, color=INK_MUTED,
             ha="right", va="bottom")
    axB.annotate("meet instruction\ncollapses the level\n(operation elicited)",
                 xy=(1.18, la_meet[0] + 0.02), xytext=(1.15, 0.62), fontsize=7.8,
                 color=VIOLET, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1.0))
    axB.set_xticks(xs)
    axB.set_xticklabels([b[0] for b in bars], fontsize=8.6)
    axB.set_ylim(0, 0.78)
    axB.set_ylabel(r"arm-A $\bar\lambda$ (overall join-ward level)")
    axB.set_title("(B) the operation: instruction collapses the\nlevel with "
                  "calibration intact", fontsize=10.5, color=INK_SECONDARY)

    fig.suptitle("F14.  The elicitation ladder (Qwen3-8B, Boolean concepts): the intersection "
                 "operation is elicitable (B), but the\nevidence statistic moves only when the "
                 "size principle is stated outright — and then only shallowly (A)",
                 fontsize=12.2, fontweight="bold", y=1.005)
    savefig(fig, OUT / "F14_elicitation_ladder")

    return dict(unins=(la_unins[0], s_unins[0], s_unins[2]),
                meet=(la_meet[0], s_meet[0], s_meet[2]),
                sizep=(la_sizep[0], s_sizep[0], s_sizep[2], s_sizep[3]))


def main():
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    r11 = fig_f11()
    r13 = fig_f13()
    r14 = fig_f14()
    print("F11:", r11)
    print("F13:", r13)
    print("F14:", r14)
    print("wrote F11_domain_boundary, F13_talk_rules_compute_similarity, "
          "F14_elicitation_ladder (.png + .pdf) to", OUT)


if __name__ == "__main__":
    main()
