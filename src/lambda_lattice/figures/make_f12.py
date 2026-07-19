"""F12 — "The boundary map: one default, five inductive identities."

Adapted (packaging only) from the lab repo
``release/lambda-icl/src/make_f12_boundary_map.py``. Only imports and the
data/figures path resolution were changed: the frozen number-game scorer is now
imported from this package's own ``lambda_lattice.scoring.numbers`` (extracted
verbatim from ``frontier_lab/ng_analyze.py``) instead of via sys.path, and the
raw data files are read from the shipped ``data/`` dir. Every gate, seed, and
statistic is unchanged.

The paper's second hero figure. Two panels tell the boundary story with the
SAME five model families (Qwen, Llama, Mistral, Gemma, OLMo):

  Panel A (Boolean domain — homogeneity): the five families' graded per-level
    lambda_j profiles at k=4 all cluster on the additive staircase
    lambda_j ~ j/k (0.25 / 0.50 / 0.75). Five families, one shared law.

  Panel B (number game — heterogeneity): a positioning scatter of the same five
    families, x = arm-A Dfit (rule-vs-similarity), y = arm-B paired contraction
    slope lambda(32) - lambda(1). Two pre-registered samples per family
    (seeds 0-35 and seeds 36-71) are drawn as two points joined by a thin line.
    Five families, five distinct inductive identities.

Every plotted number is RECOMPUTED here from the raw trials with the SAME gates
and seed-clustered statistics as the frozen pre-registered scorers:

  * Panel A: learned trial := sanity >= 0.75 (jbias inclusion rule); per-seed
    level means (pooling formats within a seed) via loader.per_seed_level_means;
    seed-clustered SE := SD(per-seed values) / sqrt(n_seeds).
  * Panel B: gated trial := sanity >= 0.75 AND parse_rate >= 0.8; Dfit,
    contraction slope and Bayes-reference slope are taken by importing the
    frozen scorer lambda_lattice.scoring.numbers directly (no re-implementation),
    so the numbers match its stdout exactly.

Reference markers in Panel B (exact-Bayes and numeric-proximity oracles) are
computed from the oracle conditions in echo_numgame_mockcheck.json with the same
scorer — never hardcoded.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from .style import (BLUE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, FAMILY_COLOR,
                    MODEL_SHORT, savefig, use_style)
from . import refs
from .loader import collect_records, per_seed_level_means
from .._paths import data_dir, figures_dir
# the frozen scorer (extracted verbatim from frontier_lab/ng_analyze.py) so
# Panel-B numbers are literally its computation
from ..scoring import numbers as ng

DATA = data_dir()
OUT = figures_dir()

GATE_SANITY = 0.75


def mean_se(xs):
    xs = [float(x) for x in xs
          if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


# ============================================================ Panel A data
# k=4 Boolean per-level lambda_j, one representative instruct model per family.
K4_FAMILIES = [
    ("Qwen", "jbias_k4.json", "Qwen/Qwen2.5-7B-Instruct"),
    ("Llama", "jbias_k4_llama.json", "unsloth/Meta-Llama-3.1-8B-Instruct"),
    ("Mistral", "jbias_k4_mistral.json", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("Gemma", "jbias_k4_gemma.json", "unsloth/gemma-2-9b-it"),
    ("OLMo", "jbias_k4_olmo.json", "allenai/OLMo-2-1124-13B-Instruct"),
]
LEVELS = refs.ladder_levels(4)  # [1, 2, 3]


def panelA_profiles():
    out = []
    for fam, fname, model in K4_FAMILIES:
        records = collect_records([fname], krel_filter=4,
                                  sanity_thresh=GATE_SANITY)
        means, ses, nseeds = [], [], 0
        for j in LEVELS:
            per_seed = per_seed_level_means(records, model, j)
            m, se = mean_se(list(per_seed.values()))
            means.append(m)
            ses.append(se)
            nseeds = max(nseeds, len(per_seed))
        out.append(dict(fam=fam, model=model, means=means, ses=ses,
                        nseeds=nseeds))
    return out


# ============================================================ Panel B data
# number game, :off (no-think) condition for every family, two prereg samples.
NG_FAMILIES = [
    ("Qwen", "echo_numgame_q8.json", "echo_numgame_q8_rep.json",
     "Qwen/Qwen3-8B:off"),
    ("Llama", "echo_numgame_llama.json", "echo_numgame_llama_rep.json",
     "unsloth/Meta-Llama-3.1-8B-Instruct:off"),
    ("Mistral", "echo_numgame_mistral.json", "echo_numgame_mistral_rep.json",
     "mistralai/Mistral-7B-Instruct-v0.3:off"),
    ("Gemma", "echo_numgame_gemma.json", "echo_numgame_gemma_rep.json",
     "unsloth/gemma-2-9b-it:off"),
    ("OLMo", "echo_numgame_olmo.json", "echo_numgame_olmo_rep.json",
     "allenai/OLMo-2-1124-13B-Instruct:off"),
]


def _score_off(fname, key):
    """Run the frozen scorer on one file/condition -> its result dict."""
    conds = dict(ng.load_conditions(str(DATA / fname)))
    return ng.score_condition(key, conds[key], fname)


def panelB_points():
    out = []
    for fam, f1, f2, key in NG_FAMILIES:
        samples = []
        for lab, fn in (("sample 1", f1), ("sample 2", f2)):
            r = _score_off(fn, key)
            samples.append(dict(
                lab=lab,
                dfit=r["dfit"], dfit_se=r["dfit_se"],
                slope=r["slope"], slope_se=r["slope_se"],
                bayes_slope=r["slope_bayes"][0],
            ))
        out.append(dict(fam=fam, samples=samples))
    return out


def oracle_refs():
    """exact-Bayes and proximity oracle (Dfit, slope) from the mockcheck file."""
    conds = dict(ng.load_conditions(str(DATA / "echo_numgame_mockcheck.json")))
    refs_ = {}
    for key, tag in (("bayes:none", "bayes"), ("prox:none", "prox")):
        r = ng.score_condition(key, conds[key], "echo_numgame_mockcheck.json")
        refs_[tag] = (r["dfit"], r["slope"])
    return refs_


# ============================================================ figure
def main():
    use_style()
    OUT.mkdir(parents=True, exist_ok=True)

    profiles = panelA_profiles()
    points = panelB_points()
    orc = oracle_refs()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.8, 5.4),
                                   gridspec_kw={"width_ratios": [1.0, 1.18]})

    # ---------------------------------------------------------- Panel A
    # additive j/k prediction line (the shared law)
    add_y = [refs.additive(j, 4) for j in LEVELS]
    axA.plot(LEVELS, add_y, color=INK_SECONDARY, ls=(0, (5, 2)), lw=1.6,
             marker=None, zorder=3,
             label="additive prediction  $\\lambda_j = j/k$")
    # meet / join envelopes as faint context
    axA.plot(LEVELS, [0, 0, 0], color=INK_MUTED, ls=(0, (1, 2)), lw=1.0,
             zorder=1, label="meet (conjunctive) = 0")
    axA.plot(LEVELS, [1, 1, 1], color=INK_MUTED, ls=(0, (1, 2)), lw=1.0,
             zorder=1, label="join (disjunctive) = 1")

    markers = {"Qwen": "o", "Llama": "s", "Mistral": "^", "Gemma": "D",
               "OLMo": "v"}
    for p in profiles:
        col = FAMILY_COLOR[p["fam"]]
        # small horizontal jitter so overlapping series stay legible
        jit = (["Qwen", "Llama", "Mistral", "Gemma", "OLMo"].index(p["fam"])
               - 2) * 0.035
        xs = [j + jit for j in LEVELS]
        axA.errorbar(xs, p["means"], yerr=p["ses"], color=col, ls="-", lw=1.7,
                     marker=markers[p["fam"]], ms=6.5, capsize=2.5,
                     markeredgecolor="white", markeredgewidth=0.7, zorder=6,
                     label=f"{MODEL_SHORT[p['model']]}  ({p['fam']})")

    axA.set_xticks(LEVELS)
    axA.set_xticklabels([f"$j$={j}\n({j}/4)" for j in LEVELS])
    axA.set_xlim(0.6, 3.4)
    axA.set_ylim(-0.05, 1.05)
    axA.set_xlabel("Boolean concept level $j$  (of $k$=4 relevant attributes)")
    axA.set_ylabel(r"$\lambda_j$ = P(judged in-concept), join-ward position")
    axA.set_title("(A) Boolean concepts: five families, one law\n"
                  "(all track the additive staircase $j/k$)",
                  fontsize=10.6, color=INK_SECONDARY)
    axA.legend(fontsize=7.3, frameon=False, loc="upper left",
               handlelength=2.0, labelspacing=0.3)
    axA.annotate("graded staircase,\nno version-space contraction",
                 xy=(2, 0.50), xytext=(1.62, 0.86), fontsize=7.8,
                 color=INK_SECONDARY, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=0.9))

    # ---------------------------------------------------------- Panel B
    # quadrant guides
    axB.axvline(0.0, color=INK_MUTED, ls=(0, (2, 2)), lw=1.0, zorder=1)
    axB.axhline(0.0, color=INK_MUTED, ls=(0, (2, 2)), lw=1.0, zorder=1)

    # oracle reference markers (faint stars)
    for tag, (dx, sy), name in (
        ("bayes", orc["bayes"], "exact-Bayes\nreference"),
        ("prox", orc["prox"], "proximity\nreference"),
    ):
        axB.scatter([dx], [sy], marker="*", s=320, color=INK_MUTED,
                    edgecolor="white", linewidth=0.8, alpha=0.85, zorder=4)
    # oracle-star labels (placed by hand to clear the data cloud)
    bx, bsy = orc["bayes"]
    px, psy = orc["prox"]
    axB.annotate("exact-Bayes\nreference", xy=(bx, bsy), xytext=(bx, bsy + 0.085),
                 fontsize=7.2, color=INK_SECONDARY, ha="center", va="bottom",
                 style="italic")
    axB.annotate("proximity\nreference", xy=(px, psy), xytext=(px, psy - 0.055),
                 fontsize=7.2, color=INK_SECONDARY, ha="center", va="top",
                 style="italic")

    # per-family display: short label + hand-placed offset, ha, va
    FAM_LABEL = {
        "Qwen": ("Qwen3-8B", (0.0, -0.075), "center", "top"),
        "Llama": ("Llama-3.1-8B", (0.0, -0.085), "center", "top"),
        "Mistral": ("Mistral-7B", (0.055, 0.0), "left", "center"),
        "Gemma": ("Gemma-2-9B", (-0.03, 0.02), "right", "bottom"),
        "OLMo": ("OLMo-13B", (0.06, 0.0), "left", "center"),
    }
    markers_pt = {"sample 1": "o", "sample 2": "s"}
    for p in points:
        col = FAMILY_COLOR[p["fam"]]
        xs = [s["dfit"] for s in p["samples"]]
        ys = [s["slope"] for s in p["samples"]]
        axB.plot(xs, ys, color=col, ls="-", lw=1.1, alpha=0.7, zorder=5)
        for s in p["samples"]:
            axB.errorbar([s["dfit"]], [s["slope"]],
                         xerr=[s["dfit_se"]], yerr=[s["slope_se"]],
                         color=col, marker=markers_pt[s["lab"]], ms=8,
                         capsize=2.5, markeredgecolor="white",
                         markeredgewidth=0.8, elinewidth=1.1, zorder=6)
        # family label anchored at the midpoint of the two samples
        lab, (dxo, dyo), ha, va = FAM_LABEL[p["fam"]]
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        axB.annotate(lab, xy=(mx, my), xytext=(mx + dxo, my + dyo),
                     fontsize=8.4, color=col, fontweight="bold", ha=ha, va=va)

    # sample-marker legend (shape encodes which pre-registered sample)
    from matplotlib.lines import Line2D
    axB.legend(handles=[
        Line2D([0], [0], marker="o", color=INK_SECONDARY, ls="", ms=7,
               markerfacecolor=INK_SECONDARY, label="sample 1 (seeds 0-35)"),
        Line2D([0], [0], marker="s", color=INK_SECONDARY, ls="", ms=7,
               markerfacecolor=INK_SECONDARY, label="sample 2 (seeds 36-71)"),
    ], fontsize=7.2, frameon=False, loc="lower left", handletextpad=0.4)

    # quadrant descriptive labels
    qkw = dict(fontsize=7.6, style="italic", color=INK_SECONDARY, alpha=0.95)
    axB.text(0.50, -0.63, "Bayes rule induction\n(rule + contraction)",
             ha="center", va="center", **qkw)
    axB.text(0.245, -0.02, "rule selection,\nfrozen evidence",
             ha="center", va="center", **qkw)
    axB.text(-0.19, -0.55, "global tightening,\nno rules", ha="center",
             va="center", **qkw)
    axB.text(-0.30, 0.19, "similarity\n+ expansion", ha="center",
             va="center", **qkw)

    axB.set_xlim(-0.42, 0.66)
    axB.set_ylim(-0.82, 0.30)
    axB.set_xlabel(r"arm-A $\Delta$fit = fit(Bayes) $-$ fit(proximity)"
                   "   (rule $\\rightarrow$   |   $\\leftarrow$ similarity)")
    axB.set_ylabel(r"arm-B contraction slope  $\lambda(32)-\lambda(1)$"
                   "   ($\\downarrow$ contraction)")
    axB.set_title("(B) Number concepts: five families, five identities\n"
                  "(same weights, no instruction)", fontsize=10.6,
                  color=INK_SECONDARY)

    fig.suptitle("F12.  The boundary map — one default, five inductive "
                 "identities: the same five families share a single graded law "
                 "on Boolean\nconcepts (A: all on the $j/k$ staircase), yet split "
                 "into five distinct inductive identities on Tenenbaum's number "
                 "game (B)",
                 fontsize=12.2, fontweight="bold", y=1.01)

    fig.text(0.5, -0.045,
             "Panel A: per-level $\\lambda_j$ over sanity$\\geq$0.75 trials, "
             "seed-clustered SE (per-seed means pooling formats). "
             "Panel B: $\\Delta$fit and contraction slope recomputed by the frozen "
             "pre-registered scorer (ng_analyze.py; sanity$\\geq$0.75, "
             "parse$\\geq$0.8, seed-clustered SE) on the :off (no-think) "
             "condition; two points = two pre-registered samples "
             "(seeds 0-35, 36-71). Panel-B verdicts are two-sample "
             "pre-registered; quadrant labels are descriptive.",
             ha="center", va="top", fontsize=6.8, color=INK_MUTED)

    savefig(fig, OUT / "F12_boundary_map")

    # -------------------------------------------------- cross-check report
    print("\nPanel A  (k=4 Boolean per-level lambda_j, seed-clustered):")
    for p in profiles:
        prof = "  ".join(f"j{j}={m:.3f}+-{se:.3f}"
                         for j, m, se in zip(LEVELS, p["means"], p["ses"]))
        print(f"  {p['fam']:8s} ({p['nseeds']} seeds)  {prof}")

    print("\nPanel B  (number game :off, Dfit / slope, two samples):")
    for p in points:
        s1, s2 = p["samples"]
        print(f"  {p['fam']:8s}  Dfit {s1['dfit']:+.3f}/{s2['dfit']:+.3f}   "
              f"slope {s1['slope']:+.3f}/{s2['slope']:+.3f}   "
              f"(Bayes-ref slope {s1['bayes_slope']:+.3f}/"
              f"{s2['bayes_slope']:+.3f})")
    print(f"\nOracle refs: Bayes (Dfit {orc['bayes'][0]:+.3f}, "
          f"slope {orc['bayes'][1]:+.3f})   "
          f"prox (Dfit {orc['prox'][0]:+.3f}, slope {orc['prox'][1]:+.3f})")
    print("\nwrote F12_boundary_map (.png + .pdf) to", OUT)


if __name__ == "__main__":
    main()
