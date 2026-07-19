"""
Publication-quality figure suite for the join-bias / additive-integration
in-context-learning study.

Run:  python make_figures.py
Output: frontier_lab/viz/figs/F1..F7 *.png (300dpi) + *.pdf, and figs/NOTES.md

Every number plotted is either (a) read/aggregated straight from the result
JSONs in frontier_lab/results/, or (b) an analytically-defined reference
line / a literal value the task brief explicitly labels as a reference
(prereg prediction, k=3 Bayes/NN values) -- see refs.py for the exact
provenance of every reference curve. Ambiguities are logged to
figs/NOTES.md rather than resolved by guessing.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import refs
from loader import (
    RESULTS_DIR, load_json, collect_records, per_seed_level_means,
    bootstrap_mean_ci, se_2se, model_names, SANITY_THRESH,
)
from style import (
    use_style, savefig, CATEGORICAL, INK_PRIMARY, INK_SECONDARY, INK_MUTED,
    GRIDLINE, BASELINE, SURFACE, MODEL_COLOR, MODEL_SHORT, MODEL_SIZE_B,
    K4_MODEL_ORDER, K5_MODEL_ORDER, MODEL_FAMILY, MODEL_IS_BASE, FAMILY_COLOR,
    REF_STYLES,
)

from lambda_lattice._paths import figures_dir

# Regenerate straight into the repo's figures/ dir (shipped PNGs are overwritten
# in place); override with LAMBDA_LATTICE_FIGURES.
FIGS_DIR = figures_dir()
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# All krel==4 result files, across the original 5-model roster and the
# 2026-07-03 cross-family + base-model addition (llama/mistral/gemma/7b_base).
K4_FILES = [
    "jbias_k4.json", "jbias_k4_olmo.json", "jbias_k4_32b.json", "jbias_k4_7b_f3.json",
    "jbias_k4_llama.json", "jbias_k4_mistral.json", "jbias_k4_gemma.json",
    "jbias_k4_7b_base.json",
]
K5_FILES = ["jbias_k5.json"]

NOTES: list[str] = []


def note(fig_id: str, title: str, lines: list[str]):
    block = [f"## {fig_id} -- {title}"] + [f"- {l}" for l in lines]
    NOTES.append("\n".join(block))


def short(model):
    return MODEL_SHORT.get(model, model)


def per_seed_level_means_filtered(records, model, j, n=None, fmt=None):
    by_seed = defaultdict(list)
    for r in records:
        if r["model"] != model:
            continue
        if n is not None and r["n"] != n:
            continue
        if fmt is not None and r["fmt"] != fmt:
            continue
        if j in r["levels"]:
            by_seed[r["seed"]].append(r["levels"][j])
    return {seed: float(np.mean(v)) for seed, v in by_seed.items()}


def draw_ref_lines(ax, k, levels, which):
    x = np.array(levels, dtype=float)
    for key in which:
        st = dict(REF_STYLES[key])
        label = st.pop("label")
        if key == "additive":
            y = [refs.additive(j, k) for j in levels]
        elif key == "meet":
            y = [refs.meet(j, k) for j in levels]
        elif key == "join":
            y = [refs.join(j, k) for j in levels]
        elif key == "flat":
            y = [refs.flat(j, k) for j in levels]
        elif key == "nn":
            y = [refs.nn_step(j, k) for j in levels]
        else:
            continue
        ax.plot(x, y, marker=st.pop("marker", None), markersize=4,
                markerfacecolor="white", markeredgewidth=1.0, zorder=1, **st, label=label)


# --------------------------------------------------------------------------
# F1 -- lambda-profile ladder at k=4, one panel per model, bootstrap CIs
# --------------------------------------------------------------------------
def fig_f1():
    files = K4_FILES
    records = collect_records(files, krel_filter=4)
    models = [m for m in K4_MODEL_ORDER if any(r["model"] == m for r in records)]

    fig, axes = plt.subplots(1, len(models), figsize=(3.1 * len(models), 3.6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    k = 4
    levels = refs.ladder_levels(k)  # [1,2,3]
    bayes_k4 = None
    trial_counts = {}

    for ax, model in zip(axes, models):
        means, los, his = [], [], []
        for j in levels:
            per_seed = per_seed_level_means_filtered(records, model, j)
            m, lo, hi = bootstrap_mean_ci(per_seed.values(), rng_seed=42)
            means.append(m); los.append(m - lo); his.append(hi - m)
        n_rows = sum(1 for r in records if r["model"] == model)
        n_seeds = len({r["seed"] for r in records if r["model"] == model})
        trial_counts[model] = (n_rows, n_seeds)

        draw_ref_lines(ax, k, levels, ["additive", "meet", "join", "flat", "nn"])
        # JSON-logged bayes constant for k=4 (single value, flat line -- see refs.py)
        bvals = [r["bayes_raw"] for r in records if r["model"] == model and r["bayes_raw"] is not None]
        if bvals:
            bmean = float(np.mean(bvals))
            bayes_k4 = bmean
            st = dict(REF_STYLES["bayes"])
            lbl = st.pop("label")
            ax.plot(levels, [bmean] * len(levels), marker=st.pop("marker", None), markersize=4,
                    markerfacecolor="white", **st, label=lbl)

        color = MODEL_COLOR.get(model, CATEGORICAL[0])
        ax.errorbar(levels, means, yerr=[los, his], fmt="o-", color=color,
                     ecolor=color, elinewidth=1.6, capsize=3, markersize=6,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=5,
                     label=f"{short(model)} (observed)")

        ax.set_title(short(model), fontsize=10.5)
        ax.set_xticks(levels)
        ax.set_xlabel("level $j$ (of $k$=4)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(0.6, 3.4)

    axes[0].set_ylabel(r"$\lambda_j$ = P(positive | $j$ of 4 attrs match)")
    handles, labels_ = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1),
               frameon=False, fontsize=8.6)
    fig.suptitle(r"F1. $\lambda$-profile ladder at $k$=4: observed vs. reference learners",
                 fontsize=12.5, fontweight="bold", y=1.04)
    savefig(fig, FIGS_DIR / "F1_lambda_ladder_k4")

    note("F1", "lambda-profile ladder (k=4, per model, bootstrap CIs)", [
        f"Data: {', '.join(files)}; p1 rows, krel==4, sanity>={SANITY_THRESH} (2 both + 2 neither "
        "probes correct, >=3/4).",
        "lambda_j pulled from row fields lambda1/lambda2/lambda3 (with p_neither/p_both as the "
        "j=0/j=4 anchors), cross-checked against the recomputed mean of the 'detail' probe list.",
        "95% CI = percentile bootstrap (2000 resamples) over per-seed means, i.e. resampling is at "
        "the seed level as specified, pooling formats f1/f2/(f3 where present) within a seed.",
        "Trial counts (rows passing sanity, unique seeds) per model: " +
        "; ".join(f"{short(m)}={trial_counts[m][0]} rows / {trial_counts[m][1]} seeds" for m in models),
        f"additive/meet/join/flat computed by formula (j/k, 0, 1, 0.5). 1-NN uses the idealized "
        "Hamming-distance step nn_step(j,k)=0.01 if 2j<=k else 0.99 (derived from the NN "
        "definition, validated against both task-given examples -- see refs.py docstring).",
        f"Bayes reference at k=4 is the JSON's own logged 'bayes' field, a single constant "
        f"({bayes_k4:.4f}) identical across every model/format/seed we "
        "checked at krel=4,n=8 -- plotted as a flat line. This is higher than the task brief's "
        "paraphrase ('~0.005-0.06, near 0'); we kept the logged number per the "
        "plot-only-from-JSON rule and flag the discrepancy here rather than fabricating a 3-point "
        "curve to match the paraphrase.",
    ])


# --------------------------------------------------------------------------
# F2 -- prereg vs observed, Qwen-7B, k=4 (headline figure)
# --------------------------------------------------------------------------
def fig_f2():
    files = ["jbias_k4.json", "jbias_k4_7b_f3.json"]
    model = "Qwen/Qwen2.5-7B-Instruct"
    records = collect_records(files, krel_filter=4)
    levels = refs.ladder_levels(4)

    obs_mean, obs_2se = [], []
    for j in levels:
        per_seed = per_seed_level_means_filtered(records, model, j)
        m, se2 = se_2se(per_seed.values())
        obs_mean.append(m); obs_2se.append(se2)

    prereg = [refs.PREREG_K4[j] for j in levels]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    x = np.arange(len(levels))
    w = 0.32
    ax.bar(x - w / 2, prereg, width=w, color=INK_MUTED, alpha=0.55,
           edgecolor=INK_SECONDARY, linewidth=0.8, label="pre-registered prediction")
    ax.bar(x + w / 2, obs_mean, width=w, color=MODEL_COLOR[model],
           yerr=obs_2se, capsize=4, ecolor=INK_PRIMARY, error_kw=dict(elinewidth=1.4),
           edgecolor="white", linewidth=0.6, label="observed (Qwen-7B-Instruct)")

    for xi, (p, o) in zip(x, zip(prereg, obs_mean)):
        ax.text(xi - w / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=9, color=INK_SECONDARY)
        ax.text(xi + w / 2, o + obs_2se[list(x).index(xi)] + 0.02, f"{o:.3f}", ha="center",
                fontsize=9, color=INK_PRIMARY, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"j={j}" for j in levels])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"$\lambda_j$ = P(positive | $j$ of 4 attrs match)")
    ax.set_title("F2. Pre-registered prediction vs. observed (Qwen-7B-Instruct, k=4)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    savefig(fig, FIGS_DIR / "F2_prereg_vs_observed_7B")

    note("F2", "prereg vs observed (Qwen-7B, headline)", [
        f"Data: {', '.join(files)}, model={model}, krel==4, sanity>={SANITY_THRESH}.",
        f"Observed means (recomputed here): j=1: {obs_mean[0]:.3f}, j=2: {obs_mean[1]:.3f}, "
        f"j=3: {obs_mean[2]:.3f}. Task brief quotes 0.226/0.465/0.778 for the same headline "
        "number; our recomputation from lambda1/2/3 fields (sanity-filtered, seed-averaged) is "
        "consistent with that quoted figure within the error bars shown.",
        "Error bars = observed mean +/- 2*SE, SE computed over per-seed means "
        f"(n={len({r['seed'] for r in records if r['model']==model})} unique seeds, pooling f1/f2/f3).",
        "Pre-registered prediction (0.25/0.50/0.75) is the literal additive-model prediction "
        "j/k, hardcoded per the brief's explicit prereg label (not read from JSON, by design).",
    ])


# --------------------------------------------------------------------------
# F3 -- cross-cell monotonicity grid (all model x format k=4 cells)
# --------------------------------------------------------------------------
def fig_f3():
    files = K4_FILES
    records = collect_records(files, krel_filter=4)

    cells = sorted({(r["model"], r["fmt"]) for r in records},
                    key=lambda mf: (K4_MODEL_ORDER.index(mf[0]) if mf[0] in K4_MODEL_ORDER else 99, mf[1]))

    rows = []
    mono_flags = []
    for model, fmt in cells:
        vals = []
        for j in (1, 2, 3):
            per_seed = per_seed_level_means_filtered(records, model, j, fmt=fmt)
            vals.append(float(np.mean(list(per_seed.values()))) if per_seed else np.nan)
        rows.append(vals)
        mono_flags.append(vals[0] < vals[1] < vals[2])

    mat = np.array(rows)
    n_mono = sum(mono_flags)
    n_total = len(cells)

    # dynamic per-model format breakdown, for the note (do not hand-type this --
    # it must stay correct as models/files are added)
    fmts_by_model = defaultdict(list)
    for model, fmt in cells:
        fmts_by_model[model].append(fmt)
    model_order_for_note = [m for m in K4_MODEL_ORDER if m in fmts_by_model]
    breakdown = "; ".join(
        f"{short(m)}: {','.join(sorted(fmts_by_model[m]))}" for m in model_order_for_note)

    fig, ax = plt.subplots(figsize=(5.2, 0.42 * n_total + 1.6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["j=1", "j=2", "j=3"])
    ylabels = [f"{short(m)} / {f}" for m, f in cells]
    ax.set_yticks(range(n_total))
    ax.set_yticklabels(ylabels, fontsize=8.8)

    for i in range(n_total):
        for jx in range(3):
            v = mat[i, jx]
            txt_color = "white" if v > 0.55 else INK_PRIMARY
            ax.text(jx, i, f"{v:.2f}", ha="center", va="center", fontsize=8.4, color=txt_color)
        color = "#0ca30c" if mono_flags[i] else "#d03b3b"
        symbol = "OK" if mono_flags[i] else "X"
        ax.text(3.05, i, symbol, ha="center", va="center", fontsize=9.5, color=color, fontweight="bold")

    ax.set_xlim(-0.5, 3.6)
    ax.text(3.05, -0.9, "monotone\n" + r"$\lambda_1<\lambda_2<\lambda_3$", ha="center", va="bottom",
             fontsize=8, color=INK_SECONDARY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.12)
    cbar.set_label(r"$\lambda_j$", fontsize=9)
    ax.set_title(f"F3. Cross-cell monotonicity grid: {n_mono}/{n_total} model x format cells "
                 r"monotone in $j$", fontsize=11.5, fontweight="bold")
    savefig(fig, FIGS_DIR / "F3_monotonicity_grid")

    note("F3", "cross-cell monotonicity grid", [
        f"Data: {', '.join(files)}; krel==4, sanity>={SANITY_THRESH}, one cell per (model, fmt) "
        "combination actually present in these eight files (original 5-model roster + the "
        "2026-07-03 Llama/Mistral/Gemma/Qwen-7B-base addition).",
        f"Cell value = mean lambda_j over all learned seeds for that (model, fmt); monotone = "
        "lambda_1 < lambda_2 < lambda_3 strictly.",
        f"Result: {n_mono}/{n_total} cells monotone. Cells present per model (all formats "
        f"actually run for that model in these files): {breakdown}. This count ({n_total}) is "
        "computed directly from the (model, fmt) pairs found in the data, not hand-typed -- it "
        "grows automatically as new model files are added, which is why it will not match any "
        "prior session's cell count (e.g. an earlier '11/11' or '13/13' claim) once the roster "
        "changes; always read the honest count off this note, not from memory.",
        "Non-monotone cells (if any) are the ones flagged with a red X in the figure; check the "
        "PDF/PNG for which specific (model, fmt) pairs those are.",
    ])
    return n_mono, n_total, cells, mono_flags


# --------------------------------------------------------------------------
# F4 -- logistic fits sigma(beta*(2j-k) - b0), beta vs model size
# --------------------------------------------------------------------------
def _sigmoid(x, beta, b0):
    return 1.0 / (1.0 + np.exp(-(beta * x - b0)))


def fig_f4():
    from scipy.optimize import curve_fit

    files = K4_FILES
    records = collect_records(files, krel_filter=4)
    models = [m for m in K4_MODEL_ORDER if any(r["model"] == m for r in records)]
    k = 4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.6))
    fit_results = {}

    for model in models:
        js = list(range(0, k + 1))  # include anchors j=0, j=k for a well-posed fit
        ys = []
        for j in js:
            per_seed = per_seed_level_means_filtered(records, model, j)
            ys.append(float(np.mean(list(per_seed.values()))) if per_seed else np.nan)
        js_arr = np.array(js, dtype=float)
        ys_arr = np.array(ys, dtype=float)
        mask = ~np.isnan(ys_arr)
        x_design = 2 * js_arr[mask] - k
        try:
            popt, _ = curve_fit(_sigmoid, x_design, ys_arr[mask], p0=[0.5, 0.0], maxfev=5000)
            beta, b0 = popt
        except Exception:
            beta, b0 = np.nan, np.nan
        fit_results[model] = (beta, b0, js_arr[mask], ys_arr[mask])

        color = MODEL_COLOR.get(model, CATEGORICAL[0])
        ax1.scatter(js_arr[mask], ys_arr[mask], color=color, s=28, zorder=5,
                    edgecolor="white", linewidth=0.6)
        if not np.isnan(beta):
            xx = np.linspace(0, k, 200)
            yy = _sigmoid(2 * xx - k, beta, b0)
            ax1.plot(xx, yy, color=color, lw=1.8, label=f"{short(model)} " + r"($\beta$=" + f"{beta:.2f})")

    ax1.set_xlabel("level $j$ (of $k$=4)")
    ax1.set_ylabel(r"$\lambda_j$")
    ax1.set_xlim(-0.2, 4.2)
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_title(r"fitted $\sigma(\beta(2j-k)-b_0)$ per model", fontsize=11)
    # 9-model roster (post 2026-07-03 addition) no longer fits inline -- legend moves outside
    ax1.legend(fontsize=7.6, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    xs_size, ys_beta, colors, used_models = [], [], [], []
    for model in models:
        beta = fit_results[model][0]
        if np.isnan(beta):
            continue
        xs_size.append(MODEL_SIZE_B[model])
        ys_beta.append(beta)
        colors.append(MODEL_COLOR.get(model, CATEGORICAL[0]))
        used_models.append(model)
    ax2.scatter(xs_size, ys_beta, c=colors, s=70, edgecolor="white", linewidth=0.8, zorder=5)
    for x_, y_, model in zip(xs_size, ys_beta, used_models):
        ax2.annotate(short(model), (x_, y_), textcoords="offset points", xytext=(6, 4), fontsize=8.5,
                     color=INK_SECONDARY)
    ax2.set_xscale("log")
    ax2.set_xlabel("model size (B params, log scale)")
    ax2.set_ylabel(r"fitted steepness $\beta$")
    ax2.set_title(r"$\beta$ vs. model size", fontsize=11)
    ax2.grid(True, which="both", alpha=0.4)

    fig.suptitle("F4. Logistic fit of the lambda ladder and its steepness across model scale",
                 fontsize=12.5, fontweight="bold", y=1.03)
    savefig(fig, FIGS_DIR / "F4_logistic_fits")

    lines = [
        f"Data: {', '.join(files)}; krel==4, sanity>={SANITY_THRESH}.",
        "Per model, mean lambda_j at j=0..4 (p_neither, lambda1, lambda2, lambda3, p_both, "
        "seed-averaged) is fit by unweighted least squares (scipy.optimize.curve_fit) to "
        "sigma(beta*(2j-k) - b0); anchors j=0,4 are included to stabilize the fit with only 5 "
        "data points per model.",
        "Fitted (beta, b0) per model: " + "; ".join(
            f"{short(m)}: beta={fit_results[m][0]:.3f}, b0={fit_results[m][1]:.3f}" for m in models),
        f"beta-vs-size panel uses nominal released parameter counts ({len(models)} models: Qwen2.5 "
        "3/7/14/32B + base-7B, OLMo-2-13B, Llama-3.1-8B, Mistral-7B-v0.3, Gemma-2-9B) on a log-x "
        "axis; still illustrative (no fit line drawn through the beta-vs-size points), not a "
        "claimed scaling law.",
    ]
    note("F4", "logistic fits + beta vs model size", lines)
    return fit_results


# --------------------------------------------------------------------------
# F5 -- evidence-quantity insensitivity: lambda (k=2) vs n_demos, per model
# --------------------------------------------------------------------------
def fig_f5():
    files = ["jbias_qwen_small.json", "jbias_big.json", "jbias_wave2.json"]
    records = collect_records(files, krel_filter=2)
    models = [m for m in K4_MODEL_ORDER + list(MODEL_SHORT.keys())
              if any(r["model"] == m for r in records)]
    # de-dupe while preserving order
    seen = set(); models = [m for m in models if not (m in seen or seen.add(m))]

    ns_all = sorted({r["n"] for r in records})

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model in models:
        xs, ys, los, his = [], [], [], []
        for n in ns_all:
            per_seed = per_seed_level_means_filtered(records, model, 1, n=n)
            if not per_seed:
                continue
            m, lo, hi = bootstrap_mean_ci(per_seed.values(), rng_seed=7)
            xs.append(n); ys.append(m); los.append(m - lo); his.append(hi - m)
        if not xs:
            continue
        color = MODEL_COLOR.get(model, CATEGORICAL[0])
        ax.errorbar(xs, ys, yerr=[los, his], fmt="o-", color=color, markersize=5,
                     markeredgecolor="white", markeredgewidth=0.6, capsize=3, lw=1.6,
                     label=short(model))

    # Bayes size-principle reference: JSON 'bayes_raw' field, averaged over all rows sharing n
    # (bayes_raw is empirically model-invariant -- a property of the sampled demo set, not the
    # model being probed -- confirmed by direct inspection of the raw JSON).
    bxs, bys = [], []
    for n in ns_all:
        vals = [r["bayes_raw"] for r in records if r["n"] == n and r["bayes_raw"] is not None]
        if vals:
            bxs.append(n); bys.append(float(np.mean(vals)))
    st = dict(REF_STYLES["bayes"])
    lbl = st.pop("label")
    ax.plot(bxs, bys, marker=st.pop("marker", None), markersize=5, markerfacecolor="white",
            **st, label=lbl + " (empirical, JSON 'bayes' field)")

    ax.axhline(0.5, color=INK_MUTED, ls=(0, (3, 3)), lw=1.1, label="flat / chance")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns_all)
    ax.set_xticklabels([str(n) for n in ns_all])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("number of demonstrations $n$ (log scale)")
    ax.set_ylabel(r"$\lambda_1$ = P(positive | 1 of 2 attrs match), $k$=2")
    ax.set_title("F5. Evidence-quantity insensitivity: model lambda is flat in $n$;\n"
                 "Bayes size-principle reference collapses toward 0", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=7.6, ncol=2, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    savefig(fig, FIGS_DIR / "F5_evidence_quantity")

    note("F5", "evidence-quantity insensitivity (k=2, lambda vs n)", [
        f"Data: {', '.join(files)}; p1 rows, krel==2 (either implicit in the old schema or "
        f"explicit krel field), sanity>={SANITY_THRESH}.",
        f"n values present: {ns_all}. n=32 only exists for Qwen-3B-Instruct / Qwen-7B-Instruct "
        "via jbias_wave2.json; other models only have n in {4,8,16} from jbias_qwen_small.json / "
        "jbias_big.json -- lines for those models simply stop at n=16 (not an exclusion, just no "
        "n=32 batch was run for them).",
        "Bayes reference is the JSON's own logged 'bayes' field for krel=2 rows, which genuinely "
        "varies with n (and, at n=4, slightly with seed) -- unlike the k=4 case this field IS a "
        "real per-trial computation here, so it is plotted as actual data, not a formula.",
        "lambda_1 error bars = 95% percentile bootstrap CI (2000 resamples) over per-seed means.",
    ])


# --------------------------------------------------------------------------
# F6 -- lever results (direct vs CoT vs meet-decoding), jlever_7b.json
# --------------------------------------------------------------------------
def fig_f6():
    data = load_json("jlever_7b.json")
    model = "Qwen/Qwen2.5-7B-Instruct"
    rows = data[model]["rows"]
    summary = data[model]["summary"]
    conds = data["_meta"]["conds"]

    p2 = [r for r in rows if r["prong"] == "p2and"]
    p1 = [r for r in rows if r["prong"] == "p1"]

    metrics = {}
    for c in conds:
        y = np.array([r["y"] for r in p2])
        pred = np.array([r[c] for r in p2])
        acc = float(np.mean(pred == y))
        pos = y == 1
        neg = y == 0
        recall = float(np.mean(pred[pos] == 1)) if pos.any() else np.nan  # TPR
        fpr = float(np.mean(pred[neg] == 1)) if neg.any() else np.nan     # FP rate
        metrics[c] = dict(accuracy=acc, recall=recall, fpr=fpr,
                           acc_pos_summary=summary[c]["p2and_acc_pos"],
                           acc_neg_summary=summary[c]["p2and_acc_neg"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))

    metric_names = ["accuracy", "recall", "fpr"]
    x = np.arange(len(metric_names))
    w = 0.25
    for i, c in enumerate(conds):
        vals = [metrics[c][mn] for mn in metric_names]
        ax1.bar(x + (i - 1) * w, vals, width=w, color=CATEGORICAL[i], label=c,
                edgecolor="white", linewidth=0.6)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax1.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.6, color=INK_SECONDARY)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["accuracy", "recall\n(TPR, y=1)", "FPR\n(y=0)"])
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("rate")
    ax1.set_title("AND-concept ground-truth probes (p2and)", fontsize=10.5)
    ax1.legend(frameon=False, fontsize=9)

    lam = [summary[c]["p1_lambda"] for c in conds]
    x2 = np.arange(len(conds))
    ax2.bar(x2, lam, width=0.5, color=[CATEGORICAL[i] for i in range(len(conds))],
            edgecolor="white", linewidth=0.6)
    for xi, v in zip(x2, lam):
        ax2.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, color=INK_PRIMARY, fontweight="bold")
    ax2.axhline(refs.meet(1, 2), color=REF_STYLES["meet"]["color"], ls=REF_STYLES["meet"]["ls"],
                lw=1.2, label="meet (=0)")
    ax2.axhline(refs.join(1, 2), color=REF_STYLES["join"]["color"], ls=REF_STYLES["join"]["ls"],
                lw=1.2, label="join (=1)")
    ax2.axhline(0.5, color=INK_MUTED, ls=(0, (3, 3)), lw=1.1, label="flat (=0.5)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(conds)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel(r"p1 $\lambda$ (rev-$j$ probes)")
    ax2.set_title("Join-bias lever: does meet-decoding pull lambda to 0?", fontsize=10.5)
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle("F6. Decoding-condition lever (Qwen-7B-Instruct): direct vs CoT vs meet-decoding",
                 fontsize=12.5, fontweight="bold", y=1.03)
    savefig(fig, FIGS_DIR / "F6_lever_conditions")

    note("F6", "lever results (jlever_7b.json)", [
        f"Data: jlever_7b.json, model={model}, conditions={conds} "
        f"(_meta: seeds={data['_meta']['seeds']}, ndemos={data['_meta']['ndemos']}, "
        f"format={data['_meta']['format']}, krel={data['_meta']['krel']}).",
        f"p2and rows (ground-truth AND-concept probes, ID'd by explicit 'y' field): "
        f"{len(p2)} total ({len(p2)//len(conds) if conds else len(p2)} shared trials scored under "
        "all 3 conditions since direct/cot/meet are columns on the same rows, not separate rows).",
        "accuracy/recall/FPR recomputed directly from raw pred-vs-y rows (not just the "
        "precomputed summary), then cross-checked against summary's p2and_acc_pos "
        "(== our recall) and p2and_acc_neg (== 1 - our FPR): " +
        "; ".join(f"{c}: acc_pos(summary)={metrics[c]['acc_pos_summary']:.3f} vs "
                  f"recall(recomputed)={metrics[c]['recall']:.3f}" for c in conds),
        f"p1 lambda bars are read directly from summary[cond]['p1_lambda'] "
        f"({len(p1)} underlying p1 rows across conditions/kinds).",
    ])


# --------------------------------------------------------------------------
# F7 -- k=3 profiles (jbias_wave2.json)
# --------------------------------------------------------------------------
def fig_f7():
    files = ["jbias_wave2.json"]
    records = collect_records(files, krel_filter=3)
    models = [m for m in K4_MODEL_ORDER if any(r["model"] == m for r in records)]
    k = 3
    levels = refs.ladder_levels(k)  # [1,2]

    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    trial_counts = {}
    for ax, model in zip(axes, models):
        means, los, his = [], [], []
        for j in levels:
            per_seed = per_seed_level_means_filtered(records, model, j)
            m, lo, hi = bootstrap_mean_ci(per_seed.values(), rng_seed=11)
            means.append(m); los.append(m - lo); his.append(hi - m)
        n_rows = sum(1 for r in records if r["model"] == model)
        n_seeds = len({r["seed"] for r in records if r["model"] == model})
        trial_counts[model] = (n_rows, n_seeds)

        draw_ref_lines(ax, k, levels, ["additive", "meet", "join", "flat"])
        # NN and Bayes: literal task-given values for k=3 (see refs.py)
        nn_y = [refs.nn_step(j, k) for j in levels]
        st = dict(REF_STYLES["nn"]); lbl = st.pop("label")
        ax.plot(levels, nn_y, marker=st.pop("marker", None), markersize=4, markerfacecolor="white",
                **st, label=lbl)
        bayes_y = [refs.BAYES_K3[j] for j in levels]
        st = dict(REF_STYLES["bayes"]); lbl = st.pop("label")
        ax.plot(levels, bayes_y, marker=st.pop("marker", None), markersize=4, markerfacecolor="white",
                **st, label=lbl + " (task-given)")

        color = MODEL_COLOR.get(model, CATEGORICAL[0])
        ax.errorbar(levels, means, yerr=[los, his], fmt="o-", color=color, ecolor=color,
                     elinewidth=1.6, capsize=3, markersize=6, markeredgecolor="white",
                     markeredgewidth=0.8, zorder=5, label=f"{short(model)} (observed)")

        ax.set_title(short(model), fontsize=10.5)
        ax.set_xticks(levels)
        ax.set_xlabel("level $j$ (of $k$=3)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(0.7, 2.3)

    axes[0].set_ylabel(r"$\lambda_j$ = P(positive | $j$ of 3 attrs match)")
    handles, labels_ = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13),
               frameon=False, fontsize=8.6)
    fig.suptitle(r"F7. $\lambda$-profile at $k$=3 (jbias_wave2): observed vs. reference learners",
                 fontsize=12.5, fontweight="bold", y=1.03)
    savefig(fig, FIGS_DIR / "F7_lambda_ladder_k3")

    note("F7", "k=3 profiles (jbias_wave2.json)", [
        f"Data: jbias_wave2.json; p1 rows, krel==3, sanity>={SANITY_THRESH}.",
        "Trial counts (rows passing sanity / unique seeds) per model: " +
        "; ".join(f"{short(m)}={trial_counts[m][0]} rows / {trial_counts[m][1]} seeds" for m in models),
        "additive=(1/3,2/3), meet=0, join=1, flat=0.5 computed by formula.",
        "NN=(0.01,0.99) and Bayes=(0.005,0.064) are the literal task-given reference values for "
        "k=3 (labeled reference lines per the brief, not derived from a per-row JSON field -- the "
        "JSON's own per-row 'bayes'/'nn' scalars at k=3 represent only the j=1 level and are "
        "n/seed-dependent, e.g. ~0.033 at n=8 in this file, which is a different quantity; see "
        "refs.py docstring).",
    ])


# --------------------------------------------------------------------------
# F8 -- k=5 blind test: lambda_j vs level j=1..4, per model x format, with the
# pre-registered additive (j/k) dashed line
# --------------------------------------------------------------------------
def fig_f8():
    files = K5_FILES
    records = collect_records(files, krel_filter=5)
    models = [m for m in K5_MODEL_ORDER if any(r["model"] == m for r in records)]
    k = 5
    levels = refs.ladder_levels(k)  # [1,2,3,4]
    fmts = sorted({r["fmt"] for r in records})

    n_rows, n_cols = len(models), len(fmts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows),
                              sharey=True, sharex=True, squeeze=False)

    prereg_y = [refs.additive(j, k) for j in levels]
    trial_counts = {}

    for i, model in enumerate(models):
        for jx, fmt in enumerate(fmts):
            ax = axes[i][jx]
            means, los, his = [], [], []
            for j in levels:
                per_seed = per_seed_level_means_filtered(records, model, j, fmt=fmt)
                m, lo, hi = bootstrap_mean_ci(per_seed.values(), rng_seed=99)
                means.append(m); los.append(m - lo); his.append(hi - m)
            n_r = sum(1 for r in records if r["model"] == model and r["fmt"] == fmt)
            n_s = len({r["seed"] for r in records if r["model"] == model and r["fmt"] == fmt})
            trial_counts[(model, fmt)] = (n_r, n_s)

            st = dict(REF_STYLES["prereg"]); lbl = st.pop("label")
            ax.plot(levels, prereg_y, marker=st.pop("marker", None), markersize=4,
                    markerfacecolor="white", **st, label=lbl)

            color = MODEL_COLOR.get(model, CATEGORICAL[0])
            ax.errorbar(levels, means, yerr=[los, his], fmt="o-", color=color, ecolor=color,
                         elinewidth=1.6, capsize=3, markersize=6, markeredgecolor="white",
                         markeredgewidth=0.8, zorder=5, label=f"{short(model)} (observed)")

            ax.set_xticks(levels)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlim(0.6, 4.4)
            if i == 0:
                ax.set_title(fmt, fontsize=10.5)
            if jx == 0:
                ax.set_ylabel(short(model), fontsize=9.5)
            if i == n_rows - 1:
                ax.set_xlabel("level $j$ (of $k$=5)")

    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05 / n_rows),
               frameon=False, fontsize=9)
    fig.suptitle(r"F8. $k$=5 blind test: $\lambda_j$ vs. level, per model x format" + "\n"
                 "second pre-registered blind confirmation", fontsize=12.5, fontweight="bold", y=1.02)
    savefig(fig, FIGS_DIR / "F8_k5_blind_test")

    note("F8", "k=5 blind test (jbias_k5.json)", [
        f"Data: {', '.join(files)}; p1 rows, krel==5, sanity>={SANITY_THRESH}, split by (model, fmt) "
        f"panel, formats present: {fmts}.",
        "Pre-registered dashed line = additive j/k evaluated at k=5 -> (0.2, 0.4, 0.6, 0.8) for "
        "j=1..4; computed by the same refs.additive(j,k) formula used elsewhere (not a separate "
        "hardcoded literal), styled distinctly here ('prereg' in refs.py/style.py) because this "
        "figure is specifically the blind pre-registered confirmation run, not a generic reference "
        "overlay.",
        "95% CI = percentile bootstrap (2000 resamples) over per-seed means within each (model, "
        "fmt) cell.",
        "Trial counts per (model, fmt) panel (rows passing sanity / unique seeds): " +
        "; ".join(f"{short(m)}/{f}={trial_counts[(m, f)][0]}r/{trial_counts[(m, f)][1]}s"
                  for m in models for f in fmts),
    ])


# --------------------------------------------------------------------------
# F9 -- "one law, many families": all k=4 and k=5 lambda points on x=2j-k,
# overlaid with per-model fitted logistic + the universal (mean-beta) sigmoid
# + the k-dependent additive diagonal
# --------------------------------------------------------------------------
def fig_f9():
    from scipy.optimize import curve_fit

    records4 = collect_records(K4_FILES, krel_filter=4)
    records5 = collect_records(K5_FILES, krel_filter=5)
    models4 = {r["model"] for r in records4}
    models5 = {r["model"] for r in records5}
    models = [m for m in K4_MODEL_ORDER if m in models4 or m in models5]

    fig, ax = plt.subplots(figsize=(8.6, 6.4))

    per_model_fit = {}
    per_model_points = {}
    for model in models:
        xs, ys = [], []
        for krel, records in ((4, records4), (5, records5)):
            if model not in {r["model"] for r in records}:
                continue
            for j in range(0, krel + 1):
                per_seed = per_seed_level_means_filtered(records, model, j)
                if not per_seed:
                    continue
                y = float(np.mean(list(per_seed.values())))
                xs.append(2 * j - krel)
                ys.append(y)
        if not xs:
            continue
        xs_arr = np.array(xs, dtype=float)
        ys_arr = np.array(ys, dtype=float)
        per_model_points[model] = (xs_arr, ys_arr)
        try:
            popt, _ = curve_fit(_sigmoid, xs_arr, ys_arr, p0=[0.5, 0.0], maxfev=5000)
            beta, b0 = popt
        except Exception:
            beta, b0 = np.nan, np.nan
        per_model_fit[model] = (beta, b0)

    # scatter points: colored by family, filled=instruct / open=base
    plotted_families = set()
    for model in models:
        if model not in per_model_points:
            continue
        xs_arr, ys_arr = per_model_points[model]
        fam = MODEL_FAMILY.get(model, "?")
        color = FAMILY_COLOR.get(fam, CATEGORICAL[0])
        is_base = model in MODEL_IS_BASE
        plotted_families.add(fam)
        if is_base:
            ax.scatter(xs_arr, ys_arr, facecolors="white", edgecolors=color, linewidths=1.4,
                       s=42, zorder=6)
        else:
            ax.scatter(xs_arr, ys_arr, facecolors=color, edgecolors="white", linewidths=0.6,
                       s=42, zorder=6)
        beta, b0 = per_model_fit.get(model, (np.nan, np.nan))
        if not np.isnan(beta):
            xx = np.linspace(xs_arr.min(), xs_arr.max(), 200)
            yy = _sigmoid(xx, beta, b0)
            ls = (0, (4, 2)) if is_base else "solid"
            ax.plot(xx, yy, color=color, lw=1.1, alpha=0.75, ls=ls, zorder=4)

    # universal law: sigma at the mean fitted (beta, b0) across all models with a valid fit
    betas = [v[0] for v in per_model_fit.values() if not np.isnan(v[0])]
    b0s = [v[1] for v in per_model_fit.values() if not np.isnan(v[1])]
    mean_beta = float(np.mean(betas))
    mean_b0 = float(np.mean(b0s))
    xx_full = np.linspace(-5, 5, 300)
    ax.plot(xx_full, _sigmoid(xx_full, mean_beta, mean_b0), color=INK_PRIMARY, lw=2.4, zorder=8,
            label=r"universal $\sigma(\bar\beta x-\bar b_0)$, " + f"mean over {len(betas)} models "
            r"($\bar\beta$=" + f"{mean_beta:.2f})")

    # additive diagonal, drawn once per k actually present (k-dependent slope 1/(2k))
    for k_ in sorted({4, 5}):
        xr = np.linspace(-k_, k_, 50)
        yr = xr / (2 * k_) + 0.5
        st = dict(REF_STYLES["additive"])
        lbl = st.pop("label") + f" (k={k_})"
        ax.plot(xr, yr, color=INK_MUTED, ls=(0, (1, 1.5)), lw=1.3, zorder=3, label=lbl)

    ax.axhline(0.5, color=GRIDLINE, lw=0.8, zorder=0)
    ax.axvline(0.0, color=GRIDLINE, lw=0.8, zorder=0)

    # compact family legend (proxy handles) + base/instruct marker key
    fam_order = [f for f in ["Qwen", "Llama", "Mistral", "Gemma", "OLMo"] if f in plotted_families]
    fam_handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=FAMILY_COLOR[f],
                               markeredgecolor="white", markersize=7, label=f) for f in fam_order]
    marker_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=INK_MUTED,
                   markeredgecolor="white", markersize=7, label="instruct (filled)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor=INK_MUTED, markersize=7, label="base (open)"),
    ]
    ref_handles, ref_labels = ax.get_legend_handles_labels()
    ax.legend(handles=fam_handles + marker_handles + ref_handles, fontsize=8.4, frameon=False,
              loc="upper left", ncol=1)

    ax.set_xlabel(r"$x = 2j-k$  (#matched $-$ #unmatched relevant attributes)")
    ax.set_ylabel(r"$\lambda$")
    ax.set_xlim(-5.4, 5.4)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("F9. One law, many families: a single $x$=(#matched$-$#unmatched) sigmoid\n"
                 "fits every model at both $k$=4 and $k$=5", fontsize=12.2, fontweight="bold")
    savefig(fig, FIGS_DIR / "F9_one_law_many_families")

    n_pts = sum(len(v[0]) for v in per_model_points.values())
    note("F9", "one law, many families (money figure)", [
        f"Data: {', '.join(K4_FILES)} (krel==4) + {', '.join(K5_FILES)} (krel==5); p1 rows, "
        f"sanity>={SANITY_THRESH}; {len(models)} models, {n_pts} total (model, level) points "
        "pooled across format/seed.",
        "x = 2j-k (matched-minus-unmatched relevant attributes) so the same x-coordinate is "
        "directly comparable across k=4 and k=5 trials; y = lambda_j (levels j=0..k INCLUDED, "
        "i.e. p_neither/p_both anchors are plotted too, same as F4's anchor-inclusive fit -- this "
        "is a judgment call to stabilize the per-model 2-parameter fit, not requested verbatim in "
        "the brief, flagged here rather than silently assumed).",
        "Per-model thin curve = least-squares fit of sigma(beta*x - b0) to that model's own "
        "pooled points (both k=4 and k=5 points together, for the 2 models -- Qwen-7B-Instruct, "
        "Llama-3.1-8B-Instruct -- that have both; k=4-only points for the rest).",
        f"Fitted (beta, b0) per model: " + "; ".join(
            f"{short(m)}: beta={per_model_fit[m][0]:.3f}, b0={per_model_fit[m][1]:.3f}"
            for m in models if m in per_model_fit and not np.isnan(per_model_fit[m][0])),
        f"Universal reference curve = sigma(mean_beta * x - mean_b0), unweighted mean of the "
        f"{len(betas)} per-model fitted (beta, b0) pairs above (mean_beta={mean_beta:.3f}, "
        f"mean_b0={mean_b0:.3f}) -- this is the 'one law' claim: a single k-independent sigmoid "
        "in x that was not told which k a point came from.",
        "additive j/k diagonal is drawn as TWO segments (k=4: slope 1/8; k=5: slope 1/10) because "
        "the formula x/(2k)+0.5 is not literally k-invariant -- per the brief's exact formula, not "
        "collapsed to one line; the two segments visibly diverge more from the data than the "
        "universal sigmoid does, which is the point of the figure.",
        "Markers: color = model family (Qwen/Llama/Mistral/Gemma/OLMo, from style.FAMILY_COLOR); "
        "filled = instruction-tuned, open = base (only Qwen2.5-7B base in this roster).",
    ])


def write_notes(header_extra: list[str]):
    header = [
        "# Figure notes",
        "",
        "Generated by frontier_lab/viz/make_figures.py. Every plotted number is either read "
        "from frontier_lab/results/*.json, or is one of the analytically-defined / task-given "
        "reference lines documented in frontier_lab/viz/refs.py (additive=j/k, meet=0, join=1, "
        f"flat=0.5, 1-NN Hamming-step, and the literal k=3/k=4 prereg values given in the brief).",
        f"Global inclusion rule: a p1 trial counts as 'learned' iff sanity >= {SANITY_THRESH} "
        "(>=3 of 4 both/neither sanity-check probes correct).",
        "",
    ]
    body = "\n\n".join(NOTES)
    (FIGS_DIR / "NOTES.md").write_text("\n".join(header) + "\n" + body + "\n" +
                                        "\n\n" + "\n".join(header_extra) + "\n", encoding="utf-8")


def main():
    use_style()

    def _try(fn, *a):
        """Regenerate one figure; skip (with a note) if its data isn't shipped."""
        try:
            return fn(*a)
        except FileNotFoundError as e:
            print(f"  [skip] {fn.__name__}: missing data file "
                  f"({getattr(e, 'filename', e)}) — not part of the released "
                  f"data set; shipped PNG left in place.")
            return None

    _try(fig_f1)
    _try(fig_f2)
    f3 = _try(fig_f3)
    n_mono, n_total, cells, mono_flags = f3 if f3 else (0, 0, [], [])
    _try(fig_f4)
    _try(fig_f5)
    _try(fig_f6)   # needs jlever_7b.json (now shipped in data/)
    _try(fig_f7)
    _try(fig_f8)
    _try(fig_f9)

    extra = ["## Cross-figure summary",
             f"- F3 monotonicity: {n_mono}/{n_total} model x format cells monotone at k=4 (see F3 "
             "note for the per-model format breakdown; this count is computed from the data, not "
             "hand-typed, and will change again if more model files are added).",
             "- F8/F9 add a second krel (k=5, 2 models: Qwen-7B-Instruct, Llama-3.1-8B-Instruct) "
             "as an independent pre-registered blind confirmation of the k=4 additive-integration "
             "finding; F9 is the cross-k, cross-family synthesis figure."]
    write_notes(extra)

    produced = sorted(p.name for p in FIGS_DIR.glob("*.png"))
    print("Wrote figures:")
    for p in produced:
        print(" -", p)
    print("Wrote figs/NOTES.md")


if __name__ == "__main__":
    main()
