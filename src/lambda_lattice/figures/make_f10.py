"""F10 — reasoning/thinking-mode extension (2026-07-18 prereg).

Two panels:
  (a) Arm B: lambda vs demo count n, thinking vs non-thinking conditions, with the
      exact size-principle Bayes reference (which contracts to 0) in muted gray.
  (b) Arm A: per-level profile lambda_j at k=4 (n=8) for the thinking conditions,
      against the graded prediction j/k and the exact Bayes per-level reference.

Style: repo palette (style.py); hue = condition, gray dashes = references.
Seed-clustered SE bars throughout.
"""
import json
import pathlib

import matplotlib.pyplot as plt

from style import (AQUA, BLUE, GREEN, GRIDLINE, INK_MUTED, INK_SECONDARY, RED,
                   VIOLET, YELLOW)

from lambda_lattice._paths import data_dir, figures_dir

HERE = pathlib.Path(__file__).parent
DATA = data_dir()
OUT = figures_dir()

CONDS = [
    ("echo_think_qwen3_8b_v2.json", "Qwen/Qwen3-8B:on", "Qwen3-8B think", BLUE, "-"),
    ("echo_think_qwen3_8b.json", "Qwen/Qwen3-8B:off", "Qwen3-8B no-think", BLUE, ":"),
    ("echo_think_qwen3_14b.json", "Qwen/Qwen3-14B:on", "Qwen3-14B think", AQUA, "-"),
    ("echo_think_qwen3_14b.json", "Qwen/Qwen3-14B:off", "Qwen3-14B no-think", AQUA, ":"),
    ("echo_think_qwen3_32b.json", "Qwen/Qwen3-32B:on", "Qwen3-32B think", GREEN, "-"),
    ("echo_think_qwen3_32b.json", "Qwen/Qwen3-32B:off", "Qwen3-32B no-think",
     GREEN, ":"),
    ("echo_think_r1d14b_v2.json", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B:native",
     "R1-Distill-14B (think)", VIOLET, "-"),
    ("echo_think_qwen25_14b.json", "Qwen/Qwen2.5-14B-Instruct:none",
     "Qwen2.5-14B control", YELLOW, ":"),
]
NS = [4, 8, 16, 32]
BAYES_LEVELS_K4_N8 = [0.002, 0.017, 0.274]  # exact, computed over the 12 seeds


def learned(trials):
    return [t for t in trials if t["sanity"] >= 0.75 and t["parse_rate"] >= 0.8]


def mean_se(vals):
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, (var / len(vals)) ** 0.5


def main():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.6, 4.1), dpi=300)

    bayes_by_n = {n: [] for n in NS}
    for ax in (ax_a, ax_b):
        ax.set_facecolor("white")
        ax.grid(True, axis="y", color=GRIDLINE, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    for fname, key, label, color, dash in CONDS:
        d = json.loads((DATA / fname).read_text(encoding="utf-8"))[key]
        tr = learned(d["trials"])
        xs, ms, ses = [], [], []
        for n in NS:
            vals = [t["lambda"] for t in tr if t["arm"] == "b" and t["n"] == n]
            if vals:
                m, se = mean_se(vals)
                xs.append(n)
                ms.append(m)
                ses.append(se)
            for t in tr:
                if t["arm"] == "b" and t["n"] == n:
                    bayes_by_n[n].append(t["bayes"])
        ax_a.errorbar(xs, ms, yerr=ses, color=color, ls=dash, lw=1.8,
                      marker="o", ms=4, capsize=2, label=label)
        # panel b: thinking conditions + control only (declutter: skip no-think)
        if dash == "-" or "control" in label:
            a = [t for t in tr if t["arm"] == "a"]
            prof, perr = [], []
            for j in (1, 2, 3):
                m, se = mean_se([t[f"lambda{j}"] for t in a])
                prof.append(m)
                perr.append(se)
            ax_b.errorbar([1, 2, 3], prof, yerr=perr, color=color, ls=dash,
                          lw=1.8, marker="o", ms=4, capsize=2, label=label)

    bayes_ref = [sum(bayes_by_n[n]) / len(bayes_by_n[n]) for n in NS]
    ax_a.plot(NS, bayes_ref, color=INK_MUTED, ls="--", lw=1.6,
              label="exact Bayes (size principle)")
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(NS, [str(n) for n in NS])
    ax_a.set_xlabel("demonstrations n", color=INK_SECONDARY)
    ax_a.set_ylabel("λ (join-ward position)", color=INK_SECONDARY)
    ax_a.set_ylim(-0.03, 1.12)
    ax_a.set_title("(a) evidence quantity: thinking stays flat,\nBayes contracts",
                   fontsize=10, color=INK_SECONDARY)
    ax_a.legend(fontsize=6.2, frameon=False, loc="upper center", ncol=2,
                columnspacing=1.0, handlelength=1.6)

    ax_b.plot([1, 2, 3], [j / 4 for j in (1, 2, 3)], color=INK_MUTED, ls="--",
              lw=1.6, label="graded prediction j/k")
    ax_b.plot([1, 2, 3], BAYES_LEVELS_K4_N8, color=INK_MUTED, ls=":", lw=1.6,
              label="exact Bayes per level")
    ax_b.set_xticks([1, 2, 3])
    ax_b.set_xlabel("lattice level j (of k=4 attributes matched)",
                    color=INK_SECONDARY)
    ax_b.set_ylabel("λ$_j$", color=INK_SECONDARY)
    ax_b.set_ylim(-0.03, 1.0)
    ax_b.set_title("(b) k=4 profile: deliberation leaves the\ngraded law intact",
                   fontsize=10, color=INK_SECONDARY)
    ax_b.legend(fontsize=6.5, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "F10_reasoning_extension.png", bbox_inches="tight")
    print("wrote", OUT / "F10_reasoning_extension.png")


if __name__ == "__main__":
    main()
