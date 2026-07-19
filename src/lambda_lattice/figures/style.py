"""
Shared plotting style for the join-bias figure suite.

Palette is the validated colorblind-safe categorical ramp from the repo's
`dataviz` skill (references/palette.md), light-mode instance. Reference /
theoretical lines (additive, meet, join, flat, Bayes, NN) are always drawn in
muted gray tones with distinct dash patterns rather than hue, so that hue is
reserved for actual data series (models / conditions) -- consistent with the
"color follows the entity" rule.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---- categorical ramp (light mode), fixed order, never cycled ----
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"

CATEGORICAL = [BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE]

# chart chrome / ink
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"

# Extended hues for the 2026-07-03 cross-family additions (Llama/Mistral/Gemma).
# The validated 8-slot categorical ramp (BLUE..ORANGE above) was already fully
# assigned to the original Qwen-size/OLMo/Qwen-base roster; these three are
# picked for visual distinctness from that set (not independently CVD-checked
# the way the 8-slot ramp was -- see figs/NOTES.md).
BROWN = "#8a5a2b"
TEAL = "#0f8a8a"
ROSE = "#b23a6b"

# fixed model -> color assignment (used consistently across every figure)
MODEL_COLOR = {
    "Qwen/Qwen2.5-3B-Instruct": BLUE,
    "Qwen/Qwen2.5-7B-Instruct": AQUA,
    "Qwen/Qwen2.5-14B-Instruct": VIOLET,
    "Qwen/Qwen2.5-32B-Instruct": RED,
    "allenai/OLMo-2-1124-13B-Instruct": ORANGE,
    "Qwen/Qwen2.5-0.5B-Instruct": YELLOW,
    "Qwen/Qwen2.5-1.5B-Instruct": GREEN,
    "Qwen/Qwen2.5-7B": MAGENTA,
    "unsloth/Meta-Llama-3.1-8B-Instruct": BROWN,
    "mistralai/Mistral-7B-Instruct-v0.3": TEAL,
    "unsloth/gemma-2-9b-it": ROSE,
}

MODEL_SHORT = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen-0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen-1.5B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen-3B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen-7B",
    "Qwen/Qwen2.5-7B": "Qwen2.5-7B (base)",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen-14B",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen-32B",
    "allenai/OLMo-2-1124-13B-Instruct": "OLMo-13B",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "unsloth/gemma-2-9b-it": "Gemma-2-9B",
}

MODEL_SIZE_B = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.5,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen/Qwen2.5-3B-Instruct": 3.0,
    "Qwen/Qwen2.5-7B-Instruct": 7.0,
    "Qwen/Qwen2.5-7B": 7.0,
    "Qwen/Qwen2.5-14B-Instruct": 14.0,
    "Qwen/Qwen2.5-32B-Instruct": 32.0,
    "allenai/OLMo-2-1124-13B-Instruct": 13.0,
    "unsloth/Meta-Llama-3.1-8B-Instruct": 8.0,
    "mistralai/Mistral-7B-Instruct-v0.3": 7.0,
    "unsloth/gemma-2-9b-it": 9.0,
}

K4_MODEL_ORDER = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "unsloth/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B",
]

# k=5 blind-test roster (jbias_k5.json)
K5_MODEL_ORDER = [
    "Qwen/Qwen2.5-7B-Instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
]

# family / base-vs-instruct metadata for F9 ("one law, many families")
MODEL_FAMILY = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen",
    "Qwen/Qwen2.5-7B": "Qwen",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen",
    "allenai/OLMo-2-1124-13B-Instruct": "OLMo",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "Llama",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral",
    "unsloth/gemma-2-9b-it": "Gemma",
}

# models that are base (not instruction-tuned) -- everything else is instruct
MODEL_IS_BASE = {"Qwen/Qwen2.5-7B"}

FAMILY_COLOR = {
    "Qwen": BLUE,
    "Llama": BROWN,
    "Mistral": TEAL,
    "Gemma": ROSE,
    "OLMo": ORANGE,
}

# reference-line styling: gray, distinguished by dash pattern + marker, not hue
REF_STYLES = {
    "additive":  dict(color=INK_SECONDARY, ls=(0, (1, 0)),      lw=1.4, marker=None, label="additive  ($j/k$)"),
    "meet":      dict(color=INK_MUTED,     ls=(0, (1, 1)),      lw=1.2, marker=None, label="meet (conjunctive)"),
    "join":      dict(color=INK_MUTED,     ls=(0, (5, 1)),      lw=1.2, marker=None, label="join (disjunctive)"),
    "flat":      dict(color=INK_MUTED,     ls=(0, (3, 3)),      lw=1.2, marker=None, label="flat / chance"),
    "nn":        dict(color=INK_SECONDARY, ls=(0, (3, 1, 1, 1)), lw=1.3, marker="s", label="1-NN (Hamming, idealized)"),
    "bayes":     dict(color=INK_SECONDARY, ls=(0, (6, 2)),      lw=1.3, marker="^", label="Bayes size-principle"),
    # same formula as "additive" (j/k) but styled/labeled distinctly for F8, where it is being
    # used specifically as the pre-registered blind prediction rather than a generic reference
    "prereg":    dict(color=INK_SECONDARY, ls=(0, (4, 2)),      lw=1.4, marker="D", label="pre-registered ($j/k$)"),
}


def use_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "text.color": INK_PRIMARY,
        "legend.frameon": False,
        "legend.fontsize": 8.8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, path_no_ext, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(str(path_no_ext) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(path_no_ext) + ".pdf", bbox_inches="tight")
    plt.close(fig)
