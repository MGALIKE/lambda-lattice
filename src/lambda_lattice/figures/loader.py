"""
Schema-tolerant loader for the join-bias (jbias_*) and lever (jlever_*)
result JSONs in frontier_lab/results/.

Observed schema (empirically introspected, 2026-07-03):

Top level: {"<model name>": {"p1": [...], "p2": [...]}, ..., "_meta": {...}}

p1 rows are one (fmt, n, seed) trial. Two generations of format:

  * OLD (jbias_qwen_small.json, jbias_big.json; implicit krel=2):
      {"fmt","n","seed","lambda","sanity","p_both","p_neither","bayes","nn"}
      "lambda" IS the j=1 (only interior) level for a 2-attribute concept.

  * NEW (jbias_wave2.json krel in {2,3}; jbias_k4*.json krel=4):
      adds "krel", "detail" (list of [kind, bitstring, p] probe records,
      kind in {"rev1","rev2","rev3","both","neither"}), and "lambda1"/
      "lambda2"/"lambda3" (precomputed per-interior-level means already
      matching what you'd recompute from "detail"). "lambda" is present
      but is NOT the simple average of lambda1..lambda{k-1} (kept only
      for backward compat) -- we do not rely on it when level fields or
      "detail" are available.

  "sanity" is trial-level accuracy (0/0.25/.../1) on the "both"+"neither"
  probes (4 sanity checks total: 2x both, 2x neither) -- a trial counts as
  "learned" iff sanity >= 0.75, per the task's stated inclusion rule.

  "bayes" / "nn" are single scalars per row (the framework's own reference
  computation), empirically confirmed here to depend only on (krel, n) for
  n>=8 (constant across seed/fmt/model), and on (krel, n, seed) for n=4
  (small residual variation, presumably size-principle numerical noise).
  They represent the j=1-level-only Bayes/NN reference, not a full
  per-level curve -- see figs/NOTES.md for how each figure handles this.

jlever_7b.json has a different per-model schema: {"rows":[...],"summary":{...}}
  rows: per-probe records with prong in {"p1","p2and"}; p1 rows carry
  kind in {rev1,rev2,both,neither} (no ground truth "y"); p2and rows carry
  an explicit ground-truth label "y" plus 0/1 predictions under "direct",
  "cot", "meet" decoding conditions.
  summary: pre-aggregated per-condition stats (p1_lambda, sanity checks,
  p2and_acc_pos/neg) -- already computed from the same rows.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from lambda_lattice._paths import data_dir

# The figure pipeline reads the shipped raw result JSONs from the repo's data/
# dir (resolved for an editable install, or via LAMBDA_LATTICE_DATA).
RESULTS_DIR = data_dir()

SANITY_THRESH = 0.75


def load_json(fname: str) -> dict:
    with open(RESULTS_DIR / fname, "r", encoding="utf-8") as fh:
        return json.load(fh)


def model_names(data: dict) -> list[str]:
    return [k for k in data.keys() if k != "_meta"]


def iter_p1_rows(data: dict, fname: str):
    """Yield normalized p1 row dicts (raw fields + _model + _source)."""
    for model in model_names(data):
        entry = data[model]
        for row in entry.get("p1", []):
            r = dict(row)
            r["_model"] = model
            r["_source"] = fname
            yield r


def recompute_levels_from_detail(row: dict) -> Optional[dict]:
    """Group the raw 'detail' probe list by level j (0..krel), averaging
    the model's P(positive) over all probes at that level. Returns
    {j: (mean_p, n_probes)} or None if no 'detail' present."""
    detail = row.get("detail")
    if not detail:
        return None
    krel = row.get("krel", 2)
    groups: dict[int, list[float]] = defaultdict(list)
    for kind, _bits, p in detail:
        if kind == "both":
            j = krel
        elif kind == "neither":
            j = 0
        elif isinstance(kind, str) and kind.startswith("rev"):
            j = int(kind[3:])
        else:
            continue
        groups[j].append(float(p))
    return {j: (float(np.mean(v)), len(v)) for j, v in groups.items()}


def get_levels(row: dict) -> tuple[int, dict[int, float]]:
    """Return (krel, {j: lambda_j}) for a p1 row, j in 0..krel.

    Preference order per level: explicit lambda{j} field > recomputed mean
    from 'detail' > (for krel==2 only) the legacy 'lambda' field which IS
    the j=1 value in that schema generation. Endpoints (j=0, j=krel) come
    from p_neither / p_both when present, else from 'detail' recompute.
    """
    krel = int(row.get("krel", 2))
    levels: dict[int, float] = {}

    if krel == 2 and "lambda" in row:
        levels[1] = row["lambda"]
    else:
        for j in range(1, krel):
            key = f"lambda{j}"
            if key in row:
                levels[j] = row[key]

    if "p_both" in row:
        levels[krel] = row["p_both"]
    if "p_neither" in row:
        levels[0] = row["p_neither"]

    recomputed = recompute_levels_from_detail(row)
    if recomputed:
        for j, (mean_p, _n) in recomputed.items():
            levels.setdefault(j, mean_p)

    return krel, levels


def passes_sanity(row: dict, thresh: float = SANITY_THRESH) -> bool:
    s = row.get("sanity")
    return s is not None and s >= thresh


def collect_records(files: list[str], krel_filter: Optional[int] = None,
                     sanity_thresh: float = SANITY_THRESH) -> list[dict]:
    """Load `files` and return one normalized record per learned p1 trial:
    {model, fmt, n, seed, source, krel, levels, bayes_raw, nn_raw, sanity}
    """
    records = []
    for fname in files:
        data = load_json(fname)
        for row in iter_p1_rows(data, fname):
            if not passes_sanity(row, sanity_thresh):
                continue
            krel, levels = get_levels(row)
            if krel_filter is not None and krel != krel_filter:
                continue
            records.append(dict(
                model=row["_model"], fmt=row.get("fmt"), n=row.get("n"),
                seed=row.get("seed"), source=fname, krel=krel, levels=levels,
                bayes_raw=row.get("bayes"), nn_raw=row.get("nn"),
                sanity=row.get("sanity"),
            ))
    return records


def per_seed_level_means(records: list[dict], model: str, j: int) -> dict:
    """{seed: mean(level j value across fmt/source rows for that seed)}"""
    by_seed = defaultdict(list)
    for r in records:
        if r["model"] != model:
            continue
        if j in r["levels"]:
            by_seed[r["seed"]].append(r["levels"][j])
    return {seed: float(np.mean(v)) for seed, v in by_seed.items()}


def bootstrap_mean_ci(values, n_boot=2000, ci=0.95, rng_seed=0):
    """Percentile bootstrap CI for the mean of `values` (resampled with
    replacement at the level the caller has already aggregated to --
    typically one value per seed)."""
    values = np.asarray(list(values), dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        return values[0], values[0], values[0]
    rng = np.random.default_rng(rng_seed)
    boots = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return float(values.mean()), lo, hi


def se_2se(values):
    values = np.asarray(list(values), dtype=float)
    n = len(values)
    if n < 2:
        return float(values.mean()) if n else np.nan, 0.0
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    return float(values.mean()), 2 * se
