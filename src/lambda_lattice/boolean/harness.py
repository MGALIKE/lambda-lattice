"""boolean/harness.py — the lambda instrument for Boolean attribute concepts.

Adapted (imports/paths/packaging only) from the pre-registered
``release/lambda-icl/src/harness.py`` (a.k.a. ``echo_join_bias.py``). The
numeric logic — stimuli, prompts, logprob scoring, and the exact reference
comparators — is unchanged; the reference comparators now live in
``lambda_lattice.references`` and are imported here.

Where in the lattice interval [meet = version-space closure, join] of
demo-consistent hypotheses does an LLM's in-context generalization sit?

Prong P1 (ambiguous demos): positives all A&B, negatives all ~A&~B; the
consistent hypotheses include A, B, A&B (meet) and A|B (join). Held-out
"A-only"/"B-only" revealer probes locate the model on the interval: join score
lambda = mean P(positive label).
Prong P2 (disambiguated concepts): AND vs OR concepts fully identified by demos;
accuracy + signed coverage error (humans: AND easier, Feldman 2000).

Two ways to drive it:
  * env-var CLI (unchanged interface): ``python -m lambda_lattice.boolean.harness``
      EJB_MODELS   comma list (default Qwen/Qwen2.5-0.5B-Instruct)
      EJB_SEEDS    int, default 20
      EJB_NDEMOS   comma list, default 4,8,16
      EJB_FORMATS  comma subset of f1,f2,f3 (default all)
      EJB_PRONGS   comma subset of p1,p2 (default both)
      EJB_KREL     comma list, default 2
      EJB_NATTRS   4|5 (default 4)
      EJB_DTYPE    float16|bfloat16 (default float16)
      EJB_BATCH    batch size, default 16
      EJB_OUT      output filename (default echo_join_bias.json)
  * python API: ``harness.run({"models": "...", "seeds": 4, ...}) -> results dict``

Both require torch + transformers + a real causal LM (there is no mock backend
for the logprob instrument; the mock-oracle power check lives in the number
game — see ``lambda_lattice.numbers.harness``).
"""
from __future__ import annotations

import itertools
import json
import math
import os
import pathlib
import random
import time

from ..references import bayes_join_score, grid_bits, nn_join_score

HERE = pathlib.Path(__file__).resolve().parent


def _results_base() -> pathlib.Path:
    from .._paths import results_dir
    return results_dir()


OUT = _results_base() / os.environ.get("EJB_OUT", "echo_join_bias.json")

MODELS = os.environ.get("EJB_MODELS", "Qwen/Qwen2.5-0.5B-Instruct").split(",")
SEEDS = int(os.environ.get("EJB_SEEDS", "20"))
NDEMOS = [int(x) for x in os.environ.get("EJB_NDEMOS", "4,8,16").split(",")]
FORMATS = os.environ.get("EJB_FORMATS", "f1,f2,f3").split(",")
PRONGS = os.environ.get("EJB_PRONGS", "p1,p2").split(",")
BATCH = int(os.environ.get("EJB_BATCH", "16"))
KREL = [int(x) for x in os.environ.get("EJB_KREL", "2").split(",")]
_DTYPE_NAME = os.environ.get("EJB_DTYPE", "float16")

LABELS = ("wug", "dax")  # counterbalanced per seed

# 4 binary attributes; first two (after per-seed shuffle) are the relevant A, B.
# EJB_NATTRS=5 adds a 5th ("material") so krel=5 profiles are possible.
WORD_ATTRS = [
    ("color", ("red", "blue")),
    ("shape", ("square", "circle")),
    ("texture", ("striped", "plain")),
    ("size", ("big", "small")),
]
if int(os.environ.get("EJB_NATTRS", "4")) >= 5:
    WORD_ATTRS.append(("material", ("wooden", "metal")))


# ---------------------------------------------------------------------------
# stimuli
# ---------------------------------------------------------------------------
def render_item(bits: tuple[int, ...], attrs, fmt: str, rng: random.Random) -> str:
    """bits[i] in {0,1} selects the value of attribute i (index 0 = value '1')."""
    if fmt == "f1":  # adjective phrase
        vals = {name: vv[0] if b else vv[1] for (name, vv), b in zip(attrs, bits)}
        adjs = [vals[n] for n in ("size", "texture", "material", "color") if n in vals]
        return f"a {', '.join(adjs[:-1])}, {adjs[-1]} {vals['shape']}"
    if fmt == "f2":  # key=value list
        parts = [f"{name}={vv[0] if b else vv[1]}" for (name, vv), b in zip(attrs, bits)]
        return ", ".join(parts)
    if fmt == "f3":  # numeric thresholds: attr i true <=> var_i > 5
        names = ("p", "q", "r", "s", "t")
        vals = [rng.randint(6, 9) if b else rng.randint(1, 4) for b in bits]
        return ", ".join(f"{n}={v}" for n, v in zip(names, vals))
    raise ValueError(fmt)


def make_p1_trial(fmt: str, n_demos: int, seed: int, krel: int = 2):
    """Ambiguous-demo trial. Relevant attrs = first `krel` positions after shuffling.
    Positives are all-1 on relevant attrs, negatives all-0; irrelevant attrs random.
    Probe kinds: 'rev<j>' = exactly j of krel relevant attrs set (0<j<krel),
    'both' = all set, 'neither' = none set. Returns (demo lines, probes, raw)."""
    rng = random.Random(10_000 + 1_000_000 * krel + seed)
    attrs = WORD_ATTRS[:]
    rng.shuffle(attrs)
    # attribute polarity also randomized: which value counts as '1'
    attrs = [(name, vv if rng.random() < 0.5 else (vv[1], vv[0])) for name, vv in attrs]
    kirr = len(attrs) - krel

    def item(*rel):
        irr = tuple(rng.randint(0, 1) for _ in range(kirr))
        return tuple(rel) + irr

    demos = []
    for _ in range(n_demos // 2):
        demos.append((item(*([1] * krel)), 1))
        demos.append((item(*([0] * krel)), 0))
    rng.shuffle(demos)
    probes = []
    for j in range(1, krel):  # every mixed level, every position pattern, twice each
        pats = [p for p in itertools.product((0, 1), repeat=krel) if sum(p) == j]
        for _ in range(2):
            for p in pats:
                probes.append((item(*p), f"rev{j}"))
    probes += [(item(*([1] * krel)), "both"), (item(*([1] * krel)), "both"),
               (item(*([0] * krel)), "neither"), (item(*([0] * krel)), "neither")]
    demo_txt = [(render_item(b, attrs, fmt, rng), y) for b, y in demos]
    probe_txt = [(render_item(b, attrs, fmt, rng), kind) for b, kind in probes]
    return demo_txt, probe_txt, demos, probes


def make_p2_trial(fmt: str, concept: str, seed: int):
    """Disambiguated concept (AND: A&B, OR: A|B). 12 demos = 3 per relevant cell,
    8 probes = 2 per cell."""
    rng = random.Random(20_000 + seed)
    attrs = WORD_ATTRS[:]
    rng.shuffle(attrs)
    attrs = [(name, vv if rng.random() < 0.5 else (vv[1], vv[0])) for name, vv in attrs]

    def lab(a, b):
        return int(a and b) if concept == "and" else int(a or b)

    def item(a, b):
        irr = (rng.randint(0, 1), rng.randint(0, 1))
        return (a, b) + irr

    demos = []
    for a, b in ((1, 1), (1, 0), (0, 1), (0, 0)):
        for _ in range(3):
            demos.append((item(a, b), lab(a, b)))
    rng.shuffle(demos)
    probes = []
    for a, b in ((1, 1), (1, 0), (0, 1), (0, 0)):
        for _ in range(2):
            probes.append((item(a, b), lab(a, b)))
    demo_txt = [(render_item(bts, attrs, fmt, rng), y) for bts, y in demos]
    probe_txt = [(render_item(bts, attrs, fmt, rng), y) for bts, y in probes]
    return demo_txt, probe_txt


def build_prompt(demo_txt, probe: str, pos_label: str, neg_label: str) -> str:
    lines = [f"Each item below is labeled '{pos_label}' or '{neg_label}'."]
    for text, y in demo_txt:
        lines.append(f"Item: {text} -> {pos_label if y else neg_label}")
    lines.append(f"Item: {probe} ->")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# model scoring (logprob teacher-forcing)
# ---------------------------------------------------------------------------
class Scorer:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        dtype = getattr(torch, _DTYPE_NAME)
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"  # label-position logic assumes right padding
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, device_map="auto"
        ).eval()
        self.dev = next(self.model.parameters()).device

    def label_probs(self, prompts: list[str], pos_label: str, neg_label: str) -> list[float]:
        """P(pos_label | prompt) vs neg_label, batched teacher-forcing."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(prompts), BATCH):
                chunk = prompts[i:i + BATCH]
                lps = {}
                for lab in (pos_label, neg_label):
                    texts = [p + " " + lab for p in chunk]
                    enc = self.tok(texts, return_tensors="pt", padding=True).to(self.dev)
                    logits = self.model(**enc).logits.float().log_softmax(-1)
                    vals = []
                    for b, p in enumerate(chunk):
                        ids = enc["input_ids"][b]
                        n_real = int(enc["attention_mask"][b].sum())
                        # label tokens = the last (n_real - plen) real tokens
                        lab_ids = self.tok(" " + lab, add_special_tokens=False)["input_ids"]
                        nlab = len(lab_ids)
                        lp = 0.0
                        for t in range(n_real - nlab, n_real):
                            lp += logits[b, t - 1, ids[t]].item()
                        vals.append(lp)
                    lps[lab] = vals
                for a, bb in zip(lps[pos_label], lps[neg_label]):
                    out.append(1.0 / (1.0 + math.exp(bb - a)))
        return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_model(name: str) -> dict:
    print(f"\n==== {name} ====", flush=True)
    sc = Scorer(name)
    res: dict = {"p1": [], "p2": []}

    if "p1" in PRONGS:
        for krel in KREL:
            for fmt in FORMATS:
                for nd in NDEMOS:
                    for seed in range(SEEDS):
                        pos_label, neg_label = (LABELS if seed % 2 == 0
                                                else (LABELS[1], LABELS[0]))
                        demo_txt, probe_txt, demos, probes = make_p1_trial(
                            fmt, nd, seed, krel)
                        prompts = [build_prompt(demo_txt, p, pos_label, neg_label)
                                   for p, _ in probe_txt]
                        ps = sc.label_probs(prompts, pos_label, neg_label)
                        kinds = [k for _, k in probe_txt]
                        rev = [p for p, k in zip(ps, kinds) if k.startswith("rev")]
                        both = [p for p, k in zip(ps, kinds) if k == "both"]
                        neither = [p for p, k in zip(ps, kinds) if k == "neither"]
                        sanity = (sum(p > 0.5 for p in both) + sum(p < 0.5 for p in neither)
                                  ) / (len(both) + len(neither))
                        # per-probe record: (kind, first-relevant-bits, p)
                        detail = [(k, "".join(map(str, g)), round(p, 4))
                                  for (g, k), p in zip(probes, ps)]
                        row = {
                            "krel": krel, "fmt": fmt, "n": nd, "seed": seed,
                            "detail": detail,
                            "lambda": sum(rev) / len(rev),
                            "sanity": sanity,
                            "p_both": sum(both) / len(both),
                            "p_neither": sum(neither) / len(neither),
                            "bayes": bayes_join_score(demos, probes),
                            "nn": nn_join_score(demos, probes),
                        }
                        for j in range(1, krel):  # per-lattice-level lambda
                            lv = [p for p, k in zip(ps, kinds) if k == f"rev{j}"]
                            row[f"lambda{j}"] = sum(lv) / len(lv)
                        res["p1"].append(row)
                done = [r for r in res["p1"] if r["fmt"] == fmt and r["krel"] == krel]
                lam = [r["lambda"] for r in done if r["sanity"] >= 0.75]
                print(f"  P1 k{krel} {fmt}: learned {len(lam)}/{len(done)}, "
                      f"lambda={sum(lam)/max(len(lam),1):.3f}", flush=True)

    if "p2" in PRONGS:
        for fmt in FORMATS:
            for concept in ("and", "or"):
                for seed in range(SEEDS):
                    pos_label, neg_label = (LABELS if seed % 2 == 0
                                            else (LABELS[1], LABELS[0]))
                    demo_txt, probe_txt = make_p2_trial(fmt, concept, seed)
                    prompts = [build_prompt(demo_txt, p, pos_label, neg_label)
                               for p, _ in probe_txt]
                    ps = sc.label_probs(prompts, pos_label, neg_label)
                    ys = [y for _, y in probe_txt]
                    acc = sum(p if y else 1 - p for p, y in zip(ps, ys)) / len(ps)
                    overcov = (sum(p for p, y in zip(ps, ys) if y == 0)
                               / max(sum(1 for y in ys if y == 0), 1))
                    undercov = (sum(1 - p for p, y in zip(ps, ys) if y == 1)
                                / max(sum(1 for y in ys if y == 1), 1))
                    res["p2"].append({
                        "fmt": fmt, "concept": concept, "seed": seed,
                        "acc": acc, "overcov": overcov, "undercov": undercov,
                    })
            for concept in ("and", "or"):
                a = [r["acc"] for r in res["p2"]
                     if r["fmt"] == fmt and r["concept"] == concept]
                print(f"  P2 {fmt} {concept}: acc={sum(a)/len(a):.3f}", flush=True)
    return res


def _run_all() -> dict:
    import torch
    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_res = {}
    if OUT.exists():
        try:
            all_res = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            all_res = {}
    for name in MODELS:
        try:
            all_res[name] = run_model(name)
        except Exception as e:  # one bad model must not sink the batch
            print(f"MODEL FAILED {name}: {type(e).__name__}: {e}", flush=True)
            all_res[name] = {"error": str(e)}
            torch.cuda.empty_cache()
            continue
        finally:
            torch.cuda.empty_cache()
        all_res["_meta"] = {"seeds": SEEDS, "ndemos": NDEMOS, "formats": FORMATS,
                            "labels": LABELS, "time_s": round(time.time() - t0, 1)}
        OUT.write_text(json.dumps(all_res), encoding="utf-8")
        print(f"wrote {OUT}", flush=True)
    return all_res


def _as_list(v):
    return v if isinstance(v, list) else str(v).split(",")


def run(config: dict | None = None) -> dict:
    """Python API entry point. Config keys mirror the EJB_* env vars without the
    prefix (lowercase): models, seeds, ndemos, formats, prongs, krel, nattrs,
    dtype, batch, out. Returns the merged results dict (also written to disk)."""
    config = config or {}
    global MODELS, SEEDS, NDEMOS, FORMATS, PRONGS, KREL, BATCH, _DTYPE_NAME, OUT, WORD_ATTRS
    if "models" in config:
        MODELS = _as_list(config["models"])
    if "seeds" in config:
        SEEDS = int(config["seeds"])
    if "ndemos" in config:
        NDEMOS = [int(x) for x in _as_list(config["ndemos"])]
    if "formats" in config:
        FORMATS = _as_list(config["formats"])
    if "prongs" in config:
        PRONGS = _as_list(config["prongs"])
    if "krel" in config:
        KREL = [int(x) for x in _as_list(config["krel"])]
    if "batch" in config:
        BATCH = int(config["batch"])
    if "dtype" in config:
        _DTYPE_NAME = str(config["dtype"])
    if int(config.get("nattrs", 4)) >= 5 and len(WORD_ATTRS) < 5:
        WORD_ATTRS = WORD_ATTRS + [("material", ("wooden", "metal"))]
    if "out" in config:
        OUT = pathlib.Path(config["out"])
    return _run_all()


def main():
    _run_all()


if __name__ == "__main__":
    main()
