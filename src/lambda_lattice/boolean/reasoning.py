"""boolean/reasoning.py — reasoning-model version of the lambda instrument.

Adapted (imports/paths/packaging only) from the pre-registered
``release/lambda-icl/src/echo_think.py``. Numeric logic unchanged.

Does reasoning training restore the version-space contraction that is absent in
all 31/31 non-reasoning Boolean cells? Generation-based lambda measurement
(validated: generation lambda=0.479 ~= logprob lambda=0.454 at 7B) with the
model's thinking mode as the manipulated variable. Same stimuli, references, and
gates as ``boolean/harness.py``.

Arms:
  a: profile        krel=4, nattrs=4, n=8 demos          (32 probes/trial)
  b: evidence sweep krel=2, nattrs=4, n in {4,8,16,32}   (8 probes/trial)

Env knobs (unchanged from the source):
  ETH_MODELS   comma list of "hf-name:mode" with mode in on|off|native|none
               (on/off = Qwen3 enable_thinking toggle; native = template always
               thinks, e.g. R1-distill; none = plain chat model)
  ETH_SEEDS    default 12
  ETH_ARMS     default a,b
  ETH_FMT      default f2
  ETH_MAXNEW   default 2048 (use 3072 for R1-distill)
  ETH_BACKEND  vllm|hf (default vllm)
  ETH_BATCH    hf-backend batch size, default 8
  ETH_OUT      default echo_think.json (merged per condition)
  ETH_NDEMOS_B default 4,8,16,32

Python API: ``reasoning.run({"models": "Qwen/Qwen3-8B:on", ...}) -> dict``.
Requires vllm or transformers+torch (no mock backend on the Boolean surface).
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import time

from .harness import LABELS, bayes_join_score, make_p1_trial, nn_join_score

HERE = pathlib.Path(__file__).resolve().parent


def _results_base() -> pathlib.Path:
    from .._paths import results_dir
    return results_dir()


OUT = _results_base() / os.environ.get("ETH_OUT", "echo_think.json")

MODELS = os.environ.get("ETH_MODELS", "Qwen/Qwen3-0.6B:on").split(",")
SEEDS = int(os.environ.get("ETH_SEEDS", "12"))
ARMS = os.environ.get("ETH_ARMS", "a,b").split(",")
FMT = os.environ.get("ETH_FMT", "f2")
MAXNEW = int(os.environ.get("ETH_MAXNEW", "2048"))
MAXLEN = int(os.environ.get("ETH_MAXLEN", "8192"))
BACKEND = os.environ.get("ETH_BACKEND", "vllm")
BATCH = int(os.environ.get("ETH_BATCH", "8"))
NDEMOS_B = [int(x) for x in os.environ.get("ETH_NDEMOS_B", "4,8,16,32").split(",")]
TEMP, TOP_P = 0.6, 0.95


def demo_block(demo_txt, pos, neg):
    lines = [f"Each item below is labeled '{pos}' or '{neg}'."]
    for text, y in demo_txt:
        lines.append(f"Item: {text} -> {pos if y else neg}")
    return "\n".join(lines)


def probe_prompt(block, probe, pos, neg):
    return (f"{block}\n\nItem: {probe}\n"
            f"Decide the label of this item. End your reply with exactly "
            f"'Answer: {pos}' or 'Answer: {neg}'.")


def strip_think(text: str) -> tuple[str, int]:
    """Return (final segment, think chars). Handles Qwen3 <think>, R1, gpt-oss."""
    think_chars = 0
    if "</think>" in text:
        head, _, tail = text.rpartition("</think>")
        think_chars = len(head)
        return tail, think_chars
    if "assistantfinal" in text:  # gpt-oss harmony decoded without special tokens
        head, _, tail = text.rpartition("assistantfinal")
        return tail, len(head)
    if "<|channel|>final<|message|>" in text:
        head, _, tail = text.rpartition("<|channel|>final<|message|>")
        return tail, len(head)
    return text, 0


def parse_label(final_seg: str, pos: str, neg: str):
    """Prefer last 'Answer: X'; fallback last occurrence of either label."""
    m = list(re.finditer(rf"answer\s*[:\-]?\s*\**\s*({pos}|{neg})",
                         final_seg.lower()))
    if m:
        return int(m[-1].group(1) == pos)
    hits = [(mm.start(), lab) for lab in (pos, neg)
            for mm in re.finditer(re.escape(lab), final_seg.lower())]
    if not hits:
        return None
    return int(max(hits)[1] == pos)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
def render_chat(tok, user_msg: str, mode: str) -> str:
    kw = {}
    if mode in ("on", "off"):
        kw["enable_thinking"] = mode == "on"
    elif mode in ("high", "low"):  # gpt-oss reasoning effort
        kw["reasoning_effort"] = mode
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True, **kw)


class VllmBackend:
    def __init__(self, name: str):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tok = AutoTokenizer.from_pretrained(name)
        self.llm = LLM(model=name, dtype="bfloat16", max_model_len=MAXLEN,
                       gpu_memory_utilization=0.92, enforce_eager=False)

    def generate(self, user_msgs: list[str], seeds: list[int],
                 mode: str) -> list[tuple[str, bool]]:
        from vllm import SamplingParams

        prompts = [render_chat(self.tok, m, mode) for m in user_msgs]
        params = [SamplingParams(temperature=TEMP, top_p=TOP_P, max_tokens=MAXNEW,
                                 seed=s) for s in seeds]
        outs = self.llm.generate(prompts, params)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append((c.text, c.finish_reason == "length"))
        return res


class HfBackend:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="auto").eval()

    def generate(self, user_msgs: list[str], seeds: list[int],
                 mode: str) -> list[tuple[str, bool]]:
        res = []
        for i in range(0, len(user_msgs), BATCH):
            chunk = [render_chat(self.tok, m, mode) for m in user_msgs[i:i + BATCH]]
            self.torch.manual_seed(seeds[i])
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=MAXNEW, do_sample=True,
                    temperature=TEMP, top_p=TOP_P,
                    pad_token_id=self.tok.pad_token_id)
            n_in = enc["input_ids"].shape[1]
            for row in out:
                text = self.tok.decode(row[n_in:], skip_special_tokens=True)
                res.append((text, len(row) - n_in >= MAXNEW))
        return res


# ---------------------------------------------------------------------------
# one condition = one (model, mode)
# ---------------------------------------------------------------------------
def trial_cells():
    cells = []
    if "a" in ARMS:
        cells += [("a", 4, 8, s) for s in range(SEEDS)]
    if "b" in ARMS:
        cells += [("b", 2, nd, s) for nd in NDEMOS_B for s in range(SEEDS)]
    return cells


def run_condition(be, name: str, mode: str) -> dict:
    cells = trial_cells()

    # build every prompt up-front; one big generate (vLLM continuous batching)
    prompts, seeds, meta = [], [], []
    trials = {}
    for arm, krel, nd, seed in cells:
        pos, neg = LABELS if seed % 2 == 0 else (LABELS[1], LABELS[0])
        demo_txt, probe_txt, demos, probes = make_p1_trial(FMT, nd, seed, krel)
        block = demo_block(demo_txt, pos, neg)
        key = (arm, krel, nd, seed)
        trials[key] = {"pos": pos, "neg": neg, "demos": demos, "probes": probes,
                       "probe_txt": probe_txt, "rows": []}
        for idx, (ptxt, kind) in enumerate(probe_txt):
            prompts.append(probe_prompt(block, ptxt, pos, neg))
            # deterministic per-probe seed (no str hash — that's randomized)
            seeds.append(1_000_000 + krel * 200_000 + nd * 4_000
                         + seed * 100 + idx)
            meta.append((key, idx, kind))

    print(f"[{name}:{mode}] {len(prompts)} prompts ...", flush=True)
    t0 = time.time()
    outs = be.generate(prompts, seeds, mode)
    gen_s = time.time() - t0
    print(f"[{name}:{mode}] generation done in {gen_s:.0f}s", flush=True)

    thinks = mode in ("on", "native", "high", "low")
    traces = []
    for (key, idx, kind), (text, truncated) in zip(meta, outs):
        tr = trials[key]
        final_seg, think_chars = strip_think(text)
        # a thinking model that got truncated before closing its reasoning block
        # produced no answer — never parse label mentions inside raw reasoning
        if truncated and thinks and think_chars == 0:
            ans = None
        elif truncated and not final_seg.strip():
            ans = None
        else:
            ans = parse_label(final_seg, tr["pos"], tr["neg"])
        bits = "".join(map(str, tr["probes"][idx][0]))
        tr["rows"].append({"kind": kind, "bits": bits, "ans": ans,
                           "think_chars": think_chars, "trunc": int(truncated)})
        if len(traces) < 20 and kind.startswith("rev"):
            traces.append({"key": str(key), "kind": kind,
                           "think": text[:2000], "final": final_seg[:300]})

    # per-trial stats
    trial_rows = []
    for (arm, krel, nd, seed), tr in trials.items():
        rows = tr["rows"]
        rev = [r["ans"] for r in rows if r["kind"].startswith("rev")
               and r["ans"] is not None]
        both = [r["ans"] for r in rows if r["kind"] == "both" and r["ans"] is not None]
        neither = [r["ans"] for r in rows
                   if r["kind"] == "neither" and r["ans"] is not None]
        n_ans = sum(1 for r in rows if r["ans"] is not None)
        sanity = ((sum(both) + sum(1 - a for a in neither))
                  / max(len(both) + len(neither), 1))
        row = {"arm": arm, "krel": krel, "n": nd, "seed": seed,
               "lambda": sum(rev) / max(len(rev), 1),
               "n_rev": len(rev), "sanity": sanity,
               "parse_rate": n_ans / len(rows),
               "trunc_rate": sum(r["trunc"] for r in rows) / len(rows),
               "think_chars": sum(r["think_chars"] for r in rows) / len(rows),
               "bayes": bayes_join_score(tr["demos"], tr["probes"]),
               "nn": nn_join_score(tr["demos"], tr["probes"]),
               "detail": [(r["kind"], r["bits"], r["ans"]) for r in rows]}
        for j in range(1, krel):
            lv = [r["ans"] for r in rows if r["kind"] == f"rev{j}"
                  and r["ans"] is not None]
            row[f"lambda{j}"] = sum(lv) / max(len(lv), 1)
        # rule-commitment: best single-attribute rule consistency on revealers
        best = 0.0
        revrows = [r for r in rows if r["kind"].startswith("rev")
                   and r["ans"] is not None]
        if revrows:
            for a in range(krel):
                agree = sum(1 for r in revrows if int(r["bits"][a]) == r["ans"])
                best = max(best, agree / len(revrows),
                           (len(revrows) - agree) / len(revrows))
        row["rule_consistency"] = best
        trial_rows.append(row)
    return {"mode": mode, "fmt": FMT, "gen_s": round(gen_s, 1),
            "sampling": {"temp": TEMP, "top_p": TOP_P, "max_new": MAXNEW},
            "trials": trial_rows, "traces": traces}


def summarize(cond: dict):
    tr = [t for t in cond["trials"] if t["sanity"] >= 0.75
          and t["parse_rate"] >= 0.8]
    out = {"n_learned": len(tr), "n_total": len(cond["trials"])}
    a = [t for t in tr if t["arm"] == "a"]
    if a:
        out["armA_lambda"] = sum(t["lambda"] for t in a) / len(a)
        for j in (1, 2, 3):
            v = [t.get(f"lambda{j}") for t in a if t.get(f"lambda{j}") is not None]
            if v:
                out[f"armA_lambda{j}"] = sum(v) / len(v)
        out["armA_rule_frac"] = sum(t["rule_consistency"] >= 0.9 for t in a) / len(a)
    for nd in NDEMOS_B:
        b = [t for t in tr if t["arm"] == "b" and t["n"] == nd]
        if b:
            out[f"armB_lambda_n{nd}"] = sum(t["lambda"] for t in b) / len(b)
            out[f"armB_bayes_n{nd}"] = sum(t["bayes"] for t in b) / len(b)
    out["mean_think_chars"] = (sum(t["think_chars"] for t in cond["trials"])
                               / max(len(cond["trials"]), 1))
    return out


def _run_all() -> dict:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_res = {}
    if OUT.exists():
        try:
            all_res = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            all_res = {}
    # group specs by model so the on/off toggle reuses ONE loaded model
    by_model: dict[str, list[str]] = {}
    for spec in MODELS:
        name, _, mode = spec.rpartition(":")
        if not name:
            name, mode = spec, "none"
        by_model.setdefault(name, []).append(mode)
    for name, modes in by_model.items():
        be = (VllmBackend if BACKEND == "vllm" else HfBackend)(name)
        for mode in modes:
            cond_key = f"{name}:{mode}"
            print(f"==== {cond_key} ====", flush=True)
            cond = run_condition(be, name, mode)
            cond["summary"] = summarize(cond)
            all_res[cond_key] = cond
            all_res["_meta"] = {"seeds": SEEDS, "arms": ARMS, "fmt": FMT,
                                "ndemos_b": NDEMOS_B, "backend": BACKEND}
            OUT.write_text(json.dumps(all_res), encoding="utf-8")
            print(json.dumps(cond["summary"], indent=1), flush=True)
            print(f"wrote {OUT}", flush=True)
        del be
    return all_res


def _as_list(v):
    return v if isinstance(v, list) else str(v).split(",")


def run(config: dict | None = None) -> dict:
    """Python API entry point. Config keys mirror the ETH_* env vars without the
    prefix (lowercase): models, seeds, arms, fmt, maxnew, maxlen, backend, batch,
    ndemos_b, out. Returns the merged results dict (also written to disk)."""
    config = config or {}
    global MODELS, SEEDS, ARMS, FMT, MAXNEW, MAXLEN, BACKEND, BATCH, NDEMOS_B, OUT
    if "models" in config:
        MODELS = _as_list(config["models"])
    if "seeds" in config:
        SEEDS = int(config["seeds"])
    if "arms" in config:
        ARMS = _as_list(config["arms"])
    if "fmt" in config:
        FMT = str(config["fmt"])
    if "maxnew" in config:
        MAXNEW = int(config["maxnew"])
    if "maxlen" in config:
        MAXLEN = int(config["maxlen"])
    if "backend" in config:
        BACKEND = str(config["backend"])
    if "batch" in config:
        BATCH = int(config["batch"])
    if "ndemos_b" in config:
        NDEMOS_B = [int(x) for x in _as_list(config["ndemos_b"])]
    if "out" in config:
        OUT = pathlib.Path(config["out"])
    return _run_all()


def main():
    _run_all()


if __name__ == "__main__":
    main()
