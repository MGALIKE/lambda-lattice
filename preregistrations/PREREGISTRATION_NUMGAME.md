# PREREG DRAFT — NUMBER GAME (NG): does the graded similarity law survive
# Tenenbaum's own task? (single-task-family objection, killer-reviewer #1)

Status: **FROZEN at commit time (2026-07-18 ~20:xx), BEFORE any model run.**
No model data for this instrument exist at freeze; only the mock power check
(committed alongside) has been run. The launch is queued behind the Amendment
D′ run currently occupying the Modal slot. `ng_analyze.py` (H-NG1 verdict
scorer) to be validated against the mock data BEFORE model data are scored
(C7-precedent: scorer pre-hoc-validated on non-hypothesis data).

## Motivation

Everything established so far (graded λ_j ≈ j/k law, no version-space
contraction, thinking-toggle null K-R, meet elicitable-but-static B, size
principle elicitable-under-statement D/D′) lives on ONE task family:
Boolean-attribute concepts. The two nearest competitor results use
Tenenbaum's number game and point the OTHER way:

- **2512.20162**: GPT-class + o1-mini are RULE-like on the number game
  (λ_fit ≈ 1 in a rules-vs-similarity mixture) — the opposite of our
  similarity finding.
- **2504.09387**: no zero-shot size principle on non-reasoning models;
  PARTIAL restoration with CoT prompting (correlational — prompt changes,
  model changes; our thinking toggle is the same-weights causal version).

Port the exact instrument — per-trial exact Bayes size-principle reference,
n-sweep, thinking toggle, ""|meet|sizep instruction branches, sanity/parse
gates — to the number game. Both outcomes publishable: a similarity result
generalizes the law and contradicts 2512.20162; a rule result maps the domain
boundary of the bias and reconciles the literatures on one gated instrument.

## Instrument (frozen; code = `frontier_lab/echo_numgame.py`)

**Domain**: hidden categories of integers in [1,100], positives-only demos
(Tenenbaum 2000), probe = "Number: y" answered `wug`/`dax` (counterbalanced
per seed), identical prompt/parse/trace machinery as `echo_think.py`.

**Hypothesis space H (fixed in code, 5082 hypotheses)**:
- 32 mathematical rules: even, odd, squares, cubes, primes ≤ 100,
  multiples of k (k=3..10), powers of k (k=2..10, k¹ included, 1 excluded),
  numbers ending in d (d=0..9). Prior share 0.5, uniform (0.5/32 each).
- ALL intervals [a,b], 1 ≤ a ≤ b ≤ 100 (5050 — Tenenbaum's actual interval
  space). Prior share 0.5, Erlang length prior p ∝ len·exp(−len/σ), σ=10.
- Extensional duplicates (mult10 ≡ end0) kept as separate hypotheses;
  priors are over hypotheses, duplicate mass stacks (documented, fixed).

**Exact Bayes reference (per trial)**: strong sampling, P(D|h) = |h|^(−n) for
consistent h (repetition counts — i.i.d. with replacement, mirroring the
Boolean harness's repeated demos when n exceeds distinct items);
P(y∈C|D) = Σ_h 1[y∈h]·P(h|D).

**Similarity reference (per trial)**: prox(y,D) = exp(−min_{x∈D}|y−x| / 10)
— the graded numeric-proximity (GCM-analog) account.

**Demo rules (10)**, each with a designated BROADER superset hypothesis in H
and 4 held-out in-rule probes ≥ 8 above the demo pool (so rule membership and
numeric proximity are decorrelated by construction):

| rule | broad | demo pool | in_far probes |
|---|---|---|---|
| mult8 | mult4 | 8..64 (8) | 72 80 88 96 |
| mult6 | mult3 | 6..66 (11) | 78 84 90 96 |
| mult9 | mult3 | 9..63 (7) | 72 81 90 99 |
| mult4 | even | 4..80 (20) | 88 92 96 100 |
| end0 | mult5 | 10..60 (6) | 70 80 90 100 |
| end5 | mult5 | 5..55 (6) | 65 75 85 95 |
| end3 | odd | 3..53 (6) | 63 73 83 93 |
| end7 | odd | 7..57 (6) | 67 77 87 97 |
| end4 | even | 4..54 (6) | 64 74 84 94 |
| primes | odd | odd primes 3..71 (19) | 79 83 89 97 |

Rule assignment: `seed % 10`. Demo multisets NESTED across the n-sweep
(paired slopes).

**Probe strata** (arm a: 4/4/4/4+2 per trial; arm b: 2/2/2/1+1):
- `in_far` — in narrow rule, held out, ≥8 from any demo. Bayes → 1 with n;
  proximity: reject.
- `off_near` — demo±1, outside rule AND broad. Bayes → 0 with n (interval
  mass dies); proximity: accept, flat.
- `broad_only` — in broad rule only, ≥3 from every demo. Bayes → 0 with n
  (size principle, (|narrow|/|broad|)^n); proximity: mid, flat.
- `out_far` — outside both rules, ≥8 from every demo. Both accounts reject
  (sanity partner).
- `repeat` — demo re-membership (sanity, must accept; Bayes = 1 exactly).

**λ (headline)** := acceptance on `off_near` ∪ `broad_only` — generalization
beyond the narrowest consistent rule. Bayes contracts λ(n) → 0; proximity is
flat. **Δfit** := per-trial mean of (1−|ans−Bayes_p|) − (1−|ans−prox_s|) over
non-repeat probes — the rules-vs-similarity positioning index.

**Arms**: a = n=8 fixed (18 probes); b = n ∈ **{1,2,4,8,16,32}** (8 probes).
The sweep extends below the Boolean instrument's n=4 because positives-only
demos permit n=1 and the exact-Bayes contraction on this space is COMPLETE by
n=4 (λ_Bayes = 0.31/0.16/0.02/0.001/0/0 at n=1/2/4/8/16/32) — a 4..32 sweep
would have had no reference dynamic range. This was caught by the mandatory
mock power check, below.

**Gates (unchanged from the Boolean instrument)**: per trial sanity ≥ 0.75
(repeat accepted + out_far rejected), parse_rate ≥ 0.8; thinking-truncation
answers never parsed; condition excluded (and reported as such, gpt-oss
precedent) if overall parse < 0.8 or fewer than 24/36 arm-a trials pass gates.

## Instrument power check (run 2026-07-18, BEFORE any model data)

Sampling oracles for both accounts, answering from the PROMPT TEXT end to end
(`ETH_BACKEND=mock`, 36 seeds; prox oracle deliberately uses scale 6 ≠ the
reference's 10 to show robustness to scale mismatch):

| statistic | Bayes oracle | proximity oracle |
|---|---|---|
| armA Δfit (positioning) | **+0.592** | **−0.317** |
| armA λ_in / λ_off / λ_broad | 1.00 / 0.00 / 0.00 | 0.06 / 0.84 / 0.37 |
| armB paired slope λ(32)−λ(1) | **−0.309 ± 0.047 (z=−6.5)** | **+0.037 ± 0.046 (z=+0.8)** |
| armB in_far slope | +0.721 (z=+15.0) | +0.059 (z=+1.7) |
| exact-Bayes ref slope on same trials | −0.309 ± 0.022 | −0.309 ± 0.022 |

Both identities recovered; at 36 seeds the slope SE ≈ 0.047, so a −0.10
contraction is detectable at z ≈ 2.1 — the same power convention as
Amendment D. Raw file: `frontier_lab/results/echo_numgame_mockcheck.json`.

## Hypotheses & decision rules (all paired by seed, seed-clustered SE)

**H-NG1 (primary): uninstructed positioning on the number game.**
Condition: Qwen3-8B thinking ON (and OFF as the same-weights toggle control),
36 seeds (0–35), arms a+b, fmt ng1, INSTRUCT="". Verdict from arm-a Δfit:
- **RULE-like** iff Δfit ≥ +0.10 with z ≥ 2 (consistency check, reported:
  λ_off ≤ 0.25 and λ_in ≥ 0.75). Reads as replicating 2512.20162 on a gated
  instrument.
- **SIMILARITY-like** iff Δfit ≤ −0.10 with z ≥ 2 (check: λ profile ordered
  by proximity, λ_off > λ_broad > λ_in). Reads as the Boolean graded law
  generalizing to Tenenbaum's own task, contradicting 2512.20162.
- **MIXED/graded** otherwise: report the stratum profile, no headline verdict.

**H-NG2 (primary): contraction dynamic.** Arm-b paired slope
λ(n=32) − λ(n=1), thinking ON:
- **CONTRACTION PRESENT** iff slope ≤ −0.10 with z ≥ 2 (Bayes ref: −0.31).
- **CONTRACTION ABSENT** (power-gated null, Amendment-D style) iff point
  estimate ≥ −0.05 AND se ≤ 0.05 (a −0.10 effect would have been seen at
  z ≥ 2).
- Otherwise intermediate, no verdict.
Secondary dynamic: in_far slope (Bayes says +0.72; proximity ~0) — reported,
not verdict-bearing.

**H-NG3 (secondary): thinking toggle.** Paired think−nothink difference on
Δfit and on the H-NG2 slope (mirror of K-R). No thresholds pre-set; reported
with z. This is the same-weights causal test of 2504.09387's "CoT partially
restores the size principle" claim.

**H-NG4 (secondary, separate runs): instructed ceiling.** meet and sizep
branches (numbers wording, frozen in code), thinking ON, same seeds.
Compliance gate as Amendment D: arm-a λ̄ must drop below 0.45 vs uninstructed,
else no verdict. sizep slope scored with the D/D′ criteria (≤ −0.10, z ≥ 2).
Cross-domain question: does the D′-tested elicitation transfer to a domain
where narrow rules are numerically natural?

**Outcome → claim ledger** (all four cells publishable):
- RULE + PRESENT: number game is the domain boundary; the graded law is a
  property of multi-attribute similarity domains, not of ICL per se. One
  instrument, both literatures reconciled.
- SIMILARITY + ABSENT: the law and the no-contraction null GENERALIZE;
  2512.20162's λ_fit ≈ 1 does not survive proximity-matched strata
  (their mixture fit has no off_near/in_far decorrelation).
- RULE + ABSENT: **"rule selection without rule statistics"** — the model
  finds the narrowest rule but ignores evidence quantity; neither account
  fits; sharpest new claim, and the cleanest reading of 2504.09387's
  zero-shot null.
- SIMILARITY + PRESENT: proximity-positioned but evidence-sensitive —
  graded contraction without rule commitment; report as such.

**Kill criteria / exclusions**: parse gate as above; if BOTH H-NG1 and H-NG2
land "intermediate", the condition is reported descriptively with no claim;
no post-hoc stratification, no threshold changes after first model data.
Two-sample rule applies: any positive verdict intended for the paper needs a
fresh-seed replication (seeds 36–71) before it is claimed.

## Models, budget, runtime

- Primary: `Qwen/Qwen3-8B:on,Qwen/Qwen3-8B:off`, 36 seeds, arms a,b,
  ETH_FMT=ng1, ETH_MAXNEW=3072 (short numeric prompts; uninstructed thinking
  ran ~5–9k chars on the Boolean task), ETH_MAXLEN=8192 auto via launcher.
- Instructed arms (H-NG4): ETH_MAXNEW=14336 / ctx 16384 (Amendment-A
  precedent: rule enumeration is expensive).
- Prompts per condition: 648 (arm a) + 1728 (arm b) = 2376.
- Launcher: ZERO changes to `modal_app.py`. `jthink_multi` jobs_spec routes
  through the fmt field — `echo_think.main()` dispatches to `echo_numgame`
  when ETH_FMT starts with "ng":
  `modal run modal_app.py::jthink_multi --seeds 36 --jobs-spec
  "Qwen/Qwen3-8B:on,Qwen/Qwen3-8B:off|3072|echo_numgame_q8.json|ng1|"`
  (instructed: `...|14336|echo_numgame_q8_sizep.json|ng1|sizep`).
- Estimated A100-80 runtime: ON ≈ 25–45 min (~4–7M think tokens), OFF ≈ 5–10
  min → primary condition ≈ 1 h. Each instructed condition ≈ 1.5–2.5 h.
  Full bundle (primary + meet + sizep) ≈ 5–6 h.

## Scorer

`summarize()` in `echo_numgame.py` already emits Δfit, per-stratum λ by n,
Bayes/prox references by n, and paired slopes with seed-clustered SE and z —
the same fields the Amendment-D scorer (`jthink_sizep_analyze.py`) consumes
(`lambda`, `bayes`, `arm`, `n`, `seed`, `sanity`, `parse_rate`), so it runs
unchanged for the H-NG2/H-NG4 slope verdicts (`lo=1` instead of 4 via its
`paired_slope` args). A dedicated `ng_analyze.py` for the H-NG1 verdict is a
~40-line adaptation, to be committed with the prereg.

---

## OUTCOME — primary condition, first sample (2026-07-19, scored after run; prereg frozen 7d7c221, scorer 79dc18e)

**H-NG1: RULE-like, decisively — both toggle conditions.**
- think ON (gates: 137/252 trials pass — arm-b low-n cells heavily sanity-gated,
  see note; arm-a 31/36 passes exclusion bar): Δfit = +0.587 ± 0.008
  (z=+72.5); strata λ_in/λ_off/λ_broad = 0.992/0.000/0.024; consistency PASS.
- think OFF (230/252 pass): Δfit = +0.561 ± 0.011 (z=+51.3); strata
  0.993/0.007/0.104; consistency PASS.

**H-NG2: CONTRACTION PRESENT — both toggle conditions, spontaneous
(uninstructed).**
- think ON: paired slope λ(32)−λ(1) = −0.562 ± 0.113 (z=−4.97, 8 seeds).
- think OFF: −0.461 ± 0.048 (z=−9.63, 19 seeds). λ(n) tracks the Bayes
  reference (0.50/0.22/0.09/0.05/0.04/0.05 vs 0.31/0.17/0.02/0.00/0.00/0.00).

**H-NG3 toggle:** Δfit(think−nothink) = +0.026 (z=+2.13) — thinking marginally
MORE rule-like; slope difference +0.125 (z=+0.77, 4 seeds) — no evidence the
contraction depends on deliberation.

**Reading (outcome-ledger cell RULE+PRESENT): the number game is the domain
boundary.** The graded-similarity law and the no-contraction null of the
Boolean instrument are properties of the multi-attribute similarity domain,
not of ICL per se: on number concepts the SAME model, uninstructed, in BOTH
decode modes, selects narrow rules and applies size-principle contraction at
near-reference magnitude. This reconciles our 31-cell Boolean results with
2512.20162 (rule-like number game) on one gated instrument, and stands in
tension with 2504.09387's zero-shot null for non-reasoning models (our
no-think condition contracts strongly; different models/protocol — reported
as tension, not refutation).

**Honest notes:** (1) the thinking condition's arm-b low-n cells are heavily
gated (n=1: 8/36 seeds, n=2: 1/36 pass) — sanity at n=1 is intrinsically hard
(single demo); the slope still clears at z=−4.97 on surviving seeds, and the
no-think condition (19 seeds at n=1) independently shows the same verdict at
z=−9.63. Gating is per-prereg, applied unchanged. (2) think-ON overall parse
0.812 sits just above the 0.8 condition bar. (3) Per the frozen two-sample
rule, BOTH verdicts are held pending the fresh-seed replication (seeds 36–71,
echo_numgame_q8_rep.json, launched immediately after scoring; same criteria;
launch precedes this commit only by minutes — no replication data existed at
commit time).

## REPLICATION OUTCOME (seeds 36-71, scored 2026-07-19 ~13:1x) — CLAIMED

Fresh-seed replication reproduces every verdict:
- think ON (150/252 pass; parse 0.804; arm-a 29/36): Δfit = +0.568 ± 0.007
  (z=+84.3), strata 1.000/0.000/0.017 → **RULE-like**; slope λ(32)−λ(1) =
  −0.417 ± 0.082 (z=−5.09, 8 seeds) → **CONTRACTION PRESENT**.
- think OFF (233/252 pass): Δfit = +0.552 ± 0.010 (z=+56.8), strata
  0.993/0.000/0.090 → **RULE-like**; slope −0.396 ± 0.045 (z=−8.81, 24 seeds)
  → **CONTRACTION PRESENT**. λ(n) tracks Bayes (0.43/0.29/0.06/0.07/0.05/0.03
  vs ref 0.35/0.18/0.02/0/0/0).
- H-NG3 toggle: Δfit diff +0.021 (z=+2.65) — thinking again marginally MORE
  rule-like; slope diff −0.03 (z=−0.59) — contraction not deliberation-driven.

**Per the two-sample rule the RULE+PRESENT cell is CLAIMED: the number game is
the domain boundary of the graded-similarity default.** Same weights,
uninstructed, both decode modes: rule selection + size-principle contraction
at near-reference magnitude on number concepts, vs graded similarity with zero
contraction on Boolean attribute concepts (31/31 cells, 5 families).
Hypothesis-space induction in LLMs is DOMAIN-GATED, not absent and not
reasoning-training-dependent. Reconciles 2512.20162 (rule-like numbers) with
our Boolean program; tension with 2504.09387's zero-shot null (our no-think
condition contracts at z=−8.8) noted for the paper.

---

## AMENDMENT NG-F (2026-07-19, committed BEFORE any cross-family run):
## cross-family + frontier sweep of the number-game instrument

Status: FROZEN at commit time. No non-Qwen number-game model data exist.
Motivation: the CLAIMED domain-boundary result (RULE+PRESENT, two samples) is
single-family (Qwen3-8B). The Boolean side of the boundary is established
across 5 families; the number side must be too, or the claim degrades to
"Qwen3 is domain-gated". This was pre-declared as the natural next step in
both the README and the replication outcome above.

**Models (exact Boolean-program roster, ungated mirrors, mode `:off` = plain
chat template; the `enable_thinking` kwarg is inert in non-Qwen templates):**

| cell | model | maxnew |
|---|---|---|
| F-L | unsloth/Meta-Llama-3.1-8B-Instruct:off | 3072 |
| F-M | mistralai/Mistral-7B-Instruct-v0.3:off | 3072 |
| F-G | unsloth/gemma-2-9b-it:off | 3072 |
| F-O | allenai/OLMo-2-1124-13B-Instruct:off | 3072 |
| F-X (frontier reasoner) | openai/gpt-oss-20b:high | 14336 |

Everything else IDENTICAL to the primary condition: fmt ng1, arms a+b,
36 seeds (sample 1 = seeds 0-35), INSTRUCT="", temp 0.6 / top_p 0.95,
gates unchanged (trial sanity >= 0.75, parse >= 0.8, exclusion bar 24/36
arm-a; a family failing gates is EXCLUDED and reported — gpt-oss C5
precedent). Scorer: `ng_analyze.py` UNCHANGED (frozen 79dc18e), H-NG1/H-NG2
thresholds unchanged.

**Decision rules (frozen):**
- Per-family verdicts from the unchanged H-NG1 (Δfit ≥ +0.10, z ≥ 2 →
  RULE-like; ≤ −0.10, z ≥ 2 → SIMILARITY-like) and H-NG2 (slope ≤ −0.10,
  z ≥ 2 → PRESENT; ≥ −0.05 with se ≤ 0.05 → ABSENT power-gated).
- **Family-universality is CLAIMED iff ≥ 3 of the 4 non-frontier families
  land RULE-like AND PRESENT in sample 1, and every family entering the
  claim replicates both verdicts on fresh seeds 36-71 (two-sample rule;
  replication launched only for families with a sample-1 verdict).**
- Any family landing SIMILARITY-like or ABSENT (and replicating) is a
  **boundary-refinement result** — the domain gate is family-dependent —
  reported per-cell with equal prominence. Mixed/intermediate cells:
  descriptive report, no claim.
- F-X (gpt-oss-20b) is a frontier bonus cell, NOT part of the universality
  count; scored with the same rules; its parse gate outcome is reported
  either way (its Boolean-instrument exclusion at 6144 is on record).
- No threshold changes, no post-hoc strata, no model swaps after first data.

**Launch (one detached app, jobs sequential on one A100-80GB):**
`ETH_SEED0=0 modal run --detach modal_app.py::jthink_multi --seeds 36
--jobs-spec "unsloth/Meta-Llama-3.1-8B-Instruct:off|3072|echo_numgame_llama.json|ng1|;mistralai/Mistral-7B-Instruct-v0.3:off|3072|echo_numgame_mistral.json|ng1|;unsloth/gemma-2-9b-it:off|3072|echo_numgame_gemma.json|ng1|;allenai/OLMo-2-1124-13B-Instruct:off|3072|echo_numgame_olmo.json|ng1|;openai/gpt-oss-20b:high|14336|echo_numgame_gptoss.json|ng1|"`
Replication pass mirrors with ETH_SEED0=36 and `_rep` filenames.

**NG-F infra deviation (2026-07-19, before any NG-F data seen):** the F-O job
(OLMo-2-1124-13B-Instruct) crashed at engine init — the model's
max_position_embeddings is 4096, below the launcher-derived max_len
5120 (= maxnew 3072 + 2048). Relaunched with maxnew 2048 → max_len 4096
(prompts are ≤ ~700 tokens; non-thinking answers are single words, so no
truncation risk). No other change; no NG-F results existed at this edit.

## NG-F OUTCOME (2026-07-19, scored after both samples; prereg 80872d8, scorer
## unchanged 79dc18e; Mistral S1 rerun after container preemption destroyed the
## first run's output pre-persist — no data from the lost run were ever seen)

Two independent 36-seed samples per family (S1 seeds 0-35, S2 seeds 36-71).
All gates passed everywhere (parse 0.956-1.000; arm-a pass 25-36/36, bar 24).

| family | Δfit S1 / S2 | H-NG1 | slope S1 / S2 | in_far slope S1 / S2 | H-NG2 |
|---|---|---|---|---|---|
| Llama-3.1-8B | +0.300 (z=10.05) / +0.347 (z=10.66) | **RULE-like ×2** | −0.167 (z=−1.51) / −0.208 (z=−1.74) | +0.000 / −0.188 | intermediate ×2 |
| Gemma-2-9B | +0.345 (z=10.68) / +0.394 (z=16.21) | **RULE-like ×2** | −0.500 (z=−9.43) / −0.450 (z=−7.60) | −0.607 / −0.550 | **PRESENT ×2** |
| OLMo-2-13B | +0.034 (z=1.26) / +0.035 (z=1.43) | MIXED ×2 | −0.750 (z=−7.35) / −0.583 (z=−5.53) | −0.625 / −0.583 | **PRESENT ×2** |
| Mistral-7B-v0.3 | −0.089 (z=−3.35) / −0.026 (z=−0.78) | MIXED ×2 (proximity-ordered strata ×2) | +0.140 (z=+2.58) / +0.183 (z=+2.98) | +0.000 / +0.038 | intermediate ×2 (EXPANSION, descriptive) |
| (Qwen3-8B, claimed above) | +0.56-0.59 / +0.55-0.57 | RULE-like ×4 cells | −0.40…−0.56 | **+0.06…+0.18 (positive ×4)** | PRESENT ×4 cells |

**Frozen decision: family-universality NOT claimed** — only Gemma joins Qwen
at RULE+PRESENT (2 families; bar was ≥3 of the 4 non-frontier families plus
replication). The pre-registered alternative branch applies: **the domain
gate is FAMILY-DEPENDENT**, and the per-cell results are claimed under the
two-sample rule as boundary refinements:
- **Gemma-2-9B: RULE + CONTRACTION claimed** (both verdicts, both samples).
- **Llama-3.1-8B: RULE-like positioning claimed** (both samples); contraction
  unresolved (intermediate ×2 — point estimates negative, SEs too large; the
  power-gated ABSENT criterion never fired).
- **OLMo-2-13B: contraction-without-rule-positioning claimed** (MIXED ×2 +
  PRESENT ×2); its in_far slopes (−0.63/−0.58) show GLOBAL tightening toward
  the demo set, not Bayes-selective narrowing.
- **Mistral-7B: no verdict either axis** (frozen rules); descriptively
  proximity-leaning with replicated EXPANSION (+0.14 z=2.6 / +0.18 z=3.0) —
  the closest number-game analog of the Boolean similarity default.

**Selectivity note (descriptive, not verdict-bearing):** only Qwen3-8B shows
the full Bayes signature — off/broad acceptance collapsing while in_far
acceptance RISES (in_far slopes positive in all 4 cells). Gemma's contraction
carries a tightening component (in_far also falls). So "size-principle
contraction at near-reference magnitude with rising in-rule confidence" is,
so far, Qwen-specific; Gemma approximates it; OLMo tightens globally;
Llama freezes; Mistral expands.

**Exploratory synthesis (post-hoc framing, flagged as such):** the same five
families that are HOMOGENEOUS on Boolean attribute concepts (31/31 cells on
one graded default) are HETEROGENEOUS on number concepts — five distinct,
individually replicated inductive profiles. The similarity default is
family-universal; hypothesis-space induction on numbers is family-
idiosyncratic. This inverts the usual expectation that "capabilities differ,
biases converge" and is the sweep's sharpest sentence for the paper — clearly
labeled exploratory since the homogeneity/heterogeneity contrast was not a
pre-registered hypothesis.

F-X (gpt-oss-20b numgame) still running at this edit; scored in a later
section when it lands.
