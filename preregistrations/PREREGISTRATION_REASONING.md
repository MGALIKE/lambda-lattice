# PRE-REGISTRATION — Does reasoning training restore the version-space contraction?
### 2026-07-18 (overnight session) · committed BEFORE any GPU run

## Background (established, verified results this builds on)

1. Across **31/31 cells, 5 model families, 0.5B–32B, base+instruct** (jbias program,
   verified 2026-07-03): ICL concept generalization is **graded similarity integration
   (GCM-class)** — monotone graded λ-profiles, log-odds ≈ linear in signed feature
   match — with **no version-space contraction**: λ is flat in demo count n=4→32 while
   the exact size-principle Bayes reference contracts 0.11→0.03. Anti-scaling vs Chan
   et al. 2210.05675.
2. **K4 (lever, 7B)**: CoT *prompting* does NOT restore the meet — it breaks
   calibration (accepts 21% of true members' complements; sanity collapses).
   Meet-decoding is a precision knob, not a capability lever.
3. All prior cells are **non-reasoning models**. RL-trained reasoning (DeepSeek-R1
   style, Qwen3 thinking mode) trains the model to *search, verify, and backtrack* —
   precisely the operations hypothesis-elimination (the lattice meet / closure
   operator) would need.

## Novelty gate (run 2026-07-18, before this file was written)

- **2504.09387** (Suspicious Coincidences): zero-shot LMs show no size-principle;
  CoT/explicit-hypothesis prompting partially restores it. **No reasoning-trained
  models tested; no quantity-controlled λ instrument.** Our K4 result is in direct
  tension with their CoT finding on a calibration-gated design — added reason to run.
- **2511.00617** (Belief dynamics): sigmoidal ICL evidence accumulation, standard
  (non-reasoning) models, no version-space/lattice design.
- **2509.01016** (LLM hypothesis search): scaffolded external search loops, not native
  thinking-mode behavior.
- The conjunction — *RL-trained thinking mode × same-weights causal toggle ×
  lattice-position instrument × evidence-quantity sweep* — is **unclaimed**.

## Hypothesis

**H-R (restoration):** Reasoning-trained models *in thinking mode* implement
hypothesis-elimination that non-reasoning ICL lacks; behaviorally, thinking mode will
show version-space contraction on the λ instrument.

**Competing outcome K-R (no restoration / anti-scaling extends):** thinking mode shows
the same graded, evidence-quantity-insensitive profile as all 31 prior cells — the
additive/graded law survives reasoning training; deliberation rides on top of, and does
not replace, similarity integration.

Both outcomes are informative and will be reported as scored below.

## Design

Instrument: the existing λ lattice-position harness (`echo_join_bias.py` stimuli,
generation-based answers as validated in the lever wave: generation λ=0.479 ≈ logprob
λ=0.454 at 7B). Answers by chat generation; final answer parsed from post-`</think>`
text via the last occurrence of either nonce label; labels counterbalanced by seed.

**Arm A — profile (contraction in shape):** krel=4, nattrs=4 (kirr=0), n=8 demos,
format f2 (f1 robustness cell for Qwen3-8B if time), 12 seeds. 32 probes/trial
(8 rev1, 12 rev2, 8 rev3, 2 both, 2 neither).

**Arm B — evidence-quantity sweep (contraction in n):** krel=2, nattrs=4,
n ∈ {4, 8, 16, 32}, format f2, 12 seeds, 8 probes/trial. Exact Bayes(size-principle)
and 1-NN references computed per trial exactly as in the jbias program.

**Conditions (models):**
| condition | model | thinking |
|---|---|---|
| C1 | Qwen/Qwen3-8B | ON (enable_thinking=True) |
| C2 | Qwen/Qwen3-8B | OFF (same weights — the causal contrast) |
| C3 | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | native (always on) |
| C4 | Qwen/Qwen2.5-14B-Instruct | none (matched-scale non-reasoning control; known logprob λ=0.500) |
| C5 (stretch) | openai/gpt-oss-20b | reasoning effort high (cross-family) |
| C6 (stretch) | Qwen/Qwen3-14B | ON vs OFF |

Sampling identical across think/no-think: temperature 0.6, top_p 0.95, seeded;
max_new_tokens 2048 (Qwen3/gpt-oss), 3072 (R1-distill). The ONLY difference between
C1 and C2 is the chat-template thinking toggle.

## Pre-registered predictions & kill criteria

Let λ̄ = mean λ over learned trials (sanity ≥ 0.75), seed-clustered SEs, z = Δ/SE.

**H-R is CONFIRMED only if BOTH:**
- (i) Arm B, thinking conditions: λ(n=32) − λ(n=4) ≤ −0.10 with z ≥ 3 (direction of
  the Bayes reference), while the matched non-thinking condition stays flat; AND
- (ii) Arm A: λ̄(C1) − λ̄(C2) ≤ −0.10 with z ≥ 3, profile displaced toward the
  per-level Bayes reference (not merely noisier).

**K-R (no restoration) is the verdict if:** |λ(n=32)−λ(n=4)| < 0.05 (or z < 2) in
thinking conditions AND the Arm-A thinking profile is monotone graded with
|λ̄(C1)−λ̄(C2)| < 0.05.

**Partial outcomes (report as such, no spin):** one signature without the other;
or contraction in C3 (R1-distill) but not C1 (could be distillation-data effect).

**Rule-commitment discriminator (crucial subtlety, pre-registered):** aggregate λ can
look graded even if each *trial* commits to a discrete single-attribute rule (random
rule choice across trials mimics additivity in the mean). Per trial we compute
best-single-attribute rule consistency over revealer answers; a trial with
consistency ≥ 0.9 is "rule-committed". Non-reasoning baseline from the jbias corpus:
9–21% of trials. If thinking conditions show > 50% rule-committed trials, we report
**"reasoning induces discrete hypothesis selection"** as a distinct positive finding
EVEN IF aggregate λ is graded — with the n-sweep telling us whether the selected
hypothesis responds to evidence quantity (true contraction) or not (rule pinning).

**Calibration failure mode:** if a thinking condition's sanity (both→pos,
neither→neg) < 0.75, that condition is excluded from λ claims and reported as
"deliberation breaks calibration survives RL training" (would replicate K4 at the
training level). Cells with > 20% unparseable/truncated answers are excluded and the
rate reported.

**What we will NOT claim:** any single-model result as a law; C3-only effects as
"reasoning" effects (C3 differs from C4 in data AND base); anything from cells that
fail sanity or parse gates.

## Analysis plan

Per condition: λ̄ + CI (seed clusters); per-level λ_j vs graded prediction j/k and vs
exact Bayes reference; Arm-B slope λ(n) with clustered SE; rule-commitment fraction;
think-length distribution; sample of 20 thinking traces read and qualitatively coded
(does the model verbalize hypothesis elimination? does its final answer follow its own
stated rule?) — coding criteria: (a) states a candidate rule, (b) tests it against ≥1
demo, (c) revises, (d) answer consistent with final stated rule.

Everything logged per-probe (bits, kind, parsed answer, think length). Raw JSON
committed. This file must be committed BEFORE the first Modal run of
`echo_think.py`; any amendment gets its own dated section committed before the run
it governs.

## OUTCOMES (scored 2026-07-18 ~04:0x, after v2/clean runs; numbers independently
## recomputed via a second code path before this section was written)

**Verdict: K-R — NO RESTORATION. H-R is dead by both of its pre-registered criteria.**

| condition | gates | Arm-A λ (profile l1/l2/l3) | Arm-B λ(n=4→32) | d(32−4), z |
|---|---|---|---|---|
| C1 Qwen3-8B think (v2) | 56/60, parse .96, 9.1k think chars | 0.699 (0.42/0.78/0.84) | .545/.614/.528/.533 | **+0.000, z=0.00** |
| C2 Qwen3-8B no-think | 57/60, parse 1.00 | 0.649 (0.43/0.70/0.79) | .800/.659/.542/.750 | −0.050, z=−0.56 |
| C3 R1-Distill-14B (v2) | 59/60, parse .99, 5.6k think chars | 0.628 (0.53/0.64/0.71) | .771/.652/.521/.632 | −0.139, z=−1.45 (ns) |
| C4 Qwen2.5-14B ctrl | 58/60, parse 1.00 | 0.696 (0.50/0.72/0.86) | .675/.688/.583/.750 | +0.050, z=+0.41 |
| C6 Qwen3-14B think | 57/60, parse 1.00, 7.5k chars | 0.643 (0.42/0.65/0.86) | .614/.727/.500/.614 | +0.000, z=0.00 |
| C6 Qwen3-14B no-think | 55/60, parse 1.00 | 0.604 (0.53/0.60/0.68) | .475/.625/.477/.646 | +0.125, z=+0.96 |
| C5 gpt-oss-20b high | **EXCLUDED** parse .55 (45% trunc at 6144) | — | — | — |
| 32B Qwen3-32B think | 57/60, parse 1.00, 6.4k chars | 0.604 (0.39/0.62/0.79) | .545/.700/.542/.528 | −0.015, z=−0.12 |
| 32B Qwen3-32B no-think | 57/60, parse 1.00 | 0.640 (0.42/0.65/0.84) | .600/.636/.604/.625 | +0.025, z=+0.19 |

**32B toggle (appended ~05:4x): same verdict at the third scale.** think−nothink
d=−0.036 (z=−1.34, paired); flat n-sensitivity both modes; graded profiles; 0%
rule commitment. The toggle null now holds at 8B, 14B, and 32B, on two formats.

- Restoration criterion (i): required λ(32)−λ(4) ≤ −0.10 at z ≥ 3 in thinking
  conditions. Observed: **+0.000 (z=0.00)** at 8B-think, +0.000 (z=0.00) at
  14B-think, −0.139 (z=−1.45, ns) at R1. Exact Bayes reference contracts
  0.19 → 0.000 over the same sweep. **FAIL.**
- Restoration criterion (ii): required think−nothink ≤ −0.10 at z ≥ 3 (paired by
  seed). Observed: **+0.050 (z=+1.24)** at 8B, **+0.039 (z=+0.54)** at 14B — wrong
  direction. **FAIL.**
- K-R conditions met: thinking Arm-A profiles are graded-monotone
  (0.42/0.78/0.84; 0.42/0.65/0.86; 0.53/0.64/0.71), |think−nothink| ≈ 0.04–0.05,
  n-insensitivity flat. **CONFIRMED.**
- Rule-commitment discriminator: ≥0.9 consistency in **0%** of trials in every
  thinking condition (8% in the non-reasoning control) — no discrete hypothesis
  selection at trial level either.
- Trace coding (60 revealer traces, 3 thinking conditions, regex markers +
  manual read of 8): **0/60 elimination language, 0/60 ambiguity/underdetermination
  acknowledgments**; single-factor election language in 3–11/20; full-conjunction
  statements in 3–10/20 — yet stated conjunctions are not applied as meets
  (profiles graded). Deliberation narrates similarity; it does not manage a
  hypothesis space.
- Exploratory (NOT a claim): v1 R1 monotonicity-violation elevation (7% vs 1.4%)
  did NOT replicate in clean v2 data (violations scatter 0.2–5.3% across all
  conditions; only 1/10 pairwise contrasts nominally z>2). Dropped.
- Interpretation, honestly bounded: with matched-weights causal toggles at two
  scales plus an RL-distilled reasoner, RL-trained deliberation leaves the graded
  similarity-integration law of ICL concept generalization intact — no
  version-space contraction, no evidence-quantity sensitivity, no rule
  commitment. Extends the anti-scaling result (vs Chan 2210.05675) to reasoning
  training; sharpens the K4 CoT result: the failure is not prompt-level, it
  survives RL training FOR deliberation. 32B toggle running at scoring time;
  will be appended when it lands.
- **Format robustness (f1, appended ~04:2x):** 8B toggle re-run on the adjective
  format: think−nothink d=+0.007 (z=+0.15); profiles graded both modes
  (0.51/0.66/0.79 think, 0.59/0.63/0.73 no-think); n-trend small and IDENTICAL
  across think (−0.111, z=−1.21) and no-think (−0.136, z=−1.00) — nothing
  thinking-specific, and nowhere near the Bayes collapse (λ stays ≥0.54 where
  Bayes reaches 0.000). The null is not a format artifact.

### AMENDMENT C (2026-07-18 ~07:3x, committed BEFORE the run it governs):
### the constructive lever — does instructed discipline IMPROVE true-AND accuracy?

Amendment B showed instructed version-space discipline contracts coherently
(calibration intact) in a reasoning model — where K4's two-step scaffold on a
non-reasoning model collapsed recall (0.42). Constructive test on the P2 prong
(disambiguated AND concept, 12 demos covering all four relevant cells, ground
truth = the conjunction): arm "c", Qwen3-8B thinking, direct instruction vs
meet-instruction, 12 seeds, f2, 8 probes/trial (2 per cell).

Pre-registered outcomes (per-trial balanced accuracy = (pos-recall +
neg-recall)/2, seed-clustered, paired by seed):
- **Lever verdict** if balanced accuracy (instructed) − (direct) ≥ +0.05 with
  z ≥ 2 AND pos-recall drop ≤ 0.10 (no K4-style collapse). Read: one-pass
  version-space instruction is a working capability lever in reasoning models.
- **Kill K-C** if balanced accuracy is not better (d < +0.05 or z < 2), or
  pos-recall drops > 0.10 (K4 failure mode replicates at the reasoning level).
- Gates as before: parse ≥ 0.8 per trial; cells > 20% unparsed excluded.

**AMENDMENT C OUTCOME (scored ~08:0x): K-C — lever NOT claimed.** 12/12 trials
pass gates in both conditions. direct: bal-acc 0.785±0.050, pos-recall 0.583,
overcov 0.014. meet-instructed: bal-acc 0.833±0.056, pos-recall 0.667, overcov
0.000. Paired diff +0.049±0.028, z=+1.74 — below BOTH pre-registered thresholds
(≥+0.05, z≥2). Reported facts, no spin: (1) the K4 failure mode does NOT
replicate — instructed discipline in a reasoning model *raised* positive recall
(+0.083) instead of collapsing it, and eliminated overcoverage; (2) the effect
direction is positive but underpowered at 12 seeds; a larger prereg'd
replication would be needed to claim a lever, and none is claimed here. Side
observation (exploratory): direct thinking mode under-accepts true conjunction
members (pos-recall 0.583) — deliberation is conservative on the AND cell.

### AMENDMENT C′ (2026-07-18 ~08:1x, committed BEFORE the run): high-power
### replication of the Amendment-C near-miss on FRESH seeds

The first Amendment-C sample (seeds 0–11) missed both thresholds narrowly
(+0.049, z=+1.74). Replication on **new seeds 12–47** (36 seeds, new stimuli),
identical conditions and criteria: **lever verdict** iff paired
bal-acc(meet−direct) ≥ +0.05 with z ≥ 2 on the NEW seeds alone AND pos-recall
drop ≤ 0.10. Secondary (reported, not decisive): pooled 48-seed estimate.
If the new sample misses again, the lever is dead (two consecutive misses) and
will be reported as such; no third attempt.

**AMENDMENT C′ OUTCOME (fetched & scored after client restart, ~13:3x): MISS —
the lever is DEAD (two consecutive misses; no third attempt, per prereg).**
36/36 fresh-seed trials pass gates in both conditions (parse 1.00). On the new
seeds alone: direct bal-acc 0.868, pos-recall 0.750, overcov 0.014;
meet-instructed bal-acc 0.854, pos-recall 0.708, overcov 0.000. Paired diff
bal-acc(meet−direct) = **−0.014 ± 0.013 (z=−1.10)** — the first sample's +0.049
not only fails to replicate, it reverses sign: a textbook
regression-to-the-mean near-miss. Secondary pooled 48-seed estimate: +0.002
(z=+0.14) — exactly zero. The single consistent effect across both samples is
overcoverage elimination (pooled −0.014, z=−2.07; meet-instruction removes the
rare false-accepts) — small, real-looking, but not the lever. The exploratory
"direct thinking under-accepts AND members" observation from the first sample
also regresses (pos-recall 0.583 → 0.750 on fresh seeds): small-sample noise,
withdrawn. Final ledger for Amendment C: **no constructive lever; instructed
meet-discipline on disambiguated conjunctions changes accuracy by ~0.**

### AMENDMENT D (2026-07-18 ~13:4x, committed BEFORE the run): is the SIZE
### PRINCIPLE elicitable? — the elicitation ceiling for evidence statistics

Amendment B's three-level ledger left one seam: instructed meet-contraction is
STATIC — λ(n) rises (z=+1.71) while exact Bayes contracts 0.19 → 0.000. Was
that because the instruction never mentioned evidence quantity? New condition
C8 (arms a+b, Qwen3-8B thinking ON, seeds 0–35, budget 14336): the C7 meet
instruction EXTENDED with an explicit statement of the size principle
(suspicious coincidence — more consistent examples ⇒ trust the narrowest
consistent rule more; few examples ⇒ stay permissive). This deliberately
TELLS the model the statistic we score; it is the maximal-elicitation
condition, so a null is the strong claim ("absent even when stated").

Novelty gate (run before this prereg): 2504.09387 (suspicious-coincidence
sensitivity) uses non-reasoning models + external hypothesis-listing scaffolds;
2503.04722 is probability estimation on coin flips; 2406.00793 is martingale
diagnostics. None instructs the size principle inside RL-trained deliberation
and scores the λ(n) dynamic on a gated lattice instrument.

Pre-registered outcomes (gates unchanged: sanity ≥ 0.75, parse ≥ 0.8; plus an
instruction-compliance gate: arm-A λ̄ must be ≤ 0.45 as in C7, else the
condition failed to elicit contraction at all and NO verdict is scored):
- **Statistics-elicitable verdict** iff arm-B λ(n=32) − λ(n=4) ≤ −0.10 with
  z ≥ 2 (paired by seed, seed-clustered SE as everywhere in this file).
- **Statistics-absent-under-instruction verdict** (the capacity-level null)
  iff point estimate ≥ −0.05 AND se ≤ 0.05 (i.e., a −0.10 effect would have
  been detected at z ≥ 2 — the power condition 36 seeds is sized for).
- Anything else: intermediate, reported without verdict.
- Secondary (reported, not decisive): sizep vs C7-meet comparison of λ̄ and
  slope on the overlapping seeds 0–11.

**AMENDMENT D OUTCOME (fetched & scored ~19:2x): HIT — first sample.** C8
(176/180 trials pass gates; parse 0.99, trunc 0.01, mean 23.4k think chars;
4 gated trials benign). Compliance gate PASSED: arm-A λ̄ = 0.386 ± 0.026
(≤ 0.45; weaker compliance than C7-meet's 0.211, as expected — the sizep
instruction explicitly licenses permissiveness at low n). Primary:
λ(n) = 0.514/0.400/0.417/0.365; paired slope λ(32) − λ(4) = **−0.169 ± 0.077,
z = −2.21** (33 seeds) — meets both prereg thresholds (≤ −0.10, z ≥ 2).
**Per prereg: statistics-elicitable** — stating the size principle inside the
instruction produces contraction dynamics that the bare meet instruction never
showed (C7: +0.152, z = +1.71, rising). Magnitude ~0.17 vs the Bayes
reference's 0.21 → 0.00 over the same range. Trace check: 18/20 sampled traces
explicitly engage narrowest-rule / suspicious-coincidence reasoning — genuine
compliance, not a parsing artifact.

Secondary (seeds 0–11 overlap): sizep λ̄ 0.381 vs meet 0.211; sizep slope on
this subsample **+0.021 ± 0.149 (z = +0.14)** vs meet +0.152 — i.e., the
full-sample contraction is carried by seeds 12–35; the overlap subsample is
flat (though its SE is wide enough to be consistent with either). Honest
caveats before claiming: (1) z = −2.21 barely clears the gate and Amendment C's
+1.74 first-sample died on fresh-seed replication; (2) the subsample
heterogeneity above; (3) λ(4) = 0.514 is elevated (the instruction's
"be permissive with few examples" clause), so part of the slope is the
instructed schedule at the low end, not only tightening at the high end — the
claim, if it replicates, is that the model can *execute an instructed
evidence-quantity schedule*, the strongest available elicitation reading.
**Claim is therefore HELD pending Amendment D′ (below).**

### AMENDMENT D′ (2026-07-18 ~19:3x, committed BEFORE the run): confirmatory
### replication of the D hit — symmetric to the C/C′ two-miss rule

Identical condition C8 (arms a+b, Qwen3-8B thinking ON, budget 14336, f2,
sizep instruction, same harness commit) on **36 FRESH seeds 36–71**
(ETH_SEED0=36). Output: `echo_think_d2_sizep.json`. Same gates, same
compliance gate (arm-A λ̄ ≤ 0.45), same scorer (`jthink_sizep_analyze.py`).

Decision rule (committed before any D′ data exist):
- **CONFIRMED (statistics-elicitable)** iff D′ paired slope λ(32) − λ(4)
  ≤ −0.10 with z ≥ 2. Two independent hits at 36 seeds each → the claim enters
  the paper.
- **If D′ misses**: the pooled 72-seed estimate decides — claim made iff pooled
  slope ≤ −0.10 with z ≥ 2 (paired by seed, seed-clustered SE); otherwise the
  D hit is reported as an unreplicated first sample, NO claim (mirror of the
  C/C′ outcome ledger).
- Compliance-gate failure in D′ → no verdict from D′; pooled rule applies to
  gated seeds only, reported as such.

**AMENDMENT D′ OUTCOME (run on H100 after two infra failures — A100 preemption
+ 12h queue starvation, then a billing-limit block; run itself clean, scored
~12:5x 2026-07-19): D′ INTERMEDIATE alone; POOLED RULE FIRES → CLAIMED.**
D′ (178/180 pass gates): compliance λ̄ = 0.339 ± 0.025 (gate passed);
λ(n) = 0.431/0.457/0.382/0.324; paired slope **−0.105 ± 0.061 (z = −1.71,
35 seeds)** — same sign as D, point estimate past the −0.10 magnitude bar,
short of z ≥ 2 → single-sample INTERMEDIATE. Per the pre-committed decision
rule, the pooled 72-seed estimate decides: **pooled slope = −0.136 ± 0.049,
z = −2.81 (68 paired seeds); pooled compliance λ̄ = 0.363 ± 0.018.** Both
pooled criteria met → **the statistics-elicitable verdict is CLAIMED**: when
the size principle is explicitly stated inside the instruction, RL-trained
deliberation produces genuine evidence-quantity contraction dynamics
(two same-sign samples, D −0.169 z=−2.21 / D′ −0.105 z=−1.71; contrast C/C′,
which died by sign reversal). Magnitude remains ~0.14 vs the Bayes reference's
0.19 → 0.000 — the elicited schedule is real but far shallower than
normative. Final three-level ledger of the reasoning program: **the meet
operation is elicitable (B); the size-principle statistic is elicitable when
stated (D+D′ pooled); the spontaneous policy of using either is absent at
every level tested (K-R, K4).**

### policy vs capacity — instructed version-space discipline in thinking mode

The K-R verdict shows reasoning models do not spontaneously contract. Follow-up
question: policy or capacity? New condition C7: Qwen3-8B thinking ON, identical
stimuli/arms/seeds, with the probe instruction replaced by explicit version-space
discipline: *"First list every rule that is consistent with ALL the labeled
items. Then answer '<pos>' ONLY IF this item satisfies EVERY one of those rules;
otherwise answer '<neg>'."* (Contrast with K4: that scaffold was two-step
external meet-decoding on a NON-reasoning model and broke calibration; this is
one-pass, inside RL-trained deliberation.)

Pre-registered outcomes:
- **Policy verdict** if: arm-A λ̄ drops ≥0.25 vs uninstructed think (0.699 →
  ≤0.45) with z ≥ 3, profile displaced toward the Bayes reference (λ1 ≤ 0.15),
  AND calibration holds (sanity ≥ 0.75, parse ≥ 0.8). Read: the meet is
  elicitable — its absence is a deliberation-policy default, not a capacity gap.
- **Capacity verdict** if: λ̄ stays ≥ 0.55 with intact calibration (instruction
  ignored), OR sanity < 0.75 (discipline destroys calibration, replicating K4 at
  the reasoning-training level).
- Intermediate (λ̄ in (0.45, 0.55) or partial): report as graded compliance,
  no verdict. Arm-B secondary: under the policy verdict, λ(n) should acquire a
  negative slope tracking Bayes.

*(Budget note, per Amendment-A precedent: first C7 run at 6144 truncated 33% —
rule enumeration is expensive — parse 0.67, 10/60, unusable; no
hypothesis-relevant numbers consulted beyond gates. Rerun at max_new=14336 /
ctx 16384. The 10/60 v1 file is kept as
`echo_think_qwen3_8b_meetinstr.json`.)*

**AMENDMENT B OUTCOME (scored ~07:1x): POLICY verdict — decisively — with an
informative failure of the secondary prediction.** C7 v2 (58/60 learned, parse
0.99, sanity 0.97, mean 18.4k think chars): arm-A λ̄ = **0.211** vs 0.699
uninstructed (drop 0.49 ≥ the 0.25 criterion; unpaired z ≈ 11); profile
(0.07/0.22/0.34), λ1 = 0.07 ≤ 0.15, displaced to the Bayes reference
(0.002/0.017/0.274); calibration INTACT. The meet is elicitable by one-pass
instruction inside RL-trained deliberation: **the absence is a policy default,
not a capacity gap** — and unlike the K4 scaffold on a non-reasoning model,
instructed contraction here does not break calibration (a genuine capability
delta of reasoning training). Secondary prediction FAILED, informatively:
λ(n) = .182/.250/.333/.326 — it RISES slightly with evidence (z=+1.71) while
Bayes falls to 0. Instructed discipline is **static**: the model executes the
form of the intersection but not the statistics of evidence accumulation. The
size principle remains absent even under explicit version-space instruction.

### AMENDMENT A (2026-07-18 ~02:45, committed BEFORE the rerun it governs)

First run outcome on the gates (no hypothesis-relevant numbers were used to decide
this): Qwen3-8B thinking-ON truncated 55% of chains at max_new_tokens=2048 → 0/60
trials pass the parse gate (prereg exclusion applied correctly). R1-Distill-14B
lost 11% at 3072. This is an instrument-budget failure, not an outcome. Fix:
max_new_tokens=6144 and max_model_len=8192 for ALL thinking conditions; rerun
C1 (Qwen3-8B:on) and C3 (R1-Distill) into *_v2.json files; v1 files kept. The
non-thinking conditions (C2, C4: parse 1.00) stand. Predictions and kill criteria
unchanged.
