# PRE-REGISTRATION — "The Join Bias" (2026-07-03, overnight)

**Hypothesis (H1, the law).** When an LLM learns a concept in-context from demos that
underdetermine the concept, its generalization sits systematically at the **join** (∨, union)
end of the lattice interval of consistent hypotheses, not at the **meet** (∧, version-space /
echo-closure of the positives) that the Bayesian size principle and documented human biases
(Feldman 2000: conjunctive advantage; Tenenbaum: size principle) favor.

**Echo-space reading.** The consistent hypotheses form an interval [cl_∧(S⁺), join-top] in the
concept lattice (thesis A.1: closure system; cl = FCA closure of the positive demos). H1 says
the model's implicit generalization operator is a JOIN-closure — the behavioral level of the
same join-native law this campaign measured at the KV level (x22: additive beats conjunctive
55–218%), the architecture level (MQAR: DeltaNet/join wins), and the training level
(echo_closure_learn: join can't extrapolate meet tasks). H1 completes it at the BEHAVIOR level.

**Mechanism conjecture (H2).** Additive attention aggregation (softmax-weighted SUM) and linear
superposition make single-forward-pass evidence combination disjunctive: features vote
additively, they do not gate multiplicatively. Predictions: (a) join bias persists or grows with
number of conjuncts k; (b) chain-of-thought (sequential gating) reduces it; (c) it appears
across model families/scales because the cause is architectural, not data-specific.

**Lever (H3, only if H1 fires).** "Meet decoding": factor the query into per-attribute
membership probes answered in-context, combine with product/min (the thesis meet operator).
At matched forward passes it should beat direct ICL on conjunctive concepts.

## Design

### Prong P1 — ambiguous demos (lattice-position instrument)
Objects = k_rel=2 relevant binary attributes (A, B) + k_irr irrelevant attributes (randomized).
Demos: n/2 positives all A∧B; n/2 negatives all ¬A∧¬B ⇒ A, B, A∧B, A∨B all consistent.
Probes: A-only and B-only items (revealers) + held-out both/neither items (sanity).
**λ (join score)** = mean P(positive-label | revealer probes), via label-token logprob.
- λ→0: meet (version-space closure); λ→1: join; λ≈0.5: indifference.

### Prong P2 — disambiguated concepts (accuracy asymmetry)
Concepts A∧B vs A∨B fully identified by demos (negatives/positives placed in the revealing
cells). Matched demo count, balanced labels. Measure accuracy on held-out grid + signed
coverage error (overcoverage = P(pos-label | out-of-concept) − P(neg-label | in-concept)).
Humans (Feldman): AND easier than OR. H1 predicts LLMs reverse or shrink this, with
AND errors = overcoverage.

### Controls (all pre-registered)
1. **Nonce labels, counterbalanced** ("wug"/"dax", assignment flipped per seed) — removes
   Yes-token response bias.
2. **Learned-trial filter**: a trial counts only if unambiguous-region accuracy ≥ 0.75
   (else the model learned nothing and λ is noise). Report unfiltered too.
3. **Formats**: F1 adjective phrases ("a red striped square"), F2 key=value lists,
   F3 numeric thresholds — surface-form robustness.
4. Demo order permuted per seed; probe order counterbalanced; ≥20 seeds/cell.
5. **Bayes comparator**: size-principle posterior λ_Bayes(n) for the same demo sets.
6. Scale: Qwen2.5-Instruct 0.5B→32B (+1 other family ≥1 model, OLMo-2-13B or Gemma).
   n_demos ∈ {4, 8, 16}.

### Pre-registered kill criteria
- **K1 (no law)**: pooled λ CI includes 0.5, or sign flips across formats/families
  ⇒ H1 dead; report null.
- **K2 (artifact)**: λ > 0.5 but disappears under nonce-label counterbalancing or is
  fully explained by unambiguous-region false-positive rate ⇒ artifact, dead.
- **K3 (meet result)**: λ < 0.5 robustly ⇒ LLMs are meet-biased like humans ⇒ H1 dead
  (interesting reversal, report as such; do NOT spin as confirmation).
- **K4 (lever)**: meet-decoding ≤ direct ICL at matched compute ⇒ H3 dead regardless of H1.
- P2 alone (accuracy asymmetry without λ signal) is NOT confirmation; both prongs must agree
  for the law claim.

### AMENDMENT (2026-07-03, ~03:50, pre-registered BEFORE the k=4 run returns)

The k=3 per-level references computed after wave 2 reframe the null: model profile
(λ₁, λ₂) = (0.23, 0.66) at 7B vs Bayes (0.005, 0.064), NN/prototype (0.01, 0.99),
meet (0,0), join (1,1). The model is close to λ_level ≈ (level/k), i.e.,
**P(generalize) ≈ fraction of satisfied conjuncts — additive evidence integration.**
Pre-registered k=4 prediction (3 free points, no fitting): levels 1/4, 2/4, 3/4 →
λ ≈ (0.25, 0.50, 0.75), allowing a modest global compression toward 0.5 and a small
meet-ward shift as seen at k=3; the ORDER and near-equal spacing are the claim.
KILLS: step-like profile (λ₁≈λ₂ low, λ₃ high = NN/prototype); flat profile
(≈0.5,0.5,0.5); or non-monotone. If the linear profile holds across 3B/7B/14B and
both formats → "additive evidence integration law" of ICL concept generalization —
the behavioral expression of the additive substrate (join-native law, level 4,
positive form).
### AMENDMENT 2 (2026-07-03, day session — pre-registered BEFORE the k=5 run)

Harness extended: 5th binary attribute ("material"), krel=5, so all 5 attributes are
relevant (kirr=0). Prediction from the additive-integration law with NO new free
parameters beyond the k≤4 fits: λ_j monotone graded in j = 1..4 with near-equal
spacing straddling 0.5 — approximately (0.2, 0.4, 0.6, 0.8) allowing the same modest
compression toward 0.5 seen at k=3,4; equivalently σ(β·(2j−k)) with β in the k=4
fitted range. KILLS: non-monotone; step-like (λ1≈λ2≈λ3 low, λ4 high, or NN-like);
flat (all ≈0.5); or spacing collapse such that adjacent levels are indistinguishable
while endpoints saturate (would indicate threshold, not additive, integration).
Models: Qwen2.5-7B-Instruct + Llama-3.1-8B-Instruct, formats f1,f2,f3, 16 seeds, n=8.

### AMENDMENT 3 (2026-07-03, day session — pre-registered BEFORE the run): GCM vs prototype

The verification audit proved that on kirr=0 designs (k=4, k=5 headline cells) the additive
law is algebraically identical to the city-block GCM exemplar model: logit P(pos) =
c·(#matched−#unmatched relevant) + log(S⁺irr/S⁻irr), where S±irr = summed exponential
similarity to positive/negative demo exemplars in the IRRELEVANT coordinates. The second
term is the only exemplar-vs-prototype discriminator: a prototype/pure-additive-in-relevant
model predicts NO within-level dependence on irrelevant overlap (positives' irrelevant attrs
are iid random, so the prototype's irrelevant component washes out), while GCM predicts a
POSITIVE within-level effect. Post-hoc residual checks at k=2/3 found positive correlations
(7B k=2 +0.22, slope +0.85 on logit; 3B +0.13; 7B k=3 j=2 +0.13).

DESIGN: nattrs=5, krel ∈ {2,3} (kirr = 3,2), full probe bits now logged; models
Qwen2.5-7B-Instruct + Llama-3.1-8B-Instruct; f1,f2,f3; 24 seeds; n=8 demos.
PREDICTION (GCM): pooled within-rev-level rank correlation between residual logit(p) and
the GCM irrelevant-overlap signal (computed at c fitted on relevant-level spacing) is
POSITIVE with pooled z > 3, in both models.
KILLS: pooled z < 3 (prototype/pure-additive wins — report as 'graded prototype
abstraction, not exemplar memory'); negative slope (neither model — report as anomaly).
Either outcome sharpens the reframed claim; neither rescues 'novel law'.

Closest work, each holding one piece but not the claim: Feldman 2000 (humans);
Wang/McCoy/Steinert-Threlkeld 2412.02823 (ICL Boolean-complexity bias; no AND/OR polarity,
no error-direction, no lattice framing); Wang 2510.01219 (quantifier monotonicity bias only —
directionally consistent with join bias, does not claim it); Bhattamishra 2310.03016 (Boolean
ICL sample-efficiency vs optimal learners; no polarity law); LIMIT+ 2605.03824 (conjunction
harder in *retrieval*); Peng et al. 2024 / Strassen attention (one-layer capability limits,
not behavioral ICL bias). The conjunction — lattice-position instrument + AND/OR polarity law
+ mechanism link + meet-corrective fix — is unclaimed as of tonight's search.
