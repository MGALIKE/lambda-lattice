# paper/

The full research article for the λ-lattice study.

- **`paper.md`** — the complete write-up (GitHub-flavored markdown, ~8.7k words):
  abstract, 10 sections, all tables and figures. This is the publication
  deliverable (TMLR-target quality).

## What the paper is

A pre-registered *instrument* (not a benchmark) that locates where a language
model sits between similarity-based and rule-based in-context induction, and
whether it applies the Bayesian size principle. It reports four things:

1. A family-universal graded **similarity default** on Boolean concepts
   (`λ_j ≈ j/k`, the city-block GCM), 31/31 cells, 5 families, blind-predicted.
2. A three-way **dissociation** in reasoning models: the intersection operation
   is elicitable, the size-principle statistic is elicitable only when stated,
   the spontaneous policy is absent at every level.
3. A **domain gate**: the same weights are rule-like with size-principle
   contraction on Tenenbaum's number game.
4. The gate is itself **family-dependent** (cross-family sweep; universality
   bar not met — that failure is the finding).

## How the numbers trace to data and scorers

Every results paragraph in `paper.md` ends with a one-line provenance note
naming the raw JSON in `../data/` and the scorer that produced it. Section 10
(Reproducibility) contains the full **data manifest** table mapping each section
to its data file(s) and scorer. No number in the paper is new: all come from
`../README.md` (the audited results summary), the three pre-registration ledgers
in `../preregistrations/`, or figure-build computations over the shipped JSONs.
The independent adversarial audit (recomputation from raw JSON, seed-clustered
errors, git-timestamp verification of the pre-registrations) found zero
mismatches.

## How to regenerate the figures

```bash
pip install -e ..            # from the repo root: instrument + scorers
lambda-lattice figures       # regenerates F1–F10 into ../figures/
```

F11–F14 are built from the number-game and reasoning JSONs listed in the paper's
Section 10 data manifest (F11 domain boundary, F12 boundary map, F13
talk-rules-compute-similarity, F14 elicitation ladder). F15 was mooted: the
gpt-oss toggle cell it would have shown was excluded by the frozen parse gate
(twice), so Section 6.4 covers it in prose with no figure.

The paper embeds figures by relative path (`../figures/Fxx_*.png`); all
referenced files exist in `../figures/`.
</content>
