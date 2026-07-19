# Pre-registrations

This study is pre-registered. The discipline is simple and verifiable: **the
prediction and the decision rule were committed to version control *before* the
run that tested them.** Mixed and negative outcomes are reported as such, and
several levers were killed by their own pre-committed criteria.

## The three ledgers (verbatim copies)

| file | what it fixes in advance |
|---|---|
| `PREREGISTRATION.md` | the original join-bias / λ instrument: the graded profile λ_j ≈ j/k predicted at k=4 (0.25/0.50/0.75) and k=5, the no-contraction and no-rule-shift-with-scale predictions, kill criteria K1–K3, and the amendment history (incl. the GCM-vs-prototype Amendment 3). |
| `PREREGISTRATION_REASONING.md` | the reasoning-model extension: the thinking-toggle null, the meet-instruction elicitation, and the size-principle Amendments A–D′ (including the two-miss death of the accuracy lever, C/C′, and the pooled decision rule for D/D′). |
| `PREREGISTRATION_NUMGAME.md` | the cross-domain number-game port: the H-NG1..H-NG4 hypotheses, probe strata that decorrelate rule membership from numeric proximity, the exact-Bayes reference, and the two-sample decision rule — frozen before any model run. |

Each ledger also records the **outcome** after the run, so the file is both the
pre-registration and the result ledger.

## How to verify the timing

Each ledger cites the **commit hashes** from the source repository at which the
pre-registration was committed and at which the corresponding results were
committed. Because these copies are verbatim, the hashes they cite are the
ground truth: in the source repo you can run, for example,

```bash
git log --oneline --reverse            # prereg commits precede result commits
git show <prereg-hash>:PREREGISTRATION_REASONING.md   # the frozen prediction
git show <result-hash> --stat                          # the run that tested it
```

and confirm that every `prereg` commit strictly precedes the `OUTCOME` commit it
gates. The amendment ledgers inside the files name those hashes explicitly
(e.g. "committed BEFORE the run" annotations on Amendments C′ and D).

The scorers in `lambda_lattice.scoring` implement exactly the thresholds written
in these ledgers, and the number-game scorer was validated against the
mock-oracle power check *before* any model data were scored (the C7 precedent
noted in `PREREGISTRATION_NUMGAME.md`).
