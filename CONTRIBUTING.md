# Contributing to lambda-lattice

Thanks for your interest. This repository backs a pre-registered scientific
study, so contributions come with one hard rule.

## The one hard rule: never silently change the science

The numbers in `data/`, and the code that produced them
(`lambda_lattice.references`, `boolean/harness.py`, `boolean/reasoning.py`,
`numbers/harness.py`, and every module in `scoring/`), are **pre-registered**.
The pre-registrations were committed to the source repository *before* the runs
they describe (see `preregistrations/`).

- Do **not** alter the numeric logic of any harness, reference learner, or
  scorer — thresholds, priors, likelihoods, gates, and statistics are frozen.
- Refactors that touch those files must be behaviour-preserving and demonstrably
  so (the reference and mock-oracle tests must still pass unchanged).
- New analyses belong in **new** modules with their **own** pre-registration if
  they make a confirmatory claim.

## Development setup

```bash
python -m pip install -e ".[dev]"      # numpy/scipy/matplotlib + pytest
python -m pip install -e ".[dev,models]"   # add torch+transformers to run models
```

## Running the tests

```bash
pytest -q                 # fast tests
pytest -q -m "not slow"   # skip the end-to-end mock-oracle run
```

The credibility test (`tests/test_mock_oracles.py`) runs both harness oracles in
mock mode and asserts the scorers recover the planted identities
(Bayes → RULE + contraction PRESENT; proximity → SIMILARITY + contraction
ABSENT). If your change breaks it, the change is wrong.

## Style

- Keep the env-var interface of the harnesses intact; the Python `run(config)`
  API is a thin wrapper over the same globals.
- Prefer adding importable functions over print-only scripts.
- Every number that appears in the README must trace back to `data/` or the
  pre-registration ledgers — do not invent results.
