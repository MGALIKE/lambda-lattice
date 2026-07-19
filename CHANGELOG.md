# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-07-19

Initial standalone open-source release, packaged from the pre-registered
`lambda-icl` study.

### Added
- `lambda_lattice` package with a torch-free top-level API
  (`run_boolean`, `run_numbergame`, `score`) and a `lambda-lattice` CLI
  (`run-boolean`, `run-numbers`, `score`, `figures`, `selftest`).
- `lambda_lattice.references` — the exact reference learners (size-principle
  Bayes on both domains, GCM/proximity, meet, join, 1-NN, the number-game
  hypothesis space) factored out of the harnesses, importable and unit-tested.
- Boolean attribute-concept instrument (`boolean/harness.py`) and its
  reasoning-model variant (`boolean/reasoning.py`).
- Tenenbaum number-game instrument (`numbers/harness.py`) with the no-GPU
  mock-oracle power check.
- Importable scorers (`scoring/`): `jbias`, `gcm`, `sizep`, `numbers`.
- Figure pipeline (`figures/`) regenerating F1–F10.
- Verbatim pre-registration ledgers, all raw result JSONs, and the F1–F10 PNGs.
- Test suite: reference-learner unit tests, the mock-oracle credibility test,
  and CLI smoke tests. CI on Python 3.10–3.12.

### Preserved
- No numeric logic in any harness, reference, or scorer was changed relative to
  the pre-registered source; this release is a packaging/adaptation of paths and
  imports only.
