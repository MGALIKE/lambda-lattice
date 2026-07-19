"""The credibility test.

Run the number-game instrument end-to-end on two *planted* oracles (no GPU):
  * a Bayes size-principle oracle, and
  * a numeric-proximity (similarity) oracle,
each of which re-parses the demos and probe straight out of the prompt text.
Then run the pre-registered scorer and assert it recovers the planted identity:

  Bayes oracle      -> RULE-like   (H-NG1) + contraction PRESENT (H-NG2)
  proximity oracle  -> SIMILARITY  (H-NG1) + contraction ABSENT  (H-NG2)

This is exactly the mock-oracle power check the number-game pre-registration
requires before any model data are scored. It runs in a couple of seconds.
"""
import json

import pytest

from lambda_lattice.numbers import harness as ng
from lambda_lattice.scoring import numbers as score


@pytest.fixture(scope="module")
def mock_scored(tmp_path_factory):
    out = tmp_path_factory.mktemp("mock") / "echo_numgame_mockcheck.json"
    res = ng.run({"models": "bayes:none,prox:none", "backend": "mock",
                  "seeds": 36, "out": str(out)})
    assert "bayes:none" in res and "prox:none" in res
    # the file the scorer reads is the same content the API returned
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert set(k for k in on_disk if not k.startswith("_")) == {"bayes:none", "prox:none"}
    scored = {r["key"]: r for r in score.score_files([str(out)])}
    return scored


def test_bayes_oracle_recovers_rule_and_contraction(mock_scored):
    r = mock_scored["bayes:none"]
    assert score.classify_ng1(r) == "RULE", (r["dfit"], r["dfit_z"])
    assert score.classify_slope(r) == "PRESENT", (r["slope"], r["slope_z"])
    # planted-identity consistency checks from the prereg
    assert r["dfit"] >= 0.10 and r["dfit_z"] >= 2
    assert r["a_lambda_off"] <= 0.25 and r["a_lambda_in"] >= 0.75
    assert r["slope"] <= -0.10 and r["slope_z"] <= -2


def test_proximity_oracle_recovers_similarity_and_flat(mock_scored):
    r = mock_scored["prox:none"]
    assert score.classify_ng1(r) == "SIMILARITY", (r["dfit"], r["dfit_z"])
    assert score.classify_slope(r) == "ABSENT", (r["slope"], r["slope_se"])
    assert r["dfit"] <= -0.10 and r["dfit_z"] <= -2
    # similarity stratum ordering: off > broad > in
    assert r["a_lambda_off"] > r["a_lambda_broad"] > r["a_lambda_in"]
    # power-gated null: near zero slope with a tight SE
    assert r["slope"] >= -0.05 and r["slope_se"] <= 0.05


def test_neither_oracle_is_excluded(mock_scored):
    # both oracles have full parse and pass the arm-a count gate
    for r in mock_scored.values():
        assert not r["excluded"]
