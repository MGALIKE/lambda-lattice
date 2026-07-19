"""Unit tests for the exact reference learners.

These pin down the pre-registered math with by-hand / logged anchors. If any of
these change, the science changed — which must not happen via a refactor.
"""
import math

import pytest

from lambda_lattice import references as R


# ---------------------------------------------------------------------------
# Boolean analytic ladder references (the "GCM profile" and its neighbours)
# ---------------------------------------------------------------------------
def test_additive_gcm_profile_k4():
    # the observed law and the k=4 pre-registered prediction: 0.25 / 0.50 / 0.75
    assert [R.additive(j, 4) for j in (1, 2, 3)] == [0.25, 0.5, 0.75]
    assert R.PREREG_K4 == {1: 0.25, 2: 0.50, 3: 0.75}


def test_additive_gcm_profile_k5():
    assert [R.additive(j, 5) for j in (1, 2, 3, 4)] == [0.2, 0.4, 0.6, 0.8]


def test_meet_join_flat_endpoints():
    for j in (1, 2, 3):
        assert R.meet(j, 4) == 0.0     # conjunctive: only j=k -> 1
        assert R.join(j, 4) == 1.0     # disjunctive: only j=0 -> 0
        assert R.flat(j, 4) == 0.5


def test_nn_step_profile():
    # idealized 1-NN: below/at half -> low, above -> high; k=4 => 1,2 low, 3 high
    assert R.nn_step(1, 4) == 0.01
    assert R.nn_step(2, 4) == 0.01     # tie 2j==k breaks low
    assert R.nn_step(3, 4) == 0.99
    # k=3 anchor from the task brief: j=1 low, j=2 high
    assert R.nn_step(1, 3) == 0.01
    assert R.nn_step(2, 3) == 0.99


# ---------------------------------------------------------------------------
# Boolean per-trial exact comparators
# ---------------------------------------------------------------------------
def test_grid_bits_size():
    for k in (2, 3, 4, 5):
        g = R.grid_bits(k)
        assert len(g) == 2 ** k
        assert len(set(g)) == len(g)


def test_boolean_bayes_and_nn_logged_constants():
    # bayes_join_score is a single constant across every k=4 model/format/seed
    # (logged 0.086212 — see references provenance); NN on the headline cell = 0.5
    from lambda_lattice.boolean.harness import make_p1_trial  # torch-free import
    _, _, demos, probes = make_p1_trial("f2", 8, 0, 4)
    assert R.bayes_join_score(demos, probes) == pytest.approx(0.086212, abs=1e-6)
    assert R.nn_join_score(demos, probes) == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Number-game hypothesis space, priors, exact Bayes
# ---------------------------------------------------------------------------
def test_hypothesis_space_size_is_5082():
    # 32 mathematical rules + all 5050 intervals [a,b] in [1,100]
    assert len(R.RULES_EXT) == 32
    assert len(R.HSPACE) == 32 + 5050 == 5082


def test_interval_prior_normalization():
    rule_w = sum(w for n, e, w in R.HSPACE if not n.startswith("["))
    iv_w = sum(w for n, e, w in R.HSPACE if n.startswith("["))
    assert rule_w == pytest.approx(0.5, abs=1e-9)      # rules share RULE_SHARE
    assert iv_w == pytest.approx(0.5, abs=1e-9)         # intervals share the rest
    assert rule_w + iv_w == pytest.approx(1.0, abs=1e-9)


def test_bayes_repeat_is_exactly_one():
    # a probe equal to a demonstrated member is in EVERY consistent hypothesis,
    # so its posterior predictive is exactly 1.0 — hand-derivable and exact.
    assert R.bayes_predict([10, 20, 30], [20])[20] == 1.0
    assert R.bayes_predict([7], [7])[7] == 1.0


def test_bayes_probabilities_are_valid():
    probs = R.bayes_predict([10, 20, 30, 40], list(range(1, 101)))
    for v, p in probs.items():
        assert 0.0 <= p <= 1.0 + 1e-12


def test_size_principle_contraction_on_off_value():
    # multiples of 10; probe 25 is off-rule and near. Once >= 2 demos pin the
    # narrow rule (mult10 / end0), each additional consistent demo makes 25 a
    # more "suspicious coincidence" -> P(25) contracts monotonically toward 0.
    # (n=1 is a single maximally-diffuse demo and is excluded from the trend, as
    # the number-game prereg notes the reference dynamic lives at n>=2.)
    demos_nested = [[10, 20], [10, 20, 30], [10, 20, 30, 40],
                    [10, 20, 30, 40, 50], [10, 20, 30, 40, 50, 60]]
    ps = [R.bayes_predict(d, [25])[25] for d in demos_nested]
    assert ps == sorted(ps, reverse=True)              # strictly non-increasing
    assert ps[0] > ps[-1]                              # and actually contracts
    assert ps[-1] < 0.05


def test_size_principle_in_rule_value_grows():
    # a held-out in-rule value (70, a multiple of 10, far from demos) rises toward
    # 1 as evidence accumulates for the narrow rule.
    demos_nested = [[10], [10, 20], [10, 20, 30], [10, 20, 30, 40]]
    ps = [R.bayes_predict(d, [70])[70] for d in demos_nested]
    assert ps[-1] > ps[0]
    assert ps[-1] > 0.9


def test_prox_reference_monotone_in_distance():
    # numeric-proximity similarity decays with distance to the nearest demo
    assert R.prox_predict([50], 50) == pytest.approx(1.0)
    assert R.prox_predict([50], 55) == pytest.approx(math.exp(-5 / R.PROX_SCALE))
    assert R.prox_predict([50], 55) > R.prox_predict([50], 60)
