"""Import-only smoke tests for the F11-F14 figure builders.

These confirm the new package-relative builders import cleanly (package plumbing,
matplotlib Agg backend) and expose their entry points, without running the full
render (which is exercised via `lambda-lattice figures`).
"""


def test_make_f11_f13_f14_imports():
    from lambda_lattice.figures import make_f11_f13_f14 as m
    assert callable(m.main)
    for fn in ("fig_f11", "fig_f13", "fig_f14"):
        assert callable(getattr(m, fn))


def test_make_f12_imports():
    from lambda_lattice.figures import make_f12 as m
    assert callable(m.main)
    # F12's Panel-B numbers come from the package's own frozen scorer
    from lambda_lattice.scoring import numbers as ng
    assert m.ng is ng
    assert callable(ng.load_conditions) and callable(ng.score_condition)
