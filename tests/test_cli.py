"""CLI smoke tests: --help works, and selftest recovers the oracle identities."""
import pytest

from lambda_lattice import cli


def test_help_exits_zero():
    # argparse -h raises SystemExit(0)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_no_subcommand_prints_help_returns_1(capsys):
    rc = cli.main([])
    assert rc == 1
    out = capsys.readouterr().out
    assert "run-boolean" in out and "selftest" in out


def test_score_help_lists_kinds():
    with pytest.raises(SystemExit) as exc:
        cli.main(["score", "--help"])
    assert exc.value.code == 0


def test_selftest_recovers_planted_identities():
    # full-sample recovery (RULE+PRESENT / SIMILARITY+ABSENT); runs in ~2s
    assert cli.selftest(seeds=36, verbose=False) is True


def test_selftest_cli_entrypoint_returns_zero():
    assert cli.main(["selftest", "--seeds", "36"]) == 0
