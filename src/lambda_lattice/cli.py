"""lambda-lattice command-line interface.

    lambda-lattice run-boolean   [--models ... --seeds N ...]
    lambda-lattice run-numbers   [--models ... --backend mock ...]
    lambda-lattice score KIND    FILE [FILE ...]          (KIND: jbias|gcm|sizep|numbers)
    lambda-lattice figures                                (regenerate F1-F14)
    lambda-lattice selftest                               (mock-oracle recovery, no GPU)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def _kv_config(pairs):
    """Turn ["--seeds", "4", ...] leftovers into a {key: value} config dict."""
    cfg = {}
    it = iter(pairs)
    for tok in it:
        if tok.startswith("--"):
            key = tok[2:].replace("-", "_")
            val = next(it, "")
            cfg[key] = val
    return cfg


def _cmd_run_boolean(args, extra):
    from . import run_boolean
    cfg = _kv_config(extra)
    reasoning = bool(cfg.pop("reasoning", False))
    res = run_boolean(cfg, reasoning=reasoning)
    print(f"conditions: {[k for k in res if not k.startswith('_')]}")
    return 0


def _cmd_run_numbers(args, extra):
    from . import run_numbergame
    res = run_numbergame(_kv_config(extra))
    print(f"conditions: {[k for k in res if not k.startswith('_')]}")
    return 0


def _cmd_score(args, extra):
    from . import score
    if not args.kind:
        print("score requires KIND (jbias|gcm|sizep|numbers) and file paths")
        return 1
    out = score(args.kind, *args.files)
    if args.kind == "numbers" and out is not None:
        from .scoring.numbers import classify_ng1, classify_slope
        for r in out:
            print(f"{r['key']}: H-NG1={classify_ng1(r)}  "
                  f"H-NG2={classify_slope(r)}  "
                  f"(Dfit={r['dfit']:+.3f} z={r['dfit_z']:+.2f}; "
                  f"slope={r['slope']:+.3f} z={r['slope_z']:+.2f})")
    return 0


def _cmd_figures(args, extra):
    from .figures import build
    build()
    return 0


def selftest(seeds: int = 36, verbose: bool = True) -> bool:
    """Run the number-game instrument on both planted oracles (no GPU) and check
    the scorers recover the identities: bayes -> RULE + contraction PRESENT,
    prox -> SIMILARITY + contraction ABSENT. Returns True on full recovery.

    Uses 36 seeds by default: the pre-registered CONTRACTION-ABSENT verdict is a
    power-gated null (SE must clear 0.05), so it needs the full sample to
    recover — fewer seeds recover RULE/SIMILARITY/PRESENT but leave the null
    'intermediate'."""
    from . import run_numbergame
    from .scoring.numbers import classify_ng1, classify_slope, score_files

    tmp = Path(tempfile.gettempdir()) / f"lambda_lattice_selftest_{seeds}.json"
    if tmp.exists():
        tmp.unlink()
    run_numbergame({"models": "bayes:none,prox:none", "backend": "mock",
                    "seeds": seeds, "out": str(tmp)})
    scored = {r["key"]: r for r in score_files([str(tmp)])}
    expect = {"bayes:none": ("RULE", "PRESENT"),
              "prox:none": ("SIMILARITY", "ABSENT")}
    ok = True
    for key, (want_ng1, want_slope) in expect.items():
        r = scored.get(key)
        got_ng1 = classify_ng1(r) if r else "MISSING"
        got_slope = classify_slope(r) if r else "MISSING"
        good = (got_ng1 == want_ng1 and got_slope == want_slope)
        ok = ok and good
        if verbose:
            mark = "PASS" if good else "FAIL"
            print(f"[{mark}] {key}: H-NG1={got_ng1} (want {want_ng1}), "
                  f"H-NG2={got_slope} (want {want_slope})")
    try:
        tmp.unlink()
    except OSError:
        pass
    if verbose:
        print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def _cmd_selftest(args, extra):
    return 0 if selftest(seeds=args.seeds) else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="lambda-lattice",
                                description="pre-registered lambda-lattice instrument")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("run-boolean", help="run the Boolean attribute-concept instrument "
                   "(pass --models/--seeds/... ; needs torch)")
    sub.add_parser("run-numbers", help="run the number-game instrument "
                   "(pass --backend mock for the no-GPU oracle check)")

    ps = sub.add_parser("score", help="score result JSONs")
    ps.add_argument("kind", nargs="?", help="jbias|gcm|sizep|numbers")
    ps.add_argument("files", nargs="*", help="result JSON paths")

    sub.add_parser("figures", help="regenerate F1-F14 into figures/")

    pt = sub.add_parser("selftest", help="mock-oracle recovery check (no GPU)")
    pt.add_argument("--seeds", type=int, default=36)

    args, extra = p.parse_known_args(argv)
    dispatch = {
        "run-boolean": _cmd_run_boolean,
        "run-numbers": _cmd_run_numbers,
        "score": _cmd_score,
        "figures": _cmd_figures,
        "selftest": _cmd_selftest,
    }
    if args.cmd not in dispatch:
        p.print_help()
        return 1
    return dispatch[args.cmd](args, extra)


if __name__ == "__main__":
    raise SystemExit(main())
