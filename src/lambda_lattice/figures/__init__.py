"""Figure pipeline: regenerates F1-F14 from the shipped data/ JSONs.

The driver modules (``make_figures``, ``make_f10``) keep the source's plain
sibling imports (``import refs`` / ``from loader import ...``) and add this
directory to ``sys.path`` at runtime, so they run both as ``python -m`` scripts
and via ``lambda_lattice.figures.build()``.
"""
from __future__ import annotations


def build():
    """Regenerate every figure into the repo's figures/ dir. Needs matplotlib."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_figures
    import make_f10
    from . import make_f11_f13_f14, make_f12
    make_figures.main()      # F1-F9 (per-figure resilient to missing optional data)
    make_f10.main()          # F10 reasoning extension
    make_f11_f13_f14.main()  # F11, F13, F14 (domain boundary / faithfulness / ladder)
    make_f12.main()          # F12 boundary map (five families)
