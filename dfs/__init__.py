"""DraftKings DFS recommender built on top of the cached scouting-report data.

The scouting pipeline already assembles everything a projection needs (lineup slot,
platoon splits, arsenal matchups, park/weather, projected team runs, bullpen quality).
This package reads those cached payloads from `.cache/report_data/` and turns them into
DK-point projections, value vs. salary, and a ranked slate board.

Nothing here re-pulls from an API, so a full slate builds in seconds.
"""

import os
import sys

# Same interpreter guard as scouting_report.py: plain `python` can resolve to a system
# install ahead of .venv, and the only symptom is a cryptic missing-module error.
try:
    from .slate import build_slate
    from .board import render_board
except ImportError as _import_error:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _venv = os.path.join(_root, ".venv", "Scripts", "python.exe")
    if not os.path.normcase(sys.executable).startswith(os.path.normcase(os.path.join(_root, ".venv"))):
        sys.stderr.write(
            f"\n[!] Missing dependency: {_import_error.name}\n"
            f"    Running under: {sys.executable}\n"
            f"    This is not the project's .venv. Call it explicitly:\n"
            f"      {_venv} -m {'.'.join(__name__.split('.'))}...\n"
            "    or activate it once: .\\.venv\\Scripts\\Activate.ps1\n"
        )
        raise SystemExit(1)
    raise

__all__ = ["build_slate", "render_board"]
