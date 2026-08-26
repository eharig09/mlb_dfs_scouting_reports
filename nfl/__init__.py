"""DraftKings NFL DFS.

A fork of the `dfs` package rather than a shared core. That is a deliberate choice made
while the MLB side is live and being submitted against: extracting a sport-agnostic core
out from under a working system mid-season is the wrong risk, and the modules worth sharing
(naming, files, adopt, pool, exposure, profiling, results, upload) are exactly the ones with
no tuned constants, so merging them later is cheap.

**What is here.** The parts needed to prove the structure transfers:

    scoring      DK NFL Classic scoring, in event units
    salaries     team-code and player-name folding
    optimizer    the MILP, with FLEX handled by the same assignment model MLB uses
    profiling    copied verbatim from dfs.profiling; sport-agnostic

**What is not, and why.** There is no projection or data layer yet. `scouting_report.py`
has no NFL analogue and should not be translated; the intended source is nflverse
(play-by-play, snap counts, depth charts, injuries back to 1999), which is a better
backtesting substrate than the MLB side ever had -- walk-forward evaluation can run over
several completed seasons before a single week is played.

**Constants are not transferable.** Everything numeric in `dfs` was fitted on baseball:
the randomness and overlap defaults, the softmax ownership temperature, the ceiling z, the
whole calibrated field model. Where a placeholder was needed here it says so in a comment
next to the MLB value. None of it should be trusted until it has been re-measured.
"""

from .scoring import DK_CLASSIC_SLOTS, DK_SALARY_CAP, offense_points, dst_points

# **The optimizer is re-exported lazily, and that is a deployment decision.** It imports
# `scipy.optimize.milp`, so an eager import here meant that *any* `from nfl import ...` --
# including `nfl.naming`, which the dashboard uses to find a file path -- pulled scipy into
# the process. On the hosted app that is about 40 MB of wheel installed and imported to
# serve pages that never solve anything. PEP 562 keeps the public names working while the
# solver stays unimported until something actually asks for it.
_LAZY = {"ROSTER": "optimizer", "ROSTER_SIZE": "optimizer",
         "eligible_positions": "optimizer"}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value          # cached: the lazy path runs once
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))

# **`optimize` is deliberately NOT re-exported here.** `nfl/optimize.py` is the CLI module,
# the twin of `dfs.optimize`, and a re-exported function of the same name shadows it: after
# `from nfl import optimize` you hold the solver function and every attribute lookup on the
# module fails with "'function' object has no attribute ...". `python -m nfl.optimize` still
# worked, because runpy loads the file directly -- so the collision only showed up on import.
# The solver is `nfl.optimizer.optimize`; the pipeline is `nfl.optimize`.

__all__ = [
    "DK_CLASSIC_SLOTS", "DK_SALARY_CAP", "offense_points", "dst_points",
    "ROSTER", "ROSTER_SIZE", "eligible_positions",
]
