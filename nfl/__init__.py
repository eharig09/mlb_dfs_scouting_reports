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
from .optimizer import ROSTER, ROSTER_SIZE, eligible_positions, optimize

__all__ = [
    "DK_CLASSIC_SLOTS", "DK_SALARY_CAP", "offense_points", "dst_points",
    "ROSTER", "ROSTER_SIZE", "eligible_positions", "optimize",
]
