"""Strikeouts, from three angles that the workbook keeps in three different tables.

A starter's K% only means something against the bats he actually draws. The pieces are all
in the cached payload already — they have just never been put on the same scale:

* **vs his arsenal** — `*_batter_arsenal` carries each hitter's K% against *this* pitch mix,
  his K% against this hand, and the `K Edge` between them.
* **vs the platoon side** — `*_pitcher_hand_splits` carries his own K% against left- and
  right-handed bats, which is only useful next to how many of each he will face.
* **over recent starts** — the game log carries strikeouts per start, which is the only one
  of the three that says whether he is currently missing bats or merely used to.

**Every K% here is normalised to a percentage.** The payload is inconsistent about it: the
starter-info dict quotes 0.189 while every frame quotes 18.9, and a chart that mixes the two
draws one series flat against the axis without erroring.
"""

import numpy as np
import pandas as pd

from dashboards import data

#: League-average strikeout rate, for the reference line. A K% panel with no anchor makes
#: every pitcher look average, since the interesting range is only about eight points wide.
LEAGUE_K = 22.5


def as_pct(value):
    """Normalise a K% to percentage points, whichever way the payload happened to store it.

    A rate is a fraction if it is at most 1. That test is safe here because a real K% below
    1% does not occur -- the lowest qualified starter in a season sits near 9%.
    """
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return np.nan
    return float(number) * 100 if abs(number) <= 1 else float(number)


def _tbf(innings, hits, walks):
    """Batters faced, approximated as 3*IP + H + BB.

    The boxscore game log does not carry batters faced, and this is the standard stand-in.
    It ignores hit batsmen and reached-on-error, so it runs a batter or two light on a messy
    start and the K% it produces is correspondingly a shade high. `K/9` is derived from
    innings alone and is exact -- both are shown, so a reader can tell which is which.
    """
    def number(value):
        # Scalars, not Series: `pd.to_numeric` on a scalar returns a scalar, which has no
        # `.fillna`, and the game log hands these over one row at a time.
        parsed = pd.to_numeric(value, errors="coerce")
        return 0.0 if pd.isna(parsed) else float(parsed)

    innings = pd.to_numeric(innings, errors="coerce")
    if pd.isna(innings) or innings <= 0:
        return np.nan
    # `Last Starts` stores IP as **true decimal thirds** (5.667 for five and two thirds),
    # already converted upstream by `scouting_report._ip_to_float`. Reading it as baseball
    # notation instead -- 5.2 meaning five and two thirds -- turns 5.667 into 5 + round(6.67)
    # = 22 outs against a true 17, and the K% that falls out is wrong by a third on every
    # fractional start while looking entirely plausible.
    outs = round(float(innings) * 3)
    return outs + number(hits) + number(walks)


def recent_starts(starter_info):
    """Per-start strikeout rate for one pitcher, newest first, with a season row attached."""
    logs = pd.DataFrame(starter_info.get("Last Starts") or [])
    if logs.empty or "SO" not in logs.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Date": logs.get("Date"),
        "Opponent": logs.get("Opponent"),
        "IP": pd.to_numeric(logs.get("IP"), errors="coerce").round(2),
        "SO": pd.to_numeric(logs.get("SO"), errors="coerce"),
    })
    blank = pd.Series(0, index=logs.index)
    faced = [_tbf(ip, h, bb) for ip, h, bb in
             zip(out["IP"], logs.get("H", blank), logs.get("BB", blank))]
    out["TBF"] = faced
    out["K%"] = (out["SO"] / out["TBF"] * 100).round(1)
    out["K/9"] = (out["SO"] * 9 / out["IP"]).round(2)
    out["Season K%"] = as_pct(starter_info.get("K%"))
    return out.dropna(subset=["K%"])


def platoon(payload, side, meta):
    """His K% against each side of the plate, and how many of those bats he will face.

    A split is only a matchup if the bats exist: an arm who buries left-handers matters much
    less against a card with two of them. The count comes from the *opposing* lineup, and
    switch hitters are resolved against his hand rather than dropped — they bat opposite the
    arm, so they land in whichever column he is weaker in.
    """
    splits = data.context_frame(payload, f"{side}_pitcher_hand_splits")
    if splits.empty or "Batter Side" not in splits.columns:
        return pd.DataFrame()

    opposing = "away" if side == "home" else "home"
    lineup = data.context_frame(payload, f"{opposing}_lineup")
    if lineup.empty:
        lineup = _lineup_frame(payload, opposing)
    throws = _throws(splits, payload, side)

    counts = {"L": 0, "R": 0}
    if not lineup.empty and "Bats" in lineup.columns:
        for bats in lineup["Bats"]:
            resolved = data.effective_side(bats, throws)
            if resolved in counts:
                counts[resolved] += 1

    out = splits.copy()
    out["K%"] = out["K%"].map(as_pct)
    out["Bats faced"] = out["Batter Side"].map(counts).fillna(0).astype(int)
    return out


def _throws(splits, payload, side):
    """The arm's throwing hand. `sp_profile` has no pitcher name and no hand; the hand-splits
    frame is where it lives, and failing that the projection is the authority."""
    for column in ("Throws", "throws"):
        if column in splits.columns and splits[column].notna().any():
            return str(splits[column].dropna().iloc[0]).upper()[:1]
    starter = (payload.get("advanced_context") or {}).get(f"{side}_sp_hand")
    return str(starter).upper()[:1] if starter else None


def _lineup_frame(payload, side):
    args = payload.get("report_args") or ()
    index = 0 if side == "home" else 7
    frame = args[index] if len(args) > index else None
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def lineup_k(payload, side):
    """Each opposing hitter's K% against this arsenal, against this hand, and the gap.

    `*_batter_arsenal` is keyed to the club whose prefix it carries and is already scored
    against the arm that club faces, so the frame for `side` is the *hitters* of that side —
    which is who the OTHER side's pitcher has to strike out.
    """
    frame = data.context_frame(payload, f"{side}_batter_arsenal")
    if frame.empty or "K%" not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    for column in ("K%", "K% vs Hand", "Whiff%"):
        if column in out.columns:
            out[column] = out[column].map(as_pct)
    if "K Edge" in out.columns:
        out["K Edge"] = pd.to_numeric(out["K Edge"], errors="coerce")
    keep = [c for c in ("Name", "Team", "PA", "K%", "K% vs Hand", "K Edge", "Whiff%")
            if c in out.columns]
    return out[keep].sort_values("K%", ascending=False)


def headline(starter_info, lineup_frame, hand_splits):
    """The one-line comparison: his strikeout rate against the card's strikeout rate.

    Returns `{his_k, lineup_k, edge, ...}` with everything already in percentage points.
    `edge` is positive when he misses more bats than this lineup usually gives up, which is
    the direction that makes a strikeout play.
    """
    his = as_pct(starter_info.get("K%"))
    lineup = np.nan
    if isinstance(lineup_frame, pd.DataFrame) and not lineup_frame.empty:
        if "Split" in lineup_frame.columns and "K%" in lineup_frame.columns:
            overall = lineup_frame[lineup_frame["Split"].astype(str).str.strip() == "Overall"]
            source = overall if not overall.empty else lineup_frame
            lineup = as_pct(source["K%"].iloc[0])
    split_spread = np.nan
    if isinstance(hand_splits, pd.DataFrame) and not hand_splits.empty and "K%" in hand_splits.columns:
        values = hand_splits["K%"].dropna()
        if len(values) >= 2:
            split_spread = float(values.max() - values.min())
    return {
        "his_k": his,
        "lineup_k": lineup,
        "edge": (his - lineup) if pd.notna(his) and pd.notna(lineup) else np.nan,
        "league": LEAGUE_K,
        "split_spread": split_spread,
    }
