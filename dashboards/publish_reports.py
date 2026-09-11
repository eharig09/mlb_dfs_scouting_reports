"""Publish generated workbooks so the hosted app can read them.

    python -m dashboards.publish_reports --date 2026-08-26
    python -m dashboards.publish_reports --latest 2          # last 2 dates per sport
    python -m dashboards.publish_reports                     # what is published now

The workbooks themselves cannot be committed -- 406 of them at 146 MB -- and most of that
weight is figures and formatting rather than data. This writes each sheet as parquet into
`reports_published/`, which is small enough to commit and is what the Reports page reads
when the workbook is not on disk.

**Publish deliberately, not everything.** A season of games is a lot of parquet to carry
around for reports nobody will open again, and the point of the page is the last few days.
`--latest` is the setting to reach for.
"""

import argparse
import os
import sys

from dashboards import workbooks


def select(frame, date=None, latest=None, sport=None):
    """Which workbooks to publish, from the full listing."""
    scoped = frame
    if sport:
        scoped = scoped[scoped["Sport"].str.upper() == sport.upper()]
    if date:
        scoped = scoped[scoped["Date"] == date]
    elif latest:
        keep = []
        for _, group in scoped.groupby("Sport"):
            dates = sorted(group["Date"].unique(), reverse=True)[:int(latest)]
            keep.append(group[group["Date"].isin(dates)])
        scoped = keep[0].iloc[0:0] if not keep else __import__("pandas").concat(keep)
    return scoped


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Publish report workbooks as parquet for the hosted app.")
    parser.add_argument("--date", default=None, help="Publish one date, e.g. 2026-08-26.")
    parser.add_argument("--latest", type=int, default=None,
                        help="Publish the most recent N dates per sport.")
    parser.add_argument("--sport", default=None, choices=["MLB", "NFL", "mlb", "nfl"])
    parser.add_argument("--root", default=workbooks.PUBLISHED_DIR)
    args = parser.parse_args(argv)

    found = workbooks.list_workbooks()
    if found.empty:
        print("No generated workbooks on disk. Run the report pipeline first.")
        return 0

    if not (args.date or args.latest or args.sport):
        current = workbooks.published_reports()
        if not current:
            print(f"Nothing published yet. {len(found)} workbooks available on disk.")
            print("Publish the recent ones with:  --latest 2")
            return 0
        size = sum(os.path.getsize(os.path.join(args.root, p, f))
                   for p in os.listdir(args.root)
                   if os.path.isdir(os.path.join(args.root, p))
                   for f in os.listdir(os.path.join(args.root, p)))
        print(f"{len(current)} reports published, {size / 1_000_000:.1f} MB.")
        for key, entry in sorted(current.items())[:12]:
            print(f"  {key}  {len(entry['sheets'])} sheets")
        if len(current) > 12:
            print(f"  ... and {len(current) - 12} more")
        return 0

    chosen = select(found, date=args.date, latest=args.latest, sport=args.sport)
    if chosen.empty:
        print("Nothing matched that selection.")
        print(f"Dates on disk: {', '.join(sorted(found['Date'].unique(), reverse=True)[:8])}")
        return 1

    print(f"Publishing {len(chosen)} workbook(s) into {args.root}")
    manifest = workbooks.publish(list(chosen["Path"]), root=args.root)
    print(f"\n{len(manifest['reports'])} reports, {manifest['bytes'] / 1_000_000:.2f} MB")
    print(f"Commit `{args.root}/` and push — the hosted Reports page reads it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
