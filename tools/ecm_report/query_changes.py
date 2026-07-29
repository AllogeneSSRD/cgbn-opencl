#!/usr/bin/env python3
"""
query_changes.py  –  Diff ECM progress between two dates (sparse history / as-of).

progress_history only stores rows when t_level/curves change. State on a date D
is the latest row with report_date <= D.

Usage:
    python query_changes.py [--db <path>] [--from YYYYMMDD] [--to YYYYMMDD]

Date defaults:
    neither  →  second-latest history date .. latest
    only --from  →  --from .. latest
    only --to    →  earliest .. --to

Environment:
    ECM_DB  –  override default DB path (same as --db)
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path(__file__).parent / "ecm_progress.db"
DATE_RE = re.compile(r"^\d{8}$")


def list_dates(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT report_date FROM progress_history ORDER BY report_date"
        )
    ]


def resolve_dates(
    dates: list[str], start: str | None, end: str | None
) -> tuple[str, str]:
    if not dates:
        sys.exit("ERROR: progress_history is empty.")

    if start is None and end is None:
        if len(dates) < 2:
            sys.exit(
                f"ERROR: Need at least 2 history dates for default diff; found {dates}."
            )
        return dates[-2], dates[-1]

    if start is None:
        start = dates[0]
    if end is None:
        end = dates[-1]

    for label, value in (("--from", start), ("--to", end)):
        if not DATE_RE.match(value):
            sys.exit(f"ERROR: {label} must be YYYYMMDD, got {value!r}")

    if start > end:
        sys.exit(f"ERROR: --from {start} is after --to {end}.")
    if start == end:
        sys.exit(f"ERROR: --from and --to are the same date ({start}).")

    return start, end


def fmt_curves(v) -> str:
    return "NULL" if v is None else str(v)


AS_OF_CTE = """
as_of_start AS (
    SELECT ph.exponent, ph.t_level, ph.curves
    FROM progress_history ph
    INNER JOIN (
        SELECT exponent, MAX(report_date) AS md
        FROM progress_history
        WHERE report_date <= ?
        GROUP BY exponent
    ) x ON ph.exponent = x.exponent AND ph.report_date = x.md
),
as_of_end AS (
    SELECT ph.exponent, ph.t_level, ph.curves
    FROM progress_history ph
    INNER JOIN (
        SELECT exponent, MAX(report_date) AS md
        FROM progress_history
        WHERE report_date <= ?
        GROUP BY exponent
    ) x ON ph.exponent = x.exponent AND ph.report_date = x.md
)
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Show ECM progress changes between two dates "
            "(as-of sparse progress_history)."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("ECM_DB", DEFAULT_DB)),
        help="SQLite database path",
    )
    parser.add_argument(
        "--from",
        dest="start",
        metavar="YYYYMMDD",
        help="Start date (default: second-latest history date)",
    )
    parser.add_argument(
        "--to",
        dest="end",
        metavar="YYYYMMDD",
        help="End date (default: latest history date)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also list exponents unchanged between as-of start and as-of end",
    )
    args = parser.parse_args()

    if not args.db.exists():
        sys.exit(f"ERROR: Database not found: {args.db}")

    conn = sqlite3.connect(args.db)
    dates = list_dates(conn)
    start, end = resolve_dates(dates, args.start, args.end)

    print(f"Comparing as-of {start} -> as-of {end}  (sparse history)")
    print(f"History event dates: {', '.join(dates)}")
    print()

    changed = conn.execute(
        f"""
        WITH {AS_OF_CTE}
        SELECT
            e.exponent,
            IFNULL(ex.factored, 0),
            s.t_level, s.curves,
            e.t_level, e.curves
        FROM as_of_end e
        INNER JOIN as_of_start s ON s.exponent = e.exponent
        LEFT JOIN exponents ex ON ex.exponent = e.exponent
        WHERE e.t_level != s.t_level
           OR IFNULL(e.curves, -1) != IFNULL(s.curves, -1)
        ORDER BY e.exponent
        """,
        (start, end),
    ).fetchall()

    first_seen = conn.execute(
        f"""
        WITH {AS_OF_CTE}
        SELECT
            e.exponent,
            IFNULL(ex.factored, 0),
            e.t_level, e.curves
        FROM as_of_end e
        LEFT JOIN as_of_start s ON s.exponent = e.exponent
        LEFT JOIN exponents ex ON ex.exponent = e.exponent
        WHERE s.exponent IS NULL
        ORDER BY e.exponent
        """,
        (start, end),
    ).fetchall()

    events = conn.execute(
        """
        SELECT ph.exponent, ph.report_date, ph.t_level, ph.curves, IFNULL(ex.factored, 0)
        FROM progress_history ph
        LEFT JOIN exponents ex ON ex.exponent = ph.exponent
        WHERE ph.report_date > ? AND ph.report_date <= ?
        ORDER BY ph.report_date, ph.exponent
        """,
        (start, end),
    ).fetchall()

    unchanged = []
    if args.all:
        unchanged = conn.execute(
            f"""
            WITH {AS_OF_CTE}
            SELECT
                e.exponent,
                IFNULL(ex.factored, 0),
                e.t_level, e.curves
            FROM as_of_end e
            INNER JOIN as_of_start s ON s.exponent = e.exponent
            LEFT JOIN exponents ex ON ex.exponent = e.exponent
            WHERE e.t_level = s.t_level
              AND IFNULL(e.curves, -1) = IFNULL(s.curves, -1)
            ORDER BY e.exponent
            """,
            (start, end),
        ).fetchall()

    newly_factored = conn.execute(
        """
        SELECT exponent, factored_date, t_level, curves
        FROM exponents
        WHERE factored_date IS NOT NULL
          AND factored_date > ?
          AND factored_date <= ?
        ORDER BY factored_date, exponent
        """,
        (start, end),
    ).fetchall()

    conn.close()

    print("Summary")
    print(f"  changed (as-of differs) : {len(changed)}")
    print(f"  first seen by {end}     : {len(first_seen)}")
    print(f"  history events in ({start}, {end}] : {len(events)}")
    print(f"  newly factored          : {len(newly_factored)}")
    if args.all:
        print(f"  unchanged (as-of)       : {len(unchanged)}")
    print()

    print(f"Changed (as-of {start} -> {end})")
    print("exponent\tfactored\tt_level_from\tcurves_from\tt_level_to\tcurves_to\tdelta_curves")
    for exp, factored, t0, c0, t1, c1 in changed:
        if c0 is not None and c1 is not None and t0 == t1:
            delta = str(c1 - c0)
        else:
            delta = "N/A"
        print(
            f"{exp}\t{factored}\t{t0}\t{fmt_curves(c0)}\t{t1}\t{fmt_curves(c1)}\t{delta}"
        )
    if not changed:
        print("(none)")
    print()

    print(f"First seen (no as-of at {start}, present by {end})")
    print("exponent\tfactored\tt_level\tcurves")
    for exp, factored, t, c in first_seen:
        print(f"{exp}\t{factored}\t{t}\t{fmt_curves(c)}")
    if not first_seen:
        print("(none)")
    print()

    print(f"History events ({start} < report_date <= {end})")
    print("exponent\treport_date\tfactored\tt_level\tcurves")
    for exp, rdate, t, c, factored in events:
        print(f"{exp}\t{rdate}\t{factored}\t{t}\t{fmt_curves(c)}")
    if not events:
        print("(none)")
    print()

    if args.all:
        print(f"Unchanged (as-of {start} == as-of {end})")
        print("exponent\tfactored\tt_level\tcurves")
        for exp, factored, t, c in unchanged:
            print(f"{exp}\t{factored}\t{t}\t{fmt_curves(c)}")
        if not unchanged:
            print("(none)")
        print()

    print(f"Newly factored ({start} < factored_date <= {end})")
    print("exponent\tfactored_date\tt_level\tcurves")
    for exp, fdate, t, c in newly_factored:
        print(f"{exp}\t{fdate}\t{t}\t{fmt_curves(c)}")
    if not newly_factored:
        print("(none)")


if __name__ == "__main__":
    main()
