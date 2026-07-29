#!/usr/bin/env python3
"""
query_min_exponents.py  –  For each t-level, show the smallest exponent
with known factors and with no known factors (from the exponents main table).

Usage:
    python query_min_exponents.py [--db <path>]

Environment:
    ECM_DB  –  override default DB path (same as --db)
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path(__file__).parent / "ecm_progress.db"

SQL = """
SELECT
    t_level AS level,
    MIN(CASE WHEN factored = 1 THEN exponent END) AS min_known,
    MIN(CASE WHEN factored = 0 THEN exponent END) AS min_no_known
FROM exponents
WHERE t_level IS NOT NULL
GROUP BY t_level
ORDER BY t_level DESC
"""


def fmt(value) -> str:
    return "N/A" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List minimum exponents per t-level (known / no known factors)."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("ECM_DB", DEFAULT_DB)),
        help="SQLite database path (default: ecm_progress.db next to this script)",
    )
    args = parser.parse_args()

    if not args.db.exists():
        sys.exit(f"ERROR: Database not found: {args.db}")

    conn = sqlite3.connect(args.db)
    rows = conn.execute(SQL).fetchall()
    conn.close()

    if not rows:
        sys.exit("No rows with t_level in exponents table.")

    col1, col2, col3 = "Level", "Exponents with known factors", "Exponents with no known factors"
    # tab-separated to match the requested layout
    print(f"{col1}\t{col2}\t{col3}")
    for level, min_known, min_no_known in rows:
        print(f"{level}\t{fmt(min_known)}\t{fmt(min_no_known)}")


if __name__ == "__main__":
    main()
