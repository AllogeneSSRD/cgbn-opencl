#!/usr/bin/env python3
"""
import_ecm.py  –  Import a PrimeNet ECM Progress HTML report into SQLite.

Usage:
    python import_ecm.py <report.html> [--db <path>] [--force]

Environment:
    ECM_DB  –  override default DB path (same as --db)

Default DB: tools/ecm_report/ecm_progress.db  (relative to this script)
"""

import argparse
import os
import re
import sqlite3
import sys
from html.parser import HTMLParser
from pathlib import Path

DEFAULT_DB = Path(__file__).parent / "ecm_progress.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS exponents (
    exponent          INTEGER PRIMARY KEY,
    factored          INTEGER NOT NULL DEFAULT 0,
    factored_date     TEXT,                       -- YYYYMMDD, only when no→known transition observed
    last_report_date  TEXT NOT NULL,              -- YYYYMMDD from Current time
    last_current_time TEXT NOT NULL,              -- full "2026-07-29 17:20 UTC"
    t_level           INTEGER,                    -- latest progress_history.t_level
    curves            INTEGER                     -- latest progress_history.curves (NULL if all Done)
);

CREATE TABLE IF NOT EXISTS progress_history (
    exponent     INTEGER NOT NULL,
    report_date  TEXT    NOT NULL,               -- YYYYMMDD
    t_level      INTEGER NOT NULL,
    curves       INTEGER,                        -- NULL when all Done
    PRIMARY KEY (exponent, report_date),
    FOREIGN KEY (exponent) REFERENCES exponents(exponent)
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    _migrate_exponents_latest_progress(conn)
    conn.commit()


def _migrate_exponents_latest_progress(conn: sqlite3.Connection) -> None:
    """Add t_level/curves to existing DBs and backfill from progress_history."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(exponents)")}
    if "t_level" not in cols:
        conn.execute("ALTER TABLE exponents ADD COLUMN t_level INTEGER")
    if "curves" not in cols:
        conn.execute("ALTER TABLE exponents ADD COLUMN curves INTEGER")

    conn.execute(
        """
        UPDATE exponents
        SET
            t_level = (
                SELECT ph.t_level FROM progress_history ph
                WHERE ph.exponent = exponents.exponent
                ORDER BY ph.report_date DESC LIMIT 1
            ),
            curves = (
                SELECT ph.curves FROM progress_history ph
                WHERE ph.exponent = exponents.exponent
                ORDER BY ph.report_date DESC LIMIT 1
            ),
            last_report_date = (
                SELECT ph.report_date FROM progress_history ph
                WHERE ph.exponent = exponents.exponent
                ORDER BY ph.report_date DESC LIMIT 1
            )
        WHERE EXISTS (
            SELECT 1 FROM progress_history ph WHERE ph.exponent = exponents.exponent
        )
        """
    )


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

class ECMHTMLParser(HTMLParser):
    """
    Extracts from the PrimeNet ECM Progress page:
      - current_time  : str  e.g. "2026-07-29 17:20 UTC"
      - report_date   : str  YYYYMMDD derived from current_time
      - ranges        : dict  { 'no': (lo, hi), 'known': (lo, hi) }
      - sections      : dict  { 'no': [(exp, t_level, curves), ...],
                                'known': [(exp, t_level, curves), ...] }
    """

    _PRE_IDS = {
        "mersenne_numbers_with_no_known_factors": "no",
        "mersenne_numbers_with_known_factors":    "known",
    }
    _RANGE_NAMES = {
        "ecmnof_lo": ("no",    "lo"),
        "ecmnof_hi": ("no",    "hi"),
        "ecm_lo":    ("known", "lo"),
        "ecm_hi":    ("known", "hi"),
    }

    def __init__(self):
        super().__init__()
        self.current_time: str | None = None
        self.report_date:  str | None = None
        self.ranges  = {"no": {}, "known": {}}
        self.sections = {"no": [], "known": []}

        self._in_pre: str | None = None   # 'no' | 'known' | None
        self._pre_buf: list[str] = []
        self._capture_time = False

    # -- tag handlers --------------------------------------------------------

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)

        if tag == "pre":
            pid = attrs_d.get("id", "")
            if pid in self._PRE_IDS:
                self._in_pre = self._PRE_IDS[pid]
                self._pre_buf = []

        if tag == "input":
            name = attrs_d.get("name", "")
            val  = attrs_d.get("value", "")
            if name in self._RANGE_NAMES:
                section, key = self._RANGE_NAMES[name]
                try:
                    self.ranges[section][key] = int(val)
                except ValueError:
                    pass

        if tag == "div":
            style = attrs_d.get("style", "")
            if "float: right" in style and "font-size: 9pt" in style:
                self._capture_time = True

    def handle_endtag(self, tag):
        if tag == "pre" and self._in_pre is not None:
            self._finish_pre(self._in_pre, self._pre_buf)
            self._in_pre = None
            self._pre_buf = []
        if tag == "div":
            self._capture_time = False

    def handle_data(self, data):
        if self._in_pre is not None:
            self._pre_buf.append(data)
            return
        if self._capture_time and "Current time:" in data:
            m = re.search(r"Current time:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)", data)
            if m:
                self.current_time = m.group(1)
                date_part = self.current_time[:10].replace("-", "")
                self.report_date = date_part
            self._capture_time = False

    # -- pre block processor -------------------------------------------------

    def _finish_pre(self, section: str, buf: list[str]) -> None:
        text = "".join(buf)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return

        # first line: "Digits in factor  25  30  35 ..."
        header = lines[0]
        if not header.startswith("Digits in factor"):
            raise ValueError(f"Unexpected header in {section!r} section: {header!r}")
        digits = [int(x) for x in header.split()[3:]]  # skip "Digits", "in", "factor"

        for line in lines[1:]:
            parts = line.split("\t")
            # skip metadata rows
            if parts[0] in ("Bound #1", "Curves to test"):
                continue
            try:
                exp = int(parts[0])
            except ValueError:
                continue

            values = parts[1:]  # may be shorter than digits if trailing columns absent
            t_level, curves = self._compute_t_curves(digits, values)
            self.sections[section].append((exp, t_level, curves))

    @staticmethod
    def _compute_t_curves(digits: list[int], values: list[str]) -> tuple[int, int | None]:
        """
        Return (t_level, curves).
        t_level = digit of first non-Done column; if all Done, t_level = highest digit.
        curves  = integer at that column; NULL (None) if all Done and no trailing number.
        """
        for i, val in enumerate(values):
            v = val.strip()
            if v.lower() != "done":
                try:
                    curves = int(v)
                except ValueError:
                    curves = None
                return digits[i], curves

        # all supplied values are "Done"
        t_level = digits[len(values) - 1] if values else digits[-1]
        return t_level, None


def parse_html(path: Path) -> ECMHTMLParser:
    parser = ECMHTMLParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_parsed(p: ECMHTMLParser) -> None:
    if not p.current_time:
        sys.exit("ERROR: Could not find 'Current time:' in the HTML. Aborting.")

    # check for exponents appearing in both sections
    no_set    = {exp for exp, *_ in p.sections["no"]}
    known_set = {exp for exp, *_ in p.sections["known"]}
    overlap   = no_set & known_set
    if overlap:
        sample = sorted(overlap)[:5]
        sys.exit(
            f"ERROR: {len(overlap)} exponent(s) appear in BOTH sections "
            f"(e.g. {sample}). This indicates a malformed report. Aborting."
        )


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def check_date_conflicts(
    conn: sqlite3.Connection,
    report_date: str,
    force: bool,
) -> None:
    """
    If any progress_history rows already exist for report_date and --force
    is not set, warn the user and exit (or ask on TTY).
    """
    rows = conn.execute(
        "SELECT COUNT(*) FROM progress_history WHERE report_date = ?",
        (report_date,),
    ).fetchone()[0]

    if rows == 0:
        return

    msg = (
        f"WARNING: {rows} row(s) for date {report_date} already exist in "
        f"progress_history.\n"
        f"  - Same-day re-import overwrites that day's snapshot (use --force).\n"
        f"  - To append a new history day, change Current time in the HTML "
        f"to a new date."
    )

    if force:
        print(msg + "\n--force set, overwriting.", file=sys.stderr)
        return

    # isatty() can be true in IDE/agent shells that still cannot provide input.
    if sys.stdin.isatty():
        print(msg, file=sys.stderr)
        try:
            answer = input("Overwrite existing data? [y/N] ").strip().lower()
        except EOFError:
            sys.exit(
                "Non-interactive session: re-run with --force to overwrite."
            )
        if answer != "y":
            sys.exit("Aborted by user.")
        return

    sys.exit(
        msg + "\nNon-interactive session: re-run with --force to overwrite."
    )


# ---------------------------------------------------------------------------
# Progress history (sparse: only store when t_level/curves change)
# ---------------------------------------------------------------------------

def _same_progress(
    t0: int, c0: int | None, t1: int, c1: int | None
) -> bool:
    return t0 == t1 and c0 == c1


def compact_progress_history(conn: sqlite3.Connection) -> int:
    """
    Remove consecutive duplicate snapshots per exponent (same t_level/curves).
    Returns number of deleted rows. Safe to run repeatedly.
    """
    rows = conn.execute(
        """
        SELECT exponent, report_date, t_level, curves
        FROM progress_history
        ORDER BY exponent, report_date
        """
    ).fetchall()
    to_delete: list[tuple[int, str]] = []
    prev_exp = prev_t = prev_c = None
    for exp, date, t, c in rows:
        if prev_exp == exp and _same_progress(prev_t, prev_c, t, c):
            to_delete.append((exp, date))
        else:
            prev_exp, prev_t, prev_c = exp, t, c
    if to_delete:
        conn.executemany(
            "DELETE FROM progress_history WHERE exponent = ? AND report_date = ?",
            to_delete,
        )
        conn.commit()
    return len(to_delete)


def _load_prev_progress(
    conn: sqlite3.Connection, report_date: str
) -> dict[int, tuple[int, int | None]]:
    """Latest (t_level, curves) for each exponent with report_date < given date."""
    rows = conn.execute(
        """
        SELECT ph.exponent, ph.t_level, ph.curves
        FROM progress_history ph
        INNER JOIN (
            SELECT exponent, MAX(report_date) AS md
            FROM progress_history
            WHERE report_date < ?
            GROUP BY exponent
        ) latest
          ON ph.exponent = latest.exponent AND ph.report_date = latest.md
        """,
        (report_date,),
    ).fetchall()
    return {int(e): (int(t), c) for e, t, c in rows}


def _load_same_day_exponents(conn: sqlite3.Connection, report_date: str) -> set[int]:
    rows = conn.execute(
        "SELECT exponent FROM progress_history WHERE report_date = ?",
        (report_date,),
    ).fetchall()
    return {int(r[0]) for r in rows}

def import_report(conn: sqlite3.Connection, p: ECMHTMLParser, report_date: str) -> None:
    cursor = conn.cursor()

    all_rows: list[tuple[int, int, int | None, int]] = []  # (exp, t, curves, factored)
    for exp, t, curves in p.sections["no"]:
        all_rows.append((exp, t, curves, 0))
    for exp, t, curves in p.sections["known"]:
        all_rows.append((exp, t, curves, 1))

    # determine intersection range for migration detection
    no_lo    = p.ranges["no"].get("lo")
    no_hi    = p.ranges["no"].get("hi")
    known_lo = p.ranges["known"].get("lo")
    known_hi = p.ranges["known"].get("hi")

    def in_intersection(exp: int) -> bool:
        if None in (no_lo, no_hi, known_lo, known_hi):
            return False
        lo = max(no_lo, known_lo)
        hi = min(no_hi, known_hi)
        return lo <= exp <= hi

    inserted = skipped = removed = updated_factored = 0
    prev_map = _load_prev_progress(conn, report_date)
    same_day = _load_same_day_exponents(conn, report_date)

    for exp, t_level, curves, factored_now in all_rows:
        # --- exponents upsert ---
        existing = cursor.execute(
            "SELECT factored, factored_date, last_report_date FROM exponents WHERE exponent = ?",
            (exp,),
        ).fetchone()

        if existing is None:
            cursor.execute(
                """INSERT INTO exponents
                   (exponent, factored, factored_date, last_report_date, last_current_time,
                    t_level, curves)
                   VALUES (?, ?, NULL, ?, ?, ?, ?)""",
                (exp, factored_now, report_date, p.current_time, t_level, curves),
            )
        else:
            prev_factored, prev_fdate, prev_report_date = existing
            new_factored = prev_factored  # only goes 0→1
            new_fdate    = prev_fdate

            if factored_now == 1 and prev_factored == 0 and in_intersection(exp):
                # genuine no→known migration observed
                new_factored = 1
                new_fdate    = report_date
                updated_factored += 1
                print(f"  [MIGRATION] Exponent {exp} moved to known factors on {report_date}")
            elif factored_now == 1 and prev_factored == 0:
                # seen in known but outside intersection – just mark factored, no date
                new_factored = 1

            if factored_now == 0 and prev_factored == 1:
                print(
                    f"  [WARNING] Exponent {exp} was factored but now appears in "
                    f"no_known_factors section. Keeping factored status.",
                    file=sys.stderr,
                )

            # Main-table t_level/curves track the latest report date only.
            is_latest = report_date >= prev_report_date
            if is_latest:
                cursor.execute(
                    """UPDATE exponents
                       SET factored = ?, factored_date = ?,
                           last_report_date = ?, last_current_time = ?,
                           t_level = ?, curves = ?
                       WHERE exponent = ?""",
                    (
                        new_factored, new_fdate,
                        report_date, p.current_time,
                        t_level, curves, exp,
                    ),
                )
            else:
                cursor.execute(
                    """UPDATE exponents
                       SET factored = ?, factored_date = ?
                       WHERE exponent = ?""",
                    (new_factored, new_fdate, exp),
                )

        # --- progress_history: sparse (only when changed vs prior snapshot) ---
        prev = prev_map.get(exp)
        unchanged = prev is not None and _same_progress(
            prev[0], prev[1], t_level, curves
        )
        if unchanged:
            if exp in same_day:
                cursor.execute(
                    "DELETE FROM progress_history WHERE exponent = ? AND report_date = ?",
                    (exp, report_date),
                )
                removed += 1
            skipped += 1
        else:
            cursor.execute(
                """INSERT INTO progress_history (exponent, report_date, t_level, curves)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(exponent, report_date) DO UPDATE SET
                       t_level = excluded.t_level,
                       curves  = excluded.curves""",
                (exp, report_date, t_level, curves),
            )
            inserted += 1

    conn.commit()
    print(
        f"Done. history written={inserted}, skipped(unchanged)={skipped}, "
        f"removed(same-day redundant)={removed}; "
        f"{updated_factored} exponent(s) newly marked as factored."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a PrimeNet ECM Progress HTML report into SQLite."
    )
    parser.add_argument("html", type=Path, help="Path to the HTML report file")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("ECM_DB", DEFAULT_DB)),
        help="SQLite database path (default: ecm_progress.db next to this script)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing data for the same report date without prompting",
    )
    args = parser.parse_args()

    if not args.html.exists():
        sys.exit(f"ERROR: File not found: {args.html}")

    print(f"Parsing {args.html} …")
    p = parse_html(args.html)
    validate_parsed(p)

    report_date = p.report_date
    print(f"  Report date : {report_date}  ({p.current_time})")
    print(f"  no_known    : {len(p.sections['no'])} exponents  range {p.ranges.get('no')}")
    print(f"  known       : {len(p.sections['known'])} exponents  range {p.ranges.get('known')}")

    all_exponents = [exp for exp, *_ in p.sections["no"] + p.sections["known"]]
    if not all_exponents:
        sys.exit("ERROR: No exponent rows found in the HTML report.")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON")
    # Large imports (100k+ rows) benefit from a bigger cache / WAL.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    init_db(conn)

    deleted = compact_progress_history(conn)
    if deleted:
        print(f"Compacted progress_history: removed {deleted} redundant row(s).")

    check_date_conflicts(conn, report_date, args.force)

    print(f"Importing into {args.db} …")
    import_report(conn, p, report_date)
    conn.close()


if __name__ == "__main__":
    main()
