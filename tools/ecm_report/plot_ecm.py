#!/usr/bin/env python3
"""
plot_ecm.py  –  Dense bar chart of ECM progress for Mersenne exponents.

Usage:
    python plot_ecm.py [--lo 1] [--hi 20000] [--db PATH] [--date YYYYMMDD]
                       [--color factored|level]
                       [--plots separate|merged|overlay|all|known|no_known]
                       [--xtick-gap 1000|auto] [--compress N] [--compress-agg avg|max]
                       [--ymin Y] [--ymax Y] [--fig-height INCHES] [--dpi N] [--vgrid]
                       [--out PATH] [--csv PATH]

Environment:
    ECM_DB  –  override default DB path
"""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DB = Path(__file__).parent / "ecm_progress.db"

# PrimeNet "Curves to test" for Digits in factor 25..75
CURVES_TO_TEST: dict[int, int] = {
    25: 280,
    30: 640,
    35: 1566,
    40: 4588,
    45: 9201,
    50: 16550,
    55: 40830,
    60: 105362,
    65: 194238,
    70: 350860,
    75: 567615,
}
LEVELS = sorted(CURVES_TO_TEST)
LEVEL_GAP = 5.0

# Default figure geometry (also overridable via CLI --fig-height / --dpi / --ymin)
DEFAULT_DPI = 100
DEFAULT_FIG_HEIGHT = 8.0  # inches; pixel height ≈ fig_height * dpi

COLOR_NO_KNOWN = "#EC3838"
COLOR_KNOWN = "#00cc00"

Row = tuple[int, int, int | None, int]  # exponent, t_level, curves, factored


def progress_y(t_level: int, curves: int | None) -> float:
    """
    t_level is the *current* digit level (not yet completed).

    y = (t_level - 5) + 5 * min(1, curves / curves_to_test[t_level])
    curves=NULL (level fully done) => fraction 1 => y = t_level

    Example: t=45, curves=5000, limit=9201 => y ≈ 42.72
    """
    if t_level not in CURVES_TO_TEST:
        raise ValueError(f"Unknown t_level {t_level}; update CURVES_TO_TEST")
    base = t_level - LEVEL_GAP
    if curves is None:
        return base + LEVEL_GAP  # == t_level
    frac = min(1.0, curves / CURVES_TO_TEST[t_level])
    return base + LEVEL_GAP * frac


def load_rows(
    conn: sqlite3.Connection,
    lo: int,
    hi: int,
    report_date: str | None,
) -> list[Row]:
    """Return sorted (exponent, t_level, curves, factored) in [lo, hi].

    Default: exponents main table (latest).
    With report_date: as-of sparse progress_history (latest row with date <= D).
    """
    if report_date is None:
        rows = conn.execute(
            """
            SELECT exponent, t_level, curves, factored
            FROM exponents
            WHERE exponent BETWEEN ? AND ?
              AND t_level IS NOT NULL
            ORDER BY exponent
            """,
            (lo, hi),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT p.exponent, p.t_level, p.curves, IFNULL(e.factored, 0)
            FROM progress_history p
            INNER JOIN (
                SELECT exponent, MAX(report_date) AS md
                FROM progress_history
                WHERE report_date <= ?
                  AND exponent BETWEEN ? AND ?
                GROUP BY exponent
            ) latest
              ON p.exponent = latest.exponent AND p.report_date = latest.md
            LEFT JOIN exponents e ON e.exponent = p.exponent
            ORDER BY p.exponent
            """,
            (report_date, lo, hi),
        ).fetchall()
    return [(int(e), int(t), c, int(f)) for e, t, c, f in rows]


def summary_rows(data: list[Row]) -> list[dict]:
    by_level: dict[int, list[tuple[int, int | None, int, float]]] = {}
    for exp, t, c, factored in data:
        y = progress_y(t, c)
        # progress within current level on [0, 5]
        by_level.setdefault(t, []).append((exp, c, factored, y - (t - LEVEL_GAP)))

    out = []
    for level in sorted(by_level, reverse=True):
        items = by_level[level]
        exps = [i[0] for i in items]
        known = sum(1 for i in items if i[2] == 1)
        no_known = len(items) - known
        mean_prog = sum(i[3] for i in items) / len(items)
        out.append(
            {
                "level": level,
                "count": len(items),
                "min_exp": min(exps),
                "max_exp": max(exps),
                "known": known,
                "no_known": no_known,
                "mean_progress": round(mean_prog, 3),
            }
        )
    return out


def print_summary(rows: list[dict]) -> None:
    headers = [
        "level",
        "count",
        "min_exp",
        "max_exp",
        "known",
        "no_known",
        "mean_progress",
    ]
    print("\t".join(headers))
    for r in rows:
        print("\t".join(str(r[h]) for h in headers))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def level_colormap():
    cmap = plt.get_cmap("viridis")
    n = len(LEVELS)
    return {lv: cmap(i / max(n - 1, 1)) for i, lv in enumerate(LEVELS)}


def compress_rows(
    data: list[Row], compress: int, agg: str = "avg"
) -> list[dict]:
    """Merge every `compress` consecutive exponents; bar height via avg or max y."""
    if compress < 1:
        raise ValueError("compress must be >= 1")
    if agg not in ("avg", "max"):
        raise ValueError("agg must be 'avg' or 'max'")
    bins: list[dict] = []
    for i in range(0, len(data), compress):
        chunk = data[i : i + compress]
        ys = [progress_y(t, c) for _, t, c, _ in chunk]
        factored = [f for *_, f in chunk]
        t_levels = [t for _, t, _, _ in chunk]
        exps = [e for e, *_ in chunk]
        if agg == "max":
            j = max(range(len(chunk)), key=lambda k: ys[k])
            bins.append(
                {
                    "exp_rep": exps[0],
                    "y": ys[j],
                    "t_level_mean": float(t_levels[j]),
                    "factored_frac": float(factored[j]),
                }
            )
        else:
            bins.append(
                {
                    "exp_rep": exps[0],
                    "y": sum(ys) / len(ys),
                    "t_level_mean": sum(t_levels) / len(t_levels),
                    "factored_frac": sum(factored) / len(factored),
                }
            )
    return bins


def nearest_index(sorted_exps: list[int], e: int) -> int:
    """Index of exponent in sorted_exps closest to e."""
    if not sorted_exps:
        raise ValueError("sorted_exps is empty")
    i = bisect.bisect_left(sorted_exps, e)
    if i <= 0:
        return 0
    if i >= len(sorted_exps):
        return len(sorted_exps) - 1
    before, after = i - 1, i
    if abs(sorted_exps[after] - e) < abs(sorted_exps[before] - e):
        return after
    return before


def nearest_level(t_mean: float) -> int:
    return min(LEVELS, key=lambda lv: abs(lv - t_mean))


def apply_xticks(ax, exp_reps: list[int], xtick_gap: str | int) -> list[float]:
    """
    xtick_gap:
      int N  – ticks near exponents N, 2N, 3N, ... (dense x may look uneven)
      'auto' – evenly spaced bar indices; labels are actual exponents (may not be round)
    exp_reps: one representative exponent per bar (first in compress bucket).
    Returns tick x positions (for optional vertical grid lines).
    """
    n = len(exp_reps)
    if n == 0:
        return []
    if n == 1:
        ax.set_xticks([0.5])
        ax.set_xticklabels([str(exp_reps[0])])
        return [0.5]

    if xtick_gap == "auto":
        n_ticks = min(8, n)
        tick_idx = list(dict.fromkeys(int(i) for i in np.linspace(0, n - 1, n_ticks)))
        tick_pos = [i + 0.5 for i in tick_idx]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(exp_reps[i]) for i in tick_idx], rotation=45, ha="right")
        return tick_pos

    gap = int(xtick_gap)
    if gap <= 0:
        raise ValueError("--xtick-gap must be positive or 'auto'")

    min_e, max_e = exp_reps[0], exp_reps[-1]
    first = ((min_e + gap - 1) // gap) * gap
    marks = list(range(first, max_e + 1, gap))
    if not marks:
        marks = [min_e]

    tick_pos: list[float] = []
    tick_labels: list[str] = []
    used_idx: set[int] = set()
    for m in marks:
        i = bisect.bisect_left(exp_reps, m)
        if i >= n:
            break
        if i in used_idx:
            continue
        used_idx.add(i)
        tick_pos.append(i + 0.5)
        tick_labels.append(str(m))

    if 0 not in used_idx and (not tick_pos or tick_pos[0] > 0.5):
        tick_pos.insert(0, 0.5)
        tick_labels.insert(0, str(exp_reps[0]))
    last_i = n - 1
    if last_i not in used_idx:
        tick_pos.append(last_i + 0.5)
        tick_labels.append(str(exp_reps[last_i]))

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    return tick_pos


def draw_vgrid(ax, tick_pos: list[float]) -> None:
    """Vertical lines at x-axis tick marks (drawn above bars)."""
    for tx in tick_pos:
        ax.axvline(
            tx, color="gray", linestyle=":", linewidth=0.7, alpha=0.7, zorder=10
        )


def plot_bars(
    data: list[Row],
    lo: int,
    hi: int,
    color_mode: str,
    out: Path,
    report_date: str | None,
    xtick_gap: str | int,
    subset_label: str,
    compress: int,
    compress_agg: str = "avg",
    *,
    ymin: float | None = None,
    ymax: float | None = None,
    fig_height: float = DEFAULT_FIG_HEIGHT,
    dpi: int = DEFAULT_DPI,
    vgrid: bool = False,
) -> None:
    raw_n = len(data)
    if raw_n == 0:
        print(f"Skip {subset_label}: no bars", file=sys.stderr)
        return

    bins = compress_rows(data, compress, compress_agg)
    n = len(bins)
    exp_reps = [b["exp_rep"] for b in bins]
    ys = [b["y"] for b in bins]
    t_means = [b["t_level_mean"] for b in bins]

    # Width: ~1 device pixel per bar. Height: fig_height inches * dpi.
    fig_w = max(n / dpi, 4.0)
    fig_h = fig_height
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    x = np.arange(n, dtype=float)
    if color_mode == "factored":
        colors = [
            COLOR_KNOWN if b["factored_frac"] >= 0.5 else COLOR_NO_KNOWN for b in bins
        ]
        ax.bar(x, ys, width=1.0, color=colors, align="edge", linewidth=0, zorder=2)
        if subset_label == "merged":
            ax.plot([], [], color=COLOR_NO_KNOWN, label="no known factors", linewidth=4)
            ax.plot([], [], color=COLOR_KNOWN, label="known factors", linewidth=4)
            ax.legend(loc="upper right", fontsize=8)
    else:
        cmap = level_colormap()
        colors = [cmap[nearest_level(tm)] for tm in t_means]
        ax.bar(x, ys, width=1.0, color=colors, align="edge", linewidth=0, zorder=2)
        used_levels = {nearest_level(tm) for tm in t_means}
        handles = [
            plt.Line2D([0], [0], color=cmap[lv], linewidth=4, label=str(lv))
            for lv in LEVELS
            if lv in used_levels
        ]
        ax.legend(handles=handles, title="t-level", loc="upper right", fontsize=7, ncol=2)

    y_min_data = min(ys)
    y_max_data = max(ys)
    if ymin is None:
        y_floor = max((lv for lv in LEVELS if lv <= y_min_data), default=LEVELS[0])
        y_bottom = y_floor - 0.5
    else:
        y_bottom = ymin
    if ymax is None:
        y_ceil = min((lv for lv in LEVELS if lv >= y_max_data), default=LEVELS[-1])
        if y_ceil < y_max_data:
            y_ceil = y_max_data
        y_top = max(y_max_data, y_ceil) + 0.5
    else:
        y_top = ymax
    if y_bottom >= y_top:
        raise ValueError(f"invalid y range: ymin={y_bottom} >= ymax={y_top}")
    ax.set_ylim(y_bottom, y_top)

    for lv in LEVELS:
        if y_bottom <= lv <= y_top:
            ax.axhline(lv, color="gray", linestyle="--", linewidth=0.6, alpha=0.7, zorder=10)

    ax.set_xlim(0, n)
    ax.set_ylabel("y = (t-level - 5) + 5*(curves / curves_to_test)")
    xlabel = "exponent (dense order)"
    if compress > 1:
        xlabel += f", compress={compress}/px ({compress_agg} y)"
    ax.set_xlabel(xlabel)
    tick_pos = apply_xticks(ax, exp_reps, xtick_gap)
    if vgrid:
        draw_vgrid(ax, tick_pos)

    date_note = report_date if report_date else "latest"
    compress_note = (
        f"compress={compress}/{compress_agg}" if compress > 1 else "1px/exp"
    )
    ax.set_title(
        f"ECM {subset_label}  [{lo}, {hi}]  bars={n} from {raw_n}  "
        f"({date_note})  color={color_mode}  xtick={xtick_gap}  {compress_note}"
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    px_h = int(round(fig_h * dpi))
    print(
        f"Wrote {out}  ({n} bars from {raw_n} exps, {subset_label}, "
        f"compress={compress}/{compress_agg}, ylim=[{y_bottom}, {y_top}], "
        f"fig_h={fig_h}in@{dpi}dpi≈{px_h}px)"
    )


def plot_overlay(
    data: list[Row],
    lo: int,
    hi: int,
    color_mode: str,
    out: Path,
    report_date: str | None,
    xtick_gap: str | int,
    compress: int,
    compress_agg: str = "avg",
    *,
    ymin: float | None = None,
    ymax: float | None = None,
    fig_height: float = DEFAULT_FIG_HEIGHT,
    dpi: int = DEFAULT_DPI,
    vgrid: bool = False,
) -> None:
    """
    X axis = dense known (factored) exponents.
    no_known maps to nearest known pixel (max y if several).
    Draw known [0, y_k]; draw no_known only excess [y_k, y_n] when y_n > y_k.
    """
    known_rows = [r for r in data if r[3] == 1]
    no_rows = [r for r in data if r[3] == 0]
    if not known_rows:
        print("Skip overlay: no known-factor exponents in range", file=sys.stderr)
        return

    known_bins = compress_rows(known_rows, compress, compress_agg)
    no_bins = compress_rows(no_rows, compress, compress_agg) if no_rows else []
    n = len(known_bins)
    exp_reps = [b["exp_rep"] for b in known_bins]
    y_known = [b["y"] for b in known_bins]
    t_means = [b["t_level_mean"] for b in known_bins]

    y_no = [0.0] * n
    for b in no_bins:
        i = nearest_index(exp_reps, b["exp_rep"])
        if b["y"] > y_no[i]:
            y_no[i] = b["y"]

    excess = [max(0.0, yn - yk) for yn, yk in zip(y_no, y_known)]

    fig_w = max(n / dpi, 4.0)
    fig_h = fig_height
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    x = np.arange(n, dtype=float)

    # Foreground known (full bar), then background excess above it
    if color_mode == "level":
        cmap = level_colormap()
        known_colors = [cmap[nearest_level(tm)] for tm in t_means]
    else:
        known_colors = COLOR_KNOWN

    ax.bar(
        x, y_known, width=1.0, color=known_colors, align="edge", linewidth=0,
        label="known factors", zorder=2,
    )
    ax.bar(
        x, excess, width=1.0, bottom=y_known, color=COLOR_NO_KNOWN,
        align="edge", linewidth=0, label="no known (excess only)", zorder=2,
    )
    ax.legend(loc="upper right", fontsize=8)

    y_min_data = min(y_known)
    y_max_data = max(max(y_known), max((yk + ex for yk, ex in zip(y_known, excess)), default=0))
    if ymin is None:
        y_floor = max((lv for lv in LEVELS if lv <= y_min_data), default=LEVELS[0])
        y_bottom = y_floor - 0.5
    else:
        y_bottom = ymin
    if ymax is None:
        y_ceil = min((lv for lv in LEVELS if lv >= y_max_data), default=LEVELS[-1])
        if y_ceil < y_max_data:
            y_ceil = y_max_data
        y_top = max(y_max_data, y_ceil) + 0.5
    else:
        y_top = ymax
    if y_bottom >= y_top:
        raise ValueError(f"invalid y range: ymin={y_bottom} >= ymax={y_top}")
    ax.set_ylim(y_bottom, y_top)

    for lv in LEVELS:
        if y_bottom <= lv <= y_top:
            ax.axhline(lv, color="gray", linestyle="--", linewidth=0.6, alpha=0.7, zorder=10)

    ax.set_xlim(0, n)
    ax.set_ylabel("y = (t-level - 5) + 5*(curves / curves_to_test)")
    xlabel = "exponent (dense known / factored baseline)"
    if compress > 1:
        xlabel += f", compress={compress}/px ({compress_agg})"
    ax.set_xlabel(xlabel)
    tick_pos = apply_xticks(ax, exp_reps, xtick_gap)
    if vgrid:
        draw_vgrid(ax, tick_pos)

    date_note = report_date if report_date else "latest"
    n_excess = sum(1 for e in excess if e > 0)
    ax.set_title(
        f"ECM overlay  [{lo}, {hi}]  known_bars={n}  excess_px={n_excess}  "
        f"({date_note})  xtick={xtick_gap}  compress={compress}/{compress_agg}"
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    px_h = int(round(fig_h * dpi))
    print(
        f"Wrote {out}  (overlay: {n} known bars, {len(no_rows)} no_known mapped, "
        f"{n_excess} excess segments, compress={compress}/{compress_agg}, "
        f"ylim=[{y_bottom}, {y_top}], fig_h={fig_h}in@{dpi}dpi≈{px_h}px)"
    )


def parse_xtick_gap(value: str) -> str | int:
    if value.lower() == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "--xtick-gap must be a positive integer or 'auto'"
        ) from e
    if n <= 0:
        raise argparse.ArgumentTypeError("--xtick-gap must be > 0 or 'auto'")
    return n


def resolve_plot_kinds(plots: str) -> list[str]:
    """Return ordered subset labels to render."""
    if plots == "separate":
        return ["no_known", "known"]
    if plots == "merged":
        return ["merged"]
    if plots == "overlay":
        return ["overlay"]
    if plots == "all":
        return ["no_known", "known", "merged", "overlay"]
    if plots == "known":
        return ["known"]
    if plots == "no_known":
        return ["no_known"]
    raise ValueError(plots)


def filter_subset(data: list[Row], kind: str) -> list[Row]:
    if kind == "merged":
        return data
    if kind == "known":
        return [r for r in data if r[3] == 1]
    if kind == "no_known":
        return [r for r in data if r[3] == 0]
    raise ValueError(kind)


def default_out_stem(
    lo: int, hi: int, report_date: str | None, color_mode: str
) -> Path:
    tag = f"_{report_date}" if report_date else ""
    return Path(__file__).parent / f"ecm_progress_{lo}-{hi}{tag}_{color_mode}"


def out_path_for(stem_or_file: Path, kind: str, multi: bool) -> Path:
    """
    If user passed --out file.png and multi outputs, write stem_known.png etc.
    If single output, use --out as-is when it has a suffix.
    """
    if not multi and stem_or_file.suffix.lower() in {".png", ".pdf", ".svg"}:
        return stem_or_file

    if stem_or_file.suffix.lower() in {".png", ".pdf", ".svg"}:
        stem = stem_or_file.with_suffix("")
        suffix = stem_or_file.suffix
    else:
        stem = stem_or_file
        suffix = ".png"
    return Path(f"{stem}_{kind}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot dense ECM progress bars (1px per exponent in range)."
    )
    parser.add_argument("--lo", type=int, default=1, help="Exponent range low (default 1)")
    parser.add_argument("--hi", type=int, default=20000, help="Exponent range high (default 20000)")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("ECM_DB", DEFAULT_DB)),
        help="SQLite database path",
    )
    parser.add_argument(
        "--date",
        metavar="YYYYMMDD",
        help=(
            "Optional as-of date from sparse progress_history "
            "(default: exponents main table / latest)"
        ),
    )
    parser.add_argument(
        "--color",
        choices=("factored", "level"),
        default="factored",
        help="Bar coloring mode (default: factored)",
    )
    parser.add_argument(
        "--plots",
        choices=("separate", "merged", "overlay", "all", "known", "no_known"),
        default="separate",
        help=(
            "Which figures to write: separate=no_known+known (default), "
            "merged, overlay (known x-baseline + no_known excess), "
            "all, or a single subset"
        ),
    )
    parser.add_argument(
        "--xtick-gap",
        type=parse_xtick_gap,
        default=1000,
        metavar="N|auto",
        help=(
            "X tick mode: integer gap in exponent space (default 1000, may look uneven), "
            "or 'auto' for evenly spaced ticks with actual exponent labels"
        ),
    )
    parser.add_argument(
        "--compress",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Merge N consecutive exponents into one pixel/bar "
            "(default 1 = no compression); aggregation via --compress-agg"
        ),
    )
    parser.add_argument(
        "--compress-agg",
        choices=("avg", "max"),
        default="avg",
        help="Within each compress bucket: avg (default) or max of y",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Y-axis lower bound (default: auto from data / digit grid)",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Y-axis upper bound (default: auto from data / digit grid)",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=DEFAULT_FIG_HEIGHT,
        metavar="INCHES",
        help=f"Figure height in inches (default {DEFAULT_FIG_HEIGHT}; pixel height ≈ inches * dpi)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PNG dpi (default {DEFAULT_DPI}; also sets ~1 bar = 1 px width)",
    )
    parser.add_argument(
        "--vgrid",
        action="store_true",
        help="Draw vertical grid lines at x-axis tick marks (from --xtick-gap)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output path or stem (multi plots append _known / _no_known / _merged)",
    )
    parser.add_argument("--csv", type=Path, help="Optional summary CSV path")
    args = parser.parse_args()

    if args.lo > args.hi:
        sys.exit("ERROR: --lo must be <= --hi")
    if args.compress < 1:
        sys.exit("ERROR: --compress must be >= 1")
    if args.fig_height <= 0:
        sys.exit("ERROR: --fig-height must be > 0")
    if args.dpi <= 0:
        sys.exit("ERROR: --dpi must be > 0")
    if args.ymin is not None and args.ymax is not None and args.ymin >= args.ymax:
        sys.exit("ERROR: --ymin must be < --ymax")
    if not args.db.exists():
        sys.exit(f"ERROR: Database not found: {args.db}")

    conn = sqlite3.connect(args.db)
    if args.date:
        row = conn.execute(
            "SELECT COUNT(*) FROM progress_history WHERE report_date <= ?",
            (args.date,),
        ).fetchone()
        if not row or row[0] == 0:
            dates = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT report_date FROM progress_history ORDER BY report_date"
                )
            ]
            conn.close()
            sys.exit(
                f"ERROR: no progress_history with report_date <= {args.date}. "
                f"Available: {', '.join(dates) or '(none)'}"
            )

    data = load_rows(conn, args.lo, args.hi, args.date)
    conn.close()

    if not data:
        sys.exit(f"ERROR: No exponents in [{args.lo}, {args.hi}] for the selected source.")

    for exp, t, c, _ in data:
        if t not in CURVES_TO_TEST:
            sys.exit(f"ERROR: exponent {exp} has unknown t_level={t}")

    kinds = resolve_plot_kinds(args.plots)
    multi = len(kinds) > 1
    stem = args.out or default_out_stem(args.lo, args.hi, args.date, args.color)

    for kind in kinds:
        out = out_path_for(stem, kind, multi)
        common = dict(
            lo=args.lo,
            hi=args.hi,
            color_mode=args.color,
            out=out,
            report_date=args.date,
            xtick_gap=args.xtick_gap,
            compress=args.compress,
            compress_agg=args.compress_agg,
            ymin=args.ymin,
            ymax=args.ymax,
            fig_height=args.fig_height,
            dpi=args.dpi,
            vgrid=args.vgrid,
        )
        if kind == "overlay":
            plot_overlay(data, **common)
        else:
            plot_bars(
                filter_subset(data, kind),
                subset_label=kind,
                **common,
            )

    summary = summary_rows(data)
    print_summary(summary)
    if args.csv:
        write_csv(args.csv, summary)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
