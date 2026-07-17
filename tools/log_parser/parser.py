"""ECM (Prime95/mprime) worker log parser.

Segments a screen log into individual ECM *runs*. A run is opened by every
``ECM on M...: Edwards curve #N`` line and closed by the next ``ECM on`` line
for the same worker (or end of file). All Stage / memory / FFT lines that fall
inside that window are attributed to the run.
"""

import re
from datetime import datetime

LINE_RE = re.compile(
    r"^\[(?P<worker>[^\]]+)\]\s+\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s?(?P<msg>.*)$"
)

WORKER_NUM_RE = re.compile(r"Worker #(\d+)")

ECM_RE = re.compile(
    r"ECM on M(?P<exp>\d+):\s*Edwards curve #(?P<curve>\d+)\s*"
    r"with s=(?P<s>\d+),\s*B1=(?P<b1>\d+),\s*B2=(?P<b2tbd>TBD|\d+)"
)

# FFT lines. "back" is captured separately so it never overrides S2.
FFT_USING_RE = re.compile(r"Using (?P<type>\w+) FFT length (?P<len>\d+)")
FFT_SWITCH_TO_RE = re.compile(r"Switching to (?P<type>\w+) FFT length (?P<len>\d+)")

B2_RE = re.compile(
    r"Actual B2 will be (?P<b2>\d+).*?Curve is worth (?P<worth>\d+(?:\.\d+)?)"
)
AVAIL_MEM_RE = re.compile(r"Available memory is (?P<mem>\d+)MB")
USING_MEM_RE = re.compile(r"Using (?P<mem>\d+)MB of memory")
STAGE1_RE = re.compile(r"Stage 1 complete\..*?Total time:\s*(?P<t>\d+(?:\.\d+)?)\s*sec")
STAGE2_INIT_RE = re.compile(
    r"Stage 2 init complete\..*?Time:\s*(?P<t>\d+(?:\.\d+)?)\s*sec"
)
STAGE2_RE = re.compile(
    r"Stage 2 complete\..*?Total time:\s*(?P<t>\d+(?:\.\d+)?)\s*sec"
)
STAGE2_GCD_RE = re.compile(
    r"Stage 2 GCD complete\..*?Time:\s*(?P<t>\d+(?:\.\d+)?)\s*sec"
)

TS_FMT = "%Y-%m-%d %H:%M:%S"


def _new_run(worker_label, worker_num, ts, m, s1_fft):
    return {
        "worker": worker_num,
        "worker_label": worker_label,
        "exponent": int(m.group("exp")),
        "curve": int(m.group("curve")),
        "s": m.group("s"),
        "b1": int(m.group("b1")),
        "b2": None,
        "worth": None,
        "avail_mem": None,
        "using_mem": None,
        "s1_time": None,
        "s2_init_time": None,
        "s2_time": None,
        "s2_gcd_time": None,
        "s1_fft": s1_fft[1] if s1_fft else None,
        "s2_fft": None,
        "fft_type": s1_fft[0] if s1_fft else None,
        "start_ts": ts,
        "end_ts": ts,
        # timestamps used for the gantt breakdown (not exported columns)
        "s1_end_ts": None,
        "s2_start_ts": None,
        "s2_end_ts": None,
        "status": "interrupted",
    }


def _finalize(run):
    if run["s2_time"] is not None:
        run["status"] = "complete"
    elif run["s1_time"] is not None:
        run["status"] = "stage1-only"
    else:
        run["status"] = "interrupted"
    return run


def parse_log(text):
    """Parse raw log text -> list of run dicts (in start order)."""
    runs = []
    open_run = {}          # worker_num -> run dict
    last_using_fft = {}    # worker_num -> (type, length)  (most recent "Using")

    for raw in text.splitlines():
        m = LINE_RE.match(raw.rstrip("\r"))
        if not m:
            continue
        worker_label = m.group("worker")
        ts = m.group("ts")
        msg = m.group("msg")

        wnum_m = WORKER_NUM_RE.search(worker_label)
        if not wnum_m:
            # [Main window] / [Comm window] never produce ECM runs.
            continue
        wnum = int(wnum_m.group(1))

        # Track the most-recent "Using ... FFT length" for this worker so an
        # ECM line can grab the FFT that precedes it.
        um = FFT_USING_RE.search(msg)
        if um:
            last_using_fft[wnum] = (um.group("type"), int(um.group("len")))

        # New ECM run.
        ecm = ECM_RE.search(msg)
        if ecm:
            if wnum in open_run:
                runs.append(_finalize(open_run.pop(wnum)))
            open_run[wnum] = _new_run(
                worker_label, wnum, ts, ecm, last_using_fft.get(wnum)
            )
            continue

        run = open_run.get(wnum)
        if run is None:
            continue

        run["end_ts"] = ts

        b2m = B2_RE.search(msg)
        if b2m:
            run["b2"] = int(b2m.group("b2"))
            run["worth"] = float(b2m.group("worth"))

        am = AVAIL_MEM_RE.search(msg)
        if am:
            run["avail_mem"] = int(am.group("mem"))

        umem = USING_MEM_RE.search(msg)
        if umem:
            run["using_mem"] = int(umem.group("mem"))

        s1 = STAGE1_RE.search(msg)
        if s1:
            run["s1_time"] = float(s1.group("t"))
            run["s1_end_ts"] = ts

        # S2 FFT: only the "Switching to" line after stage 1 completed.
        sw = FFT_SWITCH_TO_RE.search(msg)
        if sw and run["s1_time"] is not None and run["s2_fft"] is None:
            run["s2_fft"] = int(sw.group("len"))
            if not run["fft_type"]:
                run["fft_type"] = sw.group("type")

        s2i = STAGE2_INIT_RE.search(msg)
        if s2i:
            run["s2_init_time"] = float(s2i.group("t"))
            run["s2_start_ts"] = ts

        s2 = STAGE2_RE.search(msg)
        if s2:
            run["s2_time"] = float(s2.group("t"))
            run["s2_end_ts"] = ts

        s2g = STAGE2_GCD_RE.search(msg)
        if s2g:
            run["s2_gcd_time"] = float(s2g.group("t"))
            run["end_ts"] = ts

    for run in open_run.values():
        runs.append(_finalize(run))

    # If S2 FFT never switched but S1 known, inherit S1 (FFT stayed the same).
    for run in runs:
        if run["s2_fft"] is None and run["s2_time"] is not None:
            run["s2_fft"] = run["s1_fft"]

    runs.sort(key=lambda r: (r["start_ts"], r["worker"]))
    return runs


# Column definitions shared with the export layer. (key, header)
COLUMNS = [
    ("worker_label", "Worker"),
    ("exponent", "Exponent"),
    ("curve", "Curve #"),
    ("s", "s"),
    ("b1", "B1"),
    ("b2", "B2 (Actual)"),
    ("worth", "Worth"),
    ("avail_mem", "mem (MB)"), # Available mem (MB)
    ("using_mem", "Using mem (MB)"),
    ("s1_time", "S1 time (s)"), # Stage 1 time (s)
    ("s2_init_time", "S2 init time (s)"), # Stage 2 init time (s)
    ("s2_time", "S2 time (s)"), # Stage 2 time (s)
    ("s2_gcd_time", "S2 GCD time (s)"), # Stage 2 GCD time (s)
    ("s1_fft", "S1 FFT"),
    ("s2_fft", "S2 FFT"),
    ("fft_type", "FFT type"),
    ("start_ts", "Start"),
    ("end_ts", "End"),
    ("status", "Status"),
]


if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "screen_example.log"
    with open(path, encoding="utf-8", errors="replace") as fh:
        rows = parse_log(fh.read())
    print(f"Parsed {len(rows)} runs")
    print(json.dumps(rows[:3], indent=2, ensure_ascii=False))
