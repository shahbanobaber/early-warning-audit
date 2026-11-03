"""python -m ewaudit download | run"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# One core: the run should not slow the rest of the machine.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ewaudit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download", help="fetch OULAD from the UCI repository (about 45 MB)")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p = sub.add_parser("run", help="fit, score and audit every week; write results/")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--cache", type=Path, default=Path("work"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--last-week", type=int, default=20)
    p.add_argument("--budget", type=float, default=0.2, help="share of enrolled students tutors can contact")
    p.add_argument("--test", default="2014J", choices=["2013J", "2014B", "2014J"],
                   help="presentation to test on; every earlier one is used for training")
    a = ap.parse_args(argv)

    from .data import download, load

    if a.cmd == "download":
        print(download(a.raw))
        return
    from . import plots, report
    from .pipeline import run

    result = run(load(a.raw, a.cache, test=a.test), a.out, weeks=range(0, a.last_week + 1), main_budget=a.budget)
    plots.by_week(result, a.out / "by_week.png")
    first = result["audits"][str(result["settings"]["audit_weeks"][0])]
    plots.by_group(first, a.out / "by_group.png")
    plots.missing_traces(first, a.out / "missing_traces.png")
    plots.calibration(first, a.out / "calibration.png")
    (a.out / "summary.md").write_text(report.summary(result))
    print(f"wrote {a.out}/metrics.json, summary.md and 4 figures")


if __name__ == "__main__":
    main()
