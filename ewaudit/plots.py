"""Figures. One colour per ranking method, the same in every figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .audit import DIMENSIONS  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE, NULL = "#e6e5e1", "#fcfcfb", "#dddcd7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLOURS = {"boosting": BLUE, "logistic": AQUA, "quiet": ORANGE}
LABELS = {"boosting": "gradient boosting", "logistic": "logistic regression", "quiet": "inactivity rule"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "axes.axisbelow": True,
    })


def by_week(result: dict, path: Path) -> None:
    _style()
    rows = result["by_week"]
    b = str(result["settings"]["main_budget"])
    main = result["settings"]["main"]
    w = np.array([r["week"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True)

    ax = axes[0]
    for kind in ("boosting", "logistic", "quiet"):
        y = [r["models"][kind]["auc"] for r in rows]
        ax.plot(w, y, color=COLOURS[kind], label=LABELS[kind])
    lo, hi = zip(*[r["models"][main]["ci"]["auc"] for r in rows], strict=True)
    ax.fill_between(w, lo, hi, color=BLUE, alpha=0.15, lw=0)
    ax.set_ylabel("AUC on the next year's cohort")
    ax.set_title("How well each week's warning ranks the students still enrolled")
    ax.legend(loc="upper left", fontsize=8.5)

    ax = axes[1]
    for kind in ("boosting", "quiet"):
        ax.plot(w, [r["models"][kind]["budgets"][b]["precision"] for r in rows], color=COLOURS[kind],
                label=f"{LABELS[kind]}, top {float(b):.0%} flagged")
    lo, hi = zip(*[r["models"][main]["ci"]["precision"] for r in rows], strict=True)
    ax.fill_between(w, lo, hi, color=BLUE, alpha=0.15, lw=0)
    ax.plot(w, [r["base_rate"] for r in rows], color=MUTED, lw=1.5, ls=(0, (3, 2)),
            label="contacting students at random")
    ax.set_ylabel("share of flagged students\nwho go on to fail or withdraw")
    ax.set_ylim(0.2, 1.0)
    ax.set_title(f"Contacting the top {float(b):.0%}: how many contacts go to students who need it")
    ax.legend(loc="upper left", fontsize=8.5)

    ax = axes[2]
    total = rows[0]["eventual_at_risk"]
    gone = np.array([r["already_gone"] for r in rows]) / total
    reached = np.array([r["models"][main]["budgets"][b]["reached_share"] for r in rows])
    missed = 1 - gone - reached
    ax.stackplot(w, gone, reached, missed, colors=[NULL, BLUE, "#f6c3a5"], edgecolor=SURFACE, lw=1,
                 labels=["already withdrawn", f"flagged this week ({LABELS[main]})", "still enrolled, not flagged"])
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"share of the {total:,} students\nwho end up failing or withdrawing")
    ax.set_title("Who a warning can still reach")
    ax.legend(loc="upper left", fontsize=8.5, ncol=1)
    ax.set_xlabel("end of week (week 0 = before the course starts)")
    ax.set_xticks(w[::2])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def by_group(audit: dict, path: Path) -> None:
    """Miss rate and precision per group, with 95% intervals, at one week and budget."""
    _style()
    rows = audit["subgroups"]
    dims = list(DIMENSIONS)
    heights = [len(DIMENSIONS[d][1]) + 0.8 for d in dims]
    fig, axes = plt.subplots(len(dims), 2, figsize=(7.2, 6.2), sharex="col",
                             gridspec_kw={"height_ratios": heights})
    first = [r for r in rows if r["dimension"] == dims[0]]   # every student appears once per dimension
    n_all = sum(r["n"] for r in first)
    weight = {"miss": "at_risk", "precision": "flagged"}     # miss is per at-risk student, precision per flag
    overall = {m: sum(r[m] * r["n"] * r[w] for r in first) / sum(r["n"] * r[w] for r in first)
               for m, w in weight.items()}
    for i, d in enumerate(dims):
        sub = [r for r in rows if r["dimension"] == d]
        for j, (m, title) in enumerate([("miss", "at-risk students not flagged"),
                                        ("precision", "flagged students who were at risk")]):
            ax = axes[i, j]
            y = np.arange(len(sub))[::-1]
            for yy, r in zip(y, sub, strict=True):
                lo, hi = r[f"{m}_ci"]
                ax.plot([lo, hi], [yy, yy], color=BLUE, lw=2, alpha=0.45)
                ax.plot(r[m], yy, "o", ms=7, color=BLUE, mec=SURFACE, mew=1.5)
            ax.axvline(overall[m], color=MUTED, lw=1, ls=(0, (3, 2)))
            ax.set_yticks(y, [f"{r['group']} ({r['n']:,})" for r in sub] if j == 0 else [""] * len(sub))
            ax.set_ylim(-0.6, len(sub) - 0.4)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=GRID, lw=0.8)
            if i == 0:
                ax.set_title(title, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(DIMENSIONS[d][0].split(" (")[0], fontsize=8.5)
    for j in range(2):
        axes[-1, j].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.suptitle(f"Week {audit['week']}, top {audit['budget']:.0%} of {n_all:,} enrolled students flagged. "
                 "Dashed line = everyone.", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _label(dim: str, group: str) -> str:
    if dim == "deprivation":
        return f"IMD {group}" if group != "unknown" else "IMD unknown"
    if dim == "disability":
        return "disability declared" if group == "declared" else "no disability declared"
    return f"education: {group}"


def missing_traces(audit: dict, path: Path) -> None:
    """How much the absence of each trace raises the at-risk rate, per group."""
    _style()
    rows = audit["missing_traces"]
    traces = list(dict.fromkeys(r["trace"] for r in rows))
    colours = dict(zip(traces, [BLUE, ORANGE], strict=False))
    labels = [(r["dimension"], r["group"]) for r in rows if r["trace"] == traces[0]]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    y = np.arange(len(labels))[::-1]
    for k, t in enumerate(traces):
        sub = {(r["dimension"], r["group"]): r for r in rows if r["trace"] == t}
        off = 0.17 if k == 0 else -0.17
        for yy, lab in zip(y, labels, strict=True):
            r = sub[lab]
            lo, hi = r["difference_ci"]
            ax.plot([lo, hi], [yy + off, yy + off], color=colours[t], lw=2, alpha=0.45)
            ax.plot(r["difference"], yy + off, "o", ms=7, color=colours[t], mec=SURFACE, mew=1.5,
                    label=t if yy == y[0] else None)
    ax.set_yticks(y, [_label(d, g) for d, g in labels])
    for yy, (d, _), prev in zip(y, labels, [None] + labels[:-1], strict=True):
        if prev is not None and prev[0] != d:
            ax.axhline(yy + 0.5, color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:+.0f}"))
    ax.set_xlim(left=0)
    ax.set_xlabel("at-risk rate without the trace minus with it (percentage points)")
    ax.set_title(f"Week {audit['week']}: how much a missing trace raises the risk, by group")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _rows_axis(ax, labels):
    """Group rows top to bottom with a thin rule between dimensions."""
    y = np.arange(len(labels))[::-1]
    ax.set_yticks(y, [_label(d, g) for d, g in labels])
    for yy, (d, _), prev in zip(y, labels, [None] + labels[:-1], strict=True):
        if prev is not None and prev[0] != d:
            ax.axhline(yy + 0.5, color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    return y


def calibration(audit: dict, path: Path) -> None:
    """Observed at-risk rate against the model's mean predicted risk, per group."""
    _style()
    rows = [r for r in audit["calibration"] if r["dimension"] != "all"]
    labels = [(r["dimension"], r["group"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    y = _rows_axis(ax, labels)
    for yy, r in zip(y, rows, strict=True):
        ax.plot([r["predicted"], r["observed"]], [yy, yy], color=NULL, lw=3, zorder=1)
    ax.plot([r["predicted"] for r in rows], y, "o", ms=8, color=BLUE, mec=SURFACE, mew=1.5, ls="none", zorder=3,
            label="model's mean predicted risk")
    ax.plot([r["observed"] for r in rows], y, "o", ms=8, color=ORANGE, mec=SURFACE, mew=1.5, ls="none", zorder=3,
            label="share who actually failed or withdrew")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(f"Week {audit['week']}: the traces barely separate groups whose real risk differs")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
