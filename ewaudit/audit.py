"""Who the warning fails for, and whether a missing trace means the same thing for everyone."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

IMD = {"0-10%": "0-20%", "10-20": "0-20%", "10-20%": "0-20%", "20-30%": "20-40%", "30-40%": "20-40%",
       "40-50%": "40-60%", "50-60%": "40-60%", "60-70%": "60-80%", "70-80%": "60-80%",
       "80-90%": "80-100%", "90-100%": "80-100%"}
EDUCATION = {"No Formal quals": "below A level", "Lower Than A Level": "below A level",
             "A Level or Equivalent": "A level", "HE Qualification": "higher education",
             "Post Graduate Qualification": "higher education"}
DIMENSIONS = {
    "deprivation": ("IMD band of home area (0-20% = most deprived)",
                    ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%", "unknown"]),
    "disability": ("disability", ["declared", "none declared"]),
    "education": ("highest prior education", ["below A level", "A level", "higher education"]),
}


def groups(students: pd.DataFrame) -> pd.DataFrame:
    """Audit groups per student. Small categories are merged so each group has enough students."""
    return pd.DataFrame({
        "deprivation": students.imd_band.map(IMD).fillna("unknown"),
        "disability": students.disability.map({"Y": "declared", "N": "none declared"}),
        "education": students.highest_education.map(EDUCATION),
    }, index=students.index)


def _rates(g: np.ndarray, y: np.ndarray, f: np.ndarray, k: int) -> dict[str, np.ndarray]:
    n = np.bincount(g, minlength=k)
    pos = np.bincount(g, weights=y, minlength=k)
    flagged = np.bincount(g, weights=f, minlength=k)
    tp = np.bincount(g, weights=y * f, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(n=n, at_risk=pos / n, flagged=flagged / n, miss=1 - tp / pos, precision=tp / flagged,
                    false_alarm=(flagged - tp) / (n - pos))


def subgroup_errors(G: pd.DataFrame, y: np.ndarray, flagged: np.ndarray, rng, n_boot: int = 1000) -> list[dict]:
    """Per group: base rate, flag rate, miss rate (at-risk students not flagged), precision, false alarms.

    Intervals resample students. `miss_gap` is the group's miss rate minus
    everyone else's, with its interval.
    """
    y, f = y.astype(float), flagged.astype(float)
    rows = []
    for dim, (_, order) in DIMENSIONS.items():
        codes = pd.Categorical(G[dim], categories=order).codes
        k = len(order)
        obs = _rates(codes, y, f, k)
        rest = [_rates((codes != j).astype(int), y, f, 2)["miss"][1] for j in range(k)]
        boot = {m: [] for m in ("miss", "precision", "flagged", "gap")}
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            b = _rates(codes[i], y[i], f[i], k)
            boot["miss"].append(b["miss"])
            boot["precision"].append(b["precision"])
            boot["flagged"].append(b["flagged"])
            boot["gap"].append([b["miss"][j] - _rates((codes[i] != j).astype(int), y[i], f[i], 2)["miss"][1]
                                for j in range(k)])
        with warnings.catch_warnings():  # an empty group has no interval
            warnings.simplefilter("ignore", RuntimeWarning)
            ci = {m: np.nanpercentile(np.array(v), [2.5, 97.5], axis=0) for m, v in boot.items()}
        for j, name in enumerate(order):
            rows.append(dict(dimension=dim, group=name, n=int(obs["n"][j]), at_risk=obs["at_risk"][j],
                             flagged=obs["flagged"][j], flagged_ci=ci["flagged"][:, j].tolist(),
                             miss=obs["miss"][j], miss_ci=ci["miss"][:, j].tolist(),
                             precision=obs["precision"][j], precision_ci=ci["precision"][:, j].tolist(),
                             false_alarm=obs["false_alarm"][j],
                             miss_gap=obs["miss"][j] - rest[j], miss_gap_ci=ci["gap"][:, j].tolist()))
    return [{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()} for r in rows]


def _absence_effect(y: np.ndarray, absent: np.ndarray, member: np.ndarray) -> float:
    """At-risk rate among members without the trace minus members with it."""
    a, p = member & absent, member & ~absent
    return y[a].mean() - y[p].mean() if a.any() and p.any() else np.nan


# Absent traces, each tied to the self-regulation skill it is meant to show.
# Each entry returns (trace absent, trace applicable) for every student.
TRACES = {
    "no activity in the last week": ("regular study (time management)",
                                     lambda X: (X.last_week_clicks == 0, X.last_week_clicks.notna())),
    "missed an assessment deadline": ("meeting deadlines (time management)",
                                      lambda X: (X.missed_frac > 0, X.missed_frac.notna())),
    "never used the forum": ("help-seeking", lambda X: (X.forum_any == 0, X.forum_any.notna())),
    "never took a quiz": ("self-testing (monitoring)", lambda X: (X.quiz_clicks == 0, X.quiz_clicks.notna())),
    "no activity before the start": ("planning", lambda X: (X.prestart_clicks == 0, X.prestart_clicks.notna())),
}
FIGURE_TRACES = ["never used the forum", "no activity in the last week"]


def _trace(X: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    absent, ok = TRACES[name][1](X)
    return (absent & ok).to_numpy(), ok.to_numpy()


def _available(X: pd.DataFrame, traces) -> list[str]:
    """Traces whose columns exist this week (week 0 has no in-course activity, so some are dropped)."""
    out = []
    for t in traces:
        try:
            _trace(X, t)
        except AttributeError:
            continue
        out.append(t)
    return out


def missing_traces(G: pd.DataFrame, y: np.ndarray, X: pd.DataFrame, rng, n_boot: int = 1000,
                   traces=None) -> list[dict]:
    """Does the absence of a trace raise risk by the same amount in every group?

    For each absent trace and group: the share without it, the at-risk rate
    with and without it, and the difference. `gap` is that difference minus
    the same difference for everyone else. If it is far from zero, the same
    silence means something different for this group. Only students for whom
    the trace applies count (no deadline yet means no missed deadline).
    Intervals resample students.
    """
    rows = []
    y = y.astype(float)
    for trace in _available(X, traces or TRACES):
        absent, ok = _trace(X, trace)
        yt, at = y[ok], absent[ok]
        for dim, (_, order) in DIMENSIONS.items():
            codes = pd.Categorical(G[dim], categories=order).codes[ok]
            for j, name in enumerate(order):
                m = codes == j
                if min((m & at).sum(), (m & ~at).sum(), (~m & at).sum(), (~m & ~at).sum()) < 10:
                    continue   # too few on one side, inside the group or in the rest, to compare
                eff, other = _absence_effect(yt, at, m), _absence_effect(yt, at, ~m)
                boots = []
                for _ in range(n_boot):
                    i = rng.integers(0, len(yt), len(yt))
                    e = _absence_effect(yt[i], at[i], m[i])
                    boots.append((e, e - _absence_effect(yt[i], at[i], ~m[i])))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ci = np.nanpercentile(np.array(boots), [2.5, 97.5], axis=0)
                rows.append(dict(trace=trace, dimension=dim, group=name, n=int(m.sum()),
                                 share_absent=float(at[m].mean()), risk_absent=float(yt[m & at].mean()),
                                 risk_present=float(yt[m & ~at].mean()), difference=float(eff),
                                 difference_ci=ci[:, 0].tolist(), gap=float(eff - other), gap_ci=ci[:, 1].tolist()))
    return rows


def scorecard(G: pd.DataFrame, y: np.ndarray, X: pd.DataFrame, presentation: pd.Series, rng,
              n_boot: int = 300, min_n: int = 20) -> tuple[list[dict], list[dict]]:
    """Each trace against the three criteria Domino and Shaffer (2025) propose, with no model involved.

    Predictive: at-risk rate without the trace minus with it. (Their validity
    is about inferring an internal skill, which outcomes alone can't settle;
    this is the outcome side of it.)
    Stable across course runs: the same difference inside each presentation.
    Same meaning across groups: the gaps from `missing_traces`.
    """
    y = y.astype(float)
    pres = presentation.to_numpy()
    gaps = missing_traces(G, y, X, rng, n_boot=n_boot)
    rows = []
    for trace in _available(X, TRACES):
        skill = TRACES[trace][0]
        absent, ok = _trace(X, trace)
        per = []
        for run in sorted(set(pres)):
            m = (pres == run) & ok
            if (m & absent).sum() >= min_n and (m & ~absent).sum() >= min_n:
                per.append(_absence_effect(y, absent, m))
        g = [r for r in gaps if r["trace"] == trace]
        if not g or not per:   # too few students with or without the trace to say anything
            continue
        worst = max(g, key=lambda r: abs(r["gap"]))
        rows.append(dict(trace=trace, skill=skill, applicable=float(ok.mean()), share_absent=float(absent[ok].mean()),
                         difference=float(_absence_effect(y[ok], absent[ok], np.ones(ok.sum(), bool))),
                         runs=len(per), runs_total=len(set(pres)), runs_positive=int(sum(d > 0 for d in per)),
                         run_min=float(min(per)), run_median=float(np.median(per)), run_max=float(max(per)),
                         worst_group=f"{worst['dimension']}: {worst['group']}", worst_gap=worst["gap"],
                         worst_gap_ci=worst["gap_ci"], groups=len(g),
                         groups_with_clear_gap=sum(r["gap_ci"][0] > 0 or r["gap_ci"][1] < 0 for r in g)))
    return rows, gaps


def calibration(G: pd.DataFrame, y: np.ndarray, prob: np.ndarray, rng, n_boot: int = 1000) -> list[dict]:
    """Mean predicted risk against the observed at-risk rate per group.

    The model never sees group membership. If a group's observed rate sits
    above its mean prediction, the model underrates that group's risk.
    """
    y = y.astype(float)
    rows = []
    groups = [("all", "everyone", np.ones(len(y), bool))]
    for dim, (_, order) in DIMENSIONS.items():
        codes = pd.Categorical(G[dim], categories=order).codes
        groups += [(dim, name, codes == j) for j, name in enumerate(order)]
    for dim, name, m in groups:
        if m.sum() < 10:
            continue
        ym, pm = y[m], prob[m]
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(ym), len(ym))
            boots.append(ym[i].mean() - pm[i].mean())
        rows.append(dict(dimension=dim, group=name, n=int(m.sum()), observed=float(ym.mean()),
                         predicted=float(pm.mean()), under=float(ym.mean() - pm.mean()),
                         under_ci=np.percentile(boots, [2.5, 97.5]).tolist()))
    return rows
