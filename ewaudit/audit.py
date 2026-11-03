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


def missing_traces(G: pd.DataFrame, y: np.ndarray, X: pd.DataFrame, rng, n_boot: int = 1000) -> list[dict]:
    """Does the absence of a trace raise risk by the same amount in every group?

    For each absent trace and group: the share without it, the at-risk rate
    with and without it, and the difference. `gap` is that difference minus
    the same difference for everyone else. If it is far from zero, the same
    silence means something different for this group. Intervals resample
    students.
    """
    traces = {
        "never used the forum": X["forum_any"].to_numpy() == 0,
        "no activity in the last week": X["last_week_clicks"].to_numpy() == 0,
    }
    rows = []
    y = y.astype(float)
    for trace, absent in traces.items():
        for dim, (_, order) in DIMENSIONS.items():
            codes = pd.Categorical(G[dim], categories=order).codes
            for j, name in enumerate(order):
                m = codes == j
                if (m & absent).sum() < 10 or (m & ~absent).sum() < 10:
                    continue
                eff, other = _absence_effect(y, absent, m), _absence_effect(y, absent, ~m)
                boots = []
                for _ in range(n_boot):
                    i = rng.integers(0, len(y), len(y))
                    e = _absence_effect(y[i], absent[i], m[i])
                    boots.append((e, e - _absence_effect(y[i], absent[i], ~m[i])))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ci = np.nanpercentile(np.array(boots), [2.5, 97.5], axis=0)
                rows.append(dict(trace=trace, dimension=dim, group=name, n=int(m.sum()),
                                 share_absent=float(absent[m].mean()), risk_absent=float(y[m & absent].mean()),
                                 risk_present=float(y[m & ~absent].mean()), difference=float(eff),
                                 difference_ci=ci[:, 0].tolist(), gap=float(eff - other), gap_ci=ci[:, 1].tolist()))
    return rows


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
