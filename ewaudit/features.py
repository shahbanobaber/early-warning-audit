"""Trace features for each student at the end of a given week, from data strictly before that point.

Week w ends at day 7w of the module (day 0 is the official start). Week 0
uses only what happened before the start. Everything here is a behavioural
trace: clicks, timing of activity, and timing of assessment submissions.
Marks, demographics and module identity are left out on purpose. Marks are
the outcome a trace is meant to anticipate, demographics are what the audit
checks errors against, and module identity would not carry to a new course.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import GROUPS, Data


def population(students: pd.DataFrame, week: int) -> pd.DataFrame:
    """Students registered before the cutoff and not yet withdrawn: the ones a warning could still reach."""
    c = 7 * week
    registered = students.date_registration.isna() | (students.date_registration < c)
    still_in = students.date_unregistration.isna() | (students.date_unregistration >= c)
    return students[registered & still_in]


def features(data: Data, week: int, keys: pd.Index | None = None) -> pd.DataFrame:
    """One row per student in `keys` (default: the population at this week). Uses only rows dated before 7*week."""
    c = 7 * week
    if keys is None:
        keys = population(data.students, week).index
    s = data.students.loc[keys]
    d = data.daily[(data.daily.date < c) & data.daily.key.isin(keys)]
    X = pd.DataFrame(index=keys)

    per_key = d.groupby("key").clicks.sum()
    X["clicks"] = np.log1p(per_key.reindex(keys, fill_value=0))
    X["prestart_clicks"] = np.log1p(d[d.date < 0].groupby("key").clicks.sum().reindex(keys, fill_value=0))
    last_week = d[d.date >= c - 7].groupby("key").clicks.sum()
    X["last_week_clicks"] = np.log1p(last_week.reindex(keys, fill_value=0)) if week > 0 else np.nan
    for g, name in enumerate(GROUPS):
        X[f"{name}_clicks"] = np.log1p(d[d.group == g].groupby("key").clicks.sum().reindex(keys, fill_value=0))
    X["forum_any"] = (X["forum_clicks"] > 0).astype(float)

    # Regularity: how much of the elapsed course a student was active in.
    days = d.loc[d.date >= 0, ["key", "date"]].drop_duplicates().sort_values(["key", "date"])
    if week > 0:
        X["active_days"] = days.groupby("key").size().reindex(keys, fill_value=0)
        weeks_active = (days.date // 7).groupby(days.key).nunique().reindex(keys, fill_value=0)
        X["active_weeks_frac"] = weeks_active / week
        prev = days.groupby("key").date.shift(fill_value=-1)
        inner = (days.date - prev - 1).groupby(days.key).max()
        trailing = c - 1 - days.groupby("key").date.max()
        X["longest_gap"] = pd.concat([inner, trailing], axis=1).max(axis=1).reindex(keys, fill_value=c)
    else:
        X["active_days"] = X["active_weeks_frac"] = X["longest_gap"] = np.nan

    last_seen = d.groupby("key").date.max().reindex(keys)
    since = last_seen.fillna(s.date_registration.fillna(0))
    X["days_since_active"] = (c - since).clip(lower=0)

    # Time management: assessments due before the cutoff, and how early work came in.
    a = data.assessments.merge(s[["code_module", "code_presentation"]].reset_index(),
                               on=["code_module", "code_presentation"])
    due = a[a.date < c]
    sub = data.submissions[data.submissions.key.isin(keys)]
    sub = sub[(sub.date_submitted < c) | (sub.is_banked == 1)]
    sub = sub.merge(data.assessments[["id_assessment", "date"]], on="id_assessment")
    n_due = due.groupby("key").size().reindex(keys, fill_value=0)
    done = due.merge(sub[["key", "id_assessment"]], on=["key", "id_assessment"])
    n_done = done.groupby("key").size().reindex(keys, fill_value=0)
    X["assessments_due"] = n_due
    X["missed_frac"] = ((n_due - n_done) / n_due).where(n_due > 0)
    timed = sub[sub.is_banked == 0]
    X["days_early"] = (timed.date - timed.date_submitted).groupby(timed.key).mean().reindex(keys)
    X["late_count"] = (timed.date_submitted > timed.date).groupby(timed.key).sum().reindex(keys, fill_value=0)
    X["registered_early"] = -s.date_registration

    # Where the student sits in this week's cohort of the same presentation, so
    # modules with heavier or lighter VLE use are comparable.
    for col in ["clicks", "active_days", "last_week_clicks"]:
        if X[col].notna().any():
            X[f"{col}_pct"] = X[col].groupby(s.presentation).rank(pct=True)
        else:
            X[f"{col}_pct"] = np.nan
    return X.astype(float)
