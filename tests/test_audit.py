"""The missing-trace and calibration checks, on groups built so the right answer is known."""

import numpy as np
import pandas as pd

from ewaudit.audit import calibration, missing_traces


def _groups(n_declared, n_none):
    return pd.DataFrame({"deprivation": "0-20%", "education": "A level",
                         "disability": ["declared"] * n_declared + ["none declared"] * n_none})


def test_missing_trace_effect_and_gap_per_group():
    # Declared: quiet students 100% at risk, active ones 50% -> silence adds 50 points.
    # None declared: quiet 100%, active 0% -> silence adds 100 points. Gap for declared = -50.
    quiet_d, active_d = [1] * 20, [1] * 20 + [0] * 20
    quiet_n, active_n = [1] * 20, [0] * 40
    y = np.array(quiet_d + active_d + quiet_n + active_n)
    last_week = np.array([0] * 20 + [5] * 40 + [0] * 20 + [5] * 40, float)
    X = pd.DataFrame({"forum_any": np.ones(len(y)), "last_week_clicks": last_week})
    rows = {r["group"]: r for r in missing_traces(_groups(60, 60), y, X, np.random.default_rng(0), n_boot=100)
            if r["trace"] == "no activity in the last week" and r["dimension"] == "disability"}
    d, n = rows["declared"], rows["none declared"]
    assert np.isclose(d["share_absent"], 1 / 3)
    assert np.isclose(d["risk_absent"], 1) and np.isclose(d["risk_present"], 0.5)
    assert np.isclose(d["difference"], 0.5) and np.isclose(n["difference"], 1.0)
    assert np.isclose(d["gap"], -0.5) and np.isclose(n["gap"], 0.5)
    assert d["gap_ci"][0] <= -0.5 <= d["gap_ci"][1]
    # Everyone used the forum, so that trace has no "absent" students and is skipped.
    assert not any(r["trace"] == "never used the forum"
                   for r in missing_traces(_groups(60, 60), y, X, np.random.default_rng(0), n_boot=10))


def test_calibration_reports_observed_minus_predicted():
    y = np.array([1] * 10 + [0] * 10 + [1] * 5 + [0] * 15)
    prob = np.array([0.3] * 20 + [0.25] * 20)
    rows = {(r["dimension"], r["group"]): r for r in calibration(_groups(20, 20), y, prob, np.random.default_rng(0),
                                                                n_boot=100)}
    assert np.isclose(rows["disability", "declared"]["under"], 0.5 - 0.3)
    assert np.isclose(rows["disability", "none declared"]["under"], 0.25 - 0.25)
    assert np.isclose(rows["all", "everyone"]["observed"], 15 / 40)
    assert np.isclose(rows["all", "everyone"]["predicted"], 0.275)


def test_scorecard_counts_course_runs_and_finds_the_gap():
    from ewaudit.audit import scorecard

    rng = np.random.default_rng(1)
    n = 400
    pres = pd.Series(np.repeat(["A 2013J", "B 2013J"], n // 2))
    quiet = rng.random(n) < 0.3
    declared = np.arange(n) % 4 == 0
    # Quiet adds 40 points for most students but only 10 for the declared group.
    p = np.where(declared, 0.5 + 0.1 * quiet, 0.2 + 0.4 * quiet)
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"last_week_clicks": np.where(quiet, 0.0, 3.0), "missed_frac": np.nan, "forum_any": 1.0,
                      "quiz_clicks": 1.0, "prestart_clicks": 1.0})
    G = pd.DataFrame({"deprivation": "0-20%", "education": "A level",
                      "disability": np.where(declared, "declared", "none declared")})
    rows, _ = scorecard(G, y, X, pres, rng, n_boot=200)
    card = {r["trace"]: r for r in rows}
    quiet_row = card["no activity in the last week"]
    assert quiet_row["runs"] == 2 and quiet_row["runs_positive"] == 2
    assert quiet_row["worst_group"].startswith("disability") and quiet_row["worst_gap"] < 0
    assert "missed an assessment deadline" not in card   # no deadlines due, so it doesn't apply to anyone
