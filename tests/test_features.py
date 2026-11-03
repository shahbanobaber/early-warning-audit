"""Features must only see the past, and the population must only hold students a warning can reach."""

import numpy as np
import pandas as pd

from ewaudit.data import GROUPS, Data
from ewaudit.features import features, population


def _toy(seed=0, n=12):
    rng = np.random.default_rng(seed)
    students = pd.DataFrame({
        "code_module": "AAA", "code_presentation": np.where(np.arange(n) < n // 2, "2013J", "2014J"),
        "id_student": np.arange(n), "date_registration": rng.integers(-60, -1, n).astype(float),
        "date_unregistration": np.nan, "at_risk": rng.integers(0, 2, n),
    })
    students["presentation"] = students.code_module + " " + students.code_presentation
    students.loc[0, "date_unregistration"] = 10.0      # leaves in week 2
    students.loc[1, "date_registration"] = 40.0        # joins late, in week 6
    students.index = pd.RangeIndex(n, name="key")
    assessments = pd.DataFrame({"code_module": "AAA", "code_presentation": ["2013J", "2013J", "2014J", "2014J"],
                                "id_assessment": [1, 2, 3, 4], "assessment_type": "TMA", "date": [20, 50, 20, 50]})
    rows = []
    for k in range(n):
        for day in rng.choice(np.arange(-20, 120), 40, replace=False):
            rows.append((k, day, rng.integers(0, len(GROUPS)), rng.integers(1, 20)))
    daily = pd.DataFrame(rows, columns=["key", "date", "group", "clicks"])
    sub = []
    for k in range(n):
        first = 1 if students.code_presentation[k] == "2013J" else 3
        for a, due in [(first, 20), (first + 1, 50)]:
            if rng.random() < 0.8:
                sub.append((k, a, due + int(rng.integers(-10, 8)), 0))
    submissions = pd.DataFrame(sub, columns=["key", "id_assessment", "date_submitted", "is_banked"])
    return Data(students, assessments, submissions, daily)


def test_features_ignore_everything_after_the_cutoff():
    data = _toy()
    for week in (0, 1, 3, 4, 8):
        c = 7 * week
        before = features(data, week)
        # Replace everything from the cutoff on with different, random data.
        rng = np.random.default_rng(week + 100)
        late = data.daily.date >= c
        daily = data.daily.copy()
        daily.loc[late, "clicks"] = rng.integers(1, 500, late.sum())
        daily = pd.concat([daily[~late | (rng.random(len(daily)) < 0.5)],
                           pd.DataFrame({"key": [2, 3], "date": [c, c + 5], "group": [1, 2], "clicks": [999, 999]})])
        sub = data.submissions.copy()
        sub.loc[sub.date_submitted >= c, "date_submitted"] += 3
        after = features(Data(data.students, data.assessments, sub, daily.reset_index(drop=True)), week)
        pd.testing.assert_frame_equal(before, after)


def test_population_drops_students_who_left_or_have_not_joined():
    data = _toy()
    assert 0 in population(data.students, 1).index        # leaves on day 10, still in at the end of week 1
    assert 0 not in population(data.students, 2).index    # gone by day 14
    assert 1 not in population(data.students, 5).index    # registers on day 40
    assert 1 in population(data.students, 6).index


def test_feature_values_on_a_hand_built_student():
    students = pd.DataFrame({"code_module": ["AAA"], "code_presentation": ["2014J"], "id_student": [1],
                             "date_registration": [-30.0], "date_unregistration": [np.nan], "at_risk": [0],
                             "presentation": ["AAA 2014J"]}, index=pd.RangeIndex(1, name="key"))
    assessments = pd.DataFrame({"code_module": "AAA", "code_presentation": "2014J", "id_assessment": [1, 2],
                                "assessment_type": "TMA", "date": [10, 25]})
    # Active on days -3, 0, 1, 9 and 12. Forum on day 1. Submits assessment 1 two days early, misses 2.
    daily = pd.DataFrame({"key": 0, "date": [-3, 0, 1, 9, 12], "group": [0, 0, GROUPS.index("forum"), 0, 0],
                          "clicks": [5, 3, 2, 4, 6]})
    sub = pd.DataFrame({"key": [0], "id_assessment": [1], "date_submitted": [8], "is_banked": [0]})
    X = features(Data(students, assessments, sub, daily), week=4).iloc[0]   # cutoff day 28
    assert X.clicks == np.log1p(20) and X.prestart_clicks == np.log1p(5)
    assert X.active_days == 4 and X.forum_any == 1
    assert X.days_since_active == 28 - 12
    assert X.longest_gap == 28 - 1 - 12                     # days 13..27 inactive
    assert X.active_weeks_frac == 2 / 4                     # weeks 0 and 1 of 0..3
    assert X.assessments_due == 2 and X.missed_frac == 0.5 and X.days_early == 2


def test_split_trains_on_the_past_and_sets_the_future_aside():
    from ewaudit.data import assign_split

    s = pd.DataFrame({"code_presentation": ["2013B", "2013J", "2014B", "2014J"]})
    assert assign_split(s, "2014B").split.tolist() == ["train", "train", "test", "later"]
    assert assign_split(s, "2014J").split.tolist() == ["train", "train", "train", "test"]
