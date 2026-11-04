import numpy as np
import pandas as pd

from ewaudit.audit import subgroup_errors
from ewaudit.evaluate import budget_metrics, flag


def test_budget_flags_exactly_k_even_with_ties():
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    f = flag(s, 0.3)
    assert f.sum() == 3 and f[0]
    m = budget_metrics(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1]), f)
    assert m["flagged"] == 3 and m["true_pos"] == 2
    assert np.isclose(m["precision"], 2 / 3) and np.isclose(m["recall"], 2 / 3)


def test_subgroup_miss_rates_are_counted_per_group():
    # Group "declared": 4 at risk, 1 flagged -> misses 3 of 4. Group "none declared": 4 at risk, 3 flagged -> 1 of 4.
    G = pd.DataFrame({"deprivation": "0-20%", "education": "A level",
                      "disability": ["declared"] * 6 + ["none declared"] * 6})
    y = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    f = np.array([1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0], bool)
    rows = {r["group"]: r for r in subgroup_errors(G, y, f, np.random.default_rng(0), n_boot=50)
            if r["dimension"] == "disability"}
    assert np.isclose(rows["declared"]["miss"], 0.75) and np.isclose(rows["none declared"]["miss"], 0.25)
    assert np.isclose(rows["declared"]["precision"], 0.5) and np.isclose(rows["none declared"]["precision"], 1.0)
    assert np.isclose(rows["declared"]["miss_gap"], 0.5)


def test_campaign_skips_students_already_contacted():
    from ewaudit.pipeline import campaign

    keys = np.arange(10)
    ranked = {0: (keys, np.linspace(1, 0, 10)), 4: (keys[:8], np.linspace(1, 0, 8))}  # 2 have left by week 4
    at_risk = pd.Series([1, 0, 1, 1, 0, 0, 1, 0, 0, 1], index=keys)
    one = campaign(ranked, [0], 0.2, at_risk)
    assert one["contacts"] == 2 and one["reached"] == 1             # students 0 and 1
    two = campaign(ranked, [0, 4], 0.2, at_risk)
    # Week 4: 20% of 8 enrolled is 2 contacts; 0 and 1 are skipped, so 2 and 3 are next.
    assert two["contacts"] == 4 and two["reached"] == 3 and np.isclose(two["reached_share"], 3 / 5)
