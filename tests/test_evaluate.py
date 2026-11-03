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
