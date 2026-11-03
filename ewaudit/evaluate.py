"""Models, and the numbers a decision needs: who gets flagged under a fixed contact budget."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "boosting": "gradient boosting",
    "logistic": "logistic regression",
    "quiet": "inactivity rule",
}


def fit(kind: str, X: pd.DataFrame, y: pd.Series):
    if kind == "logistic":
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          LogisticRegression(max_iter=2000))
    elif kind == "boosting":
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0,
                                           early_stopping=False, random_state=0)
    else:
        raise ValueError(kind)
    return m.fit(X, y)


def score(kind: str, model, X: pd.DataFrame) -> np.ndarray:
    """Higher = more at risk. The inactivity rule needs no model: longest-quiet first, fewest clicks breaking ties."""
    if kind == "quiet":
        return X["days_since_active"].fillna(0).to_numpy() - 0.01 * X["clicks_pct"].fillna(0).to_numpy()
    return model.predict_proba(X)[:, 1]


def flag(s: np.ndarray, budget: float) -> np.ndarray:
    """The top `budget` fraction by score. Ties at the cut are broken by order, so exactly k are flagged."""
    k = int(round(budget * len(s)))
    out = np.zeros(len(s), bool)
    out[np.argsort(-s, kind="stable")[:k]] = True
    return out


def budget_metrics(y: np.ndarray, flagged: np.ndarray) -> dict:
    tp = int((flagged & (y == 1)).sum())
    return dict(flagged=int(flagged.sum()), true_pos=tp,
                precision=tp / max(1, flagged.sum()), recall=tp / max(1, (y == 1).sum()))


def bootstrap(y: np.ndarray, s: np.ndarray, budget: float, rng, n: int = 200) -> dict:
    """95% intervals for AUC, precision and recall at the budget, resampling students."""
    out = {"auc": [], "precision": [], "recall": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        out["auc"].append(roc_auc_score(y[i], s[i]))
        m = budget_metrics(y[i], flag(s[i], budget))
        out["precision"].append(m["precision"])
        out["recall"].append(m["recall"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}
