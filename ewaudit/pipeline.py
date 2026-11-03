"""Fit a warning model at the end of each week, then score it as a decision and audit who it misses.

Train on every presentation before the test one (by default 2013B, 2013J
and 2014B, testing on 2014J), as a deployed system would be used. At each
week only students still registered are scored; the ones who already left
are counted separately, because no warning can reach them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from . import audit
from .data import Data
from .evaluate import MODELS, bootstrap, budget_metrics, fit, flag, score
from .features import features, population

FAMILIES = {
    "how much they clicked": ["clicks", "content_clicks", "homepage_clicks", "clicks_pct"],
    "how recently and regularly": ["last_week_clicks", "last_week_clicks_pct", "active_days", "active_days_pct",
                                   "active_weeks_frac", "longest_gap", "days_since_active"],
    "help-seeking (forum, live tutorials)": ["forum_clicks", "forum_any", "live_clicks"],
    "self-testing (quizzes)": ["quiz_clicks"],
    "assessment timing": ["assessments_due", "missed_frac", "days_early", "late_count"],
    "planning (early registration, pre-start activity)": ["registered_early", "prestart_clicks"],
}
MAIN = "boosting"


def _family_importance(model, X: pd.DataFrame, y: np.ndarray, rng, repeats: int = 5) -> dict[str, list[float]]:
    """Drop in test AUC when one family of traces is shuffled across students (all its columns together)."""
    base = roc_auc_score(y, score(MAIN, model, X))
    out = {}
    for fam, cols in FAMILIES.items():
        cols = [c for c in cols if c in X]
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[idx]
            drops.append(base - roc_auc_score(y, score(MAIN, model, Xp)))
        out[fam] = [float(np.mean(drops)), float(np.std(drops))]
    return out


def _profile(pop: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, flagged: np.ndarray) -> dict[str, dict]:
    """What the students the warning missed looked like in the traces, next to those it caught and those who passed."""
    res = pop.final_result.to_numpy()
    groups = {
        "passed": y == 0,
        "at risk, flagged": (y == 1) & flagged,
        "failed, not flagged": (res == "Fail") & ~flagged,
        "withdrew later, not flagged": (res == "Withdrawn") & ~flagged,
    }
    out = {}
    for name, m in groups.items():
        Xm = X[m]
        out[name] = dict(n=int(m.sum()), active_last_week=float((Xm.last_week_clicks > 0).mean()),
                         median_clicks_pct=float(Xm.clicks_pct.median()),
                         missed_a_deadline=float((Xm.missed_frac.dropna() > 0).mean()),
                         median_days_early=float(Xm.days_early.median()))
    return out


def run(data: Data, out: Path, weeks=range(0, 21), budgets=(0.1, 0.2, 0.3), main_budget: float = 0.2,
        audit_weeks=(4, 8), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    s = data.students
    cohort = s[s.split == "test"]
    test = str(cohort.code_presentation.iloc[0])
    eventual = int(cohort.at_risk.sum())
    G = audit.groups(s)
    by_week, audits = [], {}
    for w in weeks:
        pop = population(s, w)
        pop = pop[pop.split.isin(["train", "test"])]
        X = features(data, w, pop.index)
        tr, te = pop.split == "train", pop.split == "test"
        X = X[[c for c in X if X.loc[tr, c].nunique() > 1]]  # week 0 has no in-course activity yet
        y_tr, y_te = pop.at_risk[tr].to_numpy(), pop.at_risk[te].to_numpy()
        gone = int((cohort.at_risk.eq(1) & cohort.date_unregistration.lt(7 * w)).sum())
        row = dict(week=w, day=7 * w, n_test=int(te.sum()), n_train=int(tr.sum()), base_rate=float(y_te.mean()),
                   eventual_at_risk=eventual, already_gone=gone, models={})
        fitted = {}
        with threadpool_limits(1):
            for kind in MODELS:
                model = None if kind == "quiet" else fit(kind, X[tr], y_tr)
                fitted[kind] = model
                sc = score(kind, model, X[te])
                m = dict(auc=float(roc_auc_score(y_te, sc)), budgets={})
                for b in budgets:
                    bm = budget_metrics(y_te, flag(sc, b))
                    m["budgets"][str(b)] = dict(bm, reached_share=bm["true_pos"] / eventual)
                if kind == MAIN:
                    m["ci"] = bootstrap(y_te, sc, main_budget, rng)
                    f = flag(sc, main_budget)
                    res = pop.final_result[te].to_numpy()
                    m["recall_by_outcome"] = {o: float(f[res == o].mean()) for o in ("Withdrawn", "Fail")}
                row["models"][kind] = m

            if w in audit_weeks:
                Xte, pte = X[te], pop[te]
                sc = score(MAIN, fitted[MAIN], Xte)
                flagged = flag(sc, main_budget)
                per_module = {mod: float(roc_auc_score(y_te[(pte.code_module == mod).to_numpy()],
                                                       sc[(pte.code_module == mod).to_numpy()]))
                              for mod in sorted(pte.code_module.unique())}
                lr = flag(score("logistic", fitted["logistic"], Xte), main_budget)
                card, card_gaps = audit.scorecard(G.loc[pop.index], pop.at_risk.to_numpy(), X, pop.presentation, rng)
                audits[w] = dict(
                    week=w, budget=main_budget,
                    subgroups=audit.subgroup_errors(G.loc[pte.index], y_te, flagged, rng),
                    subgroups_logistic=audit.subgroup_errors(G.loc[pte.index], y_te, lr, rng, n_boot=200),
                    missing_traces=audit.missing_traces(G.loc[pte.index], y_te, Xte, rng),
                    missed_profile=_profile(pte, Xte, y_te, flagged),
                    scorecard=card, scorecard_gaps=card_gaps, scorecard_runs=int(pop.presentation.nunique()),
                    calibration=audit.calibration(G.loc[pte.index], y_te, sc, rng),
                    auc_by_module=per_module,
                    importance=_family_importance(fitted[MAIN], Xte, y_te, rng),
                )
        by_week.append(row)
        mm = row["models"][MAIN]
        print(f"week {w:2d}: {row['n_test']} still enrolled, {gone}/{eventual} at-risk already gone, "
              f"AUC {mm['auc']:.3f}, precision@{main_budget:.0%} {mm['budgets'][str(main_budget)]['precision']:.3f}",
              flush=True)
    result = dict(settings=dict(weeks=list(weeks), budgets=list(budgets), main_budget=main_budget, main=MAIN,
                                audit_weeks=list(audit_weeks), test_presentation=test,
                                train_presentations=sorted(s.loc[s.split == "train", "code_presentation"].unique()),
                                seed=seed),
                  by_week=by_week, audits={str(k): v for k, v in audits.items()})
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    return result
