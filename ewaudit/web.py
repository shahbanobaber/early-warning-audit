"""Slim the two metrics.json files into web/data.json for the explorer page. No student-level data leaves here."""

from __future__ import annotations

import json
from pathlib import Path


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _year(r: dict) -> dict:
    st = r["settings"]
    weeks = []
    for row in r["by_week"]:
        models = {}
        for kind, m in row["models"].items():
            budgets = {b: [_r(v["precision"]), _r(v["recall"]), _r(v["reached_share"])]
                       for b, v in m["budgets"].items()}
            out = {"auc": _r(m["auc"]), "b": budgets}
            if "recall_by_outcome" in m:
                by = m["recall_by_outcome"]
                out["fail"], out["withdraw"] = _r(by["Fail"]), _r(by["Withdrawn"])
            models[kind] = out
        weeks.append({"week": row["week"], "n": row["n_test"], "base": _r(row["base_rate"]),
                      "gone": _r(row["already_gone"] / row["eventual_at_risk"]), "mark": _r(row.get("with_a_mark", 0)),
                      "models": models})
    grid = {b: {kind: {k: [_r(c["reached_share"]), _r(c["contacts_share"]), _r(c["precision"])]
                       for k, c in by_kind.items()} for kind, by_kind in per_b.items()}
            for b, per_b in r["campaign_grid"].items()}
    any_campaign = next(iter(r["campaigns"][st["main"]].values()))
    audits = {}
    for w, a in r["audits"].items():
        audits[w] = {
            "subgroups": [{k: (_r(v) if isinstance(v, float) else ([_r(x) for x in v] if isinstance(v, list) else v))
                           for k, v in g.items() if k in ("dimension", "group", "n", "at_risk", "flagged", "miss",
                                                          "miss_ci", "precision", "precision_ci", "miss_gap",
                                                          "miss_gap_ci")} for g in a["subgroups"]],
            "calibration": [{"dimension": c["dimension"], "group": c["group"], "n": c["n"],
                             "observed": _r(c["observed"]), "predicted": _r(c["predicted"])} for c in a["calibration"]],
            "traces": [{"trace": t["trace"], "dimension": t["dimension"], "group": t["group"], "n": t["n"],
                        "share_absent": _r(t["share_absent"]), "risk_absent": _r(t["risk_absent"]),
                        "risk_present": _r(t["risk_present"]), "difference": _r(t["difference"]),
                        "difference_ci": [_r(x) for x in t["difference_ci"]], "gap": _r(t["gap"]),
                        "gap_ci": [_r(x) for x in t["gap_ci"]]} for t in a["missing_traces"]],
            "profile": {k: {kk: (_r(vv) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                        for k, v in a["missed_profile"].items()},
            "scorecard": [{k: (_r(v) if isinstance(v, float) else v) for k, v in c.items()
                           if k in ("trace", "skill", "difference", "runs", "runs_positive", "run_min", "run_max")}
                          for c in a["scorecard"]],
        }
    return {"test": st["test_presentation"], "train": st["train_presentations"],
            "cohort": round(any_campaign["contacts"] / any_campaign["contacts_share"]),
            "eventual": r["by_week"][0]["eventual_at_risk"], "marking_lag": st.get("marking_lag_days"),
            "weeks": weeks, "grid": grid, "audits": audits}


def export(results: list[Path], out: Path) -> Path:
    data = {"years": {}}
    for path in results:
        y = _year(json.loads(Path(path).read_text()))
        data["years"][y["test"]] = y
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    return out
