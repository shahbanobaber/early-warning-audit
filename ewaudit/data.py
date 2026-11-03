"""Load OULAD and reduce the 10.6M-row click log to a compact daily table.

OULAD: Kuzilek, Hlosta and Zdrahal (2017), Scientific Data 4:170171, CC BY 4.0.
The original download link now returns 404; the UCI copy is identical
(doi:10.24432/C5KK69).
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
TABLES = ["courses", "assessments", "vle", "studentInfo", "studentRegistration", "studentAssessment", "studentVle"]
KEYS = ["code_module", "code_presentation", "id_student"]

# Activity types grouped by what they could say about self-regulation.
ACTIVITY = {
    "forumng": "forum",                                                   # help-seeking
    "oucollaborate": "live", "ouelluminate": "live",                      # live tutorials
    "quiz": "quiz", "externalquiz": "quiz", "questionnaire": "quiz",      # self-testing
    "homepage": "homepage",
}
GROUPS = ["content", "forum", "live", "quiz", "homepage"]                  # anything else is content

ORDER = ["2013B", "2013J", "2014B", "2014J"]   # presentations in time order (B starts in February, J in October)
TEST_PRESENTATION = "2014J"                    # the last one; train on everything before it


def download(raw: Path) -> Path:
    """Fetch and unpack the CSVs into raw/oulad (about 45 MB download, 460 MB unpacked)."""
    out = raw / "oulad"
    if all((out / f"{t}.csv").exists() for t in TABLES):
        return out
    raw.mkdir(parents=True, exist_ok=True)
    zpath = raw / "oulad.zip"
    if not zpath.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    return out


@dataclass
class Data:
    students: pd.DataFrame      # one row per student-presentation, index = key
    assessments: pd.DataFrame   # TMA and CMA deadlines, exams dropped
    submissions: pd.DataFrame   # key, id_assessment, date_submitted, is_banked
    daily: pd.DataFrame         # key, date, group (int code into GROUPS), clicks


def _students(src: Path) -> pd.DataFrame:
    info = pd.read_csv(src / "studentInfo.csv", na_values="?")
    reg = pd.read_csv(src / "studentRegistration.csv", na_values="?")
    courses = pd.read_csv(src / "courses.csv")
    s = info.merge(reg, on=KEYS, validate="1:1").merge(courses, on=["code_module", "code_presentation"])
    s = s.rename(columns={"module_presentation_length": "length"})
    s["presentation"] = s.code_module + " " + s.code_presentation
    s["at_risk"] = s.final_result.isin(["Fail", "Withdrawn"]).astype(int)
    s.index = pd.RangeIndex(len(s), name="key")
    return s


def assign_split(students: pd.DataFrame, test: str) -> pd.DataFrame:
    """Test on one presentation, train on every earlier one, and set later ones aside."""
    rank = students.code_presentation.map(ORDER.index)
    t = ORDER.index(test)
    out = students.copy()
    out["split"] = np.select([rank == t, rank < t], ["test", "train"], "later")
    return out


def _daily(src: Path, students: pd.DataFrame, chunk: int = 1_000_000) -> pd.DataFrame:
    """Sum clicks per student, day and activity group, reading the log in chunks to keep memory low."""
    vle = pd.read_csv(src / "vle.csv", usecols=["id_site", "activity_type"])
    site_group = pd.Series(vle.activity_type.map(ACTIVITY).fillna("content").map(GROUPS.index).astype("int8").values,
                           index=vle.id_site.values)
    keymap = students[KEYS].reset_index()
    parts = []
    dtypes = {"id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int32"}
    for c in pd.read_csv(src / "studentVle.csv", dtype=dtypes, chunksize=chunk):
        c = c.merge(keymap, on=KEYS, how="inner")
        c["group"] = site_group.reindex(c.id_site.values).values
        parts.append(c.groupby(["key", "date", "group"], sort=False).sum_click.sum())
    daily = pd.concat(parts).groupby(level=[0, 1, 2]).sum().rename("clicks").reset_index()
    return daily.astype({"key": "int32", "date": "int16", "group": "int8", "clicks": "int32"})


def load(raw: Path, cache: Path, test: str = TEST_PRESENTATION) -> Data:
    """Read everything, building the daily click table on first use and caching it."""
    src = download(raw)
    students = assign_split(_students(src), test)
    assess = pd.read_csv(src / "assessments.csv", na_values="?")
    assess = assess[assess.assessment_type != "Exam"].dropna(subset=["date"])
    sub = pd.read_csv(src / "studentAssessment.csv", na_values="?")
    sub = sub.merge(assess[["id_assessment", "code_module", "code_presentation"]], on="id_assessment")
    sub = sub.merge(students[KEYS].reset_index(), on=KEYS)[["key", "id_assessment", "date_submitted", "is_banked"]]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "daily.pkl"
    if path.exists():
        daily = pd.read_pickle(path)
    else:
        daily = _daily(src, students)
        daily.to_pickle(path)
    return Data(students, assess, sub, daily)
