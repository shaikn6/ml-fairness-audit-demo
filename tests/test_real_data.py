"""The real UCI Adult path must load, keep its schema, and surface real bias."""
import numpy as np
import pandas as pd
import pytest

import app
from audit import real_data
from audit.real_data import (
    ADULT_FEATURE_COLUMNS,
    ADULT_LABEL_COL,
    INTERSECTION_COL,
    _shape_adult,
    add_intersection,
    load_adult,
)


def _adult_or_skip():
    try:
        return load_adult()
    except Exception as exc:  # noqa: BLE001 — offline with no bundled copy
        pytest.skip(f"Adult dataset unavailable: {exc}")


def test_adult_loads_in_audit_schema():
    df = _adult_or_skip()
    for col in [*ADULT_FEATURE_COLUMNS, "sex", "race", ADULT_LABEL_COL]:
        assert col in df.columns
    assert set(df[ADULT_LABEL_COL].unique()) <= {0, 1}
    assert len(df) > 40_000
    assert set(df["sex"].unique()) == {"Male", "Female"}


def test_intersection_combines_sex_and_race():
    df = _adult_or_skip()
    di = add_intersection(df)
    assert INTERSECTION_COL in di.columns
    # Up to 2 sexes × 5 races, each label a "sex / race" pair.
    assert di[INTERSECTION_COL].nunique() == df["sex"].nunique() * df["race"].nunique()
    assert di[INTERSECTION_COL].str.contains(" / ").all()


def test_adult_sex_audit_flags_real_bias():
    _adult_or_skip()
    summary, table, _, _ = app.audit_sample("UCI Adult Income (real)", "sex")
    assert "FAIL" in summary
    female = table[table["Group"] == "Female"].iloc[0]
    # Women fall far below the 4/5ths floor in the raw Adult data.
    assert float(female["Disparate Impact (95% CI)"].split()[0]) < 0.80


def test_benchmark_improves_disparate_impact():
    _adult_or_skip()
    md = app.benchmark_sample("UCI Adult Income (real)", "sex")
    assert "Baseline" in md
    assert "Threshold" in md
    # Both mitigation rows must raise DIR above the baseline row.
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and "↑" not in ln and "---" not in ln]
    dirs = [float(r.split("|")[2]) for r in rows]
    assert dirs[0] < 0.80  # baseline fails
    assert all(d > dirs[0] for d in dirs[1:])  # every method improves it


def test_benchmark_runs_on_race_and_intersection():
    _adult_or_skip()
    # The smallest sex × race group has ~155 rows; ThresholdOptimizer must cope.
    for attr in ("race", INTERSECTION_COL):
        md = app.benchmark_sample("UCI Adult Income (real)", attr)
        assert "Baseline" in md and "Threshold" in md


def test_shape_adult_handles_trailing_period_target():
    data = pd.DataFrame(
        {
            "age": [39, 50, 23],
            "education-num": [13, 13, 9],
            "capital-gain": [2174, 0, 0],
            "capital-loss": [0, 0, 0],
            "hours-per-week": [40, 13, 40],
            "sex": ["Male", "Female", "Male"],
            "race": ["White", "White", "Black"],
        }
    )
    target = pd.Series([">50K.", "<=50K.", ">50K"])  # test set uses trailing periods
    df = _shape_adult(data, target)
    assert df[ADULT_LABEL_COL].tolist() == [1, 0, 1]
    assert list(df.columns) == [*ADULT_FEATURE_COLUMNS, "sex", "race", ADULT_LABEL_COL]


def test_load_sample_falls_back_to_synthetic_with_note(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(app, "load_adult", boom)
    df, label_col, note = app._load_sample("UCI Adult Income (real)")
    assert label_col == "loan_approved"  # synthetic fallback
    assert "race" in df.columns
    assert "Could not load" in note and "⚠️" in note
