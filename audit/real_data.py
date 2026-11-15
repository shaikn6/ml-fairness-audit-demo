"""
Real-world fairness dataset loader — UCI Adult (Census Income).

Loads the well-known UCI Adult Income dataset (predict whether a person earns
> $50K/year) and reshapes it into the schema the audit engine expects: numeric
feature columns, categorical protected attributes (``sex`` and ``race``), and a
binary ``income_over_50k`` label. Adult is a canonical fairness benchmark with a
genuine, documented income gap by sex and race, so the demo opens on a real,
recognizable bias case rather than synthetic data.

Fetched via ``fairlearn.datasets.fetch_adult`` (already a dependency) and cached
to disk so repeat launches do not re-download. If no network is available at
runtime, the caller falls back to the synthetic generator.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Numeric predictors kept for the scorecard. fnlwgt (a census sampling weight)
# and one-hot-heavy categoricals are intentionally excluded; these five are the
# standard Adult numeric features and are what _train_and_predict will pick up
# via its numeric-column fallback.
ADULT_FEATURE_COLUMNS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

ADULT_LABEL_COL = "income_over_50k"
ADULT_PROTECTED = ["sex", "race"]
INTERSECTION_COL = "sex × race"

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Shipped with the Space so a real audit runs even with no network at runtime.
_BUNDLED = _REPO_ROOT / "data" / "adult_income.csv"
_DEFAULT_CACHE = _REPO_ROOT / ".cache"
_CACHE_FILE = "adult_income.csv"


def _cache_path(cache_dir: str | os.PathLike | None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
    return base / _CACHE_FILE


def _shape_adult(data: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Reshape the raw fetch into the audit schema (features + sex + race + label)."""
    df = pd.DataFrame()
    for col in ADULT_FEATURE_COLUMNS:
        df[col] = pd.to_numeric(data[col], errors="coerce")
    df["sex"] = data["sex"].astype(str).str.strip()
    df["race"] = data["race"].astype(str).str.strip()
    # Target arrives as ">50K" / "<=50K" (sometimes with a trailing period).
    label = target.astype(str).str.replace(".", "", regex=False).str.strip()
    df[ADULT_LABEL_COL] = (label == ">50K").astype(int)
    return df.dropna().reset_index(drop=True)


def load_adult(cache_dir: str | os.PathLike | None = None) -> pd.DataFrame:
    """Return the Adult Income dataset in audit schema, using an on-disk cache.

    Raises on failure (e.g. no network and no cache) so callers can fall back to
    the synthetic generator.
    """
    if cache_dir is None and _BUNDLED.exists():
        return pd.read_csv(_BUNDLED)

    path = _cache_path(cache_dir)
    if path.exists():
        return pd.read_csv(path)

    # fetch_adult is the fairlearn convenience loader; it pulls from OpenML.
    from fairlearn.datasets import fetch_adult

    bunch = fetch_adult(as_frame=True)
    df = _shape_adult(bunch.data, bunch.target)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def add_intersection(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``sex × race`` column combining the two protected attributes.

    Intersectional analysis surfaces bias that single-attribute audits miss
    (e.g. a group disadvantaged on sex *and* race together).
    """
    out = df.copy()
    out[INTERSECTION_COL] = (
        out["sex"].astype(str) + " / " + out["race"].astype(str)
    )
    return out
