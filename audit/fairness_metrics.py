"""
Fairness metrics computation module.

Implements three core fairness metrics required for ECOA / Fair Lending audits:
  - Demographic Parity Difference (DPD)
  - Equalized Odds Difference (EOD)
  - Disparate Impact Ratio (DIR) — the ECOA 4/5ths rule threshold
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _approval_rate(y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred))


def _tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """True positive rate and false positive rate for a binary group."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr


def demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    reference_group: str | None = None,
) -> dict[str, float]:
    """
    Compute pairwise Demographic Parity Difference vs. the reference group.

    DPD = |P(ŷ=1 | A=a) - P(ŷ=1 | A=reference)|

    Args:
        y_pred: Binary predictions (0/1).
        sensitive: Protected attribute values (same length as y_pred).
        reference_group: The majority/reference group. If None, uses the group
                         with the highest approval rate.

    Returns:
        Dict mapping each group to its DPD relative to the reference group.
    """
    groups = np.unique(sensitive)
    rates: dict[str, float] = {}
    for g in groups:
        mask = sensitive == g
        rates[g] = _approval_rate(y_pred[mask])

    if reference_group is None:
        reference_group = max(rates, key=lambda k: rates[k])

    ref_rate = rates[reference_group]
    dpd: dict[str, float] = {}
    for g in groups:
        dpd[g] = round(abs(rates[g] - ref_rate), 4)

    return dpd


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    reference_group: str | None = None,
) -> dict[str, float]:
    """
    Compute Equalized Odds Difference per group vs. reference group.

    EOD = max(|TPR_a - TPR_ref|, |FPR_a - FPR_ref|)

    Returns:
        Dict mapping each group to its EOD relative to the reference group.
    """
    groups = np.unique(sensitive)
    tprs: dict[str, float] = {}
    fprs: dict[str, float] = {}

    for g in groups:
        mask = sensitive == g
        tpr, fpr = _tpr_fpr(y_true[mask], y_pred[mask])
        tprs[g] = tpr
        fprs[g] = fpr

    if reference_group is None:
        # Use highest-TPR group as reference
        reference_group = max(tprs, key=lambda k: tprs[k])

    eod: dict[str, float] = {}
    for g in groups:
        eod[g] = round(
            max(
                abs(tprs[g] - tprs[reference_group]),
                abs(fprs[g] - fprs[reference_group]),
            ),
            4,
        )
    return eod


def disparate_impact_ratio(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    reference_group: str | None = None,
) -> dict[str, float]:
    """
    Compute Disparate Impact Ratio per group.

    DIR = P(ŷ=1 | A=minority) / P(ŷ=1 | A=majority)

    ECOA / Fair Lending threshold: DIR < 0.8 constitutes a potential violation
    (the "4/5ths rule" from EEOC Uniform Guidelines).

    Returns:
        Dict mapping each group to its DIR relative to the reference (majority) group.
    """
    groups = np.unique(sensitive)
    rates: dict[str, float] = {}
    for g in groups:
        mask = sensitive == g
        rates[g] = _approval_rate(y_pred[mask])

    if reference_group is None:
        reference_group = max(rates, key=lambda k: rates[k])

    ref_rate = rates[reference_group]
    dir_scores: dict[str, float] = {}
    for g in groups:
        if ref_rate == 0:
            dir_scores[g] = float("nan")
        else:
            dir_scores[g] = round(rates[g] / ref_rate, 4)

    return dir_scores


def compute_all_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive: np.ndarray | pd.Series,
    reference_group: str | None = None,
) -> pd.DataFrame:
    """
    Compute DPD, EOD, and DIR for all groups and return a tidy DataFrame.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Binary model predictions.
        sensitive: Protected attribute values.
        reference_group: Optional reference/majority group name.

    Returns:
        DataFrame with columns: group, approval_rate, dpd, eod, dir
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)

    groups = np.unique(sensitive)

    # Compute approval rates
    approval_rates: dict[str, float] = {
        g: round(_approval_rate(y_pred[sensitive == g]), 4) for g in groups
    }

    if reference_group is None:
        reference_group = max(approval_rates, key=lambda k: approval_rates[k])

    dpd = demographic_parity_difference(y_pred, sensitive, reference_group)
    eod = equalized_odds_difference(y_true, y_pred, sensitive, reference_group)
    dir_scores = disparate_impact_ratio(y_pred, sensitive, reference_group)

    rows = []
    for g in sorted(groups):
        n_group = int((sensitive == g).sum())
        rows.append(
            {
                "group": g,
                "is_reference": g == reference_group,
                "n": n_group,
                "approval_rate": approval_rates[g],
                "dpd": dpd[g],
                "eod": eod[g],
                "dir": dir_scores[g],
            }
        )

    return pd.DataFrame(rows).set_index("group")


def approval_rates_by_group(
    y_pred: np.ndarray | pd.Series,
    sensitive: np.ndarray | pd.Series,
) -> pd.Series:
    """Return a Series of approval rates indexed by group name."""
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)
    groups = np.unique(sensitive)
    return pd.Series(
        {g: round(_approval_rate(y_pred[sensitive == g]), 4) for g in sorted(groups)},
        name="approval_rate",
    )


if __name__ == "__main__":
    # Quick smoke-test with synthetic data
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    sensitive = rng.choice(["A", "B", "C"], n)

    df = compute_all_metrics(y_true, y_pred, sensitive)
    print(df.round(3))


def bootstrap_intervals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive: np.ndarray | pd.Series,
    reference_group: str,
    n_boot: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """95% percentile bootstrap intervals for dir, dpd and eod, per group.

    Rows are resampled with the reference group held fixed, so each interval
    shows how much a metric moves from sampling noise alone. Columns are
    ``<metric>_lo`` and ``<metric>_hi``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)
    rng = np.random.default_rng(seed)
    draws: dict[str, dict[str, list[float]]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if reference_group not in set(sensitive[idx]):
            continue
        m = compute_all_metrics(y_true[idx], y_pred[idx], sensitive[idx], reference_group)
        for g, row in m.iterrows():
            for k in ("dir", "dpd", "eod"):
                draws.setdefault(g, {}).setdefault(k, []).append(row[k])
    rows = {}
    for g, ks in draws.items():
        out = {}
        for k, v in ks.items():
            lo, hi = np.nanpercentile(v, [2.5, 97.5]) if not np.all(np.isnan(v)) else (np.nan, np.nan)
            out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
        rows[g] = out
    return pd.DataFrame.from_dict(rows, orient="index")

