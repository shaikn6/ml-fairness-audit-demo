"""
Bias mitigation for the fairness audit.

The rest of the ``audit/`` package only MEASURES disparate impact; this module
REDUCES it, so the demo can show the fairness gap before vs after mitigation.
It returns new binary predictions over the same rows, which the existing audit
(``run_full_detection``) then scores unchanged — the measurement path is not
touched.

Two fairlearn strategies:

  - ``"correlation_remover"`` — pre-processing. Strips the linear correlation
    between the numeric features and the protected attribute, then refits the
    logistic scorecard on the decorrelated features (the protected attribute is
    not an input), so the model can no longer lean on proxy signal.
  - ``"threshold_optimizer"`` — post-processing. Keeps a scorecard trained on
    the same numeric features but picks group-specific decision thresholds that
    satisfy demographic parity, directly equalising selection rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# UI labels. "none" leaves the audit measurement-only (the original behaviour).
MITIGATION_METHODS = {
    "none": "None — measure only",
    "correlation_remover": "Correlation Remover (pre-processing)",
    "threshold_optimizer": "Threshold Optimizer — demographic parity (post-processing)",
}


def mitigate_predictions(
    X_num: np.ndarray,
    sensitive: np.ndarray,
    y_true: np.ndarray,
    *,
    method: str,
    random_state: int = 42,
) -> np.ndarray:
    """Return bias-mitigated binary predictions over the same rows as ``X_num``.

    Args:
        X_num: Numeric feature matrix (no protected attribute, no label).
        sensitive: Protected attribute values, one per row.
        y_true: Ground-truth binary labels, one per row.
        method: One of ``MITIGATION_METHODS`` (other than ``"none"``).
        random_state: Seed for the train/test split and the post-processing draw.
    """
    y_true = np.asarray(y_true, dtype=int)
    sensitive = np.asarray(sensitive)

    if method == "correlation_remover":
        from fairlearn.preprocessing import CorrelationRemover

        # One-hot the protected attribute and let CorrelationRemover strip its
        # linear correlation out of the numeric features; it returns only the
        # (now decorrelated) numeric columns, so the attribute is never an input.
        onehot = pd.get_dummies(pd.Series(sensitive, name="sensitive"), prefix="s")
        feat = pd.DataFrame(X_num, columns=[f"f{i}" for i in range(X_num.shape[1])])
        data = pd.concat(
            [feat.reset_index(drop=True), onehot.reset_index(drop=True)], axis=1
        )
        remover = CorrelationRemover(sensitive_feature_ids=list(onehot.columns))
        X_decorr = np.asarray(remover.fit_transform(data), dtype=float)
        X_tr, _, y_tr, _ = train_test_split(
            X_decorr, y_true, test_size=0.2, random_state=random_state, stratify=y_true
        )
        model = LogisticRegression(max_iter=2000)
        model.fit(X_tr, y_tr)
        return model.predict(X_decorr).astype(int)

    if method == "threshold_optimizer":
        from fairlearn.postprocessing import ThresholdOptimizer

        base = LogisticRegression(max_iter=2000)
        optimizer = ThresholdOptimizer(
            estimator=base,
            constraints="demographic_parity",
            objective="accuracy_score",
            predict_method="predict_proba",
            prefit=False,
        )
        optimizer.fit(X_num, y_true, sensitive_features=sensitive)
        preds = optimizer.predict(
            X_num, sensitive_features=sensitive, random_state=random_state
        )
        return np.asarray(preds, dtype=int)

    raise ValueError(
        f"Unknown mitigation method {method!r}; expected one of "
        f"{sorted(k for k in MITIGATION_METHODS if k != 'none')}."
    )
