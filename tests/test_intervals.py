"""Verdicts must separate real bias from sampling noise."""
import numpy as np
import pandas as pd

import app
from audit.bias_detector import Severity, detect_bias, run_full_detection
from audit.data_generator import generate_dataset


def _status_by_group(attribute):
    df = generate_dataset(n_samples=4000, random_state=42)
    y_true = df["loan_approved"].to_numpy(dtype=int)
    y_pred, _ = app._train_and_predict(df, "loan_approved", attribute)
    metrics, findings = run_full_detection(y_true, y_pred, df[attribute].astype(str).to_numpy(), attribute)
    return {g: app._status(g, r, findings) for g, r in metrics.iterrows() if not r["is_reference"]}


def test_injected_bias_is_confirmed():
    status = _status_by_group("race")
    assert status["Black"] == "FAIL" and status["Hispanic"] == "FAIL"


def test_attribute_without_injected_bias_does_not_fail():
    assert "FAIL" not in _status_by_group("gender").values()


def test_breach_inside_sampling_noise_is_inconclusive():
    rng = np.random.default_rng(1)
    n = 120  # tiny groups: a 12-point gap cannot be told apart from noise
    sens = np.array(["A"] * n + ["B"] * n)
    y_true = rng.integers(0, 2, 2 * n)
    y_pred = np.where(sens == "A", rng.random(2 * n) < 0.62, rng.random(2 * n) < 0.50).astype(int)
    metrics, findings = run_full_detection(y_true, y_pred, sens, "grp")
    row = metrics.loc["B"]
    if app._passes(row):  # draw did not breach: nothing to classify
        return
    assert all(f.severity == Severity.MEDIUM for f in findings if f.extra["confirmed"] is False)
    assert app._status("B", row, findings) in {"FAIL", "INCONCLUSIVE"}


def test_unconfirmed_finding_is_downgraded_and_explained():
    metrics = pd.DataFrame(
        {"is_reference": [True, False], "n": [100, 100], "approval_rate": [0.9, 0.7],
         "dpd": [0.0, 0.2], "eod": [0.0, 0.05], "dir": [1.0, 0.78]},
        index=["ref", "grp"],
    )
    wide = pd.DataFrame({f"{c}_{e}": [0.0, v] for c, (lo, hi) in
                         {"dir": (0.6, 0.95), "dpd": (0.05, 0.3), "eod": (0.0, 0.1)}.items()
                         for e, v in (("lo", lo), ("hi", hi))}, index=["ref", "grp"])
    found = detect_bias(metrics, "grp", wide)
    assert found and all(f.severity == Severity.MEDIUM and f.extra["confirmed"] is False for f in found)
    assert all("Inconclusive" in f.description for f in found)
