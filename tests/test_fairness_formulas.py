"""The fairness metrics must match their textbook definitions exactly."""
import numpy as np

from audit.fairness_metrics import compute_all_metrics


def _toy():
    # Reference group A approves 8/10; group B approves 4/10.
    sensitive = np.array(["A"] * 10 + ["B"] * 10)
    y_pred = np.array([1] * 8 + [0] * 2 + [1] * 4 + [0] * 6)
    # True labels: A all-qualified, B half-qualified (drives TPR/EOD).
    y_true = np.array([1] * 10 + [1] * 5 + [0] * 5)
    return y_true, y_pred, sensitive


def test_disparate_impact_is_rate_ratio():
    y_true, y_pred, sensitive = _toy()
    m = compute_all_metrics(y_true, y_pred, sensitive)
    # A has the higher approval rate, so it is the reference.
    assert bool(m.loc["A", "is_reference"]) is True
    assert m.loc["A", "approval_rate"] == 0.8
    assert m.loc["B", "approval_rate"] == 0.4
    # DIR = rate_B / rate_A = 0.4 / 0.8 = 0.5
    assert np.isclose(m.loc["B", "dir"], 0.5)


def test_demographic_parity_is_absolute_rate_gap():
    y_true, y_pred, sensitive = _toy()
    m = compute_all_metrics(y_true, y_pred, sensitive)
    # DPD = |rate_B - rate_A| = |0.4 - 0.8| = 0.4
    assert np.isclose(m.loc["B", "dpd"], 0.4)
    assert np.isclose(m.loc["A", "dpd"], 0.0)


def test_reference_group_has_identity_metrics():
    y_true, y_pred, sensitive = _toy()
    m = compute_all_metrics(y_true, y_pred, sensitive)
    assert np.isclose(m.loc["A", "dir"], 1.0)
    assert np.isclose(m.loc["A", "eod"], 0.0)


def _eod_toy():
    # A (reference): TPR 0.9, FPR 0.2, approval 0.55.
    # B: TPR 0.5, FPR 0.1, approval 0.30 (lower, so A stays reference).
    y_true = np.array([1] * 10 + [0] * 10 + [1] * 10 + [0] * 10)
    a_pred = [1] * 9 + [0] + [1] * 2 + [0] * 8
    b_pred = [1] * 5 + [0] * 5 + [1] * 1 + [0] * 9
    y_pred = np.array(a_pred + b_pred)
    sensitive = np.array(["A"] * 20 + ["B"] * 20)
    return y_true, y_pred, sensitive


def test_equalized_odds_is_max_of_tpr_and_fpr_gaps():
    y_true, y_pred, sensitive = _eod_toy()
    m = compute_all_metrics(y_true, y_pred, sensitive)
    assert bool(m.loc["A", "is_reference"]) is True
    # TPR gap |0.5 - 0.9| = 0.4 dominates FPR gap |0.1 - 0.2| = 0.1.
    assert np.isclose(m.loc["B", "eod"], 0.4)
    assert np.isclose(m.loc["A", "eod"], 0.0)


def test_explicit_reference_group_is_honoured():
    y_true, y_pred, sensitive = _toy()
    m = compute_all_metrics(y_true, y_pred, sensitive, reference_group="B")
    assert bool(m.loc["B", "is_reference"]) is True
    # Against B's 0.4 rate, A's 0.8 rate gives DIR = 0.8 / 0.4 = 2.0
    assert np.isclose(m.loc["A", "dir"], 2.0)
