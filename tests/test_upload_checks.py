"""Uploads that cannot be audited must say why, in terms of the user's file."""
import gradio as gr
import pandas as pd
import pytest

import app


def _frame(n=60, label=None, groups=("A", "B")):
    return pd.DataFrame({
        "race": [groups[i % len(groups)] for i in range(n)],
        "income": range(n),
        "loan_approved": label if label is not None else [i % 2 for i in range(n)],
    })


def test_non_binary_labels_are_named():
    with pytest.raises(gr.Error, match="only 0 and 1"):
        app.run_audit(_frame(label=[i % 3 for i in range(60)]), "race", "loan_approved")


def test_too_few_rows_or_a_rare_label_is_explained():
    with pytest.raises(gr.Error, match="at least 40 rows"):
        app.run_audit(_frame(n=8), "race", "loan_approved")
    with pytest.raises(gr.Error, match="at least 10 of each label"):
        app.run_audit(_frame(label=[1] * 55 + [0] * 5), "race", "loan_approved")


def test_single_group_attribute_is_explained():
    with pytest.raises(gr.Error, match="only one group"):
        app.run_audit(_frame(groups=("A",)), "race", "loan_approved")


def test_valid_file_still_audits():
    summary, table, _, mitigation = app.run_audit(_frame(), "race", "loan_approved")
    assert "Audit verdict" in summary and len(table) == 2
    assert mitigation == ""  # no mitigation requested → measurement-only


def test_mitigation_reduces_the_gap_on_the_biased_sample():
    """On the planted race bias, mitigation narrows the worst-group DPD."""
    df = app._sample_df()
    _, _, _, before_only = app.run_audit(df, "race", "loan_approved")
    assert before_only == ""
    for method in ("correlation_remover", "threshold_optimizer"):
        _, _, _, mitigation = app.run_audit(df, "race", "loan_approved", mitigation=method)
        assert "Before" in mitigation and "After" in mitigation
