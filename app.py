"""
ML Fairness Audit — interactive demo.

Wraps the REAL fairness-audit logic from
the ml-fairness-audit project (the ``audit/`` package, vendored
unmodified) in a Gradio UI. Computes genuine ECOA / Fair-Lending fairness
metrics — Disparate Impact Ratio (4/5ths rule), Demographic Parity Difference,
and Equalized Odds Difference — for a model's predictions across a protected
attribute, then applies the project's regulatory thresholds to flag violations.

Nothing is fabricated: every number comes from the vendored library running on
real data (bundled synthetic loan dataset or an uploaded CSV).
"""

from __future__ import annotations

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ── Vendored REAL fairness logic ───────────────────────────────────────────
from audit.bias_detector import (
    DIR_ECOA_THRESHOLD,
    DPD_WARNING_THRESHOLD,
    EOD_WARNING_THRESHOLD,
    findings_summary,
    run_full_detection,
)
from audit.data_generator import generate_dataset, get_feature_columns
from audit.fairness_metrics import bootstrap_intervals, compute_all_metrics
from audit.mitigation import MITIGATION_METHODS, mitigate_predictions
from audit.real_data import (
    ADULT_LABEL_COL,
    INTERSECTION_COL,
    add_intersection,
    load_adult,
)

# ── Theme ──────────────────────────────────────────────────────────────────
PASS_COLOR = "#16a34a"
FAIL_COLOR = "#dc2626"
REF_COLOR = "#94a3b8"
WARN_COLOR = "#d97706"

METRIC_RULES = {
    "dir": lambda v: np.isnan(v) or v >= DIR_ECOA_THRESHOLD,
    "dpd": lambda v: v <= DPD_WARNING_THRESHOLD,
    "eod": lambda v: v <= EOD_WARNING_THRESHOLD,
}


def _passes(row) -> bool:
    return all(METRIC_RULES[k](row[k]) for k in METRIC_RULES)


def _status(group, row, findings) -> str:
    """PASS, FAIL (breach whose 95% interval clears the threshold) or INCONCLUSIVE."""
    if _passes(row):
        return "PASS"
    mine = [f for f in findings if f.group == str(group)]
    return "FAIL" if any(f.extra.get("confirmed", True) for f in mine) else "INCONCLUSIVE"


def _train_and_predict(
    df: pd.DataFrame, label_col: str, attribute: str
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a logistic-regression scorecard to produce genuine model predictions.

    The model is trained on the data's own numeric features PLUS a one-hot
    encoding of the protected attribute, so it learns the historical bias baked
    into the decisions — exactly the disparate impact a fairness audit exists to
    catch. The vendored fairness library then audits these real predictions.

    Returns ``(predictions, X_num)``: the per-row predictions and the numeric
    feature matrix (without the protected attribute), which the mitigation step
    reuses so it operates on the same features the scorecard saw.
    """
    feature_cols = [
        c for c in get_feature_columns() if c in df.columns and c != label_col
    ]
    if not feature_cols:
        feature_cols = [
            c for c in df.select_dtypes(include="number").columns if c != label_col
        ]
    if not feature_cols:
        raise ValueError("No numeric feature columns available to train a model.")

    X_num = df[feature_cols].to_numpy(dtype=float)
    X_attr = pd.get_dummies(df[attribute].astype(str)).to_numpy(dtype=float)
    X = np.hstack([X_num, X_attr])
    y = df[label_col].to_numpy(dtype=int)

    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(X_tr, y_tr)
    return model.predict(X).astype(int), X_num


def _ci(intervals, group, metric: str) -> str:
    """Format a metric's 95% bootstrap CI as ``[lo, hi]`` when available."""
    if intervals is None or group not in intervals.index:
        return ""
    lo, hi = intervals.loc[group, f"{metric}_lo"], intervals.loc[group, f"{metric}_hi"]
    if np.isnan(lo) or np.isnan(hi):
        return ""
    return f" [{lo:.2f}, {hi:.2f}]"


def _verdict_table(metrics_df: pd.DataFrame, findings, intervals=None) -> pd.DataFrame:
    rows = []
    for group, row in metrics_df.iterrows():
        if row["is_reference"]:
            rows.append(
                {
                    "Group": f"{group} (reference)",
                    "N": int(row["n"]),
                    "Approval Rate": f"{row['approval_rate']:.3f}",
                    "Disparate Impact (95% CI)": "1.000",
                    "Dem. Parity Diff (95% CI)": "0.000",
                    "Equalized Odds Diff (95% CI)": "0.000",
                    "Verdict": "BASELINE",
                }
            )
            continue
        verdict = _status(group, row, findings)
        dir_disp = "n/a" if np.isnan(row["dir"]) else f"{row['dir']:.3f}" + _ci(intervals, group, "dir")
        rows.append(
            {
                "Group": str(group),
                "N": int(row["n"]),
                "Approval Rate": f"{row['approval_rate']:.3f}",
                "Disparate Impact (95% CI)": dir_disp,
                "Dem. Parity Diff (95% CI)": f"{row['dpd']:.3f}" + _ci(intervals, group, "dpd"),
                "Equalized Odds Diff (95% CI)": f"{row['eod']:.3f}" + _ci(intervals, group, "eod"),
                "Verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def _bar_chart(metrics_df: pd.DataFrame, attribute: str, findings):
    groups = list(metrics_df.index)
    rates = metrics_df["approval_rate"].to_list()
    colors = []
    for _, row in metrics_df.iterrows():
        if row["is_reference"]:
            colors.append(REF_COLOR)
        else:
            status = _status(_, row, findings)
            colors.append({"PASS": PASS_COLOR, "FAIL": FAIL_COLOR}.get(status, WARN_COLOR))

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
    bars = ax.bar(groups, rates, color=colors, edgecolor="white", linewidth=1.2)
    ref_rate = float(
        metrics_df.loc[metrics_df["is_reference"], "approval_rate"].iloc[0]
    )
    ax.axhline(
        ref_rate * DIR_ECOA_THRESHOLD,
        color=FAIL_COLOR,
        linestyle="--",
        linewidth=1.1,
        label=f"4/5ths floor ({DIR_ECOA_THRESHOLD}× ref)",
    )
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("Approval rate")
    ax.set_title(f"Approval rate by {attribute}", fontweight="bold")
    ax.set_ylim(0, max(rates) * 1.18 + 0.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def _summary_markdown(findings, metrics_df, attribute: str) -> str:
    counts = findings_summary(findings)
    n_audited = int((~metrics_df["is_reference"]).sum())
    statuses = [
        _status(g, row, findings)
        for g, row in metrics_df.iterrows()
        if not row["is_reference"]
    ]
    n_fail = statuses.count("FAIL")
    n_incl = statuses.count("INCONCLUSIVE")
    if n_fail:
        overall, color = "FAIL — potential disparate impact detected", FAIL_COLOR
    elif n_incl:
        overall, color = "INCONCLUSIVE — threshold breached, but within sampling noise", WARN_COLOR
    else:
        overall, color = "PASS — no threshold breaches", PASS_COLOR

    lines = [
        f"### Audit verdict: <span style='color:{color}'>{overall}</span>",
        f"Protected attribute: **{attribute}** &nbsp;|&nbsp; "
        f"groups audited (vs reference): **{n_audited}** &nbsp;|&nbsp; "
        f"groups failing: **{n_fail}**"
        + (f" &nbsp;|&nbsp; inconclusive: **{n_incl}**" if n_incl else ""),
        "",
        f"**Findings** &nbsp; Critical (ECOA): **{counts['Critical']}** &nbsp;|&nbsp; "
        f"High: **{counts['High']}**",
    ]
    if findings:
        lines.append("")
        for f in findings[:6]:
            lines.append(
                f"- **[{f.severity.value}] {f.code.value}** — group `{f.group}`: "
                f"{f.metric_name} = `{f.metric_value:.3f}` (threshold `{f.threshold}`)"
            )
    return "\n".join(lines)


def _worst_metrics(metrics_df: pd.DataFrame) -> dict[str, float]:
    """Worst non-reference value of each metric (lowest DIR, highest DPD/EOD)."""
    nonref = metrics_df[~metrics_df["is_reference"]]
    dirs = nonref["dir"].dropna()
    return {
        "dir": float(dirs.min()) if len(dirs) else float("nan"),
        "dpd": float(nonref["dpd"].max()),
        "eod": float(nonref["eod"].max()),
    }


def _mitigation_markdown(
    X_num: np.ndarray,
    sensitive: np.ndarray,
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
    method: str,
) -> str:
    """Apply a fairlearn mitigation and report the fairness gap before vs after.

    The measurement path is untouched: this re-audits mitigated predictions over
    the same rows and shows the worst-group metrics and accuracy, before → after.
    """
    y_mit = mitigate_predictions(X_num, sensitive, y_true, method=method)
    # Pin the reference group from the baseline so before and after are measured
    # against the SAME group; mitigation shifts approval rates, so letting each
    # call re-pick the max-approval reference would compare unlike baselines.
    before_df = compute_all_metrics(y_true, y_pred_base, sensitive)
    ref_group = before_df.index[before_df["is_reference"]][0]
    before = _worst_metrics(before_df)
    after = _worst_metrics(compute_all_metrics(y_true, y_mit, sensitive, reference_group=ref_group))
    acc_before = float(np.mean(y_pred_base == y_true))
    acc_after = float(np.mean(y_mit == y_true))

    def cell(v: float) -> str:
        return "n/a" if np.isnan(v) else f"{v:.3f}"

    rows = [
        ("Disparate Impact Ratio (worst group)", before["dir"], after["dir"], f"≥ {DIR_ECOA_THRESHOLD}"),
        ("Demographic Parity Diff (worst group)", before["dpd"], after["dpd"], f"≤ {DPD_WARNING_THRESHOLD}"),
        ("Equalized Odds Diff (worst group)", before["eod"], after["eod"], f"≤ {EOD_WARNING_THRESHOLD}"),
    ]
    lines = [
        f"### Mitigation applied: {MITIGATION_METHODS[method]}",
        "",
        "| Metric | Before | After | Pass rule |",
        "|---|---|---|---|",
    ]
    for name, b, a, rule in rows:
        lines.append(f"| {name} | {cell(b)} | {cell(a)} | {rule} |")
    lines.append(f"| Model accuracy | {acc_before:.3f} | {acc_after:.3f} | — |")
    lines += [
        "",
        "_Before_ = the audited scorecard above. _After_ = the same data with the "
        "chosen `fairlearn` mitigation applied, then re-audited. Lower DPD/EOD and a "
        "DIR nearer 1.0 mean less disparate impact; any accuracy change is the trade-off.",
    ]
    return "\n".join(lines)


def _benchmark_markdown(
    X_num: np.ndarray,
    sensitive: np.ndarray,
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
) -> str:
    """Compare the baseline against every ``fairlearn`` mitigation in one table.

    Each row re-audits a set of predictions over the same data and reports the
    worst-group DIR/DPD/EOD plus model accuracy, so the accuracy-vs-fairness
    trade-off of each method is visible side by side.
    """
    before_df = compute_all_metrics(y_true, y_pred_base, sensitive)
    ref_group = before_df.index[before_df["is_reference"]][0]

    def cell(v: float) -> str:
        return "n/a" if np.isnan(v) else f"{v:.3f}"

    def line(name: str, y_pred: np.ndarray) -> str:
        m = _worst_metrics(compute_all_metrics(y_true, y_pred, sensitive, reference_group=ref_group))
        acc = float(np.mean(y_pred == y_true))
        return f"| {name} | {cell(m['dir'])} | {cell(m['dpd'])} | {cell(m['eod'])} | {acc:.3f} |"

    rows = [line("Baseline (no mitigation)", y_pred_base)]
    for method, label in MITIGATION_METHODS.items():
        if method == "none":
            continue
        y_mit = mitigate_predictions(X_num, sensitive, y_true, method=method)
        rows.append(line(label, y_mit))

    return "\n".join(
        [
            "### Mitigation benchmark — worst-group metrics, baseline vs every method",
            "",
            "| Method | Disparate Impact ↑ | Dem. Parity Diff ↓ | Equalized Odds Diff ↓ | Accuracy |",
            "|---|---|---|---|---|",
            *rows,
            "",
            f"Pass rules: DIR ≥ {DIR_ECOA_THRESHOLD}, DPD ≤ {DPD_WARNING_THRESHOLD}, "
            f"EOD ≤ {EOD_WARNING_THRESHOLD}. ↑ = higher is fairer, ↓ = lower is fairer. "
            "Each method is re-audited against the same reference group; the accuracy "
            "column is the cost paid for the fairness gain.",
        ]
    )


def run_audit(df: pd.DataFrame, attribute: str, label_col: str, mitigation: str = "none"):
    if attribute not in df.columns:
        raise gr.Error(f"Protected attribute '{attribute}' not found in data.")
    if label_col not in df.columns:
        raise gr.Error(f"Label column '{label_col}' not found in data.")

    if df[label_col].isna().any():
        n_blank = int(df[label_col].isna().sum())
        raise gr.Error(
            f"Label column '{label_col}' has {n_blank} empty/NaN value(s); every row needs a 0 or 1 label."
        )
    values = set(pd.unique(df[label_col].dropna()))
    if not values <= {0, 1, 0.0, 1.0}:
        raise gr.Error(
            f"Label column '{label_col}' must hold only 0 and 1; found {sorted(map(str, values))[:6]}."
        )
    counts = df[label_col].value_counts()
    if len(df) < 40 or counts.min() < 10 or len(counts) < 2:
        raise gr.Error(
            f"Need at least 40 rows with at least 10 of each label (0 and 1); "
            f"got {len(df)} rows, label counts {counts.to_dict()}."
        )
    n_groups = df[attribute].nunique()
    if n_groups < 2:
        raise gr.Error(f"Protected attribute '{attribute}' has only one group, so there is nothing to compare.")
    if n_groups > MAX_GROUPS:
        raise gr.Error(
            f"Protected attribute '{attribute}' has {n_groups:,} distinct values. "
            f"A fairness audit compares a handful of demographic groups; pick a categorical "
            f"column with at most {MAX_GROUPS} groups (an ID-like column is not a protected attribute)."
        )

    y_true = df[label_col].to_numpy(dtype=int)
    y_pred, X_num = _train_and_predict(df, label_col, attribute)
    sensitive = df[attribute].astype(str).to_numpy()

    # ── Genuine fairness audit from the vendored library ──
    metrics_df, findings = run_full_detection(
        y_true, y_pred, sensitive, protected_attribute=attribute
    )
    # Surface the bootstrap CIs (same percentile bootstrap the findings use) so
    # the headline metrics read as interval estimates, not bare point values.
    ref_group = metrics_df.index[metrics_df["is_reference"]][0]
    n_boot = 200 if len(y_true) <= 20_000 else 60
    intervals = bootstrap_intervals(y_true, y_pred, sensitive, ref_group, n_boot=n_boot)

    table = _verdict_table(metrics_df, findings, intervals)
    chart = _bar_chart(metrics_df, attribute, findings)
    summary = _summary_markdown(findings, metrics_df, attribute)

    mitigation_md = ""
    if mitigation and mitigation != "none":
        mitigation_md = _mitigation_markdown(X_num, sensitive, y_true, y_pred, mitigation)
    return summary, table, chart, mitigation_md


def _sample_df() -> pd.DataFrame:
    return generate_dataset(n_samples=4000, random_state=42)


# Dataset registry for the sample tab. Adult is the real-world default; the
# synthetic loan set stays available (and is the offline fallback).
ADULT_DATASET = "UCI Adult Income (real)"
SYNTHETIC_DATASET = "Synthetic loan data"
DATASET_CHOICES = [ADULT_DATASET, SYNTHETIC_DATASET]
DATASET_ATTRS = {
    ADULT_DATASET: ["sex", "race", INTERSECTION_COL],
    SYNTHETIC_DATASET: ["race", "gender", "age_group"],
}


def _load_sample(dataset: str) -> tuple[pd.DataFrame, str, str]:
    """Return (df, label_col, note) for the chosen sample dataset.

    Falls back to the synthetic generator (with a note) if the real dataset
    cannot be loaded — e.g. no bundled copy and no network on the Space.
    """
    if dataset == SYNTHETIC_DATASET:
        return _sample_df(), "loan_approved", ""
    try:
        return load_adult(), ADULT_LABEL_COL, ""
    except Exception as exc:  # noqa: BLE001 — any load failure → labelled fallback
        note = (
            f"\n\n> ⚠️ Could not load the UCI Adult dataset ({exc}); "
            "showing the synthetic loan data instead."
        )
        return _sample_df(), "loan_approved", note


def _prepare_sample(dataset: str, attribute: str) -> tuple[pd.DataFrame, str, str, str]:
    """Load the dataset, add the intersectional column if requested, and resolve
    the attribute to one that exists after any fallback."""
    df, label_col, note = _load_sample(dataset)
    if attribute == INTERSECTION_COL:
        if {"sex", "race"} <= set(df.columns):
            df = add_intersection(df)
        else:  # synthetic fallback has no sex × race
            attribute = "race"
    if attribute not in df.columns:
        attribute = "race"
    return df, label_col, attribute, note


def audit_sample(dataset: str, attribute: str, mitigation: str = "none"):
    df, label_col, attribute, note = _prepare_sample(dataset, attribute)
    summary, table, chart, mitigation_md = run_audit(
        df, attribute, label_col, mitigation=mitigation
    )
    return summary + note, table, chart, mitigation_md


def benchmark_sample(dataset: str, attribute: str):
    """Run the full mitigation benchmark on the chosen sample + attribute."""
    df, label_col, attribute, _ = _prepare_sample(dataset, attribute)
    # The benchmark trains a baseline plus every fairlearn mitigation; on the Space's
    # 2-vCPU CPU the full 48k-row Adult set takes minutes, so the button looked dead.
    # Cap to a stratified-enough random sample so it returns in seconds; the single
    # "Run audit" path above still uses the full dataset.
    if len(df) > 6000:
        df = df.sample(n=6000, random_state=42).reset_index(drop=True)
    try:
        y_true = df[label_col].to_numpy(dtype=int)
        y_pred, X_num = _train_and_predict(df, label_col, attribute)
        sensitive = df[attribute].astype(str).to_numpy()
        return _benchmark_markdown(X_num, sensitive, y_true, y_pred)
    except Exception as e:  # never leave the panel silently blank
        return f"**Benchmark failed:** {type(e).__name__}: {e}"


def _update_attrs(dataset: str):
    choices = DATASET_ATTRS[dataset]
    return gr.update(choices=choices, value=choices[0])


MAX_UPLOAD_ROWS = 100_000
# A protected attribute is a handful of demographic groups. One-hot encoding it in
# _train_and_predict builds an (nunique × nunique) matrix, so an ID-like column with
# thousands of distinct values would exhaust memory and crash the worker; cap it.
MAX_GROUPS = 50

# (label, value) pairs for the mitigation dropdowns.
MITIGATION_CHOICES = [(label, value) for value, label in MITIGATION_METHODS.items()]


def audit_upload(file, attribute: str, label_col: str, mitigation: str = "none"):
    if file is None:
        raise gr.Error("Upload a CSV first, or use the bundled sample dataset.")
    try:
        df = pd.read_csv(file.name)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise gr.Error("Could not read this file as a CSV with a header row and data rows.") from exc
    if len(df) > MAX_UPLOAD_ROWS:
        raise gr.Error(f"Upload at most {MAX_UPLOAD_ROWS:,} rows (this file has {len(df):,}).")
    attribute = (attribute or "").strip()
    label_col = (label_col or "").strip()
    if not attribute or not label_col:
        raise gr.Error(
            "Specify both the protected-attribute column and the label column."
        )
    try:
        return run_audit(df, attribute, label_col, mitigation=mitigation)
    except ValueError as exc:
        # The label must be 0/1 and at least one other column must be numeric for the scorecard to train.
        raise gr.Error(f"Could not audit this data: {exc}. The label column needs 0/1 values and the file needs numeric feature columns.") from exc


INTRO = """
# ⚖️ ML Fairness Audit

Runs the **genuine** fairness-audit engine from
the `ml-fairness-audit` project
(the `audit/` package, vendored unmodified) on real data. It fits a logistic
scorecard, then measures bias across a protected attribute using three
regulatory metrics and the project's ECOA / Fair-Lending thresholds:

| Metric | Meaning | Pass rule |
|---|---|---|
| **Disparate Impact Ratio** | minority approval ÷ reference approval | ≥ 0.80 (EEOC 4/5ths rule) |
| **Demographic Parity Diff** | gap in approval rate vs reference | ≤ 0.10 |
| **Equalized Odds Diff** | gap in TPR/FPR vs reference | ≤ 0.10 |

The demo opens on the real **UCI Adult Income** dataset (48,842 US census
records). A scorecard trained on it amplifies the data's own income gap: women
and several racial groups fall far below the 4/5ths floor — a genuine disparate
impact, not a synthetic one. Headline metrics carry **95% bootstrap confidence
intervals**, so verdicts are interval estimates rather than point values.

Pick a **Mitigation** method to go beyond measuring (fairness gap
**before → after**, with the accuracy trade-off), or **Benchmark all
mitigations** to compare the baseline against every `fairlearn` method at once.
"""


with gr.Blocks(
    title="ML Fairness Audit",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
) as demo:
    gr.Markdown(INTRO)

    with gr.Tab("Sample dataset"):
        gr.Markdown(
            "Audit a real, recognizable bias case: the **UCI Adult Income** "
            "dataset (48,842 US census records; predict income > $50K). Protected "
            "attributes are **sex** and **race**, with an intersectional "
            "**sex × race** option. Switch to the synthetic loan data to compare."
        )
        with gr.Row():
            sample_dataset = gr.Dropdown(
                choices=DATASET_CHOICES,
                value=ADULT_DATASET,
                label="Dataset",
                scale=2,
            )
            sample_attr = gr.Dropdown(
                choices=DATASET_ATTRS[ADULT_DATASET],
                value="sex",
                label="Protected attribute",
                scale=2,
            )
            sample_mit = gr.Dropdown(
                choices=MITIGATION_CHOICES,
                value="none",
                label="Mitigation (show before → after)",
                scale=2,
            )
        with gr.Row():
            sample_btn = gr.Button("Run audit", variant="primary")
            bench_btn = gr.Button("Benchmark all mitigations")
        sample_dataset.change(_update_attrs, inputs=[sample_dataset], outputs=[sample_attr])

    with gr.Tab("Upload CSV"):
        gr.Markdown(
            "Upload a CSV with a binary label column (0/1), at least one numeric "
            "feature column, and a categorical protected-attribute column."
        )
        up_file = gr.File(label="CSV file", file_types=[".csv"])
        with gr.Row():
            up_attr = gr.Textbox(label="Protected-attribute column", value="race")
            up_label = gr.Textbox(label="Label column (0/1)", value="loan_approved")
        up_mit = gr.Dropdown(
            choices=MITIGATION_CHOICES,
            value="none",
            label="Mitigation (show before → after)",
        )
        up_btn = gr.Button("Run audit", variant="primary")

    gr.Markdown("## Results")
    out_summary = gr.Markdown()
    out_table = gr.Dataframe(label="Per-group fairness metrics & verdict", wrap=True)
    out_chart = gr.Plot(label="Approval rate by group")
    out_mitigation = gr.Markdown()
    # Own panel for the mitigation benchmark so a still-running baseline audit
    # (demo.load / Run audit, which write out_mitigation) can't overwrite it.
    out_benchmark = gr.Markdown()

    sample_btn.click(
        audit_sample,
        inputs=[sample_dataset, sample_attr, sample_mit],
        outputs=[out_summary, out_table, out_chart, out_mitigation],
    )
    bench_btn.click(
        benchmark_sample,
        inputs=[sample_dataset, sample_attr],
        outputs=[out_benchmark],
    )
    up_btn.click(
        audit_upload,
        inputs=[up_file, up_attr, up_label, up_mit],
        outputs=[out_summary, out_table, out_chart, out_mitigation],
    )

    # Open directly on the default bias case (UCI Adult, sex) instead of a blank page.
    demo.load(
        audit_sample,
        inputs=[sample_dataset, sample_attr, sample_mit],
        outputs=[out_summary, out_table, out_chart, out_mitigation],
    )

    gr.Markdown(
        "Source: the Files tab of this Space · "
        "Metrics computed live, no fabricated values."
    )


if __name__ == "__main__":
    demo.launch()
