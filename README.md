# ⚖️ ML Fairness Audit

**Audits a classifier for disparate impact on the real UCI Adult Income dataset. A scorecard trained on it fails the 4/5ths rule for women and for several racial groups, exposes compounded bias under intersectional (sex × race) analysis, and marks breaches inside sampling noise INCONCLUSIVE using 95% bootstrap intervals.**

Interactive demo of the **genuine** fairness-audit engine from
the `ml-fairness-audit` project.
The `audit/` package (fairness metrics + regulatory bias detector) is vendored
**unmodified** — no logic was reimplemented and no numbers are fabricated.

## What it does

Given a model's predictions over a protected attribute, it computes three
ECOA / Fair-Lending fairness metrics and applies the project's regulatory
thresholds:

| Metric | Meaning | Pass rule |
|---|---|---|
| **Disparate Impact Ratio** | minority approval ÷ reference approval | ≥ 0.80 (EEOC 4/5ths rule) |
| **Demographic Parity Difference** | gap in approval rate vs reference | ≤ 0.10 |
| **Equalized Odds Difference** | gap in TPR/FPR vs reference | ≤ 0.10 |

A Disparate Impact Ratio below 0.80 is flagged as a **Critical ECOA violation**
(15 U.S.C. § 1691, the EEOC 4/5ths rule).

## Mitigation (before → after)

Measuring bias is only half the job. Pick a **Mitigation** method and the app
re-runs the same audit on `fairlearn`-mitigated predictions, then reports the
fairness gap **before → after** with the accuracy trade-off:

| Method | Type | What it does |
|---|---|---|
| **Correlation Remover** | pre-processing | strips the linear correlation between the numeric features and the protected attribute, then refits the scorecard without the attribute as an input |
| **Threshold Optimizer** | post-processing | picks group-specific decision thresholds that satisfy demographic parity |

On the synthetic-loan `race` audit (4,000 rows, reference group Asian), the
planted bias is reduced from a **FAIL** to a **PASS**:

| Metric | Before | Correlation Remover | Threshold Optimizer | Pass rule |
|---|---|---|---|---|
| Disparate Impact Ratio (worst group) | 0.682 | 0.954 | 0.992 | ≥ 0.80 |
| Demographic Parity Diff (worst group) | 0.293 | 0.038 | 0.006 | ≤ 0.10 |
| Equalized Odds Diff (worst group) | 0.345 | 0.060 | 0.064 | ≤ 0.10 |
| Model accuracy | 0.712 | 0.704 | 0.707 | — |

The measurement path is unchanged; mitigation is an added, optional step.

## How to use

- **Sample dataset** — defaults to the **UCI Adult Income** data (48,842 rows).
  Choose `sex`, `race`, or intersectional `sex × race`. A logistic scorecard
  trained on the data amplifies its real income gap, so the audit surfaces
  genuine disparate impact. **Benchmark all mitigations** compares the baseline
  against every `fairlearn` method in one table. Switch the Dataset dropdown to
  the synthetic loan data (with its planted race bias) to compare.
- **Upload CSV** — provide a CSV with a binary label column (0/1), one or more
  numeric feature columns, and a categorical protected-attribute column.

## Dataset, method & limitations

- **Dataset.** The default is the **UCI Adult / Census Income** dataset — 48,842
  records from the 1994 US Census, the canonical fairness benchmark. The task is
  to predict whether a person earns **> $50K/year**. It is fetched via
  `fairlearn.datasets.fetch_adult` (OpenML) and a copy is bundled with the Space
  (`data/adult_income.csv`) so the audit runs with no network; the synthetic
  generator is the offline fallback. Source: Becker & Kohavi (1996), *Adult*,
  UCI Machine Learning Repository, licensed CC BY 4.0.
- **Protected attributes.** `sex` (Male/Female) and `race` (5 census
  categories), plus the intersection `sex × race`. In the raw data women are
  approved far less often than men, and intersectional groups (e.g. Black women)
  fall lowest of all — compounded bias a single-attribute audit misses.
- **Method.** A logistic-regression scorecard is trained on the numeric features
  **plus** the protected attribute, so it learns — and in fact amplifies — the
  historical disparity; the vendored engine then audits those predictions. Three ECOA / Fair-Lending
  metrics are computed with 95% bootstrap confidence intervals.
- **Legal basis (4/5ths rule).** A Disparate Impact Ratio below **0.80** is the
  EEOC "four-fifths" screening threshold (29 C.F.R. § 1607.4(D)); in lending it
  maps to ECOA (15 U.S.C. § 1691). It is a **screen**, not a legal finding.
- **Benchmark (sex, full Adult data, reference group Male).** Both `fairlearn`
  mitigations lift Disparate Impact and Demographic Parity above their
  thresholds at a small accuracy cost. Equalized Odds stays above 0.10 for both
  — Threshold Optimizer targets demographic parity, not equalized odds, so it is
  not expected to close the TPR/FPR gap:

  | Method | Disparate Impact ↑ | Dem. Parity Diff ↓ | Equalized Odds Diff ↓ | Accuracy |
  |---|---|---|---|---|
  | Baseline (no mitigation) | 0.163 | 0.165 | 0.244 | 0.823 |
  | Correlation Remover | 0.874 | 0.015 | 0.103 | 0.805 |
  | Threshold Optimizer | 1.005 | 0.001 | 0.141 | 0.803 |

- **Caveats.** Adult is a 1994 snapshot and its categories are coarse; results
  describe *this dataset and scorecard*, not any production lender. The model is
  trained on 80% of rows and scored on all rows (training rows included). A
  breach whose 95% interval still spans 0.80 is reported INCONCLUSIVE, not FAIL.

## Limits

- It audits the predictions of a logistic regression that the app trains on
  your file (80% of rows), scored on every row, training rows included. It does
  not load or audit your own model.
- The default data is the real UCI Adult set; the synthetic loan option has its
  bias planted on purpose.
- The pass rules above are screening thresholds. A FAIL means "review this",
  not a legal finding.
- Each breach is checked with a 95% bootstrap interval (200 resamples, 60 above
  20,000 rows; the reference group is held fixed). If the interval still contains the
  threshold, the verdict is INCONCLUSIVE instead of FAIL: the gap could be sampling noise,
  which happens most with small groups. On the bundled sample the Black and Hispanic
  groups are FAIL and gender passes. The CIs are bootstrapped twice per run (once
  inside the detector, once for the table), so an audit takes about 3 s on the
  4,000-row synthetic set, about 7 s on Adult by `sex`, and about 15 s on the
  10-group Adult `sex × race` audit.
- Uploads: at most 100,000 rows, label column must hold 0/1 values.

## Stack

Pure CPU — `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `gradio`. No GPU,
no external API keys.

---

Source: the Files tab of this Space
