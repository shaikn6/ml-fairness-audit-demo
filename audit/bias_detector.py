"""
Bias detector: applies regulatory thresholds to fairness metrics and
returns a structured list of findings with severity levels.

Regulatory thresholds:
  - Disparate Impact < 0.8  →  ECOA_VIOLATION    (Critical)
  - DPD > 0.1               →  FAIRNESS_WARNING  (High)
  - EOD > 0.1               →  EOD_WARNING       (High)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from audit.fairness_metrics import bootstrap_intervals, compute_all_metrics


class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    INFO = "Info"


class FindingCode(str, Enum):
    ECOA_VIOLATION = "ECOA_VIOLATION"
    FAIRNESS_WARNING = "FAIRNESS_WARNING"
    EOD_WARNING = "EOD_WARNING"
    INFO = "INFO"


# Regulatory thresholds
DIR_ECOA_THRESHOLD = 0.80   # Below this = ECOA violation (4/5ths rule)
DPD_WARNING_THRESHOLD = 0.10
EOD_WARNING_THRESHOLD = 0.10


@dataclass
class AuditFinding:
    """A single regulatory or fairness finding."""
    code: FindingCode
    severity: Severity
    group: str
    protected_attribute: str
    metric_name: str
    metric_value: float
    threshold: float
    description: str
    recommendation: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "group": self.group,
            "protected_attribute": self.protected_attribute,
            "metric_name": self.metric_name,
            "metric_value": round(self.metric_value, 4),
            "threshold": self.threshold,
            "description": self.description,
            "recommendation": self.recommendation,
            "confirmed": self.extra.get("confirmed"),
        }


def detect_bias(
    metrics_df: pd.DataFrame,
    protected_attribute: str = "race",
    intervals: pd.DataFrame | None = None,
) -> list[AuditFinding]:
    """
    Apply regulatory thresholds to a fairness metrics DataFrame.

    Args:
        metrics_df: Output of ``compute_all_metrics`` (indexed by group name).
        protected_attribute: Name of the protected attribute being audited.
        intervals: Optional output of ``bootstrap_intervals``. A finding whose
            95% interval still contains its threshold is kept but downgraded to
            Medium and marked inconclusive (``extra["confirmed"] = False``).

    Returns:
        List of AuditFinding objects, sorted by severity (Critical first).
    """
    findings: list[AuditFinding] = []

    for group, row in metrics_df.iterrows():
        if row["is_reference"]:
            continue  # reference group is the baseline; skip

        # ── Disparate Impact Ratio (ECOA 4/5ths rule) ──────────────────────
        dir_val = row["dir"]
        if not np.isnan(dir_val) and dir_val < DIR_ECOA_THRESHOLD:
            findings.append(
                AuditFinding(
                    code=FindingCode.ECOA_VIOLATION,
                    severity=Severity.CRITICAL,
                    group=str(group),
                    protected_attribute=protected_attribute,
                    metric_name="Disparate Impact Ratio",
                    metric_value=float(dir_val),
                    threshold=DIR_ECOA_THRESHOLD,
                    description=(
                        f"Group '{group}' has a Disparate Impact Ratio of "
                        f"{dir_val:.3f}, which is below the ECOA 4/5ths rule "
                        f"threshold of {DIR_ECOA_THRESHOLD}. This constitutes "
                        f"potential unlawful disparate impact under the Equal "
                        f"Credit Opportunity Act (15 U.S.C. § 1691)."
                    ),
                    recommendation=(
                        "Conduct an adverse impact analysis. Apply fairness "
                        "constraints (e.g. ExponentiatedGradient with "
                        "DemographicParity) or post-processing calibration. "
                        "Document business necessity justification if applicable."
                    ),
                )
            )

        # ── Demographic Parity Difference ──────────────────────────────────
        dpd_val = row["dpd"]
        if dpd_val > DPD_WARNING_THRESHOLD:
            findings.append(
                AuditFinding(
                    code=FindingCode.FAIRNESS_WARNING,
                    severity=Severity.HIGH,
                    group=str(group),
                    protected_attribute=protected_attribute,
                    metric_name="Demographic Parity Difference",
                    metric_value=float(dpd_val),
                    threshold=DPD_WARNING_THRESHOLD,
                    description=(
                        f"Group '{group}' has a Demographic Parity Difference "
                        f"of {dpd_val:.3f} (threshold: {DPD_WARNING_THRESHOLD}). "
                        f"The model produces meaningfully different approval rates "
                        f"across protected groups."
                    ),
                    recommendation=(
                        "Review feature engineering for proxy variables correlated "
                        "with protected attributes. Consider in-processing fairness "
                        "constraints or pre-processing reweighing."
                    ),
                )
            )

        # ── Equalized Odds Difference ──────────────────────────────────────
        eod_val = row["eod"]
        if eod_val > EOD_WARNING_THRESHOLD:
            findings.append(
                AuditFinding(
                    code=FindingCode.EOD_WARNING,
                    severity=Severity.HIGH,
                    group=str(group),
                    protected_attribute=protected_attribute,
                    metric_name="Equalized Odds Difference",
                    metric_value=float(eod_val),
                    threshold=EOD_WARNING_THRESHOLD,
                    description=(
                        f"Group '{group}' has an Equalized Odds Difference of "
                        f"{eod_val:.3f} (threshold: {EOD_WARNING_THRESHOLD}). "
                        f"The model's error rates differ significantly between "
                        f"groups, meaning qualified applicants in the affected "
                        f"group are denied at a higher rate."
                    ),
                    recommendation=(
                        "Apply post-processing threshold calibration per group, "
                        "or use in-processing methods (e.g. fairlearn's "
                        "EqualizedOdds constraint) during model training."
                    ),
                )
            )

    if intervals is not None:
        for f in findings:
            key = {"Disparate Impact Ratio": "dir", "Demographic Parity Difference": "dpd",
                   "Equalized Odds Difference": "eod"}[f.metric_name]
            lo, hi = intervals.loc[f.group, f"{key}_lo"], intervals.loc[f.group, f"{key}_hi"]
            # dir breaches when it is too low, dpd/eod when too high
            confirmed = bool(hi < f.threshold) if key == "dir" else bool(lo > f.threshold)
            f.extra["confirmed"] = confirmed
            if not confirmed:
                f.severity = Severity.MEDIUM
                f.description += (
                    f" Inconclusive: the 95% bootstrap interval [{lo:.3f}, {hi:.3f}] "
                    f"includes the threshold."
                )

    # Sort: Critical → High → Medium → Info
    _order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.INFO: 3}
    findings.sort(key=lambda f: _order[f.severity])
    return findings


def run_full_detection(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive: np.ndarray | pd.Series,
    protected_attribute: str = "race",
) -> tuple[pd.DataFrame, list[AuditFinding]]:
    """
    Convenience wrapper: compute metrics then detect bias in one call.

    Returns:
        (metrics_df, findings_list)
    """
    metrics_df = compute_all_metrics(y_true, y_pred, sensitive)
    ref = metrics_df.index[metrics_df["is_reference"]][0]
    n_boot = 200 if len(y_true) <= 20_000 else 60
    intervals = bootstrap_intervals(y_true, y_pred, sensitive, ref, n_boot=n_boot)
    findings = detect_bias(metrics_df, protected_attribute, intervals)
    return metrics_df, findings


def findings_summary(findings: list[AuditFinding]) -> dict[str, int]:
    """Return count of findings by severity."""
    counts: dict[str, int] = {s.value: 0 for s in Severity}
    for f in findings:
        counts[f.severity.value] += 1
    return counts


if __name__ == "__main__":
    # Quick integration test. train_models lives outside the shipped package, so skip
    # cleanly when it is absent instead of raising ModuleNotFoundError.
    import numpy as np
    from audit.data_generator import generate_dataset

    try:
        from audit.model_trainer import train_models
    except ModuleNotFoundError:
        import sys

        print("Skipping self-test: audit.model_trainer is not available in this checkout.")
        sys.exit(0)

    df = generate_dataset(n_samples=2000)
    result = train_models(df, save=False)
    y_pred = result["biased_model"].predict(result["X_test"])
    y_true = result["y_test"].values
    sensitive = result["s_test"].values

    metrics_df, findings = run_full_detection(y_true, y_pred, sensitive)
    print("\n[Fairness Metrics]")
    print(metrics_df.round(3))
    print(f"\n[Findings] ({len(findings)} total)")
    for f in findings:
        print(f"  [{f.severity.value}] {f.code.value} — {f.group}: {f.metric_name}={f.metric_value:.3f}")
    print("\n[Summary]", findings_summary(findings))
