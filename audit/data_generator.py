"""
Synthetic loan application dataset generator.

Generates 10,000 synthetic loan applications with realistic bias injected:
Black and Hispanic applicants face a 15% lower approval rate at equivalent
credit scores — simulating historical discriminatory lending patterns that
ECOA audits are designed to detect.
"""

import numpy as np
import pandas as pd
from pathlib import Path


RACES = ["White", "Black", "Hispanic", "Asian"]
GENDERS = ["M", "F"]
AGE_GROUPS = ["18-30", "31-50", "51+"]

RACE_WEIGHTS = [0.60, 0.13, 0.18, 0.09]  # US demographic approximation
GENDER_WEIGHTS = [0.51, 0.49]
AGE_WEIGHTS = [0.22, 0.44, 0.34]

# Groups that face injected bias
BIASED_RACES = {"Black", "Hispanic"}
BIAS_PENALTY = 0.15  # 15 pp lower approval probability at same credit score


def _base_approval_probability(
    credit_score: np.ndarray,
    income: np.ndarray,
    debt_to_income: np.ndarray,
    employment_years: np.ndarray,
) -> np.ndarray:
    """Compute merit-based approval probability from financial features."""
    # Normalised credit score contribution (300-850 range)
    score_norm = (credit_score - 300) / 550  # 0-1
    # Income in $k
    income_norm = np.clip(income / 150_000, 0, 1)
    # DTI: lower is better; above 43% is conventional cutoff
    dti_factor = np.clip(1 - debt_to_income / 0.60, 0, 1)
    # Employment stability
    emp_norm = np.clip(employment_years / 15, 0, 1)

    logit = (
        -2.0
        + 4.5 * score_norm
        + 1.2 * income_norm
        - 2.0 * (1 - dti_factor)
        + 0.8 * emp_norm
    )
    return 1.0 / (1.0 + np.exp(-logit))


def generate_dataset(
    n_samples: int = 10_000,
    random_state: int = 42,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic loan application dataset with injected demographic bias.

    Args:
        n_samples: Number of loan applications to generate.
        random_state: NumPy random seed for reproducibility.
        output_path: If provided, save CSV to this path.

    Returns:
        DataFrame with features, protected attributes, and loan_approved label.
    """
    rng = np.random.default_rng(random_state)

    # ── Protected attributes ────────────────────────────────────────────────
    race = rng.choice(RACES, size=n_samples, p=RACE_WEIGHTS)
    gender = rng.choice(GENDERS, size=n_samples, p=GENDER_WEIGHTS)
    age_group = rng.choice(AGE_GROUPS, size=n_samples, p=AGE_WEIGHTS)

    # ── Financial features (realistic distributions) ────────────────────────
    credit_score = np.clip(rng.normal(680, 80, n_samples), 300, 850).astype(int)
    income = np.clip(rng.lognormal(10.8, 0.55, n_samples), 20_000, 500_000)
    debt_to_income = np.clip(rng.beta(2, 5, n_samples), 0.05, 0.75)
    loan_amount = np.clip(rng.lognormal(11.5, 0.7, n_samples), 5_000, 500_000)
    employment_years = np.clip(rng.exponential(8, n_samples), 0, 40)
    num_accounts = rng.integers(1, 20, n_samples)
    loan_purpose = rng.choice(
        ["home", "auto", "personal", "business", "education"],
        size=n_samples,
        p=[0.35, 0.25, 0.20, 0.12, 0.08],
    )

    # ── Compute base approval probability ──────────────────────────────────
    base_prob = _base_approval_probability(
        credit_score, income, debt_to_income, employment_years
    )

    # ── Inject bias: penalise Black and Hispanic applicants ─────────────────
    bias_mask = np.array([r in BIASED_RACES for r in race])
    adjusted_prob = np.where(
        bias_mask,
        np.clip(base_prob - BIAS_PENALTY, 0.0, 1.0),
        base_prob,
    )

    # ── Sample approval outcome ─────────────────────────────────────────────
    loan_approved = (rng.uniform(size=n_samples) < adjusted_prob).astype(int)

    df = pd.DataFrame(
        {
            # Protected attributes
            "race": race,
            "gender": gender,
            "age_group": age_group,
            # Financial features
            "credit_score": credit_score,
            "income": income.round(2),
            "debt_to_income": debt_to_income.round(4),
            "loan_amount": loan_amount.round(2),
            "employment_years": employment_years.round(1),
            "num_accounts": num_accounts,
            "loan_purpose": loan_purpose,
            # Label
            "loan_approved": loan_approved,
        }
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[data_generator] Saved {len(df):,} records to {output_path}")

    return df


def get_feature_columns() -> list[str]:
    """Return ordered list of ML feature columns (no protected attributes, no label)."""
    return [
        "credit_score",
        "income",
        "debt_to_income",
        "loan_amount",
        "employment_years",
        "num_accounts",
    ]


def get_protected_attributes() -> list[str]:
    """Return list of protected attribute column names."""
    return ["race", "gender", "age_group"]


if __name__ == "__main__":
    df = generate_dataset(output_path="data/loan_applications.csv")
    print(df.head())
    print("\nApproval rate by race:")
    print(df.groupby("race")["loan_approved"].mean().round(3))
