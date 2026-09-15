"""
generate_data.py
-----------------
Creates a SYNTHETIC (fake but logically realistic) dataset of land acquisition
projects. Real government land-records data is not publicly accessible for
this hackathon project, so we simulate it: risk factors mentioned in the
problem statement (legal disputes, compensation delays, rehabilitation
progress, etc.) are combined with realistic noise so that a machine learning
model has genuine, non-trivial patterns to learn from.

Run:  python3 generate_data.py
Output: land_projects.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_PROJECTS = 1500

STATES = [
    "Uttar Pradesh", "Maharashtra", "Bihar", "West Bengal", "Madhya Pradesh",
    "Tamil Nadu", "Rajasthan", "Karnataka", "Gujarat", "Odisha",
    "Telangana", "Kerala", "Punjab", "Haryana", "Assam", "Jharkhand"
]

PROJECT_TYPES = [
    "Highway", "Railway", "Irrigation Canal", "Power Transmission Line",
    "Industrial Corridor", "Urban Metro", "Port Development", "Dam/Reservoir"
]


def generate():
    n = N_PROJECTS
    df = pd.DataFrame({
        "project_id": [f"LA-{1000 + i}" for i in range(n)],
        "state": RNG.choice(STATES, n),
        "project_type": RNG.choice(PROJECT_TYPES, n),
        "land_area_hectares": np.round(RNG.gamma(shape=2.0, scale=25, size=n), 1),
        "affected_families": RNG.integers(5, 2000, n),
        "compensation_disbursed_pct": np.round(RNG.beta(2, 2, n) * 100, 1),
        "approval_stage_days": RNG.integers(10, 900, n),          # days since application
        "legal_disputes_count": RNG.poisson(0.6, n),
        "rehabilitation_progress_pct": np.round(RNG.beta(2, 2.5, n) * 100, 1),
        "documentation_completeness_pct": np.round(RNG.beta(3, 1.2, n) * 100, 1),
        "stakeholder_responsiveness_score": np.round(RNG.uniform(1, 10, n), 1),  # 1=poor,10=excellent
        "past_dept_delay_rate_pct": np.round(RNG.beta(2, 3, n) * 100, 1),  # historical dept performance
    })

    # ---- Build a realistic "risk score" from the features (0-100) ----
    # Higher risk when: compensation low, legal disputes high, rehab low,
    # docs incomplete, stakeholders unresponsive, dept historically slow,
    # long time already stuck in approval, large number of affected families.
    risk = (
        (100 - df["compensation_disbursed_pct"]) * 0.20
        + df["legal_disputes_count"].clip(upper=5) * 12
        + (100 - df["rehabilitation_progress_pct"]) * 0.18
        + (100 - df["documentation_completeness_pct"]) * 0.15
        + (10 - df["stakeholder_responsiveness_score"]) * 3.0
        + df["past_dept_delay_rate_pct"] * 0.25
        + np.clip(df["approval_stage_days"] / 900 * 30, 0, 30)
        + np.clip(df["affected_families"] / 2000 * 15, 0, 15)
    )

    # Normalise to 0-100 and add random noise so it's not a deterministic formula
    risk = (risk - risk.min()) / (risk.max() - risk.min()) * 100
    noise = RNG.normal(0, 3, n)
    risk = np.clip(risk + noise, 0, 100)
    df["true_risk_score"] = np.round(risk, 1)

    # Binary label: was the project actually delayed? (probabilistic, not a hard cutoff)
    prob_delay = 1 / (1 + np.exp(-(risk - 50) / 7))  # logistic curve around 50, sharper than before
    df["delayed"] = (RNG.uniform(0, 1, n) < prob_delay).astype(int)

    df = df.drop(columns=["true_risk_score"])  # model shouldn't see the "answer key" directly
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("land_projects.csv", index=False)
    print(f"Generated {len(df)} synthetic project records -> land_projects.csv")
    print(f"Delay rate in data: {df['delayed'].mean():.1%}")
