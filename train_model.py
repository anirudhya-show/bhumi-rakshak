"""
train_model.py
----------------
Trains an XGBoost model to predict delay probability for each land
acquisition project, computes SHAP values (WHY the model made each
prediction), attaches a plain-English recommendation, and saves everything
the dashboard needs into predictions.csv + model.json.

Run:  python3 train_model.py
Output: predictions.csv, model.json
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

FEATURES = [
    "land_area_hectares",
    "affected_families",
    "compensation_disbursed_pct",
    "approval_stage_days",
    "legal_disputes_count",
    "rehabilitation_progress_pct",
    "documentation_completeness_pct",
    "stakeholder_responsiveness_score",
    "past_dept_delay_rate_pct",
]

# Human-readable labels + direction of "bad" for each feature, used to build
# plain-English explanations and recommendations.
FEATURE_INFO = {
    "land_area_hectares": {
        "label": "Land area",
        "bad_when": "high",
        "recommendation": "Break the acquisition into smaller phased parcels to speed up processing.",
    },
    "affected_families": {
        "label": "Number of affected families",
        "bad_when": "high",
        "recommendation": "Deploy additional rehabilitation & resettlement staff given the large affected population.",
    },
    "compensation_disbursed_pct": {
        "label": "Compensation disbursed",
        "bad_when": "low",
        "recommendation": "Expedite pending compensation disbursement; escalate to the district compensation cell.",
    },
    "approval_stage_days": {
        "label": "Time stuck in approval",
        "bad_when": "high",
        "recommendation": "Escalate pending approvals; flag for administrative fast-tracking.",
    },
    "legal_disputes_count": {
        "label": "Legal disputes",
        "bad_when": "high",
        "recommendation": "Prioritise legal mediation / fast-track court hearings to resolve disputes.",
    },
    "rehabilitation_progress_pct": {
        "label": "Rehabilitation progress",
        "bad_when": "low",
        "recommendation": "Accelerate resettlement and rehabilitation scheme implementation.",
    },
    "documentation_completeness_pct": {
        "label": "Documentation completeness",
        "bad_when": "low",
        "recommendation": "Complete pending land records / title documentation before proceeding further.",
    },
    "stakeholder_responsiveness_score": {
        "label": "Stakeholder responsiveness",
        "bad_when": "low",
        "recommendation": "Set up a dedicated coordination cell to improve inter-departmental responsiveness.",
    },
    "past_dept_delay_rate_pct": {
        "label": "Department's historical delay rate",
        "bad_when": "high",
        "recommendation": "Assign additional oversight/monitoring given this department's track record of delays.",
    },
}


def risk_bucket(p):
    if p >= 0.66:
        return "High"
    elif p >= 0.33:
        return "Medium"
    return "Low"


def main():
    df = pd.read_csv("land_projects.csv")
    X = df[FEATURES]
    y = df["delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test, test_probs)
    acc = accuracy_score(y_test, test_preds)
    print(f"Test AUC: {auc:.3f}   Test Accuracy: {acc:.3f}")

    # --- Predict + explain for ALL projects (so dashboard can show everything) ---
    all_probs = model.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # shape: (n_rows, n_features)

    results = []
    for i in range(len(df)):
        row_shap = shap_values[i]
        # Top 3 features pushing risk UP (positive SHAP = increases delay probability)
        order = np.argsort(row_shap)[::-1]
        top_factors = []
        for idx in order[:3]:
            fname = FEATURES[idx]
            contribution = float(row_shap[idx])
            if contribution <= 0:
                continue
            info = FEATURE_INFO[fname]
            top_factors.append({
                "feature": info["label"],
                "value": float(X.iloc[i][fname]),
                "contribution": round(contribution, 3),
                "recommendation": info["recommendation"],
            })

        results.append({
            "project_id": df.iloc[i]["project_id"],
            "state": df.iloc[i]["state"],
            "project_type": df.iloc[i]["project_type"],
            "delay_probability": round(float(all_probs[i]), 3),
            "risk_category": risk_bucket(all_probs[i]),
            "top_factor_1": top_factors[0]["feature"] if len(top_factors) > 0 else "N/A",
            "top_factor_1_recommendation": top_factors[0]["recommendation"] if len(top_factors) > 0 else "",
            "top_factor_2": top_factors[1]["feature"] if len(top_factors) > 1 else "",
            "top_factor_3": top_factors[2]["feature"] if len(top_factors) > 2 else "",
        })

    pred_df = pd.DataFrame(results)
    # Merge back the raw features so the dashboard detail view can show them
    full_df = pd.concat([df.reset_index(drop=True), pred_df.drop(columns=["project_id", "state", "project_type"])], axis=1)
    full_df.to_csv("predictions.csv", index=False)

    model.save_model("model.json")
    print("Saved predictions.csv and model.json")
    print(full_df["risk_category"].value_counts())


if __name__ == "__main__":
    main()
