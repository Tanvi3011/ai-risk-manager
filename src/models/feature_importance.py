"""Feature Importance — Explainability Engine.

Three complementary views of what drives fraud detection:
1. Supervised (Random Forest) — which features best predict labeled fraud
2. Permutation importance on Isolation Forest — which features most affect anomaly scores
3. Rule recall — which behavioral rules catch the most actual fraud

Also includes a supervised baseline (Random Forest) for comparison with Isolation Forest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def compute_feature_importances(df: pd.DataFrame) -> dict:
    """Compute all feature importance views.

    Returns a dict with:
      - rf_importances: dict of feature -> importance (Random Forest)
      - iso_importances: dict of feature -> permutation importance (Isolation Forest)
      - rule_recalls: dict of rule -> {triggered, total, recall}
      - rf_f1: Random Forest F1 on test set
      - iso_f1: Isolation Forest F1 on test set
      - top_features_summary: human-readable summary
    """
    # Prepare features
    from src.features.feature_engineering import get_feature_columns
    feature_cols = get_feature_columns(df)
    exclude = {"ml_top_features", "evidence_paths"}
    ml_features = [
        c for c in feature_cols
        if not c.startswith("rule_")
        and c not in exclude
        and df[c].dtype in ["float64", "int64", "bool", "float32", "int32"]
    ]

    X = df[ml_features].fillna(0).astype(float)
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Random Forest (supervised)
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_imp = pd.Series(rf.feature_importances_, index=ml_features).sort_values(
        ascending=False
    )

    # 2. Isolation Forest permutation importance
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    iso = IsolationForest(contamination=0.19, n_estimators=200, random_state=42)
    iso.fit(X_tr_s)

    iso_scores = -iso.decision_function(X_te_s)
    iso_threshold = np.percentile(iso_scores, 85)
    iso_pred = (iso_scores >= iso_threshold).astype(int)
    iso_f1 = f1_score(y_test, iso_pred)

    def iso_scorer(est, X, y_true):
        scores = -est.decision_function(X)
        t = np.percentile(scores, 18)
        return f1_score(y_true, (scores >= t).astype(int))

    perm = permutation_importance(
        iso, X_te_s, y_test, n_repeats=10, random_state=42, scoring=iso_scorer
    )
    iso_imp = pd.Series(perm.importances_mean, index=ml_features).sort_values(
        ascending=False
    )

    # 3. Rule recall
    rule_cols = [c for c in df.columns if c.startswith("rule_")]
    rule_data = {}
    total_fraud = int(df["is_fraud"].sum())
    for col in rule_cols:
        triggered = int(((df[col] == 1) & (df["is_fraud"] == 1)).sum())
        rule_data[col] = {
            "triggered": triggered,
            "total": total_fraud,
            "recall": round(triggered / total_fraud, 4),
        }

    # 4. Feature importance by category
    categories = {
        "Amount": ["amount", "log_amount", "amount_zscore",
                    "amount_deviation_from_payer_mean", "amount_ratio_to_payer_median"],
        "Velocity": ["velocity_count_5min", "velocity_count_1h", "velocity_count_24h",
                     "velocity_amount_5min", "velocity_amount_1h", "velocity_amount_24h"],
        "Device": ["device_account_count", "device_total_txn_count", "is_new_device"],
        "IP": ["ip_account_count", "ip_total_txn_count", "is_new_ip"],
        "Time": ["hour", "day_of_week", "is_odd_hour", "is_weekend",
                 "payer_mean_hour", "payer_std_hour", "hour_deviation"],
        "Payee": ["is_new_payee", "recent_new_payee_count_24h",
                  "unique_payees_24h", "recipient_concentration"],
        "Payer Stats": ["payer_mean_amount", "payer_std_amount", "payer_median_amount"],
    }

    category_importance = {}
    for cat, feats in categories.items():
        cat_imp = sum(rf_imp.get(f, 0) for f in feats if f in rf_imp.index)
        category_importance[cat] = round(cat_imp, 4)

    # Build summary
    summary_lines = []
    summary_lines.append("Random Forest (supervised) F1: {:.3f}".format(rf_f1))
    summary_lines.append("Isolation Forest (unsupervised) F1: {:.3f}".format(iso_f1))
    summary_lines.append("")
    summary_lines.append("Top 5 most important features (Random Forest):")
    for feat, imp in rf_imp.head(5).items():
        summary_lines.append("  {}: {:.4f}".format(feat, imp))
    summary_lines.append("")
    summary_lines.append("Feature importance by category:")
    for cat, imp in sorted(category_importance.items(), key=lambda x: -x[1]):
        summary_lines.append("  {}: {:.4f}".format(cat, imp))

    return {
        "rf_importances": {k: round(v, 4) for k, v in rf_imp.head(20).items()},
        "rf_f1": round(rf_f1, 4),
        "iso_importances": {k: round(v, 4) for k, v in iso_imp.head(20).items()},
        "iso_f1": round(iso_f1, 4),
        "rule_recalls": rule_data,
        "category_importance": category_importance,
        "summary": "\n".join(summary_lines),
    }


def save_feature_importances(df: pd.DataFrame, output_dir: Path | None = None) -> dict:
    """Compute and save feature importances."""
    result = compute_feature_importances(df)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "feature_importance.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved feature importances to {out_path}")

    return result


if __name__ == "__main__":
    from src.config import DATA_DIR

    df = pd.read_csv(
        DATA_DIR / "processed" / "features_with_graph.csv",
        parse_dates=["timestamp"],
    )
    result = save_feature_importances(df)
    print()
    print(result["summary"])
