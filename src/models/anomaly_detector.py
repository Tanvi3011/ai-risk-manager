"""ML Anomaly Detection — Isolation Forest.

Trains on behavioural features only (never on is_fraud or fraud_pattern).
Produces:
  - anomaly_score  (0-1 normalised, higher = more anomalous)
  - anomaly_flag   (1 if above alert threshold)
  - top_contributing_features per transaction
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engineering import get_feature_columns


# =====================================================================
# Train / Predict
# =====================================================================

def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.19,
    random_state: int = 42,
    n_estimators: int = 200,
) -> tuple[IsolationForest, StandardScaler]:
    """Fit Isolation Forest on the training feature matrix."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model, scaler


def predict_anomaly(
    model: IsolationForest,
    scaler: StandardScaler,
    X: pd.DataFrame,
    feature_names: list[str],
    alert_percentile: float = 85,
) -> pd.DataFrame:
    """Score transactions and flag anomalies.

    Returns DataFrame with columns:
      - anomaly_score (0-1, higher = riskier)
      - anomaly_flag  (1 if above alert threshold)
      - top_features  (JSON list of top contributing features)
    """
    X_scaled = scaler.transform(X)

    # Raw decision function: more negative = more anomalous
    raw_scores = model.decision_function(X_scaled)

    # Normalise to 0-1 (1 = most anomalous)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s - min_s == 0:
        anomaly_score = np.zeros(len(raw_scores))
    else:
        anomaly_score = 1.0 - (raw_scores - min_s) / (max_s - min_s)

    # Alert threshold based on percentile
    threshold = np.percentile(anomaly_score, alert_percentile)
    anomaly_flag = (anomaly_score >= threshold).astype(int)

    # Top contributing features: features where value is most extreme
    top_features_list = []
    for i in range(len(X)):
        row = X.iloc[i].values
        # Absolute z-score contribution
        abs_vals = np.abs(row)
        top_idx = np.argsort(abs_vals)[-5:][::-1]
        top_feats = [feature_names[j] for j in top_idx if abs_vals[j] > 0]
        top_features_list.append(top_feats[:3])

    result = pd.DataFrame({
        "anomaly_score": anomaly_score,
        "anomaly_flag": anomaly_flag,
        "top_features": [json.dumps(f) for f in top_features_list],
    }, index=X.index)

    return result


# =====================================================================
# Evaluation
# =====================================================================

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """Compute and print classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"Evaluation {label}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, digits=3))
    print("Confusion Matrix:")
    print(cm)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from src.config import DATA_DIR, RANDOM_SEED

    feat_path = DATA_DIR / "processed" / "features_with_rules.csv"

    if not feat_path.exists():
        print("ERROR: Run rules.py first.")
        sys.exit(1)

    print(f"Loading features from {feat_path} ...")
    df = pd.read_csv(feat_path, parse_dates=["timestamp"])

    feature_cols = get_feature_columns(df)
    # Exclude rule columns from ML features — rules are used by Detector
    ml_features = [c for c in feature_cols if not c.startswith("rule_")]
    print(f"ML features ({len(ml_features)}): {ml_features}")

    X = df[ml_features].fillna(0)
    y = df["is_fraud"].values

    # Time-based split: first 80% train, last 20% test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTrain: {len(X_train)} rows  |  Test: {len(X_test)} rows")
    print(f"Train fraud rate: {y_train.mean():.2%}")
    print(f"Test fraud rate:  {y_test.mean():.2%}")

    # Train
    contamination = y_train.mean()
    print(f"\nTraining Isolation Forest (contamination={contamination:.3f}) ...")
    model, scaler = train_isolation_forest(X_train, contamination=contamination)

    # Predict on test
    results_test = predict_anomaly(model, scaler, X_test, ml_features)
    metrics_test = evaluate(y_test, results_test["anomaly_flag"].values, "(test set)")

    # Also predict on full dataset for downstream use
    results_all = predict_anomaly(model, scaler, X, ml_features)
    metrics_all = evaluate(y, results_all["anomaly_flag"].values, "(full dataset)")

    # Save
    df_out = df.copy()
    df_out["anomaly_score"] = results_all["anomaly_score"].values
    df_out["anomaly_flag"] = results_all["anomaly_flag"].values
    df_out["ml_top_features"] = results_all["top_features"].values

    out_path = DATA_DIR / "processed" / "features_with_ml.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
