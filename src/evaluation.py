"""Evaluation & Demo Readiness.

1. Metrics: precision, recall, F1, confusion matrix
2. Critic before/after comparison
3. Scripted demo cases
4. Calibration metrics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def score_to_level(score: float) -> str:
    if score >= 80:
        return "CRITICAL"
    elif score >= 55:
        return "HIGH"
    elif score >= 30:
        return "MEDIUM"
    return "LOW"


def level_to_binary(level: str) -> int:
    """HIGH/CRITICAL = 1 (flagged), LOW/MEDIUM = 0."""
    return 1 if level in ("HIGH", "CRITICAL") else 0


def evaluate_system(
    df: pd.DataFrame,
    predictions: np.ndarray,
    labels: np.ndarray,
    split_name: str = "Full Dataset",
) -> dict:
    """Compute and print comprehensive metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, predictions)

    print(f"\n{'='*60}")
    print(f"  SYSTEM EVALUATION — {split_name}")
    print(f"{'='*60}")
    print(classification_report(labels, predictions, digits=3,
          target_names=["Normal", "Fraud"]))
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]:>5d}  FP={cm[0][1]:>5d}")
    print(f"  FN={cm[1][0]:>5d}  TP={cm[1][1]:>5d}")

    return {
        "split": split_name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": int(cm[1][1]),
        "false_positives": int(cm[0][1]),
        "true_negatives": int(cm[0][0]),
        "false_negatives": int(cm[1][0]),
        "total_flagged": int(cm[0][1] + cm[1][1]),
        "total_fraud": int(cm[1][0] + cm[1][1]),
    }


def evaluate_critic_impact(df: pd.DataFrame, explanations: dict) -> dict:
    """Compare detection before and after Critic."""
    labels = df["is_fraud"].values

    # Before Critic: use anomaly scores to derive binary predictions
    pre_scores = df["anomaly_score"].values * 100
    pre_predictions = np.array([level_to_binary(score_to_level(s)) for s in pre_scores])

    # After Critic: use explanation risk levels
    post_levels = []
    for tid in df["transaction_id"]:
        exp = explanations.get(tid, {})
        post_levels.append(exp.get("risk_level", "LOW"))
    post_predictions = np.array([level_to_binary(l) for l in post_levels])

    # Metrics before
    pre_p, pre_r, pre_f1, _ = precision_recall_fscore_support(
        labels, pre_predictions, average="binary", zero_division=0
    )
    pre_cm = confusion_matrix(labels, pre_predictions)
    pre_fp = int(pre_cm[0][1])

    # Metrics after
    post_p, post_r, post_f1, _ = precision_recall_fscore_support(
        labels, post_predictions, average="binary", zero_division=0
    )
    post_cm = confusion_matrix(labels, post_predictions)
    post_fp = int(post_cm[0][1])

    # Downgrade analysis
    n_downgraded = sum(
        1 for e in explanations.values()
        if "downgraded" in e.get("critic_note", "").lower()
    )

    results = {
        "before_critic": {
            "precision": round(pre_p, 4),
            "recall": round(pre_r, 4),
            "f1": round(pre_f1, 4),
            "false_positives": pre_fp,
        },
        "after_critic": {
            "precision": round(post_p, 4),
            "recall": round(post_r, 4),
            "f1": round(post_f1, 4),
            "false_positives": post_fp,
        },
        "improvement": {
            "precision_delta": round(post_p - pre_p, 4),
            "recall_delta": round(post_r - pre_r, 4),
            "f1_delta": round(post_f1 - pre_f1, 4),
            "fp_reduction": pre_fp - post_fp,
            "fp_reduction_pct": round((pre_fp - post_fp) / pre_fp * 100, 1) if pre_fp > 0 else 0,
        },
        "critic": {
            "total_downgraded": n_downgraded,
            "downgrade_rate": round(n_downgraded / len(explanations), 4) if explanations else 0,
        },
    }

    print(f"\n{'='*60}")
    print(f"  CRITIC IMPACT ANALYSIS")
    print(f"{'='*60}")
    print(f"  {'Metric':<25s} {'Before':>10s} {'After':>10s} {'Delta':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'Precision':<25s} {pre_p:>10.3f} {post_p:>10.3f} {post_p-pre_p:>+10.3f}")
    print(f"  {'Recall':<25s} {pre_r:>10.3f} {post_r:>10.3f} {post_r-pre_r:>+10.3f}")
    print(f"  {'F1':<25s} {pre_f1:>10.3f} {post_f1:>10.3f} {post_f1-pre_f1:>+10.3f}")
    print(f"  {'False Positives':<25s} {pre_fp:>10d} {post_fp:>10d} {post_fp-pre_fp:>+10d}")
    print(f"  {'Cases Downgraded':<25s} {'—':>10s} {'—':>10s} {n_downgraded:>+10d}")

    return results


def evaluate_by_fraud_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate detection per fraud pattern."""
    if "fraud_pattern" not in df.columns or "anomaly_flag" not in df.columns:
        return pd.DataFrame()

    results = []
    for pattern in df["fraud_pattern"].unique():
        mask = df["fraud_pattern"] == pattern
        n = mask.sum()
        detected = (df.loc[mask, "anomaly_flag"] == 1).sum()
        results.append({
            "pattern": pattern,
            "count": n,
            "detected": detected,
            "detection_rate": round(detected / n, 3) if n > 0 else 0,
        })
    return pd.DataFrame(results).sort_values("detection_rate", ascending=False)


def find_demo_cases(df: pd.DataFrame, explanations: dict) -> list[dict]:
    """Find scripted demo cases for the 3-minute live demo."""
    demo_cases = []

    # 1. Normal
    normal = df[(df["is_fraud"] == 0) & (df["anomaly_score"] < 0.3)]
    if not normal.empty:
        row = normal.iloc[len(normal) // 2]
        demo_cases.append({
            "case_type": "normal_transaction",
            "description": "Normal low-risk transaction",
            "transaction_id": row["transaction_id"],
            "amount": float(row["amount"]),
            "expected_risk": "LOW",
        })

    # 2. Odd hour high value
    odd = df[(df.get("rule_odd_hour_high_value", pd.Series([0])) == 1) & (df["is_fraud"] == 1)]
    if not odd.empty:
        row = odd.nlargest(1, "amount").iloc[0]
        demo_cases.append({
            "case_type": "odd_hour_high_value",
            "description": "High-value odd-hour transaction",
            "transaction_id": row["transaction_id"],
            "amount": float(row["amount"]),
            "expected_risk": "HIGH/CRITICAL",
        })

    # 3. Device/IP cluster
    cluster = df[
        (df.get("device_account_count", pd.Series([0])) >= 4)
        & (df.get("ip_account_count", pd.Series([0])) >= 4)
        & (df["is_fraud"] == 1)
    ]
    if not cluster.empty:
        row = cluster.iloc[0]
        demo_cases.append({
            "case_type": "device_ip_cluster",
            "description": "Device/IP sharing cluster",
            "transaction_id": row["transaction_id"],
            "amount": float(row["amount"]),
            "expected_risk": "HIGH/CRITICAL",
        })

    # 4. Velocity burst
    rapid = df[(df.get("velocity_count_5min", pd.Series([0])) >= 3) & (df["is_fraud"] == 1)]
    if not rapid.empty:
        row = rapid.nlargest(1, "velocity_count_5min").iloc[0]
        demo_cases.append({
            "case_type": "rapid_fanout",
            "description": "Rapid burst of transactions",
            "transaction_id": row["transaction_id"],
            "amount": float(row["amount"]),
            "expected_risk": "HIGH/CRITICAL",
        })

    # 5. Circular transfer
    circular = df[(df.get("rule_circular_transfer_hint", pd.Series([0])) == 1) & (df["is_fraud"] == 1)]
    if not circular.empty:
        row = circular.iloc[0]
        demo_cases.append({
            "case_type": "circular_transfer",
            "description": "Circular money flow",
            "transaction_id": row["transaction_id"],
            "amount": float(row["amount"]),
            "expected_risk": "HIGH/CRITICAL",
        })

    return demo_cases


def run_evaluation():
    """Run full evaluation."""
    from src.config import DATA_DIR

    print("Loading data ...")
    df = pd.read_csv(DATA_DIR / "processed" / "features_with_graph.csv",
                     parse_dates=["timestamp"])

    with open(DATA_DIR / "processed" / "explanations.json") as f:
        explanations = {e["transaction_id"]: e for e in json.load(f)}

    # ML-only metrics
    predictions = df["anomaly_flag"].values
    labels = df["is_fraud"].values
    metrics = evaluate_system(df, predictions, labels, "Full Dataset")

    # Per-pattern
    pattern_metrics = evaluate_by_fraud_pattern(df)
    if not pattern_metrics.empty:
        print("\n--- Detection by Fraud Pattern ---")
        print(pattern_metrics.to_string(index=False))

    # Critic impact
    critic_results = evaluate_critic_impact(df, explanations)

    # Demo cases
    demo_cases = find_demo_cases(df, explanations)
    print(f"\n--- {len(demo_cases)} Demo Cases Ready ---")
    for i, case in enumerate(demo_cases, 1):
        print(f"  {i}. {case['case_type']}: {case['transaction_id']} (${case['amount']:,.2f})")

    # Save
    out = {
        "ml_metrics": metrics,
        "critic_impact": critic_results,
        "demo_cases": demo_cases,
    }
    out_path = DATA_DIR / "processed" / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_evaluation()
