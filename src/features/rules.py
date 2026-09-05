"""Baseline Rules & Signal Detection.

Simple interpretable rules that fire before ML.  Each rule is a boolean
or numeric signal stored as a new column.  Rules are NOT the fraud
detector — they produce *evidence signals* that the Detector Agent
combines with ML and graph scores.

Signals implemented:
  - high_amount             : amount > 3 σ above payer mean
  - high_velocity_5min      : ≥ 3 transactions within 5 minutes
  - high_velocity_1h        : ≥ 6 transactions within 1 hour
  - many_new_payees_24h     : ≥ 3 new payees in 24 hours
  - shared_device_multiuser : device used by ≥ 3 different accounts
  - shared_ip_multiuser     : IP used by ≥ 3 different accounts
  - odd_hour_high_value     : odd hour (0-5) AND amount > $5 000
  - circular_transfer_hint  : payee is also a payer with reverse edge
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# Individual Rule Functions
# =====================================================================

def rule_high_amount(df: pd.DataFrame) -> pd.Series:
    """Amount > 3 σ above the payer's own mean."""
    return (
        df["amount_zscore"].abs() > 3.0
    ).astype(int)


def rule_high_velocity_5min(df: pd.DataFrame) -> pd.Series:
    """≥ 3 transactions in the last 5 minutes for same payer."""
    return (df["velocity_count_5min"] >= 3).astype(int)


def rule_high_velocity_1h(df: pd.DataFrame) -> pd.Series:
    """≥ 6 transactions in the last hour for same payer."""
    return (df["velocity_count_1h"] >= 6).astype(int)


def rule_many_new_payees_24h(df: pd.DataFrame) -> pd.Series:
    """≥ 3 new payees seen in the last 24 hours."""
    return (df["recent_new_payee_count_24h"] >= 3).astype(int)


def rule_shared_device_multiuser(df: pd.DataFrame) -> pd.Series:
    """Device shared across ≥ 3 different accounts."""
    return (df["device_account_count"] >= 3).astype(int)


def rule_shared_ip_multiuser(df: pd.DataFrame) -> pd.Series:
    """IP shared across ≥ 3 different accounts."""
    return (df["ip_account_count"] >= 3).astype(int)


def rule_odd_hour_high_value(df: pd.DataFrame) -> pd.Series:
    """Transaction at odd hours (0-5) AND amount > $5 000."""
    return ((df["is_odd_hour"] == 1) & (df["amount"] > 5_000)).astype(int)


def rule_circular_transfer_hint(df: pd.DataFrame) -> pd.DataFrame:
    """Detect if (payer → payee) has a reverse edge (payee → payer) in the data.

    This is a lightweight circular-transfer hint without full graph analysis.
    """
    edges = set(zip(df["payer_id"], df["payee_id"]))
    reverse_exists = [
        int((payee, payer) in edges)
        for payer, payee in zip(df["payer_id"], df["payee_id"])
    ]
    return pd.Series(reverse_exists, index=df.index, name="circular_transfer_hint")


# =====================================================================
# Apply All Rules
# =====================================================================

ALL_RULE_COLUMNS: list[str] = [
    "rule_high_amount",
    "rule_high_velocity_5min",
    "rule_high_velocity_1h",
    "rule_many_new_payees_24h",
    "rule_shared_device_multiuser",
    "rule_shared_ip_multiuser",
    "rule_odd_hour_high_value",
    "rule_circular_transfer_hint",
]


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all baseline rules and add signal columns.

    Returns a copy with new ``rule_*`` columns appended.
    """
    out = df.copy()

    out["rule_high_amount"] = rule_high_amount(out)
    out["rule_high_velocity_5min"] = rule_high_velocity_5min(out)
    out["rule_high_velocity_1h"] = rule_high_velocity_1h(out)
    out["rule_many_new_payees_24h"] = rule_many_new_payees_24h(out)
    out["rule_shared_device_multiuser"] = rule_shared_device_multiuser(out)
    out["rule_shared_ip_multiuser"] = rule_shared_ip_multiuser(out)
    out["rule_odd_hour_high_value"] = rule_odd_hour_high_value(out)
    out["rule_circular_transfer_hint"] = rule_circular_transfer_hint(out)

    # Aggregate signal count
    out["rules_triggered_count"] = out[ALL_RULE_COLUMNS].sum(axis=1)

    return out


# =====================================================================
# Sanity Check
# =====================================================================

def validate_rules(df: pd.DataFrame) -> None:
    """Print recall of each rule on known fraud labels (for sanity only)."""
    fraud_mask = df["is_fraud"] == 1
    n_fraud = fraud_mask.sum()

    print("\n--- Rule Recall on Known Fraud ---")
    for col in ALL_RULE_COLUMNS:
        triggered_fraud = (df.loc[fraud_mask, col] == 1).sum()
        recall = triggered_fraud / n_fraud if n_fraud > 0 else 0
        print(f"  {col:40s}  {triggered_fraud:>5d} / {n_fraud}  ({recall:.1%})")

    print("\n--- Rule Trigger Counts (all data) ---")
    for col in ALL_RULE_COLUMNS:
        count = (df[col] == 1).sum()
        print(f"  {col:40s}  {count:>5d}")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    from src.config import DATA_DIR

    feat_path = DATA_DIR / "processed" / "features.csv"
    out_path = DATA_DIR / "processed" / "features_with_rules.csv"

    if not feat_path.exists():
        print(f"ERROR: {feat_path} not found. Run feature_engineering.py first.")
        sys.exit(1)

    print(f"Loading features from {feat_path} ...")
    df = pd.read_csv(feat_path, parse_dates=["timestamp"])

    print("Applying baseline rules ...")
    df_rules = apply_rules(df)

    validate_rules(df_rules)

    df_rules.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Total columns: {len(df_rules.columns)}")
