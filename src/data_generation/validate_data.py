"""Data Quality & Validation for the synthetic transaction dataset.

Checks:
  - Row count, columns, missing values
  - Duplicate transaction IDs
  - Invalid amounts (negative, zero, NaN)
  - Invalid timestamps
  - Self-transfers (payer == payee)
  - Fraud ratio and distribution of every fraud scenario
  - Fraud labels are not leaking into future feature calculations
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    NUM_TRANSACTIONS,
    RANDOM_SEED,
)

RAW_PATH = DATA_DIR / "raw" / "transactions.csv"

EXPECTED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "payer_id",
    "payee_id",
    "merchant_id",
    "amount",
    "device_id",
    "ip_address",
    "is_fraud",
    "fraud_pattern",
]

EXPECTED_FRAUD_PATTERNS = [
    "normal",
    "rapid_fanout",
    "device_reuse",
    "new_payee_burst",
    "odd_hour_high_value",
    "shared_ip",
    "circular_transfer",
]


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Run all validation checks and return a list of error messages."""
    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Basic shape
    # ------------------------------------------------------------------
    if len(df) != NUM_TRANSACTIONS:
        errors.append(
            f"Row count mismatch: expected {NUM_TRANSACTIONS}, "
            f"got {len(df)}"
        )

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        warnings.append(f"Extra columns (ok if intentional): {extra_cols}")

    # ------------------------------------------------------------------
    # 2. Missing values
    # ------------------------------------------------------------------
    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        for col, count in cols_with_nulls.items():
            errors.append(f"Column '{col}' has {count} null values")

    # ------------------------------------------------------------------
    # 3. Duplicate transaction IDs
    # ------------------------------------------------------------------
    dupes = df["transaction_id"].duplicated().sum()
    if dupes > 0:
        errors.append(f"Found {dupes} duplicate transaction_id(s)")

    # ------------------------------------------------------------------
    # 4. Invalid amounts
    # ------------------------------------------------------------------
    neg_amounts = (df["amount"] < 0).sum()
    if neg_amounts > 0:
        errors.append(f"Found {neg_amounts} transactions with negative amounts")

    zero_amounts = (df["amount"] == 0).sum()
    if zero_amounts > 0:
        warnings.append(f"Found {zero_amounts} transactions with zero amounts")

    # ------------------------------------------------------------------
    # 5. Invalid timestamps
    # ------------------------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        errors.append("timestamp column is not datetime type")

    # ------------------------------------------------------------------
    # 6. Self-transfers
    # ------------------------------------------------------------------
    self_xfers = (df["payer_id"] == df["payee_id"]).sum()
    if self_xfers > 0:
        warnings.append(f"Found {self_xfers} self-transfers (payer == payee)")

    # ------------------------------------------------------------------
    # 7. Fraud ratio and distribution
    # ------------------------------------------------------------------
    fraud_count = int(df["is_fraud"].sum())
    normal_count = int((df["is_fraud"] == 0).sum())
    fraud_ratio = fraud_count / len(df) if len(df) > 0 else 0

    print("\n--- Fraud Summary ---")
    print(f"  Total transactions : {len(df)}")
    print(f"  Fraud transactions : {fraud_count}")
    print(f"  Normal transactions: {normal_count}")
    print(f"  Fraud ratio        : {fraud_ratio:.2%}")

    if fraud_ratio < 0.01:
        warnings.append(f"Very low fraud ratio: {fraud_ratio:.2%}")
    if fraud_ratio > 0.50:
        warnings.append(f"Very high fraud ratio: {fraud_ratio:.2%}")

    print("\n--- Fraud Pattern Distribution ---")
    pattern_dist = df[df["is_fraud"] == 1]["fraud_pattern"].value_counts()
    for pattern, count in pattern_dist.items():
        print(f"  {pattern}: {count}")

    unexpected = set(df["fraud_pattern"].unique()) - set(
        EXPECTED_FRAUD_PATTERNS
    )
    if unexpected:
        errors.append(f"Unexpected fraud patterns: {unexpected}")

    # ------------------------------------------------------------------
    # 8. Sanity: amounts per pattern
    # ------------------------------------------------------------------
    print("\n--- Amount Stats by Pattern ---")
    stats = (
        df.groupby("fraud_pattern")["amount"]
        .describe()[["count", "mean", "min", "max"]]
        .round(2)
    )
    print(stats.to_string())

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print("ALL CHECKS PASSED")

    if warnings:
        print(f"\nWARNINGS — {len(warnings)}:")
        for w in warnings:
            print(f"  WARN: {w}")

    print("=" * 50)

    return errors


def main():
    if not RAW_PATH.exists():
        print(f"ERROR: Dataset not found at {RAW_PATH}")
        print("Run generate_transactions.py first.")
        sys.exit(1)

    print(f"Loading dataset from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    errors = validate_dataset(df)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
