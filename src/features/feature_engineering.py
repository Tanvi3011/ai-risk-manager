"""Feature Engineering for transaction risk intelligence.

Converts raw transactions into a behavioural feature matrix.

Feature groups:
  - Amount features: log amount, deviation from payer's normal, z-score
  - Velocity features: count & sum over 5-min, 1-hour, 24-hour windows
  - Payee features: new-payee flag, recent new payees, recipient concentration
  - Device features: accounts per device, device count, device novelty
  - IP features: accounts per IP, IP count, IP novelty
  - Time features: hour, odd-hour flag, weekend flag, time deviation

Fraud labels (is_fraud, fraud_pattern) are NEVER used as model inputs.
They are kept alongside for evaluation only.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# Amount Features
# =====================================================================

def build_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log amount, payer-mean deviation, payer z-score."""
    out = df.copy()
    out["log_amount"] = np.log1p(out["amount"])

    payer_stats = out.groupby("payer_id")["amount"].agg(
        payer_mean_amount="mean",
        payer_std_amount="std",
        payer_median_amount="median",
    )
    out = out.merge(payer_stats, on="payer_id", how="left")
    out["payer_std_amount"] = out["payer_std_amount"].fillna(0)

    out["amount_deviation_from_payer_mean"] = (
        out["amount"] - out["payer_mean_amount"]
    )
    out["amount_zscore"] = np.where(
        out["payer_std_amount"] > 0,
        (out["amount"] - out["payer_mean_amount"]) / out["payer_std_amount"],
        0.0,
    )
    out["amount_ratio_to_payer_median"] = out["amount"] / out[
        "payer_median_amount"
    ].replace(0, np.nan)
    out["amount_ratio_to_payer_median"] = out[
        "amount_ratio_to_payer_median"
    ].fillna(1.0)

    return out


# =====================================================================
# Velocity Features
# =====================================================================

def build_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling-window transaction count and total amount per payer."""
    out = df.sort_values("timestamp").copy()

    windows = {
        "5min": pd.Timedelta(minutes=5),
        "1h": pd.Timedelta(hours=1),
        "24h": pd.Timedelta(hours=24),
    }

    for label, delta in windows.items():
        count_col = f"velocity_count_{label}"
        amount_col = f"velocity_amount_{label}"
        out[count_col] = 0
        out[amount_col] = 0.0

        for payer, group in out.groupby("payer_id"):
            idx = group.index
            ts = group["timestamp"]
            amt = group["amount"]

            for i in range(len(group)):
                window_start = ts.iloc[i] - delta
                mask = (ts >= window_start) & (ts <= ts.iloc[i])
                # Exclude current txn from velocity windows
                # (only count previous transactions)
                mask_before = (ts >= window_start) & (ts < ts.iloc[i])
                out.loc[idx[i], count_col] = mask_before.sum()
                out.loc[idx[i], amount_col] = amt[mask_before].sum()

    return out


# =====================================================================
# Payee Features
# =====================================================================

def build_payee_features(df: pd.DataFrame) -> pd.DataFrame:
    """New-payee detection and recipient concentration."""
    out = df.sort_values("timestamp").copy()

    out["is_new_payee"] = 0
    out["recent_new_payee_count_24h"] = 0
    out["unique_payees_24h"] = 0
    out["recipient_concentration"] = 0.0

    for payer, group in out.groupby("payer_id"):
        idx = group.index
        ts = group["timestamp"]
        payees = group["payee_id"]

        seen_payees: set[str] = set()
        new_payee_flags: list[int] = []
        recent_new_counts: list[int] = []
        unique_payee_counts: list[int] = []
        concentrations: list[float] = []

        for i in range(len(group)):
            current_payee = payees.iloc[i]
            current_time = ts.iloc[i]

            # Is this a new payee for this payer?
            is_new = int(current_payee not in seen_payees)
            new_payee_flags.append(is_new)
            seen_payees.add(current_payee)

            # Count new payees and unique payees in last 24h
            window_start = current_time - pd.Timedelta(hours=24)
            window_mask = (ts >= window_start) & (ts < current_time)
            window_payees = set(payees[window_mask])
            new_in_window = sum(
                1 for p in payees[window_mask]
                if p not in seen_payees - {current_payee}
            )
            recent_new_counts.append(
                sum(new_payee_flags[j] for j in range(max(0, i - 500), i)
                    if ts.iloc[j] >= window_start and ts.iloc[j] < current_time)
            )
            unique_payee_counts.append(len(window_payees) + 1)

            # Recipient concentration: fraction going to most popular payee
            all_payees_to_date = payees[:i + 1]
            if len(all_payees_to_date) > 0:
                top_payee_frac = (
                    all_payees_to_date.value_counts().iloc[0]
                    / len(all_payees_to_date)
                )
            else:
                top_payee_frac = 0.0
            concentrations.append(top_payee_frac)

        out.loc[idx, "is_new_payee"] = new_payee_flags
        out.loc[idx, "recent_new_payee_count_24h"] = recent_new_counts
        out.loc[idx, "unique_payees_24h"] = unique_payee_counts
        out.loc[idx, "recipient_concentration"] = concentrations

    return out


# =====================================================================
# Device Features
# =====================================================================

def build_device_features(df: pd.DataFrame) -> pd.DataFrame:
    """Accounts-per-device, device transaction count, device novelty."""
    out = df.copy()

    device_account_map = out.groupby("device_id")["payer_id"].nunique()
    out["device_account_count"] = out["device_id"].map(device_account_map)

    device_txn_count = out.groupby("device_id")["transaction_id"].count()
    out["device_total_txn_count"] = out["device_id"].map(device_txn_count)

    # Device novelty: is this the first time this payer uses this device?
    out = out.sort_values("timestamp")
    out["is_new_device"] = 0
    for payer, group in out.groupby("payer_id"):
        idx = group.index
        seen_devices: set[str] = set()
        flags: list[int] = []
        for dev in group["device_id"]:
            is_new = int(dev not in seen_devices)
            flags.append(is_new)
            seen_devices.add(dev)
        out.loc[idx, "is_new_device"] = flags

    return out


# =====================================================================
# IP Features
# =====================================================================

def build_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Accounts-per-IP, IP transaction count, IP novelty."""
    out = df.copy()

    ip_account_map = out.groupby("ip_address")["payer_id"].nunique()
    out["ip_account_count"] = out["ip_address"].map(ip_account_map)

    ip_txn_count = out.groupby("ip_address")["transaction_id"].count()
    out["ip_total_txn_count"] = out["ip_address"].map(ip_txn_count)

    # IP novelty
    out = out.sort_values("timestamp")
    out["is_new_ip"] = 0
    for payer, group in out.groupby("payer_id"):
        idx = group.index
        seen_ips: set[str] = set()
        flags: list[int] = []
        for ip in group["ip_address"]:
            is_new = int(ip not in seen_ips)
            flags.append(is_new)
            seen_ips.add(ip)
        out.loc[idx, "is_new_ip"] = flags

    return out


# =====================================================================
# Time Features
# =====================================================================

def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour, odd-hour flag, weekend flag, deviation from payer's usual hours."""
    out = df.sort_values("timestamp").copy()

    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["is_odd_hour"] = out["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

    # Deviation from payer's usual activity hours
    payer_hour_stats = out.groupby("payer_id")["hour"].agg(
        payer_mean_hour="mean",
        payer_std_hour="std",
    )
    out = out.merge(payer_hour_stats, on="payer_id", how="left")
    out["payer_std_hour"] = out["payer_std_hour"].fillna(0)
    out["hour_deviation"] = np.where(
        out["payer_std_hour"] > 0,
        (out["hour"] - out["payer_mean_hour"]) / out["payer_std_hour"],
        0.0,
    )

    return out


# =====================================================================
# Master Pipeline
# =====================================================================

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature builders and return a clean feature table.

    Returns a DataFrame with:
      - transaction_id, timestamp, payer_id, payee_id, merchant_id,
        device_id, ip_address (raw identifiers for graph analysis)
      - is_fraud, fraud_pattern (labels for evaluation only)
      - All engineered features
    """
    print("Building amount features ...")
    out = build_amount_features(df)

    print("Building velocity features ...")
    out = build_velocity_features(out)

    print("Building payee features ...")
    out = build_payee_features(out)

    print("Building device features ...")
    out = build_device_features(out)

    print("Building IP features ...")
    out = build_ip_features(out)

    print("Building time features ...")
    out = build_time_features(out)

    print(f"Feature matrix shape: {out.shape}")
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return only the feature column names (no IDs, no labels)."""
    label_cols = {"is_fraud", "fraud_pattern"}
    id_cols = {
        "transaction_id", "timestamp", "payer_id", "payee_id",
        "merchant_id", "device_id", "ip_address",
    }
    return [c for c in df.columns if c not in label_cols and c not in id_cols]


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    from src.config import DATA_DIR

    raw_path = DATA_DIR / "raw" / "transactions.csv"
    out_path = DATA_DIR / "processed" / "features.csv"

    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run generate_transactions.py first.")
        sys.exit(1)

    print(f"Loading {raw_path} ...")
    df = pd.read_csv(raw_path, parse_dates=["timestamp"])

    features = build_feature_matrix(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    print(f"\nSaved feature matrix to {out_path}")
    print(f"Columns: {list(features.columns)}")
    print(f"Feature count: {len(get_feature_columns(features))}")
