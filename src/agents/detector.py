"""Detector Agent — Deterministic Risk Aggregation.

Combines ML anomaly score, baseline rules, and graph-risk signals
into a single case object per transaction.

The LLM does NOT invent risk numbers — every claim is traceable to a
concrete signal or graph relationship.

Outputs per transaction:
  - risk_score    (0-100)
  - risk_level    (LOW / MEDIUM / HIGH / CRITICAL)
  - evidence_list (structured evidence items)
  - triggered_signals (which rules/ML/graph signals fired)
  - recommended_action
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# Risk Case Dataclass
# =====================================================================

@dataclass
class RiskCase:
    transaction_id: str
    payer_id: str
    payee_id: str
    amount: float
    timestamp: str

    # Component scores (0-1)
    ml_score: float = 0.0
    graph_score: float = 0.0
    rules_score: float = 0.0

    # Final output
    risk_score: float = 0.0          # 0-100
    risk_level: str = "LOW"
    evidence_list: list[dict] = field(default_factory=list)
    triggered_signals: list[str] = field(default_factory=list)
    recommended_action: str = "No action required"

    # Original detector output (before Critic adjustment)
    original_risk_score: float = 0.0
    original_risk_level: str = "LOW"


# =====================================================================
# Risk Aggregation Logic
# =====================================================================

# Weights for combining component scores
WEIGHTS = {
    "ml": 0.35,
    "graph": 0.30,
    "rules": 0.35,
}

# Risk level thresholds (on 0-100 scale)
THRESHOLDS = {
    "LOW": 0,
    "MEDIUM": 30,
    "HIGH": 55,
    "CRITICAL": 80,
}


def compute_risk_score(
    ml_score: float,
    graph_score: float,
    rules_score: float,
) -> float:
    """Deterministic weighted combination of component scores (0-100)."""
    raw = (
        WEIGHTS["ml"] * ml_score
        + WEIGHTS["graph"] * graph_score
        + WEIGHTS["rules"] * rules_score
    )
    return round(min(100.0, raw * 100), 2)


def risk_level_from_score(score: float) -> str:
    """Map risk score to a risk level."""
    if score >= THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif score >= THRESHOLDS["HIGH"]:
        return "HIGH"
    elif score >= THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def recommended_action_from_level(level: str) -> str:
    actions = {
        "LOW": "No action required — monitor normally.",
        "MEDIUM": "Flag for review — queue for analyst inspection within 24h.",
        "HIGH": "Immediate review recommended — suspend pending investigation.",
        "CRITICAL": "Urgent: freeze account and escalate to fraud team.",
    }
    return actions.get(level, "No action required.")


# =====================================================================
# Evidence Builder
# =====================================================================

def build_evidence_list(row: pd.Series) -> list[dict]:
    """Extract structured evidence items from a feature row."""
    evidence = []

    # ML evidence
    if row.get("anomaly_score", 0) > 0.7:
        evidence.append({
            "type": "ml",
            "signal": "high_anomaly_score",
            "value": round(float(row["anomaly_score"]), 3),
            "description": f"Anomaly score {row['anomaly_score']:.3f} (top 15% of all transactions).",
        })

    top_feats = row.get("ml_top_features", "[]")
    if isinstance(top_feats, str):
        try:
            top_feats = json.loads(top_feats)
        except Exception:
            top_feats = []

    if top_feats:
        evidence.append({
            "type": "ml",
            "signal": "top_features",
            "value": top_feats,
            "description": f"Top contributing features: {', '.join(top_feats[:3])}.",
        })

    # Rule evidence
    rule_map = {
        "rule_high_amount": ("rule", "Unusually high amount (>3σ above payer mean)."),
        "rule_high_velocity_5min": ("rule", "≥3 transactions in last 5 minutes."),
        "rule_high_velocity_1h": ("rule", "≥6 transactions in last hour."),
        "rule_many_new_payees_24h": ("rule", "≥3 new payees in last 24 hours."),
        "rule_shared_device_multiuser": ("rule", "Device shared across multiple accounts."),
        "rule_shared_ip_multiuser": ("rule", "IP address shared across multiple accounts."),
        "rule_odd_hour_high_value": ("rule", "High-value transaction at odd hours (0-5 AM)."),
        "rule_circular_transfer_hint": ("rule", "Possible circular transfer detected."),
    }
    for col, (rtype, desc) in rule_map.items():
        if row.get(col, 0) == 1:
            evidence.append({
                "type": rtype,
                "signal": col,
                "value": True,
                "description": desc,
            })

    # Graph evidence
    if row.get("graph_risk_score", 0) > 0.3:
        evidence.append({
            "type": "graph",
            "signal": "high_graph_risk",
            "value": round(float(row["graph_risk_score"]), 3),
            "description": f"Graph risk score {row['graph_risk_score']:.3f}.",
        })
    if row.get("shared_device_count", 0) > 0:
        evidence.append({
            "type": "graph",
            "signal": "shared_device_in_graph",
            "value": int(row["shared_device_count"]),
            "description": f"Account shares device with {int(row['shared_device_count'])} other account(s).",
        })
    if row.get("shared_ip_count", 0) > 0:
        evidence.append({
            "type": "graph",
            "signal": "shared_ip_in_graph",
            "value": int(row["shared_ip_count"]),
            "description": f"Account shares IP with {int(row['shared_ip_count'])} other account(s).",
        })
    if row.get("in_short_cycle", 0) == 1:
        evidence.append({
            "type": "graph",
            "signal": "in_cycle",
            "value": True,
            "description": "Account is part of a short cycle (potential circular transfer).",
        })

    # Transaction context
    evidence.append({
        "type": "context",
        "signal": "transaction_context",
        "value": {
            "amount": float(row["amount"]),
            "hour": int(row.get("hour", 0)),
            "is_odd_hour": bool(row.get("is_odd_hour", 0)),
            "is_weekend": bool(row.get("is_weekend", 0)),
        },
        "description": (
            f"${row['amount']:,.2f} at {int(row.get('hour', 0)):02d}:00"
            f"{' (odd hour)' if row.get('is_odd_hour', 0) else ''}"
            f"{' (weekend)' if row.get('is_weekend', 0) else ''}."
        ),
    })

    return evidence


# =====================================================================
# Case Builder
# =====================================================================

def build_case(row: pd.Series) -> RiskCase:
    """Build a full RiskCase for a single transaction row."""
    ml_score = float(row.get("anomaly_score", 0))
    graph_score = float(row.get("graph_risk_score", 0))

    # Rules score: fraction of applicable rules triggered
    rule_cols = [c for c in row.index if c.startswith("rule_")]
    n_triggered = sum(1 for c in rule_cols if row.get(c, 0) == 1)
    rules_score = min(1.0, n_triggered / 4)  # 4+ rules = max score

    risk_score = compute_risk_score(ml_score, graph_score, rules_score)
    risk_level = risk_level_from_score(risk_score)

    evidence = build_evidence_list(row)

    triggered = []
    for c in rule_cols:
        if row.get(c, 0) == 1:
            triggered.append(c)
    if ml_score > 0.7:
        triggered.append("ml_anomaly_flag")
    if graph_score > 0.3:
        triggered.append("graph_risk_flag")

    case = RiskCase(
        transaction_id=str(row["transaction_id"]),
        payer_id=str(row["payer_id"]),
        payee_id=str(row["payee_id"]),
        amount=float(row["amount"]),
        timestamp=str(row["timestamp"]),
        ml_score=round(ml_score, 4),
        graph_score=round(graph_score, 4),
        rules_score=round(rules_score, 4),
        risk_score=risk_score,
        risk_level=risk_level,
        evidence_list=evidence,
        triggered_signals=triggered,
        recommended_action=recommended_action_from_level(risk_level),
    )
    # Store original scores before Critic adjustment
    case.original_risk_score = risk_score
    case.original_risk_level = risk_level
    return case


def build_all_cases(df: pd.DataFrame) -> list[RiskCase]:
    """Build RiskCase for every transaction in the DataFrame."""
    cases = []
    for _, row in df.iterrows():
        cases.append(build_case(row))
    return cases


# =====================================================================
# Summary
# =====================================================================

def case_summary(cases: list[RiskCase]) -> pd.DataFrame:
    """Create a summary DataFrame from all cases."""
    records = []
    for c in cases:
        records.append({
            "transaction_id": c.transaction_id,
            "payer_id": c.payer_id,
            "payee_id": c.payee_id,
            "amount": c.amount,
            "ml_score": c.ml_score,
            "graph_score": c.graph_score,
            "rules_score": c.rules_score,
            "risk_score": c.risk_score,
            "risk_level": c.risk_level,
            "n_evidence": len(c.evidence_list),
            "n_signals": len(c.triggered_signals),
        })
    return pd.DataFrame(records)


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    from src.config import DATA_DIR

    graph_path = DATA_DIR / "processed" / "features_with_graph.csv"

    if not graph_path.exists():
        print("ERROR: Run graph_engine.py first.")
        sys.exit(1)

    print("Loading data ...")
    df = pd.read_csv(graph_path, parse_dates=["timestamp"])

    print("Building risk cases ...")
    cases = build_all_cases(df)

    summary = case_summary(cases)

    print("\n--- Risk Level Distribution ---")
    print(summary["risk_level"].value_counts().to_string())

    print("\n--- Risk Score Stats ---")
    print(summary["risk_score"].describe().to_string())

    # Show top 10 highest risk
    print("\n--- Top 10 Highest Risk ---")
    top10 = summary.nlargest(10, "risk_score")
    print(top10[["transaction_id", "payer_id", "amount", "risk_score", "risk_level"]].to_string())

    # Save summary
    out_path = DATA_DIR / "processed" / "cases_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")

    # Save full cases as JSON
    cases_path = DATA_DIR / "processed" / "cases.json"
    cases_dicts = [asdict(c) for c in cases]
    with open(cases_path, "w") as f:
        json.dump(cases_dicts, f, indent=2, default=str)
    print(f"Saved full cases to {cases_path}")
