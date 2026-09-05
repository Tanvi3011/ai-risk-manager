"""Multi-Agent Orchestrator — Pipeline Controller.

Runs the full investigation pipeline:
  Payment -> Feature Extract -> Detector -> Critic -> [Investigator -> Critic] -> Guardrails -> Explainer

The orchestrator manages the feedback loop where the Critic can request
additional evidence from the Investigator, creating a visible
investigation trace.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx
import pandas as pd

from src.agents.state import (
    InvestigationState,
    RiskAction,
    RiskLevel,
    ScoreBreakdown,
)
from src.agents.detector import build_case, compute_risk_score, risk_level_from_score
from src.agents.critic import run_critic
from src.agents.investigator import investigate_transaction
from src.agents.guardrails import validate_guardrails
from src.agents.explainer import run_explainer, answer_question


def run_investigation(
    row: pd.Series,
    df: pd.DataFrame,
    G: nx.Graph,
    max_rounds: int = 2,
) -> InvestigationState:
    """Run the full multi-agent investigation pipeline.

    Args:
        row: Transaction row from the features dataframe
        df: Full features dataframe (for lookups)
        G: Transaction graph
        max_rounds: Max Critic->Investigator feedback loops

    Returns:
        Complete InvestigationState with timeline and evidence
    """
    start_time = time.time()

    # Initialize state from transaction data
    state = _initialize_state(row)

    # Step 1: Detector scores the transaction
    state = _run_detector(state, row)

    # Step 2-5: Critic feedback loop
    state = _run_critic_loop(state, df, G, max_rounds)

    # Step 6: Guardrails
    state = _run_guardrails(state)

    # Step 7: Explainer
    state = _run_final_explainer(state)

    # Record timing
    state.processing_time_ms = round((time.time() - start_time) * 1000, 1)

    return state


def run_realtime_analysis(
    amount: float,
    payer_id: str,
    payee_id: str,
    device_id: str,
    ip_address: str,
    df: pd.DataFrame,
    G: nx.Graph,
) -> InvestigationState:
    """Analyze a new payment in real-time.

    Creates a synthetic feature row from user input, runs it through
    the full pipeline, and returns the investigation state.
    """
    # Build a feature row from the input
    row = _build_realtime_features(
        amount, payer_id, payee_id, device_id, ip_address, df
    )

    return run_investigation(row, df, G)


def _initialize_state(row: pd.Series) -> InvestigationState:
    """Create initial state from transaction row."""
    state = InvestigationState(
        transaction_id=str(row.get("transaction_id", "")),
        payer_id=str(row.get("payer_id", "")),
        payee_id=str(row.get("payee_id", "")),
        amount=float(row.get("amount", 0)),
        timestamp=str(row.get("timestamp", "")),
        device_id=str(row.get("device_id", "")),
        ip_address=str(row.get("ip_address", "")),
    )

    state.add_timeline_event(
        step="payment_received",
        agent="System",
        summary=f"Payment received: {state.payer_id} -> {state.payee_id}, ${state.amount:,.2f}",
        details={
            "transaction_id": state.transaction_id,
            "device": state.device_id,
            "ip": state.ip_address,
        },
    )

    return state


def _run_detector(state: InvestigationState, row: pd.Series) -> InvestigationState:
    """Run the Detector Agent — score the transaction."""
    ml_score = float(row.get("anomaly_score", 0))
    graph_score = float(row.get("graph_risk_score", 0))

    rule_cols = [c for c in row.index if c.startswith("rule_")]
    n_triggered = sum(1 for c in rule_cols if row.get(c, 0) == 1)
    rules_score = min(1.0, n_triggered / 4)

    # Build score breakdown
    weight_ml = 0.35
    weight_rules = 0.35
    weight_graph = 0.30

    ml_contribution = round(ml_score * weight_ml * 100, 1)
    rules_contribution = round(rules_score * weight_rules * 100, 1)
    graph_contribution = round(graph_score * weight_graph * 100, 1)
    context_contribution = _compute_context_contribution(row)
    base_score = ml_contribution + rules_contribution + graph_contribution + context_contribution

    state.score_breakdown = ScoreBreakdown(
        ml_raw=ml_score,
        ml_normalized=ml_score,
        ml_contribution=ml_contribution,
        rules_raw=rules_score,
        rules_contribution=rules_contribution,
        graph_raw=graph_score,
        graph_contribution=graph_contribution,
        context_contribution=context_contribution,
        base_score=base_score,
        weight_ml=weight_ml,
        weight_rules=weight_rules,
        weight_graph=weight_graph,
    )

    state.risk_score = round(min(100, base_score), 1)
    state.risk_level = RiskLevel(risk_level_from_score(state.risk_score))
    state.risk_action = state.risk_action_from_level()

    # Collect triggered signals
    triggered = []
    for c in rule_cols:
        if row.get(c, 0) == 1:
            triggered.append(c)
    if ml_score > 0.7:
        triggered.append("ml_anomaly_flag")
    if graph_score > 0.3:
        triggered.append("graph_risk_flag")
    state.triggered_signals = triggered

    # Build evidence from the row
    evidence = _build_evidence_from_row(row)
    for ev in evidence:
        state.add_evidence(ev)

    # Confidence
    strong_count = sum(1 for s in triggered if s in {
        "rule_high_amount", "rule_odd_hour_high_value", "rule_circular_transfer_hint",
        "rule_high_velocity_5min", "rule_many_new_payees_24h",
        "ml_anomaly_flag", "graph_risk_flag",
    })
    if strong_count >= 3:
        state.confidence = "HIGH"
    elif strong_count >= 1:
        state.confidence = "MEDIUM"
    else:
        state.confidence = "LOW"

    state.detector_output = {
        "ml_score": ml_score,
        "graph_score": graph_score,
        "rules_score": rules_score,
        "base_score": base_score,
        "risk_level": state.risk_level.value,
    }

    state.add_timeline_event(
        step="detector_scored",
        agent="Detector",
        summary=f"Score: {state.risk_score:.1f}/100 ({state.risk_level.value}). "
                f"ML={ml_score:.3f}, Rules={rules_score:.3f}, Graph={graph_score:.3f}",
        details=state.detector_output,
    )

    return state


def _run_critic_loop(
    state: InvestigationState,
    df: pd.DataFrame,
    G: nx.Graph,
    max_rounds: int,
) -> InvestigationState:
    """Run Critic with optional Investigator feedback loop."""
    for round_num in range(max_rounds):
        state.investigation_rounds = round_num + 1

        # Run Critic
        state = run_critic(state)

        verdict = state.critic_output.get("verdict", "confirm")

        if verdict == "request_evidence" and state.evidence_requested:
            # Investigator expands evidence
            state = investigate_transaction(
                state, df, G, state.evidence_requested
            )
            state.evidence_requested = []
            # Continue loop for Critic to reassess
        else:
            # Critic confirmed or downgraded, no need for more investigation
            break

    return state


def _run_guardrails(state: InvestigationState) -> InvestigationState:
    """Run deterministic guardrails."""
    from src.agents.guardrails import validate_guardrails as _validate

    result = _validate(state)

    state.guardrail_output = result.to_dict()

    # Apply guardrail overrides if needed
    if not result.passed:
        try:
            state.risk_action = RiskAction(result.guardrail_action.value)
            state.risk_score = result.adjusted_score
            state.risk_level = RiskLevel(risk_level_from_score(state.risk_score))
        except (ValueError, KeyError):
            pass

    state.add_timeline_event(
        step="guardrail_validated",
        agent="Guardrails",
        summary=result.summary,
        details=state.guardrail_output,
    )

    return state


def _run_final_explainer(state: InvestigationState) -> InvestigationState:
    """Run the Explainer agent."""
    state = run_explainer(state)
    return state


def _compute_context_contribution(row: pd.Series) -> float:
    """Compute context-based score contribution (max 10 points)."""
    score = 0.0

    if row.get("is_odd_hour", 0) == 1:
        score += 3.0
    if row.get("is_weekend", 0) == 1:
        score += 1.0
    if row.get("is_new_payee", 0) == 1:
        score += 2.0
    if row.get("is_new_device", 0) == 1:
        score += 1.5
    if row.get("is_new_ip", 0) == 1:
        score += 1.0

    amount = row.get("amount", 0)
    if amount > 50000:
        score += 2.5
    elif amount > 10000:
        score += 1.5
    elif amount > 5000:
        score += 0.5

    return min(10.0, score)


def _build_evidence_from_row(row: pd.Series) -> list[dict]:
    """Extract structured evidence items from a feature row."""
    evidence = []

    ml_score = float(row.get("anomaly_score", 0))
    graph_score = float(row.get("graph_risk_score", 0))

    # ML evidence
    if ml_score > 0.7:
        evidence.append({
            "type": "ml",
            "signal": "high_anomaly_score",
            "value": round(ml_score, 3),
            "description": f"Anomaly score {ml_score:.3f} (top 30% of all transactions).",
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
        "rule_high_amount": "Unusually high amount (>3σ above payer mean).",
        "rule_high_velocity_5min": "≥3 transactions in last 5 minutes.",
        "rule_high_velocity_1h": "≥6 transactions in last hour.",
        "rule_many_new_payees_24h": "≥3 new payees in last 24 hours.",
        "rule_shared_device_multiuser": "Device shared across multiple accounts.",
        "rule_shared_ip_multiuser": "IP shared across multiple accounts.",
        "rule_odd_hour_high_value": "High-value at odd hours (0-5 AM).",
        "rule_circular_transfer_hint": "Circular transfer pattern detected.",
    }
    for col, desc in rule_map.items():
        if row.get(col, 0) == 1:
            evidence.append({
                "type": "rule",
                "signal": col,
                "value": True,
                "description": desc,
            })

    # Graph evidence
    if graph_score > 0.3:
        evidence.append({
            "type": "graph",
            "signal": "high_graph_risk",
            "value": round(graph_score, 3),
            "description": f"Graph risk score {graph_score:.3f}.",
        })
    if row.get("shared_device_count", 0) > 0:
        evidence.append({
            "type": "graph",
            "signal": "shared_device_in_graph",
            "value": int(row["shared_device_count"]),
            "description": f"Shares device with {int(row['shared_device_count'])} other account(s).",
        })
    if row.get("shared_ip_count", 0) > 0:
        evidence.append({
            "type": "graph",
            "signal": "shared_ip_in_graph",
            "value": int(row["shared_ip_count"]),
            "description": f"Shares IP with {int(row['shared_ip_count'])} other account(s).",
        })
    if row.get("in_short_cycle", 0) == 1:
        evidence.append({
            "type": "graph",
            "signal": "in_cycle",
            "value": True,
            "description": "Part of a short cycle (potential circular transfer).",
        })

    # Context
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


def _build_realtime_features(
    amount: float,
    payer_id: str,
    payee_id: str,
    device_id: str,
    ip_address: str,
    df: pd.DataFrame,
) -> pd.Series:
    """Build a feature row for real-time analysis."""
    import numpy as np

    # Use historical stats to compute features
    payer_txns = df[df["payer_id"] == payer_id]

    if len(payer_txns) > 0:
        payer_mean = payer_txns["amount"].mean()
        payer_std = payer_txns["amount"].std() or 1.0
        payer_median = payer_txns["amount"].median() or 1.0
        log_amount = np.log1p(amount)
        amount_zscore = (amount - payer_mean) / payer_std
        amount_ratio = amount / payer_median
        amount_deviation = amount - payer_mean
    else:
        log_amount = np.log1p(amount)
        amount_zscore = 0.0
        amount_ratio = 1.0
        amount_deviation = 0.0
        payer_mean = amount
        payer_std = 1.0

    # Device/IP stats
    device_accounts = df[df["device_id"] == device_id]["payer_id"].nunique()
    ip_accounts = df[df["ip_address"] == ip_address]["payer_id"].nunique()

    # Time features
    from datetime import datetime
    try:
        ts = pd.Timestamp.now()
    except Exception:
        ts = datetime.now()
    hour = ts.hour
    is_odd_hour = 1 if hour in range(0, 6) else 0
    is_weekend = 1 if ts.dayofweek in (5, 6) else 0

    # Velocity (approximate from recent data)
    recent = payer_txns.sort_values("timestamp", ascending=False)
    now = pd.Timestamp.now()
    vel_5min = len(recent[recent["timestamp"] > now - pd.Timedelta(minutes=5)])
    vel_1h = len(recent[recent["timestamp"] > now - pd.Timedelta(hours=1)])
    vel_24h = len(recent[recent["timestamp"] > now - pd.Timedelta(hours=24)])

    # New payee check
    known_payees = set(payer_txns["payee_id"].unique())
    is_new_payee = 0 if payee_id in known_payees else 1

    # New device check
    known_devices = set(payer_txns["device_id"].unique())
    is_new_device = 0 if device_id in known_devices else 1

    # New IP check
    known_ips = set(payer_txns["ip_address"].unique())
    is_new_ip = 0 if ip_address in known_ips else 1

    # Circular check
    has_reverse = int(
        len(df[(df["payer_id"] == payee_id) & (df["payee_id"] == payer_id)]) > 0
    )

    row = pd.Series({
        "transaction_id": f"RT_{int(time.time())}",
        "timestamp": pd.Timestamp.now(),
        "payer_id": payer_id,
        "payee_id": payee_id,
        "merchant_id": "unknown",
        "amount": amount,
        "device_id": device_id,
        "ip_address": ip_address,
        "is_fraud": 0,
        "fraud_pattern": "normal",

        # Amount features
        "log_amount": log_amount,
        "payer_mean_amount": payer_mean,
        "payer_std_amount": payer_std,
        "payer_median_amount": payer_median,
        "amount_deviation_from_payer_mean": amount_deviation,
        "amount_zscore": amount_zscore,
        "amount_ratio_to_payer_median": amount_ratio,

        # Velocity features
        "velocity_count_5min": vel_5min,
        "velocity_amount_5min": 0.0,
        "velocity_count_1h": vel_1h,
        "velocity_amount_1h": 0.0,
        "velocity_count_24h": vel_24h,
        "velocity_amount_24h": 0.0,

        # Payee features
        "is_new_payee": is_new_payee,
        "recent_new_payee_count_24h": 1 if is_new_payee else 0,
        "unique_payees_24h": len(known_payees),
        "recipient_concentration": 0.0,

        # Device features
        "device_account_count": device_accounts,
        "device_total_txn_count": len(payer_txns[payer_txns["device_id"] == device_id]),
        "is_new_device": is_new_device,

        # IP features
        "ip_account_count": ip_accounts,
        "ip_total_txn_count": len(payer_txns[payer_txns["ip_address"] == ip_address]),
        "is_new_ip": is_new_ip,

        # Time features
        "hour": hour,
        "day_of_week": ts.dayofweek,
        "is_odd_hour": is_odd_hour,
        "is_weekend": is_weekend,
        "payer_mean_hour": payer_txns["hour"].mean() if "hour" in payer_txns.columns and len(payer_txns) > 0 else hour,
        "payer_std_hour": payer_txns["hour"].std() if "hour" in payer_txns.columns and len(payer_txns) > 1 else 0.0,
        "hour_deviation": 0.0,

        # Rules (will be computed)
        "rule_high_amount": int(abs(amount_zscore) > 3.0),
        "rule_high_velocity_5min": int(vel_5min >= 3),
        "rule_high_velocity_1h": int(vel_1h >= 6),
        "rule_many_new_payees_24h": 0,
        "rule_shared_device_multiuser": int(device_accounts >= 3),
        "rule_shared_ip_multiuser": int(ip_accounts >= 3),
        "rule_odd_hour_high_value": int(is_odd_hour and amount > 5000),
        "rule_circular_transfer_hint": has_reverse,
        "rules_triggered_count": 0,

        # ML (will be approximated)
        "anomaly_score": min(1.0, abs(amount_zscore) / 5.0) if payer_std > 0 else 0.5,
        "anomaly_flag": 0,
        "ml_top_features": "[]",

        # Graph (will be approximated)
        "graph_risk_score": min(1.0, (device_accounts + ip_accounts) / 10),
        "shared_device_count": max(0, device_accounts - 1),
        "shared_ip_count": max(0, ip_accounts - 1),
        "connected_suspicious_neighbours": 0,
        "in_short_cycle": 0,
        "evidence_paths": "[]",
    })

    # Compute rules_triggered_count
    rule_cols = [c for c in row.index if c.startswith("rule_")]
    row["rules_triggered_count"] = sum(1 for c in rule_cols if row.get(c, 0) == 1)

    # Set anomaly_flag based on score
    row["anomaly_flag"] = int(row["anomaly_score"] > 0.7)

    return row
