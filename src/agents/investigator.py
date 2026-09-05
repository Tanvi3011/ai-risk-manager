"""Investigator Agent — Expands Evidence on Request.

When the Critic requests more evidence, the Investigator:
  1. Queries the graph for additional connections
  2. Finds related transactions (shared device/IP/amount patterns)
  3. Identifies circular transfers and money flow paths
  4. Returns structured evidence additions

This enables the Critic feedback loop:
  Detector -> Critic -> Investigator -> Critic (reassesses)
"""

from __future__ import annotations

import json

import networkx as nx
import pandas as pd

from src.agents.state import InvestigationState


def investigate_transaction(
    state: InvestigationState,
    df: pd.DataFrame,
    G: nx.Graph,
    requests: list[str],
) -> InvestigationState:
    """Expand evidence based on Critic requests."""
    new_evidence = []

    for request in requests:
        req = request.lower()

        if any(w in req for w in ("graph", "connection", "neighbor")):
            new_evidence.extend(_investigate_graph_connections(state, G))

        if any(w in req for w in ("circular", "cycle", "ring")):
            new_evidence.extend(_investigate_circular_flows(state, G))

        if any(w in req for w in ("shared device", "device")):
            new_evidence.extend(_investigate_shared_device(state, df))

        if any(w in req for w in ("shared ip", "ip")):
            new_evidence.extend(_investigate_shared_ip(state, df))

        if any(w in req for w in ("velocity", "speed", "burst")):
            new_evidence.extend(_investigate_velocity(state, df))

        if any(w in req for w in ("amount", "value")):
            new_evidence.extend(_investigate_amount_patterns(state, df))

        if any(w in req for w in ("account", "related")):
            new_evidence.extend(_investigate_related_accounts(state, df, G))

    # Deduplicate
    seen = {e.get("signal") for e in state.evidence_list}
    deduped = [e for e in new_evidence if e.get("signal") not in seen]

    for ev in deduped:
        state.add_evidence(ev)

    state.investigator_output = {
        "requests_processed": len(requests),
        "new_evidence_count": len(deduped),
        "requests": requests,
    }

    state.add_timeline_event(
        step="investigator_expanded",
        agent="Investigator",
        summary=f"Investigated {len(requests)} request(s), added {len(deduped)} new evidence items",
        details={"requests": requests},
        evidence_added=deduped,
    )

    return state


def _investigate_graph_connections(state: InvestigationState, G: nx.Graph) -> list[dict]:
    """Find direct and 2-hop connections for the payer."""
    evidence = []
    payer_node = f"account:{state.payer_id}"

    if payer_node not in G:
        return evidence

    account_neighbors = [
        n.replace("account:", "")
        for n in G.neighbors(payer_node)
        if n.startswith("account:")
    ]
    if account_neighbors:
        evidence.append({
            "type": "graph",
            "signal": "direct_account_connections",
            "value": account_neighbors[:10],
            "description": (
                f"Payer has {len(account_neighbors)} direct account connections: "
                f"{', '.join(account_neighbors[:5])}"
                f"{'...' if len(account_neighbors) > 5 else ''}"
            ),
        })

    two_hop = set()
    for neighbor in G.neighbors(payer_node):
        for n2 in G.neighbors(neighbor):
            if n2.startswith("account:") and n2 != payer_node:
                two_hop.add(n2.replace("account:", ""))

    if two_hop:
        evidence.append({
            "type": "graph",
            "signal": "two_hop_connections",
            "value": list(two_hop)[:15],
            "description": (
                f"2-hop analysis reveals {len(two_hop)} indirectly connected accounts "
                f"through shared devices/IPs"
            ),
        })

    payee_node = f"account:{state.payee_id}"
    if payee_node in G:
        payee_neighbors = [
            n.replace("account:", "")
            for n in G.neighbors(payee_node)
            if n.startswith("account:")
        ]
        if payee_neighbors:
            evidence.append({
                "type": "graph",
                "signal": "payee_connections",
                "value": payee_neighbors[:10],
                "description": (
                    f"Payee has {len(payee_neighbors)} direct account connections: "
                    f"{', '.join(payee_neighbors[:5])}"
                ),
            })

    return evidence


def _investigate_circular_flows(state: InvestigationState, G: nx.Graph) -> list[dict]:
    """Detect circular money flows."""
    evidence = []
    payer_node = f"account:{state.payer_id}"
    payee_node = f"account:{state.payee_id}"

    if payer_node not in G:
        return evidence

    account_nodes = [n for n in G.nodes if n.startswith("account:")]
    subgraph = G.subgraph(account_nodes).copy()

    try:
        cycles = list(nx.cycle_basis(subgraph))
        for cycle in cycles[:5]:
            if payer_node in cycle or payee_node in cycle:
                accts = [c.replace("account:", "") for c in cycle]
                evidence.append({
                    "type": "graph",
                    "signal": "circular_transfer_detected",
                    "value": accts,
                    "description": f"Circular flow: {' -> '.join(accts)} -> {accts[0]}",
                })
    except Exception:
        pass

    if payee_node in G and G.has_edge(payee_node, payer_node):
        edge = G[payee_node][payer_node]
        evidence.append({
            "type": "graph",
            "signal": "bidirectional_transfer",
            "value": {
                "from": state.payee_id,
                "to": state.payer_id,
                "transfer_count": edge.get("weight", 0),
                "total_amount": edge.get("total_amount", 0),
            },
            "description": (
                f"Bidirectional: {state.payee_id} -> {state.payer_id} "
                f"({edge.get('weight', 0)} transfers, ${edge.get('total_amount', 0):,.2f})"
            ),
        })

    return evidence


def _investigate_shared_device(state: InvestigationState, df: pd.DataFrame) -> list[dict]:
    """Find all accounts sharing the same device."""
    evidence = []
    device_txns = df[df["device_id"] == state.device_id]
    others = [a for a in device_txns["payer_id"].unique() if a != state.payer_id]

    if others:
        other_fraud = set(
            df[(df["payer_id"].isin(others)) & (df["is_fraud"] == 1)]["payer_id"].unique()
        )
        evidence.append({
            "type": "graph",
            "signal": "shared_device_accounts",
            "value": {
                "accounts": others[:10],
                "total_accounts": len(others),
                "fraud_accounts": list(other_fraud)[:5],
            },
            "description": (
                f"Device {state.device_id} shared across {len(others)} accounts"
                f"{f', {len(other_fraud)} flagged as fraud' if other_fraud else ''}"
            ),
        })

    return evidence


def _investigate_shared_ip(state: InvestigationState, df: pd.DataFrame) -> list[dict]:
    """Find all accounts sharing the same IP."""
    evidence = []
    ip_txns = df[df["ip_address"] == state.ip_address]
    others = [a for a in ip_txns["payer_id"].unique() if a != state.payer_id]

    if others:
        other_fraud = set(
            df[(df["payer_id"].isin(others)) & (df["is_fraud"] == 1)]["payer_id"].unique()
        )
        evidence.append({
            "type": "graph",
            "signal": "shared_ip_accounts",
            "value": {
                "accounts": others[:10],
                "total_accounts": len(others),
                "fraud_accounts": list(other_fraud)[:5],
            },
            "description": (
                f"IP {state.ip_address} shared across {len(others)} accounts"
                f"{f', {len(other_fraud)} flagged as fraud' if other_fraud else ''}"
            ),
        })

    return evidence


def _investigate_velocity(state: InvestigationState, df: pd.DataFrame) -> list[dict]:
    """Investigate transaction velocity patterns."""
    evidence = []
    payer_txns = df[df["payer_id"] == state.payer_id].sort_values("timestamp")

    if len(payer_txns) < 2:
        return evidence

    total_count = len(payer_txns)
    total_amount = payer_txns["amount"].sum()

    # Check for bursts in time windows
    ts = payer_txns["timestamp"]
    for window_name, delta in [("5min", "5min"), ("1h", "1h"), ("24h", "24h")]:
        if delta == "5min":
            td = pd.Timedelta(minutes=5)
        elif delta == "1h":
            td = pd.Timedelta(hours=1)
        else:
            td = pd.Timedelta(hours=24)

        max_in_window = 0
        for i in range(len(payer_txns)):
            window_start = ts.iloc[i] - td
            count = ((ts >= window_start) & (ts < ts.iloc[i])).sum()
            max_in_window = max(max_in_window, count)

        if max_in_window > 0:
            evidence.append({
                "type": "graph",
                "signal": f"velocity_{window_name}",
                "value": max_in_window,
                "description": f"Peak velocity: {max_in_window} transactions within {window_name}",
            })

    evidence.append({
        "type": "graph",
        "signal": "total_account_activity",
        "value": {"total_txns": total_count, "total_amount": round(total_amount, 2)},
        "description": f"Account total: {total_count} transactions, ${total_amount:,.2f} volume",
    })

    return evidence


def _investigate_amount_patterns(state: InvestigationState, df: pd.DataFrame) -> list[dict]:
    """Investigate amount anomalies."""
    evidence = []
    payer_txns = df[df["payer_id"] == state.payer_id]

    if len(payer_txns) < 2:
        return evidence

    mean_amt = payer_txns["amount"].mean()
    std_amt = payer_txns["amount"].std()
    median_amt = payer_txns["amount"].median()

    zscore = 0.0
    if std_amt > 0:
        zscore = (state.amount - mean_amt) / std_amt

    evidence.append({
        "type": "graph",
        "signal": "amount_profile",
        "value": {
            "mean": round(mean_amt, 2),
            "std": round(std_amt, 2),
            "median": round(median_amt, 2),
            "zscore": round(zscore, 3),
        },
        "description": (
            f"Amount ${state.amount:,.2f} vs payer mean ${mean_amt:,.2f} "
            f"(z-score: {zscore:.2f})"
        ),
    })

    # Check for round amounts
    if state.amount >= 1000 and state.amount % 1000 == 0:
        evidence.append({
            "type": "context",
            "signal": "round_amount",
            "value": state.amount,
            "description": f"Round amount ${state.amount:,.0f} — common in structuring",
        })

    return evidence


def _investigate_related_accounts(
    state: InvestigationState, df: pd.DataFrame, G: nx.Graph
) -> list[dict]:
    """Find accounts with similar behavioral patterns."""
    evidence = []
    payer_txns = df[df["payer_id"] == state.payer_id]

    if payer_txns.empty:
        return evidence

    # Accounts that receive from this payer
    receivers = payer_txns["payee_id"].unique().tolist()
    if len(receivers) > 1:
        # Check if any receivers are also senders (potential mule chain)
        receiver_txns = df[df["payer_id"].isin(receivers)]
        receiver_receivers = set(receiver_txns["payee_id"].unique()) - set(receivers) - {state.payer_id}
        if receiver_receivers:
            evidence.append({
                "type": "graph",
                "signal": "mule_chain_pattern",
                "value": {
                    "intermediaries": receivers[:5],
                    "downstream_accounts": list(receiver_receivers)[:5],
                },
                "description": (
                    f"Funds flow through {len(receivers)} intermediaries "
                    f"to {len(receiver_receivers)} downstream accounts"
                ),
            })

    return evidence
