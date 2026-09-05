"""Graph Intelligence — NetworkX heterogeneous relationship graph.

Builds a graph of accounts, devices, IPs, and transactions, then
computes graph-risk signals and detects fraud rings / circular transfers.

Node types:  account, device, ip, merchant
Edge types:  transfer, uses_device, uses_ip, transacts_at

Outputs per transaction/account:
  - graph_degree
  - shared_device_count
  - shared_ip_count
  - connected_suspicious_neighbours
  - in_short_cycle
  - graph_risk_score
  - evidence_paths (JSON)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# Graph Construction
# =====================================================================

def build_transaction_graph(df: pd.DataFrame) -> nx.Graph:
    """Build a heterogeneous graph from the transaction dataset.

    Nodes:
      - account:payer_id, account:payee_id  (type='account')
      - device:device_id                     (type='device')
      - ip:ip_address                        (type='ip')

    Edges:
      - (account, account) transfer   — payer → payee
      - (account, device)  uses_device
      - (account, ip)      uses_ip
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        payer = f"account:{row['payer_id']}"
        payee = f"account:{row['payee_id']}"
        device = f"device:{row['device_id']}"
        ip = f"ip:{row['ip_address']}"

        # Add nodes with type attribute
        for node, ntype in [
            (payer, "account"), (payee, "account"),
            (device, "device"), (ip, "ip"),
        ]:
            if node not in G:
                G.add_node(node, node_type=ntype)

        # Transfer edge
        if G.has_edge(payer, payee):
            G[payer][payee]["weight"] += 1
            G[payer][payee]["total_amount"] += row["amount"]
        else:
            G.add_edge(payer, payee, edge_type="transfer",
                       weight=1, total_amount=row["amount"])

        # Device edge
        if not G.has_edge(payer, device):
            G.add_edge(payer, device, edge_type="uses_device")
        # IP edge
        if not G.has_edge(payer, ip):
            G.add_edge(payer, ip, edge_type="uses_ip")

    return G


# =====================================================================
# Graph Signal Computation
# =====================================================================

def compute_graph_signals(
    G: nx.Graph,
    df: pd.DataFrame,
    suspicious_accounts: set[str] | None = None,
) -> pd.DataFrame:
    """Compute per-transaction graph-risk signals.

    For each transaction, look at the payer's node in the graph and
    compute degree, shared-entity counts, and short-cycle membership.
    """
    if suspicious_accounts is None:
        suspicious_accounts = set()

    # Precompute short cycles (length 3-6)
    cycles_3_6 = set()
    try:
        for cycle in nx.cycle_basis(G):
            if 3 <= len(cycle) <= 6:
                for node in cycle:
                    cycles_3_6.add(node)
    except Exception:
        pass

    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        payer_node = f"account:{row['payer_id']}"
        payee_node = f"account:{row['payee_id']}"

        if payer_node not in G:
            records.append(_empty_signal(row))
            continue

        neighbors = list(G.neighbors(payer_node))

        degree = len(neighbors)
        shared_device_count = sum(
            1 for n in neighbors if G.nodes[n].get("node_type") == "device"
        )
        shared_ip_count = sum(
            1 for n in neighbors if G.nodes[n].get("node_type") == "ip"
        )

        # Connected suspicious neighbours
        suspicious_neighbour_count = 0
        for n in neighbors:
            if n.startswith("account:") and n in suspicious_accounts:
                suspicious_neighbour_count += 1
            # Also check 2-hop for suspicious accounts
            for n2 in G.neighbors(n):
                if n2.startswith("account:") and n2 != payer_node:
                    if n2 in suspicious_accounts:
                        suspicious_neighbour_count += 1

        in_cycle = int(payer_node in cycles_3_6 or payee_node in cycles_3_6)

        # Graph risk score: simple weighted combination
        graph_risk = min(1.0, (
            0.2 * min(degree / 20, 1.0)
            + 0.25 * min(shared_device_count / 5, 1.0)
            + 0.25 * min(shared_ip_count / 5, 1.0)
            + 0.15 * min(suspicious_neighbour_count / 5, 1.0)
            + 0.15 * in_cycle
        ))

        # Evidence paths: short paths to other accounts through shared entities
        evidence_paths = _find_evidence_paths(G, payer_node, max_hops=3)

        records.append({
            "transaction_id": row["transaction_id"],
            "graph_degree": degree,
            "shared_device_count": shared_device_count,
            "shared_ip_count": shared_ip_count,
            "connected_suspicious_neighbours": suspicious_neighbour_count,
            "in_short_cycle": in_cycle,
            "graph_risk_score": round(graph_risk, 4),
            "evidence_paths": json.dumps(evidence_paths[:5]),
        })

    return pd.DataFrame(records)


def _empty_signal(row: pd.Series) -> dict:
    return {
        "transaction_id": row["transaction_id"],
        "graph_degree": 0,
        "shared_device_count": 0,
        "shared_ip_count": 0,
        "connected_suspicious_neighbours": 0,
        "in_short_cycle": 0,
        "graph_risk_score": 0.0,
        "evidence_paths": "[]",
    }


def _find_evidence_paths(
    G: nx.Graph,
    source: str,
    max_hops: int = 3,
    max_paths: int = 5,
) -> list[list[str]]:
    """Find short paths from source to other account nodes through shared entities."""
    paths_found: list[list[str]] = []
    visited: set[str] = set()

    def _bfs(node: str, depth: int, current_path: list[str]):
        if len(paths_found) >= max_paths:
            return
        if depth > max_hops:
            return
        for neighbor in G.neighbors(node):
            if neighbor == source or neighbor in visited:
                continue
            new_path = current_path + [neighbor]
            if (
                neighbor.startswith("account:")
                and neighbor != source
                and len(new_path) >= 3
            ):
                paths_found.append(new_path)
                if len(paths_found) >= max_paths:
                    return
            visited.add(neighbor)
            _bfs(neighbor, depth + 1, new_path)
            visited.discard(neighbor)

    _bfs(source, 0, [source])
    return paths_found


# =====================================================================
# Fraud Ring Detection
# =====================================================================

def detect_fraud_rings(
    G: nx.Graph,
    min_ring_size: int = 3,
    max_ring_size: int = 8,
) -> list[dict[str, Any]]:
    """Detect dense clusters / fraud rings in the graph.

    Uses clique detection on account subgraph filtered by degree.
    """
    # Extract account-only subgraph
    account_nodes = [
        n for n in G.nodes if G.nodes[n].get("node_type") == "account"
    ]
    account_subgraph = G.subgraph(account_nodes).copy()

    rings: list[dict[str, Any]] = []

    # Find triangles and small cliques
    try:
        cliques = list(nx.enumerate_all_cliques(account_subgraph))
        for clique in cliques:
            if min_ring_size <= len(clique) <= max_ring_size:
                # Get edges in the clique
                edges_in_ring = []
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        if account_subgraph.has_edge(clique[i], clique[j]):
                            edge_data = account_subgraph[clique[i]][clique[j]]
                            edges_in_ring.append({
                                "from": clique[i],
                                "to": clique[j],
                                "transfer_count": edge_data.get("weight", 0),
                                "total_amount": edge_data.get("total_amount", 0),
                            })

                rings.append({
                    "ring_id": f"RING_{'_'.join(sorted(c.replace('account:', '') for c in clique))}",
                    "accounts": [c.replace("account:", "") for c in clique],
                    "size": len(clique),
                    "edges": edges_in_ring,
                })
    except Exception:
        pass

    # Also detect circular transfers (directed cycles in undirected graph)
    try:
        for cycle in nx.cycle_basis(account_subgraph):
            if 3 <= len(cycle) <= max_ring_size:
                ring_accounts = [c.replace("account:", "") for c in cycle]
                # Check if this ring is already captured by clique detection
                ring_ids = {r["ring_id"] for r in rings}
                test_id = f"RING_{'_'.join(sorted(ring_accounts))}"
                if test_id not in ring_ids:
                    rings.append({
                        "ring_id": test_id,
                        "accounts": ring_accounts,
                        "size": len(cycle),
                        "edges": [],
                        "type": "circular_transfer",
                    })
    except Exception:
        pass

    return rings


# =====================================================================
# Account-Level Risk
# =====================================================================

def account_graph_risk(G: nx.Graph) -> pd.DataFrame:
    """Compute graph-based risk scores per account."""
    records = []
    for node in G.nodes:
        if not node.startswith("account:"):
            continue
        account = node.replace("account:", "")
        neighbors = list(G.neighbors(node))
        degree = len(neighbors)
        device_neighbors = sum(1 for n in neighbors if G.nodes[n].get("node_type") == "device")
        ip_neighbors = sum(1 for n in neighbors if G.nodes[n].get("node_type") == "ip")

        risk = min(1.0, (
            0.3 * min(degree / 20, 1.0)
            + 0.35 * min(device_neighbors / 5, 1.0)
            + 0.35 * min(ip_neighbors / 5, 1.0)
        ))
        records.append({
            "account": account,
            "graph_degree": degree,
            "shared_devices": device_neighbors,
            "shared_ips": ip_neighbors,
            "account_graph_risk": round(risk, 4),
        })
    return pd.DataFrame(records)


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    from src.config import DATA_DIR

    ml_path = DATA_DIR / "processed" / "features_with_ml.csv"
    out_path = DATA_DIR / "processed" / "features_with_graph.csv"

    if not ml_path.exists():
        print("ERROR: Run anomaly_detector.py first.")
        sys.exit(1)

    print("Loading data ...")
    df = pd.read_csv(ml_path, parse_dates=["timestamp"])

    print("Building transaction graph ...")
    G = build_transaction_graph(df)
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # First pass: identify suspicious accounts from ML flags
    suspicious = set(
        df.loc[df["anomaly_flag"] == 1, "payer_id"].unique()
    )
    print(f"  Suspicious accounts (ML): {len(suspicious)}")

    print("Computing graph signals ...")
    graph_signals = compute_graph_signals(G, df, suspicious_accounts=suspicious)

    print("Detecting fraud rings ...")
    rings = detect_fraud_rings(G)
    print(f"  Found {len(rings)} rings/cliques")
    for r in rings[:5]:
        print(f"    {r['ring_id']} (size {r['size']})")

    # Merge
    df_out = df.merge(graph_signals, on="transaction_id", how="left")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Columns: {len(df_out.columns)}")
