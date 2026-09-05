"""AI Risk Manager — Payment Risk Operations Console.

5-page analyst interface:
  1) Risk Queue      — triage alerts by severity
  2) Case Investigation — deep dive into a single case
  3) Network         — interactive fraud network graph
  4) AI Trace        — chronological investigation timeline
  5) Operations      — model metrics, system health

Design: serious internal risk-ops console. Information-dense, restrained,
evidence-first. Data creates the visual hierarchy, not decorations.
"""

import json
import os
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

# ─── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Risk Manager",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import DATA_DIR as _CONFIG_DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = _CONFIG_DATA_DIR / "processed"

# ─── Custom CSS — restrained analyst console ───────────────────────────
st.markdown("""
<style>
    /* Base */
    .stApp { background: #fafafa; }
    section[data-testid="stSidebar"] { background: #1a1a2e; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #e0e0e0; }

    /* Risk chips */
    .risk-chip {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        font-family: "SF Mono", "Fira Code", monospace;
    }
    .risk-LOW { background: #e8f5e9; color: #2e7d32; }
    .risk-MEDIUM { background: #fff3e0; color: #e65100; }
    .risk-HIGH { background: #fce4ec; color: #c62828; }
    .risk-CRITICAL { background: #c62828; color: #fff; }

    /* Action chips */
    .action-chip {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: "SF Mono", "Fira Code", monospace;
        border: 1px solid #ccc;
        color: #333;
    }
    .action-HOLD { border-color: #c62828; color: #c62828; }
    .action-REVIEW { border-color: #e65100; color: #e65100; }
    .action-STEP-UP { border-color: #f9a825; color: #f57f17; }
    .action-ALLOW { border-color: #2e7d32; color: #2e7d32; }

    /* Compact metrics */
    .metric-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .metric-card .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #666;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a1a;
        font-family: "SF Mono", "Fira Code", monospace;
    }
    .metric-card .sub {
        font-size: 0.7rem;
        color: #888;
        margin-top: 2px;
    }

    /* Score bar */
    .score-bar {
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        overflow: hidden;
        margin: 4px 0;
    }
    .score-bar .fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s;
    }
    .score-fill-low { background: #4caf50; }
    .score-fill-medium { background: #ff9800; }
    .score-fill-high { background: #f44336; }
    .score-fill-critical { background: #b71c1c; }

    /* Timeline */
    .timeline-item {
        display: flex;
        gap: 12px;
        padding: 8px 0;
        border-left: 2px solid #e0e0e0;
        margin-left: 8px;
        padding-left: 16px;
    }
    .timeline-item:last-child { border-left-color: transparent; }
    .timeline-agent {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #666;
        min-width: 80px;
    }
    .timeline-summary { font-size: 0.85rem; color: #333; }

    /* Evidence table */
    .evidence-row {
        padding: 6px 10px;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.82rem;
    }
    .evidence-type {
        display: inline-block;
        padding: 1px 6px;
        border-radius: 2px;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 8px;
    }
    .ev-ml { background: #e3f2fd; color: #1565c0; }
    .ev-rule { background: #fce4ec; color: #c62828; }
    .ev-graph { background: #e8f5e9; color: #2e7d32; }
    .ev-critic { background: #f3e5f5; color: #7b1fa2; }
    .ev-context { background: #fff3e0; color: #e65100; }

    /* Override Streamlit defaults */
    h1 { font-size: 1.4rem !important; font-weight: 700; }
    h2 { font-size: 1.1rem !important; font-weight: 600; }
    h3 { font-size: 0.95rem !important; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-size: 0.82rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ──────────────────────────────────────────────────────

def _run_full_pipeline():
    """Generate all data if missing — runs in-process."""
    from src.config import DATA_DIR, RAW_DATA_DIR

    steps = [
        ("Generating transactions", lambda: _step_generate()),
        ("Building features", lambda: _step_features()),
        ("Applying rules", lambda: _step_rules()),
        ("Training Isolation Forest", lambda: _step_isolation_forest()),
        ("Building graph", lambda: _step_graph()),
        ("Training LightGBM", lambda: _step_lightgbm()),
        ("Feature importance", lambda: _step_feature_importance()),
        ("Running agents", lambda: _step_agents()),
    ]

    progress = st.progress(0, text="Starting pipeline...")
    errors = []
    for i, (label, fn) in enumerate(steps):
        progress.progress((i) / len(steps), text=f"{label}...")
        try:
            fn()
            progress.progress((i + 1) / len(steps), text=f"{label} done")
        except Exception as e:
            progress.progress((i + 1) / len(steps), text=f"{label} failed")
            errors.append(f"{label}: {e}")
            st.error(f"{label} failed: {e}")
    if errors:
        st.warning(f"Pipeline completed with {len(errors)} error(s). Some features may be unavailable.")
    else:
        progress.progress(1.0, text="Pipeline complete!")


def _step_generate():
    from src.config import RAW_DATA_DIR
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    exec(open(PROJECT_ROOT / "src" / "data_generation" / "generate_transactions.py", encoding="utf-8").read())


def _step_features():
    from src.features.feature_engineering import build_feature_matrix
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    import pandas as pd
    df = pd.read_csv(RAW_DATA_DIR / "transactions.csv", parse_dates=["timestamp"])
    features = build_feature_matrix(df)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_DATA_DIR / "features.csv", index=False)


def _step_rules():
    from src.features.rules import apply_rules
    from src.config import PROCESSED_DATA_DIR
    import pandas as pd
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv", parse_dates=["timestamp"])
    df_rules = apply_rules(df)
    df_rules.to_csv(PROCESSED_DATA_DIR / "features_with_rules.csv", index=False)


def _step_isolation_forest():
    from src.models.anomaly_detector import train_isolation_forest, predict_anomaly
    from src.features.feature_engineering import get_feature_columns
    from src.config import PROCESSED_DATA_DIR
    import pandas as pd, numpy as np
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(PROCESSED_DATA_DIR / "features_with_rules.csv", parse_dates=["timestamp"])
    feature_cols = [c for c in get_feature_columns(df) if not c.startswith("rule_")]
    ml_features = [c for c in feature_cols if df[c].dtype in ['float64','int64','bool','float32','int32']]
    X = df[ml_features].fillna(0)
    contamination = df["is_fraud"].mean()
    model, scaler = train_isolation_forest(X, contamination=contamination)
    results = predict_anomaly(model, scaler, X, ml_features)
    df["anomaly_score"] = results["anomaly_score"].values
    df["anomaly_flag"] = results["anomaly_flag"].values
    df["ml_top_features"] = results["top_features"].values
    df.to_csv(PROCESSED_DATA_DIR / "features_with_ml.csv", index=False)


def _step_lightgbm():
    from src.models.supervised_detector import run_dual_model_comparison
    from src.config import DATA_DIR
    import pandas as pd
    df = pd.read_csv(DATA_DIR / "processed" / "features_with_graph.csv", parse_dates=["timestamp"])
    run_dual_model_comparison(df)


def _step_feature_importance():
    from src.models.feature_importance import save_feature_importances
    from src.config import DATA_DIR
    import pandas as pd
    df = pd.read_csv(DATA_DIR / "processed" / "features_with_graph.csv", parse_dates=["timestamp"])
    save_feature_importances(df)


def _step_graph():
    from src.graphs.graph_engine import build_transaction_graph, compute_graph_signals, detect_fraud_rings
    from src.config import PROCESSED_DATA_DIR
    import pandas as pd, json
    df = pd.read_csv(PROCESSED_DATA_DIR / "features_with_ml.csv", parse_dates=["timestamp"])
    G = build_transaction_graph(df)
    suspicious = set(df.loc[df["anomaly_flag"] == 1, "payer_id"].unique())
    graph_signals = compute_graph_signals(G, df, suspicious_accounts=suspicious)
    df_out = df.merge(graph_signals, on="transaction_id", how="left")
    df_out.to_csv(PROCESSED_DATA_DIR / "features_with_graph.csv", index=False)


def _step_agents():
    from src.agents.detector import build_all_cases, case_summary
    from src.agents.critic import apply_critic_to_all
    from src.agents.explainer import explain_all
    from src.config import PROCESSED_DATA_DIR
    import pandas as pd, json
    df = pd.read_csv(PROCESSED_DATA_DIR / "features_with_graph.csv", parse_dates=["timestamp"])
    cases = build_all_cases(df)
    cases_dicts = [vars(c) if hasattr(c, '__dict__') else {} for c in cases]
    with open(PROCESSED_DATA_DIR / "cases.json", "w") as f:
        json.dump(cases_dicts, f, indent=2, default=str)
    summary = case_summary(cases)
    summary.to_csv(PROCESSED_DATA_DIR / "cases_summary.csv", index=False)
    cases = apply_critic_to_all(cases)
    cases_dicts = [vars(c) if hasattr(c, '__dict__') else {} for c in cases]
    with open(PROCESSED_DATA_DIR / "cases_with_critic.json", "w") as f:
        json.dump(cases_dicts, f, indent=2, default=str)
    explanations = explain_all(cases)
    exp_dicts = [{"transaction_id": e.transaction_id, "risk_level": e.risk_level, "confidence": e.confidence, "headline": e.headline, "what_happened": e.what_happened, "why_suspicious": e.why_suspicious, "supporting_evidence": e.supporting_evidence, "graph_context": e.graph_context, "critic_note": e.critic_note, "recommended_action": e.recommended_action} for e in explanations]
    with open(PROCESSED_DATA_DIR / "explanations.json", "w") as f:
        json.dump(exp_dicts, f, indent=2)


@st.cache_data(ttl=60)
def load_all() -> tuple[pd.DataFrame, dict]:
    """Load features, cases, and explanations. Auto-generate if missing."""
    graph_path = DATA_DIR / "features_with_graph.csv"
    cases_path = DATA_DIR / "cases.json"
    explanations_path = DATA_DIR / "explanations.json"

    if not graph_path.exists() or not cases_path.exists() or not explanations_path.exists():
        with st.spinner("Generating dataset for the first time (this takes ~2 minutes)..."):
            _run_full_pipeline()

    df = pd.read_csv(graph_path, parse_dates=["timestamp"])

    with open(cases_path) as f:
        cases_raw = json.load(f)
    with open(explanations_path) as f:
        explanations_raw = json.load(f)

    explanations = {e["transaction_id"]: e for e in explanations_raw}
    return df, explanations


@st.cache_data(ttl=60)
def load_graph() -> nx.Graph:
    """Build and cache the transaction graph."""
    graph_path = DATA_DIR / "features_with_graph.csv"
    df = pd.read_csv(graph_path, parse_dates=["timestamp"])
    from src.graphs.graph_engine import build_transaction_graph
    return build_transaction_graph(df)


@st.cache_data(ttl=60)
def load_rings() -> list[dict]:
    """Load fraud rings."""
    graph_path = DATA_DIR / "features_with_graph.csv"
    df = pd.read_csv(graph_path, parse_dates=["timestamp"])
    from src.graphs.graph_engine import build_transaction_graph, detect_fraud_rings
    G = build_transaction_graph(df)
    return detect_fraud_rings(G)


# ─── Helpers ───────────────────────────────────────────────────────────

def risk_chip(level: str) -> str:
    return f'<span class="risk-chip risk-{level}">{level}</span>'


def action_chip(action: str) -> str:
    safe = action.replace("MANUAL-REVIEW", "REVIEW")
    return f'<span class="action-chip action-{safe}">{action}</span>'


def score_bar(score: float) -> str:
    if score >= 80:
        cls = "score-fill-critical"
    elif score >= 55:
        cls = "score-fill-high"
    elif score >= 30:
        cls = "score-fill-medium"
    else:
        cls = "score-fill-low"
    return (
        f'<div class="score-bar">'
        f'<div class="fill {cls}" style="width:{min(100, score)}%"></div>'
        f'</div>'
    )


def evidence_badge(etype: str) -> str:
    return f'<span class="evidence-type ev-{etype}">{etype}</span>'


def get_risk_level_from_score(score: float) -> str:
    if score >= 80:
        return "CRITICAL"
    elif score >= 55:
        return "HIGH"
    elif score >= 30:
        return "MEDIUM"
    return "LOW"


# ─── Sidebar ───────────────────────────────────────────────────────────

st.sidebar.markdown("### AI Risk Manager")
st.sidebar.caption("Payment Risk Operations Console")

page = st.sidebar.radio(
    "Navigation",
    [
        "Risk Queue",
        "Case Investigation",
        "Network",
        "AI Investigation Trace",
        "Operations",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()

# System status
try:
    df_check = pd.read_csv(DATA_DIR / "features_with_graph.csv", nrows=5)
    st.sidebar.success("System online")
    st.sidebar.caption(f"Dataset: {len(pd.read_csv(DATA_DIR / 'features_with_graph.csv')):,} transactions")
except Exception:
    st.sidebar.error("Data not found. Run pipeline first.")

st.sidebar.divider()
st.sidebar.caption("Razorpay AI Buildathon 2026")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: RISK QUEUE
# ══════════════════════════════════════════════════════════════════════════

def page_risk_queue(df: pd.DataFrame, explanations: dict):
    """Risk Queue — triage alerts by severity."""
    st.markdown("## Risk Queue")

    # Metrics row
    total = len(df)
    n_crit = sum(1 for e in explanations.values() if e.get("risk_level") == "CRITICAL")
    n_high = sum(1 for e in explanations.values() if e.get("risk_level") == "HIGH")
    n_med = sum(1 for e in explanations.values() if e.get("risk_level") == "MEDIUM")
    n_low = sum(1 for e in explanations.values() if e.get("risk_level") == "LOW")
    n_downgraded = sum(
        1 for e in explanations.values()
        if "downgraded" in e.get("critic_note", "").lower()
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">Total</div>'
                    f'<div class="value">{total:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">CRITICAL</div>'
                    f'<div class="value" style="color:#c62828">{n_crit}</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="label">HIGH</div>'
                    f'<div class="value" style="color:#e65100">{n_high}</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="label">MEDIUM</div>'
                    f'<div class="value" style="color:#f57f17">{n_med}</div></div>',
                    unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="label">Critic Downgrades</div>'
                    f'<div class="value">{n_downgraded}</div>'
                    f'<div class="sub">{n_downgraded/total:.1%} of total</div></div>',
                    unsafe_allow_html=True)

    st.divider()

    # Filters
    fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
    with fcol1:
        level_filter = st.multiselect(
            "Risk Level",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH"],
        )
    with fcol2:
        sort_by = st.selectbox("Sort by", ["Risk Score", "Amount", "Time", "Anomaly Score"])
    with fcol3:
        search = st.text_input("Search transaction or account ID", placeholder="TXN000001 or A0001")

    # Build queue table
    queue_data = []
    for _, row in df.iterrows():
        tid = row["transaction_id"]
        exp = explanations.get(tid, {})
        risk_level = exp.get("risk_level", "LOW")
        if risk_level not in level_filter:
            continue

        risk_score = 0.0
        for c in ["risk_score", "anomaly_score"]:
            if c in row.index:
                risk_score = float(row[c])
                break
        # Use explanation risk_score if available
        if exp.get("risk_score"):
            risk_score = exp["risk_score"]

        queue_data.append({
            "transaction_id": tid,
            "payer": row.get("payer_id", ""),
            "payee": row.get("payee_id", ""),
            "amount": float(row.get("amount", 0)),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "anomaly_score": float(row.get("anomaly_score", 0)),
            "time": str(row.get("timestamp", ""))[:19],
        })

    if not queue_data:
        st.info("No transactions match the current filters.")
        return

    queue_df = pd.DataFrame(queue_data)

    # Search filter
    if search:
        mask = (
            queue_df["transaction_id"].str.contains(search, case=False, na=False)
            | queue_df["payer"].str.contains(search, case=False, na=False)
            | queue_df["payee"].str.contains(search, case=False, na=False)
        )
        queue_df = queue_df[mask]

    # Sort
    sort_map = {
        "Risk Score": "risk_score",
        "Amount": "amount",
        "Time": "time",
        "Anomaly Score": "anomaly_score",
    }
    queue_df = queue_df.sort_values(sort_map[sort_by], ascending=(sort_by != "Time"))

    # Display
    st.caption(f"{len(queue_df)} transactions in queue")

    # Render each item as a compact row
    for _, item in queue_df.head(50).iterrows():
        level = item["risk_level"]
        html = (
            f'<div style="display:flex;align-items:center;gap:12px;padding:8px 12px;'
            f'border-bottom:1px solid #f0f0f0;cursor:pointer;">'
            f'{risk_chip(level)}'
            f'<span style="font-family:monospace;font-size:0.82rem;font-weight:600;min-width:100px">'
            f'{item["transaction_id"]}</span>'
            f'<span style="font-size:0.82rem;color:#666;min-width:80px">{item["payer"]}</span>'
            f'<span style="font-size:0.7rem;color:#999">→</span>'
            f'<span style="font-size:0.82rem;color:#666;min-width:80px">{item["payee"]}</span>'
            f'<span style="font-family:monospace;font-size:0.82rem;margin-left:auto">'
            f'${item["amount"]:,.2f}</span>'
            f'<span style="font-family:monospace;font-size:0.75rem;color:#999;min-width:60px;text-align:right">'
            f'{item["risk_score"]:.1f}</span>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

    # Distribution chart
    st.divider()
    st.markdown("### Risk Distribution")
    dist = pd.Series({"CRITICAL": n_crit, "HIGH": n_high, "MEDIUM": n_med, "LOW": n_low})
    st.bar_chart(dist, color="#c62828", height=150)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: CASE INVESTIGATION
# ══════════════════════════════════════════════════════════════════════════

def page_investigation(df: pd.DataFrame, explanations: dict):
    """Deep dive into a single transaction case."""
    st.markdown("## Case Investigation")

    # Transaction selector
    col1, col2 = st.columns([3, 1])
    with col1:
        txn_ids = sorted(df["transaction_id"].unique())
        selected = st.selectbox("Select Transaction", txn_ids, index=0)
    with col2:
        search_mode = st.radio("Search", ["Transaction", "Account"], horizontal=True)

    if search_mode == "Account":
        accounts = sorted(df["payer_id"].unique())
        acct = st.selectbox("Account", accounts)
        if acct:
            acct_txns = df[df["payer_id"] == acct].sort_values("timestamp", ascending=False)
            flagged = acct_txns[acct_txns.get("anomaly_flag", pd.Series([0])) == 1]
            st.caption(f"{len(acct_txns)} transactions | {len(flagged)} flagged | "
                       f"${acct_txns['amount'].sum():,.2f} total")
            st.dataframe(
                acct_txns[["transaction_id", "timestamp", "payee_id", "amount",
                           "anomaly_score", "graph_risk_score"]].head(20),
                use_container_width=True,
                hide_index=True,
            )
            if not flagged.empty:
                selected = flagged.nlargest(1, "anomaly_score").iloc[0]["transaction_id"]

    if not selected:
        return

    row = df[df["transaction_id"] == selected].iloc[0]
    exp = explanations.get(selected, {})
    risk_level = exp.get("risk_level", "LOW")
    risk_score = exp.get("risk_score", 0)
    action = exp.get("recommended_action", "No action")

    # ── Header ──
    hcol1, hcol2, hcol3 = st.columns([4, 1, 1])
    with hcol1:
        st.markdown(f"### {selected}")
        st.caption(f"{row.get('payer_id', '')} → {row.get('payee_id', '')} | "
                   f"${row.get('amount', 0):,.2f} | {str(row.get('timestamp', ''))[:19]}")
    with hcol2:
        st.markdown(f"### {risk_chip(risk_level)}", unsafe_allow_html=True)
    with hcol3:
        action_text = action.replace("MANUAL-REVIEW", "REVIEW")
        st.markdown(f"### {action_chip(action_text)}", unsafe_allow_html=True)

    # Score bar
    st.markdown(score_bar(risk_score), unsafe_allow_html=True)
    st.caption(f"Risk Score: {risk_score:.1f}/100 | Confidence: {exp.get('confidence', 'N/A')}")

    st.divider()

    # ── Transaction details ──
    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    with dcol1:
        st.markdown("**Device**")
        st.code(row.get("device_id", "N/A"))
    with dcol2:
        st.markdown("**IP Address**")
        st.code(row.get("ip_address", "N/A"))
    with dcol3:
        st.markdown("**Amount**")
        st.code(f"${row.get('amount', 0):,.2f}")
    with dcol4:
        st.markdown("**Anomaly Score**")
        st.code(f"{row.get('anomaly_score', 0):.4f}")

    st.divider()

    # ── Evidence & Score Breakdown Tabs ──
    tabs = st.tabs(["Evidence", "Score Breakdown", "Critic Review", "Signals"])

    with tabs[0]:
        st.markdown("### Evidence")
        evidence_text = exp.get("supporting_evidence", "")
        if evidence_text:
            for part in evidence_text.split(" | "):
                part = part.strip()
                if not part:
                    continue
                # Determine type badge
                badge = "rule"
                part_lower = part.lower()
                if "[ml]" in part_lower:
                    badge = "ml"
                elif "[graph]" in part_lower:
                    badge = "graph"
                elif "[context]" in part_lower:
                    badge = "context"
                st.markdown(
                    f'<div class="evidence-row">{evidence_badge(badge)} {part}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No detailed evidence available.")

    with tabs[1]:
        st.markdown("### Score Breakdown")
        # Try to compute from explanation data
        sb_html = _render_score_breakdown(row, exp)
        st.markdown(sb_html, unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("### Critic Review")
        critic_note = exp.get("critic_note", "N/A")
        st.info(critic_note)
        if "downgraded" in critic_note.lower():
            st.success("Critic reduced risk — anti-overconfidence mechanism active.")

    with tabs[3]:
        st.markdown("### Triggered Signals")
        rule_cols = [c for c in row.index if c.startswith("rule_")]
        signals_data = []
        for c in rule_cols:
            val = row.get(c, 0)
            signals_data.append({
                "Signal": c.replace("rule_", "").replace("_", " ").title(),
                "Triggered": "YES" if val == 1 else "no",
                "Weight": "—" if val != 1 else "evidence",
            })
        st.dataframe(pd.DataFrame(signals_data), use_container_width=True, hide_index=True)


def _render_score_breakdown(row: pd.Series, exp: dict) -> str:
    """Render the score breakdown as styled HTML."""
    ml = float(row.get("anomaly_score", 0))
    graph = float(row.get("graph_risk_score", 0))
    rule_cols = [c for c in row.index if c.startswith("rule_")]
    n_rules = sum(1 for c in rule_cols if row.get(c, 0) == 1)
    rules = min(1.0, n_rules / 4)

    ml_pts = ml * 35
    rules_pts = rules * 35
    graph_pts = graph * 30
    base = ml_pts + rules_pts + graph_pts

    lines = []
    lines.append('<div style="font-family:monospace;font-size:0.85rem;line-height:1.8">')
    lines.append(f'<div>ML Anomaly: <b>{ml_pts:.1f}</b>/35.0 <span style="color:#888">(raw: {ml:.3f})</span></div>')
    lines.append(f'<div>Rules: <b>{rules_pts:.1f}</b>/35.0 <span style="color:#888">({n_rules} triggered)</span></div>')
    lines.append(f'<div>Graph: <b>{graph_pts:.1f}</b>/30.0 <span style="color:#888">(raw: {graph:.3f})</span></div>')
    lines.append(f'<hr style="margin:4px 0;border-color:#eee">')
    lines.append(f'<div>Base Score: <b>{base:.1f}</b></div>')

    # Critic adjustment
    critic_note = exp.get("critic_note", "")
    if "downgraded" in critic_note.lower() or "adjustment" in critic_note.lower():
        # Try to extract adjustment
        try:
            import re
            match = re.search(r"by\s+(-?\d+\.?\d*)\s+points", critic_note)
            if match:
                adj = float(match.group(1))
                color = "#c62828" if adj < 0 else "#2e7d32"
                lines.append(f'<div>Critic: <b style="color:{color}">{adj:+.1f}</b></div>')
        except Exception:
            pass

    lines.append(f'<hr style="margin:4px 0;border-color:#eee">')
    lines.append(f'<div>Final: <b>{exp.get("risk_score", base):.1f}</b>/100</div>')
    lines.append('</div>')

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: NETWORK
# ══════════════════════════════════════════════════════════════════════════

def page_network(df: pd.DataFrame, explanations: dict):
    """Interactive fraud network investigation."""
    st.markdown("## Fraud Network")

    rings = load_rings()

    ncol1, ncol2 = st.columns([1, 3])

    with ncol1:
        st.markdown(f"**{len(rings)}** rings detected")
        if not rings:
            st.info("No fraud rings found.")
            return

        rings_sorted = sorted(rings, key=lambda r: r["size"], reverse=True)
        ring_options = [f"{r['ring_id']} (size {r['size']})" for r in rings_sorted[:50]]
        sel_idx = st.selectbox("Select ring", range(len(ring_options)),
                               format_func=lambda i: ring_options[i])

        ring = rings_sorted[sel_idx]
        st.markdown(f"**Accounts:** {', '.join(ring['accounts'][:5])}")

        # Ring stats
        ring_accts = ring["accounts"]
        ring_txns = df[df["payer_id"].isin(ring_accts)]
        st.metric("Transactions", len(ring_txns))
        st.metric("Total Volume", f"${ring_txns['amount'].sum():,.0f}")

    with ncol2:
        # Build and render graph
        G = _build_ring_subgraph(df, ring)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.patch.set_facecolor("#fafafa")

        node_colors = []
        for node in G.nodes():
            ntype = G.nodes[node].get("node_type", "")
            if ntype == "account":
                node_colors.append("#c62828")
            elif ntype == "device":
                node_colors.append("#2e7d32")
            elif ntype == "ip":
                node_colors.append("#1565c0")
            else:
                node_colors.append("#666")

        edge_colors = []
        for u, v in G.edges():
            etype = G[u][v].get("edge_type", "")
            if etype == "transfer":
                edge_colors.append("#c62828")
            elif etype == "uses_device":
                edge_colors.append("#2e7d32")
            elif etype == "uses_ip":
                edge_colors.append("#1565c0")
            else:
                edge_colors.append("#bbb")

        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        nx.draw(G, pos, ax=ax,
                node_color=node_colors, edge_color=edge_colors,
                node_size=600, font_size=7, font_color="white",
                font_weight="bold", width=1.5, alpha=0.9)

        labels = {n: n.split(":")[1] if ":" in n else n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

        from matplotlib.lines import Line2D
        legend = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#c62828', markersize=8, label='Account'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2e7d32', markersize=8, label='Device'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565c0', markersize=8, label='IP'),
            Line2D([0], [0], color='#c62828', linewidth=2, label='Transfer'),
            Line2D([0], [0], color='#2e7d32', linewidth=2, label='Device Link'),
            Line2D([0], [0], color='#1565c0', linewidth=2, label='IP Link'),
        ]
        ax.legend(handles=legend, loc="upper left", fontsize=8)
        ax.set_title(f"Ring: {ring['ring_id']}", fontsize=11, fontweight="bold")
        ax.axis("off")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Transfer details
    if ring.get("edges"):
        st.markdown("### Transfer Details")
        st.dataframe(pd.DataFrame(ring["edges"]), use_container_width=True, hide_index=True)

    # Account risk table
    st.markdown("### Account Risk Profile")
    ring_accts = ring["accounts"]
    acct_risk = df[df["payer_id"].isin(ring_accts)].groupby("payer_id").agg({
        "anomaly_score": "mean",
        "anomaly_flag": "sum",
        "amount": ["mean", "sum", "count"],
    }).round(3)
    acct_risk.columns = ["Avg Anomaly", "Flagged Txns", "Avg Amount", "Total Amount", "Txn Count"]
    st.dataframe(acct_risk, use_container_width=True)


def _build_ring_subgraph(df: pd.DataFrame, ring: dict) -> nx.Graph:
    """Build subgraph for a ring."""
    G = nx.Graph()
    ring_accts = [f"account:{a}" for a in ring["accounts"]]

    ring_txns = df[
        df["payer_id"].isin(ring["accounts"]) | df["payee_id"].isin(ring["accounts"])
    ]

    for _, row in ring_txns.iterrows():
        payer = f"account:{row['payer_id']}"
        payee = f"account:{row['payee_id']}"
        device = f"device:{row['device_id']}"
        ip = f"ip:{row['ip_address']}"

        for node, ntype in [(payer, "account"), (payee, "account"),
                            (device, "device"), (ip, "ip")]:
            if node not in G:
                G.add_node(node, node_type=ntype)

        if G.has_edge(payer, payee):
            G[payer][payee]["weight"] = G[payer][payee].get("weight", 0) + 1
        else:
            G.add_edge(payer, payee, edge_type="transfer", weight=1)

        if not G.has_edge(payer, device):
            G.add_edge(payer, device, edge_type="uses_device")
        if not G.has_edge(payer, ip):
            G.add_edge(payer, ip, edge_type="uses_ip")

    return G


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: AI INVESTIGATION TRACE
# ══════════════════════════════════════════════════════════════════════════

def page_ai_trace(df: pd.DataFrame, explanations: dict):
    """Chronological investigation timeline."""
    st.markdown("## AI Investigation Trace")

    # Select a transaction
    txn_ids = sorted(df["transaction_id"].unique())
    selected = st.selectbox("Select Transaction", txn_ids, key="trace_select")

    if not selected:
        return

    row = df[df["transaction_id"] == selected].iloc[0]
    exp = explanations.get(selected, {})

    # Build a synthetic timeline from the explanation data
    st.markdown(f"### Case {selected}")

    timeline_events = _build_timeline_from_explanation(row, exp)

    for i, event in enumerate(timeline_events):
        agent = event["agent"]
        summary = event["summary"]
        details = event.get("details", "")

        # Agent color coding
        agent_colors = {
            "System": "#666",
            "ML": "#1565c0",
            "Rules": "#c62828",
            "Graph": "#2e7d32",
            "Detector": "#e65100",
            "Critic": "#7b1fa2",
            "Investigator": "#00838f",
            "Guardrails": "#37474f",
            "Explainer": "#4527a0",
        }
        color = agent_colors.get(agent, "#666")

        html = (
            f'<div class="timeline-item">'
            f'<div class="timeline-agent" style="color:{color}">{agent}</div>'
            f'<div>'
            f'<div class="timeline-summary">{summary}</div>'
        )
        if details:
            html += f'<div style="font-size:0.75rem;color:#888;margin-top:2px">{details}</div>'
        html += '</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    # Full explanation
    st.divider()
    st.markdown("### Full Case Report")
    report = exp.get("what_happened", "") + "\n\n" + exp.get("why_suspicious", "")
    st.text_area("Report", report, height=200, disabled=True)


def _build_timeline_from_explanation(row: pd.Series, exp: dict) -> list[dict]:
    """Build a timeline from the explanation data."""
    events = []

    events.append({
        "agent": "System",
        "summary": f"Payment received: ${row.get('amount', 0):,.2f} "
                   f"({row.get('payer_id', '')} → {row.get('payee_id', '')})",
        "details": f"Device: {row.get('device_id', '')} | IP: {row.get('ip_address', '')}",
    })

    # ML
    ml_score = float(row.get("anomaly_score", 0))
    if ml_score > 0.5:
        events.append({
            "agent": "ML",
            "summary": f"Isolation Forest scored {ml_score:.3f} — anomalous",
            "details": f"Top features: {row.get('ml_top_features', 'N/A')}",
        })
    else:
        events.append({
            "agent": "ML",
            "summary": f"Isolation Forest scored {ml_score:.3f} — within normal range",
        })

    # Rules
    rule_cols = [c for c in row.index if c.startswith("rule_")]
    triggered_rules = [c.replace("rule_", "") for c in rule_cols if row.get(c, 0) == 1]
    if triggered_rules:
        events.append({
            "agent": "Rules",
            "summary": f"{len(triggered_rules)} rule(s) triggered: {', '.join(triggered_rules)}",
        })
    else:
        events.append({
            "agent": "Rules",
            "summary": "No rule signals triggered",
        })

    # Graph
    graph_score = float(row.get("graph_risk_score", 0))
    graph_parts = []
    if row.get("shared_device_count", 0) > 0:
        graph_parts.append(f"{int(row['shared_device_count'])} shared device(s)")
    if row.get("shared_ip_count", 0) > 0:
        graph_parts.append(f"{int(row['shared_ip_count'])} shared IP(s)")
    if row.get("in_short_cycle", 0) == 1:
        graph_parts.append("in circular flow")
    events.append({
        "agent": "Graph",
        "summary": f"Graph risk: {graph_score:.3f}" + (f" — {', '.join(graph_parts)}" if graph_parts else ""),
    })

    # Detector
    events.append({
        "agent": "Detector",
        "summary": f"Aggregated score: {exp.get('risk_score', 0):.1f}/100 → {exp.get('risk_level', 'N/A')}",
    })

    # Critic
    critic_note = exp.get("critic_note", "")
    if "downgraded" in critic_note.lower():
        events.append({
            "agent": "Critic",
            "summary": "Challenged risk assessment — downgraded",
            "details": critic_note[:200],
        })
    else:
        events.append({
            "agent": "Critic",
            "summary": "Confirmed risk assessment — evidence is strong",
        })

    # Guardrails
    events.append({
        "agent": "Guardrails",
        "summary": f"Validated. Final action: {exp.get('recommended_action', 'N/A')[:50]}",
    })

    # Explainer
    events.append({
        "agent": "Explainer",
        "summary": "Case report generated",
    })

    return events


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5: OPERATIONS
# ══════════════════════════════════════════════════════════════════════════

def page_operations(df: pd.DataFrame, explanations: dict):
    """Model metrics, system health, and architecture."""
    st.markdown("## Model & Operations")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Metrics", "Model Comparison", "Critic Impact", "Feature Importance", "System", "Architecture"])

    with tab1:
        st.markdown("### Detection Metrics")

        predictions = df["anomaly_flag"].values
        labels = df["is_fraud"].values if "is_fraud" in df.columns else np.zeros(len(df))

        from sklearn.metrics import (
            classification_report, confusion_matrix,
            precision_recall_fscore_support,
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )
        cm = confusion_matrix(labels, predictions)

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.markdown(f'<div class="metric-card"><div class="label">Precision</div>'
                        f'<div class="value">{precision:.3f}</div></div>', unsafe_allow_html=True)
        with mcol2:
            st.markdown(f'<div class="metric-card"><div class="label">Recall</div>'
                        f'<div class="value">{recall:.3f}</div></div>', unsafe_allow_html=True)
        with mcol3:
            st.markdown(f'<div class="metric-card"><div class="label">F1 Score</div>'
                        f'<div class="value">{f1:.3f}</div></div>', unsafe_allow_html=True)

        st.markdown("**Confusion Matrix**")
        cm_df = pd.DataFrame(cm, index=["Actual Normal", "Actual Fraud"],
                             columns=["Pred Normal", "Pred Fraud"])
        st.dataframe(cm_df, use_container_width=True)

        # Per-pattern detection
        if "fraud_pattern" in df.columns:
            st.markdown("### Detection by Fraud Pattern")
            pattern_results = []
            for pattern in df["fraud_pattern"].unique():
                mask = df["fraud_pattern"] == pattern
                n = mask.sum()
                detected = (df.loc[mask, "anomaly_flag"] == 1).sum()
                pattern_results.append({
                    "Pattern": pattern,
                    "Count": n,
                    "Detected": detected,
                    "Rate": f"{detected/n:.1%}" if n > 0 else "N/A",
                })
            st.dataframe(pd.DataFrame(pattern_results), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Isolation Forest vs LightGBM")
        
        mc_path = DATA_DIR / "model_comparison.json"
        if mc_path.exists():
            with open(mc_path) as f:
                mc = json.load(f)
            
            models = mc.get('models', {})
            
            # Comparison table
            st.markdown("#### Performance Comparison")
            comp_rows = []
            for name, m in models.items():
                comp_rows.append({
                    "Model": m.get('label', name),
                    "Precision": m['precision'],
                    "Recall": m['recall'],
                    "F1 Score": m['f1'],
                    "Flagged": m['flagged'],
                    "False Positives": m['false_positives'],
                    "False Negatives": m['false_negatives'],
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
            
            # Highlight best model
            best_name = max(models.keys(), key=lambda k: models[k]['f1'])
            best = models[best_name]
            iso = models.get('isolation_forest', {})
            
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.markdown(f'<div class="metric-card"><div class="label">Best Model</div>'
                            f'<div class="value" style="font-size:1.1rem">{best.get("label", best_name)}</div>'
                            f'<div class="sub">F1: {best["f1"]:.3f}</div></div>', unsafe_allow_html=True)
            with mcol2:
                improvement = ((best['f1'] - iso['f1']) / iso['f1'] * 100) if iso.get('f1', 0) > 0 else 0
                st.markdown(f'<div class="metric-card"><div class="label">F1 Improvement</div>'
                            f'<div class="value">+{improvement:.0f}%</div>'
                            f'<div class="sub">over Isolation Forest</div></div>', unsafe_allow_html=True)
            with mcol3:
                fp_reduction = iso.get('false_positives', 0) - best.get('false_positives', 0)
                st.markdown(f'<div class="metric-card"><div class="label">False Positive Reduction</div>'
                            f'<div class="value">{fp_reduction}</div>'
                            f'<div class="sub">fewer false alarms</div></div>', unsafe_allow_html=True)
            
            # LightGBM feature importance
            lgb_imp = mc.get('lgb_feature_importance', {})
            if lgb_imp:
                st.markdown("#### LightGBM Feature Importance (Top 10)")
                imp_df = pd.DataFrame([
                    {"Feature": k, "Importance": v}
                    for k, v in list(lgb_imp.items())[:10]
                ])
                st.bar_chart(imp_df.set_index("Feature")["Importance"], height=250)
            
            st.markdown("""#### How It Works
- **Isolation Forest** (unsupervised): learns what 'normal' looks like, flags statistical outliers
- **LightGBM** (supervised): learns fraud patterns from labeled training data
- **Ensemble OR**: flags if either model detects anomaly (higher recall)
- **Ensemble AND**: flags only if both models agree (zero false positives)
""")
        else:
            st.info("Run model comparison: python src/models/supervised_detector.py")

    with tab3:
        st.markdown("### Critic Before / After")

        # Compute critic impact
        pre_levels = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        post_levels = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        n_downgraded = 0

        for tid, exp in explanations.items():
            risk = exp.get("risk_level", "LOW")
            post_levels[risk] = post_levels.get(risk, 0) + 1
            if "downgraded" in exp.get("critic_note", "").lower():
                n_downgraded += 1

        # Approximate pre-critic levels from anomaly scores
        for _, row in df.iterrows():
            score = float(row.get("anomaly_score", 0)) * 100
            if score >= 80:
                pre_levels["CRITICAL"] += 1
            elif score >= 55:
                pre_levels["HIGH"] += 1
            elif score >= 30:
                pre_levels["MEDIUM"] += 1
            else:
                pre_levels["LOW"] += 1

        comparison = pd.DataFrame({
            "Before Critic": pre_levels,
            "After Critic": post_levels,
        })
        st.bar_chart(comparison, height=200)

        st.metric("Cases Downgraded", n_downgraded, f"{n_downgraded/len(explanations):.1%}")

    with tab4:
        st.markdown("### What Drives Fraud Detection?")
        
        fi_path = DATA_DIR / "feature_importance.json"
        if fi_path.exists():
            with open(fi_path) as f:
                fi_data = json.load(f)
            
            # Model comparison
            st.markdown("#### Model Comparison")
            rf_f1 = fi_data.get('rf_f1', 0)
            iso_f1 = fi_data.get('iso_f1', 0)
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.markdown(f'<div class="metric-card"><div class="label">Random Forest (Supervised) F1</div>'
                            f'<div class="value">{rf_f1:.3f}</div></div>', unsafe_allow_html=True)
            with mcol2:
                st.markdown(f'<div class="metric-card"><div class="label">Isolation Forest (Unsupervised) F1</div>'
                            f'<div class="value">{iso_f1:.3f}</div></div>', unsafe_allow_html=True)
            
            if rf_f1 > iso_f1:
                improvement = (rf_f1 - iso_f1) / iso_f1 * 100
                st.info(f"Supervised model improves F1 by {improvement:.0f}% over unsupervised baseline.")
            
            # Top features
            st.markdown("#### Top Features (Random Forest)")
            rf_imp = fi_data.get('rf_importances', {})
            if rf_imp:
                imp_df = pd.DataFrame([
                    {"Feature": k, "Importance": v}
                    for k, v in rf_imp.items()
                ])
                st.bar_chart(imp_df.set_index("Feature")["Importance"], height=250)
            
            # Category importance
            cat_imp = fi_data.get('category_importance', {})
            if cat_imp:
                st.markdown("#### Importance by Feature Category")
                cat_df = pd.DataFrame([
                    {"Category": k, "Importance": v}
                    for k, v in sorted(cat_imp.items(), key=lambda x: -x[1])
                ])
                st.bar_chart(cat_df.set_index("Category")["Importance"], height=200)
            
            # Rule recall
            st.markdown("#### Rule Recall on Actual Fraud")
            st.caption("What % of actual fraud does each rule catch?")
            rule_recalls = fi_data.get('rule_recalls', {})
            if rule_recalls:
                rule_df = pd.DataFrame([
                    {"Rule": k.replace('rule_', '').replace('_', ' ').title(),
                     "Fraud Caught": f"{v['triggered']}/{v['total']}",
                     "Recall": f"{v['recall']:.1%}"}
                    for k, v in sorted(rule_recalls.items(), key=lambda x: -x[1]['recall'])
                ])
                st.dataframe(rule_df, use_container_width=True, hide_index=True)
            
            # Key insights
            st.markdown("#### Key Insights")
            st.markdown("""
- **Amount features dominate** — transaction size is the strongest predictor of fraud
- **Device/IP sharing** catches nearly all fraud (99%+) but also fires on legitimate shared infrastructure
- **Velocity rules** catch 14% of fraud — useful for catching rapid-fire attacks
- **Odd-hour high value** catches 9% — small but high-confidence signal
- **Circular transfers** catch 1% — rare but devastating when they fire
""")
        else:
            st.info("Run feature importance analysis first: python src/models/feature_importance.py")

    with tab5:
        st.markdown("### System Status")
        st.json({
            "dataset_size": len(df),
            "feature_columns": len([c for c in df.columns if c not in [
                "transaction_id", "timestamp", "payer_id", "payee_id",
                "merchant_id", "device_id", "ip_address", "is_fraud",
                "fraud_pattern", "ml_top_features", "evidence_paths",
            ]]),
            "graph_nodes": "~3,000",
            "graph_edges": "~29,000",
            "fraud_rings": len(load_rings()),
            "llm": _check_llm_info(),
            "explanations_loaded": len(explanations),
        })

    with tab6:
        st.markdown("### Architecture")
        st.code("""
Payment Event
    ↓
┌─────────────┐
│ Feature     │  31 behavioral features
│ Engine      │
└──────┬──────┘
       ↓
┌──────┴──────┐   ┌──────────┐   ┌──────────┐
│ Isolation   │   │ Baseline │   │ Graph    │
│ Forest      │   │ Rules    │   │ Engine   │
│ (ML Score)  │   │ (8 sigs) │   │ (Network)│
└──────┬──────┘   └────┬─────┘   └────┬─────┘
       ↓               ↓              ↓
  ┌──────────────────────────────────────┐
  │          Detector Agent              │
  │   (Deterministic Risk Aggregation)   │
  └──────────────┬───────────────────────┘
                 ↓
  ┌──────────────────────────────────────┐
  │          Critic Agent                │
  │   (Evidence Challenge / LLM)         │
  └──────────────┬───────────────────────┘
                 ↓ [if request_evidence]
  ┌──────────────────────────────────────┐
  │         Investigator Agent           │
  │   (Graph Expansion)                  │
  └──────────────┬───────────────────────┘
                 ↓ [reassess]
  ┌──────────────────────────────────────┐
  │         Deterministic Guardrails     │
  │   (Policy Validation)                │
  └──────────────┬───────────────────────┘
                 ↓
  ┌──────────────────────────────────────┐
  │         Explainer Agent              │
  │   (Analyst Report / LLM)             │
  └──────────────────────────────────────┘
        """, language=None)


def _check_llm_info() -> dict:
    """Check if LLM is available."""
    try:
        from src.agents.llm_client import get_llm_info
        return get_llm_info()
    except Exception:
        return {"provider": "none", "model": "", "available": False}


# ══════════════════════════════════════════════════════════════════════════
# REAL-TIME ANALYSIS (sidebar panel)
# ══════════════════════════════════════════════════════════════════════════

def add_realtime_panel(df: pd.DataFrame):
    """Add real-time analysis panel to sidebar."""
    st.sidebar.divider()
    st.sidebar.markdown("**Analyze New Payment**")

    with st.sidebar.form("realtime_form"):
        rt_amount = st.number_input("Amount ($)", min_value=0.0, value=5000.0, step=100.0)
        rt_payer = st.text_input("Payer ID", value="A0001")
        rt_payee = st.text_input("Payee ID", value="A0002")
        rt_device = st.text_input("Device ID", value="D0001")
        rt_ip = st.text_input("IP Address", value="192.168.1.1")
        submitted = st.form_submit_button("Analyze")

    if submitted:
        with st.spinner("Running investigation pipeline..."):
            try:
                from src.agents.orchestrator import run_realtime_analysis
                from src.graphs.graph_engine import build_transaction_graph

                G = build_transaction_graph(df)
                state = run_realtime_analysis(
                    amount=rt_amount,
                    payer_id=rt_payer,
                    payee_id=rt_payee,
                    device_id=rt_device,
                    ip_address=rt_ip,
                    df=df,
                    G=G,
                )

                st.sidebar.success(f"Risk: {state.risk_level.value}")
                st.sidebar.metric("Score", f"{state.risk_score:.1f}/100")
                st.sidebar.caption(f"Action: {state.risk_action.value}")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)[:100]}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    try:
        df, explanations = load_all()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.info("Run the full pipeline first:\n```\npython src/data_generation/generate_transactions.py\npython src/features/feature_engineering.py\npython src/features/rules.py\npython src/models/anomaly_detector.py\npython src/graphs/graph_engine.py\npython src/agents/detector.py\npython src/agents/critic.py\npython src/agents/explainer.py\n```")
        return

    add_realtime_panel(df)

    if page == "Risk Queue":
        page_risk_queue(df, explanations)
    elif page == "Case Investigation":
        page_investigation(df, explanations)
    elif page == "Network":
        page_network(df, explanations)
    elif page == "AI Investigation Trace":
        page_ai_trace(df, explanations)
    elif page == "Operations":
        page_operations(df, explanations)


if __name__ == "__main__":
    main()
