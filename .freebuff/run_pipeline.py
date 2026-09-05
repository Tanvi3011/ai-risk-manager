"""Run full pipeline offline (no Streamlit)."""
import sys, json
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.features.feature_engineering import build_feature_matrix, get_feature_columns
from src.features.rules import apply_rules
from src.models.anomaly_detector import train_isolation_forest, predict_anomaly
from src.graphs.graph_engine import build_transaction_graph, compute_graph_signals
from src.agents.detector import build_all_cases, case_summary
from src.agents.explainer import explain_all
from src.config import DATA_DIR
from sklearn.preprocessing import StandardScaler

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("1. Loading raw transactions...")
df = pd.read_csv(RAW_DATA_DIR / "transactions.csv", parse_dates=["timestamp"])
print(f"   {len(df)} rows, amount range: {df['amount'].min():.0f} - {df['amount'].max():.0f}")

print("2. Building features...")
features = build_feature_matrix(df)
features.to_csv(PROCESSED_DATA_DIR / "features.csv", index=False)

print("3. Applying rules...")
df_rules = apply_rules(features)
df_rules.to_csv(PROCESSED_DATA_DIR / "features_with_rules.csv", index=False)

print("4. Training Isolation Forest...")
feature_cols = [c for c in get_feature_columns(df_rules) if not c.startswith("rule_")]
ml_features = [c for c in feature_cols if df_rules[c].dtype in ["float64","int64","bool","float32","int32"]]
X = df_rules[ml_features].fillna(0)
contamination = df_rules["is_fraud"].mean()
model, scaler = train_isolation_forest(X, contamination=contamination)
results = predict_anomaly(model, scaler, X, ml_features)
df_rules["anomaly_score"] = results["anomaly_score"].values
df_rules["anomaly_flag"] = results["anomaly_flag"].values
df_rules["ml_top_features"] = results["top_features"].values
df_rules.to_csv(PROCESSED_DATA_DIR / "features_with_ml.csv", index=False)

print("5. Building graph...")
G = build_transaction_graph(df_rules)
suspicious = set(df_rules.loc[df_rules["anomaly_flag"] == 1, "payer_id"].unique())
graph_signals = compute_graph_signals(G, df_rules, suspicious_accounts=suspicious)
df_out = df_rules.merge(graph_signals, on="transaction_id", how="left")
df_out.to_csv(PROCESSED_DATA_DIR / "features_with_graph.csv", index=False)

print("6. Running agents (Detector + Critic + Explainer)...")
cases = build_all_cases(df_out)
# Save cases
cases_dicts = [vars(c) for c in cases]
with open(PROCESSED_DATA_DIR / "cases.json", "w") as f:
    json.dump(cases_dicts, f, indent=2, default=str)
summary = case_summary(cases)
summary.to_csv(PROCESSED_DATA_DIR / "cases_summary.csv", index=False)

# Run critic inline (use the detector module's case objects)
from src.agents.critic import run_critic
from src.agents.state import InvestigationState

for case in cases:
    state = InvestigationState(
        transaction_id=case.transaction_id,
        payer_id=case.payer_id,
        payee_id=case.payee_id,
        amount=case.amount,
        timestamp=case.timestamp,
        ml_score=case.ml_score,
        graph_score=case.graph_score,
        rules_score=case.rules_score,
        risk_score=case.risk_score,
        risk_level=case.risk_level,
        evidence_list=case.evidence_list,
        triggered_signals=case.triggered_signals,
    )
    state = run_critic(state)
    case.risk_score = state.risk_score
    case.risk_level = state.risk_level.value
    case.original_risk_score = state.original_risk_score

# Save critic results
cases_dicts = [vars(c) for c in cases]
with open(PROCESSED_DATA_DIR / "cases_with_critic.json", "w") as f:
    json.dump(cases_dicts, f, indent=2, default=str)

# Run explainer
explanations = explain_all(cases)
exp_dicts = [{
    "transaction_id": e.transaction_id, "risk_level": e.risk_level,
    "confidence": e.confidence, "headline": e.headline,
    "what_happened": e.what_happened, "why_suspicious": e.why_suspicious,
    "supporting_evidence": e.supporting_evidence, "graph_context": e.graph_context,
    "critic_note": e.critic_note, "recommended_action": e.recommended_action,
    "risk_score": getattr(e, "risk_score", 0),
} for e in explanations]
with open(PROCESSED_DATA_DIR / "explanations.json", "w") as f:
    json.dump(exp_dicts, f, indent=2)

print("PIPELINE COMPLETE!")
print(f"Amounts in INR: min={df_out['amount'].min():,.0f}, max={df_out['amount'].max():,.0f}, mean={df_out['amount'].mean():,.0f}")
