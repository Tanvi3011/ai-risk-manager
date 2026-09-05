"""Tests for the AI Risk Manager pipeline.

Covers: config, data generation, features, rules, graph, detector,
critic, investigator, guardrails, orchestrator, explainer, evaluation.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

class TestConfig:
    def test_dataset_sizes(self):
        from src.config import NUM_TRANSACTIONS, NUM_USERS, RANDOM_SEED
        assert NUM_TRANSACTIONS > 0
        assert NUM_USERS > 0
        assert RANDOM_SEED == 42

    def test_data_directories(self):
        from src.config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
        assert DATA_DIR.exists()
        assert RAW_DATA_DIR.exists()


# ═══════════════════════════════════════════════════════════════════
# Data Generation
# ═══════════════════════════════════════════════════════════════════

class TestDataGeneration:
    @pytest.fixture(autouse=True)
    def load_data(self):
        from src.config import DATA_DIR
        path = DATA_DIR / "raw" / "transactions.csv"
        if path.exists():
            self.df = pd.read_csv(path, parse_dates=["timestamp"])
        else:
            self.df = None

    def test_dataset_exists(self):
        assert self.df is not None, "Run generate_transactions.py first"

    def test_row_count(self):
        assert len(self.df) == 10_000

    def test_required_columns(self):
        expected = [
            "transaction_id", "timestamp", "payer_id", "payee_id",
            "merchant_id", "amount", "device_id", "ip_address",
            "is_fraud", "fraud_pattern",
        ]
        for col in expected:
            assert col in self.df.columns, f"Missing: {col}"

    def test_no_duplicate_ids(self):
        assert self.df["transaction_id"].duplicated().sum() == 0

    def test_no_negative_amounts(self):
        assert (self.df["amount"] >= 0).all()

    def test_fraud_labels_present(self):
        assert self.df["is_fraud"].sum() > 0


# ═══════════════════════════════════════════════════════════════════
# Features
# ═══════════════════════════════════════════════════════════════════

class TestFeatures:
    @pytest.fixture(autouse=True)
    def load_features(self):
        from src.config import DATA_DIR
        path = DATA_DIR / "processed" / "features.csv"
        if path.exists():
            self.df = pd.read_csv(path, parse_dates=["timestamp"])
        else:
            self.df = None

    def test_features_exist(self):
        assert self.df is not None

    def test_feature_count(self):
        from src.features.feature_engineering import get_feature_columns
        features = get_feature_columns(self.df)
        assert len(features) >= 25

    def test_amount_features(self):
        assert "log_amount" in self.df.columns
        assert "amount_zscore" in self.df.columns

    def test_no_label_leakage(self):
        from src.features.feature_engineering import get_feature_columns
        features = get_feature_columns(self.df)
        assert "is_fraud" not in features
        assert "fraud_pattern" not in features


# ═══════════════════════════════════════════════════════════════════
# Rules
# ═══════════════════════════════════════════════════════════════════

class TestRules:
    @pytest.fixture(autouse=True)
    def load_data(self):
        from src.config import DATA_DIR
        path = DATA_DIR / "processed" / "features_with_rules.csv"
        if path.exists():
            self.df = pd.read_csv(path, parse_dates=["timestamp"])
        else:
            self.df = None

    def test_rules_exist(self):
        assert self.df is not None

    def test_rule_columns(self):
        rule_cols = [c for c in self.df.columns if c.startswith("rule_")]
        assert len(rule_cols) >= 6

    def test_odd_hour_logic(self):
        odd = self.df[self.df["rule_odd_hour_high_value"] == 1]
        if len(odd) > 0:
            assert (odd["is_odd_hour"] == 1).all()
            assert (odd["amount"] > 5000).all()


# ═══════════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════════

class TestGraph:
    @pytest.fixture(autouse=True)
    def load_data(self):
        from src.config import DATA_DIR
        path = DATA_DIR / "processed" / "features_with_graph.csv"
        if path.exists():
            self.df = pd.read_csv(path, parse_dates=["timestamp"])
        else:
            self.df = None

    def test_graph_features(self):
        assert self.df is not None
        assert "graph_risk_score" in self.df.columns
        assert "shared_device_count" in self.df.columns

    def test_graph_risk_range(self):
        scores = self.df["graph_risk_score"]
        assert scores.min() >= 0
        assert scores.max() <= 1

    def test_graph_builder(self):
        from src.graphs.graph_engine import build_transaction_graph
        G = build_transaction_graph(self.df.head(100))
        assert G.number_of_nodes() > 0

    def test_ring_detection(self):
        from src.graphs.graph_engine import build_transaction_graph, detect_fraud_rings
        G = build_transaction_graph(self.df.head(500))
        rings = detect_fraud_rings(G)
        assert isinstance(rings, list)


# ═══════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════

class TestState:
    def test_state_creation(self):
        from src.agents.state import InvestigationState, RiskLevel, RiskAction
        state = InvestigationState(transaction_id="T001", amount=1000.0)
        assert state.risk_level == RiskLevel.LOW
        assert state.risk_action == RiskAction.ALLOW

    def test_timeline_event(self):
        from src.agents.state import InvestigationState
        state = InvestigationState()
        state.add_timeline_event("test", "Agent", "summary")
        assert len(state.timeline) == 1
        assert state.timeline[0].step == "test"

    def test_evidence_add(self):
        from src.agents.state import InvestigationState
        state = InvestigationState()
        state.add_evidence({"type": "ml", "signal": "test", "description": "test"})
        assert len(state.evidence_list) == 1

    def test_serialization(self):
        from src.agents.state import InvestigationState
        state = InvestigationState(transaction_id="T001", amount=500.0)
        state.add_timeline_event("test", "Agent", "summary")
        d = state.to_dict()
        assert d["transaction_id"] == "T001"
        assert len(d["timeline"]) == 1

    def test_action_from_level(self):
        from src.agents.state import InvestigationState, RiskLevel, RiskAction
        state = InvestigationState()
        state.risk_level = RiskLevel.CRITICAL
        assert state.risk_action_from_level() == RiskAction.HOLD
        state.risk_level = RiskLevel.LOW
        assert state.risk_action_from_level() == RiskAction.ALLOW


# ═══════════════════════════════════════════════════════════════════
# Detector
# ═══════════════════════════════════════════════════════════════════

class TestDetector:
    def test_risk_score_bounds(self):
        from src.agents.detector import compute_risk_score
        score = compute_risk_score(0.5, 0.5, 0.5)
        assert 0 <= score <= 100

    def test_risk_level_mapping(self):
        from src.agents.detector import risk_level_from_score
        assert risk_level_from_score(0) == "LOW"
        assert risk_level_from_score(35) == "MEDIUM"
        assert risk_level_from_score(60) == "HIGH"
        assert risk_level_from_score(85) == "CRITICAL"

    def test_case_builder(self):
        from src.agents.detector import build_case
        from src.config import DATA_DIR
        df = pd.read_csv(DATA_DIR / "processed" / "features_with_graph.csv",
                         parse_dates=["timestamp"])
        case = build_case(df.iloc[0])
        assert case.risk_score >= 0
        assert case.risk_level in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


# ═══════════════════════════════════════════════════════════════════
# Critic
# ═══════════════════════════════════════════════════════════════════

class TestCritic:
    def test_critic_never_increases(self):
        from src.agents.state import InvestigationState, RiskLevel, ScoreBreakdown
        from src.agents.critic import _run_deterministic_critic

        state = InvestigationState(
            transaction_id="T001",
            risk_score=60.0,
            risk_level=RiskLevel.HIGH,
            triggered_signals=["rule_shared_device_multiuser", "rule_shared_ip_multiuser"],
            score_breakdown=ScoreBreakdown(ml_normalized=0.3, graph_raw=0.2),
        )
        state = _run_deterministic_critic(state)
        assert state.risk_score <= 60.0

    def test_downgrade_weak_signals(self):
        from src.agents.state import InvestigationState, RiskLevel, ScoreBreakdown
        from src.agents.critic import _run_deterministic_critic

        state = InvestigationState(
            transaction_id="T002",
            risk_score=50.0,
            risk_level=RiskLevel.MEDIUM,
            triggered_signals=["rule_shared_device_multiuser", "rule_shared_ip_multiuser"],
            score_breakdown=ScoreBreakdown(ml_normalized=0.3, graph_raw=0.2),
        )
        state = _run_deterministic_critic(state)
        assert state.critic_output["verdict"] == "downgrade"

    def test_confirm_strong_signals(self):
        from src.agents.state import InvestigationState, RiskLevel, ScoreBreakdown
        from src.agents.critic import _run_deterministic_critic

        state = InvestigationState(
            transaction_id="T003",
            risk_score=80.0,
            risk_level=RiskLevel.CRITICAL,
            triggered_signals=[
                "rule_high_amount", "rule_odd_hour_high_value",
                "ml_anomaly_flag", "graph_risk_flag",
            ],
            score_breakdown=ScoreBreakdown(ml_normalized=0.9, graph_raw=0.8),
        )
        state = _run_deterministic_critic(state)
        assert state.critic_output["verdict"] == "confirm"


# ═══════════════════════════════════════════════════════════════════
# Guardrails
# ═══════════════════════════════════════════════════════════════════

class TestGuardrails:
    def test_pass_normal_case(self):
        from src.agents.state import InvestigationState, RiskLevel, RiskAction
        from src.agents.guardrails import validate_guardrails

        state = InvestigationState(
            risk_score=50.0,
            risk_level=RiskLevel.MEDIUM,
            risk_action=RiskAction.STEP_UP,
            confidence="MEDIUM",
            evidence_list=[{"type": "ml", "signal": "test"}],
        )
        result = validate_guardrails(state)
        assert result.passed
        assert result.guardrail_action == RiskAction.STEP_UP

    def test_block_insufficient_evidence(self):
        from src.agents.state import InvestigationState, RiskLevel, RiskAction
        from src.agents.guardrails import validate_guardrails

        state = InvestigationState(
            risk_score=90.0,
            risk_level=RiskLevel.CRITICAL,
            risk_action=RiskAction.HOLD,
            confidence="HIGH",
            evidence_list=[],
        )
        result = validate_guardrails(state)
        assert not result.passed

    def test_block_premature_allow(self):
        from src.agents.state import InvestigationState, RiskLevel, RiskAction
        from src.agents.guardrails import validate_guardrails

        state = InvestigationState(
            risk_score=60.0,
            risk_level=RiskLevel.HIGH,
            risk_action=RiskAction.ALLOW,
            confidence="HIGH",
            evidence_list=[
                {"type": "ml", "signal": "high_anomaly_score", "value": 0.9},
                {"type": "graph", "signal": "high_graph_risk", "value": 0.8},
            ],
        )
        result = validate_guardrails(state)
        assert result.guardrail_action != RiskAction.ALLOW


# ═══════════════════════════════════════════════════════════════════
# Investigator
# ═══════════════════════════════════════════════════════════════════

class TestInvestigator:
    def test_investigate_graph(self):
        from src.agents.state import InvestigationState, RiskLevel
        from src.agents.investigator import investigate_transaction
        from src.graphs.graph_engine import build_transaction_graph
        from src.config import DATA_DIR

        df = pd.read_csv(DATA_DIR / "processed" / "features_with_graph.csv",
                         parse_dates=["timestamp"])
        G = build_transaction_graph(df.head(200))

        state = InvestigationState(
            transaction_id="T001",
            payer_id=str(df.iloc[0]["payer_id"]),
            payee_id=str(df.iloc[0]["payee_id"]),
            amount=float(df.iloc[0]["amount"]),
        )

        state = investigate_transaction(state, df.head(200), G, ["graph connections"])
        assert len(state.evidence_list) > 0
        assert state.investigator_output["requests_processed"] == 1


# ═══════════════════════════════════════════════════════════════════
# Explainer
# ═══════════════════════════════════════════════════════════════════

class TestExplainer:
    def test_template_explainer(self):
        from src.agents.state import InvestigationState, RiskLevel
        from src.agents.explainer import _run_template_explainer

        state = InvestigationState(
            transaction_id="T001",
            payer_id="A001", payee_id="A002",
            amount=5000.0, risk_level=RiskLevel.HIGH,
            risk_score=65.0,
        )
        state = _run_template_explainer(state)
        assert "report" in state.explainer_output
        assert len(state.explainer_output["report"]) > 0

    def test_qa_without_llm(self):
        from src.agents.state import InvestigationState, RiskLevel
        from src.agents.explainer import _template_answer

        state = InvestigationState(
            transaction_id="T001",
            risk_level=RiskLevel.HIGH,
            risk_score=65.0,
            risk_action="MANUAL-REVIEW",
        )
        answer = _template_answer("Why was this flagged?", state)
        assert len(answer) > 0


# ═══════════════════════════════════════════════════════════════════
# End-to-End
# ═══════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_pipeline_files_exist(self):
        from src.config import DATA_DIR
        required = [
            "raw/transactions.csv",
            "processed/features.csv",
            "processed/features_with_rules.csv",
            "processed/features_with_ml.csv",
            "processed/features_with_graph.csv",
            "processed/cases.json",
            "processed/explanations.json",
        ]
        for f in required:
            assert (DATA_DIR / f).exists(), f"Missing: {f}"

    def test_all_explanations_generated(self):
        from src.config import DATA_DIR
        with open(DATA_DIR / "processed" / "explanations.json") as f:
            explanations = json.load(f)
        assert len(explanations) == 10_000

    def test_critic_impact_measurable(self):
        from src.config import DATA_DIR
        with open(DATA_DIR / "processed" / "explanations.json") as f:
            raw = json.load(f)
        explanations = {e["transaction_id"]: e for e in raw} if isinstance(raw, list) else raw
        n_downgraded = sum(
            1 for e in explanations.values()
            if "downgraded" in e.get("critic_note", "").lower()
        )
        assert n_downgraded > 0, "Critic should downgrade some cases"
