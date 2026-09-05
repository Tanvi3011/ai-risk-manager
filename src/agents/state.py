"""Shared Investigation State — Multi-Agent Communication Layer.

All agents read and write to this shared state object.
The state is serializable to JSON for the evidence timeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskAction(str, Enum):
    ALLOW = "ALLOW"
    STEP_UP = "STEP-UP"
    REVIEW = "MANUAL-REVIEW"
    HOLD = "HOLD"


class AgentStep(str, Enum):
    PAYMENT_RECEIVED = "payment_received"
    ML_DETECTION = "ml_detection"
    RULE_EVALUATION = "rule_evaluation"
    GRAPH_ANALYSIS = "graph_analysis"
    DETECTOR_SCORED = "detector_scored"
    CRITIC_REVIEW = "critic_review"
    EVIDENCE_REQUESTED = "evidence_requested"
    INVESTIGATOR_EXPANDED = "investigator_expanded"
    CRITIC_RECONFIRMED = "critic_reconfirmed"
    GUARDRAIL_VALIDATED = "guardrail_validated"
    FINAL_DECISION = "final_decision"
    EXPLAINER_GENERATED = "explainer_generated"


@dataclass
class TimelineEvent:
    step: str
    agent: str
    timestamp: str
    summary: str
    details: dict = field(default_factory=dict)
    evidence_added: list[dict] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    ml_raw: float = 0.0
    ml_normalized: float = 0.0
    ml_contribution: float = 0.0
    rules_raw: float = 0.0
    rules_contribution: float = 0.0
    graph_raw: float = 0.0
    graph_contribution: float = 0.0
    context_contribution: float = 0.0
    base_score: float = 0.0
    critic_adjustment: float = 0.0
    guardrail_adjustment: float = 0.0
    final_score: float = 0.0
    weight_ml: float = 0.35
    weight_rules: float = 0.35
    weight_graph: float = 0.30


@dataclass
class InvestigationState:
    """Complete state for a single transaction investigation."""

    # Transaction identity
    transaction_id: str = ""
    payer_id: str = ""
    payee_id: str = ""
    amount: float = 0.0
    timestamp: str = ""
    device_id: str = ""
    ip_address: str = ""

    # Risk decision
    risk_level: RiskLevel = RiskLevel.LOW
    risk_action: RiskAction = RiskAction.ALLOW
    risk_score: float = 0.0
    confidence: str = "LOW"

    # Evidence
    evidence_list: list[dict] = field(default_factory=list)
    triggered_signals: list[str] = field(default_factory=list)
    graph_connections: list[dict] = field(default_factory=list)

    # Score breakdown
    score_breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    # Agent outputs
    detector_output: dict = field(default_factory=dict)
    critic_output: dict = field(default_factory=dict)
    investigator_output: dict = field(default_factory=dict)
    guardrail_output: dict = field(default_factory=dict)
    explainer_output: dict = field(default_factory=dict)

    # Timeline
    timeline: list[TimelineEvent] = field(default_factory=list)

    # Feedback loop
    evidence_requested: list[str] = field(default_factory=list)
    investigation_rounds: int = 0
    max_investigation_rounds: int = 2

    # Metadata
    llm_used: bool = False
    processing_time_ms: float = 0.0

    def add_timeline_event(
        self,
        step: str,
        agent: str,
        summary: str,
        details: dict | None = None,
        evidence_added: list[dict] | None = None,
    ):
        """Add a chronological event to the investigation timeline."""
        event = TimelineEvent(
            step=step,
            agent=agent,
            timestamp=datetime.now().isoformat(),
            summary=summary,
            details=details or {},
            evidence_added=evidence_added or [],
        )
        self.timeline.append(event)

    def add_evidence(self, evidence: dict):
        """Add a piece of evidence to the case."""
        self.evidence_list.append(evidence)

    def get_evidence_by_type(self, etype: str) -> list[dict]:
        """Get all evidence of a given type."""
        return [e for e in self.evidence_list if e.get("type") == etype]

    def risk_action_from_level(self) -> RiskAction:
        """Deterministic mapping from risk level to action."""
        mapping = {
            RiskLevel.LOW: RiskAction.ALLOW,
            RiskLevel.MEDIUM: RiskAction.STEP_UP,
            RiskLevel.HIGH: RiskAction.REVIEW,
            RiskLevel.CRITICAL: RiskAction.HOLD,
        }
        return mapping.get(self.risk_level, RiskAction.ALLOW)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        d["risk_action"] = self.risk_action.value
        d["timeline"] = [
            {**asdict(e), "step": e.step} for e in self.timeline
        ]
        return d

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> InvestigationState:
        """Deserialize from dict."""
        d = dict(d)
        d["risk_level"] = RiskLevel(d.get("risk_level", "LOW"))
        d["risk_action"] = RiskAction(d.get("risk_action", "ALLOW"))
        if "timeline" in d:
            d["timeline"] = [TimelineEvent(**t) for t in d["timeline"]]
        if "score_breakdown" in d and isinstance(d["score_breakdown"], dict):
            d["score_breakdown"] = ScoreBreakdown(**d["score_breakdown"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
