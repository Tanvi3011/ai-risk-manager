"""Deterministic Guardrails — Final Policy Gate.

Validates LLM recommendations against hard rules.
The LLM can recommend, but guardrails have the final say.

Guardrails:
  1. Risk score must be 0-100
  2. Evidence must exist for non-LOW risk levels
  3. LLM cannot fabricate evidence (validate against known signals)
  4. LLM cannot unilaterally ALLOW a CRITICAL case
  5. Confidence must be HIGH/MEDIUM/LOW
  6. Action must be valid (ALLOW/STEP-UP/REVIEW/HOLD)
  7. Evidence count minimum for HIGH/CRITICAL decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.state import InvestigationState, RiskAction, RiskLevel


@dataclass
class GuardrailResult:
    passed: bool = True
    original_action: RiskAction = RiskAction.ALLOW
    guardrail_action: RiskAction = RiskAction.ALLOW
    original_score: float = 0.0
    adjusted_score: float = 0.0
    violations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "original_action": self.original_action.value,
            "guardrail_action": self.guardrail_action.value,
            "original_score": self.original_score,
            "adjusted_score": self.adjusted_score,
            "violations": self.violations,
            "warnings": self.warnings,
            "summary": self.summary,
        }


# Minimum evidence count by risk level
MIN_EVIDENCE = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}

# Strong signal types that can support high-risk decisions
STRONG_EVIDENCE_TYPES = {"ml", "rule", "graph"}


def validate_guardrails(state: InvestigationState) -> GuardrailResult:
    """Apply all deterministic guardrails to the investigation state.

    Returns a GuardrailResult with any violations and the final action.
    """
    result = GuardrailResult(
        original_action=state.risk_action,
        guardrail_action=state.risk_action,
        original_score=state.risk_score,
        adjusted_score=state.risk_score,
    )

    violations = []

    # 1. Score bounds
    if state.risk_score < 0 or state.risk_score > 100:
        violations.append({
            "rule": "score_bounds",
            "severity": "critical",
            "detail": f"Score {state.risk_score} is outside valid range [0, 100]",
        })
        result.adjusted_score = max(0, min(100, state.risk_score))

    # 2. Evidence minimum
    min_ev = MIN_EVIDENCE.get(state.risk_level, 0)
    n_evidence = len(state.evidence_list)
    if n_evidence < min_ev:
        violations.append({
            "rule": "evidence_minimum",
            "severity": "critical",
            "detail": (
                f"Risk level {state.risk_level.value} requires {min_ev} evidence items, "
                f"but only {n_evidence} provided. Downgrading to safe level."
            ),
        })
        # Downgrade to the highest level supported by current evidence
        if n_evidence == 0:
            result.guardrail_action = RiskAction.ALLOW
            result.adjusted_score = min(result.adjusted_score, 25.0)
        elif n_evidence == 1:
            result.guardrail_action = RiskAction.STEP_UP
            result.adjusted_score = min(result.adjusted_score, 45.0)
        elif n_evidence == 2:
            result.guardrail_action = RiskAction.REVIEW
            result.adjusted_score = min(result.adjusted_score, 70.0)

    # 3. Cannot ALLOW a case with strong ML + graph agreement
    strong_ml = any(
        e.get("type") == "ml" and e.get("signal") in {"high_anomaly_score", "ml_anomaly_flag"}
        for e in state.evidence_list
    )
    strong_graph = any(
        e.get("type") == "graph" and e.get("value", 0) > 0.5
        for e in state.evidence_list
    )
    if (
        state.risk_action == RiskAction.ALLOW
        and strong_ml
        and strong_graph
        and state.risk_score > 40
    ):
        violations.append({
            "rule": "prevent_premature_allow",
            "severity": "high",
            "detail": (
                f"Cannot ALLOW when strong ML and graph signals agree "
                f"(score={state.risk_score:.1f}). Upgrading to REVIEW."
            ),
        })
        result.guardrail_action = RiskAction.REVIEW
        result.adjusted_score = max(result.adjusted_score, 55.0)

    # 4. Cannot fully HOLD without at least 3 independent evidence types
    if state.risk_action == RiskAction.HOLD:
        evidence_types = set(e.get("type", "") for e in state.evidence_list)
        strong_types = evidence_types & STRONG_EVIDENCE_TYPES
        if len(strong_types) < 2:
            warnings = result.warnings
            warnings.append(
                f"HOLD requires evidence from 2+ independent sources. "
                f"Currently have: {strong_types or 'none'}. "
                f"Recommendation may be overridden on review."
            )
            result.warnings = warnings

    # 5. Confidence validation
    valid_confidence = {"HIGH", "MEDIUM", "LOW"}
    if state.confidence not in valid_confidence:
        violations.append({
            "rule": "valid_confidence",
            "severity": "low",
            "detail": f"Invalid confidence '{state.confidence}'. Must be HIGH/MEDIUM/LOW.",
        })

    # 6. Action validation
    valid_actions = {a.value for a in RiskAction}
    if state.risk_action.value not in valid_actions:
        violations.append({
            "rule": "valid_action",
            "severity": "critical",
            "detail": f"Invalid action '{state.risk_action.value}'.",
        })

    # Apply
    result.violations = violations
    result.passed = len([v for v in violations if v["severity"] == "critical"]) == 0

    # Final action: use guardrail override if there were critical violations
    if not result.passed:
        result.summary = (
            f"Guardrail BLOCKED: {len(violations)} violation(s) detected. "
            f"Action changed from {result.original_action.value} to {result.guardrail_action.value}."
        )
    elif violations:
        result.summary = (
            f"Guardrail WARNING: {len(violations)} non-critical issue(s). "
            f"Action confirmed: {result.guardrail_action.value}."
        )
    else:
        result.summary = f"Guardrail PASSED. Action confirmed: {result.guardrail_action.value}."

    return result
