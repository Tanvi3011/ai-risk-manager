"""Critic Agent — Evidence Challenge with LLM Reasoning.

The Critic receives the Detector's case and underlying evidence,
then challenges weak, duplicated, or non-independent signals.

With LLM: generates structured critique with reasoning
Without LLM: uses deterministic rules (same logic as before)

Critic output:
  - confirm / downgrade / request_evidence
  - confidence, reasons, missing evidence
  - adjustment amount
"""

from __future__ import annotations

import json
from collections import Counter

from src.agents.state import InvestigationState, RiskLevel, TimelineEvent
from src.agents.llm_client import (
    LLMResponse,
    build_critic_prompt,
    call_llm,
    is_llm_available,
)


WEAK_STANDALONE_SIGNALS = {
    "rule_shared_device_multiuser",
    "rule_shared_ip_multiuser",
}

STRONG_SIGNALS = {
    "rule_high_amount",
    "rule_odd_hour_high_value",
    "rule_circular_transfer_hint",
    "rule_high_velocity_5min",
    "rule_high_velocity_1h",
    "rule_many_new_payees_24h",
    "ml_anomaly_flag",
    "graph_risk_flag",
}


def run_critic(state: InvestigationState) -> InvestigationState:
    """Run Critic analysis on the investigation state.

    Uses LLM if available, otherwise falls back to deterministic logic.
    Always stores both the LLM recommendation and the final output.
    """
    if is_llm_available():
        state = _run_llm_critic(state)
    else:
        state = _run_deterministic_critic(state)

    state.add_timeline_event(
        step="critic_review",
        agent="Critic",
        summary=f"Critic verdict: {state.critic_output.get('verdict', 'confirm')}, "
                f"adjustment: {state.critic_output.get('adjustment', 0)}",
        details=state.critic_output,
    )

    return state


def _run_llm_critic(state: InvestigationState) -> InvestigationState:
    """LLM-powered Critic with structured output."""
    prompt = build_critic_prompt(
        evidence=state.evidence_list,
        risk_score=state.risk_score,
        risk_level=state.risk_level.value,
        signals=state.triggered_signals,
    )

    response = call_llm(
        prompt=prompt,
        system_prompt=(
            "You are a fraud risk critic. Respond ONLY with valid JSON. "
            "Never increase risk scores. Challenge weak evidence."
        ),
        temperature=0.2,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    if response.success and response.parsed:
        parsed = response.parsed
        verdict = parsed.get("verdict", "confirm")
        adjustment = max(0, min(0, parsed.get("adjustment", 0)))  # Never positive
        reasons = parsed.get("reasons", [])
        missing = parsed.get("missing_evidence", [])
        requests = parsed.get("requested_investigation", [])

        state.critic_output = {
            "verdict": verdict,
            "adjustment": adjustment,
            "confidence": parsed.get("confidence", "medium"),
            "reasons": reasons,
            "weak_signals": parsed.get("weak_signals", []),
            "missing_evidence": missing,
            "requested_investigation": requests,
            "source": "llm",
            "model": response.model,
            "usage": response.usage,
        }

        # Apply adjustment
        new_score = max(0, min(100, state.risk_score + adjustment))
        state.risk_score = round(new_score, 1)

        # Add critic evidence
        state.add_evidence({
            "type": "critic",
            "signal": "llm_critic_verdict",
            "value": verdict,
            "description": f"LLM Critic: {verdict}. " + " | ".join(reasons),
            "metadata": state.critic_output,
        })

        # If requesting evidence, store the requests
        if verdict == "request_evidence" and requests:
            state.evidence_requested = requests

        state.llm_used = True
    else:
        # LLM failed, fall back to deterministic
        state = _run_deterministic_critic(state)

    return state


def _run_deterministic_critic(state: InvestigationState) -> InvestigationState:
    """Deterministic Critic — same logic as before, adapted for shared state."""
    adjustment = 0.0
    reasoning = []
    weak_only = True
    independent_signals = set()

    strong_present = [s for s in state.triggered_signals if s in STRONG_SIGNALS]
    weak_present = [s for s in state.triggered_signals if s in WEAK_STANDALONE_SIGNALS]

    if strong_present:
        weak_only = False

    shared_ip = "rule_shared_ip_multiuser" in state.triggered_signals
    shared_dev = "rule_shared_device_multiuser" in state.triggered_signals

    if shared_ip and not strong_present:
        adjustment -= 20
        reasoning.append("Downgrade: Shared IP alone is weak evidence.")

    if shared_dev and not strong_present:
        adjustment -= 15
        reasoning.append("Downgrade: Shared device alone is weak evidence.")

    if shared_ip and shared_dev and len(strong_present) == 0:
        adjustment -= 10
        reasoning.append("Downgrade: Shared IP + device overlap (correlated signals).")

    ml_fired = "ml_anomaly_flag" in state.triggered_signals
    graph_fired = "graph_risk_flag" in state.triggered_signals
    strong_rules = [s for s in state.triggered_signals if s.startswith("rule_") and s not in WEAK_STANDALONE_SIGNALS]

    if ml_fired and graph_fired and len(strong_rules) == 0:
        adjustment -= 10
        reasoning.append("Downgrade: ML + graph agree but no behavioral rules fired.")

    ml_score = state.score_breakdown.ml_normalized
    graph_score = state.score_breakdown.graph_raw
    if ml_score < 0.5 and graph_score > 0.5:
        adjustment -= 8
        reasoning.append("Downgrade: Graph elevated but ML moderate.")

    evidence_types = [e.get("type", "") for e in state.evidence_list if e.get("type") != "critic"]
    if evidence_types:
        type_counts = Counter(evidence_types)
        dominant = type_counts.most_common(1)[0]
        if dominant[1] > len(evidence_types) * 0.7 and len(evidence_types) > 2:
            adjustment -= 5
            reasoning.append(f"Downgrade: {dominant[1]}/{len(evidence_types)} evidence from {dominant[0]}.")

    n_strong = len(strong_present)
    if n_strong >= 3:
        adjustment += 10
        reasoning.append(f"Boost: {n_strong} strong independent signals.")
    elif n_strong >= 2:
        adjustment += 5
        reasoning.append(f"Boost: {n_strong} strong signals.")

    independent_signals = set(state.triggered_signals) - WEAK_STANDALONE_SIGNALS
    adjustment = max(-40, min(0, adjustment))

    verdict = "confirm" if adjustment == 0 else "downgrade"

    state.critic_output = {
        "verdict": verdict,
        "adjustment": adjustment,
        "confidence": "high" if n_strong >= 3 else ("medium" if n_strong >= 2 else "low"),
        "reasons": reasoning,
        "weak_signals": weak_present,
        "missing_evidence": [],
        "requested_investigation": [],
        "source": "deterministic",
        "model": None,
        "usage": {},
    }

    new_score = max(0, min(100, state.risk_score + adjustment))
    state.risk_score = round(new_score, 1)

    state.add_evidence({
        "type": "critic",
        "signal": "deterministic_critic",
        "value": adjustment,
        "description": f"Critic: {verdict}. " + " | ".join(reasoning),
        "metadata": state.critic_output,
    })

    return state
