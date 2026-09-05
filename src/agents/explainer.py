"""Explainer Agent — LLM-Powered Analyst Report.

Generates a professional case report from investigation state.
With LLM: produces analyst-style narrative
Without LLM: generates structured template

Output:
  - Case summary
  - Evidence analysis
  - Investigation trace
  - Risk assessment
  - Recommended next steps
"""

from __future__ import annotations

import json

from src.agents.state import InvestigationState
from src.agents.llm_client import (
    build_explainer_prompt,
    build_qa_prompt,
    call_llm,
    is_llm_available,
)


def run_explainer(state: InvestigationState) -> InvestigationState:
    """Generate explanation for the investigation.

    Uses LLM if available, otherwise generates structured template.
    """
    if is_llm_available():
        state = _run_llm_explainer(state)
    else:
        state = _run_template_explainer(state)

    state.add_timeline_event(
        step="explainer_generated",
        agent="Explainer",
        summary="Case report generated",
        details={"method": state.explainer_output.get("source", "template")},
    )

    return state


def answer_question(state: InvestigationState, question: str) -> str:
    """Answer a case-specific question."""
    if is_llm_available():
        state_dict = _state_to_clean_dict(state)
        prompt = build_qa_prompt(question, state_dict)
        response = call_llm(
            prompt=prompt,
            system_prompt="Answer only from the provided case data. Be concise.",
            temperature=0.1,
            max_tokens=500,
        )
        if response.success:
            return response.content
        return _template_answer(question, state)
    return _template_answer(question, state)


def _run_llm_explainer(state: InvestigationState) -> InvestigationState:
    """LLM-powered case report."""
    state_dict = _state_to_clean_dict(state)
    prompt = build_explainer_prompt(state_dict)

    response = call_llm(
        prompt=prompt,
        system_prompt=(
            "You are a payment risk analyst. Write professional case reports. "
            "Use bullet points. Be precise with numbers. Do not invent facts."
        ),
        temperature=0.2,
        max_tokens=1500,
    )

    if response.success and response.content:
        state.explainer_output = {
            "report": response.content,
            "source": "llm",
            "model": response.model,
            "usage": response.usage,
            "headline": _extract_headline(state),
            "risk_level": state.risk_level.value,
            "action": state.risk_action.value,
        }
        state.llm_used = True
    else:
        state = _run_template_explainer(state)

    return state


def _run_template_explainer(state: InvestigationState) -> InvestigationState:
    """Template-based case report."""
    lines = []
    lines.append(f"## Case {state.transaction_id}")
    lines.append("")
    lines.append(f"**Risk Level:** {state.risk_level.value}  ")
    lines.append(f"**Action:** {state.risk_action.value}  ")
    lines.append(f"**Score:** {state.risk_score:.1f}/100")
    lines.append("")

    lines.append("### Transaction")
    lines.append(f"- {state.payer_id} sent **${state.amount:,.2f}** to {state.payee_id}")
    lines.append(f"- Time: {state.timestamp}")
    lines.append(f"- Device: {state.device_id}  |  IP: {state.ip_address}")
    lines.append("")

    lines.append("### Score Breakdown")
    sb = state.score_breakdown
    lines.append(f"- ML Anomaly: {sb.ml_contribution:.1f}/{sb.weight_ml * 100:.0f}")
    lines.append(f"- Behavioral Rules: {sb.rules_contribution:.1f}/{sb.weight_rules * 100:.0f}")
    lines.append(f"- Graph Risk: {sb.graph_contribution:.1f}/{sb.weight_graph * 100:.0f}")
    lines.append(f"- Context: {sb.context_contribution:.1f}/10")
    lines.append(f"- Base Score: {sb.base_score:.1f}")
    if sb.critic_adjustment != 0:
        lines.append(f"- Critic Adjustment: {sb.critic_adjustment:+.1f}")
    lines.append(f"- **Final Score: {sb.final_score:.1f}**")
    lines.append("")

    lines.append("### Evidence")
    for e in state.evidence_list:
        if e.get("type") == "critic":
            continue
        lines.append(f"- [{e.get('type', '?').upper()}] {e.get('description', '')}")
    lines.append("")

    if state.critic_output:
        lines.append("### Critic Review")
        lines.append(f"- Verdict: {state.critic_output.get('verdict', 'N/A')}")
        for reason in state.critic_output.get("reasons", []):
            lines.append(f"- {reason}")
        lines.append("")

    lines.append("### Investigation Timeline")
    for event in state.timeline:
        lines.append(f"- **{event.agent}**: {event.summary}")
    lines.append("")

    lines.append("### Recommended Next Steps")
    action = state.risk_action.value
    if action == "HOLD":
        lines.append("- Escalate to fraud team immediately")
        lines.append("- Freeze account pending investigation")
        lines.append("- Review all transactions from this device/IP cluster")
    elif action == "MANUAL-REVIEW":
        lines.append("- Assign to analyst for detailed review")
        lines.append("- Verify transaction with payer")
        lines.append("- Check for related accounts")
    elif action == "STEP-UP":
        lines.append("- Request additional verification from payer")
        lines.append("- Monitor subsequent transactions")
    else:
        lines.append("- No action required, continue monitoring")

    report = "\n".join(lines)

    state.explainer_output = {
        "report": report,
        "source": "template",
        "model": None,
        "usage": {},
        "headline": _extract_headline(state),
        "risk_level": state.risk_level.value,
        "action": state.risk_action.value,
    }

    return state


def _template_answer(question: str, state: InvestigationState) -> str:
    """Answer questions from case data without LLM."""
    q = question.lower()

    if "why" in q and ("flag" in q or "risk" in q or "suspicious" in q):
        signals = state.triggered_signals
        if signals:
            return (
                f"Transaction {state.transaction_id} was flagged because of: "
                f"{', '.join(signals)}. Risk score: {state.risk_score:.1f}/100."
            )
        return "No strong signals detected for this transaction."

    if "connected" in q or "link" in q or "related" in q:
        graph_ev = state.get_evidence_by_type("graph")
        if graph_ev:
            return " | ".join(e.get("description", "") for e in graph_ev[:3])
        return "No graph connections found."

    if "block" in q or "hold" in q or "allow" in q:
        return (
            f"Current action: {state.risk_action.value}. "
            f"Risk level: {state.risk_level.value} ({state.risk_score:.1f}/100)."
        )

    if "amount" in q:
        return f"Transaction amount: ${state.amount:,.2f}."

    if "who" in q:
        return f"From {state.payer_id} to {state.payee_id}."

    return f"Current risk: {state.risk_level.value} ({state.risk_score:.1f}/100). Action: {state.risk_action.value}."


def _extract_headline(state: InvestigationState) -> str:
    """Extract a headline from the state."""
    level = state.risk_level.value
    action = state.risk_action.value
    return f"{level} — {state.transaction_id} — {action}"


def _state_to_clean_dict(state: InvestigationState) -> dict:
    """Convert state to a clean dict for LLM consumption."""
    return {
        "transaction_id": state.transaction_id,
        "payer_id": state.payer_id,
        "payee_id": state.payee_id,
        "amount": state.amount,
        "timestamp": state.timestamp,
        "device_id": state.device_id,
        "ip_address": state.ip_address,
        "risk_level": state.risk_level.value,
        "risk_action": state.risk_action.value,
        "risk_score": state.risk_score,
        "confidence": state.confidence,
        "evidence": state.evidence_list,
        "triggered_signals": state.triggered_signals,
        "score_breakdown": {
            "ml_contribution": state.score_breakdown.ml_contribution,
            "rules_contribution": state.score_breakdown.rules_contribution,
            "graph_contribution": state.score_breakdown.graph_contribution,
            "base_score": state.score_breakdown.base_score,
            "critic_adjustment": state.score_breakdown.critic_adjustment,
            "final_score": state.score_breakdown.final_score,
        },
        "critic_output": state.critic_output,
        "investigator_output": state.investigator_output,
        "guardrail_output": state.guardrail_output,
        "timeline": [
            {"agent": e.agent, "summary": e.summary}
            for e in state.timeline
        ],
    }
