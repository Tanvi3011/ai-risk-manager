"""LLM Client — Groq/OpenAI API wrapper with deterministic fallback.

Supports:
  - Groq (default): fast inference via GROQ_API_KEY
  - OpenAI: via OPENAI_API_KEY
  - Deterministic fallback: no API key needed

The LLM is used for:
  - Critic reasoning (challenge evidence, request more data)
  - Explainer narrative (analyst-style report)
  - Case Q&A (answer questions from case evidence)

The LLM NEVER:
  - Directly executes payment actions
  - Invents evidence not in the supplied data
  - Produces risk scores (that's deterministic)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Load .env file
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict
    parsed: dict | None = None
    success: bool = True
    error: str | None = None


def _get_llm_config() -> dict[str, str]:
    """Determine which LLM provider to use."""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    if provider == "groq" or (groq_key and not openai_key):
        return {
            "provider": "groq",
            "api_key": groq_key,
            "base_url": "https://api.groq.com/openai/v1",
            "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        }
    elif openai_key:
        return {
            "provider": "openai",
            "api_key": openai_key,
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        }
    return {"provider": "none", "api_key": "", "base_url": "", "model": ""}


def is_llm_available() -> bool:
    """Check if any LLM API key is configured."""
    config = _get_llm_config()
    return config["provider"] != "none" and bool(config["api_key"])


def get_llm_info() -> dict[str, str]:
    """Return info about the configured LLM."""
    config = _get_llm_config()
    return {
        "provider": config["provider"],
        "model": config["model"],
        "available": is_llm_available(),
    }


def call_llm(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1000,
    response_format: dict | None = None,
) -> LLMResponse:
    """Call the LLM API with structured prompting."""
    config = _get_llm_config()

    if not config["api_key"]:
        return LLMResponse(
            content="",
            model="deterministic-fallback",
            usage={},
            success=False,
            error="No LLM API key configured",
        )

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": config["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        # Try to parse as JSON
        parsed = None
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            parsed=parsed,
            success=True,
        )

    except Exception as e:
        return LLMResponse(
            content="",
            model="error",
            usage={},
            success=False,
            error=str(e),
        )


def build_critic_prompt(evidence: list[dict], risk_score: float, risk_level: str, signals: list[str]) -> str:
    """Build the prompt for the Critic agent."""
    evidence_text = "\n".join(
        f"  - [{e.get('type', '?')}] {e.get('signal', '?')}: {e.get('description', '?')}"
        for e in evidence
    )
    signals_text = ", ".join(signals) if signals else "none"

    return f"""You are a fraud risk critic. Your job is to CHALLENGE the initial risk assessment.
You must be skeptical but fair. You should downgrade when evidence is weak, correlated, or insufficient.

TRANSACTION UNDER REVIEW:
  Risk Score: {risk_score}/100
  Risk Level: {risk_level}
  Triggered Signals: {signals_text}

EVIDENCE:
{evidence_text}

Provide your analysis as JSON:
{{
  "verdict": "confirm" | "downgrade" | "request_evidence",
  "adjustment": <number, negative to downgrade, 0 to confirm>,
  "confidence": "high" | "medium" | "low",
  "reasons": ["reason 1", "reason 2"],
  "weak_signals": ["signal that is weak and why"],
  "missing_evidence": ["what additional evidence would help"],
  "requested_investigation": ["specific graph lookup or data query to perform"]
}}

Rules:
- shared_ip OR shared_device ALONE is weak evidence
- Signals from the same category (e.g., two ML signals) are not independent
- request_evidence only if you genuinely need more data before deciding
- Never increase the risk score (adjustment must be <= 0)
- If fewer than 2 independent strong signals exist, consider downgrading"""


def build_explainer_prompt(state_dict: dict) -> str:
    """Build the prompt for the Explainer agent."""
    return f"""You are a payment risk analyst writing a case report for a human investigator.
Write clearly, concisely, and professionally. Use bullet points. Do not invent facts.

CASE DATA:
{json.dumps(state_dict, indent=2, default=str)}

Write an analyst report with these sections:
1. CASE SUMMARY (2-3 sentences: what happened, current risk level, action taken)
2. EVIDENCE ANALYSIS (bullet points of key evidence, grouped by source: ML, Rules, Graph)
3. INVESTIGATION TRACE (chronological steps the AI agents took)
4. RISK ASSESSMENT (score breakdown and confidence level)
5. RECOMMENDED NEXT STEPS (specific actions for the analyst)

Be precise with numbers. Reference specific signals and evidence items.
Do not use phrases like "AI-powered" or "advanced detection". Write like an internal ops report."""


def build_qa_prompt(question: str, state_dict: dict) -> str:
    """Build the prompt for case-specific Q&A."""
    return f"""You are answering a question about a specific fraud investigation case.
Only answer from the case data provided. If the information is not available, say so.

CASE DATA:
{json.dumps(state_dict, indent=2, default=str)}

QUESTION: {question}

Answer concisely (2-4 sentences). Reference specific evidence items when possible.
Do not invent information not present in the case data."""
