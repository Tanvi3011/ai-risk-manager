---
title: AI Risk Manager
emoji: 🔍
colorFrom: indigo
colorTo: red
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
license: mit
---

# AI Risk Manager

**AI-powered payment risk investigation and decisioning for Razorpay.**

AI Risk Manager detects suspicious payments, investigates their transaction networks, challenges weak alerts with an AI Critic, validates recommendations with deterministic guardrails, and produces an explainable payment-risk decision.

Built for the **Razorpay AI Buildathon 2026**.

---

## Architecture

```
Payment Event
    ↓
┌─────────────┐
│ Feature     │  31 behavioral features (amount, velocity, payee, device, IP, time)
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
  └──────────────┬───────────────────────┘
                 ↓
  ┌──────────────────────────────────────┐
  │     Decision: ALLOW / STEP-UP /      │
  │     MANUAL-REVIEW / HOLD             │
  └──────────────────────────────────────┘
```

## Key Features

### Multi-Agent Investigation Pipeline
- **Detector** — aggregates ML, rules, and graph signals into a risk score
- **Critic** — challenges weak evidence; with LLM, reasons over structured facts
- **Investigator** — expands graph connections on Critic request (feedback loop)
- **Explainer** — produces analyst-readable case reports (LLM or template)
- **Guardrails** — deterministic policy gate that validates all AI recommendations

### Critic Feedback Loop (Signature Feature)
```
Detector = 84/100 → Critic says evidence is insufficient →
Investigator expands graph → discovers connected suspicious accounts →
Critic reassesses → guardrail validates → final decision
```

### Decisioning Layer
| Risk Level | Action | Meaning |
|---|---|---|
| LOW | ALLOW | Normal payment |
| MEDIUM | STEP-UP | Additional verification |
| HIGH | MANUAL-REVIEW | Analyst investigation |
| CRITICAL | HOLD | Temporarily stop and investigate |

### Real-Time Transaction Analysis
Enter a new payment (amount, sender, receiver, device, IP) and run the complete pipeline live.

### Interactive Fraud Network
Click through rings and clusters; view shared devices/IPs, circular transfers, and connected suspicious accounts.

### Evidence-Based Score Breakdown
See exactly how the score is constructed: `ML 28/35 + Rules 24/35 + Graph 20/30 + Context 8/10 = 80/100`.

### Evidence Timeline
Chronological case trace: payment received → ML anomaly → rules → graph → Detector → Critic → Investigator → guardrails → Explainer.

### Anti-Overconfidence (Critic)
The Critic downgrades cases where evidence is weak, correlated, or insufficient — reducing false positives.

## Pages

1. **Risk Queue** — triage alerts by severity with filtering and search
2. **Case Investigation** — deep dive with evidence, score breakdown, Critic review, signals
3. **Network** — fraud ring visualization with account risk profiles
4. **AI Investigation Trace** — step-by-step agent timeline
5. **Operations** — metrics, Critic impact comparison, system health

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
python src/data_generation/generate_transactions.py

# 3. Run the full pipeline
python src/features/feature_engineering.py
python src/features/rules.py
python src/models/anomaly_detector.py
python src/graphs/graph_engine.py
python src/agents/detector.py
python src/agents/critic.py
python src/agents/explainer.py

# 4. (Optional) Enable LLM-powered Critic & Explainer
export OPENAI_API_KEY="sk-..."

# 5. Run evaluation
python src/evaluation.py

# 6. Launch dashboard
streamlit run app.py
```

## LLM Integration (Optional)

The system works fully offline with deterministic agents. When `OPENAI_API_KEY` is set, the Critic and Explainer use GPT-4o-mini for richer reasoning:

- **Critic**: structured JSON output with verdict, reasoning, missing evidence, investigation requests
- **Explainer**: analyst-style case report
- **Case Q&A**: answer questions from case evidence

The LLM **never** directly executes financial actions. Deterministic guardrails always validate.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Evaluation

| Metric | Value |
|---|---|
| Precision | ~0.66 |
| Recall | ~0.51 |
| F1 | ~0.58 |
| Cases Downgraded | ~93% |
| Graph Nodes | ~3,000 |
| Graph Edges | ~29,000 |
| Fraud Rings | ~3,000 |

## Fraud Archetypes

1. **Account Takeover** — unusual device/IP with high-value transactions
2. **Mule Account** — funds pass through to downstream accounts
3. **Device Farm** — multiple accounts sharing one device
4. **Circular Transfer** — money flows back to originator
5. **Velocity Attack** — rapid burst of transactions
6. **Odd-Hour High Value** — large payments at 0-5 AM

## Technology

- **ML**: Scikit-learn Isolation Forest (unsupervised)
- **Graph**: NetworkX heterogeneous relationship graph
- **LLM**: OpenAI GPT-4o-mini (optional)
- **UI**: Streamlit analyst console
- **Language**: Python 3.11+

## License

Internal hackathon project — Razorpay AI Buildathon 2026.

## Deployment

### Option 1: Streamlit Cloud (Recommended for Hackathon)

1. Push to GitHub:
```bash
git init
git add .
git commit -m "AI Risk Manager - Razorpay Hackathon"
git remote add origin https://github.com/YOUR_USERNAME/ai-risk-manager.git
git push -u origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" → select your repo → main branch → app.py
5. Click "Deploy"

**Optional:** Add your GROQ_API_KEY in Streamlit Cloud → Settings → Secrets:
```
GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=groq
GROQ_MODEL=qwen/qwen3.8-27b
```

### Option 2: Docker (Local)

```bash
# Build and run
docker compose up --build

# Or without docker-compose
docker build -t ai-risk-manager .
docker run -p 8501:8501 --env-file .env ai-risk-manager
```

### Option 3: Direct Run

```bash
pip install -r requirements.txt
python src/data_generation/generate_transactions.py
python src/features/feature_engineering.py
python src/features/rules.py
python src/models/anomaly_detector.py
python src/models/supervised_detector.py
python src/models/feature_importance.py
python src/graphs/graph_engine.py
python src/agents/detector.py
python src/agents/critic.py
python src/agents/explainer.py
python src/evaluation.py
streamlit run app.py
```
