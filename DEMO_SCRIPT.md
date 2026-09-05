# AI Risk Manager — 3-Minute Demo Script
# Razorpay AI Buildathon 2026

---

## Opening (0:00 – 0:20) — The Problem

**Say:**
> "Payment fraud detection has a false-positive problem. Systems flag thousands of transactions,
> but analysts can't tell which ones are real threats and which are noise. Isolated signals —
> a shared IP, an odd hour, a velocity spike — each alone means almost nothing.
> We built AI Risk Manager to **investigate**, not just flag."

**Show:** Risk Queue page with the 5 metric cards:
- TOTAL: 10,000
- CRITICAL: 336
- HIGH: 481
- MEDIUM: 9,180
- CRITIC DOWNGRADES: 9,306 (93.1%)

> "10,000 transactions. The system identified 336 critical and 481 high-risk cases.
> But here's what's interesting — the AI Critic challenged 93% of initial assessments
> and reduced risk levels where evidence was weak."

---

## Act 1 (0:20 – 0:55) — Real-Time Analysis

**Say:**
> "Let's analyze a suspicious payment live."

**Action:** Open the sidebar → "Analyze New Payment" form:
- Amount: **99745**
- Payer ID: **A0414**
- Payee ID: **A0665**
- Device ID: **D0001**
- IP Address: **10.42.1.55**

**Click "Analyze"**

**Say:**
> "We just submitted a payment through the full pipeline: feature extraction → Isolation Forest →
> rules → graph analysis → risk aggregation → Critic → guardrails → explainer.
> All in one click."

---

## Act 2 (0:55 – 1:40) — Case Investigation: TXN006642

**Say:**
> "Now let's deep-dive into a real flagged case."

**Action:** Switch to **Case Investigation** page.
Select **TXN006642** from the dropdown.

**Point out on screen:**
> "This is TXN006642: A0414 sent $99,745 to A0665 at **3 AM**."

**Show the header:** CRITICAL risk chip + HOLD action

> "Risk level: CRITICAL. Action: HOLD — freeze account and escalate to fraud team."

**Show Score Breakdown tab:**
> "Look at how the score is constructed:
> - ML Anomaly: 29.5/35 — Isolation Forest scored 0.842
> - Rules: 26.3/35 — odd_hour_high_value, shared_device, shared_ip all fired
> - Graph: 25.5/30 — high graph risk score
> - Total: 81.2/100"

**Show Evidence tab:**
> "Four evidence sources agree. This isn't a single-signal alert —
> ML, behavioral rules, and graph analysis all independently flagged this."

**Show Signals tab:**
> "The triggered signals: odd_hour_high_value (3 AM), shared_device_multiuser,
> shared_ip_multiuser, and ml_anomaly_flag."

---

## Act 3 (1:40 – 2:20) — The Critic: Anti-Overconfidence

**Say:**
> "Now here's what makes this system different. Let me show you the Critic in action."

**Action:** Search for **TXN001211** in Case Investigation.
(A0846 → A0572, $17,091.28)

**Show Critic Review tab:**
> "The Detector initially scored this HIGH. But the Critic challenged it:
> 'ML and graph scores agree but no behavioral rule signals fired.
> This may indicate model correlation rather than true fraud evidence.'
> The Critic **downgraded by 5 points**."

**Switch to AI Investigation Trace page, select TXN001211:**
> "Here's the full investigation timeline:
> 1. Payment received
> 2. ML scored 0.816 — flagged as anomalous
> 3. Rules: only shared_device and shared_ip fired — weak standalone signals
> 4. Graph risk: 0.850 — elevated
> 5. Detector: aggregated to HIGH
> 6. **Critic: challenged — downgraded because weak signals overlap**
> 7. Guardrails: validated the downgrade
> 8. Explainer: generated case report"

> "The Critic didn't just change a number — it **reasoned** about why the evidence was insufficient.
> Shared device and shared IP alone are weak. Many legitimate users share devices and IPs."

---

## Act 4 (2:20 – 2:45) — Fraud Network

**Say:**
> "Let's look at the network."

**Action:** Switch to **Network** page. Select the first ring.

**Show the graph:**
> "This is a fraud ring — interconnected accounts sharing devices and IPs.
> Each red node is an account, green is a shared device, blue is a shared IP.
> The red edges are money transfers between them."

**Point to the account risk table:**
> "Every account in this ring has elevated anomaly scores.
> The graph engine detected 3,000+ rings across 10,000 transactions —
> patterns that no single-transaction model would catch."

---

## Act 5 (2:45 – 3:00) — Close

**Say:**
> "To summarize what we built:
>
> **Detect** — Isolation Forest + 8 behavioral rules + NetworkX graph analysis
> **Investigate** — multi-agent pipeline: Detector → Critic → Investigator → Explainer
> **Challenge** — the Critic questions weak evidence and reduces false positives
> **Explain** — every decision has a traceable evidence chain
> **Guard** — deterministic guardrails prevent unsupported AI decisions
>
> We don't just flag transactions. We investigate relationships,
> challenge weak alerts, and produce an explainable risk decision."

**Final line:**
> "One-line pitch: AI Risk Manager detects suspicious payments,
> investigates their transaction networks, challenges weak alerts with an AI Critic,
> validates recommendations with deterministic guardrails,
> and produces an explainable payment-risk decision —
> purpose-built for Razorpay's payment-risk operations."

---

## Quick Reference — Transaction IDs

| Demo Moment | Transaction ID | What to Show |
|---|---|---|
| Real-time analysis | A0414→A0665, $99,745 | Full pipeline live |
| Case deep-dive | TXN006642 | CRITICAL, odd-hour, evidence breakdown |
| Critic downgrade | TXN001211 | HIGH→downgraded, weak signal challenge |
| Fraud ring | Network page ring selector | Graph visualization |
| Normal baseline | TXN007175 | LOW/MEDIUM, no signals fired |

## Key Numbers to Remember

- **9,306 / 10,000** cases downgraded by Critic (93.1%)
- **336** CRITICAL alerts
- **$99,745** largest odd-hour transaction (TXN006642)
- **3,000+** fraud rings detected
- **31** behavioral features
- **4** agent types: Detector, Critic, Investigator, Explainer
- **F1: 0.577** (ML baseline) — Critic improves precision
