"""Convert amounts in JSON files from USD to INR."""
import json, os

USD_TO_INR = 83
PROCESSED = "data/processed"

# Convert cases.json
path = os.path.join(PROCESSED, "cases.json")
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
    for case in data:
        if "amount" in case:
            case["amount"] = round(case["amount"] * USD_TO_INR, 2)
        if "risk_score" in case:
            pass  # don't change risk scores
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Converted cases.json: {len(data)} cases")

# Convert cases_with_critic.json
path = os.path.join(PROCESSED, "cases_with_critic.json")
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
    for case in data:
        if "amount" in case:
            case["amount"] = round(case["amount"] * USD_TO_INR, 2)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Converted cases_with_critic.json: {len(data)} cases")

# Convert explanations.json
path = os.path.join(PROCESSED, "explanations.json")
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
    for exp in data:
        if "amount" in exp:
            exp["amount"] = round(exp["amount"] * USD_TO_INR, 2)
        # Update text references to dollar amounts
        if "what_happened" in exp and "$" in str(exp["what_happened"]):
            import re
            def replace_dollar(m):
                val = float(m.group(1).replace(",", ""))
                return f"INR {val * USD_TO_INR:,.2f}"
            exp["what_happened"] = re.sub(r'\$([\d,]+\.?\d*)', replace_dollar, str(exp["what_happened"]))
        if "supporting_evidence" in exp and "$" in str(exp["supporting_evidence"]):
            import re
            def replace_dollar2(m):
                val = float(m.group(1).replace(",", ""))
                return f"INR {val * USD_TO_INR:,.2f}"
            exp["supporting_evidence"] = re.sub(r'\$([\d,]+\.?\d*)', replace_dollar2, str(exp["supporting_evidence"]))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Converted explanations.json: {len(data)} explanations")

print("All JSON files converted to INR!")
