import sys, os, json
sys.path.insert(0, '.')

# Force clean regeneration
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from faker import Faker

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

NUM_TXN = 10000
NUM_USERS = 1000
NUM_MERCHANTS = 200
NUM_DEVICES = 1200
NUM_IPS = 800

_users = [f"A{str(i).zfill(4)}" for i in range(1, NUM_USERS + 1)]
_merchants = [f"M{str(i).zfill(4)}" for i in range(1, NUM_MERCHANTS + 1)]
_devices = [f"D{str(i).zfill(4)}" for i in range(1, NUM_DEVICES + 1)]
_ips = [fake.ipv4() for _ in range(NUM_IPS)]

start_date = datetime(2026, 1, 1)
transactions = []

# Normal transactions - INR range: ₹200 to ₹4,00,000
for i in range(NUM_TXN):
    payer = random.choice(_users)
    hour = random.choices(range(24), weights=[1,1,1,1,1,1,2,4,7,8,8,8,8,8,8,8,7,6,5,4,3,2,1,1])[0]
    ts = start_date + timedelta(days=random.randint(0, 180), hours=hour, minutes=random.randint(0, 59), seconds=random.randint(0, 59))
    amount = round(np.random.lognormal(mean=8.0, sigma=1.2), 2)
    amount = min(amount, 400000)
    transactions.append({
        "transaction_id": f"TXN{str(i+1).zfill(6)}", "timestamp": ts,
        "payer_id": payer, "payee_id": random.choice(_users),
        "merchant_id": random.choice(_merchants), "amount": amount,
        "device_id": random.choice(_devices), "ip_address": random.choice(_ips),
        "is_fraud": 0, "fraud_pattern": "normal",
    })

# Fraud Pattern 1: High-value odd hours - ₹25,000 to ₹4,00,000 at 0-5 AM
for _ in range(30):
    payer = random.choice(_users)
    ts = start_date + timedelta(days=random.randint(0, 180), hours=random.randint(0, 5), minutes=random.randint(0, 59))
    transactions.append({
        "transaction_id": f"TXN{len(transactions)+1:06d}", "timestamp": ts,
        "payer_id": payer, "payee_id": random.choice(_users),
        "merchant_id": random.choice(_merchants),
        "amount": round(random.uniform(25000, 400000), 2),
        "device_id": random.choice(_devices), "ip_address": random.choice(_ips),
        "is_fraud": 1, "fraud_pattern": "odd_hour_high_value",
    })

# Fraud Pattern 2: Rapid fan-out - multiple fast transactions
for _ in range(25):
    payer = random.choice(_users)
    base_ts = start_date + timedelta(days=random.randint(0, 180), hours=random.randint(8, 22))
    for j in range(random.randint(4, 8)):
        ts = base_ts + timedelta(minutes=j * random.randint(1, 3))
        transactions.append({
            "transaction_id": f"TXN{len(transactions)+1:06d}", "timestamp": ts,
            "payer_id": payer, "payee_id": random.choice(_users),
            "merchant_id": random.choice(_merchants),
            "amount": round(random.uniform(8300, 415000), 2),
            "device_id": random.choice(_devices), "ip_address": random.choice(_ips),
            "is_fraud": 1, "fraud_pattern": "rapid_fanout",
        })

# Fraud Pattern 3: New payee burst
for _ in range(20):
    payer = random.choice(_users)
    base_ts = start_date + timedelta(days=random.randint(0, 180), hours=random.randint(8, 22))
    for j in range(random.randint(5, 10)):
        ts = base_ts + timedelta(minutes=j * random.randint(2, 10))
        transactions.append({
            "transaction_id": f"TXN{len(transactions)+1:06d}", "timestamp": ts,
            "payer_id": payer, "payee_id": random.choice(_users),
            "merchant_id": random.choice(_merchants),
            "amount": round(random.uniform(4150, 249000), 2),
            "device_id": random.choice(_devices), "ip_address": random.choice(_ips),
            "is_fraud": 1, "fraud_pattern": "new_payee_burst",
        })

# Fraud Pattern 4: Circular transfers
for _ in range(15):
    chain = [random.choice(_users) for _ in range(random.randint(3, 6))]
    chain.append(chain[0])
    base_ts = start_date + timedelta(days=random.randint(0, 180), hours=random.randint(10, 22))
    for j in range(len(chain) - 1):
        ts = base_ts + timedelta(minutes=j * random.randint(5, 30))
        transactions.append({
            "transaction_id": f"TXN{len(transactions)+1:06d}", "timestamp": ts,
            "payer_id": chain[j], "payee_id": chain[j+1],
            "merchant_id": random.choice(_merchants),
            "amount": round(random.uniform(83000, 1660000), 2),
            "device_id": random.choice(_devices), "ip_address": random.choice(_ips),
            "is_fraud": 1, "fraud_pattern": "circular_transfer",
        })

# Fraud Pattern 5: Shared device
for _ in range(10):
    device = random.choice(_devices)
    accts = random.sample(_users, random.randint(3, 6))
    base_ts = start_date + timedelta(days=random.randint(0, 180), hours=random.randint(8, 22))
    for j, acct in enumerate(accts):
        ts = base_ts + timedelta(minutes=j * random.randint(1, 5))
        transactions.append({
            "transaction_id": f"TXN{len(transactions)+1:06d}", "timestamp": ts,
            "payer_id": acct, "payee_id": random.choice(_users),
            "merchant_id": random.choice(_merchants),
            "amount": round(random.uniform(41500, 1245000), 2),
            "device_id": device, "ip_address": random.choice(_ips),
            "is_fraud": 1, "fraud_pattern": "shared_device",
        })

df = pd.DataFrame(transactions).sort_values("timestamp")
print(f"Generated {len(df)} transactions")
print(f"Amount range: INR {df['amount'].min():,.0f} - INR {df['amount'].max():,.0f}")
print(f"Mean: INR {df['amount'].mean():,.0f}")
print(f"Fraud: {df['is_fraud'].sum()}")

# Save
from pathlib import Path
RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
df.to_csv(RAW / "transactions.csv", index=False)
print("Saved data/raw/transactions.csv")
print("Done - now run the full pipeline")
