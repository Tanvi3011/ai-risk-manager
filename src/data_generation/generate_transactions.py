import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

fake = Faker()
Faker.seed(SEED)

NUM_TRANSACTIONS = 10000
NUM_USERS = 1000
NUM_MERCHANTS = 200
NUM_DEVICES = 1200
NUM_IPS = 800
FRAUD_SCENARIOS = 100


# --------------------------------------------------
# Generate IDs
# --------------------------------------------------

users = [f"A{str(i).zfill(4)}" for i in range(1, NUM_USERS + 1)]

merchants = [
    f"M{str(i).zfill(4)}"
    for i in range(1, NUM_MERCHANTS + 1)
]

devices = [
    f"D{str(i).zfill(4)}"
    for i in range(1, NUM_DEVICES + 1)
]

ips = [
    fake.ipv4()
    for _ in range(NUM_IPS)
]


# --------------------------------------------------
# Generate base transactions
# --------------------------------------------------

start_date = datetime(2026, 1, 1)

transactions = []

for i in range(NUM_TRANSACTIONS):

    payer = random.choice(users)

    # Most transactions happen during normal hours.
    hour = random.choices(
        range(24),
        weights=[
            1, 1, 1, 1, 1, 1,
            2, 4, 7, 8, 8, 8,
            8, 8, 8, 8, 7, 6,
            5, 4, 3, 2, 1, 1
        ]
    )[0]

    timestamp = start_date + timedelta(
        days=random.randint(0, 180),
        hours=hour,
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )

    # Most transactions are relatively small.
    amount = round(
        np.random.lognormal(mean=6.0, sigma=1.0),
        2
    )

    amount = min(amount, 50000)

    transaction = {
        "transaction_id": f"TXN{str(i + 1).zfill(6)}",
        "timestamp": timestamp,
        "payer_id": payer,
        "payee_id": random.choice(users),
        "merchant_id": random.choice(merchants),
        "amount": amount,
        "device_id": random.choice(devices),
        "ip_address": random.choice(ips),
        "is_fraud": 0,
        "fraud_pattern": "normal"
    }

    transactions.append(transaction)


df = pd.DataFrame(transactions)


# --------------------------------------------------
# Fraud Pattern 1: Rapid Fan-Out
# --------------------------------------------------

for _ in range(FRAUD_SCENARIOS):

    start_index = random.randint(
        0,
        len(df) - 6
    )

    payer = random.choice(users)

    payees = random.sample(users, 5)

    base_time = df.loc[
        start_index,
        "timestamp"
    ]

    for offset, payee in enumerate(payees):

        index = start_index + offset

        df.loc[index, "payer_id"] = payer
        df.loc[index, "payee_id"] = payee

        df.loc[index, "timestamp"] = (
            base_time +
            timedelta(seconds=offset * 20)
        )

        df.loc[index, "amount"] = round(
            random.uniform(5000, 15000),
            2
        )

        df.loc[index, "is_fraud"] = 1
        df.loc[index, "fraud_pattern"] = "rapid_fanout"

# --------------------------------------------------
# Fraud Pattern 2: Device Reuse
# --------------------------------------------------

for _ in range(FRAUD_SCENARIOS):

    shared_device = random.choice(devices)

    accounts = random.sample(
        users,
        random.randint(4, 6)
    )

    rows = random.sample(
        list(df.index),
        len(accounts)
    )

    for index, account in zip(
        rows,
        accounts
    ):

        df.loc[index, "payer_id"] = account

        df.loc[index, "device_id"] = shared_device

        df.loc[index, "is_fraud"] = 1

        df.loc[index, "fraud_pattern"] = "device_reuse"

# --------------------------------------------------
# Fraud Pattern 3: New Payee Burst
# --------------------------------------------------

for _ in range(FRAUD_SCENARIOS):

    payer = random.choice(users)

    payees = random.sample(
        users,
        5
    )

    rows = random.sample(
        list(df.index),
        5
    )

    base_time = df.loc[
        rows[0],
        "timestamp"
    ]

    for offset, (index, payee) in enumerate(
        zip(rows, payees)
    ):

        df.loc[index, "payer_id"] = payer
        df.loc[index, "payee_id"] = payee

        df.loc[index, "timestamp"] = (
            base_time +
            timedelta(minutes=offset)
        )

        df.loc[index, "amount"] = round(
            random.uniform(3000, 20000),
            2
        )

        df.loc[index, "is_fraud"] = 1
        df.loc[index, "fraud_pattern"] = (
            "new_payee_burst"
        )

# --------------------------------------------------
# Fraud Pattern 4: Odd-Hour High Value
# --------------------------------------------------

odd_hour_indices = random.sample(
    list(df.index),
    100
)

for index in odd_hour_indices:

    timestamp = df.loc[index, "timestamp"]

    timestamp = timestamp.replace(
        hour=random.choice([1, 2, 3, 4]),
        minute=random.randint(0, 59)
    )

    df.loc[index, "timestamp"] = timestamp

    df.loc[index, "amount"] = round(
        random.uniform(50000, 100000),
        2
    )

    df.loc[index, "is_fraud"] = 1
    df.loc[index, "fraud_pattern"] = "odd_hour_high_value"


# --------------------------------------------------
# Fraud Pattern 5: Shared IP
# --------------------------------------------------

for _ in range(FRAUD_SCENARIOS):

    shared_ip = random.choice(ips)

    accounts = random.sample(
        users,
        random.randint(4, 7)
    )

    rows = random.sample(
        list(df.index),
        len(accounts)
    )

    for index, account in zip(
        rows,
        accounts
    ):

        df.loc[index, "payer_id"] = account

        df.loc[index, "ip_address"] = shared_ip

        df.loc[index, "is_fraud"] = 1

        df.loc[index, "fraud_pattern"] = "shared_ip"


# --------------------------------------------------
# Fraud Pattern 6: Circular Transfers
# --------------------------------------------------

circular_accounts = random.sample(users, 3)

account_a = circular_accounts[0]
account_b = circular_accounts[1]
account_c = circular_accounts[2]

cycle_rows = random.sample(
    list(df.index),
    3
)

cycle_pairs = [
    (account_a, account_b),
    (account_b, account_c),
    (account_c, account_a)
]

for index, (payer, payee) in zip(
    cycle_rows,
    cycle_pairs
):

    df.loc[index, "payer_id"] = payer
    df.loc[index, "payee_id"] = payee
    df.loc[index, "amount"] = round(
        random.uniform(10000, 30000),
        2
    )

    df.loc[index, "is_fraud"] = 1
    df.loc[index, "fraud_pattern"] = "circular_transfer"


# --------------------------------------------------
# Save Dataset
# --------------------------------------------------

df = df.sort_values("timestamp")

output_path = "data/raw/transactions.csv"

df.to_csv(
    output_path,
    index=False
)

print("======================================")
print("Synthetic dataset created successfully")
print("======================================")

print(f"Total transactions : {len(df)}")
print(f"Fraud transactions : {df['is_fraud'].sum()}")
print(
    f"Normal transactions: "
    f"{(df['is_fraud'] == 0).sum()}"
)

print("\nFraud pattern distribution:")
print(
    df[df["is_fraud"] == 1]["fraud_pattern"]
    .value_counts()
)

print(f"\nSaved to: {output_path}")