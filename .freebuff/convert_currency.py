import pandas as pd
import os

USD_TO_INR = 83

# Convert raw transactions
raw = 'data/raw/transactions.csv'
if os.path.exists(raw):
    df = pd.read_csv(raw)
    df['amount'] = (df['amount'] * USD_TO_INR).round(2)
    df.to_csv(raw, index=False)
    print(f"Converted raw: {len(df)} rows, min={df['amount'].min()}, max={df['amount'].max()}")

# Convert all processed CSVs with amount column
for f in os.listdir('data/processed'):
    if f.endswith('.csv'):
        path = f'data/processed/{f}'
        df = pd.read_csv(path)
        if 'amount' in df.columns:
            df['amount'] = (df['amount'] * USD_TO_INR).round(2)
            df.to_csv(path, index=False)
            print(f"Converted {f}: min={df['amount'].min()}, max={df['amount'].max()}")

print("Done - all amounts converted to INR (x83)")
