import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load dataset ===
df = pd.read_csv("JSTdatasetR6.csv")
df = df.sort_values(['country', 'year'])

# === Compute bond excess returns ===
df['bond_excess_return'] = df['bond_tr'] - df['bill_rate']

# === Compute FX return ===
df['xrusd'] = df.groupby('country')['xrusd'].ffill()
df['fx_return'] = df.groupby('country')['xrusd'].transform(lambda x: x.shift(1) / x)

# === Compute USD-adjusted excess return ===
df['bond_excess_return_usd'] = df['bond_excess_return'] * df['fx_return']

# === Drop rows with any missing required values ===
required_cols = ['bond_excess_return', 'fx_return', 'bond_excess_return_usd']
df = df.dropna(subset=required_cols)

# === Compute and print summary statistics ===
def print_stats(series, label):
    print(f"\n=== Summary Statistics: {label} ===")
    print(f"Mean:        {series.mean():.4f}")
    print(f"Median:      {series.median():.4f}")
    print(f"Std Dev:     {series.std():.4f}")
    print(f"Skewness:    {series.skew():.4f}")
    print(f"Kurtosis:    {series.kurtosis():.4f}")
    print(f"Min:         {series.min():.4f}")
    print(f"Max:         {series.max():.4f}")
    print(f"Observations:{len(series)}")

print_stats(df['bond_excess_return'], "Local Currency")
print_stats(df['bond_excess_return_usd'], "USD-Adjusted")

# === Plot histograms ===
plt.figure(figsize=(12, 5))

# Local currency returns
plt.subplot(1, 2, 1)
plt.hist(df['bond_excess_return'], bins=50, color='skyblue', edgecolor='black')
plt.title("Histogram of Bond Excess Returns (Local Currency)")
plt.xlabel("Excess Return")
plt.ylabel("Frequency")
plt.grid(True)

# USD-adjusted returns
plt.subplot(1, 2, 2)
plt.hist(df['bond_excess_return_usd'], bins=50, color='salmon', edgecolor='black')
plt.title("Histogram of Bond Excess Returns (USD-Adjusted)")
plt.xlabel("Excess Return (USD)")
plt.ylabel("Frequency")
plt.grid(True)

plt.tight_layout()
plt.show()
