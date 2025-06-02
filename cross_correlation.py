import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# === Load dataset ===
df = pd.read_csv("JSTdatasetR6.csv")
df = df.sort_values(['country', 'year'])

# === Compute bond excess returns ===
df['bond_excess_return'] = df['bond_tr'] - df['bill_rate']

# === Compute FX return ===
df['xrusd'] = df.groupby('country')['xrusd'].ffill()
df['fx_return'] = df.groupby('country')['xrusd'].transform(lambda x: x.shift(1) / x)

# === Compute USD-adjusted bond excess return ===
df['bond_excess_return_usd'] = df['bond_excess_return'] * df['fx_return']

# === Filter required columns and drop rows with NaNs ===
df = df[['country', 'year', 'bond_excess_return_usd']].dropna()

# === Pivot to wide format: years x countries ===
pivot_df = df.pivot(index='year', columns='country', values='bond_excess_return_usd')
pivot_df = pivot_df[pivot_df['USA'].notna()]

# === Rolling average pairwise correlation ===
window = 20
rolling_avg_corrs = []
rolling_years = []

years = pivot_df.index

for i in range(len(years) - window + 1):
    window_data = pivot_df.iloc[i:i + window]
    window_data = window_data.dropna(axis=1, how='any')
    if window_data.shape[1] < 2:
        continue

    corr_matrix = window_data.corr()
    n = corr_matrix.shape[0]
    avg_corr = (corr_matrix.values.sum() - n) / (n * (n - 1))
    rolling_avg_corrs.append(avg_corr)
    rolling_years.append(years[i + window - 1])

# === Plot rolling average pairwise correlation ===
plt.figure(figsize=(10, 6))
plt.plot(rolling_years, rolling_avg_corrs, marker='o')
plt.title("Rolling 20-Year Avg Pairwise Correlation of USD-Adjusted International Bond Excess Returns")
plt.xlabel("Year")
plt.ylabel("Average Correlation")
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.grid(True)
plt.tight_layout()
plt.show()

# === Save correlation matrix from 1990 to 2020 ===
corr_start = 1990
corr_end = pivot_df.index.max()

corr_data = pivot_df.loc[(pivot_df.index >= corr_start) & (pivot_df.index <= corr_end)]
corr_data = corr_data.dropna(axis=1, how='any')  # Only include countries with complete data over the period

corr_matrix = corr_data.corr()

# Display heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, vmin=0, vmax=1, cmap='coolwarm', annot=False)
plt.title(f"Correlation Matrix of USD-Adjusted International Bond Excess Returns (1990–{corr_end})")
plt.tight_layout()
plt.show()

# Save to CSV
corr_matrix.to_csv(f'bond_excess_return_correlation_1990_{corr_end}.csv')
print(f"\nSaved to 'bond_excess_return_usd_correlation_1990_{corr_end}.csv'")