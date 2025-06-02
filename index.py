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

df['bond_excess_return_usd'] = df['bond_excess_return'] * df['fx_return']

# === Drop rows with any missing required values ===
required_cols = ['bond_excess_return', 'fx_return', 'bond_excess_return_usd']
df = df.dropna(subset=required_cols)

# === Build cumulative return indices ===
def build_index_with_gaps(data, return_col, index_col):
    data[index_col] = 1.0  # start at 1
    for country, group in data.groupby('country'):
        idx = group.index
        returns = group[return_col].fillna(0)  # no return if missing
        cumulative = (1 + returns).cumprod()
        data.loc[idx, index_col] = cumulative
    return data

df = build_index_with_gaps(df, 'bond_excess_return', 'term_premium_index_local')
df = build_index_with_gaps(df, 'bond_excess_return_usd', 'term_premium_index_usd')

# === Plot for one example country ===
country = 'Switzerland'
plot_df = df[df['country'] == country].copy()

plt.figure(figsize=(10, 6))
plt.plot(plot_df['year'], np.log(plot_df['term_premium_index_local']), label='Local Currency', linestyle='-', marker='o', color='b')
plt.plot(plot_df['year'], np.log(plot_df['term_premium_index_usd']), label='USD-Adjusted', linestyle='-', marker='s', color='r')
plt.title(f"Term Premium Harvesting Strategy – {country}")
plt.xlabel("Year")
plt.ylabel("Log Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Save relevant data to CSV ===
output_cols = [
    'country', 'year',
    'bond_excess_return',
    'fx_return',
    'bond_excess_return_usd',
    'term_premium_index_local',
    'term_premium_index_usd'
]
df[output_cols].to_csv("term_premium_returns.csv", index=False)
print("Saved metrics to 'term_premium_returns.csv'")
