import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf

# === Load and sort data ===
df = pd.read_csv('JSTdatasetR6.csv')
df = df.sort_values(['country', 'year']).copy()

# === Compute bond excess return (local currency) ===
df['bond_excess_return'] = df['bond_tr'] - df['bill_rate']
df['bond_excess_return'] = df['bond_excess_return'].where(df[['bond_tr', 'bill_rate']].notna().all(axis=1))

# === Term spread ===
df['term_spread'] = df['bond_rate'] - df['bill_rate']
df['term_spread'] = df['term_spread'].where(df[['bond_rate', 'bill_rate']].notna().all(axis=1))

# === FX adjustment ===
df['xrusd'] = df.groupby('country')['xrusd'].ffill()
df['fx_return'] = df.groupby('country')['xrusd'].transform(lambda x: x.shift(1) / x)
df['bond_excess_return_usd'] = df['bond_excess_return'] * df['fx_return']

# === Regressions ===
def run_regression(x, y, label):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    print(f"\n=== Regression: {label} ===")
    print(model.summary())
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data', alpha=0.6)
    plt.plot(x, model.predict(X), color='red', linewidth=2, label='OLS Fit')
    plt.xlabel('Term Spread (%)')
    plt.ylabel(label)
    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

reg1 = df[['term_spread', 'bond_excess_return']].dropna()
run_regression(reg1['term_spread'], reg1['bond_excess_return'],
               f'Term Spread vs Bond Excess Return (Local Currency)')

reg2 = df[['term_spread', 'bond_excess_return_usd']].dropna()
run_regression(reg2['term_spread'], reg2['bond_excess_return_usd'],
               f'Term Spread vs USD Bond Excess Return (USD Adjusted)')

# === PACF ===
global_spread = df['term_spread'].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
plot_pacf(global_spread, lags=10, method='ywm', zero=False, ax=ax)
ax.set_title('Partial Autocorrelation of Term Spread – All Countries')
ax.set_xlabel('Lag (Years)')
ax.set_ylabel('Partial Autocorrelation')
ax.grid(True)
plt.tight_layout()
plt.show()