import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# parameters
init_year = 1950
end_year = 2015
top_n = 3

# load dataset
df = pd.read_csv("JSTdatasetR6.csv")
df = df.sort_values(['country', 'year'])

# calculate bond excess returns
df['bond_excess_return'] = df['bond_tr'] - df['bill_rate']

# calculate FX adjusted returns
df['xrusd'] = df.groupby('country')['xrusd'].ffill()
df['fx_return'] = df.groupby('country')['xrusd'].transform(lambda x: x.shift(1) / x)
df['bond_excess_return_usd'] = df['bond_excess_return'] * df['fx_return']

# calculate term premium
df['term_premium'] = df['bond_rate'] - df['bill_rate']

# drop rows with missing values
df = df[['year', 'country', 'term_premium', 'bond_tr', 'bond_rate', 'bill_rate', 'fx_return', 'eq_tr']].dropna()

# select rows past initial year
df = df[df['year'] >= init_year]
df = df[df['year'] <= end_year]

# calculate returns of long-only US bond portfolio
benchmark = df[df['country'] == 'USA'][['year', 'bond_tr', 'eq_tr']]
benchmark['bond_tr'] = benchmark['bond_tr'].fillna(0)
benchmark['index'] = (1 + benchmark['bond_tr']).cumprod()

# calculate strategy returns
records = []
cumulative_value = 1.0  # Start with $1

grouped = df.groupby('year')

# allocate capital equally to the n countries with the highest term premium
for year, group in grouped:
    sample = group.dropna(subset=['term_premium', 'bond_tr', 'bill_rate', 'fx_return'])
    
    if 'USA' not in sample['country'].values or len(sample) < top_n:
        continue

    # select top N countries
    rank = sample.sort_values('term_premium', ascending=False)
    top_n_countries = rank.head(top_n)

    # compute strategy return
    weight = 1.0 / top_n
    strategy_pnl = 0
    selected_countries = []

    for _, row in top_n_countries.iterrows():
        position_pnl = (row['bond_tr'] - row['bill_rate']) * row['fx_return']
        strategy_pnl += weight * position_pnl
        selected_countries.append(row['country'])

    us_bill_rate = sample[sample['country'] == 'USA'].iloc[0]['bill_rate']
    strategy_return = strategy_pnl + us_bill_rate

    # update cumulative value
    cumulative_value *= (1 + strategy_return)

    # Record all data
    record = {
        'year': year,
        'strategy_return': strategy_return,
        'index': cumulative_value
    }

    for i, c in enumerate(selected_countries):
        record[f'country_{i+1}'] = c

    records.append(record)

# write strategy returns to csv
strategy = pd.DataFrame(records)

combined = pd.merge(strategy, benchmark[['year', 'bond_tr', 'index']], on='year', how='inner', suffixes=('', '_benchmark'))

combined = combined.rename(columns={
    'index': 'strategy_index',
    'bond_tr': 'benchmark_return',
    'index_benchmark': 'benchmark_index'
})

combined.to_csv('term_premium_strategy_returns.csv', index=False)
print("Saved combined strategy and benchmark metrics to 'term_premium_strategy_returns.csv'")


# plot strategy and benchmark returns
plt.figure(figsize=(10,6))
plt.plot(benchmark['year'], np.log(benchmark['index']), label='Benchmark')
plt.plot(strategy['year'], np.log(strategy['index']), label='Strategy')
plt.title('Cumulative Log Returns')
plt.xlabel('Year')
plt.ylabel('Log Cummulative Return')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(benchmark['year'], benchmark['bond_tr'], label='Benchmark')
plt.plot(strategy['year'], strategy['strategy_return'], label='Strategy')
plt.title('Annual Returns')
plt.xlabel('Year')
plt.ylabel('Return')
plt.grid()
plt.legend()
plt.show()

# print summary statistics for strategy and benchmark
avg_rf = df[df['country'] == 'USA']['bill_rate'].mean()

def summarize(log_returns, label):
    mean = log_returns.mean()
    std = log_returns.std()
    sharpe = (mean - avg_rf) / std
    print(f"\n{label}")
    print(f"  Mean Log Return:       {mean * 100:.2f}%")
    print(f"  Std Dev (Log Return):  {std * 100:.2f}%")
    print(f"  Sharpe Ratio:          {sharpe:.4f}")
    
benchmark['log_return'] = np.log(1 + benchmark['bond_tr'])
strategy['log_return'] = np.log(1 + strategy['strategy_return'])

summarize(strategy['log_return'], "Strategy (Top-N Term Premium)")
summarize(benchmark['log_return'], "Benchmark (US Bonds)")