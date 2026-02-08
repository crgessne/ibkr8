"""Run indicator calculations - standalone script."""
import sys
import importlib.util
import pandas as pd

# Load indicators module
spec = importlib.util.spec_from_file_location('indicators', r'C:\Users\Administrator\ibkr8\src\indicators.py')
ind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ind)

# Load data
input_file = r'C:\Users\Administrator\ibkr8\data\tsla_5min_2025_01.csv'
output_file = r'C:\Users\Administrator\ibkr8\data\tsla_5min_2025_01_indicators.csv'

print('Loading data...')
df = pd.read_csv(input_file, parse_dates=['time'], index_col='time')

print(f'Loaded {len(df)} bars')
print(f'Date range: {df.index.min()} to {df.index.max()}')
print(f'Price range: ${df["close"].min():.2f} - ${df["close"].max():.2f}')

# Calculate indicators
print('\nCalculating indicators...')
df_ind = ind.calc_all_indicators(df)

# Identify setups
df_setups = ind.identify_reversal_setups(df_ind, max_rsi_long=35.0, min_rsi_short=65.0, min_rel_vol=1.0)

# Calculate targets (VWAP-based mean reversion)
df_final = ind.calc_theo_targets(df_setups, stop_atr_mult=1.0, fallback_target_atr=1.5)

# Stats
df_valid = df_final.dropna()
print(f'\nValid bars (after warmup): {len(df_valid)}')

print('\n' + '='*60)
print('INDICATOR SUMMARY')
print('='*60)

print(f'\nAvg Close:     ${df_valid["close"].mean():.2f}')
print(f'Avg ATR:       ${df_valid["atr"].mean():.2f} ({df_valid["atr"].mean()/df_valid["close"].mean()*100:.2f}%)')

print(f'\nMean RSI:      {df_valid["rsi"].mean():.1f}')
print(f'RSI < 30:      {(df_valid["rsi"] < 30).sum()} bars ({(df_valid["rsi"] < 30).mean()*100:.1f}%)')
print(f'RSI > 70:      {(df_valid["rsi"] > 70).sum()} bars ({(df_valid["rsi"] > 70).mean()*100:.1f}%)')

print(f'\nAvg VWAP dist: {df_valid["vwap_dist_pct"].mean():.3f}%')
print(f'Below VWAP:    {df_valid["price_below_vwap"].sum()} bars ({df_valid["price_below_vwap"].mean()*100:.1f}%)')
print(f'VWAP dist ATR: {df_valid["vwap_dist_atr"].mean():.2f} ATR')

print(f'\nAvg BB %:      {df_valid["bb_pct"].mean():.2f}')
print(f'Below lower BB: {(df_valid["bb_pct"] < 0).sum()} bars')
print(f'Above upper BB: {(df_valid["bb_pct"] > 1).sum()} bars')

print(f'\nAvg Rel Vol:   {df_valid["rel_vol"].mean():.2f}x')
print(f'High Vol (>2x): {(df_valid["rel_vol"] > 2).sum()} bars')

print('\n' + '='*60)
print('REVERSAL SETUPS')
print('='*60)

long_setups = df_final['long_setup'].sum()
short_setups = df_final['short_setup'].sum()
print(f'\nLong setups:   {long_setups} ({long_setups/len(df_final)*100:.2f}%)')
print(f'Short setups:  {short_setups} ({short_setups/len(df_final)*100:.2f}%)')

# Show examples
if long_setups > 0:
    print('\n--- Sample Long Setups ---')
    longs = df_final[df_final['long_setup']].head(5)
    for idx, row in longs.iterrows():
        print(f'  {idx}: Close=${row["close"]:.2f}, RSI={row["rsi"]:.1f}, '
              f'VWAP=${row["vwap"]:.2f}, Target=${row["long_target"]:.2f}, '
              f'Stop=${row["long_stop"]:.2f}, RR={row["long_rr"]:.2f}')

if short_setups > 0:
    print('\n--- Sample Short Setups ---')
    shorts = df_final[df_final['short_setup']].head(5)
    for idx, row in shorts.iterrows():
        print(f'  {idx}: Close=${row["close"]:.2f}, RSI={row["rsi"]:.1f}, '
              f'VWAP=${row["vwap"]:.2f}, Target=${row["short_target"]:.2f}, '
              f'Stop=${row["short_stop"]:.2f}, RR={row["short_rr"]:.2f}')

# R:R Analysis
print('\n' + '='*60)
print('THEORETICAL R:R ANALYSIS')
print('='*60)

if long_setups > 0:
    long_rrs = df_final.loc[df_final['long_setup'], 'long_rr']
    print(f'\nLong R:R Stats:')
    print(f'  Mean:   {long_rrs.mean():.2f}')
    print(f'  Median: {long_rrs.median():.2f}')
    print(f'  Min:    {long_rrs.min():.2f}')
    print(f'  Max:    {long_rrs.max():.2f}')
    print(f'  R:R > 1: {(long_rrs > 1).sum()} ({(long_rrs > 1).mean()*100:.1f}%)')

if short_setups > 0:
    short_rrs = df_final.loc[df_final['short_setup'], 'short_rr']
    print(f'\nShort R:R Stats:')
    print(f'  Mean:   {short_rrs.mean():.2f}')
    print(f'  Median: {short_rrs.median():.2f}')
    print(f'  Min:    {short_rrs.min():.2f}')
    print(f'  Max:    {short_rrs.max():.2f}')
    print(f'  R:R > 1: {(short_rrs > 1).sum()} ({(short_rrs > 1).mean()*100:.1f}%)')

# Save
print('\n' + '='*60)
print(f'Saving to {output_file}...')
df_final.to_csv(output_file)
print(f'Done! Saved {len(df_final)} rows with {len(df_final.columns)} columns')

print('\nColumns in output:')
cols = df_final.columns.tolist()
for i in range(0, len(cols), 5):
    print('  ', cols[i:i+5])
