# Raw VWAP reversion analysis - no stops/targets, just does price return to VWAP?
import pandas as pd
import numpy as np
import sys

# Load raw price data
print("Loading data...", flush=True)
df = pd.read_csv('data/tsla_5min_2years.csv')
df['time'] = pd.to_datetime(df['time'], utc=True)
df = df.set_index('time')
df['date'] = df.index.date

print("="*70, flush=True)
print("VWAP REVERSION RATE BY EXTENSION ZONE", flush=True)
print("="*70, flush=True)
print(f"Total 5-min bars: {len(df):,}", flush=True)
print(flush=True)

# Calculate VWAP per day
typical = (df['high'] + df['low'] + df['close']) / 3
pv = typical * df['volume']
vwap = pd.Series(index=df.index, dtype=float)
for date in df['date'].unique():
    mask = df['date'] == date
    cum_pv = pv[mask].cumsum()
    cum_vol = df.loc[mask, 'volume'].cumsum()
    vwap[mask] = cum_pv / cum_vol
df['vwap'] = vwap

# Calculate ATR
tr = pd.concat([
    df['high'] - df['low'],
    abs(df['high'] - df['close'].shift(1)),
    abs(df['low'] - df['close'].shift(1))
], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

df['vwap_dist_atr'] = (df['close'] - df['vwap']) / df['atr']
df['vwap_dist_abs'] = abs(df['vwap_dist_atr'])

print("Question: If price is X ATR from VWAP, does it return to VWAP by EOD?")
print()

zones = [
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 2.5),
    (2.5, 3.0),
    (3.0, 4.0),
    (4.0, 10.0),
]

print(f"{'Zone':<15} {'Bars':>10} {'Reverts':>10} {'No Revert':>10} {'Rate':>10}")
print("-"*60)

for low, high in zones:
    mask = (df['vwap_dist_abs'] >= low) & (df['vwap_dist_abs'] < high)
    zone_df = df[mask]
    
    yes_count = 0
    no_count = 0
    
    for idx in zone_df.index:
        date = df.loc[idx, 'date']
        direction = 'long' if df.loc[idx, 'vwap_dist_atr'] < 0 else 'short'
        target_vwap = df.loc[idx, 'vwap']
        
        future_mask = (df.index > idx) & (df['date'] == date)
        future_bars = df[future_mask]
        
        if len(future_bars) == 0:
            continue
        
        if direction == 'long':
            hit = (future_bars['high'] >= target_vwap).any()
        else:
            hit = (future_bars['low'] <= target_vwap).any()
        
        if hit:
            yes_count += 1
        else:
            no_count += 1
    
    total = yes_count + no_count
    if total > 0:
        rate = 100 * yes_count / total
        print(f"{low:.1f}-{high:.1f} ATR    {len(zone_df):>10,} {yes_count:>10,} {no_count:>10,} {rate:>9.1f}%")

print()
print("Note: This is RAW reversion rate - no stops, no entry criteria.")
print("      Just: did price touch VWAP again before end of day?")
