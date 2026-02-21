"""Check if VWAP computed on 200-bar rolling window matches full-dataset VWAP."""
import sys, pandas as pd, numpy as np
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
from master_pipeline import calculate_core_indicators

# Load real data
df = pd.read_csv('data/tsla_5min_10years.csv')
df['datetime'] = pd.to_datetime(df['time'], utc=True)
df['date'] = df['datetime'].dt.date
df_2024 = df[df['datetime'].dt.year == 2024].reset_index(drop=True)

# Full dataset indicators
print("Computing full-dataset indicators...")
full_ind = calculate_core_indicators(df_2024, verbose=False)

# Check multiple bars
print("\nComparing VWAP: Full dataset vs 200-bar window")
print("=" * 90)
print(f"{'Bar':>5}  {'DateTime':>25}  {'Full VWAP':>10}  {'Win VWAP':>10}  {'Diff':>8}  {'Full ATR':>8}  {'Win ATR':>8}")
print("-" * 90)

mismatches = 0
for test_bar in [250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]:
    if test_bar >= len(df_2024):
        break
    start = max(0, test_bar - 200)
    w = df_2024.iloc[start:test_bar].copy()
    w_ind = calculate_core_indicators(w, verbose=False)
    
    full_vwap = full_ind.iloc[test_bar - 1]['vwap']
    win_vwap = w_ind.iloc[-1]['vwap']
    full_atr = full_ind.iloc[test_bar - 1]['atr']
    win_atr = w_ind.iloc[-1]['atr']
    dt = df_2024.iloc[test_bar - 1]['datetime']
    vwap_diff = abs(full_vwap - win_vwap)
    atr_diff = abs(full_atr - win_atr)
    
    flag = " ***" if vwap_diff > 0.01 else ""
    print(f"{test_bar:>5}  {str(dt):>25}  {full_vwap:>10.2f}  {win_vwap:>10.2f}  {vwap_diff:>8.4f}  {full_atr:>8.4f}  {win_atr:>8.4f}{flag}")
    if vwap_diff > 0.01:
        mismatches += 1

# Also check a mid-day bar where window starts mid-day
print("\n\nChecking mid-day bars where window might start mid-day:")
print("=" * 90)
# Find bars that are mid-day (e.g., bar 40 of a 78-bar day)
# The 200-bar window for bar 250 starts at bar 50
# If bars per day ~ 78, bar 50 is well into day 1
# The window for bar 250 spans: bars 50-250, covering ~2.5 days
# Current day VWAP should use groupby('date') which only groups current day's bars

# Let's check: how many bars of the current day are in the window?
for test_bar in [250, 500, 1000]:
    if test_bar >= len(df_2024):
        break
    start = max(0, test_bar - 200)
    w = df_2024.iloc[start:test_bar].copy()
    current_date = w.iloc[-1]['date']
    bars_today_in_window = (w['date'] == current_date).sum()
    bars_today_full = (df_2024['date'] == current_date).sum()
    first_bar_today_full = df_2024[df_2024['date'] == current_date].index[0]
    first_bar_today_window = w[w['date'] == current_date].index[0] if bars_today_in_window > 0 else None
    
    print(f"\nBar {test_bar}: date={current_date}")
    print(f"  Window range: bars {start}-{test_bar-1}")
    print(f"  Bars today in window: {bars_today_in_window} / {bars_today_full} total")
    print(f"  First bar of day in full dataset: index {first_bar_today_full}")
    print(f"  First bar of day in window: index {first_bar_today_window}")
    print(f"  Window starts from market open? {first_bar_today_window == first_bar_today_full}")

print("\nDone.", flush=True)
