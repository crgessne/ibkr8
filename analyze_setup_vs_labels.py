"""
Analyze the difference between is_long_setup and valid labels.
This will show us why master generates more trades than concurrent.
"""

import pandas as pd
import numpy as np

# Load the concurrent per-bar export (which has is_long_setup)
print("Loading concurrent per-bar features...")
df = pd.read_csv('data/concurrent_per_bar_features_concurrent.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year

print(f"Total bars: {len(df):,}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\n")

# Filter to 2024
df_2024 = df[df['year'] == 2024].copy()
print(f"2024 bars: {len(df_2024):,}\n")

# Count different scenarios
print("="*80)
print("SIGNAL GENERATION SCENARIOS (2024)")
print("="*80)

# 1. Bars with is_long_setup=True
setup_bars = df_2024['is_setup'].sum()
print(f"1. is_long_setup=True:                     {setup_bars:,} bars")

# 2. Bars with all features (regardless of setup)
has_features = df_2024['has_all_features'].sum()
print(f"2. has_all_features=True:                  {has_features:,} bars")

# 3. Bars with is_long_setup AND all features
setup_and_features = df_2024[df_2024['is_setup'] & df_2024['has_all_features']]
print(f"3. is_long_setup AND has_all_features:     {len(setup_and_features):,} bars")

# 4. Bars with prob >= 0.5 (regardless of setup)
high_prob = df_2024[df_2024['prob'] >= 0.5]
print(f"4. prob >= 0.5 (any bar):                  {len(high_prob):,} bars")

# 5. CONCURRENT APPROACH: is_long_setup AND prob >= 0.5
concurrent_signals = df_2024[df_2024['is_setup'] & (df_2024['prob'] >= 0.5)]
print(f"5. CONCURRENT: is_long_setup AND prob≥0.5: {len(concurrent_signals):,} bars")

# 6. MASTER APPROACH: Just prob >= 0.5 (no setup check)
master_signals = df_2024[df_2024['prob'] >= 0.5]
print(f"6. MASTER: prob >= 0.5 (no setup check):   {len(master_signals):,} bars")

print(f"\n{'='*80}")
print("GAP ANALYSIS")
print("="*80)

gap = len(master_signals) - len(concurrent_signals)
print(f"Difference (Master - Concurrent): {gap:,} bars")
print(f"Concurrent captures: {len(concurrent_signals)/len(master_signals)*100:.1f}% of master's signals")

# Check: are there high-prob bars that are NOT setups?
high_prob_not_setup = df_2024[(df_2024['prob'] >= 0.5) & ~df_2024['is_setup']]
print(f"\nHigh prob but NOT is_long_setup: {len(high_prob_not_setup):,} bars")
print(f"These are the MISSING signals in concurrent!\n")

# Show some examples
if len(high_prob_not_setup) > 0:
    print("="*80)
    print("SAMPLE: High-prob bars that are NOT long setups")
    print("="*80)
    sample = high_prob_not_setup.head(10)[['datetime', 'prob', 'is_setup', 
                                            'price_to_vwap_atr', 'vwap_width_atr',
                                            'crossed_vwap', 'vwap_helping']]
    print(sample.to_string())

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("Master pipeline: Generates signals on ANY bar with prob >= threshold")
print("Concurrent backtest: Only generates signals on is_long_setup=True bars")
print("\nThis is why concurrent generates fewer trades!")
print("To match master, concurrent should NOT check is_long_setup.")
