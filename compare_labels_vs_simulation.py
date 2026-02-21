"""
Compare label-based win rate (master) vs simulation-based win rate (concurrent).
Load the actual labels and compare to our simulation results on the SAME bars.
"""

import pandas as pd
import numpy as np

# Load data with labels
print("Loading data with labels...")
df = pd.read_csv('data/tsla_5min_10years.csv')
df['datetime'] = pd.to_datetime(df['time'], utc=True) if 'time' in df.columns else pd.to_datetime(df['datetime'], utc=True)
df['year'] = df['datetime'].dt.year

# Filter to 2024
df_2024 = df[df['year'] == 2024].copy()
print(f"Total 2024 bars: {len(df_2024):,}")

# Load the concurrent per-bar data (has probabilities)
per_bar = pd.read_csv('data/concurrent_per_bar_features_concurrent.csv')
per_bar['datetime'] = pd.to_datetime(per_bar['datetime'])

# Merge to get labels for the bars we signaled on
merged = per_bar.merge(df_2024[['datetime', 'label_s1_25']], on='datetime', how='left')

# Filter to signals (prob >= 0.5)
signals = merged[merged['prob'] >= 0.5].copy()
print(f"\nSignal bars (prob >= 0.5): {len(signals):,}")

# Check how many have labels
has_label = signals['label_s1_25'].notna()
print(f"Signal bars with labels: {has_label.sum():,}")

if has_label.sum() > 0:
    # Calculate label-based win rate (MASTER'S APPROACH)
    label_wr = signals.loc[has_label, 'label_s1_25'].mean()
    print(f"\n{'='*80}")
    print("LABEL-BASED WIN RATE (Master's approach)")
    print(f"{'='*80}")
    print(f"Win rate from labels: {label_wr*100:.1f}%")
    print(f"This should match master's 64.6% if we're using same bars")
    
    # Load concurrent trades
    trades = pd.read_csv('data/concurrent_backtest_trades_concurrent.csv')
    sim_wins = (trades['reason'] == 'target').sum()
    sim_wr = sim_wins / len(trades)
    
    print(f"\n{'='*80}")
    print("SIMULATION-BASED WIN RATE (Concurrent's approach)")
    print(f"{'='*80}")
    print(f"Win rate from simulation: {sim_wr*100:.1f}%")
    print(f"Total trades: {len(trades):,}")
    
    print(f"\n{'='*80}")
    print("GAP ANALYSIS")
    print(f"{'='*80}")
    print(f"Label WR - Simulation WR: {(label_wr - sim_wr)*100:.1f} percentage points")
    
    if abs(label_wr - 0.646) < 0.01:
        print("\n✅ Labels match master's 64.6%!")
    else:
        print(f"\n❌ Labels DON'T match master (expected 64.6%, got {label_wr*100:.1f}%)")
        print("This means we're not using the same bars or labels!")

else:
    print("\n❌ ERROR: No labels found for signal bars!")
    print("The data file might not have labels generated yet.")
