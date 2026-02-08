"""Quick test of updated defaults."""
import sys
import os

print("Starting...", flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
print("Imports done", flush=True)
from indicators import calc_all_indicators, identify_reversal_setups, calc_theo_targets
from outcome_sim import simulate_all_setups

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
df = pd.read_csv(os.path.join(data_dir, 'tsla_5min_2025_01.csv'), parse_dates=['time'], index_col='time')

df_ind = calc_all_indicators(df)
df_setups = identify_reversal_setups(df_ind)
df_targets = calc_theo_targets(df_setups)  # Uses new 1.5 ATR default

trades = simulate_all_setups(df_targets)
total = len(trades)
winners = trades['is_winner'].sum()
win_pnl = trades[trades['is_winner']]['pnl_per_share'].sum()
loss_pnl = abs(trades[~trades['is_winner']]['pnl_per_share'].sum())

print("=" * 60)
print("UPDATED DEFAULTS (1.5 ATR Stop)")
print("=" * 60)
print(f"Total Trades: {total}")
print(f"Win Rate: {winners/total*100:.1f}%")
print(f"Profit Factor: {win_pnl/loss_pnl:.2f}")
print(f"Avg P&L: ${trades['pnl_per_share'].mean():.2f}")

long_t = trades[trades['direction'] == 'LONG']
short_t = trades[trades['direction'] == 'SHORT']
print(f"Long: {len(long_t)} trades, {long_t['is_winner'].mean()*100:.1f}% win rate")
print(f"Short: {len(short_t)} trades, {short_t['is_winner'].mean()*100:.1f}% win rate")
