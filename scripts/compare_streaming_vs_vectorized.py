"""
Compare streaming simulation vs vectorized backtest results

This script identifies differences between:
1. Streaming simulator (bar-by-bar with market orders)
2. Vectorized backtest (master_pipeline.py)
"""

import pandas as pd
import numpy as np

# Load streaming results
streaming_trades = pd.read_csv("data/streaming_sim_results.csv")
streaming_trades['entry_time'] = pd.to_datetime(streaming_trades['entry_time'])
streaming_trades['exit_time'] = pd.to_datetime(streaming_trades['exit_time'])

# Load master_pipeline results (this is aggregated, not per-trade)
mp_results = pd.read_csv("data/master_pipeline_results_20260209_095624.csv")
mp_15_05 = mp_results[(mp_results['stop_atr'] == 1.5) & (mp_results['rf_threshold'] == 0.5)]

print("="*80)
print("STREAMING vs VECTORIZED COMPARISON (2024, stop=1.5 ATR, RF>=0.5)")
print("="*80)

print("\n📊 AGGREGATE METRICS:")
print(f"\nStreaming Simulator:")
print(f"  Trades: {len(streaming_trades)}")
print(f"  Win Rate: {(streaming_trades['net_pnl'] > 0).mean()*100:.2f}%")
print(f"  Total P&L: ${streaming_trades['net_pnl'].sum():,.2f}")
print(f"  Avg Win: ${streaming_trades[streaming_trades['net_pnl']>0]['net_pnl'].mean():.2f}")
print(f"  Avg Loss: ${streaming_trades[streaming_trades['net_pnl']<=0]['net_pnl'].mean():.2f}")
print(f"  Total Fees: ${streaming_trades['fees'].sum():,.2f}")

print(f"\nVectorized Backtest (master_pipeline):")
print(f"  Trades: {mp_15_05['n_trades'].values[0]:,.0f}")
print(f"  Win Rate: {mp_15_05['win_rate'].values[0]*100:.2f}%")
print(f"  EV per trade: ${mp_15_05['ev'].values[0]:.2f}")
print(f"  Implied Total P&L: ${mp_15_05['ev'].values[0] * mp_15_05['n_trades'].values[0]:,.2f}")

# Key differences
print("\n🔍 KEY DIFFERENCES:")
trade_diff = len(streaming_trades) - mp_15_05['n_trades'].values[0]
wr_diff = ((streaming_trades['net_pnl'] > 0).mean() - mp_15_05['win_rate'].values[0]) * 100
pnl_diff = streaming_trades['net_pnl'].sum() - (mp_15_05['ev'].values[0] * mp_15_05['n_trades'].values[0])

print(f"  Trade count difference: {trade_diff:+,.0f} ({trade_diff/mp_15_05['n_trades'].values[0]*100:+.1f}%)")
print(f"  Win rate difference: {wr_diff:+.2f} percentage points")
print(f"  P&L difference: ${pnl_diff:+,.2f}")

# Analyze streaming trade characteristics
print("\n📈 STREAMING TRADE ANALYSIS:")
print(f"  Entry slippage (avg): ${(streaming_trades['entry_price'] - streaming_trades['entry_price']).mean():.4f}")  # placeholder
print(f"  P&L distribution:")
print(f"    Min: ${streaming_trades['net_pnl'].min():.2f}")
print(f"    25%: ${streaming_trades['net_pnl'].quantile(0.25):.2f}")
print(f"    50%: ${streaming_trades['net_pnl'].median():.2f}")
print(f"    75%: ${streaming_trades['net_pnl'].quantile(0.75):.2f}")
print(f"    Max: ${streaming_trades['net_pnl'].max():.2f}")

# Sample trades
print("\n📋 FIRST 5 STREAMING TRADES:")
print(streaming_trades[['entry_time', 'entry_price', 'exit_price', 'pnl', 'fees', 'net_pnl']].head().to_string())

print("\n📋 WORST 5 LOSSES:")
print(streaming_trades.nsmallest(5, 'net_pnl')[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'net_pnl', 'duration_bars']].to_string())

print("\n📋 BEST 5 WINS:")
print(streaming_trades.nlargest(5, 'net_pnl')[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'net_pnl', 'duration_bars']].to_string())

print("\n⚠️  LIKELY CAUSES OF DISCREPANCY:")
print("1. Streaming uses bar CLOSE for stop/target trigger → worse fills")
print("2. Streaming recalculates ATR each bar → stop/target drift")
print("3. Streaming uses MARKET orders with slippage")
print("4. Vectorized uses intrabar high/low → better fills")
print("5. Streaming indicators may differ slightly from vectorized pre-calculation")

print("\n💡 TO FIX:")
print("A. Use bar HIGH/LOW for intrabar stop/target detection")
print("B. Freeze ATR at entry (don't recalculate for open positions)")
print("C. Use exact stop/target prices (limit orders) instead of market")
print("D. Verify indicator alignment with separate comparison script")
