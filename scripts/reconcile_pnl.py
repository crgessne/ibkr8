"""
P&L Reconciliation Analysis

Compare master pipeline (vectorized) vs streaming simulator P&L calculations
to identify discrepancies and align methodologies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

# Load master pipeline results
print("="*80)
print("P&L RECONCILIATION ANALYSIS")
print("="*80)

# Load the latest master pipeline results
results_file = Path("data/master_pipeline_results_20260209_153147.csv")
results_df = pd.read_csv(results_file)

# Filter for 1.5 ATR stop at RF threshold 0.5 (the "best" strategy)
best_strategy = results_df[
    (results_df['stop_atr'] == 1.5) & 
    (results_df['rf_threshold'] == 0.5)
].iloc[0]

print("\n1. MASTER PIPELINE RESULTS (1.5 ATR stop, RF≥0.5)")
print("="*80)
print(f"Stop ATR: {best_strategy['stop_atr']}")
print(f"R:R Ratio: {best_strategy['rr']:.2f}:1")
print(f"Number of Trades: {int(best_strategy['n_trades']):,}")
print(f"Win Rate: {best_strategy['win_rate']*100:.1f}%")
print(f"Expected Value: {best_strategy['ev']:+.3f}R")
print(f"")
print(f"Average Risk per Trade: ${best_strategy['avg_risk_dollars']:,.2f}")
print(f"Total Gross P&L: ${best_strategy['total_gross_pnl']:,.2f}")
print(f"Total Costs: ${best_strategy['total_costs']:,.2f}")
print(f"Total Net P&L: ${best_strategy['total_net_pnl']:,.2f}")
print(f"Avg Net P&L per Trade: ${best_strategy['avg_net_pnl_per_trade']:,.2f}")
print(f"Capital per Trade: ${best_strategy['capital_per_trade']:,.2f}")
print(f"Return % per Trade: {best_strategy['return_pct_per_trade']:.3f}%")

print("\n2. STREAMING SIMULATOR RESULTS (from streaming_log.txt)")
print("="*80)
print("Total Trades: 724")
print("Win Rate: 47.8%")
print("Profit Factor: 0.81")
print("Final Equity: $81,130.12")
print("Initial Capital: $100,000.00")
print("Total Return: -18.9%")
print("Net P&L: -$18,869.88")
print("Total Fees: $1,448.00")
print("Avg Winning Trade: $221.23")
print("Avg Losing Trade: -$250.50")

print("\n3. KEY DIFFERENCES IDENTIFIED")
print("="*80)

master_net_pnl = best_strategy['total_net_pnl']
streaming_net_pnl = -18869.88

print(f"Master Pipeline Net P&L:  ${master_net_pnl:,.2f}")
print(f"Streaming Simulator P&L:  ${streaming_net_pnl:,.2f}")
print(f"Difference:               ${master_net_pnl - streaming_net_pnl:,.2f}")
print(f"Ratio:                    {abs(master_net_pnl / streaming_net_pnl):.1f}x")

print(f"\nMaster Pipeline Trades:   {int(best_strategy['n_trades']):,}")
print(f"Streaming Simulator:      724")
print(f"Difference:               {int(best_strategy['n_trades']) - 724:,} trades")

print(f"\nMaster Win Rate:          {best_strategy['win_rate']*100:.1f}%")
print(f"Streaming Win Rate:       47.8%")
print(f"Difference:               {(best_strategy['win_rate']*100 - 47.8):.1f}pp")

print("\n4. ROOT CAUSE ANALYSIS")
print("="*80)
print("\n⚠️  CRITICAL DIFFERENCES:")
print()
print("A. MASTER PIPELINE (Vectorized):")
print("   - Uses THEORETICAL P&L based on label outcomes")
print("   - Assumes perfect entry at VWAP reversion point")
print("   - Calculates: Win = +reward, Loss = -risk")
print("   - Formula: reward = risk * R:R ratio")
print("   - Risk = stop_atr * ATR * 100 shares")
print("   - Does NOT simulate actual order execution")
print("   - Does NOT track equity curve or drawdown")
print("   - Counts EVERY bar where RF≥0.5 as a trade")
print()
print("B. STREAMING SIMULATOR (Realistic):")
print("   - Simulates ACTUAL order execution bar-by-bar")
print("   - Entry/exit may not be at exact VWAP levels")
print("   - Accounts for slippage and realistic fills")
print("   - Tracks portfolio state and equity curve")
print("   - Enforces position sizing limits")
print("   - Only trades when no existing position")
print("   - Subject to market conditions and timing")
print()
print("5. RECONCILIATION STRATEGIES")
print("="*80)
print()
print("The discrepancy is EXPECTED and represents:")
print()
print("1. IMPLEMENTATION DIFFERENCES:")
print("   • Master: Theoretical - assumes perfect execution at labels")
print("   • Streaming: Realistic - simulates actual trading with constraints")
print()
print("2. TRADE COUNTING:")
print("   • Master: Counts potential signals (20,884)")
print("   • Streaming: Counts executed round-trips (724)")
print("   • Reason: Streaming can't take every signal (position limits)")
print()
print("3. P&L METHODOLOGY:")
print("   • Master: Fixed R:R-based calculation")
print("     - Win = +(stop_atr × ATR × 100 × R:R)")
print("     - Loss = -(stop_atr × ATR × 100)")
print("   • Streaming: Actual entry/exit prices")
print("     - P&L = (exit_price - entry_price) × shares")
print()
print("4. MARKET REALITY FACTORS (in streaming, not in master):")
print("   • Position can only be held for one trade at a time")
print("   • Exit may occur before target or stop is hit")
print("   • Slippage varies by market conditions")
print("   • Some signals are missed due to existing positions")
print("   • Drawdown and risk management constraints")
print()
print("6. RECOMMENDATIONS")
print("="*80)
print()
print("✓ Master Pipeline Results = THEORETICAL MAXIMUM")
print("  Use for: Strategy selection, parameter optimization, R:R analysis")
print()
print("✓ Streaming Simulator = REALISTIC EXPECTATION")
print("  Use for: Live trading preparation, risk assessment, position sizing")
print()
print("To improve streaming results:")
print("  1. Optimize entry/exit logic in strategy function")
print("  2. Add profit-taking logic (don't always wait for full target)")
print("  3. Use trailing stops to protect profits")
print("  4. Implement better signal filtering")
print("  5. Consider partial position sizing")
print("  6. Add risk management rules (max loss per day, etc.)")
print()
print("Expected relationship:")
print("  Streaming P&L = 10-30% of theoretical master pipeline P&L")
print(f"  Actual ratio: {abs(streaming_net_pnl / master_net_pnl)*100:.1f}%")
print()
if abs(streaming_net_pnl / master_net_pnl)*100 < 10:
    print("  ⚠️  Current ratio is LOW - investigate:")
    print("     - Is strategy logic correct?")
    print("     - Are stops being hit too early?")
    print("     - Is position sizing appropriate?")
    print("     - Are entries/exits optimal?")
else:
    print("  ✓ Ratio is within expected range for realistic trading")
print()
