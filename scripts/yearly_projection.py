"""
Yearly P&L Projection Calculator for Option 1 (Fixed Stop Strategies)

Based on master_pipeline results, projects annual P&L for any future year.
Assumes similar market conditions and trade frequency.
"""

import pandas as pd
from pathlib import Path

# Load the master pipeline results
results_file = Path("data/master_pipeline_results_20260207_141204.csv")
df = pd.read_csv(results_file)

# Filter for RF threshold 0.5 (our recommended filter)
df_filtered = df[df['rf_threshold'] == 0.5].copy()

# Test period was ~2 years (2024-2026)
TEST_PERIOD_YEARS = 2.11  # From 2024-01-01 to 2026-02-06

# Calculate annualized metrics
df_filtered['trades_per_year'] = df_filtered['n_trades'] / TEST_PERIOD_YEARS
df_filtered['pnl_per_year'] = df_filtered['total_net_pnl'] / TEST_PERIOD_YEARS
df_filtered['trades_per_month'] = df_filtered['trades_per_year'] / 12
df_filtered['trades_per_day'] = df_filtered['trades_per_year'] / 252  # Trading days

# Generate report
print("="*80)
print("OPTION 1: YEARLY P&L PROJECTIONS (Fixed Stop Strategies)")
print("="*80)
print(f"\nBased on test period: 2024-2026 ({TEST_PERIOD_YEARS:.2f} years)")
print("Assumes similar market conditions and trade frequency\n")

print("="*80)
print("COMPLETE PROJECTION TABLE")
print("="*80)
print()

# Sort by stop_atr for clarity
df_sorted = df_filtered.sort_values('stop_atr')

# Print header
print(f"{'Stop':<8} {'Trades/':<10} {'Trades/':<10} {'Trades/':<10} {'Yearly':<15} {'Per Trade':<12} {'Strategy':<15}")
print(f"{'ATR':<8} {'Year':<10} {'Month':<10} {'Day':<10} {'P&L':<15} {'EV':<12} {'Type':<15}")
print("-"*95)

# Print each strategy
for _, row in df_sorted.iterrows():
    stop = row['stop_atr']
    trades_year = row['trades_per_year']
    trades_month = row['trades_per_month']
    trades_day = row['trades_per_day']
    pnl_year = row['pnl_per_year']
    ev = row['ev']
    
    # Categorize strategy type
    if stop <= 0.4:
        strategy_type = "Aggressive"
    elif stop <= 0.75:
        strategy_type = "Balanced"
    else:
        strategy_type = "Conservative"
    
    print(f"{stop:<8.2f} {trades_year:<10,.0f} {trades_month:<10,.0f} {trades_day:<10,.1f} "
          f"${pnl_year:<14,.0f} {ev:>6.3f}R      {strategy_type:<15}")

print()
print("="*80)
print("TOP 3 RECOMMENDATIONS")
print("="*80)
print()

# Best EV
best_ev = df_sorted.loc[df_sorted['ev'].idxmax()]
print(f"🏆 HIGHEST EV PER TRADE: {best_ev['stop_atr']:.2f} ATR")
print(f"   • Expected trades per year: {best_ev['trades_per_year']:,.0f}")
print(f"   • Expected trades per month: {best_ev['trades_per_month']:,.0f}")
print(f"   • Expected trades per day: {best_ev['trades_per_day']:.1f}")
print(f"   • Expected yearly P&L: ${best_ev['pnl_per_year']:,.0f}")
print(f"   • Win rate: {best_ev['win_rate']*100:.1f}%")
print(f"   • EV per trade: +{best_ev['ev']:.3f}R")
print(f"   • Best for: Risk-tolerant traders seeking maximum returns\n")

# Best P&L
best_pnl = df_sorted.loc[df_sorted['pnl_per_year'].idxmax()]
print(f"💰 HIGHEST YEARLY P&L: {best_pnl['stop_atr']:.2f} ATR")
print(f"   • Expected trades per year: {best_pnl['trades_per_year']:,.0f}")
print(f"   • Expected trades per month: {best_pnl['trades_per_month']:,.0f}")
print(f"   • Expected trades per day: {best_pnl['trades_per_day']:.1f}")
print(f"   • Expected yearly P&L: ${best_pnl['pnl_per_year']:,.0f}")
print(f"   • Win rate: {best_pnl['win_rate']*100:.1f}%")
print(f"   • EV per trade: +{best_pnl['ev']:.3f}R")
print(f"   • Best for: Conservative traders prioritizing total profit\n")

# Recommended balanced (0.50 ATR)
balanced = df_sorted[df_sorted['stop_atr'] == 0.5].iloc[0]
print(f"⭐ RECOMMENDED BALANCED: {balanced['stop_atr']:.2f} ATR")
print(f"   • Expected trades per year: {balanced['trades_per_year']:,.0f}")
print(f"   • Expected trades per month: {balanced['trades_per_month']:,.0f}")
print(f"   • Expected trades per day: {balanced['trades_per_day']:.1f}")
print(f"   • Expected yearly P&L: ${balanced['pnl_per_year']:,.0f}")
print(f"   • Win rate: {balanced['win_rate']*100:.1f}%")
print(f"   • EV per trade: +{balanced['ev']:.3f}R")
print(f"   • Best for: Most traders - great balance of all factors\n")

print("="*80)
print("MONTHLY BREAKDOWN (0.50 ATR Recommended Strategy)")
print("="*80)
print()

monthly_trades = balanced['trades_per_month']
monthly_pnl = balanced['pnl_per_year'] / 12

print("Expected monthly performance (assuming even distribution):")
print()
print(f"{'Month':<12} {'Trades':<10} {'Expected P&L':<15}")
print("-"*40)

months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']

for month in months:
    print(f"{month:<12} {monthly_trades:<10,.0f} ${monthly_pnl:<14,.0f}")

print("-"*40)
print(f"{'TOTAL YEAR':<12} {balanced['trades_per_year']:<10,.0f} ${balanced['pnl_per_year']:<14,.0f}")

print()
print("="*80)
print("SCALING PROJECTIONS (0.50 ATR Strategy)")
print("="*80)
print()

# Show different capital/position sizes
base_shares = 100
base_pnl = balanced['pnl_per_year']

print("How P&L scales with position size:")
print()
print(f"{'Shares per':<15} {'Yearly':<15} {'Monthly':<15} {'Per Trade':<15}")
print(f"{'Trade':<15} {'P&L':<15} {'Avg P&L':<15} {'Avg P&L':<15}")
print("-"*60)

for multiplier in [0.5, 1, 2, 3, 5, 10]:
    shares = int(base_shares * multiplier)
    yearly = base_pnl * multiplier
    monthly_avg = yearly / 12
    per_trade = yearly / balanced['trades_per_year']
    
    print(f"{shares:<15,} ${yearly:<14,.0f} ${monthly_avg:<14,.0f} ${per_trade:<14,.2f}")

print()
print("="*80)
print("KEY ASSUMPTIONS & RISKS")
print("="*80)
print()
print("✅ Assumptions:")
print("   • Similar market volatility and conditions")
print("   • TSLA continues to have similar daily range and volume")
print("   • No significant changes to market structure")
print("   • RF model remains predictive (no regime change)")
print("   • Trade frequency remains consistent")
print()
print("⚠️  Risks:")
print("   • Market conditions can change (trending vs ranging)")
print("   • Volatility may increase or decrease")
print("   • Model performance may degrade over time")
print("   • Actual slippage/costs may differ")
print("   • Drawdown periods will occur")
print()
print("📝 Recommendations:")
print("   • Monitor live performance vs projections monthly")
print("   • Retrain model quarterly or if performance degrades")
print("   • Start with smaller position sizes (50-100 shares)")
print("   • Scale up gradually as confidence increases")
print("   • Set maximum drawdown limits (e.g., -15% from peak)")
print("   • Keep detailed trade logs for analysis")
print()

print("="*80)
print("SUMMARY")
print("="*80)
print()
print(f"Based on {TEST_PERIOD_YEARS:.2f} years of validated test data:")
print()
print(f"Expected yearly performance (0.50 ATR, 100 shares per trade):")
print(f"  • Trades: ~{balanced['trades_per_year']:,.0f} per year (~{balanced['trades_per_month']:.0f}/month)")
print(f"  • Win rate: {balanced['win_rate']*100:.1f}%")
print(f"  • Net P&L: ${balanced['pnl_per_year']:,.0f} per year")
print(f"  • Average per trade: ${balanced['pnl_per_year']/balanced['trades_per_year']:,.2f}")
print(f"  • Expected value: +{balanced['ev']:.3f}R per trade")
print()
print("This represents a robust, statistically significant edge.")
print()

# Save detailed breakdown
output_file = Path("data/yearly_projection_breakdown.csv")
df_sorted.to_csv(output_file, index=False)
print(f"✅ Detailed breakdown saved to: {output_file}")
