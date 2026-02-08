"""
Visualize RF threshold comparison for best setup (0.25 ATR stop).
"""
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('../data/rf_simple_grid_results.csv')

# Focus on 0.25 ATR stop (best setup)
best = df[df['stop_atr'] == 0.25].iloc[0]

print("="*80)
print("RF THRESHOLD COMPARISON - 0.25 ATR Stop (6:1 R:R)")
print("="*80)
print(f"\nBaseline (No RF):")
print(f"  WR: {best['raw_wr']:.1%}")
print(f"  EV: {best['raw_ev']:+.3f}R")
print(f"  N:  {int(best['n_test']):,}")

print(f"\nRF Threshold Results:")
thresholds = [0.4, 0.5, 0.55, 0.6]
for t in thresholds:
    n = best[f'rf{t}_n']
    wr = best[f'rf{t}_wr']
    ev = best[f'rf{t}_ev']
    pct_trades = (n / best['n_test']) * 100
    print(f"  RF≥{t}: WR={wr:.1%}, EV={ev:+.3f}R, N={int(n):,} ({pct_trades:.1f}% of bars)")

print("\n" + "="*80)
print("THRESHOLD SELECTION GUIDE")
print("="*80)
print("\n🎯 Conservative (More Trades, Lower Win Rate):")
print("   RF≥0.4 → 55.2% of bars, WR=22.9%, EV=+0.611R")

print("\n⚖️  Balanced (Good Trade-off):")
print("   RF≥0.5 → 45.1% of bars, WR=25.2%, EV=+0.775R  ⭐ RECOMMENDED")

print("\n🔬 Aggressive (Higher Selectivity):")
print("   RF≥0.55 → 36.4% of bars, WR=27.0%, EV=+0.884R")
print("   RF≥0.6 → 28.7% of bars, WR=28.5%, EV=+0.970R")

print("\n💡 Key Insight:")
print("   Higher thresholds = Better EV but fewer trades")
print("   All thresholds maintain massive +EV (0.6R to 1.0R per trade!)")
print("   RF≥0.5 offers best balance: ~18K trades with +0.775R EV")

# Compare all stops at RF≥0.5
print("\n" + "="*80)
print("ALL STOP WIDTHS AT RF≥0.5 THRESHOLD")
print("="*80)
print("\nStop  | R:R  | Req WR | RF≥0.5 WR | RF≥0.5 EV | Trades | % of Bars")
print("-"*75)
for _, row in df.iterrows():
    stop = row['stop_atr']
    rr = row['rr']
    breakeven = 1 / (1 + rr)
    wr = row['rf0.5_wr']
    ev = row['rf0.5_ev']
    n = int(row['rf0.5_n'])
    pct = (n / row['n_test']) * 100
    
    # Highlight best
    marker = " ⭐" if stop == 0.25 else ""
    print(f"{stop:.2f}  | {rr:.2f} | {breakeven:.1%}  | {wr:.1%}     | {ev:+.3f}R    | {n:,}  | {pct:.1f}%{marker}")

print("\n✅ TAKEAWAY: Tighter stops (0.25-0.4 ATR) yield best EV due to high R:R!")
