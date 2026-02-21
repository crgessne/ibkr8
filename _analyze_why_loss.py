"""Quick analysis: why does 66% WR still lose money?"""
import pandas as pd
import numpy as np

# Load the stop=1.5 RF>=0.50 trade log (latest run with per-trade fix)
df = pd.read_csv("data/trades_y2024_stop1.5_thresh0.50_20260211_133724.csv")
print(f"Total trades: {len(df):,}")
print(f"Wins:   {(df.outcome=='WIN').sum():,}  ({(df.outcome=='WIN').mean()*100:.1f}%)")
print(f"Losses: {(df.outcome=='LOSS').sum():,}  ({(df.outcome=='LOSS').mean()*100:.1f}%)")
print()

wins = df[df.outcome == "WIN"]
losses = df[df.outcome == "LOSS"]

print("=== WINS ===")
print(f"  Avg reward_dollars (gross gain):  ${wins.reward_dollars.mean():,.2f}")
print(f"  Median reward_dollars:            ${wins.reward_dollars.median():,.2f}")
print(f"  Total gross win P&L:              ${wins.gross_pnl.sum():,.0f}")
print()

print("=== LOSSES ===")
print(f"  Avg risk_dollars (gross loss):    ${losses.risk_dollars.mean():,.2f}")
print(f"  Median risk_dollars:              ${losses.risk_dollars.median():,.2f}")
print(f"  Total gross loss P&L:             ${losses.gross_pnl.sum():,.0f}")
print()

avg_win_gross = wins.gross_pnl.mean()
avg_loss_gross = abs(losses.gross_pnl.mean())

print("=== KEY RATIO ===")
print(f"  Avg gross WIN:   ${avg_win_gross:,.2f}")
print(f"  Avg gross LOSS:  ${avg_loss_gross:,.2f}")
print(f"  Actual payoff ratio (avg_win / avg_loss): {avg_win_gross / avg_loss_gross:.4f}")
print()

print("=== R:R DISTRIBUTION (per-trade vwap_dist / stop_atr) ===")
print(f"  Mean R:R:   {df.rr.mean():.3f}")
print(f"  Median R:R: {df.rr.median():.3f}")
print(f"  P5  R:R:    {df.rr.quantile(0.05):.3f}")
print(f"  P10 R:R:    {df.rr.quantile(0.10):.3f}")
print(f"  P25 R:R:    {df.rr.quantile(0.25):.3f}")
print(f"  P50 R:R:    {df.rr.quantile(0.50):.3f}")
print(f"  P75 R:R:    {df.rr.quantile(0.75):.3f}")
print(f"  P90 R:R:    {df.rr.quantile(0.90):.3f}")
print(f"  P95 R:R:    {df.rr.quantile(0.95):.3f}")
print()

# Breakdown by R:R bucket
buckets = [(0, 0.1, "< 0.10"), (0.1, 0.25, "0.10-0.25"), (0.25, 0.5, "0.25-0.50"),
           (0.5, 1.0, "0.50-1.00"), (1.0, 2.0, "1.00-2.00"), (2.0, 999, ">= 2.00")]

print("=== P&L BY R:R BUCKET ===")
print(f"  {'Bucket':<12} {'Count':>7} {'%Trades':>8} {'WR':>6} {'AvgWin':>10} {'AvgLoss':>10} {'NetPnL':>12}")
for lo, hi, label in buckets:
    mask = (df.rr >= lo) & (df.rr < hi)
    sub = df[mask]
    if len(sub) == 0:
        continue
    wr = (sub.outcome == "WIN").mean()
    sw = sub[sub.outcome == "WIN"]
    sl = sub[sub.outcome == "LOSS"]
    aw = sw.gross_pnl.mean() if len(sw) > 0 else 0
    al = abs(sl.gross_pnl.mean()) if len(sl) > 0 else 0
    print(f"  {label:<12} {len(sub):>7,} {len(sub)/len(df)*100:>7.1f}% {wr*100:>5.1f}% ${aw:>8,.0f} ${al:>8,.0f} ${sub.net_pnl.sum():>11,.0f}")

print()

# Breakeven analysis
be_wr = avg_loss_gross / (avg_win_gross + avg_loss_gross)
actual_wr = (df.outcome == "WIN").mean()
print("=== BREAKEVEN ANALYSIS ===")
print(f"  Breakeven WR (given actual avg win/loss sizes): {be_wr*100:.1f}%")
print(f"  Actual WR:                                      {actual_wr*100:.1f}%")
print(f"  Gap (actual - breakeven):                       {(actual_wr - be_wr)*100:+.1f}pp")
print()

print("=== THE ANSWER ===")
print(f"  The avg WIN is only ${avg_win_gross:,.2f} but avg LOSS is ${avg_loss_gross:,.2f}")
print(f"  That's a payoff ratio of {avg_win_gross/avg_loss_gross:.3f}:1")
print(f"  To break even at this payoff ratio you need {be_wr*100:.1f}% WR")
print(f"  Your 66.4% WR is {'above' if actual_wr > be_wr else 'BELOW'} breakeven")
print()

# Also check costs impact
costs_total = df.costs.sum()
gross_total = df.gross_pnl.sum()
print(f"  Total gross P&L (before costs): ${gross_total:,.0f}")
print(f"  Total costs:                    ${costs_total:,.0f}")
print(f"  Total net P&L:                  ${gross_total - costs_total:,.0f}")

# Same analysis for stop=1.0 (best performer)
print("\n" + "="*80)
print("SAME ANALYSIS FOR STOP=1.0 ATR (best net P&L)")
print("="*80)
df2 = pd.read_csv("data/trades_y2024_stop1.0_thresh0.50_20260211_133724.csv")
wins2 = df2[df2.outcome == "WIN"]
losses2 = df2[df2.outcome == "LOSS"]
aw2 = wins2.gross_pnl.mean()
al2 = abs(losses2.gross_pnl.mean())
wr2 = (df2.outcome == "WIN").mean()
be2 = al2 / (aw2 + al2)
print(f"  Trades: {len(df2):,}, WR: {wr2*100:.1f}%")
print(f"  Avg WIN: ${aw2:,.2f}, Avg LOSS: ${al2:,.2f}")
print(f"  Payoff ratio: {aw2/al2:.3f}:1")
print(f"  Breakeven WR: {be2*100:.1f}%, Gap: {(wr2-be2)*100:+.1f}pp")
print(f"  Gross P&L: ${df2.gross_pnl.sum():,.0f}, Costs: ${df2.costs.sum():,.0f}, Net: ${df2.net_pnl.sum():,.0f}")
print()
print("  R:R distribution:")
print(f"    Median R:R: {df2.rr.median():.3f}")
print(f"    P25 R:R:    {df2.rr.quantile(0.25):.3f}")
print(f"    P75 R:R:    {df2.rr.quantile(0.75):.3f}")

print("\n=== P&L BY R:R BUCKET (stop=1.0) ===")
print(f"  {'Bucket':<12} {'Count':>7} {'%Trades':>8} {'WR':>6} {'AvgWin':>10} {'AvgLoss':>10} {'NetPnL':>12}")
for lo, hi, label in buckets:
    mask = (df2.rr >= lo) & (df2.rr < hi)
    sub = df2[mask]
    if len(sub) == 0:
        continue
    wr = (sub.outcome == "WIN").mean()
    sw = sub[sub.outcome == "WIN"]
    sl = sub[sub.outcome == "LOSS"]
    aw = sw.gross_pnl.mean() if len(sw) > 0 else 0
    al = abs(sl.gross_pnl.mean()) if len(sl) > 0 else 0
    print(f"  {label:<12} {len(sub):>7,} {len(sub)/len(df2)*100:>7.1f}% {wr*100:>5.1f}% ${aw:>8,.0f} ${al:>8,.0f} ${sub.net_pnl.sum():>11,.0f}")
