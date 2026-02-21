import pandas as pd
import numpy as np

for stop in [0.25, 0.5, 0.75, 1.0, 1.5]:
    fname = f'data/trades_y2024_stop{stop}_thresh0.50_20260211_142854.csv'
    try:
        df = pd.read_csv(fname)
    except FileNotFoundError:
        continue
    
    wr = (df['outcome'] == 'WIN').mean()
    wins = df[df['outcome'] == 'WIN']
    losses = df[df['outcome'] == 'LOSS']
    
    avg_win = wins['gross_pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['gross_pnl'].mean() if len(losses) > 0 else 0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    be_wr = 1 / (1 + payoff) if payoff > 0 else 1.0
    
    print(f"\n=== Stop={stop} ATR | Trades={len(df)} ===")
    print(f"  WR={wr*100:.1f}%  |  Avg Win=${avg_win:.0f}  |  Avg Loss=${avg_loss:.0f}  |  Payoff={payoff:.3f}")
    print(f"  Breakeven WR={be_wr*100:.1f}%  |  Gap={wr*100 - be_wr*100:+.1f}pp")
    print(f"  R:R stats: mean={df['rr'].mean():.2f}, median={df['rr'].median():.2f}, min={df['rr'].min():.2f}, max={df['rr'].max():.2f}")
    print(f"  Net P&L: ${df['net_pnl'].sum():,.0f}")
    
    # By R:R bucket
    for lo, hi in [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 99)]:
        bucket = df[(df['rr'] >= lo) & (df['rr'] < hi)]
        if len(bucket) < 5:
            continue
        bwr = (bucket['outcome'] == 'WIN').mean()
        total = bucket['net_pnl'].sum()
        bwins = bucket[bucket['outcome'] == 'WIN']
        blosses = bucket[bucket['outcome'] == 'LOSS']
        b_avg_win = bwins['gross_pnl'].mean() if len(bwins) > 0 else 0
        b_avg_loss = blosses['gross_pnl'].mean() if len(blosses) > 0 else 0
        b_payoff = abs(b_avg_win / b_avg_loss) if b_avg_loss != 0 else 0
        b_be = 1 / (1 + b_payoff) if b_payoff > 0 else 1.0
        print(f"    RR[{lo:.1f}-{hi:.1f}): n={len(bucket):>5}  WR={bwr*100:.1f}%  BE={b_be*100:.1f}%  gap={bwr*100-b_be*100:+.1f}pp  net=${total:>10,.0f}  payoff={b_payoff:.2f}")
