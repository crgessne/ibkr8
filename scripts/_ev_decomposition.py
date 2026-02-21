"""Quick EV decomposition and min-RR filter analysis."""
import pandas as pd
import numpy as np
import glob

for stop in [0.25, 0.5, 1.0, 1.5]:
    files = sorted(glob.glob(f'data/trades_realized_y2024_stop{stop}_seltop_5000_kregressor_*.csv'))
    if not files:
        continue
    t = pd.read_csv(files[-1])
    t['rr'] = t['per_trade_rr']
    
    print(f'\n=== STOP {stop} === ({len(t)} trades, net=${t["net_pnl"].sum():,.0f})')
    
    vwap = t[t['exit_reason'] == 'vwap']
    stop_t = t[t['exit_reason'] == 'stop']
    
    if len(vwap) > 0 and len(stop_t) > 0:
        wr = len(vwap) / len(t)
        avg_win = vwap['net_pnl'].mean()
        avg_loss = abs(stop_t['net_pnl'].mean())
        ev_per_trade = wr * avg_win - (1 - wr) * avg_loss
        print(f'  WR={wr:.1%}, avg_win=${avg_win:.0f}, avg_loss=${avg_loss:.0f}')
        print(f'  EV/trade = {wr:.3f} * {avg_win:.0f} - {1-wr:.3f} * {avg_loss:.0f} = ${ev_per_trade:.2f}')
        print(f'  Breakeven WR needed: {avg_loss/(avg_win+avg_loss):.1%}')
    
    # What if we only took trades with RR >= X?
    print(f'  --- Min RR filters ---')
    for min_rr in [0.5, 1.0, 1.5, 2.0, 3.0]:
        sub = t[t['rr'] >= min_rr]
        if len(sub) >= 20:
            wr_sub = (sub['net_pnl'] > 0).mean()
            v = sub[sub['exit_reason'] == 'vwap']
            s = sub[sub['exit_reason'] == 'stop']
            avg_w = v['net_pnl'].mean() if len(v) > 0 else 0
            avg_l = abs(s['net_pnl'].mean()) if len(s) > 0 else 0
            print(f'    RR>={min_rr}: n={len(sub):4d}, WR={wr_sub:.1%}, '
                  f'net=${sub["net_pnl"].sum():,.0f}, avg=${sub["net_pnl"].mean():.1f}, '
                  f'avg_win=${avg_w:.0f}, avg_loss=${avg_l:.0f}')

# Also check: what's the base rate (no model) for each stop?
print('\n\n=== BASE RATE ANALYSIS (ALL bars, no model) ===')
print('The key question: does the RF model add ANY edge over random?')

# Compare to the classifier trades
for stop in [0.5, 1.0]:
    # Classifier at threshold 0.0 (all bars)
    files_cls = sorted(glob.glob(f'data/trades_y2024_stop{stop}_thresh0.00_*.csv'))
    files_reg = sorted(glob.glob(f'data/trades_realized_y2024_stop{stop}_seltop_5000_kregressor_*.csv'))
    
    if files_cls and files_reg:
        t_cls = pd.read_csv(files_cls[-1])
        t_reg = pd.read_csv(files_reg[-1])
        print(f'\n  Stop {stop}:')
        print(f'    ALL bars (cls thresh=0.0): {len(t_cls)} trades, net=${t_cls["net_pnl"].sum():,.0f}, '
              f'avg=${t_cls["net_pnl"].mean():.2f}')
        print(f'    Regressor top 5000:        {len(t_reg)} trades, net=${t_reg["net_pnl"].sum():,.0f}, '
              f'avg=${t_reg["net_pnl"].mean():.2f}')
        
        # Is regressor better per-trade?
        if abs(t_cls["net_pnl"].mean()) > 0:
            improvement = (t_reg["net_pnl"].mean() - t_cls["net_pnl"].mean())
            print(f'    Regressor improvement per trade: ${improvement:.2f}')

print('\n\n=== CRITICAL: ONE-TRADE-PER-DAY ANALYSIS ===')
print('What if we only take the BEST trade per day (highest score)?')

for stop in [0.5, 1.0]:
    files = sorted(glob.glob(f'data/trades_realized_y2024_stop{stop}_seltop_5000_kregressor_*.csv'))
    if not files:
        continue
    t = pd.read_csv(files[-1])
    t['entry_dt'] = pd.to_datetime(t['entry_datetime'])
    t['entry_date'] = t['entry_dt'].dt.date
    
    # Just take the first trade of each day (earliest, since they're already flat-to-flat sorted)
    first_per_day = t.groupby('entry_date').first().reset_index()
    
    wr = (first_per_day['net_pnl'] > 0).mean()
    net = first_per_day['net_pnl'].sum()
    print(f'\n  Stop {stop}, first trade/day: n={len(first_per_day)}, '
          f'WR={wr:.1%}, net=${net:,.0f}, avg=${first_per_day["net_pnl"].mean():.1f}')
