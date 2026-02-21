"""Quick EV decomposition - writes output to file to avoid PowerShell issues."""
import pandas as pd
import numpy as np
import glob
import sys

OUT = open('_ev_decomp_out.txt', 'w')

def p(s=''):
    print(s, file=OUT, flush=True)
    print(s, file=sys.stderr, flush=True)

for stop in [0.25, 0.5, 1.0, 1.5]:
    files = sorted(glob.glob('data/trades_realized_y2024_stop{}_seltop_5000_kregressor_*.csv'.format(stop)))
    if not files:
        continue
    t = pd.read_csv(files[-1])
    t['rr'] = t['per_trade_rr']
    
    total_net = t['net_pnl'].sum()
    p('\n=== STOP {} === ({} trades, net=${:,.0f})'.format(stop, len(t), total_net))
    
    vwap = t[t['exit_reason'] == 'vwap']
    stop_t = t[t['exit_reason'] == 'stop']
    
    if len(vwap) > 0 and len(stop_t) > 0:
        wr = len(vwap) / len(t)
        avg_win = vwap['net_pnl'].mean()
        avg_loss = abs(stop_t['net_pnl'].mean())
        ev_per_trade = wr * avg_win - (1 - wr) * avg_loss
        p('  WR={:.1%}, avg_win=${:.0f}, avg_loss=${:.0f}'.format(wr, avg_win, avg_loss))
        p('  EV/trade = {:.3f} * {:.0f} - {:.3f} * {:.0f} = ${:.2f}'.format(
            wr, avg_win, 1-wr, avg_loss, ev_per_trade))
        be_wr = avg_loss / (avg_win + avg_loss)
        p('  Breakeven WR needed: {:.1%}'.format(be_wr))
    
    p('  --- Min RR filters ---')
    for min_rr in [0.5, 1.0, 1.5, 2.0, 3.0]:
        sub = t[t['rr'] >= min_rr]
        if len(sub) >= 20:
            wr_sub = (sub['net_pnl'] > 0).mean()
            v = sub[sub['exit_reason'] == 'vwap']
            s = sub[sub['exit_reason'] == 'stop']
            avg_w = v['net_pnl'].mean() if len(v) > 0 else 0
            avg_l = abs(s['net_pnl'].mean()) if len(s) > 0 else 0
            sub_net = sub['net_pnl'].sum()
            sub_avg = sub['net_pnl'].mean()
            p('    RR>={}: n={:4d}, WR={:.1%}, net=${:,.0f}, avg=${:.1f}, avg_win=${:.0f}, avg_loss=${:.0f}'.format(
                min_rr, len(sub), wr_sub, sub_net, sub_avg, avg_w, avg_l))

p('\n\n=== BASE RATE: ALL-BARS CLASSIFIER (thresh=0.0) vs REGRESSOR ===')
for stop in [0.5, 1.0]:
    files_cls = sorted(glob.glob('data/trades_y2024_stop{}_thresh0.00_*.csv'.format(stop)))
    files_reg = sorted(glob.glob('data/trades_realized_y2024_stop{}_seltop_5000_kregressor_*.csv'.format(stop)))
    
    if files_cls and files_reg:
        t_cls = pd.read_csv(files_cls[-1])
        t_reg = pd.read_csv(files_reg[-1])
        p('\n  Stop {}:'.format(stop))
        p('    ALL bars (cls thresh=0.0): {} trades, net=${:,.0f}, avg=${:.2f}'.format(
            len(t_cls), t_cls['net_pnl'].sum(), t_cls['net_pnl'].mean()))
        p('    Regressor top 5000:        {} trades, net=${:,.0f}, avg=${:.2f}'.format(
            len(t_reg), t_reg['net_pnl'].sum(), t_reg['net_pnl'].mean()))

p('\n\n=== ONE-TRADE-PER-DAY ===')
for stop in [0.5, 1.0, 1.5]:
    files = sorted(glob.glob('data/trades_realized_y2024_stop{}_seltop_5000_kregressor_*.csv'.format(stop)))
    if not files:
        continue
    t = pd.read_csv(files[-1])
    t['entry_dt'] = pd.to_datetime(t['entry_datetime'])
    t['entry_date'] = t['entry_dt'].dt.date
    
    first_per_day = t.groupby('entry_date').first().reset_index()
    wr = (first_per_day['net_pnl'] > 0).mean()
    net = first_per_day['net_pnl'].sum()
    avg = first_per_day['net_pnl'].mean()
    p('  Stop {}, first trade/day: n={}, WR={:.1%}, net=${:,.0f}, avg=${:.1f}'.format(
        stop, len(first_per_day), wr, net, avg))
    
    best_idx = t.groupby('entry_date')['net_pnl'].idxmax()
    best_per_day = t.loc[best_idx]
    wr_best = (best_per_day['net_pnl'] > 0).mean()
    net_best = best_per_day['net_pnl'].sum()
    p('  Stop {}, BEST trade/day:  n={}, WR={:.1%}, net=${:,.0f}, avg=${:.1f} (oracle)'.format(
        stop, len(best_per_day), wr_best, net_best, best_per_day['net_pnl'].mean()))

p('\n\n=== CRITICAL: WHAT IF SHORT SIDE ONLY (or LONG SIDE ONLY)? ===')
for stop in [0.5, 1.0]:
    files = sorted(glob.glob('data/trades_realized_y2024_stop{}_seltop_5000_kregressor_*.csv'.format(stop)))
    if not files:
        continue
    t = pd.read_csv(files[-1])
    for d, label in [(1, 'LONG'), (0, 'SHORT')]:
        sub = t[t['is_long'] == d]
        if len(sub) > 0:
            wr = (sub['net_pnl'] > 0).mean()
            vwap_pct = (sub['exit_reason'] == 'vwap').mean()
            p('  Stop {}, {}: n={}, WR={:.1%}, vwap_exit={:.1%}, net=${:,.0f}, avg=${:.1f}'.format(
                stop, label, len(sub), wr, vwap_pct, sub['net_pnl'].sum(), sub['net_pnl'].mean()))

p('\n\n=== CRITICAL: TIME-OF-DAY ANALYSIS ===')
for stop in [0.5]:
    files = sorted(glob.glob('data/trades_realized_y2024_stop{}_seltop_5000_kregressor_*.csv'.format(stop)))
    if not files:
        continue
    t = pd.read_csv(files[-1])
    t['entry_dt'] = pd.to_datetime(t['entry_datetime'])
    t['hour'] = t['entry_dt'].dt.hour
    
    p('  Stop {}:'.format(stop))
    for h in sorted(t['hour'].unique()):
        sub = t[t['hour'] == h]
        if len(sub) >= 10:
            wr = (sub['net_pnl'] > 0).mean()
            p('    Hour {:2d}: n={:4d}, WR={:.1%}, net=${:8,.0f}, avg=${:.1f}'.format(
                h, len(sub), wr, sub['net_pnl'].sum(), sub['net_pnl'].mean()))

OUT.close()
