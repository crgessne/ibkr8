"""Quick EV decomposition and min-RR filter analysis."""
import pandas as pd
import numpy as np
import glob
import sys

def main():
    for stop in [0.25, 0.5, 1.0, 1.5]:
        files = sorted(glob.glob('data/trades_realized_y2024_stop{}_seltop_5000_kregressor_*.csv'.format(stop)))
        if not files:
            continue
        t = pd.read_csv(files[-1])
        t['rr'] = t['per_trade_rr']
        
        total_net = t['net_pnl'].sum()
        print('\n=== STOP {} === ({} trades, net=${:,.0f})'.format(stop, len(t), total_net))
        
        vwap = t[t['exit_reason'] == 'vwap']
        stop_t = t[t['exit_reason'] == 'stop']
        
        if len(vwap) > 0 and len(stop_t) > 0:
            wr = len(vwap) / len(t)
            avg_win = vwap['net_pnl'].mean()
            avg_loss = abs(stop_t['net_pnl'].mean())
            ev_per_trade = wr * avg_win - (1 - wr) * avg_loss
            print('  WR={:.1%}, avg_win=${:.0f}, avg_loss=${:.0f}'.format(wr, avg_win, avg_loss))
            print('  EV/trade = {:.3f} * {:.0f} - {:.3f} * {:.0f} = ${:.2f}'.format(
                wr, avg_win, 1-wr, avg_loss, ev_per_trade))
            be_wr = avg_loss / (avg_win + avg_loss)
            print('  Breakeven WR needed: {:.1%}'.format(be_wr))
        
        # What if we only took trades with RR >= X?
        print('  --- Min RR filters ---')
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
                print('    RR>={}: n={:4d}, WR={:.1%}, net=${:,.0f}, avg=${:.1f}, avg_win=${:.0f}, avg_loss=${:.0f}'.format(
                    min_rr, len(sub), wr_sub, sub_net, sub_avg, avg_w, avg_l))
    
    # Compare regressor vs all-bars classifier 
    print('\n\n=== BASE RATE COMPARISON ===')
    for stop in [0.5, 1.0]:
        files_cls = sorted(glob.glob('data/trades_y2024_stop{}_thresh0.00_*.csv'.format(stop)))
        files_reg = sorted(glob.glob('data/trades_realized_y2024_stop{}_seltop_5000_kregressor_*.csv'.format(stop)))
        
        if files_cls and files_reg:
            t_cls = pd.read_csv(files_cls[-1])
            t_reg = pd.read_csv(files_reg[-1])
            print('\n  Stop {}:'.format(stop))
            print('    ALL bars (cls thresh=0.0): {} trades, net=${:,.0f}, avg=${:.2f}'.format(
                len(t_cls), t_cls['net_pnl'].sum(), t_cls['net_pnl'].mean()))
            print('    Regressor top 5000:        {} trades, net=${:,.0f}, avg=${:.2f}'.format(
                len(t_reg), t_reg['net_pnl'].sum(), t_reg['net_pnl'].mean()))

    # One trade per day analysis
    print('\n\n=== ONE-TRADE-PER-DAY (first trade only) ===')
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
        print('  Stop {}, first trade/day: n={}, WR={:.1%}, net=${:,.0f}, avg=${:.1f}'.format(
            stop, len(first_per_day), wr, net, avg))
        
        # Best trade per day (highest gross pnl)
        best_idx = t.groupby('entry_date')['net_pnl'].idxmax()
        best_per_day = t.loc[best_idx]
        wr_best = (best_per_day['net_pnl'] > 0).mean()
        net_best = best_per_day['net_pnl'].sum()
        print('  Stop {}, BEST trade/day:  n={}, WR={:.1%}, net=${:,.0f}, avg=${:.1f}'.format(
            stop, len(best_per_day), wr_best, net_best, best_per_day['net_pnl'].mean()))

    # The MOST IMPORTANT question: what does the RANDOM baseline look like?
    print('\n\n=== RANDOM BASELINE: What if we entered RANDOMLY? ===')
    for stop in [0.5, 1.0]:
        files_all = sorted(glob.glob('data/trades_y2024_stop{}_thresh0.00_*.csv'.format(stop)))
        if not files_all:
            continue
        t_all = pd.read_csv(files_all[-1])
        
        # Random subset of same size as regressor
        files_reg = sorted(glob.glob('data/trades_realized_y2024_stop{}_seltop_5000_kregressor_*.csv'.format(stop)))
        if not files_reg:
            continue
        t_reg = pd.read_csv(files_reg[-1])
        n_reg = len(t_reg)
        
        np.random.seed(42)
        if len(t_all) > n_reg:
            idx = np.random.choice(len(t_all), size=n_reg, replace=False)
            t_rand = t_all.iloc[idx]
        else:
            t_rand = t_all
        
        print('  Stop {}: random {} trades: net=${:,.0f}, avg=${:.2f}'.format(
            stop, len(t_rand), t_rand['net_pnl'].sum(), t_rand['net_pnl'].mean()))
        print('  Stop {}: regressor {} trades: net=${:,.0f}, avg=${:.2f}'.format(
            stop, len(t_reg), t_reg['net_pnl'].sum(), t_reg['net_pnl'].mean()))


if __name__ == '__main__':
    main()
