"""Comprehensive parameter sweep for 2-year TSLA data."""
import sys
sys.path.insert(0, r'C:\Users\Administrator\ibkr8\src')
import pandas as pd
from indicators import calc_all_indicators, calc_theo_targets
from outcome_sim import simulate_all_setups

out = open(r'C:\Users\Administrator\ibkr8\data\sweep_results.txt', 'w')
out.write("Loading data...\n")
out.flush()

df = pd.read_csv(r'C:\Users\Administrator\ibkr8\data\tsla_5min_2years.csv', parse_dates=['time'], index_col='time')
df_ind = calc_all_indicators(df)
out.write(f"Loaded {len(df)} bars\n\n")
out.flush()

out.write("="*100 + "\n")
out.write("COMPREHENSIVE PARAMETER SWEEP\n")
out.write("="*100 + "\n")
out.write(f"{'RSI_L':<6} {'RSI_S':<6} {'VWAP':<6} {'Stop':<6} {'N':<8} {'WR%':<8} {'PF':<8} {'AvgPnL':<10}\n")
out.write("-"*100 + "\n")
out.flush()

results = []
for max_rsi_long in [20, 25, 30, 35]:
    for min_rsi_short in [65, 70, 75, 80]:
        for min_vwap_dist in [0.0, 1.0, 1.5, 2.0]:
            for stop in [1.5, 2.0, 2.5]:
                df_s = df_ind.copy()
                
                if min_vwap_dist > 0:
                    df_s['long_setup'] = (
                        (df_ind['rsi'] < max_rsi_long) & 
                        (df_ind['rel_vol'] >= 1.0) & 
                        (df_ind['close'] < df_ind['bb_lower']) &
                        (df_ind['vwap_dist_atr'] < -min_vwap_dist)
                    )
                    df_s['short_setup'] = (
                        (df_ind['rsi'] > min_rsi_short) & 
                        (df_ind['rel_vol'] >= 1.0) & 
                        (df_ind['close'] > df_ind['bb_upper']) &
                        (df_ind['vwap_dist_atr'] > min_vwap_dist)
                    )
                else:
                    df_s['long_setup'] = (
                        (df_ind['rsi'] < max_rsi_long) & 
                        (df_ind['rel_vol'] >= 1.0) & 
                        (df_ind['close'] < df_ind['bb_lower'])
                    )
                    df_s['short_setup'] = (
                        (df_ind['rsi'] > min_rsi_short) & 
                        (df_ind['rel_vol'] >= 1.0) & 
                        (df_ind['close'] > df_ind['bb_upper'])
                    )
                
                n_setups = df_s['long_setup'].sum() + df_s['short_setup'].sum()
                if n_setups < 50:
                    continue
                
                df_t = calc_theo_targets(df_s, stop_atr_mult=stop)
                trades = simulate_all_setups(df_t)
                
                if len(trades) < 30:
                    continue
                
                wr = trades['is_winner'].mean()*100
                ws = trades[trades['is_winner']]['pnl_per_share'].sum()
                ls = abs(trades[~trades['is_winner']]['pnl_per_share'].sum())
                pf = ws/ls if ls > 0 else 0
                avg = trades['pnl_per_share'].mean()
                
                results.append({
                    'rsi_l': max_rsi_long, 'rsi_s': min_rsi_short,
                    'vwap': min_vwap_dist, 'stop': stop,
                    'n': len(trades), 'wr': wr, 'pf': pf, 'pnl': avg
                })
                
                if pf >= 1.0:  # Only log profitable configs
                    out.write(f"<{max_rsi_long:<5} >{min_rsi_short:<5} >{min_vwap_dist:<5.1f} {stop:<6.1f} {len(trades):<8} {wr:<8.1f} {pf:<8.2f} ${avg:<.2f}\n")
                    out.flush()

# Summary
out.write("\n" + "="*100 + "\n")
out.write("SUMMARY - PROFITABLE CONFIGURATIONS (PF >= 1.0)\n")
out.write("="*100 + "\n")

profitable = [r for r in results if r['pf'] >= 1.0]
profitable.sort(key=lambda x: x['pf'], reverse=True)

out.write(f"\nFound {len(profitable)} profitable configurations out of {len(results)} tested\n\n")

out.write("TOP 10 BY PROFIT FACTOR:\n")
out.write("-"*80 + "\n")
for i, r in enumerate(profitable[:10], 1):
    out.write(f"{i}. RSI<{r['rsi_l']}, RSI>{r['rsi_s']}, VWAP>{r['vwap']:.1f}, Stop={r['stop']:.1f}: "
              f"N={r['n']}, WR={r['wr']:.1f}%, PF={r['pf']:.2f}, Avg=${r['pnl']:.2f}\n")

# Best by sample size among profitable
if profitable:
    profitable_large = [r for r in profitable if r['n'] >= 100]
    if profitable_large:
        profitable_large.sort(key=lambda x: x['pf'], reverse=True)
        out.write(f"\nBEST WITH N>=100 TRADES:\n")
        out.write("-"*80 + "\n")
        for i, r in enumerate(profitable_large[:5], 1):
            out.write(f"{i}. RSI<{r['rsi_l']}, RSI>{r['rsi_s']}, VWAP>{r['vwap']:.1f}, Stop={r['stop']:.1f}: "
                      f"N={r['n']}, WR={r['wr']:.1f}%, PF={r['pf']:.2f}, Avg=${r['pnl']:.2f}\n")

out.write("\nDone.\n")
out.close()
