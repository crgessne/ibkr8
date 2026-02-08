"""Test more aggressive filters to find profitable edge."""
import sys
sys.path.insert(0, r'C:\Users\Administrator\ibkr8\src')
import pandas as pd
import numpy as np
from indicators import calc_all_indicators, calc_theo_targets
from outcome_sim import simulate_all_setups

print("Loading data...", flush=True)
df = pd.read_csv(r'C:\Users\Administrator\ibkr8\data\tsla_5min_2years.csv', parse_dates=['time'], index_col='time')
df_ind = calc_all_indicators(df)
print(f"Data: {len(df)} bars\n", flush=True)

print("="*90, flush=True)
print("EXTREME RSI + BB + VWAP DISTANCE SWEEP", flush=True)
print("="*90, flush=True)

results = []

# Test extreme RSI thresholds
for max_rsi_long in [20, 25, 30]:
    for min_rsi_short in [70, 75, 80]:
        for min_vwap_dist in [1.0, 1.5, 2.0]:
            for stop in [1.5, 2.0, 2.5]:
                df_s = df_ind.copy()
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

# Print best results
results.sort(key=lambda x: x['pf'], reverse=True)
print(f"\n{'RSI_L':<6} {'RSI_S':<6} {'VWAP':<6} {'Stop':<6} {'N':<6} {'WR%':<8} {'PF':<8} {'AvgPnL':<10}", flush=True)
print("-"*70, flush=True)
for r in results[:15]:
    print(f"<{r['rsi_l']:<5} >{r['rsi_s']:<5} >{r['vwap']:<5.1f} {r['stop']:<6.1f} {r['n']:<6} {r['wr']:<8.1f} {r['pf']:<8.2f} ${r['pnl']:<.2f}", flush=True)

# Highlight profitable ones
profitable = [r for r in results if r['pf'] >= 1.0]
print(f"\n{'='*90}", flush=True)
print(f"PROFITABLE CONFIGURATIONS (PF >= 1.0): {len(profitable)}", flush=True)
print("="*90, flush=True)
for r in profitable:
    print(f"RSI<{r['rsi_l']}, RSI>{r['rsi_s']}, VWAP>{r['vwap']:.1f}ATR, Stop={r['stop']:.1f}ATR: "
          f"{r['n']} trades, WR={r['wr']:.1f}%, PF={r['pf']:.2f}", flush=True)
