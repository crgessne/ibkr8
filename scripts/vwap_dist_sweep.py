"""Test VWAP distance filter and other parameters."""
import sys
sys.path.insert(0, r'C:\Users\Administrator\ibkr8\src')
import pandas as pd
from indicators import calc_all_indicators, identify_reversal_setups, calc_theo_targets
from outcome_sim import simulate_all_setups

df = pd.read_csv(r'C:\Users\Administrator\ibkr8\data\tsla_5min_2years.csv', parse_dates=['time'], index_col='time')
print(f"Loaded {len(df)} bars")
df_ind = calc_all_indicators(df)
print("Indicators calculated")

print("\n" + "="*80)
print("VWAP DISTANCE FILTER + STOP WIDTH SWEEP")
print("="*80)
print(f"MinDist  Stop   Trades   Win%     PF       AvgPnL")
print("-"*80)

results = []
for min_vwap_dist in [0.0, 0.5, 1.0, 1.5, 2.0]:
    for stop_atr in [1.0, 1.5, 2.0, 2.5]:
        df_s = df_ind.copy()
        if min_vwap_dist > 0:
            df_s['long_setup'] = (
                (df_ind['rsi'] < 35) & 
                (df_ind['rel_vol'] >= 1.0) & 
                (df_ind['close'] < df_ind['bb_lower']) &
                (df_ind['vwap_dist_atr'] < -min_vwap_dist)
            )
            df_s['short_setup'] = (
                (df_ind['rsi'] > 65) & 
                (df_ind['rel_vol'] >= 1.0) & 
                (df_ind['close'] > df_ind['bb_upper']) &
                (df_ind['vwap_dist_atr'] > min_vwap_dist)
            )
        else:
            df_s = identify_reversal_setups(df_ind)
        
        df_t = calc_theo_targets(df_s, stop_atr_mult=stop_atr)
        trades = simulate_all_setups(df_t)
        
        if len(trades) < 30:
            continue
        
        wr = trades['is_winner'].mean()*100
        ws = trades[trades['is_winner']]['pnl_per_share'].sum()
        ls = abs(trades[~trades['is_winner']]['pnl_per_share'].sum())
        pf = ws/ls if ls > 0 else 0
        avg = trades['pnl_per_share'].mean()
        
        results.append({'dist': min_vwap_dist, 'stop': stop_atr, 'n': len(trades), 
                        'wr': wr, 'pf': pf, 'pnl': avg})
        print(f"{min_vwap_dist:<8.1f} {stop_atr:<6.1f} {len(trades):<8} {wr:<8.1f} {pf:<8.2f} ${avg:<.2f}")

# Find best profitable config
print("\n" + "="*80)
print("BEST CONFIGURATIONS (PF > 1.0)")
print("="*80)
profitable = [r for r in results if r['pf'] > 1.0]
if profitable:
    profitable.sort(key=lambda x: x['pf'], reverse=True)
    for r in profitable[:5]:
        print(f"VWAP dist>{r['dist']:.1f} ATR, Stop={r['stop']:.1f} ATR: "
              f"{r['n']} trades, {r['wr']:.1f}% WR, PF={r['pf']:.2f}, ${r['pnl']:.2f}/trade")
else:
    print("No profitable configurations found with these parameters")
    print("\nBest by PF:")
    results.sort(key=lambda x: x['pf'], reverse=True)
    for r in results[:3]:
        print(f"VWAP>{r['dist']:.1f}, Stop={r['stop']:.1f}: PF={r['pf']:.2f}, WR={r['wr']:.1f}%")
