import sys
sys.path.insert(0, r'C:\Users\Administrator\ibkr8\src')
import pandas as pd
from indicators import calc_all_indicators, calc_theo_targets
from outcome_sim import simulate_all_setups

print("Loading data...", flush=True)
df = pd.read_csv(r'C:\Users\Administrator\ibkr8\data\tsla_5min_2years.csv', parse_dates=['time'], index_col='time')
df_ind = calc_all_indicators(df)
print(f"Data loaded: {len(df)} bars", flush=True)

# Test different VWAP distance filters
for min_dist in [0.5, 1.0, 1.5, 2.0]:
    for stop in [1.5, 2.0]:
        df_s = df_ind.copy()
        df_s['long_setup'] = (
            (df_ind['rsi'] < 35) & 
            (df_ind['rel_vol'] >= 1.0) & 
            (df_ind['close'] < df_ind['bb_lower']) &
            (df_ind['vwap_dist_atr'] < -min_dist)
        )
        df_s['short_setup'] = (
            (df_ind['rsi'] > 65) & 
            (df_ind['rel_vol'] >= 1.0) & 
            (df_ind['close'] > df_ind['bb_upper']) &
            (df_ind['vwap_dist_atr'] > min_dist)
        )
        
        n_setups = df_s['long_setup'].sum() + df_s['short_setup'].sum()
        if n_setups < 20:
            print(f"VWAP>{min_dist}, Stop={stop}: only {n_setups} setups, skipping", flush=True)
            continue
        
        df_t = calc_theo_targets(df_s, stop_atr_mult=stop)
        trades = simulate_all_setups(df_t)
        
        if len(trades) == 0:
            continue
        
        wr = trades['is_winner'].mean()*100
        ws = trades[trades['is_winner']]['pnl_per_share'].sum()
        ls = abs(trades[~trades['is_winner']]['pnl_per_share'].sum())
        pf = ws/ls if ls > 0 else 0
        avg = trades['pnl_per_share'].mean()
        print(f"VWAP>{min_dist:.1f}ATR, Stop={stop}: {len(trades)} trades, WR={wr:.1f}%, PF={pf:.2f}, Avg=${avg:.2f}", flush=True)
