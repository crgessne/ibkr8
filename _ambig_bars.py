"""How often does the deciding bar hit BOTH stop and target?"""
import sys, warnings
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from master_pipeline import load_and_validate_data, calculate_core_indicators, DATA_FILE

df = load_and_validate_data(DATA_FILE)
df = calculate_core_indicators(df, verbose=False)

close = df['close'].values
high = df['high'].values
low = df['low'].values
vwap = df['vwap'].values
atr = df['atr'].values
date = df['date'].values
n = len(df)

stops = [0.25, 0.35, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5]

print("How often does the DECIDING bar (first bar to resolve the trade) hit BOTH stop and target?")
print("These are currently all labeled as LOSSES (conservative: stop-first assumption).")
print("If labeled as WINS instead, win rate would increase by the 'WR boost' column.")
print()
print(f"  Stop   Resolved  BothHit  BothPct  CurWins  CurWR   AdjWR   WR_boost")
print(f"  -----  --------  -------  -------  -------  ------  ------  --------")

for stop_atr in stops:
    resolved = 0
    ambig = 0
    wins = 0
    total_bars_checked = 0

    for j in range(n):
        entry = close[j]
        v = vwap[j]
        a = atr[j]
        if np.isnan(a) or a <= 0 or np.isnan(v):
            continue
        is_long = entry < v
        stop_dist = stop_atr * a
        if is_long:
            stop_p = entry - stop_dist
            target_p = v
        else:
            stop_p = entry + stop_dist
            target_p = v

        day = date[j]
        day_end = j + 1
        while day_end < n and date[day_end] == day:
            day_end += 1

        if j + 1 >= day_end:
            continue

        for k in range(j + 1, day_end):
            if is_long:
                hit_stop = low[k] <= stop_p
                hit_target = high[k] >= target_p
            else:
                hit_stop = high[k] >= stop_p
                hit_target = low[k] <= target_p

            if hit_stop or hit_target:
                resolved += 1
                if hit_target and not hit_stop:
                    wins += 1
                if hit_stop and hit_target:
                    ambig += 1
                    # Currently labeled as loss (stop first)
                break

    pct = ambig / resolved * 100 if resolved > 0 else 0
    cur_wr = wins / resolved * 100 if resolved > 0 else 0
    adj_wr = (wins + ambig) / resolved * 100 if resolved > 0 else 0
    boost = adj_wr - cur_wr
    print(f"  {stop_atr:>5.2f}  {resolved:>8,}  {ambig:>7,}  {pct:>6.2f}%  {wins:>7,}  {cur_wr:>5.1f}%  {adj_wr:>5.1f}%  +{boost:>5.2f}pp")
