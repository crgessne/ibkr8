"""How often does the deciding bar hit BOTH stop and target? (Fast version)

Instead of iterating every bar, leverage the existing label generator's structure
and just check the DECIDING bar for ambiguity.
"""
import sys, warnings
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from master_pipeline import (
    load_and_validate_data, calculate_core_indicators, apply_setup_filter,
    DATA_FILE, TEST_YEAR,
)
from label_generator import LabelConfig, generate_labels

# Tee output
_out = open("_ambig_out.txt", "w", encoding="utf-8")
class Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for x in self.s: x.write(d); x.flush()
    def flush(self):
        for x in self.s: x.flush()
sys.stdout = Tee(sys.__stdout__, _out)

df = load_and_validate_data(DATA_FILE)
df = calculate_core_indicators(df, verbose=False)

stops = [0.25, 0.35, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5]
config = LabelConfig(stop_atrs=stops)
df = generate_labels(df, config)

close = df['close'].values
high = df['high'].values
low = df['low'].values
vwap = df['vwap'].values
atr = df['atr'].values
date = df['date'].values
n = len(df)

# Build day boundaries once
day_ends = {}
i = 0
while i < n:
    d = date[i]
    j = i
    while j < n and date[j] == d:
        j += 1
    for k in range(i, j):
        day_ends[k] = j
    i = j

print("How often does the DECIDING bar hit BOTH stop and target?")
print("These are currently labeled as LOSSES (conservative: stop-first assumption).")
print()

# --- ALL BARS ---
print("=" * 90)
print("  ALL BARS (full dataset)")
print("=" * 90)
print(f"  {'Stop':>5s}  {'Resolved':>8s}  {'BothHit':>7s}  {'BothPct':>7s}  {'CurWins':>7s}  {'CurWR':>6s}  {'AdjWR':>6s}  {'Boost':>8s}")
print(f"  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*8}")

for stop_atr in stops:
    resolved = 0
    ambig = 0
    wins = 0

    for j in range(n):
        a = atr[j]
        v = vwap[j]
        if np.isnan(a) or a <= 0 or np.isnan(v):
            continue
        entry = close[j]
        is_long = entry < v
        stop_dist = stop_atr * a
        stop_p = entry - stop_dist if is_long else entry + stop_dist
        target_p = v
        de = day_ends.get(j, j + 1)
        if j + 1 >= de:
            continue

        for k in range(j + 1, de):
            if is_long:
                hs = low[k] <= stop_p
                ht = high[k] >= target_p
            else:
                hs = high[k] >= stop_p
                ht = low[k] <= target_p
            if hs or ht:
                resolved += 1
                if ht and not hs:
                    wins += 1
                if hs and ht:
                    ambig += 1
                break

    pct = ambig / resolved * 100 if resolved > 0 else 0
    cur_wr = wins / resolved * 100 if resolved > 0 else 0
    adj_wr = (wins + ambig) / resolved * 100 if resolved > 0 else 0
    boost = adj_wr - cur_wr
    print(f"  {stop_atr:>5.2f}  {resolved:>8,}  {ambig:>7,}  {pct:>6.2f}%  {wins:>7,}  {cur_wr:>5.1f}%  {adj_wr:>5.1f}%  +{boost:.2f}pp")

# --- SETUP-FILTERED TEST BARS ONLY ---
print()
print("=" * 90)
print(f"  SETUP-FILTERED TEST BARS (>= {TEST_YEAR})")
print("=" * 90)
print(f"  {'Stop':>5s}  {'Resolved':>8s}  {'BothHit':>7s}  {'BothPct':>7s}  {'CurWins':>7s}  {'CurWR':>6s}  {'AdjWR':>6s}  {'Boost':>8s}")
print(f"  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*8}")

for stop_atr in stops:
    label_col = f"label_s{stop_atr}".replace(".", "_")
    valid = df[label_col].notna()
    df_v = df[valid].copy()
    mask = apply_setup_filter(df_v, stop_atr=stop_atr, min_dist_atr=0.5,
                               min_minutes_session=15, max_minutes_session=360, min_rr_setup=1.0)
    df_s = df_v[mask].copy()
    df_s['year'] = pd.to_datetime(df_s['datetime']).dt.year
    df_test = df_s[df_s['year'] >= TEST_YEAR]
    test_indices = df_test.index.values

    resolved = 0
    ambig = 0
    wins = 0

    for j in test_indices:
        a = atr[j]
        v = vwap[j]
        if np.isnan(a) or a <= 0 or np.isnan(v):
            continue
        entry = close[j]
        is_long = entry < v
        stop_dist = stop_atr * a
        stop_p = entry - stop_dist if is_long else entry + stop_dist
        target_p = v
        de = day_ends.get(j, j + 1)
        if j + 1 >= de:
            continue

        for k in range(j + 1, de):
            if is_long:
                hs = low[k] <= stop_p
                ht = high[k] >= target_p
            else:
                hs = high[k] >= stop_p
                ht = low[k] <= target_p
            if hs or ht:
                resolved += 1
                if ht and not hs:
                    wins += 1
                if hs and ht:
                    ambig += 1
                break

    pct = ambig / resolved * 100 if resolved > 0 else 0
    cur_wr = wins / resolved * 100 if resolved > 0 else 0
    adj_wr = (wins + ambig) / resolved * 100 if resolved > 0 else 0
    boost = adj_wr - cur_wr
    n_test = len(test_indices)
    print(f"  {stop_atr:>5.2f}  {resolved:>8,}  {ambig:>7,}  {pct:>6.2f}%  {wins:>7,}  {cur_wr:>5.1f}%  {adj_wr:>5.1f}%  +{boost:.2f}pp  (of {n_test:,} setup bars)")

print()
print("Note: 'BothHit' bars are currently all counted as LOSSES.")
print("If they were all WINS instead, win rate would increase by 'Boost'.")
print("Reality is somewhere in between (some would be wins, some losses).")
