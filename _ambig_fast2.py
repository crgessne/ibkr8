"""How often does the deciding bar hit BOTH stop and target?
Only checks setup-filtered test bars (fast: ~6K bars per stop).
"""
import sys, warnings
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')

# Tee stdout to file
_f = open("_ambig_out.txt", "w", encoding="utf-8")
class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for x in self.s: x.write(d); x.flush()
    def flush(self):
        for x in self.s: x.flush()
sys.stdout = _Tee(sys.__stdout__, _f)

import numpy as np, pandas as pd
from master_pipeline import (
    load_and_validate_data, calculate_core_indicators, apply_setup_filter,
    DATA_FILE, TEST_YEAR,
)
from label_generator import LabelConfig, generate_labels

print("Loading...")
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
day_end_map = np.zeros(n, dtype=int)
i = 0
while i < n:
    d = date[i]
    j = i
    while j < n and date[j] == d:
        j += 1
    day_end_map[i:j] = j
    i = j

print()
print("=" * 95)
print(f"  AMBIGUOUS BAR ANALYSIS: Setup-Filtered Test Bars (>= {TEST_YEAR})")
print("  Q: How often does the DECIDING bar hit BOTH stop AND target?")
print("=" * 95)
print()
print(f"  {'Stop':>5s}  {'SetupBars':>9s}  {'Resolved':>8s}  {'BothHit':>7s}  {'BothPct':>7s}  {'CurWR':>6s}  {'AdjWR':>6s}  {'Boost':>8s}  {'EOD':>5s}")
print(f"  {'-'*5}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*5}")

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
    eod_count = 0  # trades that expire at end of day

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
        de = day_end_map[j]
        if j + 1 >= de:
            continue

        found = False
        for k in range(j + 1, de):
            if is_long:
                hs = low[k] <= stop_p
                ht = high[k] >= target_p
            else:
                hs = high[k] >= stop_p
                ht = low[k] <= target_p
            if hs or ht:
                resolved += 1
                found = True
                if ht and not hs:
                    wins += 1
                if hs and ht:
                    ambig += 1
                break
        if not found:
            resolved += 1
            eod_count += 1  # neither stop nor target hit -> EOD exit (loss)

    n_setup = len(test_indices)
    pct = ambig / resolved * 100 if resolved > 0 else 0
    cur_wr = wins / resolved * 100 if resolved > 0 else 0
    adj_wr = (wins + ambig) / resolved * 100 if resolved > 0 else 0
    boost = adj_wr - cur_wr
    eod_pct = eod_count / resolved * 100 if resolved > 0 else 0
    print(f"  {stop_atr:>5.2f}  {n_setup:>9,}  {resolved:>8,}  {ambig:>7,}  {pct:>6.2f}%  {cur_wr:>5.1f}%  {adj_wr:>5.1f}%  +{boost:.2f}pp  {eod_pct:.1f}%")

print()
print("Legend:")
print("  BothHit  = deciding bar touches BOTH stop and VWAP target")
print("  CurWR    = current win rate (ambiguous bars counted as losses)")
print("  AdjWR    = adjusted win rate (if ALL ambiguous bars were wins instead)")
print("  Boost    = max possible WR increase from resolving ambiguity favorably")
print("  EOD      = % of trades that expire at end of day (neither stop nor target hit)")
print()
print("Done.")
