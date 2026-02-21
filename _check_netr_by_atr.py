"""
Check expected net_r by vwap_width_atr bucket.
This tells us whether the loss function CAN learn to prefer wider ATR setups.
If E[net_r] > 0 for wide ATR but not for narrow → network should learn the signal.
"""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
import pandas as pd
import numpy as np
from pathlib import Path

# Load pre-computed indicators if available, else recompute
ind_files = sorted(Path('data').glob('tsla_indicators_*.parquet'))
if ind_files:
    df = pd.read_parquet(ind_files[-1])
    df['date'] = pd.to_datetime(df['date']).dt.date
    print(f"Loaded indicators: {len(df):,} rows from {ind_files[-1].name}")
else:
    from master_pipeline import load_and_validate_data, calculate_core_indicators, generate_all_labels, STOP_ATRS
    from label_generator import LabelConfig, generate_labels
    df = load_and_validate_data('data/tsla_5min_10years.csv')
    df = calculate_core_indicators(df, verbose=False)
    config = LabelConfig(stop_atrs=[0.75])
    df = generate_labels(df, config)
    print(f"Recomputed: {len(df):,} rows")

STOP_ATR = 0.75
COMM = 0.005
ATR_ASSUMED = df['atr'].median()

for SLIP in [0.01, 0.05, 0.10, 0.18]:
    cost_per_share = 2 * (COMM + SLIP)
    label_col = 'label_s0_75'
    valid = df[label_col].notna() & (df['vwap_width_atr'] >= 0.3)

    d = df[valid].copy()
    d['reward_ps'] = d['vwap_width_atr'] * d['atr']
    d['risk_ps']   = STOP_ATR * d['atr']
    d['net_r'] = np.where(
        d[label_col] == 1,
        (d['reward_ps'] - cost_per_share) / d['risk_ps'],
        (-d['risk_ps']  - cost_per_share) / d['risk_ps'],
    )

    # Bucket by vwap_width_atr
    bins = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 99]
    labels = ['0.3-0.5','0.5-0.75','0.75-1.0','1.0-1.5','1.5-2.0','2.0-3.0','3.0+']
    d['atr_bucket'] = pd.cut(d['vwap_width_atr'], bins=bins, labels=labels)

    print(f"\n=== slip={SLIP:.2f} (cost/share={cost_per_share:.3f}) ===")
    print(f"{'bucket':<12} {'n':>6} {'win%':>6} {'net_r/trade':>12} {'E[net_r]':>10} {'verdict'}")
    print("-" * 60)
    for bucket in labels:
        sub = d[d['atr_bucket'] == bucket]
        if len(sub) < 10:
            continue
        wr = sub[label_col].mean()
        avg_netr = sub['net_r'].mean()
        total_netr = sub['net_r'].sum()
        verdict = "✓ TAKE" if avg_netr > 0 else "✗ SKIP"
        print(f"{bucket:<12} {len(sub):>6} {wr*100:>5.1f}% {avg_netr:>12.3f} {total_netr:>10.0f}  {verdict}")

    overall = d['net_r'].mean()
    print(f"{'OVERALL':<12} {len(d):>6} {d[label_col].mean()*100:>5.1f}% {overall:>12.3f}  {'✓' if overall>0 else '✗'}")
