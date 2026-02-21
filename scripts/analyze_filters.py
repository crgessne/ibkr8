"""Filter optimization on the profitable stop levels."""
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, 'src')

print('FILTER OPTIMIZATION ANALYSIS')
print('='*80)

for stop in [0.25, 0.35]:
    fname = f'data/trades_y2024_stop{stop}_seltop_500_kregressor_20260212_133211.csv'
    df = pd.read_csv(fname)
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['date'] = pd.to_datetime(df['datetime']).dt.date

    print(f'\nStop {stop} ATR')
    print('-'*60)

    filters = {
        'baseline': df,
        'dist>1': df[df['vwap_dist_atr'] > 1.0],
        'dist>1.5': df[df['vwap_dist_atr'] > 1.5],
        'dist 1-2.5': df[(df['vwap_dist_atr'] >= 1.0) & (df['vwap_dist_atr'] <= 2.5)],
        'dist 1-3': df[(df['vwap_dist_atr'] >= 1.0) & (df['vwap_dist_atr'] <= 3.0)],
        'dist 1.5-2': df[(df['vwap_dist_atr'] >= 1.5) & (df['vwap_dist_atr'] <= 2.0)],
        'hour<16': df[df['hour'] < 16],
        'hour 14-16': df[(df['hour'] >= 14) & (df['hour'] < 16)],
        'dist>1 + hr<16': df[(df['vwap_dist_atr'] > 1.0) & (df['hour'] < 16)],
        'dist>1 + hr14-16': df[(df['vwap_dist_atr'] > 1.0) & (df['hour'] >= 14) & (df['hour'] < 16)],
        'dist1-3 + hr<16': df[(df['vwap_dist_atr'] >= 1.0) & (df['vwap_dist_atr'] <= 3.0) & (df['hour'] < 16)],
        'RR>5': df[df['per_trade_rr'] > 5],
        'RR>8': df[df['per_trade_rr'] > 8],
        'RR 5-12': df[(df['per_trade_rr'] >= 5) & (df['per_trade_rr'] <= 12)],
    }

    # 1 per day versions of best filters
    for key in list(filters.keys()):
        sub = filters[key].copy()
        if len(sub) > 0:
            sub_sorted = sub.sort_values('per_trade_rr', ascending=False)
            sub_sorted['td'] = pd.to_datetime(sub_sorted['datetime']).dt.date
            best_day = sub_sorted.drop_duplicates(subset='td', keep='first')
            filters[key + ' (1/day)'] = best_day

    hdr = f"  {'Filter':<30s} {'N':>5s} {'WR':>7s} {'NetPnL':>12s} {'AvgPnL':>10s} {'WinAvg':>10s} {'LossAvg':>10s}"
    print(hdr)
    print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for label, sub in filters.items():
        if len(sub) == 0:
            continue
        wr = sub['net_pnl'].gt(0).mean()
        pnl = sub['net_pnl'].sum()
        avg = sub['net_pnl'].mean()
        wavg = sub.loc[sub['net_pnl'] > 0, 'net_pnl'].mean() if (sub['net_pnl'] > 0).any() else 0
        lavg = sub.loc[sub['net_pnl'] <= 0, 'net_pnl'].mean() if (sub['net_pnl'] <= 0).any() else 0
        print(f"  {label:<30s} {len(sub):>5d} {wr*100:>6.1f}% {pnl:>+12,.0f} {avg:>+10,.1f} {wavg:>+10,.1f} {lavg:>+10,.1f}")

# ============================================================================
# Feature importance
# ============================================================================
print()
print('='*80)
print('FEATURE IMPORTANCE (stop=0.25 ATR regressor)')
print('='*80)
try:
    from model_persistence import load_model
    model_path = 'models/rf_vwap_stop0.25_20260212_133211.pkl'
    model, meta = load_model(model_path)
    importances = pd.DataFrame({
        'feature': meta.get('features', []),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importances.iterrows():
        bar = '#' * int(row['importance'] * 200)
        print(f"  {row['feature']:<25s} {row['importance']:.4f} {bar}")
except Exception as e:
    print(f"  Could not load model: {e}")

# Feature importance for stop=0.35
print()
print('='*80)
print('FEATURE IMPORTANCE (stop=0.35 ATR regressor)')
print('='*80)
try:
    model_path = 'models/rf_vwap_stop0.35_20260212_133211.pkl'
    model, meta = load_model(model_path)
    importances = pd.DataFrame({
        'feature': meta.get('features', []),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importances.iterrows():
        bar = '#' * int(row['importance'] * 200)
        print(f"  {row['feature']:<25s} {row['importance']:.4f} {bar}")
except Exception as e:
    print(f"  Could not load model: {e}")

# ============================================================================
# Check: what does the regressor's predicted payoff look like for selected trades?
# ============================================================================
print()
print('='*80)
print('REGRESSOR PREDICTION DISTRIBUTION (stop=0.25)')
print('='*80)
# Need to re-run the pipeline logic briefly to get predictions
# Instead, let's look at what the top-500 vs top-200 vs top-100 look like
for stop in [0.25, 0.35]:
    fname = f'data/trades_y2024_stop{stop}_seltop_500_kregressor_20260212_133211.csv'
    df = pd.read_csv(fname)
    
    # Sort by per_trade_rr descending (proxy for regressor score order - not exact but directional)
    # Actually, the trade log is already in regressor-score order since top_n picks highest scores first
    print(f"\n  stop={stop} — P&L by top-N tranches:")
    for n in [50, 100, 150, 200, 250, 300, 400, 500]:
        sub = df.head(n)
        wr = sub['net_pnl'].gt(0).mean()
        pnl = sub['net_pnl'].sum()
        print(f"    top-{n:>3d}: WR={wr*100:5.1f}% NetP&L={pnl:>+10,.0f} avgP&L={pnl/n:>+8,.1f}")

print("\nDone.")
