"""Analyze the completed pipeline results: stop 0.25 & 0.35 ATR positive, rest negative.
Dig into trade logs, direction, time, distance, and explore filters to improve."""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 160)

# ============================================================================
# 1. Summary table from pipeline results
# ============================================================================
results = pd.read_csv('data/master_pipeline_results_20260212_133211.csv')
print("=" * 80)
print("PIPELINE RESULTS SUMMARY (all 9 stop levels)")
print("=" * 80)
cols = ['stop_atr', 'rr', 'n_trades', 'win_rate', 'total_net_pnl', 'total_return_pct', 'avg_risk_dollars']
print(results[cols].to_string(index=False))
print()

# ============================================================================
# 2. Deep-dive on the two profitable stops
# ============================================================================
timestamp = '20260212_133211'
for stop in [0.25, 0.35, 0.4, 0.5]:
    fname = f'data/trades_y2024_stop{stop}_seltop_500_kregressor_{timestamp}.csv'
    if not Path(fname).exists():
        print(f"[SKIP] {fname} not found")
        continue
    df = pd.read_csv(fname)
    print("=" * 80)
    print(f"TRADE LOG: stop={stop} ATR  ({len(df)} trades)")
    print("=" * 80)

    # Basic stats
    wins = df[df['net_pnl'] > 0]
    losses = df[df['net_pnl'] <= 0]
    print(f"  Winners: {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
    print(f"  Losers:  {len(losses)} ({len(losses)/len(df)*100:.1f}%)")
    if len(wins) > 0:
        print(f"  Avg win:  ${wins['net_pnl'].mean():.2f}")
    if len(losses) > 0:
        print(f"  Avg loss: ${losses['net_pnl'].mean():.2f}")
    print(f"  Total P&L: ${df['net_pnl'].sum():.2f}")
    print()

    # Direction breakdown
    if 'is_long' in df.columns:
        for d, lbl in [(True, 'LONG'), (False, 'SHORT')]:
            sub = df[df['is_long'] == d]
            if len(sub) > 0:
                wr = sub['net_pnl'].gt(0).mean()
                print(f"  {lbl}: {len(sub)} trades, WR={wr*100:.1f}%, NetP&L=${sub['net_pnl'].sum():.2f}, AvgP&L=${sub['net_pnl'].mean():.2f}")
        print()

    # Distance breakdown
    if 'vwap_dist_atr' in df.columns:
        df['dist_bucket'] = pd.cut(df['vwap_dist_atr'], bins=[0, 0.5, 1.0, 1.5, 2.0, 3.0, 100], labels=['0-0.5','0.5-1','1-1.5','1.5-2','2-3','3+'])
        dist_stats = df.groupby('dist_bucket', observed=True).agg(
            n=('net_pnl', 'count'),
            wr=('net_pnl', lambda x: (x > 0).mean()),
            pnl=('net_pnl', 'sum'),
            avg_pnl=('net_pnl', 'mean'),
        )
        print("  P&L by distance from VWAP:")
        print(dist_stats.to_string())
        print()

    # Time of day breakdown
    if 'datetime' in df.columns:
        df['dt'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['dt'].dt.hour
        hourly = df.groupby('hour').agg(
            n=('net_pnl', 'count'),
            wr=('net_pnl', lambda x: (x > 0).mean()),
            pnl=('net_pnl', 'sum'),
        )
        print("  P&L by hour:")
        print(hourly.to_string())
        print()

    # Monthly P&L
    if 'datetime' in df.columns:
        df['month'] = df['dt'].dt.to_period('M')
        monthly = df.groupby('month').agg(
            n=('net_pnl', 'count'),
            wr=('net_pnl', lambda x: (x > 0).mean()),
            pnl=('net_pnl', 'sum'),
        )
        print("  Monthly P&L:")
        print(monthly.to_string())
        print()

# ============================================================================
# 3. Feature importance from the saved models
# ============================================================================
print("=" * 80)
print("FEATURE IMPORTANCE (stop=0.25 ATR regressor)")
print("=" * 80)
try:
    from model_persistence import load_model
    model_path = f'models/rf_vwap_stop0.25_{timestamp}.pkl'
    model, meta = load_model(model_path)
    importances = pd.DataFrame({
        'feature': meta.get('features', []),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances.to_string(index=False))
except Exception as e:
    print(f"  Could not load model: {e}")
print()

# ============================================================================
# 4. What if we apply filters to the top-500 trades?
# ============================================================================
print("=" * 80)
print("FILTER ANALYSIS: Can we improve stop=0.25 and stop=0.35?")
print("=" * 80)

for stop in [0.25, 0.35]:
    fname = f'data/trades_y2024_stop{stop}_seltop_500_kregressor_{timestamp}.csv'
    if not Path(fname).exists():
        continue
    df = pd.read_csv(fname)
    df['dt'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['dt'].dt.hour
    
    print(f"\n  --- Stop {stop} ATR ---")
    
    filters = {}
    # Baseline
    filters['baseline'] = df
    
    # Long only
    if 'is_long' in df.columns:
        filters['long_only'] = df[df['is_long'] == True]
        filters['short_only'] = df[df['is_long'] == False]
    
    # Distance filters
    if 'vwap_dist_atr' in df.columns:
        filters['dist>0.5'] = df[df['vwap_dist_atr'] > 0.5]
        filters['dist>1.0'] = df[df['vwap_dist_atr'] > 1.0]
        filters['dist>1.5'] = df[df['vwap_dist_atr'] > 1.5]
    
    # Time filters
    filters['hour>=10'] = df[df['hour'] >= 10]
    filters['hour<15'] = df[df['hour'] < 15]
    filters['10<=hour<15'] = df[(df['hour'] >= 10) & (df['hour'] < 15)]
    
    # Combo
    if 'is_long' in df.columns and 'vwap_dist_atr' in df.columns:
        filters['long+dist>1'] = df[(df['is_long'] == True) & (df['vwap_dist_atr'] > 1.0)]
        filters['long+dist>1+10-15h'] = df[(df['is_long'] == True) & (df['vwap_dist_atr'] > 1.0) & (df['hour'] >= 10) & (df['hour'] < 15)]
    
    # Max 1 trade per day
    if 'datetime' in df.columns:
        df_sorted = df.sort_values('net_pnl', ascending=False)
        df_sorted['trade_date'] = df_sorted['dt'].dt.date
        best_per_day = df_sorted.drop_duplicates(subset='trade_date', keep='first')
        filters['1_per_day'] = best_per_day
    
    print(f"  {'Filter':<30s} {'N':>5s} {'WR':>7s} {'NetP&L':>12s} {'AvgP&L':>10s}")
    print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*12} {'-'*10}")
    for label, sub in filters.items():
        if len(sub) == 0:
            continue
        wr = sub['net_pnl'].gt(0).mean()
        pnl = sub['net_pnl'].sum()
        avg = sub['net_pnl'].mean()
        print(f"  {label:<30s} {len(sub):>5d} {wr*100:>6.1f}% ${pnl:>10,.0f} ${avg:>9,.2f}")

print("\nDone.")
