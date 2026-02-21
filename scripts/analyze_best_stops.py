"""
Analyze the profitable stop levels (0.25, 0.35 ATR) with different filters:
- Different top_n values
- Long-only vs both directions
- Minimum distance filter
- Per-month breakdown
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import io

# Tee stdout to file
_out_file = open(Path(__file__).parent.parent / "_best_stops_out.txt", "w", encoding="utf-8")
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
sys.stdout = Tee(sys.__stdout__, _out_file)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

from label_generator import LabelConfig, generate_labels

# Import pipeline helpers
sys.path.insert(0, str(Path(__file__).parent))
from master_pipeline import (
    load_and_validate_data, calculate_core_indicators, get_feature_columns,
    STOP_ATRS, SHARES_PER_TRADE, COMMISSION_PER_SHARE, SLIPPAGE_PER_SHARE,
    RF_REG_PARAMS, DATA_FILE, TEST_YEAR,
)

def main():
    print("=" * 80)
    print("DEEP ANALYSIS OF PROFITABLE STOP LEVELS")
    print("=" * 80)

    # Load and prepare data
    df = load_and_validate_data(DATA_FILE)
    df = calculate_core_indicators(df, verbose=False)
    features = get_feature_columns(df)

    # Generate labels for tight stops only
    tight_stops = [0.25, 0.35]
    config = LabelConfig(stop_atrs=tight_stops)
    df = generate_labels(df, config)

    costs = 2.0 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * SHARES_PER_TRADE

    for stop_atr in tight_stops:
        print(f"\n{'=' * 80}")
        print(f"  STOP = {stop_atr} ATR")
        print(f"{'=' * 80}")

        label_col = f"label_s{stop_atr}".replace(".", "_")
        valid = df[label_col].notna()
        df_valid = df[valid].copy()

        X = df_valid[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_raw = df_valid[label_col].astype(int)

        # Compute regression target (net_pnl)
        reward = df_valid['vwap_width_atr'] * df_valid['atr'] * SHARES_PER_TRADE
        risk = stop_atr * df_valid['atr'] * SHARES_PER_TRADE
        gross_pnl = np.where(y_raw.values == 1, reward.values, -risk.values)
        net_pnl = gross_pnl - costs
        y_reg = pd.Series(net_pnl, index=df_valid.index)

        # Split
        df_valid['year'] = pd.to_datetime(df_valid['datetime']).dt.year
        train_mask = df_valid['year'] < TEST_YEAR
        test_mask = df_valid['year'] >= TEST_YEAR

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y_reg[train_mask]
        y_raw_test = y_raw[test_mask]

        # Train
        rf = RandomForestRegressor(**RF_REG_PARAMS)
        rf.fit(X_train, y_train)
        pred_test = rf.predict(X_test)

        # Test set enrichment
        df_test = df_valid[test_mask].copy()
        df_test['pred_pnl'] = pred_test
        df_test['label'] = y_raw_test.values
        df_test['reward_dollars'] = (df_test['vwap_width_atr'] * df_test['atr'] * SHARES_PER_TRADE).values
        df_test['risk_dollars'] = (stop_atr * df_test['atr'] * SHARES_PER_TRADE).values
        df_test['actual_net_pnl'] = np.where(
            df_test['label'] == 1,
            df_test['reward_dollars'] - costs,
            -df_test['risk_dollars'] - costs,
        )
        df_test['is_long'] = df_test['is_long_setup'].astype(bool)
        df_test['month'] = pd.to_datetime(df_test['datetime']).dt.to_period('M')
        df_test['per_trade_rr'] = df_test['vwap_width_atr'] / stop_atr

        # Feature importance
        imp = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_})
        imp = imp.sort_values('importance', ascending=False)
        print("\n  Top 10 features:")
        for _, r in imp.head(10).iterrows():
            print(f"    {r['feature']:30s}  {r['importance']:.4f}")

        # ---- Sweep top_n ----
        print(f"\n  --- Top-N sweep ---")
        print(f"  {'top_n':>8s}  {'trades':>6s}  {'WR':>6s}  {'NetPnL':>10s}  {'AvgPnL':>8s}  {'LongPct':>7s}  {'AvgDist':>7s}  {'AvgRR':>6s}")
        for top_n in [50, 100, 200, 300, 500, 750, 1000, 2000, 5000]:
            order = np.argsort(pred_test)[::-1]
            idx = order[:min(top_n, len(order))]
            sel = df_test.iloc[idx]
            wr = sel['label'].mean()
            net = sel['actual_net_pnl'].sum()
            avg = sel['actual_net_pnl'].mean()
            long_pct = sel['is_long'].mean() * 100
            avg_dist = sel['vwap_width_atr'].mean()
            avg_rr = sel['per_trade_rr'].mean()
            print(f"  {top_n:8d}  {len(sel):6d}  {wr:5.1%}  ${net:>9,.0f}  ${avg:>7.1f}  {long_pct:5.1f}%  {avg_dist:6.2f}  {avg_rr:5.1f}")

        # ---- Long-only vs Short-only ----
        print(f"\n  --- Direction filter (top 500) ---")
        order = np.argsort(pred_test)[::-1][:500]
        sel = df_test.iloc[order]
        for label, mask_fn in [
            ("ALL", lambda s: pd.Series(True, index=s.index)),
            ("LONG only", lambda s: s['is_long']),
            ("SHORT only", lambda s: ~s['is_long']),
        ]:
            m = mask_fn(sel)
            sub = sel[m]
            if len(sub) == 0:
                print(f"  {label:15s}  n=0")
                continue
            wr = sub['label'].mean()
            net = sub['actual_net_pnl'].sum()
            avg = sub['actual_net_pnl'].mean()
            print(f"  {label:15s}  n={len(sub):4d}  WR={wr:5.1%}  NetPnL=${net:>9,.0f}  Avg=${avg:>7.1f}")

        # ---- Min distance filter ----
        print(f"\n  --- Min distance filter (top 500, then filter) ---")
        order = np.argsort(pred_test)[::-1][:500]
        sel = df_test.iloc[order]
        for min_dist in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            sub = sel[sel['vwap_width_atr'] >= min_dist]
            if len(sub) == 0:
                print(f"  dist>={min_dist:.1f} ATR:  n=0")
                continue
            wr = sub['label'].mean()
            net = sub['actual_net_pnl'].sum()
            avg = sub['actual_net_pnl'].mean()
            print(f"  dist>={min_dist:.1f} ATR:  n={len(sub):4d}  WR={wr:5.1%}  NetPnL=${net:>9,.0f}  Avg=${avg:>7.1f}")

        # ---- Combined: Long + min dist (top 500) ----
        print(f"\n  --- Combined: LONG + min dist (top 500) ---")
        for min_dist in [0.0, 0.5, 1.0, 1.5, 2.0]:
            sub = sel[sel['is_long'] & (sel['vwap_width_atr'] >= min_dist)]
            if len(sub) == 0:
                print(f"  LONG + dist>={min_dist:.1f}:  n=0")
                continue
            wr = sub['label'].mean()
            net = sub['actual_net_pnl'].sum()
            avg = sub['actual_net_pnl'].mean()
            print(f"  LONG + dist>={min_dist:.1f}:  n={len(sub):4d}  WR={wr:5.1%}  NetPnL=${net:>9,.0f}  Avg=${avg:>7.1f}")

        # ---- Monthly P&L (top 500 overall) ----
        print(f"\n  --- Monthly P&L (top 500) ---")
        for month, mdf in sel.groupby('month'):
            wr = mdf['label'].mean()
            net = mdf['actual_net_pnl'].sum()
            n = len(mdf)
            print(f"  {month}:  n={n:3d}  WR={wr:5.1%}  NetPnL=${net:>8,.0f}")

        # ---- Decile analysis ----
        print(f"\n  --- Predicted P&L decile analysis (all test data) ---")
        df_test['decile'] = pd.qcut(df_test['pred_pnl'], 10, labels=False, duplicates='drop')
        for d in sorted(df_test['decile'].unique()):
            sub = df_test[df_test['decile'] == d]
            wr = sub['label'].mean()
            net = sub['actual_net_pnl'].sum()
            avg_pred = sub['pred_pnl'].mean()
            avg_actual = sub['actual_net_pnl'].mean()
            print(f"  D{d}: n={len(sub):5d}  WR={wr:5.1%}  AvgPred=${avg_pred:>7.1f}  AvgActual=${avg_actual:>7.1f}  TotalNet=${net:>10,.0f}")


if __name__ == "__main__":
    main()
