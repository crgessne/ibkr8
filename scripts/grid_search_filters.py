"""
Multi-filter grid search: find the best combination of filters that produces
a robust positive-EV subset on TSLA 5-min VWAP reversion.

Filters:
  - Stop ATR: 0.25, 0.35
  - Top-N: 100, 200, 300, 500
  - Direction: all, long, short
  - Min distance: 0.0, 0.5, 1.0, 1.5
  - Session phase: all, morning (0-30min), mid-morning (30-120), afternoon (120-270), close (270+)
  - VWAP slope direction: all, slope_helping, slope_opposing

Also computes a simple "robustness" check: split the test period in half
(2024 vs 2025+) and report P&L on each half.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import io
_out_path = Path(__file__).parent.parent / "_grid_search_out.txt"
_out_file = open(_out_path, "w", encoding="utf-8")
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
from master_pipeline import (
    load_and_validate_data, calculate_core_indicators, get_feature_columns,
    SHARES_PER_TRADE, COMMISSION_PER_SHARE, SLIPPAGE_PER_SHARE,
    RF_REG_PARAMS, DATA_FILE, TEST_YEAR,
)

def main():
    print("=" * 100)
    print("MULTI-FILTER GRID SEARCH FOR POSITIVE-EV VWAP REVERSION")
    print("=" * 100)

    df = load_and_validate_data(DATA_FILE)
    df = calculate_core_indicators(df, verbose=False)
    features = get_feature_columns(df)

    tight_stops = [0.25, 0.35]
    config = LabelConfig(stop_atrs=tight_stops)
    df = generate_labels(df, config)
    print(f"Labels generated. Features: {len(features)}")

    costs = 2.0 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * SHARES_PER_TRADE

    all_rows = []

    for stop_atr in tight_stops:
        print(f"\n{'=' * 80}")
        print(f"  Training regressor for stop={stop_atr} ATR ...")

        label_col = f"label_s{stop_atr}".replace(".", "_")
        valid = df[label_col].notna()
        df_valid = df[valid].copy()

        X = df_valid[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_raw = df_valid[label_col].astype(int)

        reward = df_valid['vwap_width_atr'] * df_valid['atr'] * SHARES_PER_TRADE
        risk = stop_atr * df_valid['atr'] * SHARES_PER_TRADE
        gross_pnl = np.where(y_raw.values == 1, reward.values, -risk.values)
        net_pnl_vals = gross_pnl - costs
        y_reg = pd.Series(net_pnl_vals, index=df_valid.index)

        df_valid['year'] = pd.to_datetime(df_valid['datetime']).dt.year
        train_mask = df_valid['year'] < TEST_YEAR
        test_mask = df_valid['year'] >= TEST_YEAR

        rf = RandomForestRegressor(**RF_REG_PARAMS)
        rf.fit(X[train_mask], y_reg[train_mask])
        pred_test = rf.predict(X[test_mask])
        print(f"  Trained. Test samples: {test_mask.sum():,}")

        df_test = df_valid[test_mask].copy()
        df_test['pred_pnl'] = pred_test
        df_test['label'] = y_raw[test_mask].values
        df_test['reward_dollars'] = (df_test['vwap_width_atr'] * df_test['atr'] * SHARES_PER_TRADE).values
        df_test['risk_dollars'] = (stop_atr * df_test['atr'] * SHARES_PER_TRADE).values
        df_test['actual_net_pnl'] = np.where(
            df_test['label'] == 1,
            df_test['reward_dollars'] - costs,
            -df_test['risk_dollars'] - costs,
        )
        df_test['is_long'] = df_test['is_long_setup'].astype(bool)
        df_test['half'] = np.where(df_test['year'] == 2024, 'H1_2024', 'H2_2025+')

        # Grid
        top_ns = [100, 200, 300, 500]
        directions = ['all', 'long', 'short']
        min_dists = [0.0, 0.5, 1.0, 1.5]
        phases = ['all', 'morning', 'mid_morning', 'afternoon', 'close']
        vwap_slopes = ['all', 'helping', 'opposing']

        phase_map = {
            'all': lambda s: pd.Series(True, index=s.index),
            'morning': lambda s: s['session_phase'] == 0,
            'mid_morning': lambda s: s['session_phase'] == 1,
            'afternoon': lambda s: s['session_phase'] == 2,
            'close': lambda s: s['session_phase'] == 3,
        }
        dir_map = {
            'all': lambda s: pd.Series(True, index=s.index),
            'long': lambda s: s['is_long'],
            'short': lambda s: ~s['is_long'],
        }
        slope_map = {
            'all': lambda s: pd.Series(True, index=s.index),
            'helping': lambda s: s['vwap_helping'] == 1,
            'opposing': lambda s: s['vwap_helping'] == 0,
        }

        for top_n in top_ns:
            # Select top-N by predicted P&L
            order = np.argsort(pred_test)[::-1]
            idx = order[:min(top_n, len(order))]
            sel = df_test.iloc[idx].copy()

            for direction in directions:
                for min_dist in min_dists:
                    for phase in phases:
                        for vslope in vwap_slopes:
                            m = (
                                dir_map[direction](sel) &
                                (sel['vwap_width_atr'] >= min_dist) &
                                phase_map[phase](sel) &
                                slope_map[vslope](sel)
                            )
                            sub = sel[m]
                            n = len(sub)
                            if n < 10:
                                continue

                            wr = sub['label'].mean()
                            net = sub['actual_net_pnl'].sum()
                            avg = sub['actual_net_pnl'].mean()

                            # Half-period robustness
                            h1 = sub[sub['half'] == 'H1_2024']
                            h2 = sub[sub['half'] == 'H2_2025+']
                            h1_net = h1['actual_net_pnl'].sum() if len(h1) > 0 else 0
                            h2_net = h2['actual_net_pnl'].sum() if len(h2) > 0 else 0
                            both_positive = (h1_net > 0) and (h2_net > 0)

                            all_rows.append({
                                'stop_atr': stop_atr,
                                'top_n': top_n,
                                'direction': direction,
                                'min_dist': min_dist,
                                'phase': phase,
                                'vwap_slope': vslope,
                                'n_trades': n,
                                'win_rate': wr,
                                'total_net_pnl': net,
                                'avg_net_pnl': avg,
                                'h1_2024_pnl': h1_net,
                                'h2_2025_pnl': h2_net,
                                'both_halves_positive': both_positive,
                            })

    results = pd.DataFrame(all_rows)

    # ---- Report ----
    print(f"\n\nTotal filter combos evaluated: {len(results):,}")

    # Top 20 by total net P&L
    print(f"\n{'=' * 100}")
    print("TOP 20 CONFIGS BY TOTAL NET P&L")
    print(f"{'=' * 100}")
    top = results.sort_values('total_net_pnl', ascending=False).head(20)
    for _, r in top.iterrows():
        robust = "ROBUST" if r['both_halves_positive'] else "  weak"
        print(
            f"  stop={r['stop_atr']:.2f} top_n={int(r['top_n']):4d} dir={r['direction']:6s} "
            f"dist>={r['min_dist']:.1f} phase={r['phase']:12s} slope={r['vwap_slope']:9s} | "
            f"n={int(r['n_trades']):4d} WR={r['win_rate']:5.1%} NetPnL=${r['total_net_pnl']:>9,.0f} "
            f"Avg=${r['avg_net_pnl']:>7.1f} | H1=${r['h1_2024_pnl']:>8,.0f} H2=${r['h2_2025_pnl']:>8,.0f} [{robust}]"
        )

    # Top 20 ROBUST only (both halves positive)
    robust_df = results[results['both_halves_positive']].sort_values('total_net_pnl', ascending=False)
    print(f"\n{'=' * 100}")
    print(f"TOP 20 ROBUST CONFIGS (both halves positive) — {len(robust_df)} total")
    print(f"{'=' * 100}")
    for _, r in robust_df.head(20).iterrows():
        print(
            f"  stop={r['stop_atr']:.2f} top_n={int(r['top_n']):4d} dir={r['direction']:6s} "
            f"dist>={r['min_dist']:.1f} phase={r['phase']:12s} slope={r['vwap_slope']:9s} | "
            f"n={int(r['n_trades']):4d} WR={r['win_rate']:5.1%} NetPnL=${r['total_net_pnl']:>9,.0f} "
            f"Avg=${r['avg_net_pnl']:>7.1f} | H1=${r['h1_2024_pnl']:>8,.0f} H2=${r['h2_2025_pnl']:>8,.0f}"
        )

    # Top by avg P&L per trade (min 20 trades)
    enough = results[results['n_trades'] >= 20].sort_values('avg_net_pnl', ascending=False)
    print(f"\n{'=' * 100}")
    print(f"TOP 20 BY AVG P&L PER TRADE (min 20 trades)")
    print(f"{'=' * 100}")
    for _, r in enough.head(20).iterrows():
        robust = "ROBUST" if r['both_halves_positive'] else "  weak"
        print(
            f"  stop={r['stop_atr']:.2f} top_n={int(r['top_n']):4d} dir={r['direction']:6s} "
            f"dist>={r['min_dist']:.1f} phase={r['phase']:12s} slope={r['vwap_slope']:9s} | "
            f"n={int(r['n_trades']):4d} WR={r['win_rate']:5.1%} NetPnL=${r['total_net_pnl']:>9,.0f} "
            f"Avg=${r['avg_net_pnl']:>7.1f} | H1=${r['h1_2024_pnl']:>8,.0f} H2=${r['h2_2025_pnl']:>8,.0f} [{robust}]"
        )

    # Save full results
    out_csv = Path(__file__).parent.parent / "data" / "grid_search_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved {len(results)} rows to {out_csv}")

    _out_file.close()


if __name__ == "__main__":
    main()
