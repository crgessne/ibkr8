"""
Deep-dive on the ONE profitable configuration found:
  LONG only + dist > 1 ATR + 1 trade/day (best RF proba) = +$3,110

Questions:
1. Is this statistically significant or lucky?
2. What does the equity curve look like?
3. Can we improve it with better stop sizing?
4. What about multiple stops?
5. What if we use dist > 1.5 or > 2 ATR?
6. What's the win rate, avg win, avg loss breakdown?
7. Does walk-forward (multi-year) hold up?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from label_generator import LabelConfig, generate_labels

DATA_FILE = Path("data/tsla_5min_10years.csv")
SHARES = 100
COST_RT = 3.0

def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ============================================================================
# LOAD + INDICATORS
# ============================================================================
flush_print("Loading data...")
df = pd.read_csv(DATA_FILE)
df['datetime'] = pd.to_datetime(df['time'], utc=True)
df['date'] = df['datetime'].dt.date

# ATR
high, low, close = df['high'], df['low'], df['close']
tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

# VWAP
tp = (df['high'] + df['low'] + df['close']) / 3
pv = tp * df['volume']
df['vwap'] = df.groupby('date').apply(
    lambda g: pv.loc[g.index].cumsum() / df.loc[g.index, 'volume'].cumsum()
).reset_index(level=0, drop=True)

# Features
df['vwap_width_atr'] = abs(df['close'] - df['vwap']) / df['atr']
df['price_to_vwap_atr'] = (df['close'] - df['vwap']) / df['atr']
df['is_long_setup'] = (df['close'] < df['vwap']).astype(int)
df['vwap_slope'] = df['vwap'].diff(1)
df['vwap_slope_5'] = df['vwap'].diff(5)
df['vwap_helping'] = np.where(df['is_long_setup'], df['vwap_slope'] < 0, df['vwap_slope'] > 0).astype(int)
df['rel_vol'] = df['volume'] / df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['volume'].shift(1)
df['vol_at_extension'] = df['volume'] / df['volume'].rolling(5).mean()

delta = df['close'].diff()
gain = delta.where(delta > 0, 0.0)
loss_s = (-delta).where(delta < 0, 0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss_s.rolling(14).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df['rsi'] = 100 - (100 / (1 + rs))
df['rsi_slope'] = df['rsi'].diff(3)
df['rsi_extreme'] = ((df['rsi'] < 30) | (df['rsi'] > 70)).astype(int)

df['bar_range_atr'] = (df['high'] - df['low']) / df['atr']
df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
df['crossed_vwap'] = (df['is_long_setup'] != df['is_long_setup'].shift(1)).astype(int)
df['bars_from_vwap'] = df.groupby((df['crossed_vwap'] == 1).cumsum()).cumcount()

# Enhanced features
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['minutes_into_session'] = ((df['hour'] - 9) * 60 + df['minute'] - 30).clip(0, 390)
df['cum_vol_today'] = df.groupby('date')['volume'].cumsum()
df['total_vol_today'] = df.groupby('date')['volume'].transform('sum')
df['vol_pct_complete'] = df['cum_vol_today'] / df['total_vol_today']
df['vwap_crosses_today'] = df.groupby('date')['crossed_vwap'].cumsum()
df['day_high'] = df.groupby('date')['high'].cummax()
df['day_low'] = df.groupby('date')['low'].cummin()
df['day_range_atr'] = (df['day_high'] - df['day_low']) / df['atr']
df['pct_of_day_range'] = np.where(df['day_high'] > df['day_low'], (df['close'] - df['day_low']) / (df['day_high'] - df['day_low']), 0.5)
df['vwap_in_day_range'] = np.where(df['day_high'] > df['day_low'], (df['vwap'] - df['day_low']) / (df['day_high'] - df['day_low']), 0.5)
df['momentum_3bar_atr'] = (df['close'] - df['close'].shift(3)) / df['atr']
df['momentum_6bar_atr'] = (df['close'] - df['close'].shift(6)) / df['atr']
df['bar_reverting'] = np.where(df['is_long_setup'], (df['close'] > df['open']).astype(int), (df['close'] < df['open']).astype(int))
df['consecutive_same_side'] = df.groupby((df['is_long_setup'] != df['is_long_setup'].shift(1)).cumsum()).cumcount() + 1
daily_open = df.groupby('date')['open'].transform('first')
df['open_vs_vwap_atr'] = (daily_open - df['vwap']) / df['atr']
df['prior_bar_toward_vwap'] = np.where(df['is_long_setup'], (df['close'].shift(1) > df['close'].shift(2)).astype(float), (df['close'].shift(1) < df['close'].shift(2)).astype(float))
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema20_slope_atr'] = (df['ema20'] - df['ema20'].shift(5)) / df['atr']
df['extension_speed'] = df['vwap_width_atr'] / (df['bars_from_vwap'] + 1)

features = [
    'vwap_width_atr', 'price_to_vwap_atr', 'is_long_setup',
    'vwap_slope', 'vwap_slope_5', 'vwap_helping',
    'rel_vol', 'vol_ratio', 'vol_at_extension',
    'rsi', 'rsi_slope', 'rsi_extreme',
    'bar_range_atr', 'close_position', 'crossed_vwap', 'bars_from_vwap',
    'minutes_into_session', 'vol_pct_complete', 'vwap_crosses_today',
    'day_range_atr', 'pct_of_day_range', 'vwap_in_day_range',
    'momentum_3bar_atr', 'momentum_6bar_atr',
    'bar_reverting', 'consecutive_same_side',
    'open_vs_vwap_atr', 'prior_bar_toward_vwap',
    'ema20_slope_atr', 'extension_speed',
]

# ============================================================================
# TEST ACROSS MULTIPLE STOPS AND DISTANCE THRESHOLDS
# ============================================================================
flush_print("\nGenerating labels for multiple stops...")
stop_atrs = [0.25, 0.5, 0.75, 1.0, 1.5]
config = LabelConfig(stop_atrs=stop_atrs)
df = generate_labels(df, config)

df['year'] = df['datetime'].dt.year

# ============================================================================
# WALK-FORWARD: Train on years < test_year, test on test_year
# ============================================================================
flush_print("\n" + "="*80)
flush_print("WALK-FORWARD TEST: 1 LONG/day, best RF proba, dist > threshold")
flush_print("="*80)

test_years = [2020, 2021, 2022, 2023, 2024, 2025]
dist_thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

flush_print(f"\n{'Stop':>5} {'Dist>':>6} {'Year':>5} {'N':>5} {'WR':>6} {'AvgWin':>8} {'AvgLoss':>8} {'AvgNet':>8} {'Total':>10}")
flush_print("-"*75)

# Aggregate results
summary = []

for stop_atr in stop_atrs:
    label_col = f"label_s{stop_atr}".replace(".", "_")
    valid = df[label_col].notna()
    df_v = df[valid].copy()

    for dist_thresh in dist_thresholds:
        yearly_results = []

        for test_year in test_years:
            train_mask = df_v['year'] < test_year
            test_mask = df_v['year'] == test_year

            if train_mask.sum() < 500 or test_mask.sum() < 100:
                continue

            X = df_v[features].replace([np.inf, -np.inf], np.nan).fillna(0)
            y = df_v[label_col].astype(int)

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            rf = RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=50,
                max_features='sqrt', random_state=42, n_jobs=-1, class_weight='balanced'
            )
            rf.fit(X_train, y_train)
            proba = rf.predict_proba(X_test)[:, 1]

            test_df = df_v[test_mask].copy()
            test_df['rf_proba'] = proba
            test_df['label'] = y_test.values

            # Filter: LONG only + dist > threshold
            filt = test_df[(test_df['is_long_setup'] == 1) & (test_df['vwap_width_atr'] > dist_thresh)].copy()
            if len(filt) == 0:
                continue

            # 1 trade per day: best RF proba
            filt['rank_in_day'] = filt.groupby('date')['rf_proba'].rank(ascending=False, method='first')
            best = filt[filt['rank_in_day'] == 1].copy()

            # P&L
            best['reward_dollars'] = best['vwap_width_atr'] * best['atr'] * SHARES
            best['risk_dollars'] = stop_atr * best['atr'] * SHARES
            best['gross_pnl'] = np.where(best['label'] == 1, best['reward_dollars'], -best['risk_dollars'])
            best['net_pnl'] = best['gross_pnl'] - COST_RT

            n = len(best)
            wr = best['label'].mean()
            wins = best[best['net_pnl'] > 0]
            losses = best[best['net_pnl'] <= 0]
            avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
            avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0
            avg_net = best['net_pnl'].mean()
            total_net = best['net_pnl'].sum()

            flush_print(f"{stop_atr:>5.2f} {dist_thresh:>6.2f} {test_year:>5} {n:>5} {wr*100:>5.1f}% {avg_win:>8.1f} {avg_loss:>8.1f} {avg_net:>+8.1f} {total_net:>+10,.0f}")

            yearly_results.append({
                'year': test_year, 'n': n, 'wr': wr, 'avg_net': avg_net,
                'total_net': total_net, 'avg_win': avg_win, 'avg_loss': avg_loss,
            })

        if len(yearly_results) > 0:
            all_n = sum(r['n'] for r in yearly_results)
            all_total = sum(r['total_net'] for r in yearly_results)
            all_wr = sum(r['n'] * r['wr'] for r in yearly_results) / all_n if all_n > 0 else 0
            positive_years = sum(1 for r in yearly_results if r['total_net'] > 0)

            summary.append({
                'stop': stop_atr, 'dist': dist_thresh,
                'total_trades': all_n, 'total_pnl': all_total,
                'avg_pnl_per_trade': all_total / all_n if all_n > 0 else 0,
                'avg_wr': all_wr,
                'years_tested': len(yearly_results),
                'years_profitable': positive_years,
            })

# ============================================================================
# SUMMARY TABLE
# ============================================================================
flush_print("\n" + "="*80)
flush_print("SUMMARY: TOTAL P&L ACROSS ALL WALK-FORWARD YEARS")
flush_print("="*80)
flush_print(f"\n{'Stop':>5} {'Dist>':>6} {'Trades':>7} {'WR':>6} {'Avg/Tr':>8} {'Total P&L':>12} {'Yrs+':>5}/{' Yrs':>4}")
flush_print("-"*60)

summary.sort(key=lambda x: x['total_pnl'], reverse=True)
for s in summary:
    marker = " ***" if s['total_pnl'] > 0 else ""
    flush_print(
        f"{s['stop']:>5.2f} {s['dist']:>6.2f} {s['total_trades']:>7} "
        f"{s['avg_wr']*100:>5.1f}% {s['avg_pnl_per_trade']:>+8.1f} "
        f"{s['total_pnl']:>+12,.0f} {s['years_profitable']:>5}/{s['years_tested']:>4}{marker}"
    )

# ============================================================================
# DEEP-DIVE on best config: equity curve, monthly breakdown
# ============================================================================
if len(summary) > 0:
    best_cfg = summary[0]
    flush_print(f"\n{'='*80}")
    flush_print(f"DEEP-DIVE: Best config = stop {best_cfg['stop']} ATR, dist > {best_cfg['dist']} ATR")
    flush_print(f"{'='*80}")

    stop_atr = best_cfg['stop']
    dist_thresh = best_cfg['dist']
    label_col = f"label_s{stop_atr}".replace(".", "_")

    valid = df[label_col].notna()
    df_v = df[valid].copy()

    # Full walk-forward equity curve
    all_trades = []
    for test_year in test_years:
        train_mask = df_v['year'] < test_year
        test_mask = df_v['year'] == test_year
        if train_mask.sum() < 500 or test_mask.sum() < 100:
            continue

        X = df_v[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df_v[label_col].astype(int)
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            max_features='sqrt', random_state=42, n_jobs=-1, class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        proba = rf.predict_proba(X_test)[:, 1]

        test_df = df_v[test_mask].copy()
        test_df['rf_proba'] = proba
        test_df['label'] = y_test.values

        filt = test_df[(test_df['is_long_setup'] == 1) & (test_df['vwap_width_atr'] > dist_thresh)].copy()
        if len(filt) == 0:
            continue
        filt['rank_in_day'] = filt.groupby('date')['rf_proba'].rank(ascending=False, method='first')
        best = filt[filt['rank_in_day'] == 1].copy()
        best['reward_dollars'] = best['vwap_width_atr'] * best['atr'] * SHARES
        best['risk_dollars'] = stop_atr * best['atr'] * SHARES
        best['gross_pnl'] = np.where(best['label'] == 1, best['reward_dollars'], -best['risk_dollars'])
        best['net_pnl'] = best['gross_pnl'] - COST_RT
        best['month'] = pd.to_datetime(best['datetime']).dt.to_period('M')
        all_trades.append(best[['datetime', 'date', 'label', 'net_pnl', 'reward_dollars', 'risk_dollars', 'rf_proba', 'vwap_width_atr', 'month']])

    if len(all_trades) > 0:
        trades_df = pd.concat(all_trades).sort_values('datetime')
        trades_df['cum_pnl'] = trades_df['net_pnl'].cumsum()

        flush_print(f"\n  Total trades: {len(trades_df)}")
        flush_print(f"  Total P&L: ${trades_df['net_pnl'].sum():+,.0f}")
        flush_print(f"  Avg P&L/trade: ${trades_df['net_pnl'].mean():+.1f}")
        flush_print(f"  Win rate: {trades_df['label'].mean()*100:.1f}%")
        flush_print(f"  Avg win: ${trades_df[trades_df['net_pnl']>0]['net_pnl'].mean():.1f}")
        flush_print(f"  Avg loss: ${trades_df[trades_df['net_pnl']<=0]['net_pnl'].mean():.1f}")
        flush_print(f"  Max drawdown: ${trades_df['cum_pnl'].min():,.0f}")
        flush_print(f"  Peak equity: ${trades_df['cum_pnl'].max():,.0f}")

        # Monthly breakdown
        flush_print(f"\n  Monthly P&L:")
        monthly = trades_df.groupby('month').agg(
            n=('net_pnl', 'count'),
            total=('net_pnl', 'sum'),
            wr=('label', 'mean'),
        )
        for m, row in monthly.iterrows():
            marker = "+" if row['total'] > 0 else ""
            flush_print(f"    {str(m):>8}: n={int(row['n']):>3} WR={row['wr']*100:>5.1f}% P&L=${row['total']:>+8,.0f}")

        # Equity curve checkpoints
        flush_print(f"\n  Equity curve (every 50 trades):")
        for i in range(0, len(trades_df), 50):
            flush_print(f"    Trade {i:>4}: cum_pnl = ${trades_df.iloc[i]['cum_pnl']:>+10,.0f}")
        flush_print(f"    Trade {len(trades_df)-1:>4}: cum_pnl = ${trades_df.iloc[-1]['cum_pnl']:>+10,.0f}")

        # Save trades
        out_path = Path("data/profitable_config_trades.csv")
        trades_df.to_csv(out_path, index=False)
        flush_print(f"\n  Saved trades to {out_path}")

# ============================================================================
# ALSO TEST: What if we allow BOTH long AND short, but max 1/day, dist > X?
# ============================================================================
flush_print("\n" + "="*80)
flush_print("COMPARISON: LONG-ONLY vs BOTH DIRECTIONS (1 trade/day, walk-forward)")
flush_print("="*80)

for stop_atr in [0.25, 0.5, 0.75, 1.0]:
    label_col = f"label_s{stop_atr}".replace(".", "_")
    valid = df[label_col].notna()
    df_v = df[valid].copy()

    for dist_thresh in [0.75, 1.0, 1.5]:
        for direction in ['LONG', 'BOTH']:
            total_pnl = 0
            total_n = 0
            total_wins = 0

            for test_year in test_years:
                train_mask = df_v['year'] < test_year
                test_mask = df_v['year'] == test_year
                if train_mask.sum() < 500 or test_mask.sum() < 100:
                    continue

                X = df_v[features].replace([np.inf, -np.inf], np.nan).fillna(0)
                y = df_v[label_col].astype(int)
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]

                rf = RandomForestClassifier(
                    n_estimators=200, max_depth=8, min_samples_leaf=50,
                    max_features='sqrt', random_state=42, n_jobs=-1, class_weight='balanced'
                )
                rf.fit(X_train, y_train)
                proba = rf.predict_proba(X_test)[:, 1]

                test_df = df_v[test_mask].copy()
                test_df['rf_proba'] = proba
                test_df['label'] = y_test.values

                if direction == 'LONG':
                    filt = test_df[(test_df['is_long_setup'] == 1) & (test_df['vwap_width_atr'] > dist_thresh)].copy()
                else:
                    filt = test_df[test_df['vwap_width_atr'] > dist_thresh].copy()

                if len(filt) == 0:
                    continue
                filt['rank_in_day'] = filt.groupby('date')['rf_proba'].rank(ascending=False, method='first')
                best = filt[filt['rank_in_day'] == 1].copy()
                best['reward_dollars'] = best['vwap_width_atr'] * best['atr'] * SHARES
                best['risk_dollars'] = stop_atr * best['atr'] * SHARES
                best['gross_pnl'] = np.where(best['label'] == 1, best['reward_dollars'], -best['risk_dollars'])
                best['net_pnl'] = best['gross_pnl'] - COST_RT

                total_pnl += best['net_pnl'].sum()
                total_n += len(best)
                total_wins += best['label'].sum()

            if total_n > 0:
                wr = total_wins / total_n
                avg = total_pnl / total_n
                marker = " ***" if total_pnl > 0 else ""
                flush_print(f"  stop={stop_atr:.2f} dist>{dist_thresh:.2f} {direction:>5}: n={total_n:>5} WR={wr*100:>5.1f}% avg=${avg:>+7.1f} total=${total_pnl:>+10,.0f}{marker}")

flush_print("\nDONE.")
