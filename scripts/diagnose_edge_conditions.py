"""
Diagnose WHERE the edge in VWAP reversion actually lives.

VWAP reversion IS real — the question is:
1. The current strategy enters EVERY bar → dilutes signal with noise
2. Entry at close (not at extremes) → often entering AFTER reversion has started
3. Fixed VWAP target → ignores that VWAP moves toward price intraday
4. Symmetric long/short → ignores TSLA's long bias

This script tests specific conditions to find WHERE a real edge exists.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

# Reuse the pipeline's data loader and indicators
sys.path.insert(0, str(Path(__file__).parent))

DATA_FILE = Path("data/tsla_5min_10years.csv")
SHARES = 100
COST_RT = 2 * (0.005 + 0.01) * SHARES  # $3/trade

def load_data():
    df = pd.read_csv(DATA_FILE)
    df['datetime'] = pd.to_datetime(df['time'], utc=True)
    df['date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year

    # ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # VWAP
    tp = (df['high'] + df['low'] + df['close']) / 3
    pv = tp * df['volume']
    df['vwap'] = df.groupby('date').apply(
        lambda g: pv.loc[g.index].cumsum() / df.loc[g.index, 'volume'].cumsum()
    ).reset_index(level=0, drop=True)

    # Derived
    df['vwap_dist_atr'] = abs(df['close'] - df['vwap']) / df['atr']
    df['price_to_vwap_atr'] = (df['close'] - df['vwap']) / df['atr']
    df['is_long'] = df['close'] < df['vwap']

    # Volume
    df['rel_vol'] = df['volume'] / df['volume'].rolling(20).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss_s = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss_s.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bar position in day
    df['bar_of_day'] = df.groupby('date').cumcount()
    bars_per_day = df.groupby('date')['close'].transform('count')
    df['bars_in_day'] = bars_per_day

    # Consecutive bars on same side of VWAP
    side = df['is_long'].astype(int)
    side_change = (side != side.shift(1)).cumsum()
    df['bars_same_side'] = df.groupby(side_change).cumcount() + 1

    # Close position in bar (0=at low, 1=at high)
    bar_range = df['high'] - df['low']
    df['close_pos'] = np.where(bar_range > 0, (df['close'] - df['low']) / bar_range, 0.5)

    # Prior bar momentum (did we just move AWAY from VWAP?)
    df['prev_close'] = df['close'].shift(1)
    df['prev_vwap_dist'] = abs(df['prev_close'] - df['vwap'].shift(1)) / df['atr'].shift(1)
    df['expanding_from_vwap'] = df['vwap_dist_atr'] > df['prev_vwap_dist']

    return df


def simulate_trades(df, mask, stop_atr, label=''):
    """Simulate flat-to-flat trades for bars matching mask, return summary."""
    df_sig = df[mask].copy()
    if len(df_sig) == 0:
        return None

    trades = []
    flat_after = None

    for idx, row in df_sig.iterrows():
        entry_dt = row['datetime']
        if flat_after is not None and entry_dt <= flat_after:
            continue

        entry_price = float(row['close'])
        entry_atr = float(row['atr'])
        is_long = bool(row['is_long'])
        target_vwap = float(row['vwap'])
        entry_date = row['date']

        stop_dist = stop_atr * entry_atr
        stop_price = entry_price - stop_dist if is_long else entry_price + stop_dist

        # Walk forward same day
        df_day = df[df['date'] == entry_date]
        started = False
        exit_price = entry_price
        exit_reason = 'eod'

        for idx2, r2 in df_day.iterrows():
            if not started:
                if idx2 != idx:
                    continue
                started = True
                continue

            hi, lo = float(r2['high']), float(r2['low'])
            if is_long:
                if lo <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'stop'
                    break
                if hi >= target_vwap:
                    exit_price = target_vwap
                    exit_reason = 'vwap'
                    break
            else:
                if hi >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'stop'
                    break
                if lo <= target_vwap:
                    exit_price = target_vwap
                    exit_reason = 'vwap'
                    break
            exit_price = float(r2['close'])

        gross = (exit_price - entry_price) * SHARES if is_long else (entry_price - exit_price) * SHARES
        net = gross - COST_RT
        flat_after = pd.to_datetime(df.loc[idx2 if started else idx, 'datetime']) if started else entry_dt

        trades.append({
            'net_pnl': net,
            'gross_pnl': gross,
            'exit_reason': exit_reason,
            'is_long': is_long,
            'vwap_dist_atr': float(row['vwap_dist_atr']),
            'rsi': float(row['rsi']),
            'bar_of_day': int(row['bar_of_day']),
        })

    if len(trades) == 0:
        return None

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wr = (tdf['net_pnl'] > 0).mean()
    total = tdf['net_pnl'].sum()
    avg = tdf['net_pnl'].mean()
    vwap_pct = (tdf['exit_reason'] == 'vwap').mean()

    return {
        'label': label,
        'n_trades': n,
        'win_rate': wr,
        'vwap_touch_rate': vwap_pct,
        'total_net_pnl': total,
        'avg_net_pnl': avg,
        'avg_vwap_dist': tdf['vwap_dist_atr'].mean(),
    }


def main():
    print("Loading data...")
    df = load_data()

    # Focus on 2024 test year
    df_test = df[(df['year'] == 2024) & df['atr'].notna() & (df['atr'] > 0)].copy()
    print(f"2024 test bars: {len(df_test):,}")

    # We'll test stop = 0.5 ATR as the baseline
    stop_atr = 0.5

    print(f"\n{'='*90}")
    print(f"WHERE DOES VWAP REVERSION EDGE ACTUALLY LIVE?  (stop={stop_atr} ATR, 2024)")
    print(f"{'='*90}")

    results = []
    # =====================================================================
    # TEST 1: BASELINE — enter every bar (current strategy)
    # =====================================================================
    test_mask = df.index.isin(df_test.index)
    r = simulate_trades(df, test_mask, stop_atr, 'BASELINE: every bar')
    if r: results.append(r)

    # =====================================================================
    # TEST 2: DISTANCE FROM VWAP — only enter when far enough
    # =====================================================================
    for min_dist in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        mask = df['vwap_dist_atr'] >= min_dist
        r = simulate_trades(df, mask & test_mask, stop_atr, f'dist >= {min_dist} ATR')
        if r: results.append(r)

    # =====================================================================
    # TEST 3: LONG ONLY (TSLA has long bias)
    # =====================================================================
    mask = df['is_long'] == True
    r = simulate_trades(df, mask & test_mask, stop_atr, 'LONG only')
    if r: results.append(r)

    mask = df['is_long'] == False
    r = simulate_trades(df, mask & test_mask, stop_atr, 'SHORT only')
    if r: results.append(r)

    # =====================================================================
    # TEST 4: LONG ONLY + DISTANCE
    # =====================================================================
    for min_dist in [1.0, 1.5, 2.0, 2.5]:
        mask = (df['is_long'] == True) & (df['vwap_dist_atr'] >= min_dist)
        r = simulate_trades(df, mask & test_mask, stop_atr, f'LONG + dist >= {min_dist}')
        if r: results.append(r)

    # =====================================================================
    # TEST 5: TIME OF DAY — first hour vs midday vs last hour
    # =====================================================================
    # Assuming ~78 bars/day (6.5 hrs * 12 bars/hr)
    mask = df['bar_of_day'] <= 12  # first hour
    r = simulate_trades(df, mask & test_mask, stop_atr, 'First hour only')
    if r: results.append(r)

    mask = (df['bar_of_day'] >= 13) & (df['bar_of_day'] <= 60)  # midday
    r = simulate_trades(df, mask & test_mask, stop_atr, 'Midday (bars 13-60)')
    if r: results.append(r)

    mask = df['bar_of_day'] >= 61  # last hour
    r = simulate_trades(df, mask & test_mask, stop_atr, 'Last hour only')
    if r: results.append(r)

    # =====================================================================
    # TEST 6: RSI EXTREME + DISTANCE (classic reversion setup)
    # =====================================================================
    mask = (df['rsi'] < 30) & (df['is_long'] == True) & (df['vwap_dist_atr'] >= 1.0)
    r = simulate_trades(df, mask & test_mask, stop_atr, 'RSI<30 + LONG + dist>=1')
    if r: results.append(r)

    mask = (df['rsi'] > 70) & (df['is_long'] == False) & (df['vwap_dist_atr'] >= 1.0)
    r = simulate_trades(df, mask & test_mask, stop_atr, 'RSI>70 + SHORT + dist>=1')
    if r: results.append(r)

    mask = ((df['rsi'] < 25) | (df['rsi'] > 75)) & (df['vwap_dist_atr'] >= 1.5)
    r = simulate_trades(df, mask & test_mask, stop_atr, 'RSI extreme + dist>=1.5')
    if r: results.append(r)

    # =====================================================================
    # TEST 7: VOLUME SPIKE + DISTANCE (exhaustion signal)
    # =====================================================================
    mask = (df['rel_vol'] >= 2.0) & (df['vwap_dist_atr'] >= 1.0)
    r = simulate_trades(df, mask & test_mask, stop_atr, 'VolSpike>=2x + dist>=1')
    if r: results.append(r)

    mask = (df['rel_vol'] >= 3.0) & (df['vwap_dist_atr'] >= 1.5)
    r = simulate_trades(df, mask & test_mask, stop_atr, 'VolSpike>=3x + dist>=1.5')
    if r: results.append(r)

    # =====================================================================
    # TEST 8: EXPANDING THEN REVERSING (overextension)
    # =====================================================================
    mask = (df['expanding_from_vwap'] == False) & (df['vwap_dist_atr'] >= 1.0)
    r = simulate_trades(df, mask & test_mask, stop_atr, 'Contracting + dist>=1.0')
    if r: results.append(r)

    mask = (df['expanding_from_vwap'] == False) & (df['vwap_dist_atr'] >= 1.5) & (df['is_long'] == True)
    r = simulate_trades(df, mask & test_mask, stop_atr, 'LONG contract + dist>=1.5')
    if r: results.append(r)

    # =====================================================================
    # TEST 9: BARS ON SAME SIDE — only enter after extended time away
    # =====================================================================
    for min_bars in [5, 10, 15, 20]:
        mask = (df['bars_same_side'] >= min_bars) & (df['vwap_dist_atr'] >= 1.0)
        r = simulate_trades(df, mask & test_mask, stop_atr, f'{min_bars}+ bars same side + dist>=1')
        if r: results.append(r)

    # =====================================================================
    # TEST 10: MAX 1 TRADE PER DAY (best setup only)
    # =====================================================================
    # Pick the bar each day with max vwap_dist_atr (most extended = best reversion candidate)
    df_test_copy = df_test.copy()
    best_bar_idx = df_test_copy.groupby('date')['vwap_dist_atr'].idxmax()
    mask = df.index.isin(best_bar_idx.values)
    r = simulate_trades(df, mask & test_mask, stop_atr, 'Max-1/day (most extended)')
    if r: results.append(r)

    # Max 1/day, long only, most extended
    df_long = df_test_copy[df_test_copy['is_long'] == True]
    if len(df_long) > 0:
        best_long_idx = df_long.groupby('date')['vwap_dist_atr'].idxmax()
        mask = df.index.isin(best_long_idx.values)
        r = simulate_trades(df, mask & test_mask, stop_atr, 'Max-1/day LONG (most ext)')
        if r: results.append(r)

    # =====================================================================
    # TEST 11: DIFFERENT STOP WIDTHS with best conditions
    # =====================================================================
    for s in [0.25, 0.35, 0.5, 0.75, 1.0]:
        mask = (df['is_long'] == True) & (df['vwap_dist_atr'] >= 1.5)
        r = simulate_trades(df, mask & test_mask, s, f'LONG+dist>=1.5 stop={s}')
        if r: results.append(r)

    # =====================================================================
    # TEST 12: FIRST HOUR + LONG + DISTANCE (morning reversion)
    # =====================================================================
    mask = (df['bar_of_day'] <= 12) & (df['is_long'] == True) & (df['vwap_dist_atr'] >= 1.0)
    r = simulate_trades(df, mask & test_mask, stop_atr, '1st hr + LONG + dist>=1')
    if r: results.append(r)

    mask = (df['bar_of_day'] <= 12) & (df['is_long'] == True) & (df['vwap_dist_atr'] >= 2.0)
    r = simulate_trades(df, mask & test_mask, stop_atr, '1st hr + LONG + dist>=2')
    if r: results.append(r)

    # =====================================================================
    # TEST 13: CLOSE POSITION IN BAR (enter on wicks/reversals)
    # =====================================================================
    # Long entry: close near HIGH of bar (bouncing up) + far from VWAP
    mask = (df['is_long'] == True) & (df['close_pos'] >= 0.7) & (df['vwap_dist_atr'] >= 1.0)
    r = simulate_trades(df, mask & test_mask, stop_atr, 'LONG + close near high + dist>=1')
    if r: results.append(r)

    # Short entry: close near LOW of bar (selling off) + far from VWAP
    mask = (df['is_long'] == False) & (df['close_pos'] <= 0.3) & (df['vwap_dist_atr'] >= 1.0)
    r = simulate_trades(df, mask & test_mask, stop_atr, 'SHORT + close near low + dist>=1')
    if r: results.append(r)

    # =====================================================================
    # PRINT RESULTS
    # =====================================================================
    print(f"\n{'Label':<40s} {'N':>6s} {'WR':>7s} {'VWAPpct':>8s} {'TotalPnL':>11s} {'AvgPnL':>9s} {'AvgDist':>8s}")
    print("-" * 95)

    # Sort by avg_net_pnl descending
    results.sort(key=lambda x: x['avg_net_pnl'], reverse=True)

    for r in results:
        pnl_str = f"${r['total_net_pnl']:>+10,.0f}"
        avg_str = f"${r['avg_net_pnl']:>+8,.1f}"
        marker = " <-- POSITIVE" if r['avg_net_pnl'] > 0 else ""
        print(f"{r['label']:<40s} {r['n_trades']:>6d} {r['win_rate']*100:>6.1f}% {r['vwap_touch_rate']*100:>7.1f}% {pnl_str} {avg_str} {r['avg_vwap_dist']:>7.2f}{marker}")

    # =====================================================================
    # HIGHLIGHT FINDINGS
    # =====================================================================
    positive = [r for r in results if r['avg_net_pnl'] > 0]
    print(f"\n{'='*90}")
    if positive:
        print(f"FOUND {len(positive)} CONDITION(S) WITH POSITIVE AVG P&L:")
        for r in positive:
            print(f"  * {r['label']}: {r['n_trades']} trades, WR={r['win_rate']*100:.1f}%, avg=${r['avg_net_pnl']:+.1f}, total=${r['total_net_pnl']:+,.0f}")
    else:
        print("NO CONDITIONS FOUND WITH POSITIVE AVG P&L at this stop width.")
        print("\nClosest to breakeven:")
        for r in results[:5]:
            print(f"  {r['label']}: {r['n_trades']} trades, avg=${r['avg_net_pnl']:+.1f}")

    # Also check: what's the VWAP touch rate for bars >2 ATR away?
    print(f"\n{'='*90}")
    print("VWAP REVERSION BASE RATES BY DISTANCE (all 2024 bars, no model):")
    print(f"{'Distance bucket':<25s} {'N bars':>8s} {'VWAP touch rate':>16s}")
    print("-" * 55)

    for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 100.0)]:
        mask = (df_test['vwap_dist_atr'] >= lo) & (df_test['vwap_dist_atr'] < hi)
        n = mask.sum()
        if n == 0:
            continue
        # Quick check: does the next bar (or any bar same day) touch VWAP?
        # Use label column if available, else approximate
        label_col = f"label_s{stop_atr}".replace('.', '_')
        if label_col in df_test.columns:
            wr = df_test.loc[mask, label_col].mean()
            print(f"[{lo:.1f}, {hi:.1f}) ATR         {n:>8d} {wr*100:>15.1f}%")
        else:
            print(f"[{lo:.1f}, {hi:.1f}) ATR         {n:>8d}       (no labels)")


if __name__ == '__main__':
    main()
