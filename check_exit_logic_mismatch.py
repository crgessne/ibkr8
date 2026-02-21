"""
Compare concurrent backtest exits vs label generator logic.
Find specific examples where they diverge.
"""

import pandas as pd
import numpy as np

# Load data
bars = pd.read_csv('data/tsla_5min_10years.csv')
bars['datetime'] = pd.to_datetime(bars['datetime'])
bars_2024 = bars[bars['datetime'].dt.year == 2024].copy().reset_index(drop=True)

# Load trades
trades = pd.read_csv('data/concurrent_backtest_trades_single.csv')
trades['entry_time'] = pd.to_datetime(trades['entry_time'])
trades['exit_time'] = pd.to_datetime(trades['exit_time'])

print(f"Total trades: {len(trades)}")
print(f"Win rate: {(trades['pnl'] > 0).mean():.1%}\n")

# Get a sample of losing trades to analyze
losing_trades = trades[trades['pnl'] < 0].sample(min(5, len(trades)), random_state=42)

for idx, trade in losing_trades.iterrows():
    print(f"=== Trade {idx} ===")
    print(f"Entry: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
    print(f"Exit: {trade['exit_time']} @ ${trade['exit_price']:.2f} ({trade['reason']})")
    print(f"P&L: ${trade['pnl']:.2f}")
    
    # Find entry bar
    entry_bar_idx = bars_2024[bars_2024['datetime'] == trade['entry_time']].index
    if len(entry_bar_idx) == 0:
        print("ERROR: Entry bar not found!\n")
        continue
    
    entry_bar_idx = entry_bar_idx[0]
    entry_bar = bars_2024.iloc[entry_bar_idx]
    
    # Calculate what stop/target SHOULD be based on label generator logic
    entry_price = entry_bar['close']
    vwap = entry_bar['vwap']
    atr = entry_bar['atr']
    
    is_long = entry_price < vwap
    stop_dist = 1.25 * atr
    
    if is_long:
        stop_price = entry_price - stop_dist
        target_price = vwap
    else:
        stop_price = entry_price + stop_dist
        target_price = vwap
    
    print(f"Direction: {'LONG' if is_long else 'SHORT'}")
    print(f"Entry price: ${entry_price:.2f}, VWAP: ${vwap:.2f}")
    print(f"Stop: ${stop_price:.2f}, Target: ${target_price:.2f}")
    
    # Now check forward bars to see what SHOULD have happened
    # Find end of day
    entry_date = trade['entry_time'].date()
    day_end_idx = bars_2024[bars_2024['datetime'].dt.date == entry_date].index[-1]
    
    print(f"Checking bars from {entry_bar_idx + 1} to {day_end_idx}")
    
    hit_target = False
    hit_stop = False
    first_hit_bar = None
    
    for k in range(entry_bar_idx + 1, day_end_idx + 1):
        bar = bars_2024.iloc[k]
        
        if is_long:
            if bar['low'] <= stop_price:
                hit_stop = True
                first_hit_bar = k
                print(f"  Bar {k} ({bar['datetime']}): LOW ${bar['low']:.2f} <= STOP ${stop_price:.2f} ❌")
                break
            if bar['high'] >= target_price:
                hit_target = True
                first_hit_bar = k
                print(f"  Bar {k} ({bar['datetime']}): HIGH ${bar['high']:.2f} >= TARGET ${target_price:.2f} ✅")
                break
        else:
            if bar['high'] >= stop_price:
                hit_stop = True
                first_hit_bar = k
                print(f"  Bar {k} ({bar['datetime']}): HIGH ${bar['high']:.2f} >= STOP ${stop_price:.2f} ❌")
                break
            if bar['low'] <= target_price:
                hit_target = True
                first_hit_bar = k
                print(f"  Bar {k} ({bar['datetime']}): LOW ${bar['low']:.2f} <= TARGET ${target_price:.2f} ✅")
                break
    
    label_result = 1 if hit_target else 0
    actual_result = 1 if trade['pnl'] > 0 else 0
    
    print(f"\nLabel says: {'WIN' if label_result == 1 else 'LOSS'}")
    print(f"Concurrent says: {'WIN' if actual_result == 1 else 'LOSS'}")
    
    if label_result != actual_result:
        print("⚠️ MISMATCH! This is the bug!")
    
    print()
