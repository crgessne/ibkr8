"""
Trace exact exit checking logic to find the win rate discrepancy.

The key question: When a position is opened at bar j (at close price),
when is the first exit check performed?

Label Generator Logic:
- Entry at bar j close
- Check exits from bar j+1 to day_end
- Uses bar j+1 low/high, bar j+2 low/high, etc.

Concurrent Logic:
- Loop processes bars sequentially
- At bar j: check_exits(bar_j), then open_position(bar_j)
- At bar j+1: check_exits(bar_j+1)
- So position opened at bar_j is first checked at bar_j+1

This SHOULD match the label generator...unless there's a subtle difference.

Let's check: What if a position is opened at bar j, and the SAME bar j
has low/high that would trigger stop/target? 
- Label generator: Ignores bar j (starts at j+1)
- Concurrent: check_exits runs BEFORE open_position, so bar j is not checked

Wait...but what about the NEXT iteration?
- Bar j: open position at close
- Bar j+1: check_exits uses bar_j+1 low/high

Actually, I think the issue might be different. Let me check if concurrent
is using the entry bar's low/high to check exits.
"""

import pandas as pd
import numpy as np

# Load trades
trades = pd.read_csv('data/concurrent_backtest_trades_concurrent.csv')
print(f"Total trades: {len(trades)}")
print(f"Wins: {(trades['pnl'] > 0).sum()}")
print(f"Losses: {(trades['pnl'] <= 0).sum()}")
print(f"Win rate: {(trades['pnl'] > 0).mean():.1%}")

# Check if any trades exit on the same bar as entry
trades['entry_time'] = pd.to_datetime(trades['entry_time'])
trades['exit_time'] = pd.to_datetime(trades['exit_time'])
same_bar = trades['entry_time'] == trades['exit_time']
print(f"\nTrades exiting on same bar as entry: {same_bar.sum()}")

# Check minimum time between entry and exit
trades['duration_minutes'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 60
print(f"\nMinimum duration: {trades['duration_minutes'].min():.0f} minutes")
print(f"Trades with <5 min duration: {(trades['duration_minutes'] < 5).sum()}")

# Now let's check something else: Are we checking exits correctly?
# Let me load the per-bar data and trace a specific losing trade

bars = pd.read_csv('data/concurrent_per_bar_features_concurrent.csv')
bars['datetime'] = pd.to_datetime(bars['datetime'])
print(f"\nTotal bars: {len(bars)}")

# Get a sample losing trade
losing_trades = trades[trades['pnl'] < 0].head(5)
print("\n=== Sample Losing Trades ===")
for idx, trade in losing_trades.iterrows():
    print(f"\nTrade {idx}:")
    print(f"  Entry: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
    print(f"  Exit: {trade['exit_time']} @ ${trade['exit_price']:.2f} ({trade['exit_reason']})")
    print(f"  Stop: ${trade['stop']:.2f}, Target: ${trade['target']:.2f}")
    print(f"  Duration: {trade['duration_minutes']:.0f} minutes")
    
    # Find the entry bar and subsequent bars
    entry_bar_idx = bars[bars['datetime'] == trade['entry_time']].index
    if len(entry_bar_idx) == 0:
        print("  ERROR: Entry bar not found!")
        continue
    
    entry_bar_idx = entry_bar_idx[0]
    print(f"  Entry bar index: {entry_bar_idx}")
    
    # Get entry bar
    entry_bar = bars.iloc[entry_bar_idx]
    print(f"  Entry bar: close=${entry_bar['close']:.2f}, low=${entry_bar['low']:.2f}, high=${entry_bar['high']:.2f}")
    
    # Check if entry bar itself would have triggered exit
    if entry_bar['low'] <= trade['stop']:
        print(f"  ⚠️ ENTRY BAR LOW ({entry_bar['low']:.2f}) <= STOP ({trade['stop']:.2f})!")
    if entry_bar['high'] >= trade['target']:
        print(f"  ⚠️ ENTRY BAR HIGH ({entry_bar['high']:.2f}) >= TARGET ({trade['target']:.2f})!")
    
    # Get next few bars
    for i in range(1, min(6, len(bars) - entry_bar_idx)):
        next_bar = bars.iloc[entry_bar_idx + i]
        print(f"  Bar +{i}: {next_bar['datetime']} close=${next_bar['close']:.2f}, low=${next_bar['low']:.2f}, high=${next_bar['high']:.2f}")
        
        # Check if this bar triggered exit
        if next_bar['low'] <= trade['stop']:
            print(f"    ❌ Stop hit: low {next_bar['low']:.2f} <= stop {trade['stop']:.2f}")
            if next_bar['datetime'] == trade['exit_time']:
                print(f"    ✓ This is the exit bar")
            break
        if next_bar['high'] >= trade['target']:
            print(f"    ✅ Target hit: high {next_bar['high']:.2f} >= target {trade['target']:.2f}")
            if next_bar['datetime'] == trade['exit_time']:
                print(f"    ✓ This is the exit bar")
            break
