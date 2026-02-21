import pandas as pd

# Load trades
print("Loading trades...")
trades = pd.read_csv('data/concurrent_backtest_trades_concurrent.csv')
print(f"Total trades: {len(trades)}")
print(f"Wins: {(trades['pnl'] > 0).sum()}")
print(f"Losses: {(trades['pnl'] <= 0).sum()}")
print(f"Win rate: {(trades['pnl'] > 0).mean():.1%}")

# Check minimum time between entry and exit
trades['entry_time'] = pd.to_datetime(trades['entry_time'])
trades['exit_time'] = pd.to_datetime(trades['exit_time'])
trades['duration_minutes'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 60
print(f"\nMinimum duration: {trades['duration_minutes'].min():.0f} minutes")
print(f"Trades with 5 min duration: {(trades['duration_minutes'] == 5).sum()}")

# Load bars
print("\nLoading bars...")
bars = pd.read_csv('data/concurrent_per_bar_features_concurrent.csv')
bars['datetime'] = pd.to_datetime(bars['datetime'])
print(f"Total bars: {len(bars)}")

# Get ONE losing trade to trace
losing_trade = trades[trades['pnl'] < 0].iloc[0]
print(f"\n=== Tracing Losing Trade ===")
print(f"Entry: {losing_trade['entry_time']} @ ${losing_trade['entry_price']:.2f}")
print(f"Exit: {losing_trade['exit_time']} @ ${losing_trade['exit_price']:.2f} ({losing_trade['exit_reason']})")
print(f"Stop: ${losing_trade['stop']:.2f}, Target: ${losing_trade['target']:.2f}")
print(f"Duration: {losing_trade['duration_minutes']:.0f} minutes")

# Find the entry bar
entry_bar_idx = bars[bars['datetime'] == losing_trade['entry_time']].index[0]
print(f"Entry bar index: {entry_bar_idx}")

# Get entry bar data
entry_bar = bars.iloc[entry_bar_idx]
print(f"\nEntry bar: close=${entry_bar['close']:.2f}, low=${entry_bar['low']:.2f}, high=${entry_bar['high']:.2f}")

# Check if entry bar itself would trigger exit
if entry_bar['low'] <= losing_trade['stop']:
    print(f"⚠️ ENTRY BAR LOW ({entry_bar['low']:.2f}) <= STOP ({losing_trade['stop']:.2f})!")
if entry_bar['high'] >= losing_trade['target']:
    print(f"⚠️ ENTRY BAR HIGH ({entry_bar['high']:.2f}) >= TARGET ({losing_trade['target']:.2f})!")

# Get next 3 bars
print("\nNext bars:")
for i in range(1, 4):
    if entry_bar_idx + i >= len(bars):
        break
    next_bar = bars.iloc[entry_bar_idx + i]
    print(f"  Bar +{i}: {next_bar['datetime']} low=${next_bar['low']:.2f}, high=${next_bar['high']:.2f}")
    
    if next_bar['low'] <= losing_trade['stop']:
        print(f"    ❌ Stop hit!")
        break
    if next_bar['high'] >= losing_trade['target']:
        print(f"    ✅ Target hit!")
        break
