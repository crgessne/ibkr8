import pandas as pd
trades = pd.read_csv('data/concurrent_backtest_trades_concurrent.csv')
trades['entry_time'] = pd.to_datetime(trades['entry_time'])
trades['exit_time'] = pd.to_datetime(trades['exit_time'])
trades['duration_min'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 60
print(f"Min duration: {trades['duration_min'].min()} minutes")
print(f"Trades with 5-minute duration: {(trades['duration_min'] == 5).sum()}")
print(f"Trades with <5 minute duration: {(trades['duration_min'] < 5).sum()}")
