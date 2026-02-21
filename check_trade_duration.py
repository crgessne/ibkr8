"""
Check if concurrent trades are held across days (which would explain win rate difference).
"""

import pandas as pd

# Load trades
trades = pd.read_csv('data/concurrent_backtest_trades_concurrent.csv')
trades['entry_time'] = pd.to_datetime(trades['entry_time'])
trades['exit_time'] = pd.to_datetime(trades['exit_time'])

# Calculate trade duration
trades['duration'] = trades['exit_time'] - trades['entry_time']
trades['duration_minutes'] = trades['duration'].dt.total_seconds() / 60
trades['entry_date'] = trades['entry_time'].dt.date
trades['exit_date'] = trades['exit_time'].dt.date
trades['crosses_day'] = trades['entry_date'] != trades['exit_date']

print("="*80)
print("TRADE DURATION ANALYSIS")
print("="*80)

print(f"\nTotal trades: {len(trades):,}")
print(f"Same-day exits: {(~trades['crosses_day']).sum():,} ({(~trades['crosses_day']).sum()/len(trades)*100:.1f}%)")
print(f"Multi-day holds: {trades['crosses_day'].sum():,} ({trades['crosses_day'].sum()/len(trades)*100:.1f}%)")

print(f"\nDuration statistics:")
print(f"  Min: {trades['duration_minutes'].min():.0f} minutes")
print(f"  Median: {trades['duration_minutes'].median():.0f} minutes")
print(f"  Mean: {trades['duration_minutes'].mean():.0f} minutes")
print(f"  Max: {trades['duration_minutes'].max():.0f} minutes ({trades['duration_minutes'].max()/60/24:.1f} days)")

# Check win rate for same-day vs multi-day
same_day = trades[~trades['crosses_day']]
multi_day = trades[trades['crosses_day']]

if len(same_day) > 0:
    same_day_wr = (same_day['reason'] == 'target').sum() / len(same_day)
    print(f"\nSame-day trades win rate: {same_day_wr*100:.1f}%")

if len(multi_day) > 0:
    multi_day_wr = (multi_day['reason'] == 'target').sum() / len(multi_day)
    print(f"Multi-day trades win rate: {multi_day_wr*100:.1f}%")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print("Label generator closes ALL positions at END OF DAY.")
print("Concurrent backtest holds positions across days until stop/target hit.")
print("\nThis is why win rates differ!")
print("\nTo match master exactly, concurrent needs to close positions at EOD.")
