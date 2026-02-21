import pandas as pd

trades = pd.read_csv('data/concurrent_backtest_trades_single.csv')
trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)

raw = pd.read_csv('data/tsla_5min_10years.csv')
raw['datetime'] = pd.to_datetime(raw['time'], utc=True)
raw = raw[raw['datetime'].dt.year == 2024].reset_index(drop=True)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from master_pipeline import calculate_core_indicators

raw['date'] = raw['datetime'].dt.date
raw = calculate_core_indicators(raw, verbose=False)

idx_map = pd.Series(raw.index.values, index=raw['datetime'].values)

ok = 0
missing = 0
for et in trades['entry_time'].head(500):
    et64 = et.to_datetime64()
    if et64 in idx_map.index:
        ok += 1
    else:
        missing += 1

print('ok', ok, 'missing', missing)
print('sample missing:', trades.loc[~trades['entry_time'].apply(lambda x: x.to_datetime64() in idx_map.index), 'entry_time'].head(5).tolist())
