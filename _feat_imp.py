"""Extract feature importance rankings from saved RF models."""
import sys, pickle, json
from pathlib import Path
import pandas as pd

models_dir = Path('c:/Users/Administrator/ibkr8/models')
pattern = 'rf_vwap_stop*_20260213_120304.pkl'
files = sorted(models_dir.glob(pattern))
print(f'Found {len(files)} models\n')

all_imp = {}
for f in files:
    meta_f = f.with_suffix('.json')
    with open(meta_f) as mf:
        meta = json.load(mf)
    stop = meta['stop_atr']
    features = meta['features']
      with open(f, 'rb') as pf:
        package = pickle.load(pf)
    rf = package['model']
    
    imp = dict(zip(features, rf.feature_importances_))
    all_imp[stop] = imp

# Build DataFrame
df = pd.DataFrame(all_imp).T
df.index.name = 'stop_atr'

# Average importance across all stops
avg = df.mean().sort_values(ascending=False)
sep = '=' * 70
print(sep)
print('FEATURE IMPORTANCE RANKINGS (avg across all 9 stops)')
print(sep)
for i, (feat, val) in enumerate(avg.items(), 1):
    bar = '#' * int(val * 300)
    print(f'{i:2d}. {feat:<28s} {val:.4f}  {bar}')

print(f'\n{sep}')
print('PER-STOP BREAKDOWN (top 10 per stop)')
print(sep)
for stop in sorted(all_imp.keys()):
    print(f'\nStop {stop}:')
    stop_imp = pd.Series(all_imp[stop]).sort_values(ascending=False)
    for feat in stop_imp.head(10).index:
        print(f'  {feat:<28s} {stop_imp[feat]:.4f}')

# Show where vwap_stretch_zscore ranks per stop
print(f'\n{sep}')
print('vwap_stretch_zscore RANKING PER STOP')
print(sep)
for stop in sorted(all_imp.keys()):
    stop_imp = pd.Series(all_imp[stop]).sort_values(ascending=False)
    rank = list(stop_imp.index).index('vwap_stretch_zscore') + 1
    val = stop_imp['vwap_stretch_zscore']
    print(f'  Stop {stop:5.2f}: rank {rank:2d}/34  importance={val:.4f}')
