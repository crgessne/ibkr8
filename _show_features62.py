import sys, os
sys.path.insert(0, r'C:\Users\Administrator\ibkr8\src')
sys.path.insert(0, r'C:\Users\Administrator\ibkr8')
os.chdir(r'C:\Users\Administrator\ibkr8')

out = open(r'C:\Users\Administrator\ibkr8\_show_features_out.txt', 'w', encoding='utf-8')

from model_persistence import load_model
m, meta = load_model(r'models/rf_vwap_stop0.75_20260218_192628.pkl')
feats = meta['features']

out.write(f'N features: {len(feats)}\n\n')
out.write('Features:\n')
for i, f in enumerate(feats):
    out.write(f'  {i+1:3d}. {f}\n')
out.write('\nMetadata:\n')
for k, v in meta.items():
    if k != 'features':
        out.write(f'  {k} = {v}\n')
out.close()
print('done')
