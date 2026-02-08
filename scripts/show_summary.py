import pandas as pd

clean = pd.read_csv('data/rf_cleaned_features_results.csv')

print('='*80)
print('CLEANED FEATURES RESULTS (RF >= 0.5)')
print('='*80)
print()
print('Stop  | R:R   | Win Rate | Expected Value | Trade Count')
print('-'*80)
for i, row in clean.iterrows():
    print(f"{row['stop_atr']:.2f} ATR | {row['rr']:.1f}:1 | {row['rf0.5_wr']*100:5.1f}%    | {row['rf0.5_ev']:+.3f}R         | {int(row['rf0.5_n']):,}")

print('='*80)
print()
print('BEST SETUP: 0.25 ATR stop')
print(f"  Win Rate: {clean.iloc[0]['rf0.5_wr']*100:.1f}%")
print(f"  R:R: {clean.iloc[0]['rr']:.1f}:1")
print(f"  Expected Value: {clean.iloc[0]['rf0.5_ev']:+.3f}R")
print(f"  Trade Count: {int(clean.iloc[0]['rf0.5_n']):,}")
print(f"  => {clean.iloc[0]['rf0.5_ev']*100:.1f}% return per R risked!")
print()

print('TOP 10 FEATURES (0.25 ATR):')
features = clean.iloc[0]['top_features'].split(',')[:10]
for i, feat in enumerate(features, 1):
    print(f"  {i:2d}. {feat.strip()}")
