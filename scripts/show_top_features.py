import pandas as pd

df = pd.read_csv('data/rf_cleaned_features_results.csv')

print("="*90)
print("TOP FEATURES BY STOP WIDTH (CLEANED MODEL)")
print("="*90)

for i, row in df.iterrows():
    print(f"\n{row['stop_atr']:.2f} ATR Stop (R:R {row['rr']:.1f}:1):")
    feats = row['top_features'].split(',')[:10]
    for j, f in enumerate(feats, 1):
        print(f"  {j:2d}. {f.strip()}")
