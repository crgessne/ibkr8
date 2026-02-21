import pandas as pd
df = pd.read_csv('data/concurrent_per_bar_features_concurrent.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year
df_2024 = df[df['year'] == 2024]

print(f"Total 2024 bars: {len(df_2024)}")
print(f"is_setup=True: {df_2024['is_setup'].sum()}")
print(f"prob>=0.5 (any): {(df_2024['prob'] >= 0.5).sum()}")
print(f"is_setup AND prob>=0.5: {(df_2024['is_setup'] & (df_2024['prob'] >= 0.5)).sum()}")
print(f"prob>=0.5 but NOT setup: {((df_2024['prob'] >= 0.5) & ~df_2024['is_setup']).sum()}")
