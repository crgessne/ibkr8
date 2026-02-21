import pandas as pd
df = pd.read_csv('data/tsla_5min_10years.csv')
df['datetime'] = pd.to_datetime(df['time'], utc=True)
print(f'Total bars: {len(df)}')
print(f'Date range: {df["datetime"].min()} to {df["datetime"].max()}')
y2024_start = df[df['datetime'].dt.year == 2024].index[0]
print(f'2024 starts at index: {y2024_start}')
print(f'Bars before 2024: {y2024_start}')
