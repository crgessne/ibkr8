"""Check precomputed parquet columns and sample rows."""
import pandas as pd
df = pd.read_parquet("data/tsla_5min_streaming_2023-01-01_2024-12-31.parquet")
print(f"Shape: {df.shape}")
print(f"Columns ({len(df.columns)}):")
for c in sorted(df.columns):
    print(f"  {c}")
print(f"\nFirst 3 rows datetime/date:")
print(df[['datetime', 'date']].head(3))
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
print(f"\n2024 rows: {(df['date'] >= '2024-01-01').sum()}")
