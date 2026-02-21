"""Quick check of precomputed parquet structure."""
import pandas as pd
pq = pd.read_parquet("data/tsla_5min_streaming_2023-01-01_2024-12-31.parquet")
print(f"Shape: {pq.shape}")
print(f"\nColumns ({len(pq.columns)}):")
for c in sorted(pq.columns):
    print(f"  {c}")
print(f"\nDatetime range: {pq['datetime'].min()} to {pq['datetime'].max()}")
print(f"\nSample datetimes (first 5):")
print(pq['datetime'].head())
print(f"\nSample row (first non-NaN vwap_width_atr):")
valid = pq[pq['vwap_width_atr'].notna()]
if len(valid) > 0:
    row = valid.iloc[0]
    print(f"  datetime: {row['datetime']}")
    print(f"  vwap_width_atr: {row['vwap_width_atr']:.6f}")
    print(f"  rsi: {row.get('rsi', 'N/A')}")
    print(f"  atr: {row.get('atr', 'N/A')}")
    print(f"  vol_pct_complete: {row.get('vol_pct_complete', 'N/A')}")
