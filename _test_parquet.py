"""Quick test: verify pyarrow parquet roundtrip with streaming indicator data types."""
import sys
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import numpy as np
from datetime import date
import os

# Simulate the exact data types the streaming script produces
rows = []
for i in range(100):
    row = {
        'datetime': pd.Timestamp('2023-01-03 09:30:00', tz='UTC') + pd.Timedelta(minutes=5*i),
        'date': date(2023, 1, 3),
        'open': 100.0 + i*0.1,
        'high': 101.0 + i*0.1,
        'low': 99.0 + i*0.1,
        'close': 100.5 + i*0.1,
        'volume': float(1000 + i),
        'vwap_width_atr': 5.69 + np.random.randn()*0.1,
        'rsi': 38.1 + np.random.randn()*5,
        'vol_pct_complete': 1.0,
        'atr': 2.5,
        'ema20_slope_atr': 0.01,
        'vwap_stretch_zscore': -0.5,
        'reversal_quality': 0.7,
        'is_long_setup': int(i % 2),
    }
    rows.append(row)

# Add a row with None/NaN to simulate failed bars (only OHLCV, no indicators)
rows.append({
    'datetime': pd.Timestamp('2023-01-03 18:00:00', tz='UTC'),
    'date': date(2023, 1, 3),
    'open': 110.0, 'high': 111.0, 'low': 109.0, 'close': 110.5, 'volume': 500.0,
})

df = pd.DataFrame(rows)
print(f"DataFrame shape: {df.shape}")
print(f"dtypes:")
for col in df.columns[:10]:
    print(f"  {col}: {df[col].dtype}")

# Convert date column same as the script does
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df['date'] = pd.to_datetime(df['date']).dt.date

out_path = 'data/_test_parquet_roundtrip.parquet'
print(f"\nSaving to {out_path}...")
df.to_parquet(out_path, index=False, engine='pyarrow')
print("Save OK")

# Read back
df2 = pd.read_parquet(out_path, engine='pyarrow')
print(f"\nRead back shape: {df2.shape}")
print(f"dtypes after read:")
for col in df2.columns[:10]:
    print(f"  {col}: {df2[col].dtype}")

# Check values roundtrip
print(f"\nvwap_width_atr[0]: {df2['vwap_width_atr'].iloc[0]:.4f}")
print(f"vol_pct_complete[0]: {df2['vol_pct_complete'].iloc[0]}")
print(f"date[0]: {df2['date'].iloc[0]} (type={type(df2['date'].iloc[0]).__name__})")
print(f"datetime[0]: {df2['datetime'].iloc[0]}")
print(f"NaN in last row vwap_width_atr: {pd.isna(df2['vwap_width_atr'].iloc[-1])}")
print(f"is_long_setup last row NaN: {pd.isna(df2['is_long_setup'].iloc[-1])}")

# Verify numeric precision
orig_val = df['vwap_width_atr'].iloc[0]
read_val = df2['vwap_width_atr'].iloc[0]
assert abs(orig_val - read_val) < 1e-10, f"Precision mismatch: {orig_val} vs {read_val}"

# Verify all columns survived
assert set(df.columns) == set(df2.columns), f"Column mismatch: {set(df.columns) - set(df2.columns)}"

# Verify row count
assert len(df) == len(df2), f"Row count mismatch: {len(df)} vs {len(df2)}"

print("\n=== All roundtrip checks PASSED ===")

# Check file size
size_mb = os.path.getsize(out_path) / (1024 * 1024)
print(f"File size: {size_mb:.2f} MB for {len(df)} rows")

# Estimate full dataset size: ~197K rows with ~143 columns
est_size = size_mb * (197419 / len(df)) * (143 / len(df.columns))
print(f"Estimated full dataset size: ~{est_size:.0f} MB")

os.remove(out_path)
print("Cleaned up test file")
