"""
Smoke test: compute streaming indicators for 1 month, save to parquet,
then verify it can be loaded and used by master_pipeline's downstream steps.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import time
import numpy as np
import pandas as pd
from datetime import date

# Import the precompute machinery
from precompute_streaming_indicators import load_data, precompute_all, DEFAULT_WARMUP
from master_pipeline import calculate_core_indicators, get_feature_columns

DATA_FILE = ROOT / "data" / "tsla_5min_10years.csv"
OUT_FILE = ROOT / "data" / "_smoke_test_streaming_jan2023.parquet"

def main():
    print("=" * 70)
    print("SMOKE TEST: 1-month streaming indicators → parquet → pipeline check")
    print("=" * 70)

    # --- Step 1: Load full dataset ---
    print("\n[1] Loading full dataset...")
    df_full = load_data(DATA_FILE)
    print(f"    Total bars: {len(df_full):,}")    # --- Step 2: Keep only Nov 2022 + Dec 2022 + Jan 2023 for a small test ---
    # Nov+Dec provide warmup/prior-day context; Jan 2023 is the target month.
    dt = df_full['datetime']
    keep_mask = (
        ((dt.dt.year == 2022) & (dt.dt.month >= 11)) |
        ((dt.dt.year == 2023) & (dt.dt.month == 1))
    )
    df_subset = df_full[keep_mask].copy().reset_index(drop=True)
    jan2023_mask = (df_subset['datetime'].dt.year == 2023) & \
                   (df_subset['datetime'].dt.month == 1)
    n_jan = jan2023_mask.sum()
    print(f"    Subset (Nov22+Dec22+Jan23): {len(df_subset):,} bars")
    print(f"    Jan 2023 target bars: {n_jan}")

    # --- Step 3: Run streaming precompute (all days in subset) ---
    print("\n[2] Computing streaming indicators (all days up to Jan 2023)...")
    t0 = time.time()
    df_streaming = precompute_all(
        df_subset,
        warmup_bars=DEFAULT_WARMUP,
        n_workers=6,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s ({len(df_streaming):,} rows)")

    # --- Step 4: Save to parquet ---
    print(f"\n[3] Saving to {OUT_FILE}...")
    if "datetime" in df_streaming.columns:
        df_streaming["datetime"] = pd.to_datetime(df_streaming["datetime"], utc=True)
    if "date" in df_streaming.columns:
        df_streaming["date"] = pd.to_datetime(df_streaming["date"]).dt.date
    df_streaming.to_parquet(OUT_FILE, index=False, engine="pyarrow")
    size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print(f"    Saved: {size_mb:.1f} MB")

    # --- Step 5: Read back and verify ---
    print(f"\n[4] Reading back parquet...")
    df_read = pd.read_parquet(OUT_FILE, engine="pyarrow")
    print(f"    Shape: {df_read.shape}")
    print(f"    Columns: {len(df_read.columns)}")

    # --- Step 6: Check feature columns exist ---
    print(f"\n[5] Checking feature columns...")
    features = get_feature_columns(df_read)
    print(f"    Features found: {len(features)}")

    # Check which features have NaN
    jan_rows = df_read[
        pd.to_datetime(df_read['datetime']).dt.year == 2023
    ]
    print(f"    Jan 2023 rows in output: {len(jan_rows)}")

    n_missing = 0
    for f in features:
        if f not in df_read.columns:
            print(f"    [MISSING] {f}")
            n_missing += 1
    if n_missing == 0:
        print(f"    All {len(features)} features present!")

    # NaN stats for Jan 2023 rows
    print(f"\n[6] NaN stats for Jan 2023 bars (key features):")
    key_feats = [
        "vwap_width_atr", "vol_pct_complete", "rsi", "ema20_slope_atr",
        "vwap_stretch_zscore", "reversal_quality", "atr", "is_long_setup",
        "minutes_into_session", "vwap_slope_atr",
    ]
    for f in key_feats:
        if f in jan_rows.columns:
            n_nan = int(jan_rows[f].isna().sum())
            pct = n_nan / len(jan_rows) * 100
            print(f"    {f:30s}: {n_nan:5,} NaN ({pct:5.1f}%)")

    # --- Step 7: Compare streaming vs vectorized for a sample bar ---
    print(f"\n[7] Comparing streaming vs vectorized (sample from Jan 2023)...")
    # Compute vectorized on the same subset
    df_vec = calculate_core_indicators(df_subset.copy(), verbose=False)
    # Pick a bar in the middle of January
    jan_dates = sorted(set(jan_rows['date']))
    mid_date = jan_dates[len(jan_dates) // 2]
    # Get a bar from that date
    stream_day = jan_rows[jan_rows['date'] == mid_date]
    vec_day = df_vec[df_vec['date'] == mid_date]

    if len(stream_day) > 0 and len(vec_day) > 0:
        # Compare the 10th bar of the day (avoid first-bar edge effects)
        bar_idx = min(10, len(stream_day) - 1)
        s_row = stream_day.iloc[bar_idx]
        v_row = vec_day.iloc[bar_idx]
        print(f"    Sample date: {mid_date}, bar #{bar_idx}")
        print(f"    {'Feature':30s} {'Streaming':>12s} {'Vectorized':>12s} {'Diff':>10s}")
        print(f"    {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
        compare_feats = [
            "vwap_width_atr", "rsi", "vol_pct_complete", "atr",
            "ema20_slope_atr", "vwap_stretch_zscore", "minutes_into_session",
            "is_long_setup", "reversal_quality",
        ]
        for f in compare_feats:
            sv = s_row.get(f, float('nan'))
            vv = v_row.get(f, float('nan'))
            if pd.notna(sv) and pd.notna(vv):
                diff = float(sv) - float(vv)
                print(f"    {f:30s} {float(sv):12.4f} {float(vv):12.4f} {diff:+10.4f}")
            else:
                print(f"    {f:30s} {str(sv):>12s} {str(vv):>12s}      ---")

    # --- Step 8: Verify vol_pct_complete is 1.0 ---
    print(f"\n[8] Checking vol_pct_complete...")
    if 'vol_pct_complete' in jan_rows.columns:
        vpc_vals = jan_rows['vol_pct_complete'].dropna()
        all_one = (vpc_vals == 1.0).all()
        print(f"    All 1.0: {all_one} (unique values: {vpc_vals.unique()[:5]})")
    
    print(f"\n{'='*70}")
    print("SMOKE TEST COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
