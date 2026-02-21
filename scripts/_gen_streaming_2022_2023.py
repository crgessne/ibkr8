"""
Generate streaming indicators for 2022+2023 (with warmup from late 2021).
Train on 2022, test on 2023.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import time
import pandas as pd
from precompute_streaming_indicators import load_data, precompute_all, DEFAULT_WARMUP

DATA_FILE = ROOT / "data" / "tsla_5min_10years.csv"
OUT_FILE = ROOT / "data" / "tsla_5min_streaming_2022_2023.parquet"

def main():
    print("=" * 70, flush=True)
    print("GENERATE STREAMING INDICATORS: 2022-2023 (train/test split)", flush=True)
    print("=" * 70, flush=True)

    # Load full dataset
    print("\nLoading full dataset...", flush=True)
    df_full = load_data(DATA_FILE)
    print(f"  Total bars: {len(df_full):,}", flush=True)

    # Keep Oct 2021 through Dec 2023
    # Oct-Dec 2021 = warmup context for Jan 2022
    # 2022 = train year
    # 2023 = test year
    dt = df_full['datetime']
    keep_mask = (
        ((dt.dt.year == 2021) & (dt.dt.month >= 10)) |
        (dt.dt.year == 2022) |
        (dt.dt.year == 2023)
    )
    df_subset = df_full[keep_mask].copy().reset_index(drop=True)

    n_2021 = ((df_subset['datetime'].dt.year == 2021)).sum()
    n_2022 = ((df_subset['datetime'].dt.year == 2022)).sum()
    n_2023 = ((df_subset['datetime'].dt.year == 2023)).sum()
    print(f"  Subset: {len(df_subset):,} bars", flush=True)
    print(f"    2021 warmup: {n_2021:,} bars", flush=True)
    print(f"    2022 (train): {n_2022:,} bars", flush=True)
    print(f"    2023 (test):  {n_2023:,} bars", flush=True)

    # Compute streaming indicators
    print(f"\nComputing streaming indicators (7 workers)...", flush=True)
    t0 = time.time()
    df_streaming = precompute_all(
        df_subset,
        warmup_bars=DEFAULT_WARMUP,
        n_workers=7,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # Save to parquet
    print(f"\nSaving to {OUT_FILE}...", flush=True)
    if "datetime" in df_streaming.columns:
        df_streaming["datetime"] = pd.to_datetime(df_streaming["datetime"], utc=True)
    if "date" in df_streaming.columns:
        df_streaming["date"] = pd.to_datetime(df_streaming["date"]).dt.date
    df_streaming.to_parquet(OUT_FILE, index=False, engine="pyarrow")
    size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print(f"  Saved: {size_mb:.1f} MB ({len(df_streaming):,} rows x {len(df_streaming.columns)} cols)", flush=True)

    # Verify
    print(f"\nVerification:", flush=True)
    df_read = pd.read_parquet(OUT_FILE, engine="pyarrow")
    yr = pd.to_datetime(df_read['datetime']).dt.year
    print(f"  2021 rows: {(yr == 2021).sum():,}", flush=True)
    print(f"  2022 rows: {(yr == 2022).sum():,}", flush=True)
    print(f"  2023 rows: {(yr == 2023).sum():,}", flush=True)

    # NaN check on 2022+2023
    target = df_read[yr >= 2022]
    key_feats = ["vwap_width_atr", "vol_pct_complete", "rsi", "atr", "ema20_slope_atr"]
    print(f"\n  NaN in 2022+2023 ({len(target):,} rows):", flush=True)
    for f in key_feats:
        if f in target.columns:
            nn = int(target[f].isna().sum())
            print(f"    {f:30s}: {nn:,}", flush=True)

    print(f"\nDone. Use with master_pipeline:", flush=True)
    print(f"  python scripts/master_pipeline.py --indicators-file {OUT_FILE} "
          f"--train-years 2022-2022 --test-years 2023-2023 --model-kind nn_pnl "
          f"--select-mode prob_weighted --prob-risk-pct 0.01", flush=True)


if __name__ == "__main__":
    main()
