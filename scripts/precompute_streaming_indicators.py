"""
Pre-compute streaming-style indicators for all bars in the dataset.

This script simulates how indicators would be computed in live paper trading:
  - For each bar, the streaming simulator has a window of:
      warmup_bars BEFORE prior-day open + all prior-day bars + current-day bars up to now
  - It calls `calculate_core_indicators()` on that window and extracts the last row

Instead of calling `calculate_core_indicators()` 197K times (once per bar), this
script uses an optimised day-based approach:
  1. For each target day, build the full window (warmup + prior day + all current day bars)
  2. Call `calculate_core_indicators()` ONCE on that window
  3. Extract indicator values for each current-day bar from the corresponding row
  4. Patch forward-looking features to match streaming behaviour:
     - vol_pct_complete: set to streaming value (cum_vol / cum_vol = 1.0 for current bar)
     - Fractals/volume profile: accept minor approximation from full-day window

This reduces ~197K calls to ~2,500 calls (one per day), giving ~78x speedup.
With 6 workers: ~2500 calls / 6 = ~420 per worker x ~7s each = ~49 minutes.

The output is saved as a Parquet file that can be fed to `master_pipeline.py`
so that models are trained on indicators matching live inference conditions.

Key differences from vectorized (full-dataset) indicators:
  1. EMA edge effects: streaming windows have ~200-300 bars of history,
     so EMA-60/EMA-20 don't see the full dataset history
  2. Rolling windows (ATR-14, RSI-14, rolling-20, rolling-60): limited warmup
  3. vol_pct_complete: streaming = 1.0 always (no future volume lookahead)
  4. Prior-day stats: correctly computed from the prior full day in the window
  5. Fractal swing pivots: minor approximation (full-day window vs true streaming)
  6. Volume profile: minor approximation (full-day vs developing profile)

Usage:
    python scripts/precompute_streaming_indicators.py
    python scripts/precompute_streaming_indicators.py --warmup 60 --workers 6
    python scripts/precompute_streaming_indicators.py --ticker tsla --date-range 2022-01-01:2023-12-31
    python scripts/precompute_streaming_indicators.py --ticker tqqq --date-range 2024-01-01:2024-06-30

Output:
    data/tsla_5min_streaming_indicators.parquet
    data/tsla_5min_streaming_2022-01-01_2023-12-31.parquet  (with --date-range)
"""

import sys
from pathlib import Path

# Ensure imports from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import argparse
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date as dt_date, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = ROOT / "data" / "tsla_5min_10years.csv"
OUTPUT_DIR = ROOT / "data"
DEFAULT_WARMUP = 60       # bars before prior-day open for rolling indicator warmup
DEFAULT_WORKERS = 6       # number of parallel worker processes

# Minimum bars needed to produce valid indicators (ATR-14 needs 14+, RSI needs 14+, etc.)
MIN_BARS_FOR_INDICATORS = 20

# Known ticker -> data file mapping pattern
TICKER_DATA_PATTERN = "{ticker}_5min_10years.csv"


def resolve_ticker_data_file(ticker: str) -> Path:
    """Resolve a ticker symbol to its data file path.

    Args:
        ticker: Ticker symbol (e.g., 'tsla', 'tqqq'). Case-insensitive.

    Returns:
        Path to the data file.

    Raises:
        FileNotFoundError: If the data file doesn't exist.
    """
    ticker_lower = ticker.lower()
    filename = TICKER_DATA_PATTERN.format(ticker=ticker_lower)
    filepath = ROOT / "data" / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found for ticker '{ticker}': {filepath}\n"
            f"Expected pattern: data/{TICKER_DATA_PATTERN.format(ticker='<ticker>')}"
        )
    return filepath


def parse_date_range(date_range_str: str) -> tuple:
    """Parse a date range string like '2022-01-01:2023-12-31'.

    Args:
        date_range_str: Date range in format 'YYYY-MM-DD:YYYY-MM-DD'.

    Returns:
        Tuple of (start_date, end_date) as datetime.date objects.

    Raises:
        ValueError: If the format is invalid.
    """
    parts = date_range_str.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid date range format: '{date_range_str}'. "
            f"Expected 'YYYY-MM-DD:YYYY-MM-DD' (e.g., '2022-01-01:2023-12-31')"
        )
    try:
        start = pd.Timestamp(parts[0].strip()).date()
        end = pd.Timestamp(parts[1].strip()).date()
    except Exception as e:
        raise ValueError(
            f"Invalid date in range '{date_range_str}': {e}"
        )
    if start > end:
        raise ValueError(
            f"Start date ({start}) must be before end date ({end})"
        )
    return start, end


def filter_with_warmup(
    df: pd.DataFrame,
    start_date: dt_date,
    end_date: dt_date,
    warmup_bars: int = DEFAULT_WARMUP,
    verbose: bool = True,
) -> tuple:
    """Filter DataFrame to date range with warmup context before start_date.

    Includes enough bars before start_date to provide rolling indicator warmup.
    The warmup is at least `warmup_bars` bars before the first day's prior-day
    open, matching the window logic in precompute_all().

    Args:
        df: Full OHLCV DataFrame with 'date' and 'datetime' columns.
        start_date: First target date (inclusive).
        end_date: Last target date (inclusive).
        warmup_bars: Number of warmup bars for rolling indicators.
        verbose: Print info about filtering.

    Returns:
        Tuple of (filtered_df, target_dates_set) where:
          - filtered_df includes warmup + target bars
          - target_dates_set is the set of dates in the target range
    """
    all_dates = sorted(df["date"].unique())

    # Find target dates in range
    target_dates = [d for d in all_dates if start_date <= d <= end_date]
    if not target_dates:
        raise ValueError(
            f"No data found in date range {start_date} to {end_date}. "
            f"Available range: {all_dates[0]} to {all_dates[-1]}"
        )

    # Find the first target date's position in the full date list
    first_target_idx = all_dates.index(target_dates[0])

    # We need warmup context: prior day + warmup_bars before that
    # Go back enough days to cover warmup_bars + one prior day
    # Estimate ~78 bars per day, so warmup_bars/78 + 2 days should be enough
    days_back = max(3, warmup_bars // 50 + 3)  # generous estimate
    warmup_start_idx = max(0, first_target_idx - days_back)
    warmup_start_date = all_dates[warmup_start_idx]

    # Filter: warmup_start_date through end_date
    mask = (df["date"] >= warmup_start_date) & (df["date"] <= end_date)
    df_filtered = df[mask].copy().reset_index(drop=True)

    target_dates_set = set(target_dates)

    if verbose:
        n_warmup = len(df_filtered[df_filtered["date"] < start_date])
        n_target = len(df_filtered[df_filtered["date"] >= start_date])
        print(f"  Date range: {start_date} to {end_date}", flush=True)
        print(f"  Target days: {len(target_dates)}", flush=True)
        print(f"  Warmup from: {warmup_start_date} ({n_warmup:,} bars)", flush=True)
        print(f"  Target bars: {n_target:,}", flush=True)
        print(f"  Total subset: {len(df_filtered):,} bars", flush=True)

    return df_filtered, target_dates_set


def load_data(filepath: Path) -> pd.DataFrame:
    """Load and prepare the raw OHLCV data."""
    df = pd.read_csv(filepath)
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
        df["date"] = df["datetime"].dt.date
    elif df.index.name == "time":
        df = df.reset_index()
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
        df["date"] = df["datetime"].dt.date
    else:
        raise ValueError("No 'time' column found in data")
    return df


def _worker_compute_day(
    day_task: dict,
    all_bars_records: list,
    all_bars_columns: list,
    warmup_bars: int,
) -> dict:
    """Worker function: compute streaming indicators for one target day.

    Approach:
      1. Build the window: warmup bars + prior day (if any) + all target day bars
      2. Call calculate_core_indicators() ONCE on the full window
      3. Extract rows for each target-day bar
      4. Patch vol_pct_complete to match streaming behaviour

    Args:
        day_task: dict with keys:
            'target_date': the date to compute
            'window_start': iloc start of the window in all_bars_records
            'window_end': iloc end (exclusive) of the window
            'target_start': iloc of first bar of target day
            'target_end': iloc end (exclusive) of last bar of target day
        all_bars_records: list of lists (all bars as records)
        all_bars_columns: list of column names
        warmup_bars: number of warmup bars

    Returns:
        dict with key 'results': list of (bar_iloc, indicator_dict) tuples
    """
    # Import inside worker to avoid pickling issues
    from master_pipeline import calculate_core_indicators

    target_date = day_task['target_date']
    window_start = day_task['window_start']
    window_end = day_task['window_end']
    target_start = day_task['target_start']
    target_end = day_task['target_end']

    extract_ilocs = list(range(target_start, target_end))

    window_size = window_end - window_start
    if window_size < MIN_BARS_FOR_INDICATORS:
        return {'results': [(iloc, None) for iloc in extract_ilocs]}

    # Build DataFrame for the window
    window_records = all_bars_records[window_start:window_end]
    window_df = pd.DataFrame(window_records, columns=all_bars_columns)

    # Ensure datetime and date columns
    if "datetime" not in window_df.columns and "time" in window_df.columns:
        window_df["datetime"] = pd.to_datetime(window_df["time"], utc=True)
    if "date" not in window_df.columns:
        window_df["date"] = pd.to_datetime(window_df["datetime"]).dt.date

    try:
        df_ind = calculate_core_indicators(window_df, verbose=False)
    except Exception:
        return {'results': [(iloc, None) for iloc in extract_ilocs]}

    # The offset from global iloc to df_ind row index
    # df_ind has rows 0..len(window)-1, corresponding to global ilocs window_start..window_end-1
    offset = window_start

    results = []
    for bar_iloc in extract_ilocs:
        row_idx = bar_iloc - offset
        if row_idx < 0 or row_idx >= len(df_ind):
            results.append((bar_iloc, None))
            continue

        row = df_ind.iloc[row_idx]
        ind_dict = {}
        for col in df_ind.columns:
            val = row[col]
            if isinstance(val, (np.integer,)):
                ind_dict[col] = int(val)
            elif isinstance(val, (np.floating,)):
                ind_dict[col] = float(val)
            else:
                ind_dict[col] = val

        # Patch vol_pct_complete to streaming behaviour.
        # In streaming, at bar j the window only has bars up to j, so
        # groupby('date')['volume'].transform('sum') = cum_vol at bar j.
        # Therefore vol_pct_complete = cum_vol[j] / cum_vol[j] = 1.0.
        ind_dict['vol_pct_complete'] = 1.0

        results.append((bar_iloc, ind_dict))

    return {'results': results}


def precompute_all(
    df: pd.DataFrame,
    warmup_bars: int = DEFAULT_WARMUP,
    n_workers: int = DEFAULT_WORKERS,
    verbose: bool = True,
    target_dates_only: set = None,
) -> pd.DataFrame:
    """Pre-compute streaming indicators for ALL bars using multiprocessing.

    Strategy: process one day at a time. For each day, build a window
    containing warmup + prior day + all of target day, call calculate_core_indicators
    once, and extract each target bar's row.

    Args:
        df: Full OHLCV DataFrame with datetime and date columns.
        warmup_bars: Number of bars before prior-day open to include for rolling warmup.
        n_workers: Number of parallel worker processes.
        verbose: Print progress.
        target_dates_only: If provided, only compute indicators for these dates.
            The df should still include warmup bars before the first target date.
            If None, all dates in df are processed.

    Returns:
        DataFrame with one row per target bar, containing all indicator columns.
    """
    # Build day index: date -> (start_iloc, end_iloc)  [half-open]
    dates = df["date"].values
    unique_dates_ordered = []
    day_ranges = {}  # date -> (start_iloc, end_iloc)
    prev_date = None
    start_idx = 0
    for i, d in enumerate(dates):
        if d != prev_date:
            if prev_date is not None:
                day_ranges[prev_date] = (start_idx, i)
            unique_dates_ordered.append(d)
            start_idx = i
            prev_date = d
    if prev_date is not None:
        day_ranges[prev_date] = (start_idx, len(dates))

    date_to_order = {d: i for i, d in enumerate(unique_dates_ordered)}

    # Select target dates — either all, or filtered subset
    if target_dates_only is not None:
        target_dates = [d for d in unique_dates_ordered if d in target_dates_only]
    else:
        target_dates = list(unique_dates_ordered)

    # Count total target bars
    n_total_bars = 0
    for d in target_dates:
        s, e = day_ranges[d]
        n_total_bars += e - s

    if verbose:
        print(f"\n{'='*80}", flush=True)
        print(f"PRE-COMPUTING STREAMING INDICATORS", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"  Total bars in dataset: {len(df):,}", flush=True)
        print(f"  Target days: {len(target_dates):,}", flush=True)
        print(f"  Target bars: {n_total_bars:,}", flush=True)
        print(f"  Warmup bars: {warmup_bars}", flush=True)
        print(f"  Workers: {n_workers}", flush=True)
        print(flush=True)

    # Build day tasks
    all_bars_columns = df.columns.tolist()
    all_bars_records = df.values.tolist()

    day_tasks = []
    for target_date in target_dates:
        target_start, target_end = day_ranges[target_date]
        order = date_to_order[target_date]

        # Window starts at: prior_day_start - warmup_bars (or 0)
        if order >= 1:
            prior_date = unique_dates_ordered[order - 1]
            prior_start, _ = day_ranges[prior_date]
            window_start = max(0, prior_start - warmup_bars)
        else:
            window_start = max(0, target_start - warmup_bars)

        window_end = target_end  # include all of target day

        day_tasks.append({
            'target_date': target_date,
            'window_start': window_start,
            'window_end': window_end,
            'target_start': target_start,
            'target_end': target_end,
        })

    n_tasks = len(day_tasks)
    if verbose:
        print(f"  Day tasks: {n_tasks}", flush=True)
        print(flush=True)

    # Process day tasks in parallel
    results_dict = {}  # bar_iloc -> indicator_dict
    completed = 0
    bars_completed = 0
    t_start = time.time()

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {}
        for task_idx, task in enumerate(day_tasks):
            future = executor.submit(
                _worker_compute_day,
                task,
                all_bars_records,
                all_bars_columns,
                warmup_bars,
            )
            futures[future] = task_idx

        for future in as_completed(futures):
            task_idx = futures[future]
            task = day_tasks[task_idx]
            try:
                result = future.result()
                chunk_results = result['results']
                for bar_iloc, ind_dict in chunk_results:
                    results_dict[bar_iloc] = ind_dict
                bars_completed += len(chunk_results)
                completed += 1

                if verbose and (completed % 50 == 0 or completed == n_tasks):
                    elapsed = time.time() - t_start
                    pct = completed / n_tasks * 100
                    rate = bars_completed / elapsed if elapsed > 0 else 0
                    remaining_bars = n_total_bars - bars_completed
                    eta = remaining_bars / rate if rate > 0 else 0
                    print(
                        f"  [{pct:5.1f}%] {completed}/{n_tasks} days | "
                        f"{bars_completed:,}/{n_total_bars:,} bars | "
                        f"{rate:.0f} bars/s | ETA {eta/60:.1f}m",
                        flush=True,
                    )
            except Exception as e:
                print(f"  [ERROR] Day task {task_idx} ({task['target_date']}): {e}", flush=True)
                import traceback; traceback.print_exc()
                for iloc in range(task['target_start'], task['target_end']):
                    results_dict[iloc] = None
                completed += 1

    elapsed_total = time.time() - t_start
    if verbose:
        print(f"\n  Completed in {elapsed_total/60:.1f} minutes ({elapsed_total:.0f}s)", flush=True)

    # Assemble results into a DataFrame
    if verbose:
        print("  Assembling output DataFrame...", flush=True)

    all_target_ilocs = []
    for d in target_dates:
        s, e = day_ranges[d]
        all_target_ilocs.extend(range(s, e))

    out_rows = []
    n_ok = 0
    n_fail = 0
    for bar_iloc in all_target_ilocs:
        ind_dict = results_dict.get(bar_iloc)
        if ind_dict is not None:
            out_rows.append(ind_dict)
            n_ok += 1
        else:
            orig = df.iloc[bar_iloc]
            row = {"datetime": orig["datetime"], "date": orig["date"],
                   "open": orig["open"], "high": orig["high"],
                   "low": orig["low"], "close": orig["close"],
                   "volume": orig["volume"]}
            out_rows.append(row)
            n_fail += 1

    df_out = pd.DataFrame(out_rows)

    if verbose:
        print(f"  OK: {n_ok:,} bars | Failed: {n_fail:,} bars", flush=True)

    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute streaming-style indicators for all bars",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-file", type=str, default=None,
        help="Path to 5-min bar CSV (default: auto-resolved from --ticker)",
    )
    parser.add_argument(
        "--ticker", type=str, default="tsla",
        help="Ticker symbol to resolve data file (default: tsla). "
             "Auto-resolves to data/<ticker>_5min_10years.csv",
    )
    parser.add_argument(
        "--date-range", type=str, default=None,
        help="Date range to compute, format 'YYYY-MM-DD:YYYY-MM-DD' "
             "(e.g., '2022-01-01:2023-12-31'). Warmup context is added "
             "automatically. If omitted, computes all bars.",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Warmup bars before prior-day open (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel worker processes (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: auto-generated in data/)",
    )
    parser.add_argument(
        "--format", choices=["parquet", "csv"], default="parquet",
        help="Output format (default: parquet)",
    )

    args = parser.parse_args()

    # Resolve data file: explicit --data-file takes precedence over --ticker
    if args.data_file:
        data_path = Path(args.data_file)
        ticker = data_path.stem.split("_")[0].lower()
    else:
        data_path = resolve_ticker_data_file(args.ticker)
        ticker = args.ticker.lower()

    print(f"Loading data from {data_path}...", flush=True)
    df = load_data(data_path)
    print(f"Loaded {len(df):,} bars ({ticker.upper()})", flush=True)

    # Apply date range filter if specified
    target_dates_set = None
    if args.date_range:
        start_date, end_date = parse_date_range(args.date_range)
        print(f"\nFiltering to date range with warmup...", flush=True)
        df, target_dates_set = filter_with_warmup(
            df, start_date, end_date,
            warmup_bars=args.warmup,
            verbose=True,
        )

    # Compute streaming indicators
    df_streaming = precompute_all(
        df,
        warmup_bars=args.warmup,
        n_workers=args.workers,
        verbose=True,
        target_dates_only=target_dates_set,
    )

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        ext = ".parquet" if args.format == "parquet" else ".csv"
        if args.date_range:
            start_date, end_date = parse_date_range(args.date_range)
            out_path = OUTPUT_DIR / f"{ticker}_5min_streaming_{start_date}_{end_date}{ext}"
        else:
            out_path = OUTPUT_DIR / f"{ticker}_5min_streaming_indicators{ext}"

    # Save
    print(f"\nSaving to {out_path}...", flush=True)
    if args.format == "parquet":
        if "datetime" in df_streaming.columns:
            df_streaming["datetime"] = pd.to_datetime(df_streaming["datetime"], utc=True)
        if "date" in df_streaming.columns:
            df_streaming["date"] = pd.to_datetime(df_streaming["date"]).dt.date
        df_streaming.to_parquet(out_path, index=False, engine="pyarrow")
    else:
        df_streaming.to_csv(out_path, index=False)

    print(f"[OK] Saved {len(df_streaming):,} rows to {out_path}", flush=True)

    # Summary statistics
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    n_cols = len([c for c in df_streaming.columns
                  if c not in ["datetime", "date", "open", "high", "low", "close",
                               "volume", "time", "wap", "bar_count", "symbol"]])
    print(f"  Indicator columns: {n_cols}", flush=True)
    print(f"  Rows: {len(df_streaming):,}", flush=True)

    # Show NaN stats for key features
    key_feats = [
        "vwap_width_atr", "vol_pct_complete", "rsi", "ema20_slope_atr",
        "vwap_stretch_zscore", "reversal_quality",
    ]
    print(f"\n  NaN counts for key features:", flush=True)
    for f in key_feats:
        if f in df_streaming.columns:
            n_nan = int(df_streaming[f].isna().sum())
            pct = n_nan / len(df_streaming) * 100
            print(f"    {f:30s}: {n_nan:6,} ({pct:.1f}%)", flush=True)

    # Sample indicator values
    if len(df_streaming) > 100:
        print(f"\n  Sample indicator values (bar 100):", flush=True)
        sample = df_streaming.iloc[100]
        for f in ["vwap_width_atr", "rsi", "vol_pct_complete", "vwap_stretch_zscore",
                   "ema20_slope_atr", "reversal_quality"]:
            if f in df_streaming.columns:
                v = sample[f]
                if pd.notna(v):
                    print(f"    {f:30s}: {v:.4f}", flush=True)
                else:
                    print(f"    {f:30s}: NaN", flush=True)

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
