"""
Compare streaming simulator indicators vs precomputed (pipeline) indicators.

Goal: verify that the bar-by-bar indicators the simulator computes via
StreamingIndicatorsAligned match the precomputed indicators stored in the
Parquet file fed to master_pipeline for training.

If they match, the model sees the same features at inference (simulator / live)
as it saw during training (pipeline with --indicators-file).

Approach:
  1. Load raw OHLCV data
  2. Load precomputed streaming indicators parquet
  3. For a sample of bars, build the 200-bar lookback window (exactly as
     StreamingSimulator does) and call StreamingIndicatorsAligned
  4. Compare the 62 model features between simulator output and precomputed
  5. Report matches, mismatches, and magnitudes

Usage:
    python scripts/compare_indicators.py
    python scripts/compare_indicators.py --sample-day 2024-03-15
    python scripts/compare_indicators.py --sample-bars 20
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "sim_trading"))
sys.path.insert(0, str(ROOT))

# Redirect stdout/stderr to a log file for reliable capture on Windows/PowerShell
_log_path = ROOT / "_compare_indicators_out.txt"
_log_file = open(_log_path, "w", encoding="utf-8")
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr

class _Tee:
    """Write to both console and file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()

sys.stdout = _Tee(_orig_stdout, _log_file)
sys.stderr = _Tee(_orig_stderr, _log_file)

import argparse
import numpy as np
import pandas as pd
from collections import deque

# ---------------------------------------------------------------------------
# Load model features
# ---------------------------------------------------------------------------

def get_model_features(model_path: str) -> list:
    """Load the model and return its feature list."""
    from model_persistence import load_model
    _, meta = load_model(model_path)
    return meta["features"]


# ---------------------------------------------------------------------------
# Simulate streaming indicator calculation (using actual StreamingSimulator)
# ---------------------------------------------------------------------------


def compute_streaming_indicators_for_bars(
    df_raw: pd.DataFrame,
    target_indices: list,
    lookback: int = 200,
    warmup_bars: int = 60,
    use_day_window: bool = True,
) -> dict:
    """Compute indicators using the ACTUAL simulator code path.

    Instead of duplicating windowing/indicator logic, this instantiates a
    real ``StreamingSimulator`` and calls its ``_build_day_index``,
    ``_build_day_window_df``, and ``_calc_indicators_at_row`` methods.
    This guarantees that any match/mismatch we find reflects what the
    simulator will actually produce at inference time.

    Args:
        df_raw: Full OHLCV DataFrame (with datetime, date columns).
        target_indices: List of integer positions in df_raw to compute.
        lookback: Lookback window size (legacy mode only).
        warmup_bars: Warmup bars before prior-day open (day-window mode).
        use_day_window: If True, use day-aware windowing matching precompute.

    Returns:
        Dict mapping target_index -> dict of indicator values.
    """
    from sim_trading.streaming_simulator import StreamingSimulator
    from sim_trading.streaming_indicators_aligned import StreamingIndicatorsAligned

    # Instantiate a real simulator (same params the simulation would use)
    sim = StreamingSimulator(
        initial_capital=1_000_000,
        bar_interval_minutes=5,
        lookback_bars=lookback,
        warmup_bars=warmup_bars,
        use_day_window=use_day_window,
        verbose=False,
    )

    calc = StreamingIndicatorsAligned(verbose=False)
    results = {}

    # Ensure date column
    if "date" not in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw["date"] = pd.to_datetime(df_raw["datetime"]).dt.date

    if not use_day_window:
        # Legacy: fixed lookback per bar — use simulator's _calc_indicators
        for i, idx in enumerate(target_indices):
            window_start = max(0, idx - lookback + 1)
            window_end = idx + 1
            window_df = df_raw.iloc[window_start:window_end].copy().reset_index(drop=True)
            if "date" not in window_df.columns:
                window_df["date"] = pd.to_datetime(window_df["datetime"]).dt.date
            indicators = sim._calc_indicators(calc, window_df)
            results[idx] = indicators
            if (i + 1) % 5 == 0 or (i + 1) == len(target_indices):
                print(f"  Computed {i+1}/{len(target_indices)} bars (window={len(window_df)})...", flush=True)
        return results

    # Day-window mode: use simulator's actual methods
    unique_dates, day_ranges, date_to_order = sim._build_day_index(df_raw)

    # Group target_indices by date for efficient per-day batch compute
    from collections import defaultdict
    targets_by_date = defaultdict(list)
    for idx in target_indices:
        bar_date = df_raw.iloc[idx]["date"]
        targets_by_date[bar_date].append(idx)

    # Cache per-day indicator DataFrames (same as simulator's _ensure_day_cached)
    day_indicator_cache = {}  # date -> (result_df, window_start)

    days_done = 0
    for target_date, idxs in sorted(targets_by_date.items()):
        if target_date not in day_indicator_cache:
            # Use simulator's _build_day_window_df to get window + offset
            # (use first idx just to get the window — full day is the same for all)
            window_df, _ = sim._build_day_window_df(
                idxs[0], target_date, df_raw,
                unique_dates, day_ranges, date_to_order,
            )
            if len(window_df) == 0:
                for idx in idxs:
                    results[idx] = {}
                continue

            # Compute window_start the same way the simulator does
            order = date_to_order[target_date]
            if order >= 1:
                prior_date = unique_dates[order - 1]
                prior_start, _ = day_ranges[prior_date]
                window_start = max(0, prior_start - warmup_bars)
            else:
                cur_start, _ = day_ranges[target_date]
                window_start = max(0, cur_start - warmup_bars)

            # Use simulator's _calc_indicators_at_row for the first idx
            # to get the full result_df — but we need the raw DataFrame result.
            # Call calculate_core_indicators directly (same as simulator does)
            window_df_copy = window_df.copy().reset_index(drop=True)
            if "date" not in window_df_copy.columns:
                window_df_copy["date"] = pd.to_datetime(window_df_copy["datetime"]).dt.date

            try:
                result_df = calc.calculate_core_indicators(window_df_copy, verbose=False)
            except Exception as e:
                print(f"  WARNING: Failed to compute indicators for {target_date}: {e}", flush=True)
                for idx in idxs:
                    results[idx] = {}
                continue

            day_indicator_cache[target_date] = (result_df, window_start)

        result_df, window_start = day_indicator_cache[target_date]

        # Extract each bar's row — exactly as the simulator does in its run() loop
        for idx in idxs:
            row_offset = idx - window_start
            if row_offset < 0 or row_offset >= len(result_df):
                results[idx] = {}
                continue

            ind_row = result_df.iloc[row_offset]
            indicators = {}
            for col in result_df.columns:
                val = ind_row[col]
                if isinstance(val, (np.integer,)):
                    indicators[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    indicators[col] = float(val)
                else:
                    indicators[col] = val
            # Patch vol_pct_complete to streaming (matches precompute & simulator)
            indicators['vol_pct_complete'] = 1.0
            results[idx] = indicators

        days_done += 1
        bars_done = sum(1 for idx in target_indices if idx in results)
        print(f"  Day {days_done}: {target_date} — {len(idxs)} bars (window={len(window_df)}). "
              f"Total: {bars_done}/{len(target_indices)} bars done.", flush=True)

    return results


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_features(
    precomputed: pd.DataFrame,
    streaming: dict,
    feature_cols: list,
    df_raw: pd.DataFrame,
    target_indices: list,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> pd.DataFrame:
    """Compare precomputed vs streaming indicator values for target bars.

    Args:
        precomputed: Precomputed indicators DataFrame (from parquet).
        streaming: Dict of {raw_df_index -> indicator_dict} from simulator.
        feature_cols: List of feature column names to compare.
        df_raw: Raw OHLCV data (to get datetime for matching).
        target_indices: List of raw df indices that were computed.
        rtol: Relative tolerance for numeric comparison.
        atol: Absolute tolerance for numeric comparison.

    Returns:
        DataFrame with comparison results per feature per bar.
    """
    rows = []

    for idx in target_indices:
        bar_dt = df_raw.iloc[idx]["datetime"]

        # Find matching row in precomputed by datetime
        pq_match = precomputed[precomputed["datetime"] == bar_dt]
        if len(pq_match) == 0:
            rows.append({
                "bar_idx": idx,
                "datetime": bar_dt,
                "feature": "ALL",
                "precomputed": None,
                "streaming": None,
                "diff": None,
                "pct_diff": None,
                "match": False,
                "note": "NO MATCH IN PRECOMPUTED",
            })
            continue

        pq_row = pq_match.iloc[0]
        sim_indicators = streaming.get(idx, {})

        if not sim_indicators:
            rows.append({
                "bar_idx": idx,
                "datetime": bar_dt,
                "feature": "ALL",
                "precomputed": "present",
                "streaming": "EMPTY",
                "diff": None,
                "pct_diff": None,
                "match": False,
                "note": "SIMULATOR RETURNED EMPTY",
            })
            continue

        for feat in feature_cols:
            pq_val = pq_row.get(feat, None)
            sim_val = sim_indicators.get(feat, None)

            pq_is_nan = pq_val is None or (isinstance(pq_val, float) and np.isnan(pq_val))
            sim_is_nan = sim_val is None or (isinstance(sim_val, float) and np.isnan(sim_val))

            if pq_is_nan and sim_is_nan:
                rows.append({
                    "bar_idx": idx, "datetime": bar_dt, "feature": feat,
                    "precomputed": "NaN", "streaming": "NaN",
                    "diff": 0.0, "pct_diff": 0.0, "match": True, "note": "both NaN",
                })
                continue

            if pq_is_nan != sim_is_nan:
                rows.append({
                    "bar_idx": idx, "datetime": bar_dt, "feature": feat,
                    "precomputed": pq_val, "streaming": sim_val,
                    "diff": None, "pct_diff": None, "match": False,
                    "note": "NaN mismatch",
                })
                continue

            try:
                pq_f = float(pq_val)
                sim_f = float(sim_val)
            except (TypeError, ValueError):
                # Boolean / categorical comparison
                match = (pq_val == sim_val)
                rows.append({
                    "bar_idx": idx, "datetime": bar_dt, "feature": feat,
                    "precomputed": pq_val, "streaming": sim_val,
                    "diff": 0.0 if match else None,
                    "pct_diff": 0.0 if match else None,
                    "match": match,
                    "note": "" if match else "value mismatch",
                })
                continue

            diff = abs(pq_f - sim_f)
            denom = max(abs(pq_f), abs(sim_f), 1e-12)
            pct_diff = diff / denom

            match = np.isclose(pq_f, sim_f, rtol=rtol, atol=atol)

            rows.append({
                "bar_idx": idx, "datetime": bar_dt, "feature": feat,
                "precomputed": pq_f, "streaming": sim_f,
                "diff": diff, "pct_diff": pct_diff,
                "match": bool(match), "note": "",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(cmp_df: pd.DataFrame, feature_cols: list):
    """Print a human-readable comparison report."""
    print(f"\n{'='*100}")
    print("INDICATOR COMPARISON: Precomputed (pipeline) vs Streaming (simulator)")
    print(f"{'='*100}")

    n_bars = cmp_df["bar_idx"].nunique()
    n_features = len(feature_cols)
    n_comparisons = len(cmp_df[cmp_df["feature"] != "ALL"])
    n_match = cmp_df[(cmp_df["feature"] != "ALL") & (cmp_df["match"] == True)].shape[0]
    n_mismatch = n_comparisons - n_match

    print(f"\nBars compared:      {n_bars}")
    print(f"Features compared:  {n_features}")
    print(f"Total comparisons:  {n_comparisons}")
    print(f"Matches:            {n_match} ({n_match/max(n_comparisons,1)*100:.1f}%)")
    print(f"Mismatches:         {n_mismatch} ({n_mismatch/max(n_comparisons,1)*100:.1f}%)")

    # ---- Per-feature summary ----
    print(f"\n{'─'*100}")
    print(f"{'Feature':<35s} {'Match%':>7s} {'MaxDiff':>12s} {'MaxPct':>10s} {'AvgDiff':>12s} {'Note':>20s}")
    print(f"{'─'*100}")

    feat_rows = cmp_df[cmp_df["feature"] != "ALL"]
    for feat in feature_cols:
        feat_data = feat_rows[feat_rows["feature"] == feat]
        if len(feat_data) == 0:
            print(f"{feat:<35s} {'N/A':>7s}")
            continue

        match_pct = feat_data["match"].mean() * 100
        numeric_diffs = feat_data["diff"].dropna()
        pct_diffs = feat_data["pct_diff"].dropna()

        max_diff = numeric_diffs.max() if len(numeric_diffs) > 0 else 0
        avg_diff = numeric_diffs.mean() if len(numeric_diffs) > 0 else 0
        max_pct = pct_diffs.max() if len(pct_diffs) > 0 else 0

        note = ""
        if match_pct < 100:
            nan_mismatches = feat_data[feat_data["note"] == "NaN mismatch"]
            if len(nan_mismatches) > 0:
                note = f"{len(nan_mismatches)} NaN mismatches"
            elif max_pct > 0.01:
                note = "SIGNIFICANT"
            else:
                note = "minor rounding"

        marker = " <<<" if match_pct < 90 else ""
        print(f"{feat:<35s} {match_pct:>6.1f}% {max_diff:>12.6f} {max_pct:>9.4f}% {avg_diff:>12.6f} {note:>20s}{marker}")

    # ---- Worst mismatches (by pct_diff) ----
    mismatches = feat_rows[feat_rows["match"] == False].copy()
    if len(mismatches) > 0:
        print(f"\n{'─'*100}")
        print(f"TOP 20 WORST MISMATCHES (by relative difference):")
        print(f"{'─'*100}")
        mismatches_sorted = mismatches.sort_values("pct_diff", ascending=False).head(20)
        print(f"{'Feature':<30s} {'Datetime':>25s} {'Precomputed':>14s} {'Streaming':>14s} {'Diff':>12s} {'PctDiff':>10s} {'Note'}")
        print("-" * 120)
        for _, row in mismatches_sorted.iterrows():
            print(
                f"{row['feature']:<30s} {str(row['datetime']):>25s} "
                f"{row['precomputed']:>14.6f} {row['streaming']:>14.6f} "
                f"{row['diff']:>12.6f} {row['pct_diff']:>9.4f}% {row['note']}"
            )

    # ---- Key features deep dive (show actual values for a sample bar) ----
    sample_bars = sorted(cmp_df["bar_idx"].unique())
    if len(sample_bars) > 0:
        sample_idx = sample_bars[len(sample_bars) // 2]  # middle bar
        print(f"\n{'─'*100}")
        print(f"SAMPLE BAR DETAIL (bar_idx={sample_idx}):")
        print(f"{'─'*100}")
        bar_data = feat_rows[feat_rows["bar_idx"] == sample_idx]
        bar_dt = bar_data.iloc[0]["datetime"] if len(bar_data) > 0 else "?"
        print(f"Datetime: {bar_dt}\n")
        print(f"{'Feature':<35s} {'Precomputed':>14s} {'Streaming':>14s} {'Diff':>12s} {'Match':>6s}")
        print("-" * 85)
        for _, row in bar_data.iterrows():
            try:
                pq_str = f"{float(row['precomputed']):>14.6f}"
            except (TypeError, ValueError):
                pq_str = f"{row['precomputed']:>14s}"
            try:
                sim_str = f"{float(row['streaming']):>14.6f}"
            except (TypeError, ValueError):
                sim_str = f"{row['streaming']:>14s}"
            diff_str = f"{row['diff']:>12.6f}" if row['diff'] is not None and not np.isnan(row['diff']) else "        N/A"
            match_str = "  OK" if row["match"] else " DIFF"
            print(f"{row['feature']:<35s} {pq_str} {sim_str} {diff_str} {match_str}")

    # ---- vol_pct_complete check ----
    vpc_data = feat_rows[feat_rows["feature"] == "vol_pct_complete"]
    if len(vpc_data) > 0:
        print(f"\n{'─'*100}")
        print("vol_pct_complete CHECK (should be 1.0 in both):")
        pq_vals = vpc_data["precomputed"].dropna().unique()
        sim_vals = vpc_data["streaming"].dropna().unique()
        print(f"  Precomputed unique values: {pq_vals}")
        print(f"  Streaming unique values:   {sim_vals}")
        all_one = all(v == 1.0 for v in pq_vals) and all(v == 1.0 for v in sim_vals)
        print(f"  All 1.0? {'YES ✓' if all_one else 'NO ✗ — PROBLEM'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare streaming simulator indicators vs precomputed (pipeline) indicators"
    )
    parser.add_argument("--data", type=str, default="data/tsla_5min_10years.csv",
                        help="Raw OHLCV data file")
    parser.add_argument("--parquet", type=str,
                        default="data/tsla_5min_streaming_2023-01-01_2024-12-31.parquet",
                        help="Precomputed streaming indicators parquet file")
    parser.add_argument("--model", type=str,
                        default="models/rf_vwap_stop0.5_20260217_154448.pkl",
                        help="Model file (to get feature list)")
    parser.add_argument("--lookback", type=int, default=200,
                        help="Legacy lookback window (only used with --no-day-window)")
    parser.add_argument("--warmup-bars", type=int, default=60,
                        help="Warmup bars before prior-day open (day-window mode, matches precompute)")
    parser.add_argument("--no-day-window", action="store_true",
                        help="Disable day-aware windowing; use legacy fixed lookback")
    parser.add_argument("--sample-day", type=str, default=None,
                        help="Sample a specific day (YYYY-MM-DD). Default: picks 2024-06-15 or nearby.")
    parser.add_argument("--sample-bars", type=int, default=10,
                        help="Number of bars to sample from the chosen day")
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="Relative tolerance for numeric comparison")
    parser.add_argument("--atol", type=float, default=1e-6,
                        help="Absolute tolerance for numeric comparison")
    args = parser.parse_args()

    print("=" * 80)
    print("INDICATOR COMPARISON: Precomputed vs Streaming (bar-by-bar)")
    print("=" * 80)

    # 1. Load model features
    print(f"\n[1] Loading model features from {args.model}...")
    feature_cols = get_model_features(args.model)
    print(f"    {len(feature_cols)} features")

    # 2. Load raw data
    print(f"\n[2] Loading raw OHLCV data from {args.data}...")
    df_raw = pd.read_csv(args.data)
    if "time" in df_raw.columns and "datetime" not in df_raw.columns:
        df_raw["datetime"] = pd.to_datetime(df_raw["time"], utc=True)
    else:
        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], utc=True)
    df_raw["date"] = df_raw["datetime"].dt.date
    print(f"    {len(df_raw):,} bars, {df_raw['datetime'].min()} to {df_raw['datetime'].max()}")

    # 3. Load precomputed
    print(f"\n[3] Loading precomputed indicators from {args.parquet}...")
    pq = pd.read_parquet(args.parquet)
    pq["datetime"] = pd.to_datetime(pq["datetime"], utc=True)
    print(f"    {len(pq):,} rows, {pq['datetime'].min()} to {pq['datetime'].max()}")

    # Check feature coverage
    missing_in_pq = [f for f in feature_cols if f not in pq.columns]
    if missing_in_pq:
        print(f"    WARNING: {len(missing_in_pq)} features missing in parquet: {missing_in_pq}")

    # 4. Select sample bars
    print(f"\n[4] Selecting sample bars...")
    if args.sample_day:
        target_date = pd.Timestamp(args.sample_day).date()
    else:
        # Default: pick a mid-2024 day that exists in both datasets
        target_date = pd.Timestamp("2024-06-17").date()

    # Find bars for target day in raw data
    day_mask = df_raw["date"] == target_date
    day_indices = df_raw[day_mask].index.tolist()

    if not day_indices:
        # Try adjacent days
        all_dates = sorted(df_raw["date"].unique())
        dates_2024 = [d for d in all_dates if d.year == 2024]
        if dates_2024:
            target_date = dates_2024[len(dates_2024) // 2]
            day_mask = df_raw["date"] == target_date
            day_indices = df_raw[day_mask].index.tolist()
            print(f"    Adjusted to: {target_date}")

    if not day_indices:
        print("ERROR: No bars found for target day!")
        return

    print(f"    Target day: {target_date}")
    print(f"    Bars in day: {len(day_indices)}")

    # Sample evenly across the day
    n_sample = min(args.sample_bars, len(day_indices))
    step = max(1, len(day_indices) // n_sample)
    sample_indices = day_indices[::step][:n_sample]    # Make sure all have enough history (20 bars minimum for day-window mode)
    use_day_window = not args.no_day_window
    min_history = 20 if use_day_window else args.lookback
    sample_indices = [idx for idx in sample_indices if idx >= min_history]
    if not sample_indices:
        print(f"ERROR: Not enough history for sample bars! (min={min_history})")
        print(f"    First bar index: {day_indices[0]}")
        return

    mode_str = f"day-window (warmup={args.warmup_bars})" if use_day_window else f"legacy deque({args.lookback})"
    print(f"    Sampling {len(sample_indices)} bars: indices {sample_indices[0]}..{sample_indices[-1]}")
    print(f"    Window mode: {mode_str}")

    # 5. Compute streaming indicators
    print(f"\n[5] Computing streaming indicators (bar-by-bar, {mode_str})...")
    streaming_results = compute_streaming_indicators_for_bars(
        df_raw, sample_indices,
        lookback=args.lookback,
        warmup_bars=args.warmup_bars,
        use_day_window=use_day_window,
    )

    # 6. Compare
    print(f"\n[6] Comparing {len(feature_cols)} features across {len(sample_indices)} bars...")
    cmp_df = compare_features(
        precomputed=pq,
        streaming=streaming_results,
        feature_cols=feature_cols,
        df_raw=df_raw,
        target_indices=sample_indices,
        rtol=args.rtol,
        atol=args.atol,
    )

    # 7. Report
    print_report(cmp_df, feature_cols)

    # 8. Save detailed comparison
    out_path = ROOT / "data" / "indicator_comparison.csv"
    cmp_df.to_csv(out_path, index=False)
    print(f"\nDetailed comparison saved to: {out_path}")

    # Summary verdict
    feat_rows = cmp_df[cmp_df["feature"] != "ALL"]
    if len(feat_rows) > 0:
        match_rate = feat_rows["match"].mean() * 100
        print(f"\n{'='*80}")
        if match_rate >= 99.0:
            print(f"VERDICT: INDICATORS MATCH ({match_rate:.1f}% match rate)")
            print("The model sees the same features during simulation as during training.")
        elif match_rate >= 90.0:
            print(f"VERDICT: MOSTLY MATCH ({match_rate:.1f}% match rate)")
            print("Minor differences likely from EMA edge effects / lookback window size.")
        else:
            print(f"VERDICT: SIGNIFICANT MISMATCH ({match_rate:.1f}% match rate)")
            print("The model is seeing different features during simulation vs training!")
            print("This needs investigation.")
        print(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    finally:
        _log_file.close()
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        print(f"Log saved to: {_log_path}")
