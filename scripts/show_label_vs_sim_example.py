"""Show concrete examples where forward-looking label outcome differs from streaming/concurrent simulation.

This script:
1) Runs concurrent_backtest.py in simulate mode to get executed trades with real exit times/prices.
2) Recomputes the forward-looking label outcome for the same entry bar and stop_atr.
3) Prints a few mismatches with surrounding bars so it's obvious why.

Usage (powershell):
  .\.venv\Scripts\python.exe scripts\show_label_vs_sim_example.py --year 2024 --stop-atr 1.25 --rf-threshold 0.5 --n 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_forward_label_for_entry(df_year: pd.DataFrame, entry_idx: int, stop_atr: float):
    """Match src/label_generator.py logic for one entry index within df_year."""
    bar = df_year.iloc[entry_idx]
    entry_price = float(bar["close"])
    vwap = float(bar["vwap"])
    atr = float(bar["atr"])

    if not np.isfinite(entry_price) or not np.isfinite(vwap) or not np.isfinite(atr) or atr <= 0:
        return np.nan, None, np.nan, pd.NaT, None, np.nan, np.nan

    is_long = entry_price < vwap
    stop_dist = stop_atr * atr

    if is_long:
        stop_price = entry_price - stop_dist
        target_price = vwap
    else:
        stop_price = entry_price + stop_dist
        target_price = vwap

    # within same day only
    d = bar["date"]
    day_mask = df_year["date"] == d
    day_indices = np.flatnonzero(day_mask.to_numpy())
    j_rel = int(np.where(day_indices == entry_idx)[0][0])

    # can't evaluate last bar
    if j_rel + 1 >= len(day_indices):
        return np.nan, None, np.nan, pd.NaT, is_long, stop_price, target_price

    for k_rel in range(j_rel + 1, len(day_indices)):
        k = int(day_indices[k_rel])
        r = df_year.iloc[k]
        hi = float(r["high"])
        lo = float(r["low"])

        if is_long:
            if lo <= stop_price:
                return 0, "stop", stop_price, r["datetime"], is_long, stop_price, target_price
            if hi >= target_price:
                return 1, "target", target_price, r["datetime"], is_long, stop_price, target_price
        else:
            if hi >= stop_price:
                return 0, "stop", stop_price, r["datetime"], is_long, stop_price, target_price
            if lo <= target_price:
                return 1, "target", target_price, r["datetime"], is_long, stop_price, target_price

    # EOD without target => loss
    k = int(day_indices[-1])
    r = df_year.iloc[k]
    return 0, "eod", float(r["close"]), r["datetime"], is_long, stop_price, target_price


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--stop-atr", type=float, default=1.25)
    ap.add_argument("--rf-threshold", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--trades-csv", type=str, default="data/concurrent_backtest_trades_single.csv")
    args = ap.parse_args()

    trades_path = Path(args.trades_csv)
    if not trades_path.exists():
        raise SystemExit(f"Missing trades file: {trades_path}. Run concurrent_backtest.py in simulate mode first.")

    trades = pd.read_csv(trades_path)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)

    # Load and indicator-enrich year data the same way concurrent does
    df = pd.read_csv("data/tsla_5min_10years.csv")
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df_year = df[df["datetime"].dt.year == args.year].reset_index(drop=True)

    # calculate indicators
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).parent.parent / "scripts"))
    from master_pipeline import calculate_core_indicators

    df_year["date"] = df_year["datetime"].dt.date
    df_year = calculate_core_indicators(df_year, verbose=False)

    # Map entry_time -> index
    idx_map = pd.Series(df_year.index.values, index=df_year["datetime"].values)

    examples = []
    for _, t in trades.iterrows():
        et = t["entry_time"].to_datetime64()
        if et not in idx_map.index:
            continue
        entry_idx = int(idx_map.loc[et])

        label, label_reason, label_exit_price, label_exit_time, is_long, stop_price, target_price = compute_forward_label_for_entry(
            df_year, entry_idx, args.stop_atr
        )
        if np.isnan(label):
            continue

        sim_win = 1 if float(t["pnl"]) > 0 else 0
        if int(label) != sim_win:
            examples.append((t, entry_idx, label, label_reason, label_exit_price, label_exit_time, is_long, stop_price, target_price))
            if len(examples) >= args.n:
                break

    if not examples:
        print("No mismatches found between label and simulate outcomes for the provided trades file.")
        return

    for i, (t, entry_idx, label, label_reason, label_exit_price, label_exit_time, is_long, stop_price, target_price) in enumerate(examples, 1):
        entry_bar = df_year.iloc[entry_idx]
        print("=" * 100)
        print(f"Example {i}")
        print(f"Direction: {'LONG' if is_long else 'SHORT'}")
        print(f"Entry idx: {entry_idx}  Entry time: {entry_bar['datetime']}  Entry close: {entry_bar['close']:.2f}")
        print(f"Entry VWAP: {entry_bar['vwap']:.2f}  ATR: {entry_bar['atr']:.4f}  StopATR: {args.stop_atr}")
        print(f"Computed stop: {stop_price:.2f}  target: {target_price:.2f}")
        print("-")
        print(f"LABEL outcome: {int(label)} ({'WIN' if label==1 else 'LOSS'}) via {label_reason} at {label_exit_time} @ {label_exit_price:.2f}")
        print(f"SIM outcome  : {'WIN' if float(t['pnl'])>0 else 'LOSS'} via {t['reason']} at {t['exit_time']} @ {t['exit_price']:.2f}  pnl={t['pnl']:.2f}")

        # Print the next few bars until either label exit or sim exit time
        d = entry_bar["date"]
        day_mask = df_year["date"] == d
        day_indices = np.flatnonzero(day_mask.to_numpy())
        j_rel = int(np.where(day_indices == entry_idx)[0][0])

        # show next up to 12 bars
        print("\nForward bars (next 12 within day):")
        for step in range(1, 13):
            rel = j_rel + step
            if rel >= len(day_indices):
                break
            k = int(day_indices[rel])
            r = df_year.iloc[k]
            flag = []
            if is_long:
                if float(r['low']) <= stop_price:
                    flag.append("STOP")
                if float(r['high']) >= target_price:
                    flag.append("TGT")
            else:
                if float(r['high']) >= stop_price:
                    flag.append("STOP")
                if float(r['low']) <= target_price:
                    flag.append("TGT")

            mark = ""
            if pd.Timestamp(r["datetime"]) == pd.Timestamp(t["exit_time"]):
                mark += " <SIM_EXIT>"
            if pd.Timestamp(r["datetime"]) == pd.Timestamp(label_exit_time):
                mark += " <LABEL_EXIT>"
            print(
                f"  +{step:02d} {r['datetime']}  O:{r['open']:.2f} H:{r['high']:.2f} L:{r['low']:.2f} C:{r['close']:.2f}"
                + (f"  [{'|'.join(flag)}]" if flag else "")
                + mark
            )

    print("=" * 100)


if __name__ == "__main__":
    main()
