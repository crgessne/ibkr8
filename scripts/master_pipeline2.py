r"""master_pipeline2.py

A clean, deterministic end-to-end pipeline for the VWAP reversion RF strategy.

Goals:
- Train a RandomForestClassifier on the same per-bar indicators.
- Evaluate with *realized-path* execution (VWAP/stop/EOD) and explicit trading costs.
- Output a summary CSV/markdown that clearly documents:
  - label win rate (target touched first)
  - net-profitable win rate (net_pnl > 0)
  - realized net P&L

Design choices (explicit):
- Entry at bar close.
- Direction: long if close < vwap else short.
- Target: vwap.
- Stop: entry +/- stop_atr * atr.
- Forward scan starts at next bar (j+1) and is restricted to the same day.
- If both stop and target touched on same bar, stop is assumed first.
- If neither touched by EOD, exit at EOD close.
- Costs: round-trip (entry+exit) commission+slippage per share.

Usage:
  .\.venv\Scripts\python.exe scripts\master_pipeline2.py --year 2024 --auto-threshold

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import heapq

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


DATA_FILE = Path("data/tsla_5min_10years.csv")

# Defaults to match the rest of the project
SHARES_PER_TRADE = 100
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_PER_SHARE = 0.01
CAPITAL_CAP = 1_000_000
RISK_PCT_PER_TRADE = 1.0  # e.g. 1.0% of current equity per trade

# Model params for the *net-profitable* classification objective.
# (We are not using label_sX for training; it's only reported.)
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_leaf": 40,
    "min_samples_split": 100,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",
}


def _fmt_secs(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


@dataclass(frozen=True)
class ExecCosts:
    commission_per_share: float
    slippage_per_share: float

    def round_trip_cost(self, shares: int) -> float:
        return 2.0 * (self.commission_per_share + self.slippage_per_share) * float(shares)


def calculate_shares_for_risk(*, entry_price: float, risk_per_share: float, account_equity: float, risk_pct: float) -> int:
    """Calculate shares to risk a specific percentage of equity."""
    risk_dollars = account_equity * (risk_pct / 100.0)
    if risk_per_share <= 0:
        return 0
    shares = int(risk_dollars / risk_per_share)
    return shares


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column in CSV")

    df["datetime"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["datetime"].dt.date
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ATR(14)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # VWAP (daily reset)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    df["vwap"] = (
        df.groupby("date", sort=False)
        .apply(lambda g: (pv.loc[g.index].cumsum() / df.loc[g.index, "volume"].cumsum()))
        .reset_index(level=0, drop=True)
    )

    # Core features (keep in sync with project)
    df["vwap_width_atr"] = (df["close"] - df["vwap"]).abs() / df["atr"]
    df["price_to_vwap_atr"] = (df["close"] - df["vwap"]) / df["atr"]
    df["is_long_setup"] = (df["close"] < df["vwap"]).astype(int)

    df["vwap_slope"] = df["vwap"].diff(1)
    df["vwap_slope_5"] = df["vwap"].diff(5)
    df["vwap_helping"] = np.where(df["close"] < df["vwap"], df["vwap_slope"] < 0, df["vwap_slope"] > 0).astype(int)

    df["rel_vol"] = df["volume"] / df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["volume"].shift(1)
    df["vol_at_extension"] = df["volume"] / df["volume"].rolling(5).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_slope"] = df["rsi"].diff(3)
    df["rsi_extreme"] = ((df["rsi"] < 30) | (df["rsi"] > 70)).astype(int)

    df["bar_range_atr"] = (df["high"] - df["low"]) / df["atr"]
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

    crossed = (df["is_long_setup"] != df["is_long_setup"].shift(1)).astype(int)
    df["bars_from_vwap"] = df.groupby((crossed == 1).cumsum()).cumcount()

    # --- New Features ---
    # Exponential Moving Averages (short-term trend context)
    ema9 = df["close"].ewm(span=9, adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    df["dist_ema9_atr"] = (df["close"] - ema9) / df["atr"]
    df["dist_ema21_atr"] = (df["close"] - ema21) / df["atr"]

    # Momentum / Returns (log returns)
    # Fillna(0) to handle the first bar
    log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df["log_ret_1"] = log_ret
    df["log_ret_3"] = log_ret.rolling(3).sum().fillna(0.0)
    df["log_ret_10"] = log_ret.rolling(10).sum().fillna(0.0)

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "time",
        "datetime",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "atr",
        "year",
    }
    feats: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype.kind in {"i", "u", "f", "b"}:
            feats.append(c)
    return feats


def realized_exit_for_entry(
    *,
    entry_idx: int,
    stop_atr: float,
    full_date: np.ndarray,
    full_high: np.ndarray,
    full_low: np.ndarray,
    full_close: np.ndarray,
    entry_close: float,
    entry_vwap: float,
    entry_atr: float,
    want_exit_dt: bool = False,
    full_dt: np.ndarray | None = None,
):
    is_long = bool(entry_close < entry_vwap)
    stop_level = entry_close - stop_atr * entry_atr if is_long else entry_close + stop_atr * entry_atr
    target_level = entry_vwap

    d = full_date[entry_idx]

    # find day end
    n = len(full_date)
    i = entry_idx
    while i + 1 < n and full_date[i + 1] == d:
        i += 1
    day_end = i

    for j in range(entry_idx + 1, day_end + 1):
        hi = float(full_high[j])
        lo = float(full_low[j])

        if is_long:
            hit_stop = lo <= stop_level
            hit_target = hi >= target_level
        else:
            hit_stop = hi >= stop_level
            hit_target = lo <= target_level

        # stop priority
        if hit_stop:
            if want_exit_dt:
                if full_dt is None:
                    raise ValueError("full_dt is required when want_exit_dt=True")
                return float(stop_level), full_dt[j], "stop", is_long
            return float(stop_level), None, "stop", is_long
        if hit_target:
            if want_exit_dt:
                if full_dt is None:
                    raise ValueError("full_dt is required when want_exit_dt=True")
                return float(target_level), full_dt[j], "target", is_long
            return float(target_level), None, "target", is_long

    # EOD close
    if want_exit_dt:
        if full_dt is None:
            raise ValueError("full_dt is required when want_exit_dt=True")
        return float(full_close[day_end]), full_dt[day_end], "eod", is_long
    return float(full_close[day_end]), None, "eod", is_long


def realized_net_pnl_for_entry(
    *,
    entry_idx: int,
    stop_atr: float,
    shares: int,
    costs: ExecCosts,
    full_date: np.ndarray,
    full_high: np.ndarray,
    full_low: np.ndarray,
    full_close: np.ndarray,
    full_vwap: np.ndarray,
    full_atr: np.ndarray,
    dynamic_sizing: bool = False,
    risk_pct: float = 1.0,
    capital: float = 1_000_000.0,
) -> float:
    """Compute realized-path *net* P&L for a hypothetical entry at bar close.

    Performance note:
      - This function is called in a tight loop to build training targets.
      - All arrays must be precomputed once and passed in (no DataFrame access here).
      - No datetime conversions are performed here.
    """
    entry_close = float(full_close[entry_idx])
    entry_vwap = float(full_vwap[entry_idx])
    entry_atr = float(full_atr[entry_idx])

    if not np.isfinite(entry_close) or not np.isfinite(entry_vwap) or not np.isfinite(entry_atr) or entry_atr <= 0:
        return float("nan")

    exit_price, _, _, is_long = realized_exit_for_entry(
        entry_idx=int(entry_idx),
        stop_atr=stop_atr,
        full_date=full_date,
        full_high=full_high,
        full_low=full_low,
        full_close=full_close,
        entry_close=entry_close,
        entry_vwap=entry_vwap,
        entry_atr=entry_atr,
        want_exit_dt=False,
        full_dt=None,
    )

    gross = (exit_price - entry_close) * shares if is_long else (entry_close - exit_price) * shares
    net = gross - costs.round_trip_cost(shares)
    return float(net)


def backtest_realized_path(
    *,
    df: pd.DataFrame,
    proba: np.ndarray,
    proba_threshold: float,
    stop_atr: float,
    fixed_shares: int,
    costs: ExecCosts,
    capital_cap: float,
    use_risk_sizing: bool = False,
    risk_pct: float = 1.0,
    leverage: float = 1.0, 
) -> pd.DataFrame:
    # Use full arrays for scanning (dt only needed for export)
    full_date = df["date"].to_numpy()
    full_dt = df["datetime"].to_numpy()
    full_high = df["high"].to_numpy()
    full_low = df["low"].to_numpy()
    full_close = df["close"].to_numpy()
    full_vwap = df["vwap"].to_numpy()
    full_atr = df["atr"].to_numpy()

    mask = proba >= proba_threshold
    idxs = np.where(mask)[0]

    potential_trades = []

    for entry_idx in idxs:
        entry_close = float(full_close[entry_idx])
        entry_vwap = float(full_vwap[entry_idx])
        entry_atr = float(full_atr[entry_idx])

        if not np.isfinite(entry_close) or not np.isfinite(entry_vwap) or not np.isfinite(entry_atr) or entry_atr <= 0:
            continue

        exit_price, exit_dt, reason, is_long = realized_exit_for_entry(
            entry_idx=int(entry_idx),
            stop_atr=stop_atr,
            full_date=full_date,
            full_high=full_high,
            full_low=full_low,
            full_close=full_close,
            entry_close=entry_close,
            entry_vwap=entry_vwap,
            entry_atr=entry_atr,
            want_exit_dt=True,
            full_dt=full_dt,
        )

        potential_trades.append({
            "entry_idx": int(entry_idx),
            "entry_dt": full_dt[entry_idx],
            "exit_dt": exit_dt,
            "entry_price": entry_close,
            "exit_price": exit_price,
            "is_long": is_long,
            "reason": reason,
            "entry_atr": entry_atr,
        })

    trades = pd.DataFrame(potential_trades)
    if trades.empty:
        return trades

    trades = trades.sort_values("entry_dt").reset_index(drop=True)
    
    # Active trades: (exit_dt, notional) - used for buying power check
    active: list[tuple[np.ndarray, float]] = []
    
    # Pending exits for PnL realization: (exit_dt, net_pnl)
    # Using a heap to efficiently find closed trades
    pending_exits: list[tuple[float, float]] = [] 
    
    executed = []

    # Dynamic equity tracking
    # Start with initial capital.
    # As trades close (exit_dt < current entry_dt), we add their net_pnl to realized_equity.
    realized_equity = capital_cap
    
    for _, t in trades.iterrows():
        now = t["entry_dt"]
        
        # 1. Update Realized Equity (process closed trades)
        # We use a float representation of time for comparison if possible, or just compare datetimes
        # 'now' is a Timestamp. 
        while pending_exits and pending_exits[0][0] <= now:
            _, pnl = heapq.heappop(pending_exits)
            realized_equity += pnl

        # 2. Update Active Notional (Buying Power usage)
        # Remove trades that have exited from the active list
        active = [(ex, ntn) for (ex, ntn) in active if ex > now]
        active_notional = float(sum(ntn for (_, ntn) in active))

        # Base for risk calc is current realized equity
        account_equity = realized_equity

        if use_risk_sizing:
            entry_price = float(t["entry_price"])
            entry_atr = float(t["entry_atr"])
            
            # Risk per share = distance to stop
            risk_per_share = (entry_price - (entry_price - stop_atr * entry_atr)) if t["is_long"] else ((entry_price + stop_atr * entry_atr) - entry_price)
            risk_per_share = abs(risk_per_share)

            shares = calculate_shares_for_risk(
                entry_price=entry_price,
                risk_per_share=risk_per_share,
                account_equity=account_equity,
                risk_pct=risk_pct
            )
            shares = max(1, shares)
        else:
            shares = fixed_shares
            entry_price = float(t["entry_price"]) # ensure defined

        # 3. Apply Capital/Margin Constraint
        # Limit total simultaneous notional to capital_cap * leverage
        # (If leverage=1.0, this is cash account behavior)
        max_buying_power = capital_cap * leverage
        available_cap = max(0.0, max_buying_power - active_notional)
        
        if entry_price > 0:
            max_affordable_shares = int(available_cap / entry_price)
            shares = min(shares, max_affordable_shares)
        else:
            shares = 0

        if shares > 0:
            notional = float(shares * entry_price)
            gross = (t["exit_price"] - t["entry_price"]) * shares if t["is_long"] else (t["entry_price"] - t["exit_price"]) * shares
            c = costs.round_trip_cost(shares)
            net = gross - c

            t_exec = t.to_dict()
            t_exec["shares"] = shares
            t_exec["gross_pnl"] = gross
            t_exec["costs"] = c
            t_exec["net_pnl"] = net
            t_exec["net_win"] = int(net > 0)
            t_exec["label_win"] = int(t["reason"] == "target")
            t_exec["notional"] = notional
            t_exec["equity_start"] = account_equity # Debug info

            executed.append(t_exec)
            
            active.append((t["exit_dt"], notional))
            heapq.heappush(pending_exits, (t["exit_dt"], net))

    return pd.DataFrame(executed)


def select_threshold_by_validation(
    *,
    df: pd.DataFrame,
    proba_full: np.ndarray,
    valid_mask: np.ndarray,
    stop_atr: float,
    shares: int,
    costs: ExecCosts,
    capital_cap: float,
    grid: np.ndarray,
    progress: bool,
    use_risk_sizing: bool,
    risk_pct: float,
) -> tuple[float, pd.DataFrame]:
    """Pick proba threshold that maximizes validation net P&L."""
    best_t = float(grid[0])
    best_pnl = -np.inf
    best_table = []

    proba_valid_only = np.full(len(proba_full), -np.inf, dtype=float)
    proba_valid_only[valid_mask] = proba_full[valid_mask]

    t0 = time.time()
    for idx, t in enumerate(grid, 1):
        trades = backtest_realized_path(
            df=df,
            proba=proba_valid_only,
            proba_threshold=float(t),
            stop_atr=stop_atr,
            fixed_shares=shares,
            costs=costs,
            capital_cap=capital_cap,
            use_risk_sizing=use_risk_sizing,
            risk_pct=risk_pct,
        )

        pnl = float(trades["net_pnl"].sum()) if not trades.empty else 0.0
        n = int(len(trades))
        wr_net = float(trades["net_win"].mean()) if n else float("nan")

        best_table.append({"threshold": float(t), "trades": n, "net_pnl": pnl, "wr_net": wr_net})

        if pnl > best_pnl:
            best_pnl = pnl
            best_t = float(t)

        if progress:
            elapsed = time.time() - t0
            print(
                f"Threshold sweep [{idx}/{len(grid)}] t={float(t):.3f} | trades={n:,} | net=${pnl:,.0f} | best_t={best_t:.3f} best_net=${best_pnl:,.0f} | elapsed {_fmt_secs(elapsed)}",
                flush=True,
            )

    return best_t, pd.DataFrame(best_table).sort_values("threshold").reset_index(drop=True)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(log_loss(y_true, y_score, labels=[0, 1]))
    except Exception:
        return float("nan")


def compute_classification_metrics(*, y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    base_rate = float(np.mean(y_true)) if len(y_true) else float("nan")
    auc = _safe_auc(y_true, proba)
    ll = _safe_logloss(y_true, proba)
    brier = float(brier_score_loss(y_true, proba)) if len(y_true) else float("nan")

    pred = (proba >= float(threshold)).astype(int)
    acc = float(accuracy_score(y_true, pred)) if len(y_true) else float("nan")

    try:
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    except Exception:
        tn = fp = fn = tp = 0

    return {
        "n": int(len(y_true)),
        "base_rate": base_rate,
        "auc": auc,
        "logloss": ll,
        "brier": brier,
        "acc@0.5": acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _threshold_sweep_table_for_test(
    *,
    df: pd.DataFrame,
    test_mask: np.ndarray,
    full_proba: np.ndarray,
    stop_atr: float,
    shares: int,
    costs: ExecCosts,
    capital_cap: float,
    thresholds: np.ndarray,
    use_risk_sizing: bool,
    risk_pct: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    proba_test_only = np.full(len(df), -np.inf, dtype=float)
    proba_test_only[test_mask] = full_proba[test_mask]

    for t in thresholds:
        trades = backtest_realized_path(
            df=df,
            proba=proba_test_only,
            proba_threshold=float(t),
            stop_atr=stop_atr,
            fixed_shares=shares,
            costs=costs,
            capital_cap=capital_cap,
            use_risk_sizing=use_risk_sizing,
            risk_pct=risk_pct,
        )

        n = int(len(trades))
        net = float(trades["net_pnl"].sum()) if n else 0.0
        gross = float(trades["gross_pnl"].sum()) if n else 0.0
        wr_net = float(trades["net_win"].mean()) if n else float("nan")
        wr_label = float(trades["label_win"].mean()) if n else float("nan")

        rows.append(
            {
                "threshold": float(t),
                "executed": n,
                "wr_net": wr_net,
                "wr_label": wr_label,
                "gross_pnl": gross,
                "net_pnl": net,
                "return_pct": (net / float(capital_cap) * 100.0) if capital_cap else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def write_markdown_summary(
    *,
    path: Path,
    args: argparse.Namespace,
    trades: pd.DataFrame,
    chosen_threshold: float,
    threshold_source: str,
    model_metrics: dict | None = None,
    test_threshold_table: pd.DataFrame | None = None,
) -> None:
    pnl_def = "realized_path_dollars"
    win_defs = "WR(label)=target-first, WR(net)=net_pnl>0"
    model_target = "classification: P(net_pnl>0) under realized-path execution + costs"

    sizing_desc = f"{args.risk_pct}% risk/trade" if args.use_risk_sizing else f"{args.shares} shares/trade"

    n = int(len(trades))
    net = float(trades["net_pnl"].sum()) if n else 0.0
    gross = float(trades["gross_pnl"].sum()) if n else 0.0
    wr_label = float(trades["label_win"].mean()) if n else float("nan")
    wr_net = float(trades["net_win"].mean()) if n else float("nan")

    stop_hits = int((trades["reason"] == "stop").sum()) if n else 0
    tgt_hits = int((trades["reason"] == "target").sum()) if n else 0
    eod_hits = int((trades["reason"] == "eod").sum()) if n else 0

    lines = [
        "# master_pipeline2 summary",
        "",
        f"- Year (test): {args.year}",
        f"- Train years: [{args.train_start_year}, {args.year})",
        f"- Validation (Option A): {args.year - 1}",
        f"- Stop (ATR): {args.stop_atr}",
        f"- Proba threshold: {chosen_threshold} ({threshold_source})",
        f"- Sizing: {sizing_desc}",
        f"- Capital cap: {args.capital:,.0f}",
        f"- Costs (round-trip): 2*(commission+slippage)*shares = 2*({args.commission}+{args.slippage})*shares",
        "",
        "## Semantics",
        f"- Model target: {model_target}",
        f"- P&L definition: {pnl_def}",
        f"- Win definitions: {win_defs}",
        "",
    ]

    if model_metrics:
        lines.extend(
            [
                "## Model fit diagnostics (classification)",
                "Metrics computed on `net_profitable = (net_pnl>0)` labels.",
                "- AUC/logloss/Brier use probabilities; acc/confusion use threshold=0.5.",
                "",
            ]
        )

        for split_name in ("train", "valid", "test"):
            m = model_metrics.get(split_name)
            if not m:
                continue
            lines.extend(
                [
                    f"### {split_name}",
                    f"- n: {m['n']:,}",
                    f"- base_rate (P(y=1)): {m['base_rate']:.4f}",
                    f"- AUC: {m['auc']:.4f}",
                    f"- logloss: {m['logloss']:.4f}",
                    f"- brier: {m['brier']:.4f}",
                    f"- acc@0.5: {m['acc@0.5']:.4f}",
                    f"- confusion@0.5 (tn fp / fn tp): {m['tn']} {m['fp']} / {m['fn']} {m['tp']}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Results",
            f"- Trades executed: {n:,}",
            f"- WR(label): {wr_label*100:.2f}%",
            f"- WR(net): {wr_net*100:.2f}%",
            f"- Gross P&L: ${gross:,.2f}",
            f"- Net P&L: ${net:,.2f}",
            "",
            "## Exit reasons",
            f"- target: {tgt_hits:,}",
            f"- stop: {stop_hits:,}",
            f"- eod: {eod_hits:,}",
        ]
    )

    if test_threshold_table is not None and not test_threshold_table.empty:
        lines.extend(
            [
                "## Test-year results by probability threshold",
                f"Test year = {args.year}. Backtest uses the same realized-path execution + costs + capital cap.",
                "",
                "| Threshold | Executed | WR(net) | WR(label) | Gross P&L | Net P&L | Return % |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for _, r in test_threshold_table.iterrows():
            lines.append(
                "| "
                f"{float(r['threshold']):.2f} | "
                f"{int(r['executed']):,} | "
                f"{(float(r['wr_net'])*100.0 if np.isfinite(r['wr_net']) else float('nan')):.2f}% | "
                f"{(float(r['wr_label'])*100.0 if np.isfinite(r['wr_label']) else float('nan')):.2f}% | "
                f"${float(r['gross_pnl']):,.2f} | "
                f"${float(r['net_pnl']):,.2f} | "
                f"{float(r['return_pct']):.3f}%"
                " |"
            )

        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_trades(trades: pd.DataFrame) -> dict:
    n = int(len(trades))
    if n == 0:
        return {"trades": 0, "net_pnl": 0.0, "gross_pnl": 0.0, "wr_net": float("nan"), "wr_label": float("nan")}
    return {
        "trades": n,
        "net_pnl": float(trades["net_pnl"].sum()),
        "gross_pnl": float(trades["gross_pnl"].sum()),
        "wr_net": float(trades["net_win"].mean()),
        "wr_label": float(trades["label_win"].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024, help="Test year (exact year, not 2024+)")
    ap.add_argument("--train-start-year", type=int, default=2016)
    ap.add_argument("--stop-atr", type=float, default=1.5)
    ap.add_argument("--proba-threshold", type=float, default=0.5)
    ap.add_argument("--auto-threshold", action="store_true", help="Select threshold on validation year (year-1) to maximize net P&L")
    ap.add_argument("--threshold-grid", type=int, default=37, help="Number of thresholds between 0.05 and 0.95")
    ap.add_argument("--progress-every", type=int, default=25_000, help="Print progress every N bars while precomputing targets")
    ap.add_argument("--progress-seconds", type=float, default=5.0, help="Also print progress at least every N seconds while precomputing targets")
    ap.add_argument("--verbose", action="store_true", help="Print more progress logging")
    ap.add_argument("--shares", type=int, default=SHARES_PER_TRADE, help="Fixed shares per trade (if not using risk sizing)")
    ap.add_argument("--use-risk-sizing", action="store_true", help="Use % risk based sizing instead of fixed shares")
    ap.add_argument("--risk-pct", type=float, default=RISK_PCT_PER_TRADE, help="Risk % per trade (if using risk sizing)")
    ap.add_argument("--capital", type=float, default=CAPITAL_CAP)
    ap.add_argument("--commission", type=float, default=COMMISSION_PER_SHARE)
    ap.add_argument("--slippage", type=float, default=SLIPPAGE_PER_SHARE)
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    ap.add_argument(
        "--target-years",
        type=str,
        default="needed",
        choices=["needed", "all"],
        help="Compute targets for only needed years (train-fit+valid+test) or for all years.",
    )
    ap.add_argument(
        "--test-threshold-range",
        type=str,
        default="",
        help="Optional: report test-year realized-path results across thresholds. Format '0.5:0.8' or '0.5:0.8:0.05' (start:end[:step]).",
    )
    args = ap.parse_args()

    t_start = time.time()

    print(f"Loading: {DATA_FILE}", flush=True)
    df_raw = load_data(DATA_FILE)
    print(f"Loaded {len(df_raw):,} bars", flush=True)

    print("Calculating indicators...", flush=True)
    t0 = time.time()
    df = calculate_indicators(df_raw)
    print(f"Indicators done in {_fmt_secs(time.time() - t0)}", flush=True)

    df["year"] = pd.to_datetime(df["datetime"]).dt.year

    feats = feature_columns(df)
    print(f"Feature columns: {len(feats)}", flush=True)

    costs = ExecCosts(args.commission, args.slippage)
    print(
        "Costs: round-trip per trade = "
        f"{2.0 * (args.commission + args.slippage):.4f} * shares",
        flush=True,
    )

    if args.use_risk_sizing:
        print(f"Sizing: Risk {args.risk_pct}% of ${args.capital:,.0f} per trade", flush=True)
    else:
        print(f"Sizing: Fixed {args.shares} shares per trade", flush=True)

    valid_year = int(args.year - 1)
    if valid_year < args.train_start_year:
        raise ValueError("Validation year (year-1) is before train-start-year")

    train_years_mask = (df["year"] >= args.train_start_year) & (df["year"] < args.year)

    train_fit_mask = train_years_mask

    valid_mask = (df["year"] == valid_year)
    test_mask = (df["year"] == args.year)

    print("Precomputing realized-path net P&L targets (optimized)...", flush=True)

    full_date = df["date"].to_numpy()
    full_high = df["high"].to_numpy()
    full_low = df["low"].to_numpy()
    full_close = df["close"].to_numpy()
    full_vwap = df["vwap"].to_numpy()
    full_atr = df["atr"].to_numpy()

    needed_mask = (train_fit_mask | valid_mask | test_mask).to_numpy()
    compute_mask = needed_mask if (str(args.target_years).lower() == "needed") else np.ones(len(df), dtype=bool)

    net_pnl_all = np.full(len(df), np.nan, dtype=float)
    idxs = np.where(compute_mask)[0]

    t0 = time.time()
    last_print = t0
    pe = int(args.progress_every)
    ps = float(args.progress_seconds)

    for k, i in enumerate(idxs):
        net_pnl_all[i] = realized_net_pnl_for_entry(
            entry_idx=int(i),
            stop_atr=float(args.stop_atr),
            shares=int(args.shares),
            costs=costs,
            full_date=full_date,
            full_high=full_high,
            full_low=full_low,
            full_close=full_close,
            full_vwap=full_vwap,
            full_atr=full_atr,
        )

        now = time.time()
        bars_done = k + 1
        total = len(idxs)
        by_count = pe > 0 and (bars_done == 1 or bars_done % pe == 0 or bars_done == total)
        by_time = ps > 0 and (now - last_print >= ps)
        if by_count or by_time:
            elapsed = now - t0
            rate = bars_done / elapsed if elapsed > 0 else float("inf")
            pct = 100.0 * bars_done / float(total) if total else 100.0
            remaining = (total - bars_done) / rate if rate > 0 else float("inf")

            print(
                f"Targets: {bars_done:,}/{total:,} ({pct:.1f}%) | "
                f"{rate:,.0f} bars/s | elapsed {_fmt_secs(elapsed)} | ETA {_fmt_secs(remaining)}",
                flush=True,
            )
            last_print = now

    print(f"Target precompute done in {_fmt_secs(time.time() - t0)}", flush=True)

    finite_mask = np.isfinite(net_pnl_all)
    y_all = (net_pnl_all > 0).astype(int)

    train_fit_mask = (train_fit_mask.to_numpy()) & finite_mask

    print(
        "Split sizes: "
        f"train_fit={int(train_fit_mask.sum()):,} | valid={int(valid_mask.sum()):,} | test={int(test_mask.sum()):,}",
        flush=True,
    )

    X_train = df.loc[train_fit_mask, feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = y_all[train_fit_mask].astype(int)

    print("Fitting RandomForestClassifier...", flush=True)
    t0 = time.time()
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    print(f"Model fit done in {_fmt_secs(time.time() - t0)}", flush=True)

    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        idxs = np.argsort(imps)[::-1]
        print("Top 10 Feature Importances:", flush=True)
        for i in range(min(10, len(feats))):
            print(f"  {feats[idxs[i]]}: {imps[idxs[i]]:.4f}", flush=True)

    print(f"Predicting probabilities for train ({int(train_fit_mask.sum()):,} bars)...", flush=True)
    t0 = time.time()
    proba_train = model.predict_proba(X_train)[:, 1]
    print(f"Train predict done in {_fmt_secs(time.time() - t0)}", flush=True)

    eval_mask = valid_mask | test_mask
    X_eval = df.loc[eval_mask, feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"Predicting probabilities for valid+test ({int(eval_mask.sum()):,} bars)...", flush=True)
    t0 = time.time()
    proba_eval = model.predict_proba(X_eval)[:, 1]
    print(f"Predict done in {_fmt_secs(time.time() - t0)}", flush=True)

    eval_full_idx = df.index[eval_mask].to_numpy()
    full_proba = np.full(len(df), -np.inf, dtype=float)
    full_proba[eval_full_idx] = proba_eval

    eval_idx_all = df.index[eval_mask].to_numpy()
    is_valid_row = valid_mask.loc[eval_mask].to_numpy()
    proba_valid = proba_eval[is_valid_row]
    proba_test = proba_eval[~is_valid_row]

    y_valid = y_all[valid_mask.to_numpy()].astype(int)
    y_test = y_all[test_mask.to_numpy()].astype(int)

    model_metrics = {
        "train": compute_classification_metrics(y_true=y_train, proba=proba_train, threshold=0.5),
        "valid": compute_classification_metrics(y_true=y_valid, proba=proba_valid, threshold=0.5),
        "test": compute_classification_metrics(y_true=y_test, proba=proba_test, threshold=0.5),
    }

    def _pm(split: str) -> None:
        m = model_metrics[split]
        print(
            f"{split}: n={m['n']:,} base={m['base_rate']:.3f} "
            f"auc={m['auc']:.3f} logloss={m['logloss']:.3f} brier={m['brier']:.3f} acc@0.5={m['acc@0.5']:.3f} | "
            f"cm@0.5 tn={m['tn']} fp={m['fp']} fn={m['fn']} tp={m['tp']}",
            flush=True,
        )

    print("Model fit diagnostics:", flush=True)
    _pm("train")
    _pm("valid")
    _pm("test")

    chosen_threshold = float(args.proba_threshold)
    threshold_source = "cli"

    threshold_table = None
    if bool(args.auto_threshold):
        grid_n = max(5, int(args.threshold_grid))
        grid = np.linspace(0.05, 0.95, grid_n)
        print(
            f"Auto-threshold enabled: validation={valid_year} | grid_n={grid_n} | "
            f"range=[{grid[0]:.2f}, {grid[-1]:.2f}]",
            flush=True,
        )
        chosen_threshold, threshold_table = select_threshold_by_validation(
            df=df,
            proba_full=full_proba,
            valid_mask=valid_mask.to_numpy(),
            stop_atr=float(args.stop_atr),
            shares=int(args.shares),
            costs=costs,
            capital_cap=float(args.capital),
            grid=grid,
            progress=bool(args.verbose),
            use_risk_sizing=bool(args.use_risk_sizing),
            risk_pct=float(args.risk_pct),
        )
        threshold_source = f"auto(validation={valid_year})"

        if not bool(args.verbose):
            best_row = threshold_table.loc[threshold_table["net_pnl"].idxmax()]
            print(
                f"Auto-threshold result: best_t={float(best_row['threshold'])::.3f} | "
                f"valid_net=${float(best_row['net_pnl']):,.0f} | valid_trades={int(best_row['trades']):,}",
                flush=True,
            )

    print(f"Chosen threshold: {chosen_threshold} ({threshold_source})", flush=True)

    test_sweep_thresholds = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80], dtype=float)
    test_threshold_table = _threshold_sweep_table_for_test(
        df=df,
        test_mask=test_mask.to_numpy(),
        full_proba=full_proba,
        stop_atr=float(args.stop_atr),
        shares=int(args.shares),
        costs=costs,
        capital_cap=float(args.capital),
        thresholds=test_sweep_thresholds,
        use_risk_sizing=bool(args.use_risk_sizing),
        risk_pct=float(args.risk_pct),
    )

    proba_test_only = np.full(len(df), -np.inf, dtype=float)
    proba_test_only[test_mask.to_numpy()] = full_proba[test_mask.to_numpy()]

    print(f"Backtesting test year {args.year}...", flush=True)
    t0 = time.time()
    trades = backtest_realized_path(
        df=df,
        proba=proba_test_only,
        proba_threshold=chosen_threshold,
        stop_atr=args.stop_atr,
        fixed_shares=args.shares,
        costs=costs,
        capital_cap=args.capital,
        use_risk_sizing=bool(args.use_risk_sizing),
        risk_pct=float(args.risk_pct),
    )
    print(f"Backtest done in {_fmt_secs(time.time() - t0)}", flush=True)

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    stem = f"master_pipeline2_{args.year}_stop{str(args.stop_atr).replace('.', '_')}_pt{str(chosen_threshold).replace('.', '_')}_{ts}"

    trades_path = outdir / f"{stem}_trades.csv"
    summary_path = outdir / f"{stem}_summary.md"
    thresh_path = outdir / f"{stem}_thresholds.csv"

    trades.to_csv(trades_path, index=False)

    if threshold_table is not None:
        threshold_table.to_csv(thresh_path, index=False)

    if trades.empty:
        print("No trades executed", flush=True)
        write_markdown_summary(
            path=summary_path,
            args=args,
            trades=trades,
            chosen_threshold=chosen_threshold,
            threshold_source=threshold_source,
            model_metrics=model_metrics,
            test_threshold_table=test_threshold_table,
        )
        print(f"Saved summary: {summary_path}", flush=True)
        print(f"Total runtime: {_fmt_secs(time.time() - t_start)}", flush=True)
        return 0

    net = float(trades["net_pnl"].sum())
    wr_label = float(trades["label_win"].mean())
    wr_net = float(trades["net_win"].mean())

    print(f"Trades: {len(trades):,}", flush=True)
    print(f"Net P&L: ${net:,.0f}", flush=True)
    print(f"WR(label): {wr_label*100:.1f}%", flush=True)
    print(f"WR(net): {wr_net*100:.1f}%", flush=True)
    print(f"Saved trades: {trades_path}", flush=True)
    if threshold_table is not None:
        print(f"Saved thresholds: {thresh_path}", flush=True)

    write_markdown_summary(
        path=summary_path,
        args=args,
        trades=trades,
        chosen_threshold=chosen_threshold,
        threshold_source=threshold_source,
        model_metrics=model_metrics,
        test_threshold_table=test_threshold_table,
    )
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Total runtime: {_fmt_secs(time.time() - t_start)}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
