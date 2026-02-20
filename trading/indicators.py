"""
Indicator Calculator for live / paper trading.

Wraps the same ``calculate_core_indicators`` used by master_pipeline.py
so that feature vectors match exactly.

Maintains a rolling window of bars (deque) and recalculates indicators
on every new bar.  Only the *last row* (current bar) is used for signals.
"""

from __future__ import annotations

import logging
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_log = logging.getLogger("trading.indicators")
# Also write indicator errors to the bar-debug log file
_bar_log = logging.getLogger("trading.barlog")

# Lazy-loaded reference to calculate_core_indicators.
# Importing master_pipeline at module level is slow (~10s) because it pulls in
# torch, sklearn, etc.  We defer to first use so that ``import trading`` is fast.
_calc_fn = None


def _get_calc_fn():
    """Lazy-import calculate_core_indicators from master_pipeline."""
    global _calc_fn
    if _calc_fn is not None:
        return _calc_fn
    _scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from master_pipeline import calculate_core_indicators
    _calc_fn = calculate_core_indicators
    return _calc_fn


class LiveIndicators:
    """Rolling-window indicator calculator for live bar streams.

    Parameters
    ----------
    lookback : int
        Number of historical bars kept in the rolling window.
        Must be ≥ the longest indicator period used by
        ``calculate_core_indicators`` (typically ~80–100 bars).
    """

    def __init__(self, lookback: int = 200):
        self.lookback = lookback
        self._bars: deque = deque(maxlen=lookback)
        self._last_indicators: Dict[str, float] = {}

    # ── public API ──────────────────────────────────────────────────────

    def push(self, bar: Dict) -> Dict[str, float]:
        """Append a new bar and return the indicators for it.

        Parameters
        ----------
        bar : dict
            Must contain at minimum:
            ``datetime``, ``open``, ``high``, ``low``, ``close``, ``volume``.

        Returns
        -------
        dict
            Feature dict for the current bar.  Empty if the window is too
            short for indicator calculation.
        """
        self._bars.append(bar)

        if len(self._bars) < 20:
            return {}

        df = pd.DataFrame(list(self._bars))

        # Ensure required columns
        if "datetime" not in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["time"], utc=True)
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        if "date" not in df.columns:
            df["date"] = df["datetime"].dt.date

        # Workaround: pandas groupby('date').apply() behaves differently
        # when there is only one group (returns DataFrame instead of Series).
        # calculate_core_indicators uses groupby('date') for VWAP etc.
        # If all bars share the same date, inject a synthetic prior-day row
        # at index 0 so there are always >=2 groups, then drop it afterward.
        _injected_dummy = False
        if df["date"].nunique() == 1:
            dummy = df.iloc[[0]].copy()
            dummy["date"] = dummy["date"].iloc[0] - pd.Timedelta(days=1)
            dummy["datetime"] = dummy["datetime"].iloc[0] - pd.Timedelta(days=1)
            df = pd.concat([dummy, df], ignore_index=True)
            _injected_dummy = True

        try:
            result = _get_calc_fn()(df, verbose=False)
        except Exception as e:
            _log.warning("calculate_core_indicators failed: %s\n%s",
                         e, traceback.format_exc())
            _bar_log.warning("INDICATOR ERROR: %s\n%s", e, traceback.format_exc())
            return {}

        if result is None or result.empty:
            return {}

        # Drop the synthetic dummy row if we injected one
        if _injected_dummy and len(result) > 1:
            result = result.iloc[1:].reset_index(drop=True)

        if result.empty:
            return {}

        # Take the last row — that's the "current" bar
        last = result.iloc[-1]
        indicators: Dict[str, float] = {}
        for col in result.columns:
            val = last[col]
            if isinstance(val, (np.integer,)):
                indicators[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                v = float(val)
                indicators[col] = v
            else:
                indicators[col] = val

        # Live bars are complete (closed)
        indicators["vol_pct_complete"] = 1.0

        # Fix NaN/inf from zero-range bars (high == low -> division by zero)
        for key in list(indicators.keys()):
            v = indicators[key]
            if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')):
                if key == "close_position":
                    indicators[key] = 0.5  # midpoint default
                elif key == "bar_range_atr":
                    indicators[key] = 0.0
                else:
                    indicators[key] = 0.0  # safe default for any NaN/inf

        self._last_indicators = indicators
        return indicators

    @property
    def last(self) -> Dict[str, float]:
        """Most recently computed indicators."""
        return self._last_indicators

    def reset(self) -> None:
        """Clear the bar window."""
        self._bars.clear()
        self._last_indicators = {}

    def seed(self, bars: List[Dict]) -> None:
        """Pre-load historical bars (warm-up) without returning indicators.

        Useful at startup to fill the window with prior data so that
        the first real-time bar already has full indicators.
        """
        for b in bars:
            self._bars.append(b)
