"""
VWAP Mean-Reversion Strategy

Translates model predictions + indicators into trade signals.
Matches the logic in master_pipeline.py / simulate_streaming_clean.py:

  1. Bar closes → compute indicators → extract feature vector
  2. model.predict_proba(features) → prob
  3. If prob ≥ threshold → enter at bar close
     • Direction: is_long_setup (close < VWAP → long, else short)
     • Stop: entry ± stop_atr × ATR
     • Target: entry-bar VWAP (fixed target)
  4. Exits managed by IBKR bracket order (stop + limit)
  5. EOD flatten: cancel bracket, market-close any open position

Joint model (nn_pnl_joint):
  - Model has a predict_best_stop(X, atr, vwap_dist_atr) method
  - 9 forward passes sweep stop_atr ∈ STOP_ATRS → pick (entry, stop) with max E[net P&L]
  - VWAPReversionJointStrategy wraps this logic

This module is *pure logic* — no IB connection.  The engine calls it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("trading.strategy")


@dataclass
class Signal:
    """A trade signal produced by the strategy."""

    direction: str           # "long" or "short"
    entry_price: float       # bar close
    stop_price: float
    target_price: float      # entry-bar VWAP
    shares: int
    prob: float
    atr: float
    stop_atr: float
    indicators: Dict[str, float]


class VWAPReversionStrategy:
    """Stateless signal generator.

    Parameters
    ----------
    model : Any
        Sklearn-compatible model with ``predict_proba(X)`` → (n, 2).
    features : list[str]
        Ordered list of feature column names the model expects.
    stop_atr : float
        Stop distance in ATR multiples.
    threshold : float
        Minimum model probability to enter.
    risk_pct : float
        Fraction of capital risked per trade.
    prob_scale_min : float
        Minimum probability scaling fraction.
    capital : float
        Available capital for sizing.
    """

    def __init__(
        self,
        model: Any,
        features: List[str],
        stop_atr: float = 0.75,
        threshold: float = 0.50,
        risk_pct: float = 0.01,
        prob_scale_min: float = 0.30,
        capital: float = 1_000_000,
    ):
        self.model = model
        self.features = features
        self.stop_atr = stop_atr
        self.threshold = threshold
        self.risk_pct = risk_pct
        self.prob_scale_min = prob_scale_min
        self.capital = capital

    def evaluate(self, indicators: Dict[str, float]) -> Tuple[Optional[Signal], Optional[str]]:
        """Evaluate a single bar's indicators and return a Signal or rejection reason.

        Parameters
        ----------
        indicators : dict
            Must contain all ``self.features`` keys plus ``atr``, ``vwap``,
            ``close``, ``is_long_setup``.

        Returns
        -------
        (Signal, None) if a trade should be taken, or
        (None, reason_string) explaining why the trade was rejected.
        """
        close = indicators.get("close", "?") if indicators else "?"

        if not indicators or "atr" not in indicators:
            return None, f"indicators empty or missing 'atr' (keys: {sorted(indicators.keys()) if indicators else 'None'})"

        # Direction (not a filter — pipeline trades both directions)
        is_long_setup = indicators.get("is_long_setup")
        if is_long_setup is None:
            return None, "'is_long_setup' not in indicators (close/vwap comparison missing)"

        # Build feature vector
        feature_vec = [indicators.get(c) for c in self.features]
        missing = [c for c, v in zip(self.features, feature_vec)
                   if v is None or (isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)))]
        if missing:
            return None, (f"{len(missing)}/{len(self.features)} features missing/NaN: "
                          f"{missing[:15]}{'...' if len(missing) > 15 else ''}")

        # Model prediction
        prob = float(self.model.predict_proba([feature_vec])[0, 1])
        if prob < self.threshold:
            direction = "LONG" if is_long_setup else "SHORT"
            return None, (f"prob too low: {prob:.4f} < threshold {self.threshold:.4f} "
                          f"(setup={direction}, C={close})")

        entry_price = float(indicators["close"])
        atr = float(indicators["atr"])
        vwap = float(indicators.get("vwap", entry_price))

        # Stop
        stop_dist = self.stop_atr * atr
        if is_long_setup:
            stop_price = entry_price - stop_dist
        else:
            stop_price = entry_price + stop_dist

        # Target = entry-bar VWAP (fixed)
        target_price = vwap

        # Shares
        from trading.risk import compute_shares  # avoid circular at module level
        shares = compute_shares(
            prob=prob,
            threshold=self.threshold,
            atr=atr,
            stop_atr=self.stop_atr,
            risk_pct=self.risk_pct,            min_frac=self.prob_scale_min,
            capital=self.capital,
            entry_price=entry_price,
        )

        return Signal(
            direction="long" if is_long_setup else "short",
            entry_price=entry_price,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            shares=shares,
            prob=prob,
            atr=atr,
            stop_atr=self.stop_atr,
            indicators=indicators,
        ), None


class VWAPReversionJointStrategy:
    """Signal generator for the ``nn_pnl_joint`` model.

    At each bar the model runs 9 forward passes (one per stop width) and
    selects the ``(entry, stop_atr)`` pair that maximises expected net P&L:

        E[net_pnl | stop_atr] = prob(stop) * (reward - costs)
                               + (1-prob(stop)) * (-risk - costs)

    A trade is entered only when the best-stop expected P&L exceeds
    ``min_exp_pnl`` (default $0, i.e. positive expectancy required).

    Parameters
    ----------
    model : PnLModelWrapperJoint
        Joint model with ``predict_best_stop()`` method.
    features : list[str]
        Market feature names (without stop_atr).
    threshold : float
        Minimum probability at the best stop to take the trade.
    min_exp_pnl : float
        Minimum expected net P&L ($) at best stop to enter.
    risk_pct : float
        Fraction of capital to risk per trade.
    prob_scale_min : float
        Minimum probability scaling fraction for position sizing.
    capital : float
        Available capital.
    """

    def __init__(
        self,
        model,
        features: List[str],
        threshold: float = 0.50,
        min_exp_pnl: float = 0.0,
        risk_pct: float = 0.01,
        prob_scale_min: float = 0.30,
        capital: float = 1_000_000,
    ):
        self.model = model
        self.features = features
        self.threshold = threshold
        self.min_exp_pnl = min_exp_pnl
        self.risk_pct = risk_pct
        self.prob_scale_min = prob_scale_min
        self.capital = capital

    def evaluate(self, indicators: Dict[str, float]) -> Tuple[Optional[Signal], Optional[str]]:
        """Evaluate a bar and return (Signal, None) or (None, reason)."""
        if not indicators or "atr" not in indicators:
            return None, "indicators empty or missing 'atr'"

        is_long_setup = indicators.get("is_long_setup")
        if is_long_setup is None:
            return None, "'is_long_setup' not in indicators"

        feature_vec = [indicators.get(c) for c in self.features]
        missing = [c for c, v in zip(self.features, feature_vec)
                   if v is None or (isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)))]
        if missing:
            return None, (f"{len(missing)}/{len(self.features)} features missing/NaN: "
                          f"{missing[:15]}{'...' if len(missing) > 15 else ''}")

        atr = float(indicators["atr"])
        vwap = float(indicators.get("vwap", indicators.get("close", 0.0)))
        entry_price = float(indicators["close"])
        vwap_dist_atr = abs(entry_price - vwap) / max(atr, 1e-8)

        # Run joint model: 9 stops → pick best
        result = self.model.predict_best_stop(
            [feature_vec],
            atr_series=np.array([atr]),
            vwap_dist_atr_series=np.array([vwap_dist_atr]),
        )
        prob = float(result['prob'][0])
        best_stop_atr = float(result['stop_atr'][0])
        exp_net_pnl = float(result['exp_net_pnl'][0])

        if prob < self.threshold:
            direction = "LONG" if is_long_setup else "SHORT"
            return None, (f"prob too low: {prob:.4f} < {self.threshold:.4f} "
                          f"(best_stop={best_stop_atr:.2f}, E[P&L]=${exp_net_pnl:.0f}, "
                          f"setup={direction})")

        if exp_net_pnl < self.min_exp_pnl:
            return None, (f"exp_net_pnl ${exp_net_pnl:.0f} < min ${self.min_exp_pnl:.0f} "
                          f"(prob={prob:.4f}, best_stop={best_stop_atr:.2f})")

        stop_dist = best_stop_atr * atr
        if is_long_setup:
            stop_price = entry_price - stop_dist
        else:
            stop_price = entry_price + stop_dist
        target_price = vwap

        from trading.risk import compute_shares
        shares = compute_shares(
            prob=prob,
            threshold=self.threshold,
            atr=atr,
            stop_atr=best_stop_atr,
            risk_pct=self.risk_pct,
            min_frac=self.prob_scale_min,
            capital=self.capital,
            entry_price=entry_price,
        )

        return Signal(
            direction="long" if is_long_setup else "short",
            entry_price=entry_price,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            shares=shares,
            prob=prob,
            atr=atr,
            stop_atr=best_stop_atr,
            indicators=indicators,
        ), None
