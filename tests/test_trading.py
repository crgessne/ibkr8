"""
Tests for the trading platform — end-to-end through the full pipeline:
  indicators → strategy → risk sizing → order submission

Verifies that a bar with favorable conditions actually produces a trade,
and that unfavorable bars are correctly rejected with reasons.

All IB connectivity is mocked — no live connection needed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.config import CAPITAL, RISK_PER_TRADE
from trading.orders import OrderManager, OrderTicket, Side
from trading.risk import compute_shares, ibkr_margin_cost, round_trip_cost
from trading.strategy import Signal, VWAPReversionStrategy


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# The 62 features the real model expects (from rf_vwap_stop0.75 metadata).
# We only need the *names* — values will be synthetic.
FEATURES_62 = [
    "atr", "vwap", "vwap_width_atr", "price_to_vwap_atr", "is_long_setup",
    "vwap_stretch_zscore", "vwap_slope", "vwap_slope_5", "vwap_helping",
    "rel_vol", "vol_ratio", "vol_at_extension", "rsi", "rsi_slope",
    "rsi_extreme", "bar_range_atr", "close_position", "crossed_vwap",
    "bars_from_vwap", "hour", "minute", "minutes_into_session",
    "session_phase", "cum_vol_today", "total_vol_today", "vol_pct_complete",
    "vwap_crosses_today", "day_high", "day_low", "day_range_atr",
    "pct_of_day_range", "vwap_in_day_range", "momentum_3bar_atr",
    "momentum_6bar_atr", "bar_reverting", "consecutive_same_side",
    "open_vs_vwap_atr", "prior_bar_toward_vwap", "ema20", "ema20_slope_atr",
    "extension_speed", "bb_z_score", "bb_width_atr", "vwap_sigma",
    "upper_wick_pct", "lower_wick_pct", "body_pct", "rejection_wick_pct",
    "vol_zscore", "vol_climax", "vol_declining_3bar", "atr_ratio_5",
    "atr_ratio_20", "atr_regime", "ema60", "ema60_slope_atr",
    "price_vs_ema60_atr", "trend_aligned", "counter_trend_discount",
    "reversion_potential_atr", "touch_count_20bar", "multi_touch",
]


def _make_model(prob: float) -> MagicMock:
    """Create a mock model whose predict_proba always returns `prob`."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1.0 - prob, prob]])
    return model


def _make_indicators(
    *,
    close: float = 400.0,
    vwap: float = 402.0,
    atr: float = 1.50,
    is_long: bool = True,
) -> Dict[str, float]:
    """Build a full indicator dict with 62 features + required keys.

    close < vwap → long setup (default).
    """
    ind: Dict[str, float] = {}
    for f in FEATURES_62:
        ind[f] = 0.5  # safe default for all features

    # Override key fields
    ind["close"] = close
    ind["vwap"] = vwap
    ind["atr"] = atr
    ind["is_long_setup"] = 1.0 if is_long else 0.0
    ind["high"] = close + atr * 0.5
    ind["low"] = close - atr * 0.5
    ind["open"] = close - 0.10
    ind["volume"] = 10000
    ind["rsi"] = 35.0 if is_long else 65.0
    ind["vwap_width_atr"] = abs(close - vwap) / atr
    ind["price_to_vwap_atr"] = (close - vwap) / atr
    ind["vol_pct_complete"] = 1.0
    return ind


def _mock_ib() -> MagicMock:
    """Create a mock IB connection with placeOrder / cancelOrder stubs."""
    ib = MagicMock()

    def _place_order(contract, order):
        trade = MagicMock()
        trade.order = order
        trade.orderStatus.status = "PreSubmitted"
        trade.orderStatus.avgFillPrice = 0.0
        trade.isActive.return_value = True
        # Give orders an incrementing ID
        if not hasattr(order, "orderId") or order.orderId == 0:
            _place_order._next_id = getattr(_place_order, "_next_id", 100) + 1
            order.orderId = _place_order._next_id
        return trade

    ib.placeOrder.side_effect = _place_order
    ib.cancelOrder.return_value = None
    return ib


# ===========================================================================
# Strategy tests
# ===========================================================================


class TestStrategySignal:
    """Test that VWAPReversionStrategy.evaluate() produces / rejects signals."""

    def test_high_prob_long_produces_signal(self):
        """High probability + long setup → trade signal."""
        model = _make_model(prob=0.85)
        strat = VWAPReversionStrategy(
            model=model,
            features=FEATURES_62,
            stop_atr=0.75,
            threshold=0.50,
            capital=CAPITAL,
        )
        indicators = _make_indicators(close=400.0, vwap=402.0, atr=1.50, is_long=True)

        signal, reason = strat.evaluate(indicators)

        assert signal is not None, f"Expected a signal, got rejection: {reason}"
        assert reason is None
        assert signal.direction == "long"
        assert signal.entry_price == 400.0
        assert signal.stop_price == round(400.0 - 0.75 * 1.50, 2)  # 398.88
        assert signal.target_price == 402.0  # VWAP
        assert signal.shares >= 1
        assert signal.prob == pytest.approx(0.85, abs=0.01)

    def test_high_prob_short_produces_signal(self):
        """High probability + short setup → trade signal."""
        model = _make_model(prob=0.80)
        strat = VWAPReversionStrategy(
            model=model,
            features=FEATURES_62,
            stop_atr=0.75,
            threshold=0.50,
            capital=CAPITAL,
        )
        indicators = _make_indicators(close=404.0, vwap=402.0, atr=1.50, is_long=False)

        signal, reason = strat.evaluate(indicators)

        assert signal is not None, f"Expected a signal, got rejection: {reason}"
        assert signal.direction == "short"
        assert signal.entry_price == 404.0
        assert signal.stop_price == round(404.0 + 0.75 * 1.50, 2)  # 405.12
        assert signal.target_price == 402.0

    def test_low_prob_rejected(self):
        """Probability below threshold → rejection with reason."""
        model = _make_model(prob=0.10)
        strat = VWAPReversionStrategy(
            model=model,
            features=FEATURES_62,
            stop_atr=0.75,
            threshold=0.50,
            capital=CAPITAL,
        )
        indicators = _make_indicators(close=400.0, vwap=402.0, atr=1.50)

        signal, reason = strat.evaluate(indicators)

        assert signal is None
        assert "prob too low" in reason
        assert "0.1000" in reason
        assert "threshold 0.5000" in reason

    def test_missing_atr_rejected(self):
        """Missing ATR key → rejection."""
        model = _make_model(prob=0.90)
        strat = VWAPReversionStrategy(
            model=model, features=FEATURES_62, threshold=0.50,
        )
        indicators = {"close": 400.0, "vwap": 402.0}  # no 'atr'

        signal, reason = strat.evaluate(indicators)

        assert signal is None
        assert "atr" in reason.lower()

    def test_missing_features_rejected(self):
        """Features that are NaN → rejection listing the bad features."""
        model = _make_model(prob=0.90)
        strat = VWAPReversionStrategy(
            model=model, features=FEATURES_62, threshold=0.50,
        )
        indicators = _make_indicators()
        # Poison a few features
        indicators["rsi"] = float("nan")
        indicators["bb_z_score"] = float("inf")

        signal, reason = strat.evaluate(indicators)

        assert signal is None
        assert "features missing/NaN" in reason
        assert "rsi" in reason
        assert "bb_z_score" in reason

    def test_empty_indicators_rejected(self):
        """Empty dict → rejection."""
        model = _make_model(prob=0.90)
        strat = VWAPReversionStrategy(
            model=model, features=FEATURES_62, threshold=0.50,
        )

        signal, reason = strat.evaluate({})

        assert signal is None
        assert "empty" in reason.lower() or "atr" in reason.lower()

    def test_borderline_prob_accepted(self):
        """Probability exactly at threshold → accepted."""
        model = _make_model(prob=0.50)
        strat = VWAPReversionStrategy(
            model=model, features=FEATURES_62, threshold=0.50,
        )
        indicators = _make_indicators()

        signal, reason = strat.evaluate(indicators)

        assert signal is not None, f"Borderline prob should be accepted, got: {reason}"


# ===========================================================================
# Risk / share-sizing tests
# ===========================================================================


class TestRiskSizing:
    """Test compute_shares produces reasonable position sizes."""

    def test_basic_sizing(self):
        """Standard case: $1M capital, 1% risk, 0.75 ATR stop."""
        shares = compute_shares(
            prob=0.80, threshold=0.50, atr=1.50,
            stop_atr=0.75, risk_pct=0.01, min_frac=0.30,
            capital=1_000_000, entry_price=400.0,
        )
        assert shares >= 1
        assert shares <= 9999
        # Risk per share = 0.75 * 1.50 = $1.125
        # Full shares = 10000 / 1.125 = 8888
        # With prob scaling: scale_frac should be > 0.30
        assert shares > 2000  # substantial position

    def test_prob_at_threshold_gets_minimum_fraction(self):
        """prob == threshold → scale_frac == min_frac."""
        shares_at_threshold = compute_shares(
            prob=0.50, threshold=0.50, atr=1.50,
            stop_atr=0.75, risk_pct=0.01, min_frac=0.30,
            capital=1_000_000, entry_price=400.0,
        )
        shares_high_prob = compute_shares(
            prob=0.99, threshold=0.50, atr=1.50,
            stop_atr=0.75, risk_pct=0.01, min_frac=0.30,
            capital=1_000_000, entry_price=400.0,
        )
        assert shares_at_threshold < shares_high_prob

    def test_notional_cap(self):
        """Shares capped so notional doesn't exceed capital."""
        shares = compute_shares(
            prob=0.99, threshold=0.50, atr=0.01,  # tiny ATR → huge raw shares
            stop_atr=0.75, risk_pct=0.01, min_frac=0.30,
            capital=100_000, entry_price=400.0,
        )
        assert shares <= 100_000 / 400.0  # can't exceed capital / price

    def test_minimum_one_share(self):
        """Even with bad inputs, at least 1 share."""
        shares = compute_shares(
            prob=0.50, threshold=0.50, atr=10000.0,
            stop_atr=0.75, risk_pct=0.0001, min_frac=0.01,
            capital=1_000, entry_price=400.0,
        )
        assert shares >= 1

    def test_margin_cost(self):
        """ibkr_margin_cost returns positive cost for positive inputs."""
        cost = ibkr_margin_cost(borrowed=500_000, hold_hours=1.0)
        assert cost > 0

    def test_round_trip_cost(self):
        """round_trip_cost scales with shares."""
        cost_100 = round_trip_cost(100)
        cost_1000 = round_trip_cost(1000)
        assert cost_1000 == pytest.approx(cost_100 * 10)


# ===========================================================================
# Order submission tests (mocked IB)
# ===========================================================================


class TestOrderSubmission:
    """Test OrderManager creates correct bracket orders via mocked IB."""

    def test_bracket_order_long(self):
        """Long bracket: BUY entry + SELL stop + SELL target."""
        ib = _mock_ib()
        mgr = OrderManager(ib=ib)
        contract = MagicMock()
        contract.symbol = "TSLA"

        ticket = mgr.submit_bracket(
            contract=contract,
            side=Side.BUY,
            shares=100,
            stop_price=398.0,
            target_price=402.0,
        )

        assert ticket is not None
        assert ticket.symbol == "TSLA"
        assert ticket.side == Side.BUY
        assert ticket.shares == 100
        assert ticket.status == "pending"
        assert ticket.is_open
        # 3 orders placed: entry + stop + target
        assert ib.placeOrder.call_count == 3

    def test_bracket_order_short(self):
        """Short bracket: SELL entry + BUY stop + BUY target."""
        ib = _mock_ib()
        mgr = OrderManager(ib=ib)
        contract = MagicMock()
        contract.symbol = "TSLA"

        ticket = mgr.submit_bracket(
            contract=contract,
            side=Side.SELL,
            shares=50,
            stop_price=405.0,
            target_price=402.0,
        )

        assert ticket.side == Side.SELL
        assert ticket.shares == 50
        assert ib.placeOrder.call_count == 3

    def test_open_tickets_tracking(self):
        """OrderManager.open_tickets tracks pending orders."""
        ib = _mock_ib()
        mgr = OrderManager(ib=ib)
        contract = MagicMock()
        contract.symbol = "TSLA"

        assert len(mgr.open_tickets) == 0

        t1 = mgr.submit_bracket(contract, Side.BUY, 100, 398.0, 402.0)
        assert len(mgr.open_tickets) == 1

        t2 = mgr.submit_bracket(contract, Side.SELL, 50, 405.0, 402.0)
        assert len(mgr.open_tickets) == 2

    def test_cancel_ticket(self):
        """Cancelling a ticket removes it from open list."""
        ib = _mock_ib()
        mgr = OrderManager(ib=ib)
        contract = MagicMock()
        contract.symbol = "TSLA"

        ticket = mgr.submit_bracket(contract, Side.BUY, 100, 398.0, 402.0)
        assert len(mgr.open_tickets) == 1

        mgr.cancel_ticket(ticket.ticket_id)
        assert len(mgr.open_tickets) == 0
        assert ticket.status == "cancelled"

    def test_market_order(self):
        """submit_market places exactly 1 order."""
        ib = _mock_ib()
        mgr = OrderManager(ib=ib)
        contract = MagicMock()
        contract.symbol = "TSLA"

        ticket = mgr.submit_market(contract, Side.SELL, 100)
        assert ib.placeOrder.call_count == 1
        assert ticket.side == Side.SELL


# ===========================================================================
# End-to-end: indicators → strategy → order
# ===========================================================================


class TestEndToEnd:
    """Full pipeline: given indicators, does a trade get submitted?"""

    def test_favorable_bar_produces_trade(self):
        """
        Simulates a bar where the model returns high probability.
        Verifies the full chain: strategy.evaluate() → signal → 
        order_mgr.submit_bracket() → ticket created.
        """
        # Setup
        model = _make_model(prob=0.85)
        strat = VWAPReversionStrategy(
            model=model,
            features=FEATURES_62,
            stop_atr=0.75,
            threshold=0.50,
            capital=CAPITAL,
        )
        ib = _mock_ib()
        mgr = OrderManager(ib=ib)
        contract = MagicMock()
        contract.symbol = "TSLA"

        # Act — same flow as engine._process_bar()
        indicators = _make_indicators(close=400.0, vwap=402.0, atr=1.50, is_long=True)
        signal, reason = strat.evaluate(indicators)

        assert signal is not None, f"Signal should fire, got rejection: {reason}"

        side = Side.BUY if signal.direction == "long" else Side.SELL
        ticket = mgr.submit_bracket(
            contract=contract,
            side=side,
            shares=signal.shares,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
        )

        # Verify
        assert ticket is not None
        assert ticket.is_open
        assert ticket.side == Side.BUY
        assert ticket.shares == signal.shares
        assert signal.stop_price == round(400.0 - 0.75 * 1.50, 2)
        assert signal.target_price == 402.0
        assert ib.placeOrder.call_count == 3  # entry + stop + target

    def test_unfavorable_bar_no_trade(self):
        """Low-probability bar → no order submitted."""
        model = _make_model(prob=0.05)
        strat = VWAPReversionStrategy(
            model=model,
            features=FEATURES_62,
            stop_atr=0.75,
            threshold=0.50,
            capital=CAPITAL,
        )
        ib = _mock_ib()
        mgr = OrderManager(ib=ib)

        indicators = _make_indicators(close=400.0, vwap=402.0, atr=1.50)
        signal, reason = strat.evaluate(indicators)

        assert signal is None
        assert "prob too low" in reason
        assert ib.placeOrder.call_count == 0  # nothing submitted

    def test_signal_shares_are_risk_consistent(self):
        """Shares from signal respect risk budget."""
        model = _make_model(prob=0.85)
        strat = VWAPReversionStrategy(
            model=model,
            features=FEATURES_62,
            stop_atr=0.75,
            threshold=0.50,
            risk_pct=0.01,
            capital=1_000_000,
        )
        indicators = _make_indicators(close=400.0, vwap=402.0, atr=1.50)
        signal, _ = strat.evaluate(indicators)

        assert signal is not None
        # Max loss per share = stop_atr * atr = 0.75 * 1.50 = $1.125
        # Max risk = capital * risk_pct = $10,000
        # Max shares from risk = 10000 / 1.125 = 8888
        max_risk_shares = int(math.floor(1_000_000 * 0.01 / (0.75 * 1.50)))
        assert signal.shares <= max_risk_shares
        # Also capped by notional
        max_notional_shares = int(math.floor(1_000_000 / 400.0))
        assert signal.shares <= max_notional_shares

    def test_multiple_trades_blocked_by_max_concurrent(self):
        """
        Simulates the engine's max_concurrent=1 guard.
        After one ticket is open, the second bar should be blocked.
        """
        ib = _mock_ib()
        mgr = OrderManager(ib=ib)
        contract = MagicMock()
        contract.symbol = "TSLA"

        # First trade
        t1 = mgr.submit_bracket(contract, Side.BUY, 100, 398.0, 402.0)
        assert len(mgr.open_tickets) == 1

        # Engine would check: open_count >= max_concurrent
        max_concurrent = 1
        open_count = len(mgr.open_tickets)
        blocked = open_count >= max_concurrent

        assert blocked, "Second trade should be blocked by max_concurrent=1"

    def test_fill_callback_fires(self):
        """When entry fills, on_fill callback is invoked."""
        ib = _mock_ib()
        fill_callback = MagicMock()
        mgr = OrderManager(ib=ib, on_fill=fill_callback)
        contract = MagicMock()
        contract.symbol = "TSLA"

        ticket = mgr.submit_bracket(contract, Side.BUY, 100, 398.0, 402.0)

        # Simulate IBKR reporting the entry fill
        mock_trade = MagicMock()
        mock_trade.order.orderId = ticket.entry_order_id
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.avgFillPrice = 400.50

        mgr._on_order_status(mock_trade)

        assert ticket.status == "filled"
        assert ticket.fill_price == 400.50
        fill_callback.assert_called_once()

    def test_stop_exit_callback_fires(self):
        """When stop fills, on_exit callback is invoked with 'stop' reason."""
        ib = _mock_ib()
        exit_callback = MagicMock()
        mgr = OrderManager(ib=ib, on_exit=exit_callback)
        contract = MagicMock()
        contract.symbol = "TSLA"

        ticket = mgr.submit_bracket(contract, Side.BUY, 100, 398.0, 402.0)

        # First: fill entry
        entry_trade = MagicMock()
        entry_trade.order.orderId = ticket.entry_order_id
        entry_trade.orderStatus.status = "Filled"
        entry_trade.orderStatus.avgFillPrice = 400.50
        mgr._on_order_status(entry_trade)

        # Then: fill stop
        stop_trade = MagicMock()
        stop_trade.order.orderId = ticket.stop_order_id
        stop_trade.orderStatus.status = "Filled"
        stop_trade.orderStatus.avgFillPrice = 398.0
        mgr._on_order_status(stop_trade)

        assert ticket.status == "stopped"
        exit_callback.assert_called_once_with(ticket, "stop", 398.0)

    def test_target_exit_callback_fires(self):
        """When target fills, on_exit callback is invoked with 'vwap' reason."""
        ib = _mock_ib()
        exit_callback = MagicMock()
        mgr = OrderManager(ib=ib, on_exit=exit_callback)
        contract = MagicMock()
        contract.symbol = "TSLA"

        ticket = mgr.submit_bracket(contract, Side.BUY, 100, 398.0, 402.0)

        # Fill entry
        entry_trade = MagicMock()
        entry_trade.order.orderId = ticket.entry_order_id
        entry_trade.orderStatus.status = "Filled"
        entry_trade.orderStatus.avgFillPrice = 400.50
        mgr._on_order_status(entry_trade)

        # Fill target
        target_trade = MagicMock()
        target_trade.order.orderId = ticket.target_order_id
        target_trade.orderStatus.status = "Filled"
        target_trade.orderStatus.avgFillPrice = 402.0
        mgr._on_order_status(target_trade)

        assert ticket.status == "target"
        exit_callback.assert_called_once_with(ticket, "vwap", 402.0)
