"""
Trading Engine — the main event loop.

Connects to IBKR, subscribes to real-time bars, computes indicators,
evaluates the strategy, submits bracket orders, and flattens at EOD.

Works identically for paper and live — only the port differs.
P&L is managed entirely by the IBKR account; no local accounting.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz

from ib_insync import IB, Contract, Stock, BarData, RealTimeBar, util

from trading.config import (
    BAR_SIZE_ALIASES,
    CAPITAL,
    DEFAULT_BAR_SIZE,
    DEFAULT_CLIENT_ID,
    DEFAULT_HOST,
    DEFAULT_STOP_ATR,
    DEFAULT_SYMBOL,
    IB_BAR_SIZES,
    LIVE_PORT,
    MAX_CONCURRENT,
    PAPER_PORT,
    PROB_SCALE_MIN,
    PROB_THRESHOLD,
    RISK_PER_TRADE,
)
from trading.indicators import LiveIndicators
from trading.orders import OrderManager, OrderTicket, Side
from trading.strategy import Signal, VWAPReversionStrategy

log = logging.getLogger("trading.engine")

# Dedicated bar-by-bar debug log -- writes directly to file, bypasses
# stdout/stderr so it works even when output is redirected on Windows.
_bar_log = logging.getLogger("trading.barlog")
_bar_log.propagate = False  # don't send to root logger / console
_bar_handler = logging.FileHandler("_trading_debug.log", mode="w", encoding="utf-8")
_bar_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
_bar_handler.setLevel(logging.DEBUG)
_bar_log.addHandler(_bar_handler)
_bar_log.setLevel(logging.DEBUG)

# US/Eastern for session times
ET = pytz.timezone("US/Eastern")

# Regular trading hours (ET)
MARKET_OPEN_ET  = (9, 30)
MARKET_CLOSE_ET = (16, 0)
FLATTEN_MINUTES_BEFORE_CLOSE = 5   # flatten 5 min before close
NO_NEW_ENTRIES_BEFORE_CLOSE = 15   # stop opening new trades 15 min before close


class TradingEngine:
    """Real-time trading engine that bridges IBKR ↔ strategy.

    Parameters
    ----------
    mode : str
        ``"paper"`` or ``"live"``.
    symbol : str
        Ticker to trade (e.g. ``"TSLA"``).
    bar_size : str
        Bar resolution (e.g. ``"5 mins"``, ``"1 min"``, ``"tick"``).
    model : Any
        Sklearn-compatible model with ``predict_proba``.
    features : list[str]
        Feature column names the model expects.
    stop_atr : float
    threshold : float
    risk_pct : float
    capital : float
    max_concurrent : int
    host : str
    client_id : int
    lookback : int
        Number of bars for indicator warm-up window.
    """

    def __init__(
        self,
        mode: str = "paper",
        symbol: str = DEFAULT_SYMBOL,
        bar_size: str = DEFAULT_BAR_SIZE,
        model: Any = None,
        features: Optional[List[str]] = None,
        stop_atr: float = DEFAULT_STOP_ATR,
        threshold: float = PROB_THRESHOLD,
        risk_pct: float = RISK_PER_TRADE,
        prob_scale_min: float = PROB_SCALE_MIN,
        capital: float = CAPITAL,
        max_concurrent: int = MAX_CONCURRENT,
        host: str = DEFAULT_HOST,
        client_id: int = DEFAULT_CLIENT_ID,
        lookback: int = 200,
    ):
        self.mode = mode
        self.symbol = symbol
        self.bar_size = bar_size
        self.stop_atr = stop_atr
        self.threshold = threshold
        self.risk_pct = risk_pct
        self.prob_scale_min = prob_scale_min
        self.capital = capital
        self.max_concurrent = max_concurrent
        self.host = host
        self.port = PAPER_PORT if mode == "paper" else LIVE_PORT
        self.client_id = client_id
        self.lookback = lookback

        self.model = model
        self.features = features or []

        # Components (initialised on start)
        self.ib: Optional[IB] = None
        self.contract: Optional[Contract] = None
        self.order_mgr: Optional[OrderManager] = None
        self.strategy: Optional[VWAPReversionStrategy] = None
        self.indicators: Optional[LiveIndicators] = None

        # State
        self._running = False
        self._bars_received = 0
        self._signals_evaluated = 0
        self._trades_submitted = 0
        self._eod_flattened = False

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Connect, warm up, subscribe, and enter the event loop."""
        self._print_banner()
        self._connect()
        self._init_components()
        self._warmup()
        self._subscribe()
        self._running = True

        log.info("Engine running -- waiting for bars...")
        print(f"\n  [OK] Engine running.  Ctrl+C to stop.\n", flush=True)

        try:
            while self._running:
                self.ib.sleep(1)  # ib_insync event loop tick
                self._check_eod_flatten()
        except KeyboardInterrupt:
            print("\n  Ctrl+C received -- shutting down...", flush=True)
        finally:
            self.stop()

    def stop(self) -> None:
        """Flatten all positions and disconnect."""
        self._running = False
        if self.order_mgr:
            self._flatten_all("shutdown")
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            log.info("Disconnected from IBKR")
        print("  [OK] Engine stopped.\n", flush=True)

    # ── connection ──────────────────────────────────────────────────────

    def _connect(self) -> None:
        self.ib = IB()
        print(f"  Connecting to IBKR ({self.mode.upper()}) "
              f"at {self.host}:{self.port} ...", flush=True)
        self.ib.connect(
            host=self.host,
            port=self.port,
            clientId=self.client_id,
            readonly=False,
        )
        print(f"  [OK] Connected  (account: {self.ib.managedAccounts()})", flush=True)

        # Qualify contract
        self.contract = Stock(self.symbol, "SMART", "USD")
        self.ib.qualifyContracts(self.contract)
        print(f"  [OK] Contract qualified: {self.contract}", flush=True)

    # ── component initialisation ────────────────────────────────────────

    def _init_components(self) -> None:
        self.indicators = LiveIndicators(lookback=self.lookback)

        self.strategy = VWAPReversionStrategy(
            model=self.model,
            features=self.features,
            stop_atr=self.stop_atr,
            threshold=self.threshold,
            risk_pct=self.risk_pct,
            prob_scale_min=self.prob_scale_min,
            capital=self.capital,
        )

        self.order_mgr = OrderManager(
            ib=self.ib,
            on_fill=self._handle_fill,
            on_exit=self._handle_exit,
        )

    # ── warm-up: seed indicator window with recent history ──────────────

    def _warmup(self) -> None:
        """Fetch recent historical bars to fill the indicator window."""
        print(f"  Warming up indicators ({self.lookback} bars of "
              f"{self.bar_size})...", flush=True)

        ib_bar_size = self._resolve_bar_size()
        if ib_bar_size is None:
            # tick / 1-sec modes — warm up with 5-min bars then switch
            ib_bar_size = "5 mins"

        # Duration string: enough to cover lookback bars
        # Rough: 78 five-min bars per day, so lookback/78 days + buffer
        bars_per_day = self._bars_per_day(ib_bar_size)
        days_needed = max(3, int(self.lookback / max(bars_per_day, 1)) + 2)
        duration = f"{days_needed} D"

        hist_bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=ib_bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
        )

        if not hist_bars:
            print(f"  [WARN] No historical bars returned for warm-up", flush=True)
            return
        # Seed indicators
        bar_dicts = []
        for b in hist_bars:
            ts = pd.Timestamp(b.date)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            bar_dicts.append({
                "datetime": ts,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(b.volume),
            })

        self.indicators.seed(bar_dicts[:-1])  # all but last
        # Push the last one to get initial indicators
        if bar_dicts:
            ind = self.indicators.push(bar_dicts[-1])
            if ind:
                log.info("Warm-up complete: %d bars, %d indicators",
                         len(bar_dicts), len(ind))

        print(f"  [OK] Seeded {len(bar_dicts)} historical bars", flush=True)

    # ── real-time subscription ──────────────────────────────────────────

    def _subscribe(self) -> None:
        """Subscribe to real-time bar updates."""
        ib_bar_size = self._resolve_bar_size()

        if self.bar_size == "tick":
            # Tick-by-tick: AllLast
            print(f"  Subscribing to tick-by-tick data for {self.symbol}...", flush=True)
            self.ib.reqTickByTickData(self.contract, "AllLast")
            self.ib.pendingTickersEvent += self._on_tick
        elif self.bar_size == "1 sec":
            # 1-second: tick-by-tick AllLast, aggregate ourselves
            print(f"  Subscribing to tick-by-tick (1-sec aggregation) "
                  f"for {self.symbol}...", flush=True)
            self.ib.reqTickByTickData(self.contract, "AllLast")
            self.ib.pendingTickersEvent += self._on_tick
        elif ib_bar_size == "5 secs":
            # 5-second real-time bars (the only size reqRealTimeBars supports)
            print(f"  Subscribing to 5-second real-time bars for {self.symbol}...",
                  flush=True)
            self.ib.reqRealTimeBars(
                self.contract, 5, "TRADES", useRTH=True,
            )
            self.ib.barUpdateEvent += self._on_realtime_bar
        else:
            # Everything else: historical bars kept up-to-date
            print(f"  Subscribing to {self.bar_size} bars (keepUpToDate) "
                  f"for {self.symbol}...", flush=True)
            bars = self.ib.reqHistoricalData(
                self.contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting=ib_bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,
                keepUpToDate=True,
            )
            bars.updateEvent += self._on_historical_bar_update

    # ── bar handlers ────────────────────────────────────────────────────

    def _on_realtime_bar(self, bars, hasNewBar: bool) -> None:
        """Handler for reqRealTimeBars (5-second bars)."""
        if not hasNewBar or not bars:
            return
        b = bars[-1]
        raw = b.date if hasattr(b, 'date') else b.time
        ts = pd.Timestamp(raw)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        bar_dict = {
            "datetime": ts,
            "open": float(b.open_),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": int(b.volume),        }
        self._process_bar(bar_dict)

    def _on_historical_bar_update(self, bars, hasNewBar: bool) -> None:
        """Handler for reqHistoricalData with keepUpToDate=True."""
        b = bars[-1] if bars else None
        if not hasNewBar or not bars:
            return
        if b:
            _bar_log.info("NEW BAR | date=%s | O=%.2f H=%.2f L=%.2f C=%.2f V=%d | total_bars=%d",
                          b.date, b.open, b.high, b.low, b.close,
                          b.volume, len(bars))
        ts = pd.Timestamp(b.date)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        bar_dict = {
            "datetime": ts,
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": int(b.volume),
        }
        self._process_bar(bar_dict)

    def _on_tick(self, tickers) -> None:
        """Handler for tick-by-tick data.

        For 'tick' mode we evaluate on every tick.
        For '1 sec' mode we would aggregate — for now, treat each tick
        as a micro-bar (price = last, volume = lastSize).
        """
        for ticker in tickers:
            if ticker.contract != self.contract:
                continue
            if ticker.last != ticker.last:  # NaN check
                continue
            now = pd.Timestamp.now(tz="UTC")
            bar_dict = {
                "datetime": now,
                "open": float(ticker.last),
                "high": float(ticker.last),
                "low": float(ticker.last),
                "close": float(ticker.last),
                "volume": int(ticker.lastSize) if ticker.lastSize == ticker.lastSize else 0,
            }
            self._process_bar(bar_dict)

    # ── core bar processing ─────────────────────────────────────────────

    def _process_bar(self, bar: Dict) -> None:
        """Central pipeline: bar -> indicators -> strategy -> order."""
        self._bars_received += 1
        dt_str = str(bar.get("datetime", ""))[:19]
        close_px = bar.get("close", 0)
        vol = bar.get("volume", 0)

        hdr = f"Bar #{self._bars_received}  {dt_str}  C={close_px:.2f}  V={vol}"

        # Compute indicators
        indicators = self.indicators.push(bar)
        if not indicators:
            _bar_log.info("%s | NO TRADE | indicators=None (not enough history)", hdr)
            return

        self._signals_evaluated += 1

        # Block new entries near EOD — no point opening just to flatten
        now_et = datetime.now(ET)
        close_et = now_et.replace(
            hour=MARKET_CLOSE_ET[0], minute=MARKET_CLOSE_ET[1], second=0, microsecond=0
        )
        cutoff_et = close_et - timedelta(minutes=NO_NEW_ENTRIES_BEFORE_CLOSE)
        if now_et >= cutoff_et:
            _bar_log.info("%s | NO TRADE | EOD cutoff (%d min before close)",
                          hdr, NO_NEW_ENTRIES_BEFORE_CLOSE)
            return

        # Check capacity
        open_count = len(self.order_mgr.open_tickets)
        if open_count >= self.max_concurrent:
            _bar_log.info("%s | NO TRADE | max concurrent reached (%d/%d open)",
                          hdr, open_count, self.max_concurrent)
            return

        # Evaluate strategy -- returns (Signal, None) or (None, reason_string)
        signal, reject_reason = self.strategy.evaluate(indicators)
        if signal is None:
            _bar_log.info("%s | NO TRADE | %s", hdr, reject_reason)
            return

        _bar_log.info("%s | >>> TRADE <<< | %s %d shares @ $%.2f  "
                      "prob=%.4f  stop=$%.2f  target=$%.2f",
                      hdr, signal.direction.upper(), signal.shares,
                      signal.entry_price, signal.prob,
                      signal.stop_price, signal.target_price)        # Cancel any stale pending-entry tickets before submitting a new one.
        # This prevents the race condition (bar 5) where the new bar's bracket
        # orders get voided because the previous bar's cancel sweeps them too.
        self._cancel_stale_pending_entries()

        # Submit bracket order
        self._submit_signal(signal)

    def _cancel_stale_pending_entries(self) -> None:
        """Cancel open tickets whose entry has NOT yet filled.

        When a new bar fires, any bracket from the previous bar that is still
        waiting for an entry fill is stale (the bar close price is gone).
        Cancel them now, BEFORE submitting new orders, and give IB a moment
        to process the cancellations so they don't bleed into the new bracket.
        """
        stale = [
            t for t in self.order_mgr.open_tickets
            if t.status == "pending"   # entry not yet filled
        ]
        for ticket in stale:
            _bar_log.info("CANCEL STALE PENDING ticket=%s (entry not filled, new bar)",
                          ticket.ticket_id)
            self.order_mgr.cancel_ticket(ticket.ticket_id)

        if stale:
            self.ib.sleep(0.3)   # wait for IB to ack the cancellations

    # Maximum gap between signal close and bar open before we abort entry.
    # Limit orders beyond this gap simply won't fill, skipping the trade
    # instead of taking a large adverse market-order fill.
    ENTRY_LIMIT_BUFFER = 0.10   # $0.10 beyond close in the entry direction

    def _submit_signal(self, signal: Signal) -> None:
        """Convert a Signal into an IBKR bracket order.

        Uses a LIMIT entry at signal close ± ENTRY_LIMIT_BUFFER so that
        large gap-opens (which caused $0.20-0.46/share adverse slippage in
        paper trading) result in a missed trade rather than a terrible fill.

        If the bar opens within $0.10 of the signal close, the limit fills
        immediately at the open (almost always better than MKT).
        If the bar gaps more than $0.10 against us the order stays open and
        may fill on a pullback, or is cancelled when the next bar fires.
        """
        side = Side.BUY if signal.direction == "long" else Side.SELL

        # Limit price: signal close ± buffer in the entry direction.
        # LONG:  willing to buy up to (close + buffer)
        # SHORT: willing to sell down to (close - buffer)
        if signal.direction == "long":
            entry_limit = round(signal.entry_price + self.ENTRY_LIMIT_BUFFER, 2)
        else:
            entry_limit = round(signal.entry_price - self.ENTRY_LIMIT_BUFFER, 2)

        ticket = self.order_mgr.submit_bracket(
            contract=self.contract,
            side=side,
            shares=signal.shares,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            entry_type="LMT",
            entry_limit_price=entry_limit,
        )

        self._trades_submitted += 1
        print(f"\n  >> SIGNAL #{self._trades_submitted}  "
              f"{signal.direction.upper()} {signal.shares} {self.symbol} "
              f"@ LMT ${entry_limit:.2f}  (signal close=${signal.entry_price:.2f})  "
              f"stop=${signal.stop_price:.2f}  "
              f"target=${signal.target_price:.2f}  "
              f"prob={signal.prob:.3f}  "
              f"ticket={ticket.ticket_id}",
              flush=True)

    # ── fill / exit callbacks ───────────────────────────────────────────

    def _handle_fill(self, ticket: OrderTicket, trade) -> None:
        fill_px = ticket.fill_price or 0
        print(f"  [FILLED]  ticket={ticket.ticket_id}  "
              f"{ticket.side.name} {ticket.shares} {ticket.symbol} "
              f"@ ${fill_px:.2f}", flush=True)

    def _handle_exit(self, ticket: OrderTicket, reason: str, fill_price: float) -> None:
        entry_px = ticket.fill_price or 0
        if ticket.side == Side.BUY:
            pnl = (fill_price - entry_px) * ticket.shares
        else:
            pnl = (entry_px - fill_price) * ticket.shares

        icon = "[WIN]" if pnl >= 0 else "[LOSS]"
        print(f"  {icon} EXIT  ticket={ticket.ticket_id}  "
              f"reason={reason}  exit=${fill_price:.2f}  "
              f"gross=${pnl:+,.2f}", flush=True)

    # ── EOD flatten ─────────────────────────────────────────────────────

    def _check_eod_flatten(self) -> None:
        """At 3:55 PM ET: cancel all orders, close all positions via IB directly."""
        if self._eod_flattened:
            return
        now_et = datetime.now(ET)
        close_et = now_et.replace(
            hour=MARKET_CLOSE_ET[0], minute=MARKET_CLOSE_ET[1], second=0, microsecond=0
        )
        flatten_at = close_et - timedelta(minutes=FLATTEN_MINUTES_BEFORE_CLOSE)

        if now_et < flatten_at:
            return

        _bar_log.info("=== EOD FLATTEN TRIGGERED at %s ET ===", now_et.strftime("%H:%M:%S"))
        print(f"\n  === EOD FLATTEN {now_et.strftime('%H:%M:%S')} ET ===", flush=True)

        # 1. Cancel tracked tickets (best-effort)
        if self.order_mgr:
            self._flatten_all("eod")

        # 2. Hard safety: cancel ALL IB orders + close actual position
        self._flatten_ib_position(reason="eod")

        self._eod_flattened = True
        _bar_log.info("=== EOD FLATTEN COMPLETE ===")

    def _flatten_all(self, reason: str) -> None:
        """Cancel all brackets and market-close open positions."""
        for ticket in list(self.order_mgr.open_tickets):
            _bar_log.info("FLATTEN (%s) ticket=%s  side=%s  shares=%d  status=%s",
                          reason, ticket.ticket_id, ticket.side.name,
                          ticket.shares, ticket.status)
            print(f"  [FLATTEN] ({reason})  ticket={ticket.ticket_id}", flush=True)

            # Cancel stop/target
            self.order_mgr.cancel_exit_orders(ticket.ticket_id)
            self.ib.sleep(0.2)

            # If entry was filled, submit closing market order
            if ticket.status == "filled":
                close_side = Side.SELL if ticket.side == Side.BUY else Side.BUY
                self.order_mgr.submit_market(
                    contract=self.contract,
                    side=close_side,
                    shares=ticket.shares,
                )
                ticket.status = "cancelled"
            else:
                self.order_mgr.cancel_ticket(ticket.ticket_id)

    def _flatten_ib_position(self, reason: str = "eod_safety") -> None:
        """Query IB for actual open orders and positions, close everything.

        This is the hard safety net — ignores ticket tracking entirely
        and talks directly to IB.  Cancels ALL open orders for the symbol,
        then market-closes any remaining position.
        """
        # 1. Cancel every open order on this symbol
        open_orders = self.ib.openOrders()
        open_trades = self.ib.openTrades()
        cancelled = 0
        for trade in open_trades:
            if (trade.contract.symbol == self.symbol
                    and trade.orderStatus.status in
                    ("PreSubmitted", "Submitted", "PendingSubmit")):
                _bar_log.info("CANCEL IB ORDER (%s) orderId=%d action=%s qty=%.0f status=%s",
                              reason, trade.order.orderId, trade.order.action,
                              trade.order.totalQuantity, trade.orderStatus.status)
                self.ib.cancelOrder(trade.order)
                cancelled += 1
        if cancelled:
            _bar_log.info("Cancelled %d open IB orders for %s", cancelled, self.symbol)
            print(f"  [FLATTEN IB] ({reason}) cancelled {cancelled} open orders", flush=True)
            self.ib.sleep(0.5)  # let cancellations propagate

        # 2. Close any remaining position
        self.ib.reqPositions()  # refresh
        self.ib.sleep(0.3)
        positions = self.ib.positions()
        for pos in positions:
            if pos.contract.symbol == self.symbol and pos.position != 0:
                qty = int(abs(pos.position))
                side = Side.SELL if pos.position > 0 else Side.BUY
                _bar_log.info("FLATTEN IB POSITION (%s) %s %d shares of %s (actual IB position)",
                              reason, side.name, qty, self.symbol)
                print(f"  [FLATTEN IB] ({reason}) {side.name} {qty} {self.symbol}", flush=True)
                self.order_mgr.submit_market(
                    contract=self.contract,
                    side=side,
                    shares=qty,
                )

    # ── helpers ─────────────────────────────────────────────────────────

    def _resolve_bar_size(self) -> Optional[str]:
        """Resolve the user-specified bar_size to an IB API bar size string."""
        # Direct match
        if self.bar_size in IB_BAR_SIZES:
            return IB_BAR_SIZES[self.bar_size]
        # Alias match
        canonical = BAR_SIZE_ALIASES.get(self.bar_size, self.bar_size)
        return IB_BAR_SIZES.get(canonical)

    @staticmethod
    def _bars_per_day(bar_size: str) -> int:
        """Rough estimate of bars per RTH day for a given bar size."""
        mapping = {
            "5 secs": 4680,
            "10 secs": 2340,
            "15 secs": 1560,
            "30 secs": 780,
            "1 min": 390,
            "2 mins": 195,
            "3 mins": 130,
            "5 mins": 78,
            "10 mins": 39,
            "15 mins": 26,
            "30 mins": 13,
            "1 hour": 7,
        }
        return mapping.get(bar_size, 78)

    def _print_banner(self) -> None:
        risk_dollars = self.capital * self.risk_pct
        print(f"\n{'='*72}", flush=True)
        print(f"  IBKR TRADING ENGINE", flush=True)
        print(f"{'='*72}", flush=True)
        print(f"  Mode:           {self.mode.upper()}", flush=True)
        print(f"  Symbol:         {self.symbol}", flush=True)
        print(f"  Bar size:       {self.bar_size}", flush=True)
        print(f"  Capital:        ${self.capital:,.0f}", flush=True)
        print(f"  Risk/trade:     {self.risk_pct:.1%}  "
              f"(${risk_dollars:,.0f})", flush=True)
        print(f"  Stop ATR:       {self.stop_atr}", flush=True)
        print(f"  Threshold:      {self.threshold}", flush=True)
        print(f"  Max concurrent: {self.max_concurrent}", flush=True)
        print(f"  Features:       {len(self.features)}", flush=True)
        print(f"  Lookback:       {self.lookback} bars", flush=True)
        print(f"  Port:           {self.port}", flush=True)
        print(f"{'='*72}", flush=True)
