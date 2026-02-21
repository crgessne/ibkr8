"""
Streaming Trading Simulator - Paper trading simulation

Simulates real-time paper trading by:
1. Reading bars one-by-one (simulating live data feed)
2. Building day-aware indicator windows (warmup + prior day + current day)
3. Calculating indicators when new bar completes
4. Running model/strategy on current state
5. Managing orders and positions in real-time

Window construction matches precompute_streaming_indicators.py exactly:
  window = warmup bars before prior_day_open + prior day + current day up to now

This ensures indicator parity between training (precomputed) and inference (simulator).
"""

import sys
from typing import Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np

from .order_manager import OrderManager, Order, OrderType, OrderSide, OrderStatus
from .execution_model import ExecutionModel, ExecutionQuality, SlippageModelFactory
from .portfolio import Portfolio
from .broker_fees import BrokerFees, IBKRFees


class StreamingSimulator:
    """
    Streaming simulator that processes bars one at a time,
    calculating indicators on-the-fly like paper trading.

    Supports two windowing modes:
      - **day-aware** (default, ``use_day_window=True``):  window per bar =
        warmup bars before prior-day open + full prior day + current day up to
        current bar.  Matches ``precompute_streaming_indicators.py``.
      - **legacy rolling deque** (``use_day_window=False``): fixed-size
        ``deque(maxlen=lookback_bars)`` sliding window.

    Example usage:
        sim = StreamingSimulator(
            initial_capital=100000,
            bar_interval_minutes=5,
            warmup_bars=60,
        )
        
        def strategy(current_bar, indicators, model, portfolio, order_manager):
            # Your strategy logic using current indicators
            if indicators['rsi'] < 30 and current_bar['close'] < indicators['vwap']:
                # Submit buy order
                ...
        
        results = sim.run(data, strategy, model, target_start_iloc=0)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        bar_interval_minutes: int = 5,
        lookback_bars: int = 200,
        warmup_bars: int = 60,
        use_day_window: bool = True,
        execution_quality: ExecutionQuality = ExecutionQuality.AVERAGE,
        broker_fees: Optional[BrokerFees] = None,
        max_position_size: Optional[int] = None,
        slippage_model: Optional[ExecutionModel] = None,
        verbose: bool = True,
        log_every_n_bars: int = 0,
        log_indicators: Optional[List[str]] = None,
    ):
        """
        Initialize streaming simulator
        
        Args:
            initial_capital: Starting capital
            bar_interval_minutes: Time interval for bars (e.g., 5 for 5-min bars)
            lookback_bars: Number of historical bars for indicator calc (legacy mode only)
            warmup_bars: Number of bars before prior-day open for warmup (day-window mode)
            use_day_window: If True, use day-aware windowing matching precompute.
                            If False, use legacy deque(maxlen=lookback_bars).
            execution_quality: Execution quality level
            broker_fees: Broker fee model (default: IBKR Pro)
            max_position_size: Maximum shares per position
            slippage_model: Custom slippage model
            verbose: Print detailed logging
            log_every_n_bars: If > 0, log a per-bar line every N bars with OHLC + indicators + P&L
            log_indicators: Optional list of indicator keys to print (defaults to a small set)
        """
        self.initial_capital = initial_capital
        self.bar_interval = timedelta(minutes=bar_interval_minutes)
        self.lookback_bars = lookback_bars
        self.warmup_bars = warmup_bars
        self.use_day_window = use_day_window
        self.verbose = verbose
        self.log_every_n_bars = int(log_every_n_bars or 0)
        self.log_indicators = log_indicators
        
        # Initialize components
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            max_position_size=max_position_size,
        )
        
        self.broker_fees = broker_fees if broker_fees is not None else IBKRFees()
        
        self.order_manager = OrderManager(
            commission_per_share=0.005 if isinstance(self.broker_fees, IBKRFees) else 0.0,
            min_commission=1.0 if isinstance(self.broker_fees, IBKRFees) else 0.0,
        )
        
        # Execution model
        if slippage_model is not None:
            self.slippage_model = slippage_model
        else:
            self.slippage_model = SlippageModelFactory.create_model(execution_quality)
        
        # Rolling window of historical bars for indicator calculation (legacy mode)
        self.bar_history = deque(maxlen=lookback_bars)
        
        # Current state
        self.current_bar = None
        self.current_timestamp = None
        self.bar_count = 0
        self.trade_count = 0
        self.cumulative_pnl = 0.0  # running total net P&L from CLOSED trades
        self.last_trade_pnl = 0.0  # last CLOSED trade net P&L

    # ------------------------------------------------------------------
    # Day-aware window building (matches precompute_streaming_indicators.py)
    # ------------------------------------------------------------------

    def _build_day_index(self, data: pd.DataFrame) -> Tuple[List, Dict, Dict]:
        """Build day boundary index from the full dataset.

        Returns:
            unique_dates: ordered list of unique dates
            day_ranges: {date: (start_iloc, end_iloc)}  half-open
            date_to_order: {date: ordinal_position}
        """
        dates = data["date"].values
        unique_dates: List = []
        day_ranges: Dict = {}
        prev_date = None
        start_idx = 0
        for i, d in enumerate(dates):
            if d != prev_date:
                if prev_date is not None:
                    day_ranges[prev_date] = (start_idx, i)
                unique_dates.append(d)
                start_idx = i
                prev_date = d
        if prev_date is not None:
            day_ranges[prev_date] = (start_idx, len(dates))
        date_to_order = {d: i for i, d in enumerate(unique_dates)}
        return unique_dates, day_ranges, date_to_order

    def _build_day_window_df(
        self,
        current_iloc: int,
        current_date,
        data: pd.DataFrame,
        unique_dates: List,
        day_ranges: Dict,
        date_to_order: Dict,
    ) -> Tuple[pd.DataFrame, int]:
        """Build the indicator window for a bar, matching precompute logic.

        **Critical**: the window includes the FULL current day (all bars, not
        just up to current_iloc).  This matches precompute_streaming_indicators
        which calls calculate_core_indicators() once per day on the complete
        window and then extracts each bar's row.  Features like swing pivots
        (fractal look-forward) and volume profile (groupby-date) are computed
        over the full day — the training data reflects this, so the simulator
        must too.

        Returns:
            (window_df, row_offset)  where row_offset is the position of
            current_iloc within window_df (so the caller can extract the
            correct indicator row).
        """
        order = date_to_order[current_date]

        if order >= 1:
            prior_date = unique_dates[order - 1]
            prior_start, _ = day_ranges[prior_date]
            window_start = max(0, prior_start - self.warmup_bars)
        else:
            # First day — just warmup before current day start
            cur_start, _ = day_ranges[current_date]
            window_start = max(0, cur_start - self.warmup_bars)

        # Include full current day (not just up to current bar)
        _, target_day_end = day_ranges[current_date]
        window_end = target_day_end

        if window_end - window_start < 20:
            return pd.DataFrame(), -1

        row_offset = current_iloc - window_start
        return data.iloc[window_start:window_end], row_offset

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_pnl(self, x: float) -> str:
        return f"${x:+,.2f}"

    def _get_indicator_snapshot(self, indicators: Dict) -> str:
        """Format a small set of indicators for per-bar logging."""
        keys = self.log_indicators or ["vwap", "atr", "rsi", "is_long_setup"]
        parts = []
        for k in keys:
            if not indicators or k not in indicators:
                continue
            v = indicators.get(k)
            if isinstance(v, (bool, np.bool_)):
                parts.append(f"{k}={int(bool(v))}")
            elif v is None:
                parts.append(f"{k}=nan")
            else:
                try:
                    fv = float(v)
                    if not np.isfinite(fv):
                        parts.append(f"{k}=nan")
                    else:
                        parts.append(f"{k}={fv:.4f}")
                except Exception:
                    parts.append(f"{k}={v}")
        return " ".join(parts)

    def _log_bar_state(self, bar: dict, indicators: Dict, symbol: str):
        if not self.verbose:
            return
        if self.log_every_n_bars <= 0:
            return
        if self.bar_count % self.log_every_n_bars != 0:
            return

        o = float(bar.get("open", float("nan")))
        h = float(bar.get("high", float("nan")))
        l = float(bar.get("low", float("nan")))
        c = float(bar.get("close", float("nan")))

        ind_str = self._get_indicator_snapshot(indicators)
        mid = " | " + ind_str if ind_str else ""

        print(
            "  BAR "
            f"{self.bar_count:,} {bar.get('datetime')} "
            f"O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}"
            f"{mid} | Trade P&L={self._format_pnl(self.last_trade_pnl)} | Total P&L={self._format_pnl(self.cumulative_pnl)}",
            file=sys.stderr,
            flush=True,
        )

    def _calc_indicators(self, indicator_calculator, bars_df: pd.DataFrame) -> Dict:
        """Support either a callable calculator or an object with calculate()."""
        if indicator_calculator is None:
            return {}
        if hasattr(indicator_calculator, "calculate") and callable(getattr(indicator_calculator, "calculate")):
            return indicator_calculator.calculate(bars_df)
        return indicator_calculator(bars_df)

    def _calc_indicators_at_row(
        self, indicator_calculator, bars_df: pd.DataFrame, row_offset: int,
    ) -> Dict:
        """Calculate indicators on bars_df, then extract row at row_offset.

        This is used in day-window mode where the full current day is passed
        to calculate_core_indicators() (matching precompute), but we need
        the indicators for a specific bar (not the last row).
        """
        if indicator_calculator is None:
            return {}

        # We need the full DataFrame result, not just the last row.
        # StreamingIndicatorsAligned.calculate() returns only the last row,
        # so we call calculate_core_indicators directly.
        if hasattr(indicator_calculator, "calculate_core_indicators"):
            calc_fn = indicator_calculator.calculate_core_indicators
        elif hasattr(indicator_calculator, "calculate"):
            # Fallback: the calculator's calculate() returns last-row dict.
            # For row_offset == len(bars_df)-1, this is fine.
            if row_offset == len(bars_df) - 1:
                return indicator_calculator.calculate(bars_df)
            # Otherwise, we need the underlying function
            calc_fn = getattr(indicator_calculator, "calculate_core_indicators", None)
            if calc_fn is None:
                # Last resort: call calculate() — will be wrong for non-last rows
                return indicator_calculator.calculate(bars_df)
        else:
            # Callable — assume it returns last-row dict
            return indicator_calculator(bars_df)

        try:
            result_df = calc_fn(bars_df, verbose=False)
        except Exception:
            return {}

        if row_offset < 0 or row_offset >= len(result_df):
            return {}

        row = result_df.iloc[row_offset]
        indicators = {}
        for col in result_df.columns:
            val = row[col]
            if isinstance(val, (np.integer,)):
                indicators[col] = int(val)
            elif isinstance(val, (np.floating,)):
                indicators[col] = float(val)
            else:
                indicators[col] = val

        # Patch vol_pct_complete to streaming behaviour (matches precompute)
        indicators['vol_pct_complete'] = 1.0

        return indicators

    def run(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        indicator_calculator: Callable,
        model: Optional[any] = None,
        symbol: str = 'TSLA',
        target_start_iloc: int = 0,
    ) -> Dict:
        """
        Run streaming simulation

        Args:
            data: DataFrame with OHLCV data (datetime, open, high, low, close, volume).
                  Should include context/warmup bars BEFORE the target period.
            strategy_func: Strategy function with signature:
                func(bar, indicators, model, portfolio, order_manager)
            indicator_calculator: Function to calculate indicators from bar history:
                func(bars_df) -> dict of indicators for current bar
            model: Optional ML model (e.g., trained RandomForest)
            symbol: Trading symbol
            target_start_iloc: iloc index into ``data`` where the target (tradeable)
                period begins.  Bars before this index are used as warmup context
                only — no orders are processed or strategy executed on them.

        Returns:
            Dictionary with simulation results
        """
        # Ensure date column exists
        if "date" not in data.columns:
            data = data.copy()
            data["date"] = pd.to_datetime(data["datetime"]).dt.date

        # Reset index so iloc matches row position
        data = data.reset_index(drop=True)

        # Pre-build day index for day-aware windowing
        if self.use_day_window:
            unique_dates, day_ranges, date_to_order = self._build_day_index(data)

        if self.verbose:
            mode_str = f"day-window (warmup={self.warmup_bars})" if self.use_day_window else f"rolling deque({self.lookback_bars})"
            target_bars = len(data) - target_start_iloc
            print(f"\n{'='*80}", file=sys.stderr, flush=True)
            print(f"STARTING STREAMING SIMULATION (Paper Trading Mode)", file=sys.stderr, flush=True)
            print(f"{'='*80}", file=sys.stderr, flush=True)
            print(f"Initial Capital: ${self.initial_capital:,.2f}", file=sys.stderr, flush=True)
            print(f"Bar Interval: {self.bar_interval.seconds // 60} minutes", file=sys.stderr, flush=True)
            print(f"Window Mode: {mode_str}", file=sys.stderr, flush=True)
            print(f"Symbol: {symbol}", file=sys.stderr, flush=True)
            print(f"Data: {len(data):,} total bars ({target_bars:,} target, {target_start_iloc:,} context)", file=sys.stderr, flush=True)
            print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}", file=sys.stderr, flush=True)
            print(f"Target from iloc {target_start_iloc}: {data.iloc[target_start_iloc]['datetime']}", file=sys.stderr, flush=True)
            print(f"{'='*80}\n", file=sys.stderr, flush=True)

        # ---- Day-window mode: compute indicators lazily, one day at a time ----
        # Matches precompute_streaming_indicators.py exactly:
        #   build window (warmup + prior day + full current day),
        #   call calculate_core_indicators() ONCE per day,
        #   extract each bar's row from the result.
        #
        # Computed lazily on first bar of each new day — like paper trading
        # where you'd compute indicators when the day's data is available.
        day_indicator_cache = {}  # date -> pd.DataFrame (full indicator result)
        day_window_offset_cache = {}  # date -> window_start (so we can map iloc -> row)

        if self.use_day_window:
            # Get the calculate_core_indicators function from the calculator
            if hasattr(indicator_calculator, "calculate_core_indicators"):
                _core_calc_fn = indicator_calculator.calculate_core_indicators
            else:
                _core_calc_fn = None

        def _ensure_day_cached(bar_date):
            """Lazily compute and cache indicators for bar_date on first access."""
            if bar_date in day_indicator_cache:
                return
            if _core_calc_fn is None:
                return

            order = date_to_order[bar_date]
            if order >= 1:
                prior_date = unique_dates[order - 1]
                prior_start, _ = day_ranges[prior_date]
                w_start = max(0, prior_start - self.warmup_bars)
            else:
                cur_start, _ = day_ranges[bar_date]
                w_start = max(0, cur_start - self.warmup_bars)

            _, target_day_end = day_ranges[bar_date]
            w_end = target_day_end

            if w_end - w_start < 20:
                return

            window_df = data.iloc[w_start:w_end].copy().reset_index(drop=True)
            if "date" not in window_df.columns:
                window_df["date"] = pd.to_datetime(window_df["datetime"]).dt.date

            try:
                result_df = _core_calc_fn(window_df, verbose=False)
            except Exception:
                return

            day_indicator_cache[bar_date] = result_df
            day_window_offset_cache[bar_date] = w_start

            if self.verbose:
                print(f"  Day {len(day_indicator_cache):>3d}: computed indicators for {bar_date} "
                      f"(window={len(window_df)} bars)",
                      file=sys.stderr, flush=True)

        # Process each bar sequentially (simulating live feed)
        for iloc_idx in range(len(data)):
            row = data.iloc[iloc_idx]
            self.bar_count += 1
            self.current_timestamp = row['datetime']

            # Convert row to dict
            bar = row.to_dict()
            self.current_bar = bar
            is_target_bar = iloc_idx >= target_start_iloc

            # Legacy mode: always append to deque
            if not self.use_day_window:
                self.bar_history.append(bar)

            # Process existing orders against current bar (target bars only)
            if is_target_bar:
                self._process_pending_orders(bar, symbol)

            # Calculate indicators
            indicators = {}
            if is_target_bar:
                if self.use_day_window:
                    # Day-aware: lazily compute on first bar of each new day,
                    # then look up from cached full-day indicator DataFrame
                    bar_date = row["date"]
                    _ensure_day_cached(bar_date)
                    cached_df = day_indicator_cache.get(bar_date)
                    if cached_df is not None:
                        window_start = day_window_offset_cache[bar_date]
                        row_offset = iloc_idx - window_start
                        if 0 <= row_offset < len(cached_df):
                            ind_row = cached_df.iloc[row_offset]
                            indicators = {}
                            for col in cached_df.columns:
                                val = ind_row[col]
                                if isinstance(val, (np.integer,)):
                                    indicators[col] = int(val)
                                elif isinstance(val, (np.floating,)):
                                    indicators[col] = float(val)
                                else:
                                    indicators[col] = val
                            # Patch vol_pct_complete to streaming (matches precompute)
                            indicators['vol_pct_complete'] = 1.0
                else:
                    # Legacy: use deque
                    if len(self.bar_history) >= self.lookback_bars:
                        indicators = self._calc_indicators(indicator_calculator, self._get_bars_df())

            # Per-bar log from simulator (not indicator calculator)
            if is_target_bar:
                self._log_bar_state(bar, indicators, symbol)

            # Execute strategy (target bars only)
            if is_target_bar:
                strategy_func(bar, indicators, model, self.portfolio, self.order_manager)

            # Update equity curve (target bars only)
            if is_target_bar:
                current_prices = {}
                current_prices[symbol] = bar['close']
                for pos_symbol in list(self.portfolio.positions.keys()):
                    if pos_symbol.startswith(symbol) and pos_symbol != symbol:
                        current_prices[pos_symbol] = bar['close']
                self.portfolio.update_equity_curve(self.current_timestamp, current_prices)

            # Progress indicator every 1000 target bars
            if is_target_bar and self.verbose and (iloc_idx - target_start_iloc + 1) % 1000 == 0:
                current_prices = {symbol: bar['close']}
                equity = self.portfolio.get_equity(current_prices)
                n_trades = len(self.portfolio.get_trade_history())
                target_processed = iloc_idx - target_start_iloc + 1
                total_target = len(data) - target_start_iloc
                print(
                    f"  Processed {target_processed:,}/{total_target:,} target bars | "
                    f"Days cached: {len(day_indicator_cache)} | Trades: {n_trades} | "
                    f"Equity: ${equity:,.2f} | Total P&L: {self._format_pnl(self.cumulative_pnl)}",
                    file=sys.stderr,
                    flush=True,
                )

            # Evict completed days from cache to save memory
            # (keep at most current + 1 prior day)
            if self.use_day_window and is_target_bar and len(day_indicator_cache) > 2:
                bar_date = row["date"]
                for cached_date in list(day_indicator_cache.keys()):
                    if cached_date < bar_date:
                        # Check it's not the immediate prior day (might still
                        # be needed for warmup of tomorrow)
                        order_cur = date_to_order.get(bar_date, 0)
                        order_cached = date_to_order.get(cached_date, 0)
                        if order_cur - order_cached > 1:
                            del day_indicator_cache[cached_date]
                            del day_window_offset_cache[cached_date]

        # Close any remaining positions (all symbols)
        self._close_all_positions_multi(data.iloc[-1]['close'], data.iloc[-1]['datetime'])
        
        # Generate results
        results = self._generate_results(symbol)
        
        return results
    
    def _get_bars_df(self) -> pd.DataFrame:
        """Convert bar history deque to DataFrame for indicator calculation"""
        return pd.DataFrame(list(self.bar_history))
    
    def _process_pending_orders(self, bar: dict, symbol: str):
        """Process all pending orders against current bar"""
        active_orders = self.order_manager.get_active_orders()
        
        for order in active_orders:
            # Try to fill the order
            filled = self.order_manager.process_order(
                order=order,
                current_bar=bar,
                timestamp=self.current_timestamp,
                slippage_model=self.slippage_model,
            )
            
            if filled and order.status == OrderStatus.FILLED:
                # For readability, keep trade_count as "number of fills".
                self.trade_count += 1
                
                # Update portfolio with fill
                if order.side == OrderSide.BUY:
                    self.portfolio.open_position(
                        symbol=order.symbol,
                        quantity=order.filled_quantity,
                        price=order.avg_fill_price,
                        timestamp=self.current_timestamp,
                        fees=order.fees,
                    )
                    if self.verbose:
                        print(
                            f"  ENTRY #{self.trade_count}: {self.current_timestamp} | {order.symbol} | "
                            f"{order.filled_quantity} shares @ ${order.avg_fill_price:.2f} | "
                            f"Fees: ${order.fees:.2f}",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    self.portfolio.close_position(
                        symbol=order.symbol,
                        quantity=order.filled_quantity,
                        price=order.avg_fill_price,
                        timestamp=self.current_timestamp,
                        fees=order.fees,
                    )

                    # Authoritative P&L comes from portfolio trade history
                    self._refresh_pnl_state()

                    if self.verbose:
                        print(
                            f"  EXIT  #{self.trade_count}: {self.current_timestamp} | {order.symbol} | "
                            f"{order.filled_quantity} shares @ ${order.avg_fill_price:.2f} | "
                            f"Fees: ${order.fees:.2f} | Trade P&L={self._format_pnl(self.last_trade_pnl)} | "
                            f"Total P&L={self._format_pnl(self.cumulative_pnl)}",
                            file=sys.stderr,
                            flush=True,
                        )
    
    def _close_all_positions(self, symbol: str, price: float, timestamp: datetime):
        """Close all remaining positions at end of simulation (single symbol - legacy)"""
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            fees = self.broker_fees.calculate_commission(abs(pos.quantity), price)
            
            self.portfolio.close_position(
                symbol=symbol,
                quantity=None,  # Close all
                price=price,
                timestamp=timestamp,
                fees=fees,
            )

            self._refresh_pnl_state()

            if self.verbose:
                print(
                    f"\n  FINAL EXIT: {timestamp} | {symbol} | {abs(pos.quantity)} shares @ ${price:.2f} | "
                    f"Fees: ${fees:.2f} | Trade P&L={self._format_pnl(self.last_trade_pnl)} | "
                    f"Total P&L={self._format_pnl(self.cumulative_pnl)}",
                    file=sys.stderr,
                    flush=True,
                )
    
    def _close_all_positions_multi(self, price: float, timestamp: datetime):
        """Close all remaining positions at end of simulation (multi-symbol support)"""
        symbols_to_close = list(self.portfolio.positions.keys())
        
        for symbol in symbols_to_close:
            pos = self.portfolio.positions[symbol]
            fees = self.broker_fees.calculate_commission(abs(pos.quantity), price)
            
            self.portfolio.close_position(
                symbol=symbol,
                quantity=None,  # Close all
                price=price,
                timestamp=timestamp,
                fees=fees,
            )

            self._refresh_pnl_state()

            if self.verbose:
                print(
                    f"\n  FINAL EXIT: {timestamp} | {symbol} | {abs(pos.quantity)} shares @ ${price:.2f} | "
                    f"Fees: ${fees:.2f} | Trade P&L={self._format_pnl(self.last_trade_pnl)} | "
                    f"Total P&L={self._format_pnl(self.cumulative_pnl)}",
                    file=sys.stderr,
                    flush=True,
                )
    
    def _generate_results(self, symbol: str) -> Dict:
        """Generate simulation results"""
        if self.verbose:
            print(f"\n{'='*80}", file=sys.stderr, flush=True)
            print(f"STREAMING SIMULATION COMPLETE", file=sys.stderr, flush=True)
            print(f"{'='*80}\n", file=sys.stderr, flush=True)
        
        # Get statistics
        stats = self.portfolio.get_statistics()
        
        if self.verbose:
            print("Performance Summary:", file=sys.stderr, flush=True)
            print(f"  Total Trades: {stats['total_trades']:,}", file=sys.stderr, flush=True)
            print(f"  Win Rate: {stats['win_rate']*100:.1f}%", file=sys.stderr, flush=True)
            print(f"  Profit Factor: {stats['profit_factor']:.2f}", file=sys.stderr, flush=True)
            print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}", file=sys.stderr, flush=True)
            print(f"  Max Drawdown: {stats['max_drawdown_pct']:.1f}%", file=sys.stderr, flush=True)
            print(f"  Total Return: {stats['total_return_pct']:.1f}%", file=sys.stderr, flush=True)
            print(f"  Final Equity: ${stats['final_equity']:,.2f}", file=sys.stderr, flush=True)
            print(f"  Total Fees: ${stats['total_fees']:,.2f}", file=sys.stderr, flush=True)
            print(f"\nAverage Trade:", file=sys.stderr, flush=True)
            print(f"  Winning: ${stats['avg_win']:,.2f}", file=sys.stderr, flush=True)
            print(f"  Losing: ${stats['avg_loss']:,.2f}", file=sys.stderr, flush=True)
        
        # Compile results
        results = {
            'statistics': stats,
            'equity_curve': pd.DataFrame({
                'datetime': self.portfolio.timestamps,
                'equity': self.portfolio.equity_curve,
            }),
            'trade_history': self.portfolio.get_trade_history(),
            'portfolio': self.portfolio,
            'order_manager': self.order_manager,
            'bars_processed': self.bar_count,
        }
        
        return results
    
    def get_current_position(self, symbol: str):
        """Get current position for symbol"""
        return self.portfolio.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in symbol"""
        return symbol in self.portfolio.positions
    
    def get_equity(self, symbol: str, price: float) -> float:
        """Get current equity"""
        return self.portfolio.get_equity({symbol: price})

    def _get_cumulative_pnl_from_history(self) -> float:
        """Compute cumulative net P&L from the portfolio trade history.

        This avoids drift/double-counting from maintaining a separate counter and
        makes results directly comparable to vectorized analysis outputs.
        """
        trades = self.portfolio.get_trade_history()
        if trades is None or len(trades) == 0:
            return 0.0
        if 'net_pnl' in trades.columns:
            return float(trades['net_pnl'].sum())
        if 'pnl' in trades.columns:
            return float(trades['pnl'].sum())
        return 0.0

    def _refresh_pnl_state(self):
        """Refresh cached P&L fields from portfolio history (authoritative)."""
        self.cumulative_pnl = self._get_cumulative_pnl_from_history()
        trades = self.portfolio.get_trade_history()
        if trades is None or len(trades) == 0:
            self.last_trade_pnl = 0.0
            return
        last = trades.iloc[-1]
        self.last_trade_pnl = float(last.get("net_pnl", last.get("pnl", 0.0)) or 0.0)
