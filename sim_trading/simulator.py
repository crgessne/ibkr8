"""
Trading Simulator - Main simulation engine

Coordinates all components to simulate realistic trading:
- Processes bar-by-bar market data
- Executes strategy signals
- Manages orders and positions
- Tracks performance metrics
- Generates detailed reports
"""

from typing import Callable, Dict, List, Optional
from datetime import datetime
import sys
import pandas as pd
import numpy as np

from .order_manager import OrderManager, Order, OrderType, OrderSide, OrderStatus
from .execution_model import ExecutionModel, ExecutionQuality, SlippageModelFactory
from .portfolio import Portfolio
from .broker_fees import BrokerFees, IBKRFees


class TradingSimulator:
    """
    Main trading simulator
    
    Example usage:
        # Create simulator
        sim = TradingSimulator(
            initial_capital=100000,
            execution_quality=ExecutionQuality.GOOD,
        )
        
        # Define strategy function
        def strategy(bar, portfolio, order_manager):
            # Your strategy logic here
            if should_buy(bar):
                order = order_manager.create_order(
                    symbol='TSLA',
                    side=OrderSide.BUY,
                    quantity=100,
                    order_type=OrderType.MARKET,
                )
                order_manager.submit_order(order, bar['datetime'])
        
        # Run simulation
        results = sim.run(data, strategy)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        execution_quality: ExecutionQuality = ExecutionQuality.AVERAGE,
        broker_fees: Optional[BrokerFees] = None,
        max_position_size: Optional[int] = None,
        slippage_model: Optional[ExecutionModel] = None,
    ):
        """
        Initialize simulator
        
        Args:
            initial_capital: Starting capital
            execution_quality: Execution quality level (if slippage_model not provided)
            broker_fees: Broker fee model (default: IBKR Pro)
            max_position_size: Maximum shares per position
            slippage_model: Custom slippage model (overrides execution_quality)
        """
        self.initial_capital = initial_capital
        
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
        
        # Tracking
        self.current_bar = None
        self.current_timestamp = None
        self.bar_count = 0
    
    def run(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        symbol: str = 'TSLA',
    ) -> Dict:
        """
        Run simulation on historical data
        
        Args:
            data: DataFrame with OHLCV data (columns: datetime, open, high, low, close, volume)
            strategy_func: Strategy function with signature: func(bar, portfolio, order_manager)
            symbol: Trading symbol
            
        Returns:
            Dictionary with simulation results
        """
        print(f"\n{'='*80}")
        print(f"STARTING TRADING SIMULATION")
        print(f"{'='*80}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Broker: {self.broker_fees.get_name()}")
        print(f"Symbol: {symbol}")
        print(f"Data: {len(data):,} bars from {data['datetime'].min()} to {data['datetime'].max()}")
        print(f"{'='*80}\n")
        
        # Process each bar
        for idx, row in data.iterrows():
            self.bar_count += 1
            self.current_timestamp = row['datetime']
            
            # Convert row to dict for easier access
            bar = row.to_dict()
            self.current_bar = bar
            
            # Process existing orders
            self._process_pending_orders(bar)
            
            # Execute strategy
            strategy_func(bar, self.portfolio, self.order_manager)
            
            # Update equity curve
            current_prices = {symbol: bar['close']}
            self.portfolio.update_equity_curve(self.current_timestamp, current_prices)
            
            # Progress indicator every 1000 bars
            if self.bar_count % 1000 == 0:
                equity = self.portfolio.get_equity(current_prices)
                print(f"Processed {self.bar_count:,} bars | Equity: ${equity:,.2f}")
        
        # Close any remaining positions
        self._close_all_positions(symbol, data.iloc[-1]['close'], data.iloc[-1]['datetime'])
          # Generate results
        results = self._generate_results(symbol)
        
        return results
    
    def _process_pending_orders(self, bar: dict):
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
                # Update portfolio with fill
                if order.side == OrderSide.BUY:
                    self.portfolio.open_position(
                        symbol=order.symbol,
                        quantity=order.filled_quantity,
                        price=order.avg_fill_price,
                        timestamp=self.current_timestamp,
                        fees=order.fees,
                    )
                    print(f"  ✅ ENTRY: {self.current_timestamp} | {order.symbol} | {order.filled_quantity} shares @ ${order.avg_fill_price:.2f} | Fees: ${order.fees:.2f}", file=sys.stderr, flush=True)
                else:
                    self.portfolio.close_position(
                        symbol=order.symbol,
                        quantity=order.filled_quantity,
                        price=order.avg_fill_price,
                        timestamp=self.current_timestamp,
                        fees=order.fees,
                    )
                    # Calculate P&L for this trade
                    trades = self.portfolio.get_trade_history()
                    if len(trades) > 0:
                        last_trade = trades[-1]
                        pnl = last_trade.get('pnl', 0)
                        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                        print(f"  ❌ EXIT:  {self.current_timestamp} | {order.symbol} | {order.filled_quantity} shares @ ${order.avg_fill_price:.2f} | Fees: ${order.fees:.2f} | P&L: {pnl_str}", file=sys.stderr, flush=True)
    
    def _close_all_positions(self, symbol: str, price: float, timestamp: datetime):
        """Close all remaining positions at end of simulation"""
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
            
            # Calculate final P&L
            trades = self.portfolio.get_trade_history()
            if len(trades) > 0:
                last_trade = trades[-1]
                pnl = last_trade.get('pnl', 0)
                pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                print(f"\n  ❌ FINAL EXIT: {timestamp} | {symbol} | {abs(pos.quantity)} shares @ ${price:.2f} | Fees: ${fees:.2f} | P&L: {pnl_str}", file=sys.stderr, flush=True)
            else:
                print(f"\n  ❌ FINAL EXIT: {timestamp} | {symbol} | {abs(pos.quantity)} shares @ ${price:.2f} | Fees: ${fees:.2f}", file=sys.stderr, flush=True)
    
    def _generate_results(self, symbol: str) -> Dict:
        """Generate simulation results"""
        print(f"\n{'='*80}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*80}\n")
        
        # Get statistics
        stats = self.portfolio.get_statistics()
        
        # Print summary
        print("📊 Performance Summary:")
        print(f"  Total Trades: {stats['total_trades']:,}")
        print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {stats['max_drawdown_pct']:.1f}%")
        print(f"  Total Return: {stats['total_return_pct']:.1f}%")
        print(f"  Final Equity: ${stats['final_equity']:,.2f}")
        print(f"  Total Fees: ${stats['total_fees']:,.2f}")
        print(f"\n💰 Average Trade:")
        print(f"  Winning: ${stats['avg_win']:,.2f}")
        print(f"  Losing: ${stats['avg_loss']:,.2f}")
        
        # Compile results
        results = {
            'statistics': stats,
            'equity_curve': pd.DataFrame({
                'datetime': self.portfolio.timestamps,
                'equity': self.portfolio.equity_curve,
            }),
            'trade_history': self.portfolio.get_trade_history(),
            'portfolio': self.portfolio,
            'order_manager': self.order_manager,        }
        
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


# Helper function to create a basic strategy
def create_simple_strategy(
    entry_signal_func: Callable,
    exit_signal_func: Callable,
    position_size: int = 100,
) -> Callable:
    """
    Create a simple strategy from entry/exit signal functions
    
    Args:
        entry_signal_func: Function that takes a bar and returns True to enter
        exit_signal_func: Function that takes a bar and returns True to exit
        position_size: Number of shares per trade
        
    Returns:
        Strategy function compatible with simulator
    """
    def strategy(bar, portfolio, order_manager):
        symbol = 'TSLA'  # Default symbol
        
        # Check if we have a position
        has_position = symbol in portfolio.positions
        
        if not has_position:
            # Look for entry signal
            if entry_signal_func(bar):
                order = order_manager.create_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=position_size,
                    order_type=OrderType.MARKET,
                )
                order_manager.submit_order(order, bar['datetime'])
        else:
            # Look for exit signal
            if exit_signal_func(bar):
                pos = portfolio.positions[symbol]
                order = order_manager.create_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=abs(pos.quantity),
                    order_type=OrderType.MARKET,
                )
                order_manager.submit_order(order, bar['datetime'])
    
    return strategy
