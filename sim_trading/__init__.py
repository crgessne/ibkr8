"""
Simulated Trading Package

This package provides a realistic trading simulation environment for backtesting
the RF VWAP reversion strategy with accurate execution modeling.

Components:
- simulator: Main trading simulator with position management
- order_manager: Order execution with realistic fills, slippage, and rejections
- execution_model: Execution quality models (limit orders, market orders, etc.)
- portfolio: Portfolio tracking with P&L, drawdown, and risk metrics
- broker_fees: Commission and fee models for different brokers
"""

__version__ = "0.1.0"

from .simulator import TradingSimulator
from .streaming_simulator import StreamingSimulator
from .streaming_indicators import StreamingIndicators
from .order_manager import OrderManager, Order, OrderType, OrderStatus, OrderSide
from .execution_model import ExecutionModel, LimitOrderModel, MarketOrderModel, ExecutionQuality
from .portfolio import Portfolio, Position
from .broker_fees import BrokerFees, IBKRFees

__all__ = [
    'TradingSimulator',
    'StreamingSimulator',
    'StreamingIndicators',
    'OrderManager',
    'Order',
    'OrderType',
    'OrderStatus',
    'OrderSide',
    'ExecutionModel',
    'LimitOrderModel',
    'MarketOrderModel',
    'ExecutionQuality',
    'Portfolio',
    'Position',
    'BrokerFees',
    'IBKRFees',
]
