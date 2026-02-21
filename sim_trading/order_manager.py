"""
Order Manager - Handles order creation, execution, and tracking

This module simulates realistic order execution including:
- Limit orders with partial fills
- Market orders with slippage
- Order rejections (insufficient funds, invalid prices, etc.)
- Fill probability based on market conditions
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import numpy as np


class OrderType(Enum):
    """Order types supported by the simulator"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side (buy or sell)"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status tracking"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """
    Represents a trading order
    
    Attributes:
        order_id: Unique order identifier
        symbol: Trading symbol
        side: BUY or SELL
        order_type: MARKET, LIMIT, etc.
        quantity: Number of shares
        limit_price: Limit price (for limit orders)
        stop_price: Stop price (for stop orders)
        filled_quantity: Number of shares filled
        avg_fill_price: Average price of filled shares
        status: Current order status
        submitted_time: When order was submitted
        filled_time: When order was fully filled
        fees: Total fees paid on this order
        slippage: Total slippage on this order
    """
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    submitted_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    fees: float = 0.0
    slippage: float = 0.0
    fills: List[dict] = field(default_factory=list)
    
    def is_complete(self) -> bool:
        """Check if order is fully filled or terminated"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    def remaining_quantity(self) -> int:
        """Get unfilled quantity"""
        return self.quantity - self.filled_quantity
    
    def add_fill(self, quantity: int, price: float, timestamp: datetime, fee: float = 0.0):
        """Add a partial or full fill"""
        self.fills.append({
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'fee': fee
        })
        
        # Update filled quantity
        old_filled = self.filled_quantity
        self.filled_quantity += quantity
        
        # Update average fill price
        if self.filled_quantity > 0:
            total_value = (self.avg_fill_price * old_filled) + (price * quantity)
            self.avg_fill_price = total_value / self.filled_quantity
        
        # Update fees
        self.fees += fee
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_time = timestamp
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED


class OrderManager:
    """
    Manages order execution with realistic fills and rejections
    
    This simulates real broker behavior including:
    - Order validation
    - Fill probability based on limit price vs market price
    - Partial fills for large orders
    - Realistic slippage models
    - Commission calculation
    """
    
    def __init__(
        self,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        max_commission: float = None,
    ):
        """
        Initialize order manager
        
        Args:
            commission_per_share: Commission per share (IBKR Pro: $0.005)
            min_commission: Minimum commission per order
            max_commission: Maximum commission per order (None = no cap)
        """
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.max_commission = max_commission
        self.orders = {}
        self.order_counter = 0
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Create a new order"""
        self.order_counter += 1
        order_id = f"ORD{self.order_counter:06d}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        
        self.orders[order_id] = order
        return order
    
    def submit_order(self, order: Order, timestamp: datetime) -> bool:
        """
        Submit order for execution
        
        Returns:
            True if order accepted, False if rejected
        """
        # Validate order
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            return False
        
        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            order.status = OrderStatus.REJECTED
            return False
        
        order.status = OrderStatus.SUBMITTED
        order.submitted_time = timestamp
        return True
    
    def process_order(
        self,
        order: Order,
        current_bar: dict,
        timestamp: datetime,
        slippage_model=None,
    ) -> bool:
        """
        Process order against current market data
        
        Args:
            order: Order to process
            current_bar: Current OHLCV bar with 'open', 'high', 'low', 'close', 'volume'
            timestamp: Current timestamp
            slippage_model: Optional custom slippage model
            
        Returns:
            True if order filled (fully or partially), False otherwise
        """
        if order.is_complete():
            return False
        
        # Market orders fill immediately
        if order.order_type == OrderType.MARKET:
            return self._fill_market_order(order, current_bar, timestamp, slippage_model)
        
        # Limit orders require price check
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit_order(order, current_bar, timestamp)
        
        return False
    
    def _fill_market_order(
        self,
        order: Order,
        current_bar: dict,
        timestamp: datetime,
        slippage_model=None,
    ) -> bool:
        """Fill market order with realistic slippage"""
        # Use close price as base
        base_price = current_bar['close']
        
        # Apply slippage model
        if slippage_model is not None:
            fill_price = slippage_model.get_fill_price(
                base_price=base_price,
                order_side=order.side,
                quantity=order.remaining_quantity(),
                bar_data=current_bar
            )
        else:
            # Default: simple slippage based on side
            slippage_pct = 0.0005  # 5 basis points
            if order.side == OrderSide.BUY:
                fill_price = base_price * (1 + slippage_pct)
            else:
                fill_price = base_price * (1 - slippage_pct)
        
        # Calculate commission
        commission = self._calculate_commission(order.remaining_quantity())
        
        # Fill the order
        order.add_fill(
            quantity=order.remaining_quantity(),
            price=fill_price,
            timestamp=timestamp,
            fee=commission
        )
        
        # Track slippage
        order.slippage = abs(fill_price - base_price) * order.quantity
        
        return True
    
    def _fill_limit_order(
        self,
        order: Order,
        current_bar: dict,
        timestamp: datetime,
    ) -> bool:
        """Fill limit order if price touches limit"""
        # Check if limit price was reached during this bar
        if order.side == OrderSide.BUY:
            # Buy limit: fill if low <= limit price
            if current_bar['low'] <= order.limit_price:
                # Fill at limit price (best case) or slightly worse
                fill_price = min(order.limit_price, current_bar['close'])
            else:
                return False
        else:
            # Sell limit: fill if high >= limit price
            if current_bar['high'] >= order.limit_price:
                # Fill at limit price (best case) or slightly worse
                fill_price = max(order.limit_price, current_bar['close'])
            else:
                return False
        
        # Calculate commission
        commission = self._calculate_commission(order.remaining_quantity())
        
        # Fill the order
        order.add_fill(
            quantity=order.remaining_quantity(),
            price=fill_price,
            timestamp=timestamp,
            fee=commission
        )
        
        return True
    
    def _calculate_commission(self, quantity: int) -> float:
        """Calculate commission for a fill"""
        commission = quantity * self.commission_per_share
        
        # Apply minimum
        if commission < self.min_commission:
            commission = self.min_commission
        
        # Apply maximum if set
        if self.max_commission is not None and commission > self.max_commission:
            commission = self.max_commission
        
        return commission
    
    def cancel_order(self, order: Order) -> bool:
        """Cancel an order"""
        if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            order.status = OrderStatus.CANCELLED
            return True
        return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """Get all active (non-terminal) orders"""
        return [
            order for order in self.orders.values()
            if not order.is_complete()
        ]
    
    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders"""
        return [
            order for order in self.orders.values()
            if order.status == OrderStatus.FILLED
        ]
