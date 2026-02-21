"""
Execution Models - Realistic fill simulation with slippage

Models different execution scenarios:
- Market orders with variable slippage
- Limit orders with probability-based fills
- Volume-based partial fills
- Spread-aware execution
"""

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from .order_manager import OrderSide


class ExecutionQuality(Enum):
    """Execution quality levels"""
    EXCELLENT = "excellent"  # HFT-like, sub-penny slippage
    GOOD = "good"            # Limit orders, join bid/ask
    AVERAGE = "average"      # Market orders, cross spread
    POOR = "poor"            # High slippage, market impact


class ExecutionModel(ABC):
    """Base class for execution models"""
    
    @abstractmethod
    def get_fill_price(
        self,
        base_price: float,
        order_side: OrderSide,
        quantity: int,
        bar_data: dict,
    ) -> float:
        """
        Calculate realistic fill price
        
        Args:
            base_price: Reference price (usually close)
            order_side: BUY or SELL
            quantity: Number of shares
            bar_data: OHLCV data for the current bar
            
        Returns:
            Actual fill price including slippage
        """
        pass


class MarketOrderModel(ExecutionModel):
    """
    Market order execution with configurable slippage
    
    Simulates crossing the spread and market impact
    """
    
    def __init__(
        self,
        base_slippage_pct: float = 0.0005,  # 5 bps
        volume_impact_factor: float = 0.0001,  # Additional slippage per 1% of volume
        volatility_multiplier: float = 1.0,  # Slippage increases with volatility
    ):
        """
        Initialize market order model
        
        Args:
            base_slippage_pct: Base slippage as percentage of price
            volume_impact_factor: Additional slippage based on order size vs bar volume
            volatility_multiplier: Multiplier for high volatility periods
        """
        self.base_slippage_pct = base_slippage_pct
        self.volume_impact_factor = volume_impact_factor
        self.volatility_multiplier = volatility_multiplier
    
    def get_fill_price(
        self,
        base_price: float,
        order_side: OrderSide,
        quantity: int,
        bar_data: dict,
    ) -> float:
        """Calculate market order fill price"""
        # Start with base slippage
        slippage_pct = self.base_slippage_pct
        
        # Add volume impact if volume data available
        if 'volume' in bar_data and bar_data['volume'] > 0:
            volume_pct = (quantity / bar_data['volume']) * 100
            slippage_pct += volume_pct * self.volume_impact_factor
        
        # Add volatility impact if ATR available
        if 'atr' in bar_data and bar_data.get('atr', 0) > 0:
            # Higher volatility = more slippage
            volatility_factor = (bar_data['atr'] / base_price) * self.volatility_multiplier
            slippage_pct += volatility_factor * 0.1  # Scale it down
        
        # Add random component (normal distribution)
        random_factor = np.random.normal(1.0, 0.2)  # Mean=1.0, StdDev=0.2
        slippage_pct *= random_factor
        
        # Ensure slippage is non-negative
        slippage_pct = max(0.0, slippage_pct)
        
        # Apply slippage based on side
        if order_side == OrderSide.BUY:
            fill_price = base_price * (1 + slippage_pct)
        else:
            fill_price = base_price * (1 - slippage_pct)
        
        return fill_price


class LimitOrderModel(ExecutionModel):
    """
    Limit order execution with probability-based fills
    
    Models:
    - Fill probability based on how far limit is from market
    - Partial fills in volatile markets
    - Queue position simulation
    """
    
    def __init__(
        self,
        aggressive_fill_prob: float = 0.8,  # Prob of fill if limit touches price
        passive_fill_prob: float = 0.3,     # Prob of fill if price is near limit
        spread_threshold: float = 0.001,    # 10 bps threshold for "near"
    ):
        """
        Initialize limit order model
        
        Args:
            aggressive_fill_prob: Probability of fill when limit price clearly touched
            passive_fill_prob: Probability of fill when price is just near limit
            spread_threshold: Threshold for determining if price is "near" limit
        """
        self.aggressive_fill_prob = aggressive_fill_prob
        self.passive_fill_prob = passive_fill_prob
        self.spread_threshold = spread_threshold
    
    def get_fill_price(
        self,
        base_price: float,
        order_side: OrderSide,
        quantity: int,
        bar_data: dict,
    ) -> float:
        """
        Calculate limit order fill price
        
        For limit orders, this returns the limit price itself if filled,
        but execution is determined in the order manager based on price action.
        """
        # Limit orders fill at the limit price (or better)
        # The order manager handles whether the order actually fills
        return base_price
    
    def should_fill(
        self,
        limit_price: float,
        current_price: float,
        bar_low: float,
        bar_high: float,
        order_side: OrderSide,
    ) -> bool:
        """
        Determine if limit order should fill based on probability
        
        Args:
            limit_price: The order's limit price
            current_price: Current market price (close)
            bar_low: Low of the bar
            bar_high: High of the bar
            order_side: BUY or SELL
            
        Returns:
            True if order should fill
        """
        if order_side == OrderSide.BUY:
            # Buy limit: needs price to drop to limit
            if bar_low <= limit_price:
                # Price touched limit, high probability of fill
                return np.random.random() < self.aggressive_fill_prob
            elif abs(current_price - limit_price) / limit_price < self.spread_threshold:
                # Price is near limit, lower probability
                return np.random.random() < self.passive_fill_prob
        else:
            # Sell limit: needs price to rise to limit
            if bar_high >= limit_price:
                # Price touched limit, high probability of fill
                return np.random.random() < self.aggressive_fill_prob
            elif abs(current_price - limit_price) / limit_price < self.spread_threshold:
                # Price is near limit, lower probability
                return np.random.random() < self.passive_fill_prob
        
        return False


class SlippageModelFactory:
    """Factory for creating execution models based on quality level"""
    
    @staticmethod
    def create_model(quality: ExecutionQuality) -> ExecutionModel:
        """
        Create execution model for given quality level
        
        Args:
            quality: Execution quality level
            
        Returns:
            Appropriate ExecutionModel instance
        """
        if quality == ExecutionQuality.EXCELLENT:
            # Sub-penny slippage, like HFT or very good limit orders
            return MarketOrderModel(
                base_slippage_pct=0.0001,  # 1 bp
                volume_impact_factor=0.00001,
                volatility_multiplier=0.5,
            )
        
        elif quality == ExecutionQuality.GOOD:
            # Good limit orders, minimal slippage
            return MarketOrderModel(
                base_slippage_pct=0.0005,  # 5 bps
                volume_impact_factor=0.0001,
                volatility_multiplier=0.8,
            )
        
        elif quality == ExecutionQuality.AVERAGE:
            # Market orders, crossing spread
            return MarketOrderModel(
                base_slippage_pct=0.001,   # 10 bps
                volume_impact_factor=0.0005,
                volatility_multiplier=1.0,
            )
        
        elif quality == ExecutionQuality.POOR:
            # Poor execution, high slippage
            return MarketOrderModel(
                base_slippage_pct=0.003,   # 30 bps
                volume_impact_factor=0.001,
                volatility_multiplier=1.5,
            )
        
        else:
            # Default to average
            return MarketOrderModel()
