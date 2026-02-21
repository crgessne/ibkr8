"""
Broker Fee Models - Realistic commission structures

Implements commission models for different brokers:
- Interactive Brokers (IBKR) Pro and Lite
- Other popular brokers
- Custom fee structures
"""

from abc import ABC, abstractmethod


class BrokerFees(ABC):
    """Base class for broker fee models"""
    
    @abstractmethod
    def calculate_commission(self, quantity: int, price: float) -> float:
        """
        Calculate commission for a trade
        
        Args:
            quantity: Number of shares
            price: Price per share
            
        Returns:
            Commission amount in dollars
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get broker name"""
        pass


class IBKRFees(BrokerFees):
    """
    Interactive Brokers Pro commission structure
    
    US Stocks:
    - $0.005 per share
    - Minimum $1.00 per order
    - Maximum 1% of trade value
    """
    
    def __init__(
        self,
        per_share: float = 0.005,
        min_per_order: float = 1.00,
        max_pct_of_value: float = 0.01,
    ):
        """
        Initialize IBKR fee model
        
        Args:
            per_share: Commission per share
            min_per_order: Minimum commission per order
            max_pct_of_value: Maximum commission as % of trade value
        """
        self.per_share = per_share
        self.min_per_order = min_per_order
        self.max_pct_of_value = max_pct_of_value
    
    def calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate IBKR Pro commission"""
        # Base commission
        commission = quantity * self.per_share
        
        # Apply minimum
        if commission < self.min_per_order:
            commission = self.min_per_order
        
        # Apply maximum (1% of trade value)
        trade_value = quantity * price
        max_commission = trade_value * self.max_pct_of_value
        if commission > max_commission:
            commission = max_commission
        
        return commission
    
    def get_name(self) -> str:
        return "IBKR Pro"


class IBKRLiteFees(BrokerFees):
    """
    Interactive Brokers Lite (commission-free)
    
    Note: Lite has wider spreads and payment for order flow,
    so effective cost is still non-zero (modeled as slippage)
    """
    
    def calculate_commission(self, quantity: int, price: float) -> float:
        """IBKR Lite has $0 commission"""
        return 0.0
    
    def get_name(self) -> str:
        return "IBKR Lite"


class FidelityFees(BrokerFees):
    """
    Fidelity commission structure (commission-free for stocks)
    """
    
    def calculate_commission(self, quantity: int, price: float) -> float:
        """Fidelity has $0 commission for stocks"""
        return 0.0
    
    def get_name(self) -> str:
        return "Fidelity"


class SchwabFees(BrokerFees):
    """
    Charles Schwab commission structure (commission-free for stocks)
    """
    
    def calculate_commission(self, quantity: int, price: float) -> float:
        """Schwab has $0 commission for stocks"""
        return 0.0
    
    def get_name(self) -> str:
        return "Charles Schwab"


class CustomFees(BrokerFees):
    """
    Custom fee structure
    
    Supports:
    - Per-share commission
    - Flat fee per trade
    - Percentage of trade value
    - Minimum and maximum caps
    """
    
    def __init__(
        self,
        name: str = "Custom",
        per_share: float = 0.0,
        flat_per_trade: float = 0.0,
        pct_of_value: float = 0.0,
        min_commission: float = 0.0,
        max_commission: float = None,
    ):
        """
        Initialize custom fee model
        
        Args:
            name: Broker name
            per_share: Commission per share
            flat_per_trade: Flat fee per trade
            pct_of_value: Commission as % of trade value
            min_commission: Minimum commission
            max_commission: Maximum commission (None = no cap)
        """
        self.broker_name = name
        self.per_share = per_share
        self.flat_per_trade = flat_per_trade
        self.pct_of_value = pct_of_value
        self.min_commission = min_commission
        self.max_commission = max_commission
    
    def calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate custom commission"""
        trade_value = quantity * price
        
        # Calculate all components
        commission = 0.0
        commission += quantity * self.per_share
        commission += self.flat_per_trade
        commission += trade_value * self.pct_of_value
        
        # Apply minimum
        if commission < self.min_commission:
            commission = self.min_commission
        
        # Apply maximum
        if self.max_commission is not None and commission > self.max_commission:
            commission = self.max_commission
        
        return commission
    
    def get_name(self) -> str:
        return self.broker_name


# Factory for creating broker fee models
class BrokerFeeFactory:
    """Factory for creating broker fee models"""
    
    @staticmethod
    def create(broker_name: str) -> BrokerFees:
        """
        Create fee model for specified broker
        
        Args:
            broker_name: One of 'ibkr', 'ibkr_lite', 'fidelity', 'schwab'
            
        Returns:
            Appropriate BrokerFees instance
        """
        broker_name = broker_name.lower()
        
        if broker_name in ['ibkr', 'ibkr_pro']:
            return IBKRFees()
        elif broker_name == 'ibkr_lite':
            return IBKRLiteFees()
        elif broker_name == 'fidelity':
            return FidelityFees()
        elif broker_name == 'schwab':
            return SchwabFees()
        else:
            # Default to IBKR Pro
            return IBKRFees()
