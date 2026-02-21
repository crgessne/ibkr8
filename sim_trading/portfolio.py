"""
Portfolio Management - Track positions, P&L, and risk metrics

Handles:
- Position tracking (long/short)
- Realized and unrealized P&L
- Drawdown calculation
- Risk metrics (Sharpe, Sortino, etc.)
- Trade statistics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class Position:
    """
    Represents an open position
    
    Attributes:
        symbol: Trading symbol
        quantity: Number of shares (positive=long, negative=short)
        avg_entry_price: Average entry price
        entry_time: When position was opened
        unrealized_pnl: Current unrealized P&L
        realized_pnl: Realized P&L from partial closes
    """
    symbol: str
    quantity: int
    avg_entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def is_long(self) -> bool:
        return self.quantity > 0
    
    def is_short(self) -> bool:
        return self.quantity < 0
    
    def market_value(self, current_price: float) -> float:
        """Calculate current market value"""
        return abs(self.quantity) * current_price
    
    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L"""
        if self.quantity > 0:
            # Long position
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.quantity
        else:
            # Short position
            self.unrealized_pnl = (self.avg_entry_price - current_price) * abs(self.quantity)


@dataclass
class Trade:
    """Completed trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    fees: float
    net_pnl: float
    duration_bars: int
    return_pct: float


class Portfolio:
    """
    Portfolio manager tracking positions, P&L, and metrics
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_size: Optional[int] = None,
    ):
        """
        Initialize portfolio
        
        Args:
            initial_capital: Starting capital
            max_position_size: Maximum shares per position (None = unlimited)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_size = max_position_size
        
        # Positions
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        
        # P&L tracking
        self.equity_curve = []
        self.timestamps = []
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0.0
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
    
    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total equity (cash + positions)
        
        Args:
            current_prices: Dict of symbol -> current price
            
        Returns:
            Total portfolio equity
        """
        equity = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_unrealized_pnl(current_prices[symbol])
                equity += position.market_value(current_prices[symbol])
        
        return equity
    
    def open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        fees: float = 0.0,
    ) -> bool:
        """
        Open a new position or add to existing
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares (positive=buy, negative=sell short)
            price: Entry price
            timestamp: Entry timestamp
            fees: Commission/fees paid
            
        Returns:
            True if successful, False if insufficient capital
        """
        # Calculate cost
        cost = abs(quantity) * price + fees
        
        # Check if we have enough cash
        if cost > self.cash:
            return False
        
        # Check position size limit
        if self.max_position_size is not None:
            if symbol in self.positions:
                total_quantity = self.positions[symbol].quantity + quantity
                if abs(total_quantity) > self.max_position_size:
                    return False
            elif abs(quantity) > self.max_position_size:
                return False
        
        # Deduct cash
        self.cash -= cost
        self.total_fees += fees
        
        # Update or create position
        if symbol in self.positions:
            # Add to existing position
            pos = self.positions[symbol]
            old_quantity = pos.quantity
            old_value = old_quantity * pos.avg_entry_price
            new_value = quantity * price
            
            pos.quantity += quantity
            if pos.quantity != 0:
                pos.avg_entry_price = (old_value + new_value) / pos.quantity
            
            # If position closed completely, remove it
            if pos.quantity == 0:
                del self.positions[symbol]
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=price,
                entry_time=timestamp,
            )
        
        return True
    
    def close_position(
        self,
        symbol: str,
        quantity: Optional[int],
        price: float,
        timestamp: datetime,
        fees: float = 0.0,
    ) -> Optional[Trade]:
        """
        Close position (fully or partially)
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares to close (None = close all)
            price: Exit price
            timestamp: Exit timestamp
            fees: Commission/fees paid
            
        Returns:
            Trade record if position closed, None if position still open
        """
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # Determine quantity to close
        if quantity is None:
            close_qty = abs(pos.quantity)
        else:
            close_qty = min(abs(quantity), abs(pos.quantity))
        
        # Calculate P&L
        if pos.is_long():
            pnl = (price - pos.avg_entry_price) * close_qty
        else:
            pnl = (pos.avg_entry_price - price) * close_qty
        
        net_pnl = pnl - fees
        
        # Add cash back
        proceeds = close_qty * price - fees
        self.cash += proceeds
        self.total_fees += fees
        
        # Calculate return
        entry_value = close_qty * pos.avg_entry_price
        return_pct = (net_pnl / entry_value) * 100 if entry_value > 0 else 0.0
        
        # Create trade record
        duration_bars = (timestamp - pos.entry_time).total_seconds() / 300  # Assuming 5min bars
        
        trade = Trade(
            symbol=symbol,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            quantity=close_qty if pos.is_long() else -close_qty,
            entry_price=pos.avg_entry_price,
            exit_price=price,
            pnl=pnl,
            fees=fees,
            net_pnl=net_pnl,
            duration_bars=int(duration_bars),
            return_pct=return_pct,
        )
        
        self.closed_trades.append(trade)
        self.total_trades += 1
        
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update position
        pos.quantity -= close_qty if pos.is_long() else -close_qty
        pos.realized_pnl += net_pnl
        
        # Remove position if fully closed
        if abs(pos.quantity) < 1:
            del self.positions[symbol]
            return trade
        
        return None
    
    def update_equity_curve(self, timestamp: datetime, current_prices: Dict[str, float]):
        """Update equity curve with current value"""
        equity = self.get_equity(current_prices)
        self.equity_curve.append(equity)
        self.timestamps.append(timestamp)
        
        # Update drawdown tracking
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def get_statistics(self) -> dict:
        """Calculate portfolio statistics"""
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'total_return_pct': 0.0,
            }
        
        # Calculate basic stats
        wins = [t.net_pnl for t in self.closed_trades if t.net_pnl > 0]
        losses = [t.net_pnl for t in self.closed_trades if t.net_pnl <= 0]
        
        win_rate = self.winning_trades / self.total_trades
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # Calculate Sharpe ratio if we have equity curve
        sharpe_ratio = 0.0
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            if returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 78)  # Annualized for 5min bars
        
        # Total return
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_return_pct': total_return_pct,
            'total_fees': self.total_fees,
            'final_equity': final_equity,
        }
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.closed_trades:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'symbol': t.symbol,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'quantity': t.quantity,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'fees': t.fees,
                'net_pnl': t.net_pnl,
                'return_pct': t.return_pct,
                'duration_bars': t.duration_bars,
            }
            for t in self.closed_trades
        ])
