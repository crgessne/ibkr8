"""
P&L Calculator for VWAP Reversal Strategy

Calculates realistic profit/loss with:
- Position sizing: 100 shares per trade
- Maximum exposure: $1,000,000 simultaneous positions
- VWAP mean-reversion targets
- ATR-based stops

Integrates with indicators.py pipeline.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PnLConfig:
    """Configuration for P&L calculations."""
    shares_per_trade: int = 100
    max_simultaneous_exposure: float = 1_000_000.0
    stop_atr_width: float = 0.25
    target_rr: float = 6.0  # Risk:Reward ratio
    commission_per_share: float = 0.005  # $0.005 per share per side
    slippage_per_share: float = 0.01  # $0.01 per share


@dataclass
class TradeResult:
    """Single trade result."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    shares: int
    pnl_gross: float
    pnl_net: float
    outcome: str  # 'TARGET', 'STOP', 'TIMEOUT'
    bars_held: int
    capital_used: float
    
    def to_dict(self) -> Dict:
        return {
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'shares': self.shares,
            'pnl_gross': self.pnl_gross,
            'pnl_net': self.pnl_net,
            'outcome': self.outcome,
            'bars_held': self.bars_held,
            'capital_used': self.capital_used,
        }


@dataclass
class PnLSummary:
    """Summary statistics for P&L."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_bars_held: float
    total_commission: float
    total_slippage: float
    # EV metrics (Expected Value)
    ev_per_trade: float = 0.0
    total_ev: float = 0.0
    avg_pnl_per_trade: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_bars_held': self.avg_bars_held,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'ev_per_trade': self.ev_per_trade,
            'total_ev': self.total_ev,
            'avg_pnl_per_trade': self.avg_pnl_per_trade,
        }


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    stop_price: float,
    target_price: float,
    shares: int,
    config: PnLConfig,
    max_bars_held: int = 100
) -> TradeResult:
    """
    Simulate a single trade from entry to exit.
    
    Args:
        df: DataFrame with OHLC data
        entry_idx: Index where trade enters
        direction: 'LONG' or 'SHORT'
        stop_price: Stop loss price
        target_price: Target profit price
        shares: Number of shares to trade
        config: P&L configuration
        max_bars_held: Maximum bars to hold (timeout)
        
    Returns:
        TradeResult with outcome
    """
    entry_bar = df.iloc[entry_idx]
    entry_price = entry_bar['close']
    entry_time = entry_bar.name if hasattr(entry_bar, 'name') else entry_bar.get('time', entry_idx)
    
    # Look for exit in future bars
    exit_idx = entry_idx + 1
    outcome = 'TIMEOUT'
    exit_price = entry_price
    exit_time = entry_time
    bars_held = 0
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars_held + 1, len(df))):
        bar = df.iloc[i]
        bars_held = i - entry_idx
        
        if direction == 'LONG':
            # Check if hit stop (low of bar)
            if bar['low'] <= stop_price:
                outcome = 'STOP'
                exit_price = stop_price
                exit_time = bar.name if hasattr(bar, 'name') else bar.get('time', i)
                break
            # Check if hit target (high of bar)
            elif bar['high'] >= target_price:
                outcome = 'TARGET'
                exit_price = target_price
                exit_time = bar.name if hasattr(bar, 'name') else bar.get('time', i)
                break
        
        elif direction == 'SHORT':
            # Check if hit stop (high of bar)
            if bar['high'] >= stop_price:
                outcome = 'STOP'
                exit_price = stop_price
                exit_time = bar.name if hasattr(bar, 'name') else bar.get('time', i)
                break
            # Check if hit target (low of bar)
            elif bar['low'] <= target_price:
                outcome = 'TARGET'
                exit_price = target_price
                exit_time = bar.name if hasattr(bar, 'name') else bar.get('time', i)
                break
    
    # If timeout, exit at close of last bar checked
    if outcome == 'TIMEOUT':
        final_bar = df.iloc[min(entry_idx + max_bars_held, len(df) - 1)]
        exit_price = final_bar['close']
        exit_time = final_bar.name if hasattr(final_bar, 'name') else final_bar.get('time', len(df) - 1)
    
    # Calculate P&L
    if direction == 'LONG':
        pnl_gross = (exit_price - entry_price) * shares
    else:  # SHORT
        pnl_gross = (entry_price - exit_price) * shares
    
    # Subtract costs
    commission = 2 * config.commission_per_share * shares  # Entry + Exit
    slippage = 2 * config.slippage_per_share * shares  # Entry + Exit
    pnl_net = pnl_gross - commission - slippage
    
    capital_used = entry_price * shares
    
    return TradeResult(
        entry_time=entry_time,
        exit_time=exit_time,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_price=stop_price,
        target_price=target_price,
        shares=shares,
        pnl_gross=pnl_gross,
        pnl_net=pnl_net,
        outcome=outcome,
        bars_held=bars_held,
        capital_used=capital_used,
    )


def calculate_pnl(
    df: pd.DataFrame,
    entry_signals: pd.Series,
    config: Optional[PnLConfig] = None
) -> Tuple[List[TradeResult], PnLSummary]:
    """
    Calculate P&L for a series of entry signals.
    
    Args:
        df: DataFrame with OHLC data and indicators (must have 'close', 'vwap', 'atr')
        entry_signals: Boolean series indicating entry bars (True = enter trade)
        config: P&L configuration (uses defaults if None)
        
    Returns:
        Tuple of (list of TradeResults, PnLSummary)
    """
    if config is None:
        config = PnLConfig()
    
    trades: List[TradeResult] = []
    active_exposure = 0.0
    last_exit_idx = -1
    
    # Iterate through entry signals
    for idx in range(len(df)):
        if not entry_signals.iloc[idx]:
            continue
        
        # Skip if we just exited (prevent immediate re-entry)
        if idx <= last_exit_idx:
            continue
        
        bar = df.iloc[idx]
        close = bar['close']
        vwap = bar['vwap']
        atr = bar['atr']
        
        # Determine direction based on price vs VWAP
        if close < vwap:
            direction = 'LONG'
            stop_price = close - (config.stop_atr_width * atr)
            risk_per_share = close - stop_price
            target_price = close + (risk_per_share * config.target_rr)
            # But cap target at VWAP for mean reversion
            target_price = min(target_price, vwap)
        elif close > vwap:
            direction = 'SHORT'
            stop_price = close + (config.stop_atr_width * atr)
            risk_per_share = stop_price - close
            target_price = close - (risk_per_share * config.target_rr)
            # But cap target at VWAP for mean reversion
            target_price = max(target_price, vwap)
        else:
            continue  # Price at VWAP, skip
        
        # Check if we have room for this trade (max exposure limit)
        trade_exposure = close * config.shares_per_trade
        if active_exposure + trade_exposure > config.max_simultaneous_exposure:
            continue  # Skip, would exceed max exposure
        
        # Simulate the trade
        result = simulate_trade(
            df=df,
            entry_idx=idx,
            direction=direction,
            stop_price=stop_price,
            target_price=target_price,
            shares=config.shares_per_trade,
            config=config,
            max_bars_held=100
        )
        
        trades.append(result)
        
        # Update tracking
        active_exposure += trade_exposure
        last_exit_idx = idx + result.bars_held
        # Reduce exposure when trade exits
        active_exposure = max(0, active_exposure - trade_exposure)
    
    # Calculate summary statistics
    if not trades:
        return trades, PnLSummary(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            total_pnl=0.0, avg_win=0.0, avg_loss=0.0, largest_win=0.0,
            largest_loss=0.0, profit_factor=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, avg_bars_held=0.0, total_commission=0.0,
            total_slippage=0.0
        )
    
    pnls = [t.pnl_net for t in trades]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p < 0]
    
    total_pnl = sum(pnls)
    win_rate = len(winning) / len(trades) if trades else 0
    avg_win = np.mean(winning) if winning else 0
    avg_loss = np.mean(losing) if losing else 0
    largest_win = max(winning) if winning else 0
    largest_loss = min(losing) if losing else 0
    
    gross_profit = sum(winning) if winning else 0
    gross_loss = abs(sum(losing)) if losing else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Calculate Sharpe ratio (annualized, assuming 252 trading days)
    if len(pnls) > 1:
        pnl_std = np.std(pnls)
        if pnl_std > 0:
            sharpe_ratio = (np.mean(pnls) / pnl_std) * np.sqrt(252)
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
    
    # Calculate max drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
      avg_bars_held = np.mean([t.bars_held for t in trades])
    
    total_commission = sum(2 * config.commission_per_share * t.shares for t in trades)
    total_slippage = sum(2 * config.slippage_per_share * t.shares for t in trades)
    
    # Calculate EV metrics
    avg_pnl_per_trade = np.mean(pnls)
    ev_per_trade = avg_pnl_per_trade
    total_ev = ev_per_trade * len(trades)
    
    summary = PnLSummary(
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        avg_bars_held=avg_bars_held,
        total_commission=total_commission,
        total_slippage=total_slippage,
        ev_per_trade=ev_per_trade,
        total_ev=total_ev,
        avg_pnl_per_trade=avg_pnl_per_trade,
    )
    
    return trades, summary


def print_pnl_summary(summary: PnLSummary) -> None:
    """Print formatted P&L summary."""
    print("\n" + "="*70)
    print("P&L SUMMARY")
    print("="*70)
    print(f"Total Trades:        {summary.total_trades:,}")
    print(f"Winning Trades:      {summary.winning_trades:,} ({summary.win_rate*100:.1f}%)")
    print(f"Losing Trades:       {summary.losing_trades:,}")
    print()
    print(f"Total P&L:           ${summary.total_pnl:,.2f}")
    print(f"Total EV:            ${summary.total_ev:,.2f}")
    print(f"EV per Trade:        ${summary.ev_per_trade:,.2f}")
    print(f"Avg P&L per Trade:   ${summary.avg_pnl_per_trade:,.2f}")
    print(f"Average Win:         ${summary.avg_win:,.2f}")
    print(f"Average Loss:        ${summary.avg_loss:,.2f}")
    print(f"Largest Win:         ${summary.largest_win:,.2f}")
    print(f"Largest Loss:        ${summary.largest_loss:,.2f}")
    print()
    print(f"Profit Factor:       {summary.profit_factor:.2f}")
    print(f"Sharpe Ratio:        {summary.sharpe_ratio:.2f}")
    print(f"Max Drawdown:        ${summary.max_drawdown:,.2f}")
    print(f"Avg Bars Held:       {summary.avg_bars_held:.1f}")
    print()
    print(f"Total Commission:    ${summary.total_commission:,.2f}")
    print(f"Total Slippage:      ${summary.total_slippage:,.2f}")
    print("="*70)


def save_trade_log(trades: List[TradeResult], filepath: str) -> None:
    """Save trade log to CSV."""
    df = pd.DataFrame([t.to_dict() for t in trades])
    df.to_csv(filepath, index=False)
    print(f"\nTrade log saved to: {filepath}")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, 'src')
    from indicators import calc_all_indicators
    
    # Load data
    df = pd.read_csv('data/tsla_5min_2025_01.csv', parse_dates=['time'])
    if 'time' in df.columns:
        df = df.set_index('time')
    
    print(f"Loaded {len(df)} bars")
    
    # Calculate indicators
    print("Calculating indicators...")
    df = calc_all_indicators(df)
    
    # Create simple entry signals (price extended from VWAP)
    entry_signals = (
        (df['vwap_width_atr'] >= 0.5) & 
        (df['vwap_width_atr'] <= 2.0)
    )
    
    print(f"Found {entry_signals.sum()} potential entry signals")
    
    # Calculate P&L
    print("\nCalculating P&L...")
    config = PnLConfig(
        shares_per_trade=100,
        max_simultaneous_exposure=1_000_000,
        stop_atr_width=0.25,
        target_rr=6.0
    )
    
    trades, summary = calculate_pnl(df, entry_signals, config)
    
    # Print results
    print_pnl_summary(summary)
    
    # Save trade log
    save_trade_log(trades, 'data/pnl_trade_log.csv')
