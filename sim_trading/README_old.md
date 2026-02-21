# Simulated Trading Package

A realistic trading simulation framework for backtesting strategies with accurate execution modeling.

## Features

- ✅ **Realistic Order Execution** - Market orders, limit orders, with probability-based fills
- ✅ **Slippage Models** - Configurable slippage based on volume, volatility, and execution quality
- ✅ **Commission Models** - Support for IBKR Pro, IBKR Lite, and custom fee structures
- ✅ **Portfolio Management** - Track positions, P&L, drawdown, and risk metrics
- ✅ **Complete Trade History** - Every fill, partial fill, and fee tracked
- ✅ **Performance Metrics** - Sharpe ratio, profit factor, win rate, drawdown, etc.

## Components

### 1. **TradingSimulator**
Main simulation engine that coordinates all components.

### 2. **OrderManager**
Handles order creation, validation, and execution with realistic fills.

### 3. **ExecutionModel**
Models slippage and execution quality:
- `ExecutionQuality.EXCELLENT` - HFT-like, ~1bp slippage
- `ExecutionQuality.GOOD` - Limit orders, ~5bp slippage
- `ExecutionQuality.AVERAGE` - Market orders, ~10bp slippage
- `ExecutionQuality.POOR` - High slippage, ~30bp

### 4. **Portfolio**
Tracks positions, P&L, and calculates risk metrics.

### 5. **BrokerFees**
Commission models for different brokers:
- IBKR Pro: $0.005/share (min $1, max 1% of value)
- IBKR Lite: $0 commission
- Custom: Define your own structure

## Quick Start

```python
from sim_trading import TradingSimulator, ExecutionQuality, OrderSide, OrderType
import pandas as pd

# Load your data
data = pd.read_csv('data/tsla_5min_10years.csv')
data['datetime'] = pd.to_datetime(data['time'], utc=True)

# Create simulator
sim = TradingSimulator(
    initial_capital=100000,
    execution_quality=ExecutionQuality.GOOD,
)

# Define strategy
def my_strategy(bar, portfolio, order_manager):
    symbol = 'TSLA'
    
    # Entry logic
    if bar['rf_prob'] >= 0.5 and symbol not in portfolio.positions:
        order = order_manager.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        order_manager.submit_order(order, bar['datetime'])
    
    # Exit logic
    elif symbol in portfolio.positions:
        pos = portfolio.positions[symbol]
        entry_price = pos.avg_entry_price
        current_price = bar['close']
        
        # Take profit or stop loss
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        if pnl_pct >= 2.0 or pnl_pct <= -1.0:
            order = order_manager.create_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=abs(pos.quantity),
                order_type=OrderType.MARKET,
            )
            order_manager.submit_order(order, bar['datetime'])

# Run simulation
results = sim.run(data, my_strategy, symbol='TSLA')

# Access results
print(f"Total Return: {results['statistics']['total_return_pct']:.1f}%")
print(f"Sharpe Ratio: {results['statistics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['statistics']['max_drawdown_pct']:.1f}%")

# Get trade history
trades = results['trade_history']
print(trades.head())

# Get equity curve
equity = results['equity_curve']
equity.plot(x='datetime', y='equity', title='Equity Curve')
```

## Advanced Usage

### Custom Slippage Model

```python
from sim_trading import MarketOrderModel

# Create custom slippage model
slippage_model = MarketOrderModel(
    base_slippage_pct=0.0005,  # 5 basis points
    volume_impact_factor=0.0001,
    volatility_multiplier=1.0,
)

sim = TradingSimulator(
    initial_capital=100000,
    slippage_model=slippage_model,
)
```

### Custom Broker Fees

```python
from sim_trading import CustomFees

# Define custom fee structure
fees = CustomFees(
    name="My Broker",
    per_share=0.01,
    flat_per_trade=5.0,
    min_commission=10.0,
)

sim = TradingSimulator(
    initial_capital=100000,
    broker_fees=fees,
)
```

### Limit Orders

```python
def strategy_with_limits(bar, portfolio, order_manager):
    symbol = 'TSLA'
    
    # Place limit order at 1% below current price
    if bar['rf_prob'] >= 0.5:
        limit_price = bar['close'] * 0.99
        
        order = order_manager.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
        )
        order_manager.submit_order(order, bar['datetime'])
```

## Integration with RF VWAP Strategy

See `examples/simulate_rf_vwap.py` for a complete example of simulating the RF VWAP reversion strategy.

## Differences from `master_pipeline.py`

| Feature | `master_pipeline.py` | `sim_trading` |
|---------|---------------------|---------------|
| **Approach** | Statistical backtest (vectorized) | Event-driven simulation (bar-by-bar) |
| **Order Modeling** | Assumes all orders fill | Realistic fills, rejections, partial fills |
| **Slippage** | Fixed per-share amount | Dynamic based on volume, volatility |
| **Position Management** | Theoretical | Actual position tracking |
| **Realism** | Optimistic | Realistic |
| **Speed** | Very fast | Slower (more detailed) |
| **Use Case** | Quick P&L estimates | Pre-live validation |

**Recommendation:** Use `master_pipeline.py` for rapid strategy development and optimization, then validate with `sim_trading` before going live.

## Output

The simulator returns a dictionary with:

```python
{
    'statistics': {
        'total_trades': 1234,
        'win_rate': 0.652,
        'profit_factor': 1.85,
        'sharpe_ratio': 1.42,
        'max_drawdown_pct': 8.5,
        'total_return_pct': 45.2,
        'final_equity': 145234.56,
        'total_fees': 3456.78,
    },
    'equity_curve': DataFrame with datetime and equity columns,
    'trade_history': DataFrame with all trades,
    'portfolio': Portfolio object,
    'order_manager': OrderManager object,
}
```

## Testing

Run the test suite:

```bash
python -m pytest sim_trading/tests/
```

## Next Steps

1. ✅ Basic simulator created
2. 🔄 Add example with RF VWAP strategy
3. 🔄 Add stop-loss and take-profit helpers
4. 🔄 Add position sizing strategies
5. 🔄 Add concurrent position management
6. 🔄 Add visualization tools
7. 🔄 Add Monte Carlo simulation
8. 🔄 Add walk-forward testing

## License

MIT License - See LICENSE file for details
