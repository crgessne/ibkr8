"""
Example: Simulate RF VWAP Reversion Strategy

This script demonstrates how to use the trading simulator
with the RF VWAP reversion strategy from master_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Import simulator
from sim_trading import (
    TradingSimulator,
    ExecutionQuality,
    OrderSide,
    OrderType,
    IBKRFees,
)

# Import strategy components
from src.indicators import calc_all_indicators
from src.label_generator import LabelConfig, generate_labels


def prepare_data(filepath: str = "data/tsla_5min_10years.csv"):
    """Load and prepare data with indicators and signals"""
    print("Loading data...")
    df = pd.read_csv(filepath)
      # Handle time column
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'], utc=True)
    print(f"Loaded {len(df):,} bars")
    
    # Calculate indicators
    print("Calculating indicators...")
    df = calc_all_indicators(df)
      # Generate labels for a specific stop (e.g., 1.25 ATR)
    print("Generating labels...")
    stop_atr = 1.25
    config = LabelConfig(stop_atrs=[stop_atr])  # Only generate for 1.25 ATR
    labels_df = generate_labels(df, config)
      # Merge labels
    df = df.merge(labels_df[['label', 'rr', 'stop_price', 'target_price']], 
                  left_index=True, right_index=True, how='left')
    
    print(f"✓ Data prepared with {len(df.columns)} columns")
    print(f"  Columns: {', '.join(df.columns[:20])}...")
    print(f"  Has 'is_long_setup': {'is_long_setup' in df.columns}")
    print(f"  Has 'rf_prob': {'rf_prob' in df.columns}")
    print(f"  Has 'vwap': {'vwap' in df.columns}")
    
    return df, stop_atr


def train_rf_model(df, stop_atr, test_year=2024):
    """Train RF model on data before test_year"""
    print(f"\nTraining RF model (test year: {test_year})...")
    
    # Get feature columns (exclude targets, labels, etc.)
    exclude_cols = ['datetime', 'time', 'date', 'open', 'high', 'low', 'close', 'volume',
                   'label', 'rr', 'stop_price', 'target_price', 'wap']
    feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('label_')]
    
    # Filter valid data
    valid_mask = df['label'].notna()
    df_valid = df.loc[valid_mask].copy()
    df_valid['year'] = pd.to_datetime(df_valid['datetime']).dt.year
    
    # Split train/test
    train_mask = df_valid['year'] < test_year
    test_mask = df_valid['year'] >= test_year
    
    X_train = df_valid.loc[train_mask, feature_cols]
    y_train = df_valid.loc[train_mask, 'label']
    X_test = df_valid.loc[test_mask, feature_cols]
    y_test = df_valid.loc[test_mask, 'label']
    
    print(f"Training: {len(X_train):,} samples")
    print(f"Test: {len(X_test):,} samples")
    
    # Train RF
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=50,
        min_samples_split=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
    )
    
    rf.fit(X_train, y_train)
    
    # Get predictions on full dataset
    df.loc[valid_mask, 'rf_prob'] = rf.predict_proba(df.loc[valid_mask, feature_cols])[:, 1]
    
    print(f"✓ Model trained")
    print(f"  Train win rate: {y_train.mean()*100:.1f}%")
    print(f"  Test win rate: {y_test.mean()*100:.1f}%")
    
    return df


def create_rf_vwap_strategy(
    rf_threshold: float = 0.5,
    position_size: int = 100,
    stop_atr: float = 1.25,
    allow_concurrent: bool = False,  # NEW: Allow unlimited concurrent positions
):
    """
    Create RF VWAP strategy function
    
    Args:
        rf_threshold: Minimum RF probability to enter (e.g., 0.5)
        position_size: Number of shares per trade
        stop_atr: Stop width in ATR multiples
        allow_concurrent: If True, take ALL signals concurrently (ignore position limits)
        
    Returns:
        Strategy function compatible with simulator
    """
    
    entry_count = 0
    no_rf_count = 0
    no_setup_count = 0
    active_positions = {}  # Track multiple positions with unique IDs
    next_position_id = 1
    
    def strategy(bar, portfolio, order_manager):
        nonlocal entry_count, no_rf_count, no_setup_count, next_position_id
        symbol = 'TSLA'
        current_datetime = bar['datetime']
        
        # Check if we have RF prediction for this bar
        if pd.isna(bar.get('rf_prob', np.nan)):
            no_rf_count += 1
            if no_rf_count % 1000 == 0:
                print(f"  DEBUG: {no_rf_count:,} bars with no RF prediction", file=sys.stderr, flush=True)
            return
        
        # ENTRY LOGIC - Modified to allow concurrent positions
        rf_prob = bar['rf_prob']
        is_setup = bar.get('is_long_setup', False)
        
        # Check if we should enter (either no position OR concurrent mode)
        can_enter = allow_concurrent or (symbol not in portfolio.positions)
        
        if can_enter and rf_prob >= rf_threshold and is_setup:
            # Create market order
            entry_count += 1
            position_id = next_position_id
            next_position_id += 1
            
            print(f"  🎯 SIGNAL #{entry_count} (POS#{position_id}): {current_datetime} | RF={rf_prob:.3f} | Close=${bar['close']:.2f} | VWAP=${bar.get('vwap', 0):.2f}", file=sys.stderr, flush=True)
            order = order_manager.create_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=position_size,
                order_type=OrderType.MARKET,
            )
            order_manager.submit_order(order, current_datetime)
            
            # Track position with entry details
            active_positions[position_id] = {
                'entry_price': bar['close'],
                'entry_time': current_datetime,
                'atr': bar.get('atr', 0),
                'rr': bar.get('rr', 1.2),
                'quantity': position_size,
            }
        elif rf_prob >= rf_threshold and not is_setup:
            no_setup_count += 1
            if no_setup_count <= 5:  # Show first 5 missed signals
                print(f"  ⚠️ No setup: {current_datetime} | RF={rf_prob:.3f} | Close=${bar['close']:.2f} | VWAP=${bar.get('vwap', 0):.2f}", file=sys.stderr, flush=True)
        
        # EXIT LOGIC - Check ALL active positions for stops/targets
        if allow_concurrent:
            # Check each tracked position for exit
            positions_to_close = []
            
            for pos_id, pos_data in active_positions.items():
                entry_price = pos_data['entry_price']
                current_price = bar['close']
                atr = pos_data['atr']
                rr = pos_data['rr']
                
                # Stop loss
                stop_price = entry_price - (stop_atr * atr)
                # Target
                target_price = entry_price + (rr * stop_atr * atr)
                
                # Check if stop or target hit
                if current_price <= stop_price:
                    print(f"  🛑 STOP HIT (POS#{pos_id}): {current_datetime} | Entry=${entry_price:.2f} | Stop=${stop_price:.2f} | Current=${current_price:.2f}", file=sys.stderr, flush=True)
                    order = order_manager.create_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=pos_data['quantity'],
                        order_type=OrderType.MARKET,
                    )
                    order_manager.submit_order(order, current_datetime)
                    positions_to_close.append(pos_id)
                    
                elif current_price >= target_price:
                    print(f"  🎯 TARGET HIT (POS#{pos_id}): {current_datetime} | Entry=${entry_price:.2f} | Target=${target_price:.2f} | Current=${current_price:.2f}", file=sys.stderr, flush=True)
                    order = order_manager.create_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=pos_data['quantity'],
                        order_type=OrderType.MARKET,
                    )
                    order_manager.submit_order(order, current_datetime)
                    positions_to_close.append(pos_id)
            
            # Remove closed positions from tracking
            for pos_id in positions_to_close:
                del active_positions[pos_id]
        
        else:
            # Original single-position logic
            if symbol in portfolio.positions:
                pos = portfolio.positions[symbol]
                entry_price = pos.avg_entry_price
                current_price = bar['close']
                atr = bar.get('atr', 0)
                
                # Stop loss
                stop_price = entry_price - (stop_atr * atr)
                # Target
                rr = bar.get('rr', 1.2)
                target_price = entry_price + (rr * stop_atr * atr)
                
                # Exit if stop hit or target reached
                if current_price <= stop_price:
                    print(f"  🛑 STOP HIT: {current_datetime} | Entry=${entry_price:.2f} | Stop=${stop_price:.2f} | Current=${current_price:.2f}", file=sys.stderr, flush=True)
                    order = order_manager.create_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=abs(pos.quantity),
                        order_type=OrderType.MARKET,
                    )
                    order_manager.submit_order(order, current_datetime)
                elif current_price >= target_price:
                    print(f"  🎯 TARGET HIT: {current_datetime} | Entry=${entry_price:.2f} | Target=${target_price:.2f} | Current=${current_price:.2f}", file=sys.stderr, flush=True)
                    order = order_manager.create_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=abs(pos.quantity),
                        order_type=OrderType.MARKET,
                    )
                    order_manager.submit_order(order, current_datetime)
    
    return strategy


def main():
    """Run simulation"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Simulate RF VWAP Reversion Strategy')
    parser.add_argument('--year', type=int, default=2024,
                        help='Year to simulate (default: 2024). Use 0 for all years.')
    parser.add_argument('--rf-threshold', type=float, default=0.5,
                        help='RF probability threshold (default: 0.5)')
    parser.add_argument('--position-size', type=int, default=100,
                        help='Position size in shares (default: 100)')
    parser.add_argument('--stop-atr', type=float, default=1.25,
                        help='Stop width in ATR units (default: 1.25)')
    parser.add_argument('--concurrent', action='store_true',
                        help='Allow unlimited concurrent positions (test all signals)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" SIMULATED TRADING: RF VWAP REVERSION STRATEGY")
    print("="*80)
    print(f"Configuration:")
    print(f"  Year filter: {args.year if args.year > 0 else 'All years'}")
    print(f"  RF threshold: {args.rf_threshold}")
    print(f"  Position size: {args.position_size} shares")
    print(f"  Stop width: {args.stop_atr} ATR")
    print(f"  Concurrent positions: {'ENABLED (take all signals)' if args.concurrent else 'DISABLED (one at a time)'}")
    print("="*80)
    
    # 1. Prepare data
    df, stop_atr = prepare_data()
    stop_atr = args.stop_atr  # Use command-line value
    
    # 2. Train RF model (use year before test year for training)
    test_year = args.year if args.year > 0 else 2024
    df = train_rf_model(df, stop_atr, test_year=test_year)
    
    # 3. Filter to test period
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    
    if args.year > 0:
        # Filter to specific year
        df_test = df[df['year'] == args.year].copy()
        print(f"\n✓ Filtering to year {args.year}...")
    else:
        # Use all years
        df_test = df.copy()
        print(f"\n✓ Using all years...")
    
    df_test = df_test.reset_index(drop=True)
    
    if len(df_test) == 0:
        print(f"\n❌ No data found for year {args.year}")
        return None
    
    print(f"✓ Test data: {len(df_test):,} bars ({df_test['datetime'].min()} to {df_test['datetime'].max()})")
    
    # 4. Create simulator
    print("\nInitializing simulator...")
    sim = TradingSimulator(
        initial_capital=100000,
        execution_quality=ExecutionQuality.GOOD,  # Assume good limit order execution
        broker_fees=IBKRFees(),  # IBKR Pro fees
        max_position_size=None if args.concurrent else 1000,  # Unlimited if concurrent mode
    )
      # 5. Create strategy
    strategy = create_rf_vwap_strategy(
        rf_threshold=args.rf_threshold,
        position_size=args.position_size,
        stop_atr=stop_atr,
        allow_concurrent=args.concurrent,  # NEW: Enable concurrent mode
    )
    
    # 6. Run simulation
    results = sim.run(
        data=df_test,
        strategy_func=strategy,
        symbol='TSLA',
    )
    
    # 7. Analyze results
    print(f"\n{'='*80}")
    print(" DETAILED RESULTS")
    print(f"{'='*80}\n")
    
    stats = results['statistics']
    
    print("📈 Returns:")
    print(f"  Initial Capital: ${sim.initial_capital:,.2f}")
    print(f"  Final Equity: ${stats['final_equity']:,.2f}")
    print(f"  Total Return: {stats['total_return_pct']:+.2f}%")
    print(f"  Total Fees: ${stats['total_fees']:,.2f}")
    
    print(f"\n📊 Trade Statistics:")
    print(f"  Total Trades: {stats['total_trades']:,}")
    print(f"  Winning Trades: {stats['winning_trades']:,}")
    print(f"  Losing Trades: {stats['losing_trades']:,}")
    print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
    
    print(f"\n💰 P&L Metrics:")
    print(f"  Average Win: ${stats['avg_win']:,.2f}")
    print(f"  Average Loss: ${stats['avg_loss']:,.2f}")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    
    print(f"\n⚠️ Risk Metrics:")
    print(f"  Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    
    # 8. Show sample trades
    trades = results['trade_history']
    if len(trades) > 0:
        print(f"\n📝 First 5 Trades:")
        print(trades.head().to_string())
        
        print(f"\n📝 Last 5 Trades:")
        print(trades.tail().to_string())
    
    # 9. Save results
    output_path = Path("data/sim_trading_results.csv")
    trades.to_csv(output_path, index=False)
    print(f"\n✓ Trade history saved to: {output_path}")
    
    equity_path = Path("data/sim_trading_equity.csv")
    results['equity_curve'].to_csv(equity_path, index=False)
    print(f"✓ Equity curve saved to: {equity_path}")
    
    print(f"\n{'='*80}")
    print(" SIMULATION COMPLETE")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    results = main()
