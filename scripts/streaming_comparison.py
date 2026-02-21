"""
Streaming Simulator Comparison - Test Multiple Stop Widths

Runs the streaming simulator with different stop widths to find
the optimal balance between theoretical edge and realistic execution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import argparse
from datetime import datetime

# Import the strategy and simulator
from sim_trading.streaming_simulator import StreamingSimulator
from sim_trading.order_manager import Order, OrderType, OrderSide


def load_trained_model(stop_atr: float):
    """Load the trained model for a specific stop width"""
    from model_persistence import load_model
    
    # Find the latest model file for this stop
    models_dir = Path("models")
    stop_str = f"{stop_atr:.2f}".replace(".", "_")
    pattern = f"rf_vwap_stop{stop_str}_*.pkl"
    
    model_files = list(models_dir.glob(pattern))
    if not model_files:
        raise FileNotFoundError(f"No model found for stop {stop_atr} ATR")
    
    # Use the most recent
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading model: {latest_model.name}")
    
    model_data = load_model(str(latest_model))
    return model_data['model'], model_data['metadata']


def create_indicator_calculator(stop_atr: float):
    """Create indicator calculator matching master pipeline"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.master_pipeline import calculate_core_indicators, get_feature_columns
    
    class IndicatorCalculator:
        def __init__(self, stop_atr):
            self.stop_atr = stop_atr
            self.features = None
            
        def calculate(self, bars_df: pd.DataFrame) -> dict:
            """Calculate indicators for current bar"""
            # Calculate all indicators
            df_with_indicators = calculate_core_indicators(bars_df.copy(), verbose=False)
            
            # Get feature columns (first time only)
            if self.features is None:
                self.features = get_feature_columns(df_with_indicators)
            
            # Return indicators for the LAST bar (current)
            if len(df_with_indicators) == 0:
                return {}
            
            last_row = df_with_indicators.iloc[-1]
            indicators = {}
            
            # Add all features
            for col in self.features:
                if col in last_row.index:
                    indicators[col] = last_row[col]
            
            # Add key indicators for strategy
            for col in ['vwap', 'atr', 'vwap_width_atr', 'is_long_setup', 'close']:
                if col in last_row.index:
                    indicators[col] = last_row[col]
            
            return indicators
    
    return IndicatorCalculator(stop_atr)


def create_strategy(model, metadata, rf_threshold=0.5):
    """Create strategy function"""
    stop_atr = metadata['stop_atr']
    features = metadata['features']
    
    def strategy(bar, indicators, model_obj, portfolio, order_manager):
        """VWAP reversion strategy with RF filtering"""
        symbol = 'TSLA'
        
        # Skip if no indicators yet
        if not indicators or 'vwap_width_atr' not in indicators:
            return
        
        # Check if we have a position
        has_position = portfolio.positions.get(symbol) is not None
        
        if has_position:
            # Manage existing position
            position = portfolio.positions[symbol]
            current_price = bar['close']
            atr = indicators.get('atr', 10.0)
            
            # Calculate stop and target
            if position.quantity > 0:  # Long position
                stop_price = position.avg_entry_price - (stop_atr * atr)
                target_price = position.avg_entry_price + (stop_atr * atr * metadata['rr'])
                
                # Check for stop or target hit
                if current_price <= stop_price or current_price >= target_price:
                    # Submit market sell order
                    order = Order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        quantity=position.quantity,
                        timestamp=bar['datetime'],
                    )
                    order_manager.submit_order(order)
        else:
            # Look for entry signal
            # Only trade long setups (price < VWAP)
            if not indicators.get('is_long_setup', False):
                return
            
            # Check distance from VWAP
            vwap_dist = indicators.get('vwap_width_atr', 0)
            if vwap_dist < 0.3 or vwap_dist > 3.0:  # Too close or too far
                return
            
            # Get model prediction
            try:
                # Prepare features
                X = np.array([[indicators.get(f, 0) for f in features]])
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Get probability
                proba = model_obj.predict_proba(X)[0, 1]
                
                # Check threshold
                if proba >= rf_threshold:
                    # Submit market buy order (100 shares)
                    order = Order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        quantity=100,
                        timestamp=bar['datetime'],
                    )
                    order_manager.submit_order(order)
            except Exception as e:
                # Skip on error
                pass
    
    return strategy


def run_comparison(stop_atrs, data_file, initial_capital=100000, rf_threshold=0.5):
    """Run streaming simulation for multiple stop widths"""
    
    print("="*80)
    print("STREAMING SIMULATOR COMPARISON - MULTIPLE STOP WIDTHS")
    print("="*80)
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"RF Threshold: {rf_threshold}")
    print(f"Stop Widths: {stop_atrs}")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from {data_file}...")
    data = pd.read_csv(data_file)
    
    # Filter to 2024 only (test period)
    data['datetime'] = pd.to_datetime(data['time'], utc=True)
    data['year'] = data['datetime'].dt.year
    data_2024 = data[data['year'] == 2024].copy()
    print(f"Filtered to 2024: {len(data_2024):,} bars")
    
    results = []
    
    for stop_atr in stop_atrs:
        print(f"\n{'='*80}")
        print(f"TESTING STOP WIDTH: {stop_atr} ATR")
        print(f"{'='*80}")
        
        try:
            # Load model
            model, metadata = load_trained_model(stop_atr)
            
            # Create indicator calculator
            indicator_calc = create_indicator_calculator(stop_atr)
            
            # Create strategy
            strategy = create_strategy(model, metadata, rf_threshold)
            
            # Create simulator
            sim = StreamingSimulator(
                initial_capital=initial_capital,
                bar_interval_minutes=5,
                lookback_bars=200,
                verbose=True,
                log_every_n_bars=0,  # Disable per-bar logging for speed
            )
            
            # Run simulation
            sim_results = sim.run(
                data=data_2024,
                strategy_func=strategy,
                indicator_calculator=indicator_calc,
                model=model,
                symbol='TSLA',
            )
            
            stats = sim_results['statistics']
            
            # Store results
            results.append({
                'stop_atr': stop_atr,
                'total_trades': stats['total_trades'],
                'win_rate': stats['win_rate'],
                'profit_factor': stats['profit_factor'],
                'sharpe_ratio': stats['sharpe_ratio'],
                'max_drawdown_pct': stats['max_drawdown_pct'],
                'total_return_pct': stats['total_return_pct'],
                'final_equity': stats['final_equity'],
                'net_pnl': stats['final_equity'] - initial_capital,
                'total_fees': stats['total_fees'],
                'avg_win': stats['avg_win'],
                'avg_loss': stats['avg_loss'],
            })
            
        except Exception as e:
            print(f"❌ Error with stop {stop_atr}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create comparison DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        # Sort by net P&L
        results_df = results_df.sort_values('net_pnl', ascending=False)
        
        # Display table
        print(results_df.to_string(index=False))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"data/streaming_comparison_{timestamp}.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        
        # Identify best strategy
        best = results_df.iloc[0]
        print(f"\n{'='*80}")
        print("BEST PERFORMING STRATEGY")
        print(f"{'='*80}")
        print(f"Stop Width: {best['stop_atr']} ATR")
        print(f"Net P&L: ${best['net_pnl']:,.2f}")
        print(f"Win Rate: {best['win_rate']*100:.1f}%")
        print(f"Profit Factor: {best['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        print(f"Total Trades: {int(best['total_trades']):,}")
        
        return results_df
    else:
        print("\n❌ No results to display")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare streaming simulator across stop widths")
    parser.add_argument(
        "--stops",
        type=str,
        default="0.5,0.75,1.0,1.25,1.5",
        help="Comma-separated list of stop ATR values (default: 0.5,0.75,1.0,1.25,1.5)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/tsla_5min_10years.csv",
        help="Path to data file"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="RF probability threshold (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Parse stop widths
    stop_atrs = [float(x.strip()) for x in args.stops.split(',')]
    
    # Run comparison
    run_comparison(
        stop_atrs=stop_atrs,
        data_file=args.data,
        initial_capital=args.capital,
        rf_threshold=args.threshold,
    )
