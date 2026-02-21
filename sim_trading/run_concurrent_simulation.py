"""
Concurrent Trading Simulation - BRAND NEW IMPLEMENTATION

This script runs a streaming bar-by-bar simulation with support for concurrent positions.
Each qualifying signal opens a new position with a unique symbol identifier.

Usage:
    python sim_trading/run_concurrent_simulation.py --concurrent
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from datetime import datetime

from sim_trading import (
    StreamingSimulator,
    ExecutionQuality,
    OrderSide,
    OrderType,
    IBKRFees,
)
from sim_trading.streaming_indicators_aligned import StreamingIndicatorsAligned
from src.model_selector import load_model_for_stop


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run concurrent trading simulation")
    parser.add_argument("--year", type=int, default=2024, help="Year to test")
    parser.add_argument("--stop-atr", type=float, default=1.5, help="Stop loss in ATR units")
    parser.add_argument("--rf-threshold", type=float, default=0.5, help="RF probability threshold")
    parser.add_argument("--position-size", type=int, default=100, help="Shares per trade")
    parser.add_argument("--concurrent", action="store_true", help="Enable concurrent positions")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--lookback", type=int, default=200, help="Lookback bars")
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"CONCURRENT SIMULATION STARTING")
    print(f"{'='*80}")
    print(f"Mode: {'CONCURRENT' if args.concurrent else 'SINGLE POSITION'}")
    print(f"Year: {args.year}")
    print(f"Stop ATR: {args.stop_atr}")
    print(f"RF Threshold: {args.rf_threshold}")
    print(f"Position Size: {args.position_size}")
    print(f"Initial Capital: ${args.capital:,.0f}")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv("data/tsla_5min_10years.csv")
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    
    df["year"] = df["datetime"].dt.year
    df_year = df[df["year"] == args.year].reset_index(drop=True)
    print(f"Loaded {len(df_year):,} bars for year {args.year}")
    
    # Load model
    print(f"Loading model for stop_atr={args.stop_atr}...")
    model, metadata = load_model_for_stop(stop_atr=args.stop_atr, models_dir="models", latest=True)
    feature_cols = metadata["features"]
    rr = float(metadata.get("rr", 1.2))
    print(f"Model loaded. RR ratio: {rr}")
    print(f"Features: {len(feature_cols)}")
    
    # Create simulator
    sim = StreamingSimulator(
        initial_capital=args.capital,
        bar_interval_minutes=5,
        lookback_bars=args.lookback,
        execution_quality=ExecutionQuality.GOOD,
        broker_fees=IBKRFees(),
        max_position_size=1000,
        verbose=True,
    )
    
    # Create indicator calculator
    indicator_calc = StreamingIndicatorsAligned(verbose=False)
    
    # Strategy state
    position_counter = [0]  # Mutable counter for position IDs
    stats = {
        "bars_processed": 0,
        "signals_generated": 0,
        "trades_opened": 0,
        "max_concurrent": 0,
    }
    
    # Define strategy function
    def strategy_function(bar, indicators, model, portfolio, order_manager):
        stats["bars_processed"] += 1
        
        # Check for required indicators
        if not indicators or "atr" not in indicators:
            return
        
        # ENTRY LOGIC
        if args.concurrent or len(portfolio.positions) == 0:
            # Check if setup exists
            if not indicators.get("is_long_setup", False):
                return
            
            # Get feature vector
            feature_vector = [indicators.get(col) for col in feature_cols]
            if any(pd.isna(x) for x in feature_vector):
                return
            
            # Check RF probability
            prob = model.predict_proba([feature_vector])[0, 1]
            if prob < args.rf_threshold:
                return
            
            # Generate signal
            stats["signals_generated"] += 1
            
            # Create unique symbol for concurrent mode
            if args.concurrent:
                position_counter[0] += 1
                symbol = f"TSLA_{position_counter[0]}"
            else:
                symbol = "TSLA"
            
            # Submit buy order
            order = order_manager.create_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=args.position_size,
                order_type=OrderType.MARKET,
            )
            order_manager.submit_order(order, bar["datetime"])
            stats["trades_opened"] += 1
            
            # Track max concurrent
            current_positions = len(portfolio.positions) + 1
            if current_positions > stats["max_concurrent"]:
                stats["max_concurrent"] = current_positions
            
            # Print progress
            if stats["trades_opened"] % 50 == 0:
                print(f"Opened {stats['trades_opened']} trades, max concurrent: {stats['max_concurrent']}")
            
            return
        
        # EXIT LOGIC - Check all open positions
        positions_to_check = list(portfolio.positions.keys())
        
        for symbol in positions_to_check:
            pos = portfolio.positions[symbol]
            entry_price = pos.avg_entry_price
            
            # Get ATR
            atr = float(indicators.get("atr", 0.0) or 0.0)
            if atr == 0.0:
                continue
            
            # Calculate stop and target
            stop_price = entry_price - (args.stop_atr * atr)
            target_price = entry_price + (rr * args.stop_atr * atr)
            
            # Check if stop or target hit (using bar high/low)
            stop_hit = bar["low"] <= stop_price
            target_hit = bar["high"] >= target_price
            
            if stop_hit or target_hit:
                # Submit sell order
                order = order_manager.create_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=abs(pos.quantity),
                    order_type=OrderType.MARKET,
                )
                order_manager.submit_order(order, bar["datetime"])
    
    # Run simulation
    print("\nRunning simulation...")
    print(f"{'='*80}\n")
    
    results = sim.run(
        data=df_year,
        strategy_func=strategy_function,
        indicator_calculator=indicator_calc,
        model=model,
        symbol="TSLA",
    )
    
    # Print final statistics
    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Bars processed: {stats['bars_processed']:,}")
    print(f"Signals generated: {stats['signals_generated']:,}")
    print(f"Trades opened: {stats['trades_opened']:,}")
    print(f"Max concurrent positions: {stats['max_concurrent']}")
    print(f"{'='*80}\n")
    
    # Analyze results
    trade_history = results["trade_history"]
    if len(trade_history) > 0:
        total_pnl = trade_history["pnl"].sum()
        win_rate = (trade_history["pnl"] > 0).mean()
        avg_win = trade_history[trade_history["pnl"] > 0]["pnl"].mean() if (trade_history["pnl"] > 0).any() else 0
        avg_loss = trade_history[trade_history["pnl"] < 0]["pnl"].mean() if (trade_history["pnl"] < 0).any() else 0
        
        print("PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Total trades: {len(trade_history):,}")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Win rate: {win_rate:.1%}")
        print(f"Avg win: ${avg_win:.2f}")
        print(f"Avg loss: ${avg_loss:.2f}")
        print(f"{'='*80}\n")
        
        # Save results
        output_suffix = "_concurrent" if args.concurrent else "_single"
        trade_file = f"data/streaming_results{output_suffix}.csv"
        equity_file = f"data/streaming_equity{output_suffix}.csv"
        
        trade_history.to_csv(trade_file, index=False)
        results["equity_curve"].to_csv(equity_file, index=False)
        
        print(f"Results saved:")
        print(f"  Trade history: {trade_file}")
        print(f"  Equity curve: {equity_file}")
    else:
        print("WARNING: No trades were executed!")
    
    print("\nDone.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
