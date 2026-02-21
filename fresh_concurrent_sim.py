"""
Concurrent Streaming Simulator - Built from Scratch

Takes all qualifying signals simultaneously (no position blocking) to compare with master pipeline.
"""
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sim_trading import StreamingSimulator, ExecutionQuality, OrderSide, OrderType, IBKRFees
from sim_trading.streaming_indicators_aligned import StreamingIndicatorsAligned
from src.model_selector import load_model_for_stop


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Concurrent streaming simulation")
    parser.add_argument("--year", type=int, default=2024, help="Year to test")
    parser.add_argument("--stop-atr", type=float, default=1.5, help="Stop loss in ATR multiples")
    parser.add_argument("--rf-threshold", type=float, default=0.5, help="RF probability threshold")
    parser.add_argument("--position-size", type=int, default=100, help="Position size in shares")
    parser.add_argument("--concurrent", action="store_true", help="Enable concurrent positions")
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"CONCURRENT MODE: {args.concurrent}")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv("data/tsla_5min_10years.csv")
    df["datetime"] = pd.to_datetime(df["time"], utc=True)
    df["year"] = df["datetime"].dt.year
    df_test = df[df["year"] == args.year].reset_index(drop=True)
    print(f"Loaded {len(df_test)} bars for {args.year}")
    
    # Load model
    print(f"Loading model for stop_atr={args.stop_atr}...")
    model, metadata = load_model_for_stop(stop_atr=args.stop_atr, models_dir="models", latest=True)
    feature_cols = metadata["features"]
    rr = float(metadata.get("rr", 1.2))
    print(f"Model loaded. RR={rr}, Features={len(feature_cols)}")
    
    # Initialize simulator
    sim = StreamingSimulator(
        initial_capital=100000,
        bar_interval_minutes=5,
        lookback_bars=200,
        execution_quality=ExecutionQuality.GOOD,
        broker_fees=IBKRFees(),
        max_position_size=1000,
        verbose=True
    )
    
    # Initialize indicators
    indicator_calc = StreamingIndicatorsAligned(verbose=False)
    
    # Strategy state
    position_counter = [1]  # Mutable counter for unique position IDs
    entry_count = [0]
    max_concurrent = [0]
    
    # Strategy function
    def strategy_func(bar, indicators, model, portfolio, order_manager):
        """
        Strategy that takes entries when:
        1. Price < VWAP (is_long_setup)
        2. RF model predicts >= threshold
        
        In concurrent mode: Takes every signal regardless of existing positions
        In single mode: Only takes signal if no position exists
        """
        # Skip if indicators not ready
        if not indicators or "atr" not in indicators:
            return
        
        # ENTRY LOGIC
        if args.concurrent:
            can_enter = True  # Always allow entry in concurrent mode
        else:
            can_enter = "TSLA" not in portfolio.positions  # Block if position exists
        
        if can_enter:
            # Check setup condition
            if not indicators.get("is_long_setup", False):
                return
            
            # Get features
            feature_vector = [indicators.get(c) for c in feature_cols]
            if any(pd.isna(x) for x in feature_vector):
                return
            
            # Check RF probability
            prob = model.predict_proba([feature_vector])[0, 1]
            if prob < args.rf_threshold:
                return
            
            # ENTER TRADE
            entry_count[0] += 1
            
            if args.concurrent:
                # Unique symbol for each concurrent position
                symbol = f"TSLA_{position_counter[0]}"
                position_counter[0] += 1
                
                # Track max concurrent positions
                num_positions = len(portfolio.positions)
                if num_positions > max_concurrent[0]:
                    max_concurrent[0] = num_positions
                    
                if entry_count[0] % 100 == 1:
                    print(f"Entry #{entry_count[0]}: {num_positions} concurrent positions")
            else:
                symbol = "TSLA"
            
            # Submit buy order
            order = order_manager.create_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=args.position_size,
                order_type=OrderType.MARKET
            )
            order_manager.submit_order(order, bar["datetime"])
            return
        
        # EXIT LOGIC
        atr = float(indicators.get("atr", 0.0) or 0.0)
        if atr == 0.0:
            return
        
        # Check all positions for exits
        for pos_symbol in list(portfolio.positions.keys()):
            # In single mode, only check TSLA; in concurrent mode, check all TSLA_* positions
            if not pos_symbol.startswith("TSLA"):
                continue
            
            pos = portfolio.positions[pos_symbol]
            entry_price = pos.avg_entry_price
            
            # Calculate stop and target
            stop_price = entry_price - (args.stop_atr * atr)
            target_price = entry_price + (rr * args.stop_atr * atr)
            
            # Check if stop or target hit (using intrabar high/low)
            stop_hit = bar["low"] <= stop_price
            target_hit = bar["high"] >= target_price
            
            if stop_hit or target_hit:
                # EXIT TRADE
                order = order_manager.create_order(
                    symbol=pos_symbol,
                    side=OrderSide.SELL,
                    quantity=abs(pos.quantity),
                    order_type=OrderType.MARKET
                )
                order_manager.submit_order(order, bar["datetime"])
    
    # Run simulation
    print(f"\nRunning simulation...")
    results = sim.run(
        data=df_test,
        strategy_func=strategy_func,
        indicator_calculator=indicator_calc,
        model=model,
        symbol="TSLA"
    )
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"  Total signals taken: {entry_count[0]}")
    if args.concurrent:
        print(f"  Max concurrent positions: {max_concurrent[0]}")
    print(f"  Completed trades: {len(results['trade_history'])}")
    
    if len(results['trade_history']) > 0:
        total_pnl = results['trade_history']['pnl'].sum()
        win_rate = (results['trade_history']['pnl'] > 0).mean()
        print(f"  Total P&L: ${total_pnl:,.2f}")
        print(f"  Win rate: {win_rate:.1%}")
    print(f"{'='*80}\n")
    
    # Save results
    output_suffix = "concurrent" if args.concurrent else "single"
    results["trade_history"].to_csv(f"data/sim_results_{output_suffix}.csv", index=False)
    results["equity_curve"].to_csv(f"data/sim_equity_{output_suffix}.csv", index=False)
    print(f"Results saved to data/sim_results_{output_suffix}.csv")


if __name__ == "__main__":
    main()
