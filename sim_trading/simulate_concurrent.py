"""
Concurrent Streaming Simulation - Takes All Signals Simultaneously

This script runs a bar-by-bar simulation that allows multiple concurrent positions,
unlike the standard streaming simulator which blocks overlapping signals.

Usage:
    python sim_trading/simulate_concurrent.py --year 2024 --stop-atr 1.5
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


def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare data with datetime column."""
    df = pd.read_csv(filepath)
    if "time" in df.columns and "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def create_concurrent_strategy(
    feature_cols: list,
    rf_threshold: float,
    position_size: int,
    stop_atr: float,
    rr: float,
    model,
    verbose: bool = True
):
    """
    Create a strategy that takes all qualifying signals concurrently.
    
    Key differences from single-position mode:
    - Always checks for entry (no position blocking)
    - Uses unique symbol names for each position (TSLA_1, TSLA_2, etc.)
    - Exits each position independently based on its own stop/target
    """
    symbol = "TSLA"
    
    # Statistics tracking
    stats = {
        'total_bars': 0,
        'signals_found': 0,
        'positions_opened': 0,
        'max_concurrent': 0,
        'current_positions': 0,
    }
    
    # Position ID counter
    next_id = [1]
    
    def strategy(bar, indicators, model_obj, portfolio, order_manager):
        """Strategy function called on each bar."""
        stats['total_bars'] += 1
        
        # Skip if indicators not ready
        if not indicators or "atr" not in indicators:
            return
        
        # =================================================================
        # ENTRY LOGIC - Check for new signals (always, no blocking)
        # =================================================================
        
        # Check if we have a valid long setup
        is_setup = indicators.get("is_long_setup", False)
        if not is_setup:
            # Price is not below VWAP, no setup
            pass
        else:
            # We have a setup, check ML probability
            feature_vector = [indicators.get(c) for c in feature_cols]
            
            # Skip if any features are missing
            if any(pd.isna(x) for x in feature_vector):
                pass
            else:
                # Calculate probability
                prob = model_obj.predict_proba([feature_vector])[0, 1]
                
                if prob >= rf_threshold:
                    # SIGNAL FOUND! Take it immediately
                    stats['signals_found'] += 1
                    
                    # Create unique symbol name for this position
                    position_symbol = f"{symbol}_{next_id[0]}"
                    next_id[0] += 1
                    stats['positions_opened'] += 1
                    stats['current_positions'] += 1
                    
                    # Update max concurrent tracker
                    if stats['current_positions'] > stats['max_concurrent']:
                        stats['max_concurrent'] = stats['current_positions']
                    
                    # Log every 500 signals
                    if verbose and stats['signals_found'] % 500 == 0:
                        print(f"\n[SIGNAL #{stats['signals_found']}] Bar {stats['total_bars']}", flush=True)
                        print(f"  Current concurrent positions: {stats['current_positions']}", flush=True)
                        print(f"  Max concurrent so far: {stats['max_concurrent']}", flush=True)
                    
                    # Submit market order
                    order = order_manager.create_order(
                        symbol=position_symbol,
                        side=OrderSide.BUY,
                        quantity=position_size,
                        order_type=OrderType.MARKET,
                    )
                    order_manager.submit_order(order, bar["datetime"])
        
        # =================================================================
        # EXIT LOGIC - Check all open positions for stop/target hits
        # =================================================================
        
        atr = float(indicators.get("atr", 0.0) or 0.0)
        if atr == 0.0:
            return
        
        # Check each position independently
        for pos_symbol in list(portfolio.positions.keys()):
            # Only check our TSLA positions
            if not pos_symbol.startswith(symbol):
                continue
            
            pos = portfolio.positions[pos_symbol]
            entry_price = pos.avg_entry_price
            
            # Calculate stop and target based on entry
            stop_price = entry_price - (stop_atr * atr)
            target_price = entry_price + (rr * stop_atr * atr)
            
            # Check if stop or target hit using intrabar high/low
            # This is critical - don't just check close price!
            stop_hit = bar["low"] <= stop_price
            target_hit = bar["high"] >= target_price
            
            if stop_hit or target_hit:
                # Exit this position
                order = order_manager.create_order(
                    symbol=pos_symbol,
                    side=OrderSide.SELL,
                    quantity=abs(pos.quantity),
                    order_type=OrderType.MARKET,
                )
                order_manager.submit_order(order, bar["datetime"])
                stats['current_positions'] -= 1
    
    # Attach stats getter
    def get_stats():
        return stats
    
    strategy.get_stats = get_stats
    return strategy


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("CONCURRENT STREAMING SIMULATION")
    print("="*80 + "\n")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Concurrent streaming simulation")
    parser.add_argument("--year", type=int, default=2024, help="Year to simulate")
    parser.add_argument("--stop-atr", type=float, default=1.5, help="Stop loss in ATR multiples")
    parser.add_argument("--rf-threshold", type=float, default=0.5, help="RF probability threshold")
    parser.add_argument("--position-size", type=int, default=100, help="Shares per position")
    parser.add_argument("--lookback", type=int, default=200, help="Indicator lookback bars")
    parser.add_argument("--data", type=str, default="data/tsla_5min_10years.csv")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--initial-capital", type=float, default=1000000, help="Starting capital")
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Year: {args.year}")
    print(f"  Stop ATR: {args.stop_atr}x")
    print(f"  RF Threshold: {args.rf_threshold}")
    print(f"  Position Size: {args.position_size} shares")
    print(f"  Initial Capital: ${args.initial_capital:,.0f}")
    print(f"  Mode: CONCURRENT (all signals taken)")
    print()
    
    # Load data
    print("Loading data...", flush=True)
    df = load_data(args.data)
    df["year"] = df["datetime"].dt.year
    df_test = df[df["year"] == args.year].reset_index(drop=True)
    
    if df_test.empty:
        raise SystemExit(f"ERROR: No data found for year {args.year}")
    
    print(f"  Data loaded: {len(df_test):,} bars for {args.year}")
    print()
    
    # Load model
    print("Loading model...", flush=True)
    model, metadata = load_model_for_stop(
        stop_atr=args.stop_atr,
        models_dir=args.models_dir,
        latest=True
    )
    feature_cols = metadata["features"]
    rr = float(metadata.get("rr", 1.2))
    print(f"  Model loaded: {len(feature_cols)} features, RR={rr}")
    print()
    
    # Create simulator
    print("Initializing simulator...", flush=True)
    sim = StreamingSimulator(
        initial_capital=args.initial_capital,
        bar_interval_minutes=5,
        lookback_bars=args.lookback,
        execution_quality=ExecutionQuality.GOOD,
        broker_fees=IBKRFees(),
        max_position_size=10000,  # High limit for concurrent mode
        verbose=True,
    )
    print()
    
    # Create strategy
    indicator_calc = StreamingIndicatorsAligned(verbose=False)
    strategy = create_concurrent_strategy(
        feature_cols=feature_cols,
        rf_threshold=args.rf_threshold,
        position_size=args.position_size,
        stop_atr=args.stop_atr,
        rr=rr,
        model=model,
        verbose=True,
    )
    
    # Run simulation
    print("="*80)
    print("RUNNING SIMULATION (this may take a few minutes)...")
    print("="*80 + "\n")
    
    start_time = datetime.now()
    
    results = sim.run(
        data=df_test,
        strategy_func=strategy,
        indicator_calculator=indicator_calc,
        model=model,
        symbol="TSLA",
    )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    # Print results
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80 + "\n")
    
    stats = strategy.get_stats()
    trades = results["trade_history"]
    
    print(f"Execution Time: {elapsed:.1f} seconds")
    print(f"\nStrategy Statistics:")
    print(f"  Total bars processed: {stats['total_bars']:,}")
    print(f"  Signals found: {stats['signals_found']:,}")
    print(f"  Positions opened: {stats['positions_opened']:,}")
    print(f"  Max concurrent positions: {stats['max_concurrent']}")
    print()
    
    print(f"Trading Results:")
    print(f"  Total trades: {len(trades):,}")
    
    if len(trades) > 0:
        total_pnl = trades["pnl"].sum()
        wins = (trades["pnl"] > 0).sum()
        losses = (trades["pnl"] < 0).sum()
        win_rate = wins / len(trades) * 100
        avg_win = trades[trades["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
        avg_loss = trades[trades["pnl"] < 0]["pnl"].mean() if losses > 0 else 0
        
        print(f"  Total P&L: ${total_pnl:,.2f}")
        print(f"  Win rate: {win_rate:.1f}% ({wins} wins, {losses} losses)")
        print(f"  Avg win: ${avg_win:,.2f}")
        print(f"  Avg loss: ${avg_loss:,.2f}")
        
        # Check unique symbols to verify concurrent mode worked
        unique_symbols = trades["symbol"].nunique()
        print(f"\n  Unique symbols traded: {unique_symbols}")
        if unique_symbols == 1:
            print(f"    WARNING: Only 1 symbol found - concurrent mode may not have worked!")
        else:
            print(f"    ✓ Multiple symbols confirmed - concurrent mode working")
    
    # Save results
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    trades_file = out_dir / "concurrent_sim_results.csv"
    equity_file = out_dir / "concurrent_sim_equity.csv"
    
    results["trade_history"].to_csv(trades_file, index=False)
    results["equity_curve"].to_csv(equity_file, index=False)
    
    print(f"\nResults saved:")
    print(f"  Trades: {trades_file}")
    print(f"  Equity: {equity_file}")
    print()
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print("="*80 + "\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
