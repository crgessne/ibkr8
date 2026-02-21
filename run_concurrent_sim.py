"""
Concurrent Streaming Simulation - Takes ALL signals simultaneously

This script runs a bar-by-bar simulation that allows multiple concurrent positions
to test if taking all signals (rather than blocking overlapping signals) can achieve
the master pipeline's performance.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

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
    print("="*80)
    print("CONCURRENT STREAMING SIMULATION")
    print("="*80)
    
    # Configuration
    YEAR = 2024
    STOP_ATR = 1.5
    RF_THRESHOLD = 0.5
    POSITION_SIZE = 100
    LOOKBACK = 200
    INITIAL_CAPITAL = 100000
    
    print(f"\nConfiguration:")
    print(f"  Year: {YEAR}")
    print(f"  Stop ATR: {STOP_ATR}")
    print(f"  RF Threshold: {RF_THRESHOLD}")
    print(f"  Position Size: {POSITION_SIZE} shares")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"  Mode: CONCURRENT (all signals taken)")
    
    # Load data
    print(f"\nLoading data...")
    df = pd.read_csv("data/tsla_5min_10years.csv")
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    
    df["year"] = df["datetime"].dt.year
    df_test = df[df["year"] == YEAR].reset_index(drop=True)
    print(f"  Loaded {len(df_test):,} bars for {YEAR}")
    
    # Load model
    print(f"\nLoading model...")
    model, metadata = load_model_for_stop(stop_atr=STOP_ATR, models_dir="models", latest=True)
    feature_cols = metadata["features"]
    rr = float(metadata.get("rr", 1.2))
    print(f"  Model loaded with {len(feature_cols)} features")
    print(f"  Risk/Reward Ratio: {rr}")
    
    # Create simulator
    print(f"\nInitializing simulator...")
    sim = StreamingSimulator(
        initial_capital=INITIAL_CAPITAL,
        bar_interval_minutes=5,
        lookback_bars=LOOKBACK,
        execution_quality=ExecutionQuality.GOOD,
        broker_fees=IBKRFees(),
        max_position_size=1000,
        verbose=True,
    )
    
    # Strategy state
    symbol = "TSLA"
    position_counter = [1]  # Track unique position IDs
    stats = {
        'total_signals': 0,
        'entries_taken': 0,
        'max_concurrent': 0,
    }
    
    def concurrent_strategy(bar, indicators, model, portfolio, order_manager):
        """
        Concurrent strategy that takes ALL qualifying signals.
        Each position gets a unique symbol (TSLA_1, TSLA_2, etc.)
        """
        # Check for valid indicators
        if not indicators or "atr" not in indicators:
            return
        
        # CHECK FOR NEW ENTRIES (always allow in concurrent mode)
        if indicators.get("is_long_setup", False):
            # Get features and check model prediction
            feature_vector = [indicators.get(c) for c in feature_cols]
            
            if not any(pd.isna(x) for x in feature_vector):
                prob = model.predict_proba([feature_vector])[0, 1]
                
                if prob >= RF_THRESHOLD:
                    stats['total_signals'] += 1
                    
                    # Create unique symbol for this position
                    position_symbol = f"{symbol}_{position_counter[0]}"
                    position_counter[0] += 1
                    
                    # Enter the position
                    order = order_manager.create_order(
                        symbol=position_symbol,
                        side=OrderSide.BUY,
                        quantity=POSITION_SIZE,
                        order_type=OrderType.MARKET,
                    )
                    order_manager.submit_order(order, bar["datetime"])
                    
                    stats['entries_taken'] += 1
                    current_positions = len(portfolio.positions)
                    stats['max_concurrent'] = max(stats['max_concurrent'], current_positions)
                    
                    # Log every 100 entries
                    if stats['entries_taken'] % 100 == 0:
                        print(f"  [{stats['entries_taken']} entries] Max concurrent: {stats['max_concurrent']}")
        
        # CHECK FOR EXITS on all open positions
        atr = float(indicators.get("atr", 0.0) or 0.0)
        if atr == 0.0:
            return
        
        for pos_symbol in list(portfolio.positions.keys()):
            if not pos_symbol.startswith(symbol):
                continue
            
            pos = portfolio.positions[pos_symbol]
            entry_price = pos.avg_entry_price
            
            # Calculate stop and target
            stop_price = entry_price - (STOP_ATR * atr)
            target_price = entry_price + (rr * STOP_ATR * atr)
            
            # Check if stop or target hit (using intrabar high/low)
            stop_hit = bar["low"] <= stop_price
            target_hit = bar["high"] >= target_price
            
            if stop_hit or target_hit:
                order = order_manager.create_order(
                    symbol=pos_symbol,
                    side=OrderSide.SELL,
                    quantity=abs(pos.quantity),
                    order_type=OrderType.MARKET,
                )
                order_manager.submit_order(order, bar["datetime"])
    
    # Run simulation
    print(f"\nRunning simulation...")
    print(f"="*80)
    
    indicator_calc = StreamingIndicatorsAligned(verbose=False)
    
    results = sim.run(
        data=df_test,
        strategy_func=concurrent_strategy,
        indicator_calculator=indicator_calc,
        model=model,
        symbol=symbol,
    )
    
    # Print results
    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nStrategy Statistics:")
    print(f"  Total qualifying signals: {stats['total_signals']:,}")
    print(f"  Entries taken: {stats['entries_taken']:,}")
    print(f"  Max concurrent positions: {stats['max_concurrent']}")
    
    trade_history = results["trade_history"]
    print(f"\nTrade Results:")
    print(f"  Total trades closed: {len(trade_history):,}")
    print(f"  Total P&L: ${trade_history['pnl'].sum():,.2f}")
    print(f"  Win rate: {(trade_history['pnl'] > 0).mean():.1%}")
    print(f"  Avg win: ${trade_history[trade_history['pnl'] > 0]['pnl'].mean():.2f}")
    print(f"  Avg loss: ${trade_history[trade_history['pnl'] < 0]['pnl'].mean():.2f}")
    
    # Save results
    print(f"\nSaving results...")
    trade_history.to_csv("data/concurrent_sim_results.csv", index=False)
    results["equity_curve"].to_csv("data/concurrent_sim_equity.csv", index=False)
    print(f"  Saved to data/concurrent_sim_results.csv")
    print(f"  Saved to data/concurrent_sim_equity.csv")
    
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
