"""
Calculate realistic P&L with capital constraints.
Accounts for:
- Maximum capital available ($1M)
- Overlapping positions (trades close over time, not instantly)
- Skipped trades when capital limit reached
- Realistic position durations
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Capital constraints
MAX_CAPITAL = 500_000  # $500K available
SHARES_PER_TRADE = 100
CURRENT_PRICE = 400  # TSLA @ $400/share
CAPITAL_PER_TRADE = SHARES_PER_TRADE * CURRENT_PRICE  # $40,000

# Costs
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_PER_SHARE = 0.01
COST_PER_TRADE = 2 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * SHARES_PER_TRADE  # $3

# Strategy parameters (0.50 ATR)
STOP_ATR = 0.50
WIN_RATE = 0.438
TRADES_PER_YEAR = 8245
TRADES_PER_DAY = 33
RR_RATIO = 3.05  # From results
EV_PER_TRADE = 0.763

# Average trade duration (in 5-min bars)
# Estimate: tight stop (0.5 ATR) means quick exits
# Assume avg trade lasts 4 bars (20 minutes)
AVG_BARS_TO_STOP = 4  # Hit stop loss
AVG_BARS_TO_TARGET = 12  # Hit target (takes longer)
AVG_TRADE_DURATION = WIN_RATE * AVG_BARS_TO_TARGET + (1 - WIN_RATE) * AVG_BARS_TO_STOP

# ============================================================================
# CALCULATIONS
# ============================================================================

def calculate_unconstrained_pnl():
    """Original calculation (assumes unlimited capital)."""
    avg_atr = CURRENT_PRICE * 0.01  # 1% of price = $4.00
    
    risk_dollars = STOP_ATR * avg_atr * SHARES_PER_TRADE
    reward_dollars = risk_dollars * RR_RATIO
    
    avg_win = reward_dollars
    avg_loss = -risk_dollars
    
    gross_pnl_per_trade = (WIN_RATE * avg_win) + ((1 - WIN_RATE) * avg_loss)
    net_pnl_per_trade = gross_pnl_per_trade - COST_PER_TRADE
    
    total_net_pnl = net_pnl_per_trade * TRADES_PER_YEAR
    
    return {
        'scenario': 'Unconstrained (Original)',
        'trades_per_year': TRADES_PER_YEAR,
        'trades_skipped': 0,
        'skip_rate': 0.0,
        'net_pnl_per_trade': net_pnl_per_trade,
        'total_net_pnl': total_net_pnl,
        'avg_positions_open': 0,  # Not tracked
        'max_positions_open': 0,  # Not tracked
        'capital_utilized': 0  # Not tracked
    }


def calculate_constrained_pnl():
    """
    Realistic calculation with capital constraints.
    Simulates position management over time.
    """
    
    # Calculate P&L per trade
    avg_atr = CURRENT_PRICE * 0.01
    risk_dollars = STOP_ATR * avg_atr * SHARES_PER_TRADE
    reward_dollars = risk_dollars * RR_RATIO
    
    avg_win = reward_dollars
    avg_loss = -risk_dollars
    
    gross_pnl_per_trade = (WIN_RATE * avg_win) + ((1 - WIN_RATE) * avg_loss)
    net_pnl_per_trade = gross_pnl_per_trade - COST_PER_TRADE
    
    # Simulate trading year
    bars_per_year = 19_093
    trades_per_bar = TRADES_PER_YEAR / bars_per_year  # ~0.43 trades/bar
    
    max_concurrent_positions = int(MAX_CAPITAL / CAPITAL_PER_TRADE)
    
    # Simple simulation
    positions_open = []
    trades_taken = 0
    trades_skipped = 0
    total_pnl = 0
    position_counts = []
    
    for bar in range(bars_per_year):
        # Close positions that have reached their duration
        positions_open = [p - 1 for p in positions_open if p > 1]
        
        position_counts.append(len(positions_open))
        
        # Try to take new trades
        # Use Poisson distribution for trade arrivals
        n_signals_this_bar = np.random.poisson(trades_per_bar)
        
        for _ in range(n_signals_this_bar):
            if len(positions_open) < max_concurrent_positions:
                # Take the trade
                is_win = np.random.random() < WIN_RATE
                duration = AVG_BARS_TO_TARGET if is_win else AVG_BARS_TO_STOP
                
                positions_open.append(int(duration))
                trades_taken += 1
                total_pnl += net_pnl_per_trade
            else:
                # Skip due to capital constraint
                trades_skipped += 1
    
    return {
        'scenario': f'Capital Constrained (${MAX_CAPITAL/1e6:.1f}M)',
        'trades_per_year': trades_taken,
        'trades_skipped': trades_skipped,
        'skip_rate': trades_skipped / (trades_taken + trades_skipped) if (trades_taken + trades_skipped) > 0 else 0,
        'net_pnl_per_trade': net_pnl_per_trade,
        'total_net_pnl': total_pnl,
        'avg_positions_open': np.mean(position_counts),
        'max_positions_open': max_concurrent_positions,
        'capital_utilized': np.mean(position_counts) * CAPITAL_PER_TRADE
    }


def estimate_analytical_constrained():
    """
    Analytical approximation of constrained performance.
    Based on queuing theory / capacity utilization.
    """
    
    avg_atr = CURRENT_PRICE * 0.01
    risk_dollars = STOP_ATR * avg_atr * SHARES_PER_TRADE
    reward_dollars = risk_dollars * RR_RATIO
    
    avg_win = reward_dollars
    avg_loss = -risk_dollars
    
    gross_pnl_per_trade = (WIN_RATE * avg_win) + ((1 - WIN_RATE) * avg_loss)
    net_pnl_per_trade = gross_pnl_per_trade - COST_PER_TRADE
    
    # Calculate capacity
    max_concurrent_positions = int(MAX_CAPITAL / CAPITAL_PER_TRADE)
    
    # Estimate average positions open
    # arrival_rate = trades_per_day / bars_per_day
    bars_per_day = 78  # 6.5 hours * 12 bars/hour
    arrival_rate = TRADES_PER_DAY / bars_per_day  # trades per bar
    service_rate = 1.0 / AVG_TRADE_DURATION  # positions closed per bar
    
    # Steady-state average positions (M/M/c queue approximation)
    rho = arrival_rate / service_rate  # utilization
    avg_positions = min(rho, max_concurrent_positions)
    
    # Estimate skip rate
    # When utilization exceeds capacity, we skip trades
    if rho <= max_concurrent_positions:
        skip_rate = 0.0
        trades_taken = TRADES_PER_YEAR
    else:
        # Approximate skip rate based on overcapacity
        skip_rate = 1.0 - (max_concurrent_positions / rho)
        trades_taken = int(TRADES_PER_YEAR * (1 - skip_rate))
    
    total_pnl = net_pnl_per_trade * trades_taken
    trades_skipped = TRADES_PER_YEAR - trades_taken
    
    return {
        'scenario': 'Analytical Estimate',
        'trades_per_year': trades_taken,
        'trades_skipped': trades_skipped,
        'skip_rate': skip_rate,
        'net_pnl_per_trade': net_pnl_per_trade,
        'total_net_pnl': total_pnl,
        'avg_positions_open': avg_positions,
        'max_positions_open': max_concurrent_positions,
        'capital_utilized': avg_positions * CAPITAL_PER_TRADE
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.flush()
    
    print("="*80, flush=True)
    print("CAPITAL-CONSTRAINED P&L ANALYSIS", flush=True)
    print("="*80, flush=True)
    print(f"\nConfiguration:", flush=True)
    print(f"  Max Capital: ${MAX_CAPITAL:,}", flush=True)
    print(f"  TSLA Price: ${CURRENT_PRICE}", flush=True)
    print(f"  Position Size: {SHARES_PER_TRADE} shares = ${CAPITAL_PER_TRADE:,}", flush=True)
    print(f"  Max Positions: {int(MAX_CAPITAL / CAPITAL_PER_TRADE)}", flush=True)
    print(f"\nStrategy (0.50 ATR):")
    print(f"  Win Rate: {WIN_RATE*100:.1f}%")
    print(f"  R:R: {RR_RATIO:.2f}:1")
    print(f"  EV: +{EV_PER_TRADE:.3f}R")
    print(f"  Expected trades/year: {TRADES_PER_YEAR:,}")
    print(f"  Expected trades/day: {TRADES_PER_DAY}")
    print(f"  Avg trade duration: {AVG_TRADE_DURATION:.1f} bars ({AVG_TRADE_DURATION*5:.0f} min)")
    
    print(f"\n{'='*80}")
    print("SCENARIO COMPARISON")
    print(f"{'='*80}\n")
    
    # Calculate all scenarios
    scenarios = []
    
    # 1. Original (unconstrained)
    scenarios.append(calculate_unconstrained_pnl())
    
    # 2. Analytical estimate
    scenarios.append(estimate_analytical_constrained())
    
    # 3. Monte Carlo simulation (run 10 times, take average)
    print("Running Monte Carlo simulations...")
    mc_results = []
    for i in range(10):
        np.random.seed(42 + i)
        mc_results.append(calculate_constrained_pnl())
    
    # Average MC results
    avg_mc = {
        'scenario': 'Monte Carlo (10 runs)',
        'trades_per_year': int(np.mean([r['trades_per_year'] for r in mc_results])),
        'trades_skipped': int(np.mean([r['trades_skipped'] for r in mc_results])),
        'skip_rate': np.mean([r['skip_rate'] for r in mc_results]),
        'net_pnl_per_trade': mc_results[0]['net_pnl_per_trade'],
        'total_net_pnl': np.mean([r['total_net_pnl'] for r in mc_results]),
        'avg_positions_open': np.mean([r['avg_positions_open'] for r in mc_results]),
        'max_positions_open': mc_results[0]['max_positions_open'],
        'capital_utilized': np.mean([r['capital_utilized'] for r in mc_results])
    }
    scenarios.append(avg_mc)
    
    # Print comparison table
    df = pd.DataFrame(scenarios)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80 + "\n")
    
    for scenario in scenarios:
        print(f"\n{scenario['scenario']}:")
        print(f"  Trades taken: {scenario['trades_per_year']:,}")
        print(f"  Trades skipped: {scenario['trades_skipped']:,}")
        print(f"  Skip rate: {scenario['skip_rate']*100:.1f}%")
        print(f"  P&L per trade: ${scenario['net_pnl_per_trade']:.2f}")
        print(f"  Total P&L: ${scenario['total_net_pnl']:,.0f}")
        if scenario['avg_positions_open'] > 0:
            print(f"  Avg positions: {scenario['avg_positions_open']:.1f}")
            print(f"  Capital used: ${scenario['capital_utilized']:,.0f} ({scenario['capital_utilized']/MAX_CAPITAL*100:.1f}%)")
    
    # Impact summary
    original_pnl = scenarios[0]['total_net_pnl']
    realistic_pnl = scenarios[2]['total_net_pnl']  # Use MC result
    pnl_reduction = original_pnl - realistic_pnl
    pct_reduction = (pnl_reduction / original_pnl) * 100
    
    print(f"\n{'='*80}")
    print("CAPITAL CONSTRAINT IMPACT")
    print(f"{'='*80}")
    print(f"\n  Original P&L (unconstrained): ${original_pnl:,.0f}")
    print(f"  Realistic P&L (constrained):  ${realistic_pnl:,.0f}")
    print(f"  Reduction: ${pnl_reduction:,.0f} ({pct_reduction:.1f}%)")
    print(f"\n  Key insight: With ${MAX_CAPITAL:,.0f} capital and ${CAPITAL_PER_TRADE:,} per trade,")
    print(f"  you can hold max {int(MAX_CAPITAL/CAPITAL_PER_TRADE)} positions simultaneously.")
    print(f"  With ~{TRADES_PER_DAY} signals/day and {AVG_TRADE_DURATION:.1f} bar duration,")
    print(f"  you'll skip ~{scenarios[2]['skip_rate']*100:.0f}% of trades due to capital limits.")
    
    print("\n" + "="*80)
