"""
P&L Add-on for existing RF pipeline

Converts R-multiple outcomes to dollar P&L with position sizing.
Adds dollar-based metrics ON TOP OF existing EV calculations.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def add_pnl_columns(
    df: pd.DataFrame,
    shares_per_trade: int = 100,
    stop_atr_width: float = 0.25,
    target_rr: float = 6.0,
    commission_per_share: float = 0.005,
    slippage_per_share: float = 0.01
) -> pd.DataFrame:
    """
    Add dollar-based P&L columns to existing dataframe with 'outcome' column.
    
    Assumes df has:
    - 'outcome': 'TARGET', 'STOP', or 'TIMEOUT'
    - 'close': entry price
    - 'atr': ATR value
    
    Adds columns:
    - pnl_dollars: Gross P&L in dollars
    - pnl_net: Net P&L after commission & slippage
    - capital_used: Entry price * shares
    - risk_dollars: Dollar risk (stop distance * shares)
    
    Args:
        df: DataFrame with outcome column
        shares_per_trade: Shares per trade (default 100)
        stop_atr_width: Stop width in ATR (default 0.25)
        target_rr: Target risk:reward ratio (default 6.0)
        commission_per_share: Commission per share per side
        slippage_per_share: Slippage per share per side
        
    Returns:
        DataFrame with P&L columns added
    """
    result = df.copy()
    
    # Calculate dollar risk and reward
    result['risk_dollars'] = stop_atr_width * result['atr'] * shares_per_trade
    result['reward_dollars'] = result['risk_dollars'] * target_rr
    result['capital_used'] = result['close'] * shares_per_trade
    
    # Calculate gross P&L based on outcome
    result['pnl_dollars'] = np.where(
        result['outcome'] == 'TARGET',
        result['reward_dollars'],  # Hit target = +6R
        np.where(
            result['outcome'] == 'STOP',
            -result['risk_dollars'],  # Hit stop = -1R
            0.0  # Timeout = 0
        )
    )
    
    # Subtract costs
    costs = (2 * commission_per_share * shares_per_trade +  # Entry + Exit commission
             2 * slippage_per_share * shares_per_trade)      # Entry + Exit slippage
    
    result['pnl_net'] = result['pnl_dollars'] - costs
    result['costs'] = costs
    
    return result


def calculate_pnl_metrics(
    df: pd.DataFrame,
    filter_col: str = None,
    filter_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate P&L metrics from dataframe with P&L columns.
    
    Args:
        df: DataFrame with pnl_net column
        filter_col: Optional column to filter by (e.g., 'rf_prob')
        filter_threshold: Threshold for filtering
        
    Returns:
        Dictionary of P&L metrics
    """
    # Apply filter if specified
    if filter_col and filter_col in df.columns:
        df_filt = df[df[filter_col] >= filter_threshold].copy()
    else:
        df_filt = df.copy()
    
    if len(df_filt) == 0:
        return {
            'total_trades': 0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'total_ev': 0.0,
            'ev_per_trade': 0.0,
        }
    
    pnls = df_filt['pnl_net']
    
    total_pnl = pnls.sum()
    avg_pnl = pnls.mean()
    total_ev = avg_pnl * len(pnls)  # EV = avg * count
    
    # Win/loss breakdown
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    
    return {
        'total_trades': len(pnls),
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'total_ev': total_ev,
        'ev_per_trade': avg_pnl,
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'avg_win': wins.mean() if len(wins) > 0 else 0,
        'avg_loss': losses.mean() if len(losses) > 0 else 0,
        'largest_win': wins.max() if len(wins) > 0 else 0,
        'largest_loss': losses.min() if len(losses) > 0 else 0,
        'total_costs': df_filt['costs'].sum(),
    }


def print_pnl_comparison(
    metrics_unfiltered: Dict[str, float],
    metrics_filtered: Dict[str, float],
    scenario_names: Tuple[str, str] = ('Unfiltered', 'RF Filtered')
) -> None:
    """Print side-by-side P&L comparison."""
    
    print("\n" + "="*80)
    print("P&L COMPARISON (Dollar-Based)")
    print("="*80)
    
    print(f"\n{'Metric':<30} {scenario_names[0]:>20} {scenario_names[1]:>20}")
    print("-"*72)
    
    # Trade counts
    print(f"{'Total Trades':<30} {metrics_unfiltered['total_trades']:>20,} {metrics_filtered['total_trades']:>20,}")
    print(f"{'Winning Trades':<30} {metrics_unfiltered['winning_trades']:>20,} {metrics_filtered['winning_trades']:>20,}")
    print(f"{'Losing Trades':<30} {metrics_unfiltered['losing_trades']:>20,} {metrics_filtered['losing_trades']:>20,}")
    
    print()
    # P&L metrics
    print(f"{'Total P&L':<30} ${metrics_unfiltered['total_pnl']:>19,.2f} ${metrics_filtered['total_pnl']:>19,.2f}")
    print(f"{'Total EV':<30} ${metrics_unfiltered['total_ev']:>19,.2f} ${metrics_filtered['total_ev']:>19,.2f}")
    print(f"{'EV per Trade':<30} ${metrics_unfiltered['ev_per_trade']:>19,.2f} ${metrics_filtered['ev_per_trade']:>19,.2f}")
    print(f"{'Avg P&L per Trade':<30} ${metrics_unfiltered['avg_pnl']:>19,.2f} ${metrics_filtered['avg_pnl']:>19,.2f}")
    
    print()
    # Win/Loss breakdown
    print(f"{'Avg Win':<30} ${metrics_unfiltered['avg_win']:>19,.2f} ${metrics_filtered['avg_win']:>19,.2f}")
    print(f"{'Avg Loss':<30} ${metrics_unfiltered['avg_loss']:>19,.2f} ${metrics_filtered['avg_loss']:>19,.2f}")
    print(f"{'Largest Win':<30} ${metrics_unfiltered['largest_win']:>19,.2f} ${metrics_filtered['largest_win']:>19,.2f}")
    print(f"{'Largest Loss':<30} ${metrics_unfiltered['largest_loss']:>19,.2f} ${metrics_filtered['largest_loss']:>19,.2f}")
    
    print()
    # Costs
    print(f"{'Total Costs':<30} ${metrics_unfiltered['total_costs']:>19,.2f} ${metrics_filtered['total_costs']:>19,.2f}")
    
    print("="*80)
    
    # Calculate improvement
    if metrics_unfiltered['total_pnl'] != 0:
        pnl_improvement = ((metrics_filtered['total_pnl'] - metrics_unfiltered['total_pnl']) / 
                          abs(metrics_unfiltered['total_pnl'])) * 100
        print(f"\nP&L Improvement: {pnl_improvement:+.1f}%")


if __name__ == "__main__":
    # Example usage
    print("P&L Add-on Module")
    print("Use this to add dollar-based P&L to your existing RF pipeline")
    print()
    print("Example:")
    print("  from pnl_addon import add_pnl_columns, calculate_pnl_metrics")
    print("  df = add_pnl_columns(df, shares_per_trade=100)")
    print("  metrics = calculate_pnl_metrics(df, filter_col='rf_prob', filter_threshold=0.5)")
