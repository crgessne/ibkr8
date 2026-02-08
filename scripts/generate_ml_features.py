"""
Generate ML-ready dataset with reversal context features.

This script creates a dataset that can be used for downstream ML to:
1. Predict which reversal trades will be profitable
2. Optimize stop/target parameters per trade
3. Filter out low-quality setups
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from indicators import calc_all_indicators, identify_reversal_setups, calc_theo_targets
from outcome_sim import simulate_trade_outcome


def generate_ml_dataset(input_file: str, 
                        output_file: str = None,
                        max_rsi_long: float = 35.0,
                        min_rsi_short: float = 65.0,
                        min_rel_vol: float = 1.0,
                        stop_atr: float = 1.5) -> pd.DataFrame:
    """
    Generate ML feature dataset from bar data.
    
    Returns a DataFrame where each row is a potential reversal trade with:
    - Entry conditions/context (features for ML)
    - Trade parameters (stop, target)
    - Outcome (win/loss/EOD exit, P&L, R:R achieved)
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, parse_dates=['time'], index_col='time')
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Calculate all indicators
    print("\nCalculating indicators...")
    df_ind = calc_all_indicators(df)
    
    # Identify setups
    df_setups = identify_reversal_setups(df_ind, max_rsi_long, min_rsi_short, min_rel_vol)
    
    # Get trade rows
    long_mask = df_setups['long_setup']
    short_mask = df_setups['short_setup']
    
    print(f"\nFound {long_mask.sum()} LONG setups and {short_mask.sum()} SHORT setups")
    
    # Build feature set for each trade
    trades = []
    
    # Process LONG trades
    for idx in df_setups[long_mask].index:
        row = df_setups.loc[idx]
        trade = extract_trade_features(row, 'LONG', stop_atr)
        trade['entry_time'] = idx
        trades.append(trade)
    
    # Process SHORT trades
    for idx in df_setups[short_mask].index:
        row = df_setups.loc[idx]
        trade = extract_trade_features(row, 'SHORT', stop_atr)
        trade['entry_time'] = idx
        trades.append(trade)
    
    if not trades:
        print("No trades found!")
        return pd.DataFrame()
      # Create DataFrame
    trade_df = pd.DataFrame(trades)
    trade_df = trade_df.sort_values('entry_time').reset_index(drop=True)
    
    print(f"\nGenerated {len(trade_df)} trade feature rows")
    
    # Simulate outcomes
    print("\nSimulating trade outcomes...")
    
    # Create a time-to-index map for quick lookup
    time_to_idx = {t: i for i, t in enumerate(df.index)}
    
    outcomes = []
    for _, trade in trade_df.iterrows():
        entry_time = trade['entry_time']
        if entry_time not in time_to_idx:
            # Skip if entry time not in data
            outcomes.append({'outcome': 'SKIP', 'exit_price': np.nan, 
                           'exit_time': None, 'bars_held': 0, 
                           'pnl': np.nan, 'rr_achieved': np.nan})
            continue
            
        entry_idx = time_to_idx[entry_time]
        result = simulate_trade_outcome(
            df=df,
            entry_idx=entry_idx,
            direction=trade['direction'],
            entry_price=trade['entry_price'],
            target=trade['target'],
            stop=trade['stop']
        )
        outcomes.append({
            'outcome': result.outcome,
            'exit_price': result.exit_price,
            'exit_time': result.exit_time,
            'bars_held': result.bars_held,
            'pnl': result.pnl_per_share,
            'rr_achieved': result.rr_achieved
        })
    
    # Merge outcomes with features
    outcome_df = pd.DataFrame(outcomes)
    for col in outcome_df.columns:
        trade_df[col] = outcome_df[col].values
    
    # Add derived outcome features
    trade_df['win'] = (trade_df['outcome'] == 'TARGET').astype(int)
    trade_df['loss'] = (trade_df['outcome'] == 'STOP').astype(int)
    trade_df['eod'] = (trade_df['outcome'] == 'EOD').astype(int)
    
    # Summary stats
    print("\n=== DATASET SUMMARY ===")
    print(f"Total trades: {len(trade_df)}")
    print(f"  LONG: {(trade_df['direction'] == 'LONG').sum()}")
    print(f"  SHORT: {(trade_df['direction'] == 'SHORT').sum()}")
    print(f"\nOutcomes:")
    print(f"  Wins (TARGET): {trade_df['win'].sum()} ({100*trade_df['win'].mean():.1f}%)")
    print(f"  Losses (STOP): {trade_df['loss'].sum()} ({100*trade_df['loss'].mean():.1f}%)")
    print(f"  EOD exits: {trade_df['eod'].sum()} ({100*trade_df['eod'].mean():.1f}%)")
    
    if trade_df['pnl'].notna().any():
        print(f"\nP&L Stats:")
        print(f"  Total P&L: ${trade_df['pnl'].sum():.2f}")
        print(f"  Avg P&L per trade: ${trade_df['pnl'].mean():.2f}")
        print(f"  Win avg: ${trade_df[trade_df['win']==1]['pnl'].mean():.2f}")
        print(f"  Loss avg: ${trade_df[trade_df['loss']==1]['pnl'].mean():.2f}")
    
    # Save to file
    if output_file:
        trade_df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")
    
    return trade_df


def extract_trade_features(row: pd.Series, direction: str, stop_atr: float) -> dict:
    """Extract features for a single trade."""
    
    close = row['close']
    vwap = row['vwap']
    atr = row['atr']
    
    # Basic trade setup
    if direction == 'LONG':
        stop = close - stop_atr * atr
        target = vwap if vwap > close else close + 1.5 * atr
        rr_theo = (target - close) / (close - stop)
    else:
        stop = close + stop_atr * atr
        target = vwap if vwap < close else close - 1.5 * atr
        rr_theo = (close - target) / (stop - close)
      # Core features
    features = {
        'direction': direction,
        'entry_price': close,
        'stop': stop,
        'target': target,
        'stop_atr': stop_atr,
        'rr_theo': rr_theo,
        
        # Price context
        'atr': atr,
        'close': close,
        'vwap': vwap,
        
        # Distance features
        'price_to_vwap_atr': row.get('price_to_vwap_atr', (close - vwap) / atr),
        'vwap_width_atr': row.get('vwap_width_atr', abs(close - vwap) / atr),
        'vwap_width_pct': row.get('vwap_width_pct', abs(close - vwap) / vwap * 100),
        
        # Momentum indicators
        'rsi': row['rsi'],
        'rsi_extremity': row.get('rsi_extremity', abs(row['rsi'] - 50)),
        'bb_pct': row['bb_pct'],
        
        # Extension
        'bb_extension': row.get('bb_extension', 0),
        'bb_extension_abs': row.get('bb_extension_abs', 0),
        
        # Volume
        'rel_vol': row['rel_vol'],
        'vol_at_extension': row.get('vol_at_extension', row['rel_vol']),
        'vol_declining': row.get('vol_declining', 0),
        'vol_trend_3': row.get('vol_trend_3', 0),
        
        # === REVERSAL QUALITY FEATURES ===
        # Momentum divergence (key reversal signal)
        'momentum_divergence_3': row.get('momentum_divergence_3', 0),
        'momentum_divergence_5': row.get('momentum_divergence_5', 0),
        
        # RSI slope (turning?)
        'rsi_slope_3': row.get('rsi_slope_3', 0),
        'rsi_slope_5': row.get('rsi_slope_5', 0),
        
        # Price action
        'reversal_wick': row.get('reversal_wick', 0),
        'reversal_close_position': row.get('reversal_close_position', 0),
        
        # Momentum exhaustion
        'bar_range_atr': row.get('bar_range_atr', 0),
        'range_vs_prev': row.get('range_vs_prev', 1),
        'consecutive_shrinking': row.get('consecutive_shrinking', 0),
        
        # Extension dynamics
        'extension_velocity_3': row.get('extension_velocity_3', 0),
        'extension_velocity_5': row.get('extension_velocity_5', 0),
        'extension_accel': row.get('extension_accel', 0),
        
        # VWAP dynamics
        'vwap_slope_5': row.get('vwap_slope_5', 0),
        'vwap_helping': row.get('vwap_helping', 0),        'vwap_dist_delta_3': row.get('vwap_dist_delta_3', 0),
        'vwap_dist_delta_5': row.get('vwap_dist_delta_5', 0),
        
        # R:R at various stops
        'rr_0_5atr': row.get('rr_0_5atr', 0),
        'rr_1_0atr': row.get('rr_1_0atr', 0),
        'rr_1_5atr': row.get('rr_1_5atr', 0),
        'rr_2_0atr': row.get('rr_2_0atr', 0),
        'rr_2_5atr': row.get('rr_2_5atr', 0),
        'rr_3_0atr': row.get('rr_3_0atr', 0),
        'avg_rr': row.get('avg_rr', 0),
        
        # Profit potential
        'profit_potential_atr': row.get('profit_potential_atr', abs(close - vwap) / atr),
        
        # VWAP Zone classification (empirical: 1-1.5 ATR = 53% WR, >3 ATR = 17%, >4 ATR = 6%)
        'in_sweet_spot': row.get('in_sweet_spot', False),
        'in_tradeable_zone': row.get('in_tradeable_zone', False),
        'over_extended': row.get('over_extended', False),
        'extreme_extension': row.get('extreme_extension', False),
        
        # Other
        'atr_move': row.get('atr_move', 0),
    }
    
    return features


def feature_analysis(df: pd.DataFrame):
    """Analyze feature importance for winning vs losing trades."""
    
    print("\n=== REVERSAL QUALITY FEATURE ANALYSIS ===")
    print("(Focus: What predicts successful reversal given significant extension?)")
    
    winners = df[df['win'] == 1]
    losers = df[df['loss'] == 1]
    
    if len(winners) == 0 or len(losers) == 0:
        print("Not enough data for analysis")
        return
    
    # Group features by category
    feature_groups = {
        'Momentum Divergence': ['momentum_divergence_3', 'momentum_divergence_5', 'rsi_slope_3', 'rsi_slope_5'],
        'Price Action': ['reversal_wick', 'reversal_close_position', 'bar_range_atr'],
        'Exhaustion Signals': ['consecutive_shrinking', 'range_vs_prev', 'vol_declining', 'vol_trend_3'],
        'Extension Dynamics': ['extension_velocity_3', 'extension_accel', 'vwap_helping'],
        'Setup Quality': ['rsi_extremity', 'bb_extension_abs', 'vwap_width_atr', 'rel_vol'],
    }
    
    print("\n" + "="*70)
    print(f"{'Feature':<30} {'Winners':>10} {'Losers':>10} {'Diff':>10} {'Corr':>8}")
    print("="*70)
    
    for group_name, features in feature_groups.items():
        print(f"\n--- {group_name} ---")
        for col in features:
            if col in df.columns:
                w_mean = winners[col].mean()
                l_mean = losers[col].mean()
                diff = w_mean - l_mean
                corr = df[[col, 'win']].corr().iloc[0, 1]
                marker = "**" if abs(corr) > 0.1 else "  "
                print(f"{marker}{col:<28} {w_mean:>10.3f} {l_mean:>10.3f} {diff:>+10.3f} {corr:>+8.3f}")
    
    # Top correlations
        print("\n" + "="*70)
    print("TOP FEATURES BY CORRELATION WITH WIN:")
    print("="*70)
    
    all_features = [f for features in feature_groups.values() for f in features]
    correlations = []
    for col in all_features:
        if col in df.columns:
            corr = df[[col, 'win']].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations.append((col, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in correlations[:10]:
        print(f"  {col:<30} {corr:>+8.3f}")
    
    # === ANALYSIS FOR SIGNIFICANT EXTENSIONS ONLY ===
    # Filter to trades where there's actually a reversal opportunity
    print("\n" + "="*70)
    print("ANALYSIS FOR SIGNIFICANT EXTENSIONS (>1.5 ATR from VWAP)")
    print("="*70)
    
    if 'vwap_width_atr' in df.columns:
        sig_ext = df[df['vwap_width_atr'] > 1.5]
        sig_winners = sig_ext[sig_ext['win'] == 1]
        sig_losers = sig_ext[sig_ext['loss'] == 1]
        
        print(f"\nTrades: {len(sig_ext)} (Winners: {len(sig_winners)}, Losers: {len(sig_losers)})")
        print(f"Win rate: {100*sig_ext['win'].mean():.1f}%")
        
        if len(sig_winners) > 0 and len(sig_losers) > 0:
            print(f"\n{'Feature':<30} {'Winners':>10} {'Losers':>10} {'Diff':>10} {'Corr':>8}")
            print("-"*70)
            
            key_features = [
                'reversal_wick', 'reversal_close_position', 'vol_declining',
                'consecutive_shrinking', 'rsi_slope_3', 'extension_velocity_3',
                'extension_accel', 'vwap_helping', 'rel_vol', 'bar_range_atr'
            ]
            
            sig_correlations = []
            for col in key_features:
                if col in sig_ext.columns:
                    w_mean = sig_winners[col].mean()
                    l_mean = sig_losers[col].mean()
                    diff = w_mean - l_mean
                    corr = sig_ext[[col, 'win']].corr().iloc[0, 1]
                    if not np.isnan(corr):
                        sig_correlations.append((col, corr, w_mean, l_mean, diff))
                        marker = "**" if abs(corr) > 0.05 else "  "
                        print(f"{marker}{col:<28} {w_mean:>10.3f} {l_mean:>10.3f} {diff:>+10.3f} {corr:>+8.3f}")
            
            print("\nTop predictive features for significant extensions:")
            sig_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            for col, corr, w, l, d in sig_correlations[:5]:
                direction = "HIGHER better" if corr > 0 else "LOWER better"
                print(f"  {col:<28} corr={corr:>+.3f} ({direction})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ML features for reversal trades')
    parser.add_argument('input', help='Input CSV file with bar data')
    parser.add_argument('-o', '--output', help='Output CSV file', default=None)
    parser.add_argument('--stop-atr', type=float, default=1.5, help='Stop distance in ATR')
    parser.add_argument('--rsi-long', type=float, default=35.0, help='Max RSI for long setups')
    parser.add_argument('--rsi-short', type=float, default=65.0, help='Min RSI for short setups')
    parser.add_argument('--min-vol', type=float, default=1.0, help='Min relative volume')
    parser.add_argument('--analyze', action='store_true', help='Run feature analysis')
    
    args = parser.parse_args()
    
    if args.output is None:
        # Generate output filename from input
        base = args.input.replace('.csv', '')
        args.output = f"{base}_ml_features.csv"
    
    # Generate dataset
    df = generate_ml_dataset(
        args.input,
        args.output,
        max_rsi_long=args.rsi_long,
        min_rsi_short=args.rsi_short,
        min_rel_vol=args.min_vol,
        stop_atr=args.stop_atr
    )
    
    if args.analyze and len(df) > 0:
        feature_analysis(df)
