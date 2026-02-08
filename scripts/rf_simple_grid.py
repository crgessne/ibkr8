"""
Simple RF Grid Search - sweep stop widths only, let RF use vwap_width_atr as continuous feature.
NOW WITH DOLLAR-BASED P&L CALCULATIONS!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from label_generator import LabelConfig, generate_labels

# Stop widths to test
STOP_ATRS = [0.25, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0]

# P&L Configuration (100 shares per trade)
SHARES_PER_TRADE = 100
COMMISSION_PER_SHARE = 0.005  # $0.005/share
SLIPPAGE_PER_SHARE = 0.01     # $0.01/share

def calculate_dollar_pnl(stop_atr, rr, win_rate, n_trades, avg_entry_price=250.0):
    """
    Calculate dollar-based P&L from R-multiple EV.
    
    Args:
        stop_atr: Stop width in ATR
        rr: Risk:Reward ratio
        win_rate: Win rate (0-1)
        n_trades: Number of trades
        avg_entry_price: Average entry price for position sizing
        
    Returns:
        Dict with P&L metrics
    """
    # Assume median ATR ~$2.50 for TSLA (rough estimate)
    avg_atr = avg_entry_price * 0.01  # 1% of price
    
    # Calculate dollar amounts per trade
    risk_dollars = stop_atr * avg_atr * SHARES_PER_TRADE
    reward_dollars = risk_dollars * rr
    
    # Costs per trade (entry + exit)
    costs_per_trade = 2 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * SHARES_PER_TRADE
    
    # Gross P&L
    avg_win = reward_dollars
    avg_loss = -risk_dollars
    gross_pnl_per_trade = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Net P&L (after costs)
    net_pnl_per_trade = gross_pnl_per_trade - costs_per_trade
    
    # Total P&L
    total_gross_pnl = gross_pnl_per_trade * n_trades
    total_costs = costs_per_trade * n_trades
    total_net_pnl = net_pnl_per_trade * n_trades
    
    # Capital required (peak exposure assuming sequential trades)
    capital_per_trade = avg_entry_price * SHARES_PER_TRADE
    
    # Return metrics
    return_pct = (net_pnl_per_trade / capital_per_trade) * 100 if capital_per_trade > 0 else 0
    
    return {
        'risk_dollars': risk_dollars,
        'reward_dollars': reward_dollars,
        'gross_pnl_per_trade': gross_pnl_per_trade,
        'costs_per_trade': costs_per_trade,
        'net_pnl_per_trade': net_pnl_per_trade,
        'total_gross_pnl': total_gross_pnl,
        'total_costs': total_costs,
        'total_net_pnl': total_net_pnl,
        'capital_per_trade': capital_per_trade,
        'return_pct_per_trade': return_pct,
    }

def get_features(df):
    """Get feature columns - include vwap_width_atr as continuous feature."""
    exclude = ['date', 'year', 'open', 'high', 'low', 'close', 'volume', 
               'vwap', 'atr', 'bb_upper', 'bb_lower', 'bb_middle', 
               'price_below_vwap', 'price_to_vwap', 'price_to_vwap_pct',
               'vwap_width_pct', 'vwap_dist_pct', 'vwap_dist_atr',
               'profit_potential_atr', 'in_sweet_spot', 'in_tradeable_zone',
               'over_extended', 'extreme_extension']
    
    exclude_prefixes = ['label_', 'zone_', 'long_rr', 'short_rr', 'rr_']
    
    features = []
    for c in df.columns:
        if c in exclude or any(c.startswith(p) for p in exclude_prefixes):
            continue
        if df[c].dtype in ['float64', 'int64', 'float32', 'int32']:
            features.append(c)
    
    return features

def run_rf(df, stop_atr, features, test_year=2024):
    """Train RF for a single stop width."""
    label_col = f"label_s{stop_atr}".replace(".", "_")
    
    # Filter valid labels
    valid = df[label_col].notna()
    df_valid = df[valid].copy()
    
    if len(df_valid) < 500:
        return None
    
    X = df_valid[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_valid[label_col].astype(int)
    
    # Train/test split
    df_valid['year'] = pd.to_datetime(df_valid['date']).apply(lambda x: x.year)
    train_mask = df_valid['year'] < test_year
    test_mask = df_valid['year'] >= test_year
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if len(X_train) < 200 or len(X_test) < 50:
        return None
    
    # Train RF
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=50,
        min_samples_split=100, max_features='sqrt',
        random_state=42, n_jobs=-1, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    # Test predictions
    proba = rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics - use median vwap_width_atr for R:R
    median_dist = df_valid['vwap_width_atr'].median()
    rr = median_dist / stop_atr
    breakeven = 1 / (1 + rr)
      raw_wr = y_test.mean()
    raw_ev = raw_wr * rr - (1 - raw_wr)
    
    # Calculate dollar P&L for baseline (raw)
    raw_pnl = calculate_dollar_pnl(stop_atr, rr, raw_wr, len(y_test))
    
    result = {
        'stop_atr': stop_atr,
        'rr': rr,
        'breakeven_wr': breakeven,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'raw_wr': raw_wr,
        'raw_ev': raw_ev,
        'raw_pnl_per_trade': raw_pnl['net_pnl_per_trade'],
        'raw_total_pnl': raw_pnl['total_net_pnl'],
        'raw_return_pct': raw_pnl['return_pct_per_trade'],
    }
    
    # RF filtered at thresholds
    for thresh in [0.4, 0.5, 0.55, 0.6]:
        mask = proba >= thresh
        n = mask.sum()
        if n >= 30:
            wr = y_test[mask].mean()
            ev = wr * rr - (1 - wr)
            
            # Calculate dollar P&L for this threshold
            thresh_pnl = calculate_dollar_pnl(stop_atr, rr, wr, n)
            
            result[f'rf{thresh}_n'] = n
            result[f'rf{thresh}_wr'] = wr
            result[f'rf{thresh}_ev'] = ev
            result[f'rf{thresh}_pnl_per_trade'] = thresh_pnl['net_pnl_per_trade']
            result[f'rf{thresh}_total_pnl'] = thresh_pnl['total_net_pnl']
            result[f'rf{thresh}_return_pct'] = thresh_pnl['return_pct_per_trade']
        else:
            result[f'rf{thresh}_n'] = n
            result[f'rf{thresh}_wr'] = np.nan
            result[f'rf{thresh}_ev'] = np.nan
            result[f'rf{thresh}_pnl_per_trade'] = np.nan
            result[f'rf{thresh}_total_pnl'] = np.nan
            result[f'rf{thresh}_return_pct'] = np.nan
    
    # Feature importances (top 10)
    importances = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    result['top_features'] = ', '.join(importances['feature'].tolist())
    
    return result

def main():
    print("="*80)
    print("SIMPLE RF GRID SEARCH - Stop Width Sweep Only")
    print("="*80)
    
    data_path = Path(__file__).parent.parent / "data" / "tsla_5min_10years_indicators.csv"
    print(f"\nLoading {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
    print(f"Loaded {len(df):,} bars")    # Generate labels
    print("\nGenerating labels...")
    config = LabelConfig(stop_atrs=STOP_ATRS)
    df = generate_labels(df, config)
    
    features = get_features(df)
    print(f"\nUsing {len(features)} features (including vwap_width_atr)")
    
    # Run RF for each stop
    print(f"\nRunning RF for {len(STOP_ATRS)} stop widths...\n")
    results = []
      for i, stop in enumerate(STOP_ATRS, 1):
        print(f"[{i}/{len(STOP_ATRS)}] Stop {stop} ATR...", end=" ")
        result = run_rf(df, stop, features)
        if result:
            results.append(result)
            raw_ev = result['raw_ev']
            raw_pnl = result['raw_pnl_per_trade']
            rf_ev = result.get('rf0.5_ev', np.nan)
            rf_pnl = result.get('rf0.5_pnl_per_trade', np.nan)
            print(f"Raw: EV={raw_ev:+.3f}R, ${raw_pnl:+.2f}/trade | RF≥0.5: EV={rf_ev:+.3f}R, ${rf_pnl:+.2f}/trade")
        else:
            print("skipped")
    
    # Results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save
    output_path = Path(__file__).parent.parent / "data" / "rf_simple_grid_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print("\nALL RESULTS:")
    cols = ['stop_atr', 'rr', 'n_test', 'raw_wr', 'raw_ev', 'rf0.5_n', 'rf0.5_wr', 'rf0.5_ev']
    print(results_df[cols].to_string(index=False))
    
    print("\nPOSITIVE EV (Raw):")
    pos_raw = results_df[results_df['raw_ev'] > 0].sort_values('raw_ev', ascending=False)
    if len(pos_raw) > 0:
        print(pos_raw[cols].to_string(index=False))
    else:
        print("None found")
    
    print("\nPOSITIVE EV (RF >= 0.5):")
    pos_rf = results_df[results_df['rf0.5_ev'] > 0].sort_values('rf0.5_ev', ascending=False)
    if len(pos_rf) > 0:
        print(pos_rf[cols].to_string(index=False))
    else:
        print("None found")
      print("\nBEST SETUP:")
    best = results_df.loc[results_df['rf0.5_ev'].idxmax()]
    print(f"Stop: {best['stop_atr']} ATR")
    print(f"R:R: {best['rr']:.2f}:1")
    print(f"Raw WR: {best['raw_wr']:.1%} (EV={best['raw_ev']:+.3f})")
    print(f"RF>=0.5 WR: {best['rf0.5_wr']:.1%} (EV={best['rf0.5_ev']:+.3f}, N={best['rf0.5_n']:.0f})")
    print(f"Top features: {best['top_features']}")
    
    # === DOLLAR P&L SUMMARY ===
    print("\n" + "="*80)
    print("DOLLAR P&L SUMMARY (100 shares/trade, $0.005 commission, $0.01 slippage)")
    print("="*80)
    
    print("\n🎯 BEST SETUP P&L BREAKDOWN:")
    print(f"Stop: {best['stop_atr']} ATR (R:R = {best['rr']:.2f}:1)")
    print(f"\nBaseline (No RF Filter):")
    print(f"  Win Rate: {best['raw_wr']:.1%}")
    print(f"  EV: {best['raw_ev']:+.3f}R")
    print(f"  Net P&L per trade: ${best['raw_pnl_per_trade']:+.2f}")
    print(f"  Return per trade: {best['raw_return_pct']:+.2f}%")
    print(f"  Total P&L ({int(best['n_test'])} trades): ${best['raw_total_pnl']:+,.2f}")
    
    print(f"\nRF ≥ 0.5 Filtered:")
    print(f"  Win Rate: {best['rf0.5_wr']:.1%}")
    print(f"  EV: {best['rf0.5_ev']:+.3f}R")
    print(f"  Net P&L per trade: ${best['rf0.5_pnl_per_trade']:+.2f}")
    print(f"  Return per trade: {best['rf0.5_return_pct']:+.2f}%")
    print(f"  Total P&L ({int(best['rf0.5_n'])} trades): ${best['rf0.5_total_pnl']:+,.2f}")
    
    pnl_improvement = ((best['rf0.5_pnl_per_trade'] - best['raw_pnl_per_trade']) / 
                       abs(best['raw_pnl_per_trade']) * 100) if best['raw_pnl_per_trade'] != 0 else 0
    print(f"\n💰 P&L Improvement: {pnl_improvement:+.1f}%")
    
    # Compare all stops at RF≥0.5
    print("\n" + "="*80)
    print("ALL STOPS @ RF≥0.5: DOLLAR P&L COMPARISON")
    print("="*80)
    print(f"\n{'Stop':<6} {'R:R':<6} {'WR':<8} {'EV':<10} {'$/Trade':<12} {'Total P&L':<15} {'Trades':<8}")
    print("-"*75)
    
    for _, row in results_df.iterrows():
        stop = row['stop_atr']
        rr = row['rr']
        wr = row['rf0.5_wr']
        ev = row['rf0.5_ev']
        pnl_per = row['rf0.5_pnl_per_trade']
        total_pnl = row['rf0.5_total_pnl']
        n = int(row['rf0.5_n'])
        
        marker = " ⭐" if stop == best['stop_atr'] else ""
        print(f"{stop:<6.2f} {rr:<6.2f} {wr:<8.1%} {ev:<+10.3f} ${pnl_per:<+11.2f} ${total_pnl:<+14,.0f} {n:<8,}{marker}")
    
    print("\n✅ KEY TAKEAWAYS:")
    print(f"   • Best EV: {best['stop_atr']} ATR stop → {best['rf0.5_ev']:+.3f}R per trade")
    print(f"   • Best $/Trade: {best['stop_atr']} ATR stop → ${best['rf0.5_pnl_per_trade']:+.2f} per trade")
    print(f"   • Total Test P&L: ${best['rf0.5_total_pnl']:+,.2f} across {int(best['rf0.5_n']):,} trades")
    print(f"   • Costs matter: ~${2*(COMMISSION_PER_SHARE+SLIPPAGE_PER_SHARE)*SHARES_PER_TRADE:.2f} per trade in commissions + slippage")

if __name__ == "__main__":
    main()
