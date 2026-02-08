"""
Dynamic Stop Selection Strategy

This script tests whether dynamically choosing the best stop width per trade
can improve upon using a single fixed stop width.

Approach:
- Train all 9 RF models (one per stop width)
- For each test bar, evaluate all 9 models
- Select the "best" stop width using various criteria
- Compare results to fixed stop strategies
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

from label_generator import LabelConfig, generate_labels

# Configuration
DATA_FILE = Path("data/tsla_5min_10years.csv")
OUTPUT_DIR = Path("data")
STOP_ATRS = [0.25, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5]
TEST_YEAR = 2024

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_leaf': 50,
    'min_samples_split': 100,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

# P&L Configuration
SHARES_PER_TRADE = 100
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_PER_SHARE = 0.01
AVG_ENTRY_PRICE = 250.0


def load_data():
    """Load and prepare data."""
    print("\n" + "="*80)
    print("LOADING DATA FOR DYNAMIC STOP SELECTION")
    print("="*80)
    
    df = pd.read_csv(DATA_FILE)
    df['datetime'] = pd.to_datetime(df['time'], utc=True)
    df['date'] = df['datetime'].dt.date
    
    print(f"✓ Loaded {len(df):,} bars")
    return df


def calculate_indicators(df):
    """Calculate all indicators."""
    print("\nCalculating indicators...")
    df = df.copy()
    
    # ATR
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    pv = typical_price * df['volume']
    df['vwap'] = df.groupby('date').apply(
        lambda g: (pv.loc[g.index].cumsum() / df.loc[g.index, 'volume'].cumsum())
    ).reset_index(level=0, drop=True)
    
    # VWAP metrics
    df['vwap_width_atr'] = abs(df['close'] - df['vwap']) / df['atr']
    df['price_to_vwap_atr'] = (df['close'] - df['vwap']) / df['atr']
    df['is_long_setup'] = df['close'] < df['vwap']
    
    # VWAP dynamics
    df['vwap_slope'] = df['vwap'].diff(1)
    df['vwap_slope_5'] = df['vwap'].diff(5)
    df['vwap_helping'] = np.where(df['is_long_setup'], df['vwap_slope'] < 0, df['vwap_slope'] > 0).astype(int)
    
    # Volume
    df['rel_vol'] = df['volume'] / df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['volume'].shift(1)
    df['vol_at_extension'] = df['volume'] / df['volume'].rolling(5).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_slope'] = df['rsi'].diff(3)
    df['rsi_extreme'] = ((df['rsi'] < 30) | (df['rsi'] > 70)).astype(int)
    
    # Bar context
    df['bar_range_atr'] = (df['high'] - df['low']) / df['atr']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['crossed_vwap'] = (df['is_long_setup'] != df['is_long_setup'].shift(1)).astype(int)
    df['bars_from_vwap'] = df.groupby((df['crossed_vwap'] == 1).cumsum()).cumcount()
    
    # R:R metrics
    for stop_atr in STOP_ATRS:
        df[f'rr_{stop_atr}'] = df['vwap_width_atr'] / stop_atr
    
    rr_cols = [f'rr_{s}' for s in STOP_ATRS]
    df['avg_rr'] = df[rr_cols].mean(axis=1)
    df['min_rr'] = df[rr_cols].min(axis=1)
    df['max_rr'] = df[rr_cols].max(axis=1)
    
    return df


def get_features(df):
    """Get feature columns."""
    exclude = ['datetime', 'date', 'time', 'year', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'atr']
    exclude_prefixes = ['label_', 'rr_0', 'rr_1']
    
    features = []
    for col in df.columns:
        if col in exclude or any(col.startswith(p) for p in exclude_prefixes):
            continue
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32', 'bool']:
            features.append(col)
    
    return features


def train_all_models(df, features):
    """Train RF model for each stop width."""
    print("\n" + "="*80)
    print("TRAINING ALL RF MODELS")
    print("="*80)
    
    models = {}
    evs = {}
    rrs = {}
    
    for stop_atr in STOP_ATRS:
        print(f"\nTraining model for {stop_atr} ATR stop...")
        
        label_col = f"label_s{stop_atr}".replace(".", "_")
        valid = df[label_col].notna()
        df_valid = df[valid].copy()
        
        X = df_valid[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df_valid[label_col].astype(int)
        
        df_valid['year'] = pd.to_datetime(df_valid['datetime']).dt.year
        train_mask = df_valid['year'] < TEST_YEAR
        
        X_train, y_train = X[train_mask], y[train_mask]
        
        # Train
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train, y_train)
        
        # Calculate R:R and EV
        median_dist = df_valid['vwap_width_atr'].median()
        rr = median_dist / stop_atr
        
        # Store
        models[stop_atr] = rf
        rrs[stop_atr] = rr
        
        # Calculate baseline EV for reference
        raw_wr = y[~train_mask].mean()
        raw_ev = raw_wr * rr - (1 - raw_wr)
        evs[stop_atr] = raw_ev
        
        print(f"  ✓ R:R={rr:.2f}:1, Raw EV={raw_ev:+.3f}R")
    
    return models, rrs, evs


def evaluate_dynamic_selection(df, features, models, rrs):
    """Evaluate different dynamic selection strategies."""
    print("\n" + "="*80)
    print("EVALUATING DYNAMIC SELECTION STRATEGIES")
    print("="*80)
    
    # Get test data
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    test_mask = df['year'] >= TEST_YEAR
    df_test = df[test_mask].copy()
    
    X_test = df_test[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Get predictions from all models
    print("\nGenerating predictions from all models...")
    predictions = {}
    for stop_atr in STOP_ATRS:
        predictions[stop_atr] = models[stop_atr].predict_proba(X_test)[:, 1]
    
    # Create DataFrame with all predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.index = df_test.index
    
    results = []
    
    # Strategy 1: Max RF Probability
    print("\n--- Strategy 1: Max RF Probability ---")
    strategy_1 = evaluate_strategy_max_prob(df_test, pred_df, rrs, "Max RF Prob")
    results.append(strategy_1)
    
    # Strategy 2: Max Expected Value (RF_prob × R:R)
    print("\n--- Strategy 2: Max Expected Value ---")
    strategy_2 = evaluate_strategy_max_ev(df_test, pred_df, rrs, "Max EV")
    results.append(strategy_2)
    
    # Strategy 3: Threshold + Max EV (only trade if RF ≥ 0.5)
    print("\n--- Strategy 3: Threshold (RF≥0.5) + Max EV ---")
    strategy_3 = evaluate_strategy_threshold_ev(df_test, pred_df, rrs, "Threshold + MaxEV", 0.5)
    results.append(strategy_3)
    
    # Strategy 4: Adaptive (tight stops when high confidence, wide when low)
    print("\n--- Strategy 4: Adaptive (Confidence-based) ---")
    strategy_4 = evaluate_strategy_adaptive(df_test, pred_df, rrs, "Adaptive")
    results.append(strategy_4)
    
    # Baseline: Fixed stop strategies for comparison
    print("\n--- Baseline: Fixed Stop Strategies ---")
    for stop_atr in [0.25, 0.5, 1.0, 1.5]:
        baseline = evaluate_fixed_stop(df_test, pred_df[stop_atr], rrs[stop_atr], stop_atr)
        results.append(baseline)
    
    return pd.DataFrame(results)


def evaluate_strategy_max_prob(df_test, pred_df, rrs, strategy_name):
    """Pick stop width with highest RF probability."""
    selected_stops = pred_df.idxmax(axis=1)
    
    trades = []
    for idx in selected_stops.index:
        stop_atr = selected_stops[idx]
        rf_prob = pred_df.loc[idx, stop_atr]
        
        if rf_prob >= 0.5:  # Only trade if confident
            label_col = f"label_s{stop_atr}".replace(".", "_")
            outcome = df_test.loc[idx, label_col]
            
            if pd.notna(outcome):
                rr = rrs[stop_atr]
                pnl = calculate_trade_pnl(outcome, stop_atr, rr)
                trades.append({'stop': stop_atr, 'outcome': outcome, 'pnl': pnl, 'rf_prob': rf_prob})
    
    return summarize_trades(trades, strategy_name)


def evaluate_strategy_max_ev(df_test, pred_df, rrs, strategy_name):
    """Pick stop width with highest (RF_prob × R:R)."""
    # Calculate EV score for each stop
    ev_scores = pred_df.copy()
    for stop_atr in STOP_ATRS:
        ev_scores[stop_atr] = pred_df[stop_atr] * rrs[stop_atr] - (1 - pred_df[stop_atr])
    
    selected_stops = ev_scores.idxmax(axis=1)
    
    trades = []
    for idx in selected_stops.index:
        stop_atr = selected_stops[idx]
        rf_prob = pred_df.loc[idx, stop_atr]
        ev_score = ev_scores.loc[idx, stop_atr]
        
        if rf_prob >= 0.5 and ev_score > 0:  # Only trade if positive EV
            label_col = f"label_s{stop_atr}".replace(".", "_")
            outcome = df_test.loc[idx, label_col]
            
            if pd.notna(outcome):
                rr = rrs[stop_atr]
                pnl = calculate_trade_pnl(outcome, stop_atr, rr)
                trades.append({'stop': stop_atr, 'outcome': outcome, 'pnl': pnl, 'rf_prob': rf_prob})
    
    return summarize_trades(trades, strategy_name)


def evaluate_strategy_threshold_ev(df_test, pred_df, rrs, strategy_name, threshold):
    """Only trade if RF ≥ threshold, then pick max EV."""
    ev_scores = pred_df.copy()
    for stop_atr in STOP_ATRS:
        ev_scores[stop_atr] = pred_df[stop_atr] * rrs[stop_atr] - (1 - pred_df[stop_atr])
        # Mask out below threshold
        ev_scores.loc[pred_df[stop_atr] < threshold, stop_atr] = -999
    
    trades = []
    for idx in ev_scores.index:
        row = ev_scores.loc[idx]
        max_ev = row.max()
        
        if max_ev > 0:  # Has at least one positive EV trade
            stop_atr = row.idxmax()
            rf_prob = pred_df.loc[idx, stop_atr]
            
            label_col = f"label_s{stop_atr}".replace(".", "_")
            outcome = df_test.loc[idx, label_col]
            
            if pd.notna(outcome):
                rr = rrs[stop_atr]
                pnl = calculate_trade_pnl(outcome, stop_atr, rr)
                trades.append({'stop': stop_atr, 'outcome': outcome, 'pnl': pnl, 'rf_prob': rf_prob})
    
    return summarize_trades(trades, strategy_name)


def evaluate_strategy_adaptive(df_test, pred_df, rrs, strategy_name):
    """Use tighter stops when confidence is high, wider when low."""
    trades = []
    
    for idx in pred_df.index:
        # Get all predictions for this bar
        probs = pred_df.loc[idx]
        max_prob = probs.max()
        
        if max_prob >= 0.65:  # High confidence - use tight stop
            candidates = [0.25, 0.35, 0.4]
        elif max_prob >= 0.55:  # Medium confidence - use medium stop
            candidates = [0.5, 0.6, 0.75]
        elif max_prob >= 0.5:  # Low confidence - use wide stop
            candidates = [1.0, 1.25, 1.5]
        else:
            continue  # Skip if all below 0.5
        
        # Among candidates, pick highest EV
        best_stop = None
        best_ev = -999
        
        for stop_atr in candidates:
            if stop_atr not in STOP_ATRS:
                continue
            prob = probs[stop_atr]
            ev = prob * rrs[stop_atr] - (1 - prob)
            if ev > best_ev:
                best_ev = ev
                best_stop = stop_atr
        
        if best_stop and best_ev > 0:
            rf_prob = pred_df.loc[idx, best_stop]
            label_col = f"label_s{best_stop}".replace(".", "_")
            outcome = df_test.loc[idx, label_col]
            
            if pd.notna(outcome):
                rr = rrs[best_stop]
                pnl = calculate_trade_pnl(outcome, best_stop, rr)
                trades.append({'stop': best_stop, 'outcome': outcome, 'pnl': pnl, 'rf_prob': rf_prob})
    
    return summarize_trades(trades, strategy_name)


def evaluate_fixed_stop(df_test, rf_probs, rr, stop_atr):
    """Baseline: Fixed stop width."""
    label_col = f"label_s{stop_atr}".replace(".", "_")
    
    trades = []
    for idx, prob in rf_probs.items():
        if prob >= 0.5:
            outcome = df_test.loc[idx, label_col]
            if pd.notna(outcome):
                pnl = calculate_trade_pnl(outcome, stop_atr, rr)
                trades.append({'stop': stop_atr, 'outcome': outcome, 'pnl': pnl, 'rf_prob': prob})
    
    return summarize_trades(trades, f"Fixed {stop_atr} ATR")


def calculate_trade_pnl(outcome, stop_atr, rr):
    """Calculate P&L for a single trade."""
    avg_atr = AVG_ENTRY_PRICE * 0.01
    risk_dollars = stop_atr * avg_atr * SHARES_PER_TRADE
    reward_dollars = risk_dollars * rr
    costs = 2 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * SHARES_PER_TRADE
    
    if outcome == 1:  # Win
        return reward_dollars - costs
    else:  # Loss
        return -risk_dollars - costs


def summarize_trades(trades, strategy_name):
    """Summarize trade results."""
    if not trades:
        return {
            'strategy': strategy_name,
            'n_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'avg_rf_prob': 0
        }
    
    df = pd.DataFrame(trades)
    
    return {
        'strategy': strategy_name,
        'n_trades': len(df),
        'win_rate': df['outcome'].mean(),
        'total_pnl': df['pnl'].sum(),
        'avg_pnl': df['pnl'].mean(),
        'avg_rf_prob': df['rf_prob'].mean(),
        'stops_used': df['stop'].value_counts().to_dict() if 'stop' in df else {}
    }


def main():
    # Load and prepare data
    df = load_data()
    df = calculate_indicators(df)
    
    # Generate labels
    print("\nGenerating labels...")
    config = LabelConfig(stop_atrs=STOP_ATRS)
    df = generate_labels(df, config)
    
    # Get features
    features = get_features(df)
    print(f"\n✓ Using {len(features)} features")
    
    # Train all models
    models, rrs, evs = train_all_models(df, features)
    
    # Evaluate dynamic selection
    results = evaluate_dynamic_selection(df, features, models, rrs)
    
    # Display results
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(results[['strategy', 'n_trades', 'win_rate', 'total_pnl', 'avg_pnl']].to_string(index=False))
    
    # Save results
    output_file = OUTPUT_DIR / "dynamic_stop_selection_results.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✓ Saved results to: {output_file}")
    
    # Identify best strategy
    best_idx = results['total_pnl'].idxmax()
    best = results.loc[best_idx]
    
    print("\n" + "="*80)
    print("BEST STRATEGY")
    print("="*80)
    print(f"Strategy: {best['strategy']}")
    print(f"Trades: {best['n_trades']:,.0f}")
    print(f"Win Rate: {best['win_rate']*100:.1f}%")
    print(f"Total P&L: ${best['total_pnl']:,.0f}")
    print(f"Avg P&L/Trade: ${best['avg_pnl']:.2f}")
    
    return results


if __name__ == "__main__":
    results = main()
