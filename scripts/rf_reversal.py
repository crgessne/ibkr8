"""
Random Forest classifier for reversal trade prediction.

Trains on reversal context features to predict which trades will hit target vs stop.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def load_ml_data(filepath: str) -> pd.DataFrame:
    """Load ML features dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} trades from {filepath}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix and target vector.
    
    Returns:
        X: Feature matrix
        y: Target (1=win, 0=loss)
        feature_names: List of feature names
    """
    # Define feature columns (reversal quality indicators)
    feature_cols = [
        # Distance metrics
        'vwap_width_atr', 'price_to_vwap_atr',
        
        # Momentum indicators
        'rsi', 'rsi_extremity', 'bb_pct',
        'rsi_slope_3', 'rsi_slope_5',
        
        # Divergence
        'momentum_divergence_3', 'momentum_divergence_5',
        
        # Price action
        'reversal_wick', 'reversal_close_position',
        'bar_range_atr',
        
        # Exhaustion signals
        'consecutive_shrinking', 'range_vs_prev',
        'vol_declining', 'vol_trend_3',
        
        # Extension dynamics
        'extension_velocity_3', 'extension_velocity_5',
        'extension_accel',
        'vwap_helping',
        
        # Volume
        'rel_vol', 'vol_at_extension',
        
        # Extension metrics
        'bb_extension', 'bb_extension_abs',
        
        # R:R metrics
        'rr_theo', 'avg_rr',
        'rr_1_0atr', 'rr_1_5atr', 'rr_2_0atr',
    ]
    
    # Filter to available columns
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
    
    print(f"Using {len(available_cols)} features")
    
    # Filter to trades with definitive outcomes (exclude EOD)
    df_filtered = df[df['outcome'].isin(['TARGET', 'STOP'])].copy()
    print(f"Trades with definitive outcomes: {len(df_filtered)} (excluded {len(df) - len(df_filtered)} EOD)")
    
    # Create feature matrix
    X = df_filtered[available_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Target: 1 = win (TARGET), 0 = loss (STOP)
    y = (df_filtered['outcome'] == 'TARGET').astype(int)
    
    return X, y, available_cols, df_filtered


def train_random_forest(X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2,
                        n_estimators: int = 100,
                        max_depth: int = 10,
                        min_samples_leaf: int = 20,
                        random_state: int = 42) -> dict:
    """
    Train Random Forest and return results.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} ({100*y_train.mean():.1f}% wins)")
    print(f"Test set: {len(X_test)} ({100*y_test.mean():.1f}% wins)")
    
    # Train model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
    
    return {
        'model': rf,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred, 'y_prob': y_prob,
        'cv_scores': cv_scores
    }


def evaluate_model(results: dict, feature_names: list):
    """Print model evaluation metrics."""
    y_test = results['y_test']
    y_pred = results['y_pred']
    y_prob = results['y_prob']
    rf = results['model']
    cv_scores = results['cv_scores']
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Basic metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['STOP', 'TARGET']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              STOP    TARGET")
    print(f"Actual STOP   {cm[0,0]:>5}    {cm[0,1]:>5}")
    print(f"Actual TARGET {cm[1,0]:>5}    {cm[1,1]:>5}")
    
    # ROC AUC
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC: {auc:.3f}")
    print(f"CV AUC (5-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Top 15)")
    print("="*60)
    
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importances.head(15).iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"{row['feature']:<30} {row['importance']:.3f} {bar}")
    
    return importances


def simulate_trading(results: dict, df_filtered: pd.DataFrame, threshold: float = 0.5):
    """
    Simulate trading using model predictions.
    """
    y_prob = results['y_prob']
    y_test = results['y_test']
    X_test = results['X_test']
    
    # Get test indices
    test_idx = X_test.index
    test_df = df_filtered.loc[test_idx].copy()
    test_df['pred_prob'] = y_prob
    test_df['pred_win'] = (y_prob >= threshold).astype(int)
    
    print("\n" + "="*60)
    print(f"TRADING SIMULATION (threshold={threshold})")
    print("="*60)
    
    # All trades (baseline)
    all_wins = (test_df['outcome'] == 'TARGET').sum()
    all_trades = len(test_df)
    all_pnl = test_df['pnl'].sum()
    print(f"\nBaseline (all trades):")
    print(f"  Trades: {all_trades}, Wins: {all_wins} ({100*all_wins/all_trades:.1f}%)")
    print(f"  Total P&L: ${all_pnl:.2f}")
    
    # Filtered trades
    filtered = test_df[test_df['pred_win'] == 1]
    if len(filtered) > 0:
        filt_wins = (filtered['outcome'] == 'TARGET').sum()
        filt_trades = len(filtered)
        filt_pnl = filtered['pnl'].sum()
        print(f"\nFiltered by RF (prob >= {threshold}):")
        print(f"  Trades: {filt_trades}, Wins: {filt_wins} ({100*filt_wins/filt_trades:.1f}%)")
        print(f"  Total P&L: ${filt_pnl:.2f}")
        print(f"  Trades filtered out: {all_trades - filt_trades}")
    
    # Try different thresholds
    print("\n" + "-"*60)
    print("Performance at different probability thresholds:")
    print("-"*60)
    print(f"{'Threshold':<12} {'Trades':>8} {'Wins':>8} {'WinRate':>10} {'P&L':>12}")
    
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        subset = test_df[test_df['pred_prob'] >= thresh]
        if len(subset) > 0:
            wins = (subset['outcome'] == 'TARGET').sum()
            wr = 100 * wins / len(subset)
            pnl = subset['pnl'].sum()
            print(f"{thresh:<12} {len(subset):>8} {wins:>8} {wr:>9.1f}% ${pnl:>10.2f}")


def analyze_by_zone(df: pd.DataFrame, results: dict):
    """Analyze model performance by VWAP zone."""
    X_test = results['X_test']
    y_test = results['y_test']
    y_prob = results['y_prob']
    
    print("\n" + "="*60)
    print("PERFORMANCE BY VWAP ZONE")
    print("="*60)
    
    # Add predictions to test data
    test_data = X_test.copy()
    test_data['actual'] = y_test.values
    test_data['pred_prob'] = y_prob
    
    zones = [
        ('Sweet spot (1.0-1.5 ATR)', (1.0, 1.5)),
        ('Moderate (1.5-2.5 ATR)', (1.5, 2.5)),
        ('Extended (2.5-3.5 ATR)', (2.5, 3.5)),
        ('Over-extended (>3.5 ATR)', (3.5, 100)),
    ]
    
    print(f"\n{'Zone':<30} {'N':>6} {'Actual WR':>12} {'Pred Prob':>12}")
    print("-"*60)
    
    for zone_name, (low, high) in zones:
        mask = (test_data['vwap_width_atr'] >= low) & (test_data['vwap_width_atr'] < high)
        subset = test_data[mask]
        if len(subset) > 0:
            actual_wr = 100 * subset['actual'].mean()
            avg_prob = 100 * subset['pred_prob'].mean()
            print(f"{zone_name:<30} {len(subset):>6} {actual_wr:>11.1f}% {avg_prob:>11.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Random Forest for reversal trades')
    parser.add_argument('input', nargs='?', default='data/tsla_5min_2years_ml_features.csv',
                        help='Input ML features CSV')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max-depth', type=int, default=10, help='Max tree depth')
    parser.add_argument('--min-samples-leaf', type=int, default=20, help='Min samples per leaf')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    args = parser.parse_args()
    
    # Load data
    df = load_ml_data(args.input)
    
    # Prepare features
    X, y, feature_names, df_filtered = prepare_features(df)
    
    print(f"\nClass balance: {100*y.mean():.1f}% wins, {100*(1-y.mean()):.1f}% losses")
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    print(f"Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"min_samples_leaf={args.min_samples_leaf}")
    
    results = train_random_forest(
        X, y,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf
    )
    
    # Evaluate
    importances = evaluate_model(results, feature_names)
    
    # Trading simulation
    simulate_trading(results, df_filtered, threshold=args.threshold)
    
    # Zone analysis
    analyze_by_zone(df, results)
