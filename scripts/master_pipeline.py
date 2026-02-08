"""
MASTER PIPELINE: Complete RF VWAP Reversion Analysis

This script runs the complete analysis pipeline:
1. Load data (ensure 100K+ bars)
2. Calculate indicators (remove redundant ones)
3. Generate labels for all stop widths
4. Train RF models for each width
5. Calculate EV metrics
6. Generate P&L projections
7. Save comprehensive results

Usage:
    python scripts/master_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

from label_generator import LabelConfig, generate_labels
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = Path("data/tsla_5min_10years.csv")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Stop widths to analyze (ATR multiples)
STOP_ATRS = [0.25, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5]

# Train/test split
TEST_YEAR = 2024

# RF Parameters
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
COMMISSION_PER_SHARE = 0.005  # $0.005/share
SLIPPAGE_PER_SHARE = 0.01     # $0.01/share
AVG_ENTRY_PRICE = 400.0       # TSLA average price (used for notional/risk sizing)

# RF Threshold to analyze
RF_THRESHOLDS = [0.5, 0.55, 0.6, 0.65]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_validate_data(filepath):
    """Load data and ensure minimum bar count."""
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    df = pd.read_csv(filepath)    # Handle time column
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'], utc=True)
        df['date'] = df['datetime'].dt.date  # Date-only for label generation
    elif df.index.name == 'time':
        df = df.reset_index()
        df['datetime'] = pd.to_datetime(df['time'], utc=True)
        df['date'] = df['datetime'].dt.date
    else:
        raise ValueError("No 'time' column found")
    
    print(f"✓ Loaded {len(df):,} bars")
    print(f"✓ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"✓ Data span: {(df['datetime'].max() - df['datetime'].min()).days} days")
    
    if len(df) < 100000:
        print(f"\n⚠ WARNING: Only {len(df):,} bars (< 100K minimum)")
        print("Consider using a larger dataset for robust analysis")
    else:
        print(f"✓ Dataset exceeds 100K bars requirement")
    
    return df


def calculate_core_indicators(df):
    """Calculate only essential, non-redundant indicators."""
    print(f"\n{'='*80}")
    print("CALCULATING INDICATORS")
    print(f"{'='*80}")
    
    df = df.copy()
    
    # ========================================
    # 1. ATR (14-period)
    # ========================================
    print("Calculating ATR...")
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
      # ========================================
    # 2. VWAP (daily reset)
    # ========================================
    print("Calculating VWAP...")
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    pv = typical_price * df['volume']
    
    df['vwap'] = df.groupby('date').apply(
        lambda g: (pv.loc[g.index].cumsum() / df.loc[g.index, 'volume'].cumsum())
    ).reset_index(level=0, drop=True)
    
    # ========================================
    # 3. VWAP Distance Metrics
    # ========================================
    print("Calculating VWAP distance metrics...")
    df['vwap_width_atr'] = abs(df['close'] - df['vwap']) / df['atr']
    df['price_to_vwap_atr'] = (df['close'] - df['vwap']) / df['atr']  # Signed
    df['is_long_setup'] = df['close'] < df['vwap']
    
    # ========================================
    # 4. VWAP Dynamics
    # ========================================
    print("Calculating VWAP dynamics...")
    df['vwap_slope'] = df['vwap'].diff(1)
    df['vwap_slope_5'] = df['vwap'].diff(5)
    
    # Is VWAP "helping" (moving toward price)?
    df['vwap_helping'] = np.where(
        df['is_long_setup'],
        df['vwap_slope'] < 0,  # VWAP moving down helps long
        df['vwap_slope'] > 0   # VWAP moving up helps short
    ).astype(int)
    
    # ========================================
    # 5. Volume Metrics
    # ========================================
    print("Calculating volume metrics...")
    df['rel_vol'] = df['volume'] / df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['volume'].shift(1)
      # Volume at extension (current volume relative to nearby bars)
    df['vol_at_extension'] = df['volume'] / df['volume'].rolling(5).mean()
    
    # ========================================
    # 6. Momentum Indicators
    # ========================================
    print("Calculating momentum indicators...")
    
    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # RSI momentum
    df['rsi_slope'] = df['rsi'].diff(3)
    df['rsi_extreme'] = ((df['rsi'] < 30) | (df['rsi'] > 70)).astype(int)
    
    # ========================================
    # 7. Bar Context
    # ========================================
    print("Calculating bar context...")
    df['bar_range_atr'] = (df['high'] - df['low']) / df['atr']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Time since crossed VWAP
    df['crossed_vwap'] = (df['is_long_setup'] != df['is_long_setup'].shift(1)).astype(int)
    df['bars_from_vwap'] = df.groupby((df['crossed_vwap'] == 1).cumsum()).cumcount()
    
    # ========================================
    # 8. R:R Metrics (for each stop width)
    # ========================================
    print("Calculating R:R metrics...")
    for stop_atr in STOP_ATRS:
        df[f'rr_{stop_atr}'] = df['vwap_width_atr'] / stop_atr
      # Average R:R across all stops
    rr_cols = [f'rr_{s}' for s in STOP_ATRS]
    df['avg_rr'] = df[rr_cols].mean(axis=1)
    df['min_rr'] = df[rr_cols].min(axis=1)
    df['max_rr'] = df[rr_cols].max(axis=1)
    
    print(f"✓ Calculated {len([c for c in df.columns if c not in ['datetime', 'date', 'open', 'high', 'low', 'close', 'volume', 'time']])} features")
    
    return df


def get_feature_columns(df):
    """Get non-redundant feature columns for RF."""
    exclude = [
        'datetime', 'date', 'time', 'year', 'open', 'high', 'low', 'close', 'volume',
        'vwap', 'atr',
        'avg_rr', 'min_rr', 'max_rr'  # Exclude aggregate R:R features (data leakage)
    ]
    
    exclude_prefixes = ['label_', 'rr_']  # Exclude all R:R columns
    
    features = []
    for col in df.columns:
        if col in exclude:
            continue
        if any(col.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32', 'bool']:
            features.append(col)
    
    return features


def generate_all_labels(df):
    """Generate labels for all stop widths."""
    print(f"\n{'='*80}")
    print("GENERATING LABELS")
    print(f"{'='*80}")
    
    config = LabelConfig(stop_atrs=STOP_ATRS)
    df_labeled = generate_labels(df, config)
    
    # Count valid labels per stop
    for stop_atr in STOP_ATRS:
        label_col = f"label_s{stop_atr}".replace(".", "_")
        n_valid = df_labeled[label_col].notna().sum()
        win_rate = df_labeled[label_col].mean() * 100
        print(f"  Stop {stop_atr:4.2f} ATR: {n_valid:7,} valid labels ({win_rate:5.2f}% win rate)")
    
    return df_labeled


def train_rf_model(df, stop_atr, features, test_year=2024, train_start_year=None):
    """Train RF model for a single stop width."""
    label_col = f"label_s{stop_atr}".replace(".", "_")
    
    # Filter valid labels
    valid = df[label_col].notna()
    df_valid = df[valid].copy()
    
    if len(df_valid) < 500:
        return None
    
    X = df_valid[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_valid[label_col].astype(int)
    
    # Train/test split by year
    df_valid['year'] = pd.to_datetime(df_valid['datetime']).dt.year
    if train_start_year is None:
        train_mask = df_valid['year'] < test_year
    else:
        train_mask = (df_valid['year'] >= train_start_year) & (df_valid['year'] < test_year)
    test_mask = df_valid['year'] >= test_year
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if len(X_train) < 200 or len(X_test) < 50:
        return None
    
    # Train model
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    
    # Predictions
    proba_test = rf.predict_proba(X_test)[:, 1]
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': rf,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'proba_test': proba_test,
        'importance': importance,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }


def walk_forward_resample_fixed_stop(df, stop_atr, features, start_train_year, start_test_year, end_test_year, rf_threshold=0.5, slippage_per_share=SLIPPAGE_PER_SHARE):
    """Walk-forward by year with a fixed stop ATR (Option A).

    Each fold trains on [start_train_year, test_year-1] and tests on exactly test_year.
    """
    rows = []

    # Precompute rr for this stop for consistency (median dist on all valid labels)
    label_col = f"label_s{stop_atr}".replace(".", "_")
    valid_mask = df[label_col].notna()
    median_dist = df.loc[valid_mask, 'vwap_width_atr'].median()
    rr = float(median_dist / stop_atr) if stop_atr else np.nan

    # Build a valid-labeled frame once for year masking
    df_valid = df.loc[valid_mask].copy()
    df_valid['year'] = pd.to_datetime(df_valid['datetime']).dt.year

    for test_year in range(start_test_year, end_test_year + 1):
        # Train up to year-1
        res = train_rf_model(
            df,
            stop_atr=stop_atr,
            features=features,
            test_year=test_year,
            train_start_year=start_train_year,
        )
        if res is None:
            continue        # IMPORTANT: train_rf_model() returns X_test/y_test/proba_test for years >= test_year.
        # For walk-forward we want a *single-year* slice, so we must compute predictions on that year only.
        df_test_year = df_valid[(df_valid['year'] == test_year)].copy()
        if len(df_test_year) < 50:
            continue
        
        X_year = df_test_year[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        proba_year = res['model'].predict_proba(X_year)[:, 1]
        y_year = df_test_year[label_col].astype(int).values
        
        sel = proba_year >= rf_threshold
        if sel.sum() < 10:
            continue
        
        wr = float(y_year[sel].mean())
        ev_r = wr * rr - (1 - wr)
        n_trades = int(sel.sum())        # Per-trade P&L using actual ATR for this year
        atr_year_selected = df_test_year['atr'].values[sel]
        pnl = calculate_dollar_pnl_pertrade(
            stop_atr=stop_atr,
            rr=rr,
            y_actual=y_year[sel],
            atr_series=atr_year_selected,
            slippage_per_share=slippage_per_share,
        )

        rows.append({
            'train_years': f"{start_train_year}-{test_year-1}",
            'test_year': int(test_year),
            'test_bars': int(len(df_test_year)),
            'trades': n_trades,
            'win_rate': wr,
            'ev_r': ev_r,
            'net_pnl': float(pnl['total_net_pnl']),
        })

    return pd.DataFrame(rows)


def calculate_ev_metrics(y_test, proba_test, stop_atr, median_dist):
    """Calculate EV metrics for various RF thresholds."""
    rr = median_dist / stop_atr
    breakeven_wr = 1 / (1 + rr)
    
    results = []
    
    # Raw (no filtering)
    raw_wr = y_test.mean()
    raw_ev = raw_wr * rr - (1 - raw_wr)
    results.append({
        'threshold': 0.0,
        'n_trades': len(y_test),
        'win_rate': raw_wr,
        'ev': raw_ev,
        'pct_filtered': 0.0
    })
    
    # RF filtered
    for thresh in RF_THRESHOLDS:
        mask = proba_test >= thresh
        if mask.sum() < 10:
            continue
        
        filtered_wr = y_test[mask].mean()
        filtered_ev = filtered_wr * rr - (1 - filtered_wr)
        pct_filtered = (1 - mask.mean()) * 100
        
        results.append({
            'threshold': thresh,
            'n_trades': mask.sum(),
            'win_rate': filtered_wr,
            'ev': filtered_ev,
            'pct_filtered': pct_filtered
        })
    
    return pd.DataFrame(results), rr, breakeven_wr


def calculate_dollar_pnl_pertrade(stop_atr, rr, y_actual, atr_series, avg_price=AVG_ENTRY_PRICE, slippage_per_share=SLIPPAGE_PER_SHARE):
    """Calculate dollar P&L using per-trade ATR.
    
    Args:
        stop_atr: stop width in ATR multiples
        rr: risk:reward ratio
        y_actual: array of actual outcomes (1=win, 0=loss) for selected trades
        atr_series: array of ATR values ($/share) for selected trades
        avg_price: average entry price for notional capital calculation
        slippage_per_share: slippage cost per share in dollars (default: SLIPPAGE_PER_SHARE)
    
    Returns:
        dict with total P&L and per-trade averages
    """
    n_trades = len(y_actual)
    if n_trades == 0:
        return {
            'total_gross_pnl': 0.0,
            'total_net_pnl': 0.0,
            'total_costs': 0.0,
            'avg_risk_dollars': 0.0,
            'avg_net_pnl_per_trade': 0.0,
            'capital_per_trade': avg_price * SHARES_PER_TRADE,
            'return_pct_per_trade': 0.0,
        }
    
    # Per-trade calculations
    costs_per_trade = 2 * (COMMISSION_PER_SHARE + slippage_per_share) * SHARES_PER_TRADE
    
    # Vectorized: risk/reward for each trade based on its ATR
    risk_per_trade = stop_atr * atr_series * SHARES_PER_TRADE
    reward_per_trade = risk_per_trade * rr
    
    # P&L per trade: wins get +reward, losses get -risk
    gross_pnl_per_trade = np.where(y_actual == 1, reward_per_trade, -risk_per_trade)
    net_pnl_per_trade = gross_pnl_per_trade - costs_per_trade
    
    # Totals
    total_gross_pnl = float(gross_pnl_per_trade.sum())
    total_net_pnl = float(net_pnl_per_trade.sum())
    total_costs = float(costs_per_trade * n_trades)
    
    # Averages
    avg_risk_dollars = float(risk_per_trade.mean())
    avg_net_pnl_per_trade = float(net_pnl_per_trade.mean())
    
    capital_per_trade = avg_price * SHARES_PER_TRADE
    return_pct = (avg_net_pnl_per_trade / capital_per_trade) * 100 if capital_per_trade > 0 else 0.0
    
    return {
        'total_gross_pnl': total_gross_pnl,
        'total_net_pnl': total_net_pnl,
        'total_costs': total_costs,
        'avg_risk_dollars': avg_risk_dollars,
        'avg_net_pnl_per_trade': avg_net_pnl_per_trade,
        'capital_per_trade': capital_per_trade,
        'return_pct_per_trade': return_pct,
    }


def _median_test_atr_dollars(df, label_col, test_year=TEST_YEAR):
    """Median realized ATR (in dollars/share) over labeled bars in the test period."""
    try:
        valid = df[label_col].notna()
        df_valid = df.loc[valid].copy()
        df_valid['year'] = pd.to_datetime(df_valid['datetime']).dt.year
        df_test = df_valid[df_valid['year'] >= test_year]
        if 'atr' not in df_test.columns or len(df_test) == 0:
            return None
        atr_med = float(pd.to_numeric(df_test['atr'], errors='coerce').dropna().median())
        return atr_med if np.isfinite(atr_med) and atr_med > 0 else None
    except Exception:
        return None


def save_results(all_results, features_used, n_features=None, walk_forward_df=None, slippage_per_share=SLIPPAGE_PER_SHARE):
    """Save comprehensive results to CSV and markdown with all summary tables."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If n_features not provided, use length of provided feature list
    if n_features is None:
        n_features = len(features_used)

    # Save detailed CSV
    csv_path = OUTPUT_DIR / f"master_pipeline_results_{timestamp}.csv"
    all_results.to_csv(csv_path, index=False)
    print(f"\n✓ Saved detailed results to: {csv_path}")

    # --------------------------------------------------------------------
    # Compute test-period bar counts from actual data (remove static 0.432)
    # --------------------------------------------------------------------
    # We use the smallest-stop label column as a proxy for "eligible bars" since
    # label generation produces non-NaN values only when a forward outcome exists.
    total_test_bars = None
    test_years = None
    bars_per_year = None

    try:
        df_stats = pd.read_csv(DATA_FILE)
        if 'time' in df_stats.columns:
            dt = pd.to_datetime(df_stats['time'], utc=True)
        else:
            # fallback
            dt = pd.to_datetime(df_stats.index, utc=True)
        df_stats['datetime'] = dt
        df_stats['date'] = df_stats['datetime'].dt.date

        df_stats = calculate_core_indicators(df_stats)
        df_stats = generate_all_labels(df_stats)

        stop_for_stats = STOP_ATRS[0]
        label_col_stats = f"label_s{stop_for_stats}".replace(".", "_")
        mask_valid = df_stats[label_col_stats].notna()

        # eligible bars in test period
        years = pd.to_datetime(df_stats.loc[mask_valid, 'datetime']).dt.year
        mask_test = years >= TEST_YEAR
        total_test_bars = int(mask_test.sum())

        if total_test_bars > 0:
            dt_min = df_stats.loc[mask_valid].loc[mask_test, 'datetime'].min()
            dt_max = df_stats.loc[mask_valid].loc[mask_test, 'datetime'].max()
            test_years = max((dt_max - dt_min).days / 365.25, 1e-9)
            bars_per_year = total_test_bars / test_years
        else:
            # fallback to previous constants
            total_test_bars = 40293
            test_years = 2.11
            bars_per_year = 19093
    except Exception:
        # fallback to previous constants
        total_test_bars = 40293
        test_years = 2.11
        bars_per_year = 19093

    # Convenience filtered subsets
    rf_05 = all_results[all_results['rf_threshold'] == 0.5].copy()
    rf_0 = all_results[all_results['rf_threshold'] == 0.0].copy()

    # Helper: format percent
    def fmt_pct(x, digits=1):
        try:
            return f"{float(x) * 100:.{digits}f}%"
        except Exception:
            return "n/a"    # Helper: compute per-year results for the recommended strategy
    def compute_recommended_results_by_year(rec_stop_atr: float, rec_rr: float, rf_threshold: float = 0.5):
        """Return a per-year DataFrame for the recommended stop (test years only)."""
        try:
            df_local = load_and_validate_data(DATA_FILE)
            df_local = calculate_core_indicators(df_local)
            features_local = get_feature_columns(df_local)
            df_local = generate_all_labels(df_local)

            model_res = train_rf_model(df_local, rec_stop_atr, features_local, TEST_YEAR)
            if model_res is None:
                return None

            # Recompute mask in the same order as train_rf_model()
            label_col = f"label_s{rec_stop_atr}".replace(".", "_")
            valid = df_local[label_col].notna()
            df_valid = df_local.loc[valid].copy()
            df_valid['year'] = pd.to_datetime(df_valid['datetime']).dt.year
            test_mask = df_valid['year'] >= TEST_YEAR

            df_test = df_valid.loc[test_mask].copy()
            y_test = model_res['y_test']
            proba_test = model_res['proba_test']            # Align prediction arrays to df_test index (they should match lengths)
            if len(df_test) != len(y_test) or len(y_test) != len(proba_test):
                df_test = df_test.reset_index(drop=True)
                y_test = pd.Series(np.asarray(y_test), index=df_test.index)
                proba_test = pd.Series(np.asarray(proba_test), index=df_test.index)
            else:
                y_test = pd.Series(y_test.values if hasattr(y_test, 'values') else np.asarray(y_test), index=df_test.index)
                proba_test = pd.Series(np.asarray(proba_test), index=df_test.index)
            
            sel = proba_test >= rf_threshold
            df_sel = df_test.loc[sel].copy()
            y_sel = y_test.loc[sel]
            
            if len(df_sel) == 0:
                return pd.DataFrame(columns=['year', 'trades', 'win_rate', 'ev_r', 'net_pnl'])

            out_rows = []
            for yr, grp in df_sel.groupby(pd.to_datetime(df_sel['datetime']).dt.year):
                mask = grp.index
                y_grp = y_sel.loc[mask]
                trades = int(y_grp.shape[0])
                wr = float(y_grp.mean()) if trades else 0.0
                ev_r = wr * rec_rr - (1 - wr)                # Per-trade P&L using actual ATR for this year's selected bars
                atr_year = grp['atr'].values
                pnl = calculate_dollar_pnl_pertrade(
                    stop_atr=rec_stop_atr,
                    rr=rec_rr,
                    y_actual=y_grp.values,
                    atr_series=atr_year,
                    slippage_per_share=slippage_per_share,
                )

                out_rows.append({
                    'year': int(yr),
                    'trades': trades,
                    'win_rate': wr,
                    'ev_r': ev_r,
                    'net_pnl': float(pnl['total_net_pnl']),
                })

            return pd.DataFrame(out_rows).sort_values('year')
        except Exception:
            return None

    # Generate comprehensive markdown summary
    md_path = OUTPUT_DIR / f"master_pipeline_summary_{timestamp}.md"

    with open(md_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# 📊 VWAP Reversion Strategy - Complete Analysis\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Dataset**: {DATA_FILE.name}\n")
        f.write(f"**Test Period**: {TEST_YEAR}+ ({total_test_bars:,.0f} eligible bars, {test_years:.2f} years)\n")
        f.write(f"**Eligible bars per year**: ~{bars_per_year:,.0f}\n")
        f.write(f"**Features Used**: {n_features}\n")
        f.write(f"**Position Size**: {SHARES_PER_TRADE} shares @ ${AVG_ENTRY_PRICE:.0f}/share\n\n")

        # ====================================================================
        # SECTION: SUMMARY TABLES
        # ====================================================================
        f.write("---\n\n")
        f.write("## 📌 Summary Tables\n\n")

        if len(all_results) == 0:
            f.write("No results were generated. Check data availability, label generation, and model training.\n\n")
        else:
            # Best EV strategies (RF≥0.5)
            if len(rf_05) > 0:
                top_ev = rf_05.sort_values('ev', ascending=False).head(10)
                f.write("### Top 10 Strategies (by EV, RF≥0.50)\n\n")
                f.write("| Stop (ATR) | R:R | Win Rate | EV (R) | Trades (test) | % Filtered | Net P&L (test) |\n")
                f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
                for _, r in top_ev.iterrows():
                    f.write(
                        f"| {r['stop_atr']:.2f} | {r['rr']:.2f} | {r['win_rate']*100:.1f}% | {r['ev']:+.3f} | "
                        f"{int(r['n_trades']):,} | {r['pct_filtered']:.1f}% | ${r['total_net_pnl']:,.0f} |\n"
                    )
                f.write("\n")

            # Raw baseline table
            if len(rf_0) > 0:
                raw = rf_0.sort_values('stop_atr')
                f.write("### Baseline (No RF Filter, threshold=0.00)\n\n")
                f.write("| Stop (ATR) | R:R | Win Rate | EV (R) | Trades (test) | Net P&L (test) |\n")
                f.write("|---:|---:|---:|---:|---:|---:|\n")
                for _, r in raw.iterrows():
                    f.write(
                        f"| {r['stop_atr']:.2f} | {r['rr']:.2f} | {r['win_rate']*100:.1f}% | {r['ev']:+.3f} | "
                        f"{int(r['n_trades']):,} | ${r['total_net_pnl']:,.0f} |\n"
                    )
                f.write("\n")

        # ====================================================================
        # SECTION: TOP FEATURES
        # ====================================================================
        f.write("---\n\n")
        f.write("## 🔑 Top 15 Features (by importance)\n\n")
        f.write("Based on 0.25 ATR stop model:\n\n")
        for i, feat in enumerate(features_used[:15], 1):
            f.write(f"{i:2d}. `{feat}`\n")
        f.write("\n")

        # ====================================================================
        # SECTION: KEY FINDINGS & RECOMMENDATIONS
        # ====================================================================
        f.write("---\n\n")
        f.write("## 🎯 Key Findings & Recommendations\n\n")

        if len(all_results) == 0 or len(rf_05) == 0:
            f.write("No RF≥0.50 results available to summarize.\n\n")
        else:
            best_ev_row = rf_05.loc[rf_05['ev'].idxmax()]
            best_pnl_row = rf_05.loc[rf_05['total_net_pnl'].idxmax()]            # Option 1: maximize net P&L (RF≥0.50)
            rec_row = best_pnl_row
            
            rec_stop_atr = float(rec_row['stop_atr'])
            rec_rr = float(rec_row['rr'])
            rec_wr = float(rec_row['win_rate'])
            rec_ev = float(rec_row['ev'])
            rec_trades_test = int(rec_row['n_trades'])
            rec_pnl_test = float(rec_row['total_net_pnl'])
            rec_pnl_per_trade = float(rec_row['avg_net_pnl_per_trade'])
            rec_capital_per_trade = float(rec_row['capital_per_trade'])

            # Annualized projection using bars/year estimate
            selection_rate = rec_trades_test / total_test_bars if total_test_bars else 0.0
            est_trades_per_year = bars_per_year * selection_rate
            rec_pnl_year = rec_pnl_per_trade * est_trades_per_year

            f.write("### Best Strategies\n\n")
            f.write(
                f"- **Best EV (RF≥0.50):** {best_ev_row['stop_atr']:.2f} ATR | "
                f"EV={best_ev_row['ev']:+.3f}R | WR={best_ev_row['win_rate']*100:.1f}% | "
                f"R:R={best_ev_row['rr']:.2f}:1\n"
            )
            f.write(
                f"- **Best Net P&L (test, RF≥0.50):** {best_pnl_row['stop_atr']:.2f} ATR | "
                f"Net P&L=${best_pnl_row['total_net_pnl']:,.0f} | Trades={int(best_pnl_row['n_trades']):,}\n\n"
            )

            f.write("### Recommended (for scaling tables)\n\n")
            f.write(
                "Scaling/projection tables use the **max net P&L** strategy at **RF≥0.50**.\n\n"
            )
            f.write(
                f"- **Recommended stop:** {rec_stop_atr:.2f} ATR\n"
                f"- **RF threshold:** 0.50\n"
                f"- **Win rate:** {rec_wr*100:.1f}%\n"
                f"- **R:R:** {rec_rr:.2f}:1\n"
                f"- **EV:** {rec_ev:+.3f}R\n"
                f"- **Net P&L (test period):** ${rec_pnl_test:,.0f} across {rec_trades_test:,} trades\n"
                f"- **Estimated trades/year:** ~{est_trades_per_year:,.0f}\n"                f"- **Estimated net P&L/year (100 shares):** ${rec_pnl_year:,.0f}\n\n"
            )
            
            # New section: results by year (recommended strategy)
            f.write("### Results by Year (Recommended Strategy, RF≥0.50)\n\n")
            yearly_df = compute_recommended_results_by_year(rec_stop_atr, rec_rr, rf_threshold=0.5)
            if yearly_df is None or len(yearly_df) == 0:
                f.write("Yearly breakdown not available.\n\n")
            else:
                f.write("| Year | Trades | Win Rate | EV (R) | Net P&L |\n")
                f.write("|---:|---:|---:|---:|---:|\n")
                for _, r in yearly_df.iterrows():
                    f.write(
                        f"| {int(r['year'])} | {int(r['trades']):,} | {r['win_rate']*100:.1f}% | {r['ev_r']:+.3f} | ${r['net_pnl']:,.0f} |\n"
                    )
                f.write("\n")
            
            f.write("### Capital & Execution Assumptions\n\n")
            f.write(
                f"- **Capital per trade (notional):** ${rec_capital_per_trade:,.0f} "
                f"({SHARES_PER_TRADE} shares × ${AVG_ENTRY_PRICE:.0f})\n"
                f"- **Commission per share:** ${COMMISSION_PER_SHARE:.4f}\n"
                f"- **Slippage per share:** ${slippage_per_share:.4f}\n"
                f"- **Costs per round trip:** commission+slippage = "
                f"${2*(COMMISSION_PER_SHARE+slippage_per_share)*SHARES_PER_TRADE:,.2f}\n"
                f"- **Price assumption:** Projections use **AVG_ENTRY_PRICE=${AVG_ENTRY_PRICE:.0f}** for risk sizing and notional.\n\n"
            )

            # Scaling table (recommended strategy only)
            f.write("### Position Scaling (Recommended Strategy)\n\n")
            f.write("| Shares | Net P&L / Year | Notional / Trade |\n")
            f.write("|---:|---:|---:|\n")
            for shares in [1, 10, 25, 50, 100, 200, 500]:
                scale = shares / SHARES_PER_TRADE
                f.write(
                    f"| {shares:,} | ${rec_pnl_year*scale:,.0f} | ${AVG_ENTRY_PRICE*shares:,.0f} |\n"
                )
            f.write("\n")

            f.write("### Summary\n\n")
            f.write(
                "Max-P&L selection tends to move toward wider stops because larger stop widths reduce stop-outs and "
                "increase win rate, even as R:R compresses. This choice is objective-dependent: max P&L is not the same "
                "as max EV(R) per trade.\n\n"
            )

            f.write("### Next Steps\n\n")
            f.write(
                "1. Confirm max-P&L stability via walk-forward resampling (to avoid overfitting stop width to one period).\n"
                "2. Add drawdown/volatility stats so P&L can be compared on a risk-adjusted basis.\n"
                "3. Re-estimate selection rate (trades per bar) from the actual test set and remove the static 0.432 shortcut.\n"
                "4. Integrate explicit capital constraints (max concurrent trades / margin) into the P&L projection.\n\n"
            )

        # Optional: Walk-forward table
        if walk_forward_df is not None:
            f.write("---\n\n")
            f.write("## 🔁 Walk-Forward Resampling (Optional)\n\n")
            if len(walk_forward_df) == 0:
                f.write("Walk-forward run requested, but no valid folds were produced (insufficient samples).\n\n")
            else:
                f.write("Fixed stop (Option A): evaluate the same stop width across sequential yearly test folds.\n\n")
                f.write("| Train Years | Test Year | Test Bars (labeled) | Trades | Win Rate | EV (R) | Net P&L |\n")
                f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
                for _, r in walk_forward_df.iterrows():
                    f.write(
                        f"| {r['train_years']} | {int(r['test_year'])} | {int(r['test_bars']):,} | "
                        f"{int(r['trades']):,} | {r['win_rate']*100:.1f}% | {r['ev_r']:+.3f} | ${r['net_pnl']:,.0f} |\n"
                    )
                f.write("\n")

    print(f"✓ Saved markdown summary to: {md_path}")

    return csv_path, md_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RF VWAP reversion master pipeline")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run optional walk-forward evaluation (fixed stop, yearly folds) and append results to the markdown report.",
    )
    parser.add_argument(
        "--wf-start-train-year",
        type=int,
        default=2016,
        help="Walk-forward: first year included in the training window (default: 2016).",
    )
    parser.add_argument(
        "--wf-start-test-year",
        type=int,
        default=2020,
        help="Walk-forward: first test year (default: 2020).",
    )
    parser.add_argument(
        "--wf-end-test-year",
        type=int,
        default=2025,
        help="Walk-forward: last test year (default: 2025).",
    )
    parser.add_argument(
        "--wf-stop-atr",
        type=float,
        default=None,
        help="Walk-forward: stop ATR to evaluate (default: use recommended stop from results).",
    )
    parser.add_argument(
        "--wf-threshold",
        type=float,
        default=0.5,
        help="Walk-forward: RF probability threshold (default: 0.5).",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.01,
        help="Slippage per share in dollars (default: 0.01). Use this to model different slippage scenarios.",
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print(" MASTER PIPELINE: RF VWAP REVERSION ANALYSIS")
    print("="*80)

    # Step 1: Load data
    df = load_and_validate_data(DATA_FILE)

    # Step 2: Calculate indicators
    df = calculate_core_indicators(df)

    # Step 3: Get feature columns
    features = get_feature_columns(df)
    print(f"\n✓ Using {len(features)} features:")
    for i, feat in enumerate(features, 1):
        print(f"  {i:2d}. {feat}")

    # Step 4: Generate labels
    df = generate_all_labels(df)

    # Step 5: Train RF models and calculate metrics
    print(f"\n{'='*80}")
    print("TRAINING RF MODELS & CALCULATING METRICS")
    print(f"{'='*80}\n")

    all_results = []

    for stop_atr in STOP_ATRS:
        print(f"\n--- Stop Width: {stop_atr} ATR ---")

        # Train model
        result = train_rf_model(df, stop_atr, features, TEST_YEAR)

        if result is None:
            print(f"  ⚠ Insufficient data for {stop_atr} ATR")
            continue

        print(f"  ✓ Trained: {result['n_train']:,} train, {result['n_test']:,} test samples")

        # Calculate median distance for R:R
        label_col = f"label_s{stop_atr}".replace(".", "_")
        valid_mask = df[label_col].notna()
        median_dist = df.loc[valid_mask, 'vwap_width_atr'].median()

        # Median realized ATR (dollars/share) in the test period for this stop
        atr_med_test = _median_test_atr_dollars(df, label_col, test_year=TEST_YEAR)        # Calculate EV metrics
        ev_df, rr, breakeven_wr = calculate_ev_metrics(
            result['y_test'], result['proba_test'], stop_atr, median_dist
        )
        
        print(f"  ✓ R:R = {rr:.2f}:1, Breakeven WR = {breakeven_wr*100:.1f}%")

        # Get ATR series for test set (to calculate per-trade P&L)
        # We need the original df_valid used in train_rf_model to align indices
        valid_mask_local = df[label_col].notna()
        df_valid_local = df.loc[valid_mask_local].copy()
        df_valid_local['year'] = pd.to_datetime(df_valid_local['datetime']).dt.year
        test_mask_local = df_valid_local['year'] >= TEST_YEAR
        df_test_local = df_valid_local.loc[test_mask_local].copy()
        
        # Ensure alignment: result['y_test'] and result['proba_test'] should match df_test_local length
        if len(df_test_local) != len(result['y_test']):
            print(f"  ⚠ Warning: test set length mismatch, skipping per-trade P&L")
            continue
        
        atr_test_array = df_test_local['atr'].values

        # Store results for each threshold
        for _, ev_row in ev_df.iterrows():
            thresh = ev_row['threshold']
            mask = result['proba_test'] >= thresh
              # Per-trade P&L using actual ATR at each trade
            y_selected = result['y_test'][mask]
            atr_selected = atr_test_array[mask]
            
            pnl_metrics = calculate_dollar_pnl_pertrade(
                stop_atr=stop_atr,
                rr=rr,
                y_actual=y_selected,
                atr_series=atr_selected,
                slippage_per_share=args.slippage,
            )

            # Raw win rate (for reference)
            raw_wr = ev_df.iloc[0]['win_rate']
            raw_ev = ev_df.iloc[0]['ev']

            all_results.append({
                'stop_atr': stop_atr,
                'rr': rr,
                'breakeven_wr': breakeven_wr,
                'rf_threshold': thresh,
                'n_trades': ev_row['n_trades'],
                'win_rate': ev_row['win_rate'],
                'ev': ev_row['ev'],
                'pct_filtered': ev_row['pct_filtered'],
                'raw_win_rate': raw_wr,
                'raw_ev': raw_ev,
                'avg_risk_dollars': pnl_metrics['avg_risk_dollars'],
                'total_gross_pnl': pnl_metrics['total_gross_pnl'],
                'total_net_pnl': pnl_metrics['total_net_pnl'],
                'total_costs': pnl_metrics['total_costs'],
                'avg_net_pnl_per_trade': pnl_metrics['avg_net_pnl_per_trade'],
                'capital_per_trade': pnl_metrics['capital_per_trade'],
                'return_pct_per_trade': pnl_metrics['return_pct_per_trade'],
            })

            if thresh == 0.5:
                print(f"  → RF≥0.5: WR={ev_row['win_rate']*100:.1f}%, EV={ev_row['ev']:+.3f}R, "
                      f"P&L=${pnl_metrics['total_net_pnl']:,.0f}")

    # Step 6: Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Determine recommended stop (max net P&L at RF≥0.50)
    recommended_stop_atr = None
    if len(results_df) > 0:
        rf_05 = results_df[results_df['rf_threshold'] == 0.5]
        if len(rf_05) > 0:
            recommended_stop_atr = float(rf_05.loc[rf_05['total_net_pnl'].idxmax(), 'stop_atr'])    # Optional walk-forward
    walk_forward_df = None
    if args.walk_forward:
        stop_for_wf = args.wf_stop_atr if args.wf_stop_atr is not None else recommended_stop_atr
        if stop_for_wf is None:
            walk_forward_df = pd.DataFrame()
        else:
            walk_forward_df = walk_forward_resample_fixed_stop(
                df=df,
                stop_atr=stop_for_wf,
                features=features,
                start_train_year=args.wf_start_train_year,
                start_test_year=args.wf_start_test_year,
                end_test_year=args.wf_end_test_year,
                rf_threshold=args.wf_threshold,
                slippage_per_share=args.slippage,
            )
            print("\nWalk-forward results:")
            if len(walk_forward_df) == 0:
                print("  (no valid folds)")
            else:
                for _, r in walk_forward_df.iterrows():
                    print(
                        f"  Test {int(r['test_year'])}: trades={int(r['trades']):,}, "
                        f"WR={r['win_rate']*100:.1f}%, EV={r['ev_r']:+.3f}R, P&L=${r['net_pnl']:,.0f}"
                    )

    # Step 7: Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    # Get top features for summary
    if len(results_df) > 0:
        first_model = train_rf_model(df, STOP_ATRS[0], features, TEST_YEAR)
        if first_model is not None:
            top_features = first_model['importance']['feature'].head(15).tolist()
        else:
            top_features = features[:15]
    else:
        top_features = features[:15]    # Pass both: top features for display and the full feature count for the header
    csv_path, md_path = save_results(results_df, top_features, n_features=len(features), walk_forward_df=walk_forward_df, slippage_per_share=args.slippage)

    # Step 8: Final summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}\n")

    rf_05 = results_df[results_df['rf_threshold'] == 0.5].copy()

    if len(rf_05) == 0:
        print("⚠ No RF≥0.50 results to summarize.")
    else:
        # IMPORTANT: Do NOT sum across stop widths (double-counting). Select a single strategy.
        best_pnl_row = rf_05.loc[rf_05['total_net_pnl'].idxmax()]
        print(f"✓ Analyzed {len(STOP_ATRS)} stop widths")
        print("✓ Recommended (max net P&L, RF≥0.50):")
        print(f"   - Stop: {best_pnl_row['stop_atr']:.2f} ATR")
        print(f"   - Trades (test): {int(best_pnl_row['n_trades']):,}")
        print(f"   - Net P&L (test): ${best_pnl_row['total_net_pnl']:,.0f}")
        print(f"   - Win rate: {best_pnl_row['win_rate']*100:.1f}%")
        print(f"   - EV: {best_pnl_row['ev']:+.3f}R")

    print(f"\n📊 Results saved to:")
    print(f"   - {csv_path}")
    print(f"   - {md_path}")

    return results_df


if __name__ == "__main__":
    results = main()
