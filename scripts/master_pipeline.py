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

# Ensure stdout can handle Unicode on Windows (cp1252 fallback)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(errors='replace')

import argparse

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import torch
import torch.nn as tnn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from label_generator import LabelConfig, generate_labels
from datetime import datetime
from model_persistence import save_model, load_model

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = Path("data/tsla_5min_10years.csv")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

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

# Separate regressor params (can differ slightly)
RF_REG_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_leaf': 50,
    'min_samples_split': 100,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
}

# Logistic regression params (strong L2 regularization to prevent overfit)
LOGISTIC_PARAMS = {
    'C': 0.1,               # Moderate regularisation (was 0.01, too strong/underfitting)
    'penalty': 'l2',
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': 42,
    # class_weight removed — using explicit random oversampling instead
}

# Ridge (L2-regularised linear regression on P&L)
RIDGE_PARAMS = {
    'alpha': 100.0,          # Strong regularisation
    'fit_intercept': True,
}

# Neural network (MLP) classifier params
# 2 hidden layers capture nonlinear feature interactions (like RF) while staying
# within the sklearn API (same predict_proba interface, no GPU required).
# Early stopping + L2 regularisation prevent overfitting on ~50K training bars.
NN_PARAMS = {
    'hidden_layer_sizes': (64, 32),   # Smaller network to reduce overfitting
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.05,            # Much stronger L2 (was 0.001 — massive overfit)
    'batch_size': 256,
    'learning_rate': 'adaptive',
    'learning_rate_init': 0.001,
    'max_iter': 300,
    'early_stopping': True,   # Hold out 10% of training data for validation
    'validation_fraction': 0.1,
    'n_iter_no_change': 15,   # Patience: stop if no improvement for 15 epochs
    'random_state': 42,
}

# PyTorch profit-maximising NN params (nn_pnl mode)
# Uses a custom loss: -mean(action_prob * per_trade_net_pnl) where action_prob = sigmoid(logit).
# The network learns to output high probability for bars where net P&L is positive
# and low probability for bars where net P&L is negative, weighted by magnitude.
NN_PNL_PARAMS = {
    'hidden_sizes': (64, 32),
    'dropout': 0.2,
    'lr': 0.001,
    'weight_decay': 0.01,     # L2 regularisation
    'batch_size': 256,
    'max_epochs': 300,
    'patience': 15,           # Early stopping patience
    'val_fraction': 0.1,
    'random_state': 42,
}

# P&L Configuration
SHARES_PER_TRADE = 100
COMMISSION_PER_SHARE = 0.005  # $0.005/share
SLIPPAGE_PER_SHARE = 0.01     # $0.01/share
AVG_ENTRY_PRICE = 400.0       # TSLA average price (used for notional/risk sizing)
CAPITAL_CAP = 1_000_000       # Maximum capital available ($1M)
DEFAULT_RISK_PER_TRADE = 0.01 # 1% of capital risked per trade when using risk-based sizing

# IBKR Margin / Borrowing Cost Configuration
# Rates are annual, prorated to actual hold duration per trade.
# IBKR Pro tiered rates (USD): benchmark (Fed Funds) + spread.
# We store (upper_bound, annual_rate) pairs sorted ascending.
IBKR_MARGIN_RATE_TIERS = [
    (100_000,       0.0683),   # <= $100K borrowed: BM + 1.50% ≈ 6.83%
    (1_000_000,     0.0633),   # $100K-$1M:        BM + 1.00% ≈ 6.33%
    (50_000_000,    0.0608),   # $1M-$50M:         BM + 0.75% ≈ 6.08%
    (200_000_000,   0.0583),   # $50M-$200M:       BM + 0.50% ≈ 5.83%
    (float('inf'),  0.0558),   # >$200M:           BM + 0.25% ≈ 5.58%
]
IBKR_DAYS_PER_YEAR = 360      # IBKR uses 360-day convention for interest


def ibkr_margin_cost(borrowed: float, hold_hours: float) -> float:
    """Compute IBKR tiered margin interest for a given borrowed amount and hold duration.

    Args:
        borrowed: Dollar amount borrowed on margin (notional - cash). Must be >= 0.
        hold_hours: Duration the position is held, in hours.

    Returns:
        Interest cost in dollars (always >= 0).
    """
    if borrowed <= 0 or hold_hours <= 0:
        return 0.0
    remaining = borrowed
    total_interest = 0.0
    prev_bound = 0.0
    for upper_bound, rate in IBKR_MARGIN_RATE_TIERS:
        tier_capacity = upper_bound - prev_bound
        tier_amount = min(remaining, tier_capacity)
        if tier_amount <= 0:
            break
        # annual_rate / 360 days / 24 hours * hold_hours
        total_interest += tier_amount * rate * hold_hours / (IBKR_DAYS_PER_YEAR * 24.0)
        remaining -= tier_amount
        prev_bound = upper_bound
    return total_interest


# RF Threshold to analyze
RF_THRESHOLDS = [0.5, 0.55, 0.6, 0.65]

# ============================================================================
# SETUP FILTER DEFAULTS
# ============================================================================
# Only bars matching these criteria are considered "proper reversion setups".
# Non-setup bars are excluded from both training and evaluation entirely —
# they are not labeled 0, they simply don't exist in the dataset.

SETUP_DEFAULTS = {
    'min_dist_atr': 0.5,      # Minimum distance from VWAP in ATR units
    'min_minutes_session': 15, # Minimum minutes into session (VWAP needs to stabilize)
    'max_minutes_session': 360,# Maximum minutes into session (need time for reversion)
    'min_rr_setup': 1.0,      # Minimum R:R (vwap_width_atr / stop_atr) for the trade to qualify
}

# ============================================================================
# PYTORCH PROFIT-MAXIMISING NN (nn_pnl mode)
# ============================================================================

class PnLNet(tnn.Module):
    """Small MLP that outputs a single logit (sigmoid → trade probability).

    The network is trained with a *profit-aware* loss:
        loss = -mean( sigma(logit) * net_pnl_per_bar )  +  L2

    This directly optimises for expected dollar P&L, not classification accuracy.
    Bars with large positive net_pnl pull the output toward 1; bars with large
    negative net_pnl push it toward 0 — weighted by magnitude.
    """

    def __init__(self, n_features: int, hidden_sizes=(64, 32), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_sizes:
            layers.append(tnn.Linear(prev, h))
            layers.append(tnn.ReLU())
            layers.append(tnn.Dropout(dropout))
            prev = h
        layers.append(tnn.Linear(prev, 1))  # single logit output
        self.net = tnn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # shape: (batch,)


class PnLModelWrapper:
    """Sklearn-like wrapper around PnLNet so it plugs into the existing pipeline.

    Exposes predict_proba(X) → array of shape (n, 2) matching MLPClassifier API.
    Also exposes a dummy fit() so sklearn's permutation_importance accepts it.
    """

    def __init__(self, net: PnLNet, scaler: StandardScaler, features: list):
        self.net = net
        self.scaler = scaler
        self.features = features

    def fit(self, X, y=None):
        """No-op — model is already trained. Required by sklearn validation."""
        return self

    def predict_proba(self, X):
        self.net.eval()
        if isinstance(X, pd.DataFrame):
            X_np = X[self.features].values
        else:
            X_np = np.asarray(X)
        X_scaled = self.scaler.transform(X_np)
        with torch.no_grad():
            logits = self.net(torch.tensor(X_scaled, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy()
        # Return (n, 2) array: [P(loss), P(win)]
        return np.column_stack([1 - probs, probs])


def _train_nn_pnl(
    X_train: np.ndarray,
    pnl_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    pnl_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    params: dict,
) -> PnLNet:
    """Train PnLNet with profit-maximising loss + early stopping on val P&L.

    Loss = -mean( sigmoid(logit_i) * pnl_i )
    where pnl_i = net dollar P&L for bar i (positive for winners, negative for losers).

    The network learns to assign high probability to profitable trades and
    low probability to losing trades, *weighted by how much they win or lose*.
    """
    torch.manual_seed(params.get('random_state', 42))
    np.random.seed(params.get('random_state', 42))

    hidden = params.get('hidden_sizes', (64, 32))
    dropout = params.get('dropout', 0.2)
    lr = params.get('lr', 0.001)
    wd = params.get('weight_decay', 0.01)
    batch_size = params.get('batch_size', 256)
    max_epochs = params.get('max_epochs', 300)
    patience = params.get('patience', 15)

    net = PnLNet(n_features, hidden_sizes=hidden, dropout=dropout)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    # Normalise pnl magnitudes to stabilise gradients (zero-mean, unit-variance)
    pnl_mean = float(np.mean(pnl_train))
    pnl_std = float(np.std(pnl_train)) or 1.0
    pnl_train_norm = (pnl_train - pnl_mean) / pnl_std
    pnl_val_norm = (pnl_val - pnl_mean) / pnl_std

    # Build datasets
    t_X = torch.tensor(X_train, dtype=torch.float32)
    t_pnl = torch.tensor(pnl_train_norm, dtype=torch.float32)
    dataset = TensorDataset(t_X, t_pnl)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    t_X_val = torch.tensor(X_val, dtype=torch.float32)
    t_pnl_val = torch.tensor(pnl_val_norm, dtype=torch.float32)

    best_val_metric = -np.inf
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(max_epochs):
        net.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, pnl_batch in loader:
            optimizer.zero_grad()
            logits = net(X_batch)
            probs = torch.sigmoid(logits)
            # Profit-maximising loss: maximise expected P&L
            loss = -torch.mean(probs * pnl_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation: measure *actual* expected P&L on val set (un-normalised)
        net.eval()
        with torch.no_grad():
            val_logits = net(t_X_val)
            val_probs = torch.sigmoid(val_logits).numpy()
            # Val metric: sum of (prob * raw_pnl) — proxy for expected dollar P&L
            val_expected_pnl = float(np.sum(val_probs * pnl_val))

        if val_expected_pnl > best_val_metric:
            best_val_metric = val_expected_pnl
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Restore best model
    if best_state is not None:
        net.load_state_dict(best_state)

    n_epochs = epoch + 1
    print(f"  [NN_PNL] Stopped at epoch {n_epochs}, best epoch {best_epoch}, "
          f"best val E[PnL]=${best_val_metric:,.0f}")

    return net


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def apply_setup_filter(
    df: pd.DataFrame,
    stop_atr: float,
    min_dist_atr: float = 0.5,
    min_minutes_session: int = 15,
    max_minutes_session: int = 360,
    min_rr_setup: float = 1.0,
) -> pd.Series:
    """Return a boolean mask identifying bars that qualify as proper reversion setups.

    A "setup" requires ALL of the following:
      1. Price is meaningfully extended from VWAP (vwap_width_atr >= min_dist_atr).
      2. Session has been open long enough for VWAP to stabilize
         (minutes_into_session >= min_minutes_session).
      3. Enough session time remains for reversion to play out
         (minutes_into_session <= max_minutes_session).
      4. Per-trade R:R is attractive (vwap_width_atr / stop_atr >= min_rr_setup).

    Bars that fail ANY criterion are excluded from training and evaluation.
    This focuses the RF on bars where a discretionary trader would actually
    consider entering a VWAP reversion trade.
    """
    mask = pd.Series(True, index=df.index)

    # 1. Minimum distance from VWAP
    if 'vwap_width_atr' in df.columns and min_dist_atr > 0:
        mask &= df['vwap_width_atr'] >= min_dist_atr

    # 2. VWAP stabilisation — skip first N minutes of session
    if 'minutes_into_session' in df.columns and min_minutes_session > 0:
        mask &= df['minutes_into_session'] >= min_minutes_session

    # 3. Enough time remaining for reversion
    if 'minutes_into_session' in df.columns and max_minutes_session < 390:
        mask &= df['minutes_into_session'] <= max_minutes_session

    # 4. Minimum R:R for THIS stop level
    if 'vwap_width_atr' in df.columns and min_rr_setup > 0 and stop_atr > 0:
        mask &= (df['vwap_width_atr'] / stop_atr) >= min_rr_setup

    return mask


def load_and_validate_data(filepath):
    """Load data and ensure minimum bar count."""
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    df = pd.read_csv(filepath)
    # Handle time column
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'], utc=True)
        df['date'] = df['datetime'].dt.date  # Date-only for label generation
    elif df.index.name == 'time':
        df = df.reset_index()
        df['datetime'] = pd.to_datetime(df['time'], utc=True)
        df['date'] = df['datetime'].dt.date
    else:
        raise ValueError("No 'time' column found")
    
    # Replace unicode checkmark with ASCII to avoid cp1252 console encoding issues on Windows
    print(f"[OK] Loaded {len(df):,} bars")
    print(f"[OK] Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"[OK] Data span: {(df['datetime'].max() - df['datetime'].min()).days} days")
    
    if len(df) < 100000:
        print(f"\n[WARNING] Only {len(df):,} bars (< 100K minimum)")
        print("Consider using a larger dataset for robust analysis")
    else:
        print(f"[OK] Dataset exceeds 100K bars requirement")
    
    return df


def calculate_core_indicators(df, verbose: bool = True):
    """Calculate only essential, non-redundant indicators.

    Args:
        df: Input OHLCV dataframe
        verbose: If False, suppress all progress printing (useful for streaming)
    """
    # NOTE: All printing in this function must be guarded by `if verbose:`.
    if verbose:
        print(f"\n{'='*80}")
        print("CALCULATING INDICICATORS" if False else "CALCULATING INDICATORS")
        print(f"{'='*80}")

    df = df.copy()

    # ========================================
    # 1. ATR (14-period)
    # ========================================
    if verbose:
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
    if verbose:
        print("Calculating VWAP...")
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    pv = typical_price * df['volume']
    
    df['vwap'] = df.groupby('date').apply(
        lambda g: (pv.loc[g.index].cumsum() / df.loc[g.index, 'volume'].cumsum())
    ).reset_index(level=0, drop=True)
    
    # ========================================
    # 3. VWAP Distance Metrics
    # ========================================
    if verbose:
        print("Calculating VWAP distance metrics...")
    df['vwap_width_atr'] = abs(df['close'] - df['vwap']) / df['atr']
    df['price_to_vwap_atr'] = (df['close'] - df['vwap']) / df['atr']  # Signed
    df['is_long_setup'] = df['close'] < df['vwap']

    # Z-score of VWAP stretch: how unusual is the current distance vs recent history
    # Uses rolling 60-bar (5 hours) window of absolute VWAP distance in ATR units
    _stretch_roll = df['vwap_width_atr'].rolling(60, min_periods=20)
    df['vwap_stretch_zscore'] = (
        (df['vwap_width_atr'] - _stretch_roll.mean()) / _stretch_roll.std().replace(0, np.nan)
    ).fillna(0)
    
    # ========================================
    # 4. VWAP Dynamics
    # ========================================
    if verbose:
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
    if verbose:
        print("Calculating volume metrics...")
    df['rel_vol'] = df['volume'] / df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['volume'].shift(1)
      # Volume at extension (current volume relative to nearby bars)
    df['vol_at_extension'] = df['volume'] / df['volume'].rolling(5).mean()
    
    # ========================================
    # 6. Momentum Indicators
    # ========================================
    if verbose:
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
    if verbose:
        print("Calculating bar context...")
    df['bar_range_atr'] = (df['high'] - df['low']) / df['atr']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
      # Time since crossed VWAP
    df['crossed_vwap'] = (df['is_long_setup'] != df['is_long_setup'].shift(1)).astype(int)
    df['bars_from_vwap'] = df.groupby((df['crossed_vwap'] == 1).cumsum()).cumcount()

    # ========================================
    # 8. ENHANCED FEATURES (time, session, trend)
    # ========================================
    if verbose:
        print("Calculating enhanced features (time/session/trend)...")

    # 8a. Time-of-day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['minutes_into_session'] = ((df['hour'] - 9) * 60 + df['minute'] - 30).clip(0, 390)
    df['session_phase'] = pd.cut(
        df['minutes_into_session'], bins=[-1, 30, 120, 270, 391],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # 8b. Daily cumulative volume profile
    df['cum_vol_today'] = df.groupby('date')['volume'].cumsum()
    df['total_vol_today'] = df.groupby('date')['volume'].transform('sum')
    df['vol_pct_complete'] = df['cum_vol_today'] / df['total_vol_today']

    # 8c. VWAP crosses today
    df['vwap_crosses_today'] = df.groupby('date')['crossed_vwap'].cumsum()

    # 8d. Day range context
    df['day_high'] = df.groupby('date')['high'].cummax()
    df['day_low'] = df.groupby('date')['low'].cummin()
    df['day_range_atr'] = (df['day_high'] - df['day_low']) / df['atr']
    df['pct_of_day_range'] = np.where(
        df['day_high'] > df['day_low'],
        (df['close'] - df['day_low']) / (df['day_high'] - df['day_low']),
        0.5
    )

    # 8e. VWAP position within day range
    df['vwap_in_day_range'] = np.where(
        df['day_high'] > df['day_low'],
        (df['vwap'] - df['day_low']) / (df['day_high'] - df['day_low']),
        0.5
    )

    # 8f. Momentum into extension
    df['momentum_3bar_atr'] = (df['close'] - df['close'].shift(3)) / df['atr']
    df['momentum_6bar_atr'] = (df['close'] - df['close'].shift(6)) / df['atr']

    # 8g. Bar-level mean reversion signal
    df['bar_reverting'] = np.where(
        df['is_long_setup'],
        (df['close'] > df['open']).astype(int),
        (df['close'] < df['open']).astype(int),
    )

    # 8h. Consecutive bars on same side of VWAP
    df['consecutive_same_side'] = df.groupby(
        (df['is_long_setup'] != df['is_long_setup'].shift(1)).cumsum()
    ).cumcount() + 1

    # 8i. Open-to-VWAP distance
    daily_open = df.groupby('date')['open'].transform('first')
    df['open_vs_vwap_atr'] = (daily_open - df['vwap']) / df['atr']

    # 8j. Prior bar toward VWAP
    df['prior_bar_toward_vwap'] = np.where(
        df['is_long_setup'],
        (df['close'].shift(1) > df['close'].shift(2)).astype(float),
        (df['close'].shift(1) < df['close'].shift(2)).astype(float),
    )

    # 8k. EMA20 trend context
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema20_slope_atr'] = (df['ema20'] - df['ema20'].shift(5)) / df['atr']

    # 8l. Extension speed
    df['extension_speed'] = df['vwap_width_atr'] / (df['bars_from_vwap'] + 1)

    # ========================================
    # 8m–8z+ REVERSAL-SPECIFIC FEATURES
    # ========================================
    if verbose:
        print("Calculating reversal-specific features...")

    # 8m. Bollinger Band context (20-period, 2σ)
    #   - How far price is from its own mean relative to volatility
    #   - z_score > 2 = statistically overextended
    bb_mean = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_z_score'] = (df['close'] - bb_mean) / bb_std.replace(0, np.nan)
    df['bb_width_atr'] = (2.0 * bb_std) / df['atr']  # band width normalised

    # 8n. VWAP standard deviation bands
    #   - Institutional-level overextension: how many σ from VWAP
    #   - Computed intraday: rolling σ of (close - vwap) within each session
    vwap_dev = df['close'] - df['vwap']
    vwap_dev_std = df.groupby('date')['close'].transform(
        lambda x: (x - x.expanding().mean()).expanding().std()
    )
    # Fallback: use rolling 20-bar std of price-to-vwap deviation
    vwap_dev_std_rolling = vwap_dev.rolling(20).std()
    vwap_std_final = vwap_dev_std.fillna(vwap_dev_std_rolling)
    df['vwap_sigma'] = (vwap_dev / vwap_std_final.replace(0, np.nan)).fillna(0)

    # 8o. Wicking / rejection signals (pin bar detection)
    #   - Upper/lower wick ratios detect rejection from extremes
    bar_range = (df['high'] - df['low']).replace(0, np.nan)
    body_top = df[['open', 'close']].max(axis=1)
    body_bot = df[['open', 'close']].min(axis=1)
    df['upper_wick_pct'] = (df['high'] - body_top) / bar_range
    df['lower_wick_pct'] = (body_bot - df['low']) / bar_range
    df['body_pct'] = (body_top - body_bot) / bar_range

    # Rejection wick: wick pointing away from VWAP (toward extension)
    #   Long setup (price below VWAP): lower wick = rejection of further downside
    #   Short setup (price above VWAP): upper wick = rejection of further upside
    df['rejection_wick_pct'] = np.where(
        df['is_long_setup'],
        df['lower_wick_pct'],   # long: lower wick rejects more downside
        df['upper_wick_pct'],   # short: upper wick rejects more upside
    )

    # 8p. Volume climax / exhaustion
    #   - Volume spike at extension suggests capitulation / exhaustion
    df['vol_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std().replace(0, np.nan)
    df['vol_climax'] = (df['vol_zscore'] > 2.0).astype(int)

    # Volume declining while extended = momentum drying up
    df['vol_declining_3bar'] = (
        (df['volume'] < df['volume'].shift(1)) &
        (df['volume'].shift(1) < df['volume'].shift(2))
    ).astype(int)

    # 8q. ATR expansion / contraction regime
    #   - Expanding ATR = trending, contracting = mean-reverting environment
    df['atr_ratio_5'] = df['atr'] / df['atr'].shift(5)
    df['atr_ratio_20'] = df['atr'] / df['atr'].shift(20)
    df['atr_regime'] = np.where(df['atr_ratio_5'] > 1.1, 1,    # expanding
                       np.where(df['atr_ratio_5'] < 0.9, -1, 0)) # contracting / neutral

    # 8r. Higher-timeframe trend context (simulated via longer lookback)
    #   - 60-bar EMA ≈ 5-hour trend on 5-min bars
    #   - 240-bar EMA ≈ 1-day trend
    df['ema60'] = df['close'].ewm(span=60).mean()
    df['ema60_slope_atr'] = (df['ema60'] - df['ema60'].shift(10)) / df['atr']
    df['price_vs_ema60_atr'] = (df['close'] - df['ema60']) / df['atr']

    # Trend alignment: is reversion direction aligned with higher-TF trend?
    #   Long reversion (buy) aligned with uptrend = good
    #   Short reversion (sell) aligned with downtrend = good
    df['htf_trend_aligned'] = np.where(
        df['is_long_setup'],
        (df['ema60_slope_atr'] > 0).astype(int),    # long + uptrend
        (df['ema60_slope_atr'] < 0).astype(int),     # short + downtrend
    )

    # 8s. Stochastic oscillator (14,3,3) — short-term overbought/oversold
    stoch_low = df['low'].rolling(14).min()
    stoch_high = df['high'].rolling(14).max()
    stoch_range = (stoch_high - stoch_low).replace(0, np.nan)
    df['stoch_k'] = ((df['close'] - stoch_low) / stoch_range * 100).fillna(50)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # Stochastic favoring reversion:
    #   Long setup: stoch_k < 20 (oversold) → favorable
    #   Short setup: stoch_k > 80 (overbought) → favorable
    df['stoch_reversal_signal'] = np.where(
        df['is_long_setup'],
        (df['stoch_k'] < 20).astype(int),
        (df['stoch_k'] > 80).astype(int),
    )

    # 8t. Reversion velocity — is price already pulling back toward VWAP?
    #   Positive = moving toward VWAP, negative = extending further
    df['reversion_velocity_1'] = np.where(
        df['is_long_setup'],
        (df['close'] - df['close'].shift(1)) / df['atr'],   # long: positive = up toward VWAP
        (df['close'].shift(1) - df['close']) / df['atr'],    # short: positive = down toward VWAP
    )
    df['reversion_velocity_3'] = np.where(
        df['is_long_setup'],
        (df['close'] - df['close'].shift(3)) / df['atr'],
        (df['close'].shift(3) - df['close']) / df['atr'],
    )

    # 8u. Max excursion from VWAP today — how far has price gone beyond current level?
    #   Helps distinguish "first touch of extreme" from "already been there, bounced, came back"
    df['max_dist_from_vwap_today'] = df.groupby('date')['vwap_width_atr'].cummax()
    df['pct_of_max_extension'] = df['vwap_width_atr'] / df['max_dist_from_vwap_today'].replace(0, np.nan)

    # 8v. Extension deceleration (2nd derivative of distance from VWAP)
    #   - If extension is decelerating, reversal more likely
    dist_delta_1 = df['vwap_width_atr'].diff(1)
    dist_delta_2 = df['vwap_width_atr'].diff(2)
    df['extension_accel'] = dist_delta_1.diff(1)    # 2nd derivative
    df['extension_decelerating'] = (
        (dist_delta_1 < dist_delta_2) &  # rate of extension slowing
        (df['vwap_width_atr'] > df['vwap_width_atr'].shift(1))  # but still extended
    ).astype(int)    # 8w. Prior day context
    #   - Gap, prior day range, prior close vs prior VWAP

    # Shift by 1 day to get "prior day" values
    # Use a day-level mapping to avoid look-ahead
    daily_open_price = df.groupby('date')['open'].transform('first')
    day_stats = df.groupby('date').agg(
        prev_close=('close', 'last'),
        prev_range_high=('high', 'max'),
        prev_range_low=('low', 'min'),
        prev_vwap=('vwap', 'last'),
        prev_atr=('atr', 'last'),
    )
    day_stats['prev_day_range'] = day_stats['prev_range_high'] - day_stats['prev_range_low']
    day_stats = day_stats.shift(1)  # shift so today maps to yesterday's stats

    df = df.merge(day_stats, left_on='date', right_index=True, how='left')
    df['gap_atr'] = (daily_open_price - df['prev_close']) / df['atr']
    df['prior_day_range_atr'] = df['prev_day_range'] / df['atr']
    df['prior_close_vs_prior_vwap_atr'] = (df['prev_close'] - df['prev_vwap']) / df['atr']    # ========================================
    # 8w2. SUPPORT / RESISTANCE FRAMEWORK
    # ========================================
    # Comprehensive S/R detection with 5 sources:
    #   1) VWAP sigma bands (intraday fair value)
    #   2) Prior day levels (PDH/PDL/PDC/prior VWAP)
    #   3) Opening range + session extremes
    #   4) Swing pivot zones (fractal-based, clustered)
    #   5) Approximate volume profile (POC, VAH/VAL)
    # Plus 3-part reversal quality signals (location + rejection + trigger)
    if verbose:
        print("Calculating S/R framework (5 sources + reversal quality)...")

    # ------------------------------------------------------------------
    # 1) VWAP SIGMA BANDS
    # ------------------------------------------------------------------
    # vwap_sigma already computed above (section 8n): how many σ from VWAP
    # Add band-relative features: are we at/beyond key band levels?
    df['beyond_vwap_1sigma'] = (df['vwap_sigma'].abs() > 1.0).astype(int)
    df['beyond_vwap_2sigma'] = (df['vwap_sigma'].abs() > 2.0).astype(int)
    # Distance to nearest sigma band (fractional: 0 = on a band, 0.5 = halfway)
    vwap_sigma_abs = df['vwap_sigma'].abs()
    df['dist_to_nearest_vwap_band'] = vwap_sigma_abs - vwap_sigma_abs.round(0)

    # ------------------------------------------------------------------
    # 2) PRIOR DAY LEVELS (institutional memory)
    # ------------------------------------------------------------------
    # prev_range_high/low already computed. Add prior day close & prior VWAP.
    df['dist_prior_high_atr'] = (df['close'] - df['prev_range_high']) / df['atr']
    df['dist_prior_low_atr'] = (df['close'] - df['prev_range_low']) / df['atr']
    df['dist_prior_close_atr'] = (df['close'] - df['prev_close']) / df['atr']
    df['dist_prior_vwap_atr'] = (df['close'] - df['prev_vwap']) / df['atr']
    df['dist_open_atr'] = (df['close'] - daily_open_price) / df['atr']

    # Sweep detection: did price break PDH/PDL then close back inside?
    # "Liquidity grab" — breaks level by small amount then reverses
    df['swept_pdh'] = (
        (df['high'] > df['prev_range_high']) &
        (df['close'] <= df['prev_range_high'])
    ).astype(int)
    df['swept_pdl'] = (
        (df['low'] < df['prev_range_low']) &
        (df['close'] >= df['prev_range_low'])
    ).astype(int)
    # Directional: swept the level relevant to our setup direction
    df['swept_key_level'] = np.where(
        df['is_long_setup'],
        df['swept_pdl'],    # long: swept support = liquidity grab below
        df['swept_pdh'],    # short: swept resistance = liquidity grab above
    )

    # ------------------------------------------------------------------
    # 3) OPENING RANGE + SESSION EXTREMES
    # ------------------------------------------------------------------
    # Opening range = first 3 bars (15 min on 5-min data)
    def _opening_range(g):
        first_3 = g.head(3)
        orh = first_3['high'].max()
        orl = first_3['low'].min()
        return pd.Series({'or_high': orh, 'or_low': orl}, dtype=float)

    or_stats = df.groupby('date').apply(_opening_range, include_groups=False)
    df = df.merge(or_stats, left_on='date', right_index=True, how='left')

    df['dist_or_high_atr'] = (df['close'] - df['or_high']) / df['atr']
    df['dist_or_low_atr'] = (df['close'] - df['or_low']) / df['atr']
    # Are we inside or outside the opening range?
    df['outside_or'] = (
        (df['close'] > df['or_high']) | (df['close'] < df['or_low'])
    ).astype(int)

    # Session extremes (already have day_high / day_low from section 8d)
    df['dist_day_high_atr'] = (df['close'] - df['day_high']) / df['atr']
    df['dist_day_low_atr'] = (df['close'] - df['day_low']) / df['atr']
    # At session extreme: are we making new highs/lows right now?
    df['at_session_extreme'] = (
        (df['dist_day_high_atr'].abs() < 0.1) |
        (df['dist_day_low_atr'].abs() < 0.1)
    ).astype(int)

    # ------------------------------------------------------------------
    # 4) SWING PIVOT ZONES (fractal-based, clustered)
    # ------------------------------------------------------------------
    # Detect swing highs/lows using fractals (n=3: bar must be highest/lowest
    # among 7 bars centered on it). Then find distance to nearest recent pivot.
    n_fractal = 3  # look back/forward 3 bars

    # Swing highs: high[i] > max(high[i-n:i], high[i+1:i+n+1])
    roll_max_left = df['high'].rolling(n_fractal, min_periods=n_fractal).max().shift(1)
    roll_max_right = df['high'].iloc[::-1].rolling(n_fractal, min_periods=n_fractal).max().shift(1).iloc[::-1]
    swing_high_mask = (df['high'] > roll_max_left) & (df['high'] > roll_max_right)
    df['_swing_high_price'] = np.where(swing_high_mask, df['high'], np.nan)

    # Swing lows
    roll_min_left = df['low'].rolling(n_fractal, min_periods=n_fractal).min().shift(1)
    roll_min_right = df['low'].iloc[::-1].rolling(n_fractal, min_periods=n_fractal).min().shift(1).iloc[::-1]
    swing_low_mask = (df['low'] < roll_min_left) & (df['low'] < roll_min_right)
    df['_swing_low_price'] = np.where(swing_low_mask, df['low'], np.nan)

    # Forward-fill the last N swing pivots so each bar knows the recent levels
    # Use last 20 swing highs/lows (covers ~100 bars of history)
    # Distance to nearest recent swing high (resistance above) and swing low (support below)
    last_swing_high = df['_swing_high_price'].ffill(limit=60)  # ~5 hours lookback
    last_swing_low = df['_swing_low_price'].ffill(limit=60)

    df['dist_swing_high_atr'] = (df['close'] - last_swing_high) / df['atr']
    df['dist_swing_low_atr'] = (df['close'] - last_swing_low) / df['atr']

    # Nearest swing level (absolute distance)
    df['nearest_swing_atr'] = pd.DataFrame({
        'sh': df['dist_swing_high_atr'].abs(),
        'sl': df['dist_swing_low_atr'].abs(),
    }).min(axis=1)

    # Did we just break a swing level? (sweep + potential reversal)
    df['broke_swing_high'] = (
        (df['high'] > last_swing_high) & (df['close'] <= last_swing_high)
    ).astype(int).fillna(0)
    df['broke_swing_low'] = (
        (df['low'] < last_swing_low) & (df['close'] >= last_swing_low)
    ).astype(int).fillna(0)

    # Clean up temp columns
    df.drop(columns=['_swing_high_price', '_swing_low_price'], inplace=True)

    # ------------------------------------------------------------------
    # 5) APPROXIMATE VOLUME PROFILE (POC proxy)
    # ------------------------------------------------------------------
    # True volume profile needs tick data. Approximate using 5-min bars:
    # POC = price level with highest cumulative volume today (discretized to ATR/4 bins)
    # VAH/VAL = levels containing 70% of volume
    def _volume_profile(g):
        """Compute POC and value area for a single day's bars."""
        prices = g['close'].values
        volumes = g['volume'].values
        atr_val = g['atr'].iloc[-1] if len(g) > 0 else 1.0
        if atr_val <= 0 or np.isnan(atr_val):
            atr_val = 1.0
        bin_size = atr_val / 4.0  # bin width = ATR/4

        if len(prices) == 0 or bin_size <= 0:
            return pd.Series({'poc': np.nan, 'vah': np.nan, 'val': np.nan}, dtype=float)

        # Discretize prices into bins
        price_min = prices.min()
        bins = ((prices - price_min) / bin_size).astype(int)
        # Accumulate volume per bin
        vol_by_bin = {}
        for b, v in zip(bins, volumes):
            vol_by_bin[b] = vol_by_bin.get(b, 0) + v

        # POC = bin with max volume
        poc_bin = max(vol_by_bin, key=vol_by_bin.get)
        poc_price = price_min + (poc_bin + 0.5) * bin_size

        # Value area: expand from POC until 70% of total volume
        total_vol = sum(vol_by_bin.values())
        if total_vol == 0:
            return pd.Series({'poc': poc_price, 'vah': poc_price, 'val': poc_price}, dtype=float)

        va_vol = vol_by_bin[poc_bin]
        lo_bin, hi_bin = poc_bin, poc_bin
        all_bins = sorted(vol_by_bin.keys())
        while va_vol / total_vol < 0.70:
            # Expand in direction with more volume
            next_lo = lo_bin - 1 if lo_bin - 1 in vol_by_bin else None
            next_hi = hi_bin + 1 if hi_bin + 1 in vol_by_bin else None
            lo_vol = vol_by_bin.get(lo_bin - 1, 0) if next_lo is not None else 0
            hi_vol = vol_by_bin.get(hi_bin + 1, 0) if next_hi is not None else 0
            if lo_vol == 0 and hi_vol == 0:
                break
            if hi_vol >= lo_vol:
                hi_bin += 1
                va_vol += hi_vol
            else:
                lo_bin -= 1
                va_vol += lo_vol

        vah = price_min + (hi_bin + 1) * bin_size
        val_price = price_min + lo_bin * bin_size

        return pd.Series({'poc': poc_price, 'vah': vah, 'val': val_price}, dtype=float)

    # Compute developing volume profile (expanding window within each day)
    # For performance, compute once per day using all bars up to current point
    # Approximation: use full-day profile (slight lookahead for early bars, but
    # the features are cross-sectional within a day so the model can't exploit it
    # across days). For strict no-lookahead, we'd need cumulative profile.
    vp_daily = df.groupby('date').apply(_volume_profile, include_groups=False)
    df = df.merge(vp_daily, left_on='date', right_index=True, how='left', suffixes=('', '_vp'))

    df['dist_poc_atr'] = (df['close'] - df['poc']) / df['atr']
    df['dist_vah_atr'] = (df['close'] - df['vah']) / df['atr']
    df['dist_val_atr'] = (df['close'] - df['val']) / df['atr']
    # Are we inside or outside the value area?
    df['outside_value_area'] = (
        (df['close'] > df['vah']) | (df['close'] < df['val'])
    ).astype(int)

    # ------------------------------------------------------------------
    # COMPOSITE S/R: nearest level from ALL sources + levels blocking reversion
    # ------------------------------------------------------------------
    all_sr_dists = pd.DataFrame({
        'prior_high': df['dist_prior_high_atr'].abs(),
        'prior_low': df['dist_prior_low_atr'].abs(),
        'prior_close': df['dist_prior_close_atr'].abs(),
        'open': df['dist_open_atr'].abs(),
        'or_high': df['dist_or_high_atr'].abs(),
        'or_low': df['dist_or_low_atr'].abs(),
        'swing_high': df['dist_swing_high_atr'].abs(),
        'swing_low': df['dist_swing_low_atr'].abs(),
        'poc': df['dist_poc_atr'].abs(),
    })
    df['nearest_sr_atr'] = all_sr_dists.min(axis=1)

    # Count S/R levels between price and VWAP (blocking reversion path)
    close_col = df['close']
    vwap_col = df['vwap']
    lo = pd.DataFrame({'close': close_col, 'vwap': vwap_col}).min(axis=1)
    hi = pd.DataFrame({'close': close_col, 'vwap': vwap_col}).max(axis=1)

    sr_level_prices = pd.DataFrame({
        'pdh': df['prev_range_high'],
        'pdl': df['prev_range_low'],
        'pdc': df['prev_close'],
        'open': daily_open_price,
        'or_high': df['or_high'],
        'or_low': df['or_low'],
        'swing_high': last_swing_high,
        'swing_low': last_swing_low,
        'poc': df['poc'],
    })
    sr_between_count = pd.Series(0, index=df.index)
    for col_name in sr_level_prices.columns:
        lvl = sr_level_prices[col_name]
        sr_between_count += ((lvl > lo) & (lvl < hi)).astype(int)
    df['sr_levels_between'] = sr_between_count

    # ------------------------------------------------------------------
    # 3-PART REVERSAL QUALITY SIGNALS
    # ------------------------------------------------------------------
    if verbose:
        print("Calculating reversal quality signals...")

    # A) LOCATION FILTER — is price at a meaningful S/R confluence?
    #    Score 0-3: how many conditions are met
    loc_at_vwap_band = (vwap_sigma_abs > 1.5).astype(int)
    loc_at_prior_day = (
        (df['dist_prior_high_atr'].abs() < 0.3) |
        (df['dist_prior_low_atr'].abs() < 0.3)
    ).astype(int)
    loc_at_swing = (df['nearest_swing_atr'] < 0.3).astype(int)
    loc_outside_value = df['outside_value_area']
    df['sr_location_score'] = loc_at_vwap_band + loc_at_prior_day + loc_at_swing + loc_outside_value

    # B) REJECTION / SWEEP — evidence the level is defended
    #    rejection_wick_pct already computed (section 8o)
    #    swept_key_level computed above
    #    Add: failed breakdown — broke level, couldn't hold for 2+ bars
    df['strong_rejection_wick'] = (df['rejection_wick_pct'] > 0.5).astype(int)
    df['sr_rejection_score'] = (
        df['strong_rejection_wick'] +
        df['swept_key_level'] +
        df['broke_swing_high'] + df['broke_swing_low']
    ).clip(0, 3)

    # C) TRIGGER — objective entry signal
    #    VWAP band reclaim: was beyond ±2σ, now back inside ±1.5σ
    df['vwap_band_reclaim'] = (
        (df['vwap_sigma'].abs().shift(1) > 2.0) &
        (df['vwap_sigma'].abs() <= 1.5)
    ).astype(int)
    #    Engulfing: current bar body engulfs prior bar body (toward VWAP)
    prev_body_top = df[['open', 'close']].max(axis=1).shift(1)
    prev_body_bot = df[['open', 'close']].min(axis=1).shift(1)
    cur_body_top = df[['open', 'close']].max(axis=1)
    cur_body_bot = df[['open', 'close']].min(axis=1)
    engulfing_bullish = (cur_body_bot < prev_body_bot) & (cur_body_top > prev_body_top) & (df['close'] > df['open'])
    engulfing_bearish = (cur_body_bot < prev_body_bot) & (cur_body_top > prev_body_top) & (df['close'] < df['open'])
    df['engulfing_toward_vwap'] = np.where(
        df['is_long_setup'],
        engulfing_bullish.astype(int),
        engulfing_bearish.astype(int),
    )
    df['sr_trigger_score'] = (
        df['vwap_band_reclaim'] + df['engulfing_toward_vwap']
    ).clip(0, 2)

    # COMPOSITE reversal quality: sum of all three (0-8 scale)
    df['reversal_quality'] = df['sr_location_score'] + df['sr_rejection_score'] + df['sr_trigger_score']

    # 8x. VWAP anchoring strength — how flat/stable is VWAP?
    #   - Flat VWAP = strong anchor for reversion; steep = trending, less mean-reverting
    df['vwap_slope_atr'] = df['vwap_slope'].abs() / df['atr']
    df['vwap_stability'] = 1.0 / (1.0 + df['vwap_slope_atr'])  # 1 = perfectly flat, 0 = steep

    # 8y. Consecutive reverting bars (price moving toward VWAP)
    df['consec_reverting'] = df.groupby(
        (df['bar_reverting'] != df['bar_reverting'].shift(1)).cumsum()
    ).cumcount() + 1
    df['consec_reverting'] = np.where(df['bar_reverting'] == 1, df['consec_reverting'], 0)

    # 8z. Spread proxy (WAP vs close — proxy for liquidity/market maker positioning)
    if 'wap' in df.columns:
        df['wap_vs_close_atr'] = (df['wap'] - df['close']) / df['atr']
    else:
        df['wap_vs_close_atr'] = 0.0

    # ========================================
    # 9. R:R Metrics (for each stop width)
    # ========================================
    if verbose:
        print("Calculating R:R metrics...")
    for stop_atr in STOP_ATRS:
        df[f'rr_{stop_atr}'] = df['vwap_width_atr'] / stop_atr
      # Average R:R across all stops
    rr_cols = [f'rr_{s}' for s in STOP_ATRS]
    df['avg_rr'] = df[rr_cols].mean(axis=1)
    df['min_rr'] = df[rr_cols].min(axis=1)
    df['max_rr'] = df[rr_cols].max(axis=1)
    
    if verbose:
        print(f"[OK] Calculated {len([c for c in df.columns if c not in ['datetime', 'date', 'open', 'high', 'low', 'close', 'volume', 'time']])} features")
    
    return df


def get_feature_columns(df):
    """Get non-redundant feature columns for RF."""
    exclude = [
        'datetime', 'date', 'time', 'year', 'open', 'high', 'low', 'close', 'volume',
        'vwap', 'atr',
        'avg_rr', 'min_rr', 'max_rr',  # Exclude aggregate R:R features (data leakage)
        'cum_vol_today', 'total_vol_today',  # raw volume helpers (use vol_pct_complete instead)
        'day_high', 'day_low',  # raw price helpers (use day_range_atr / pct_of_day_range)
        'ema20',  # raw price (use ema20_slope_atr)
        'ema60',  # raw price (use ema60_slope_atr / price_vs_ema60_atr)
        'bar_count', 'symbol', 'wap',  # metadata columns from raw CSV
        # Prior-day raw helpers (use gap_atr / prior_day_range_atr / prior_close_vs_prior_vwap_atr)
        'prev_close', 'prev_range_high', 'prev_range_low', 'prev_vwap', 'prev_atr', 'prev_day_range',
        'max_dist_from_vwap_today',  # raw helper (use pct_of_max_extension)
        # S/R raw price helpers (use ATR-normalised distance features)
        'or_high', 'or_low',  # opening range raw prices
        'poc', 'vah', 'val',  # volume profile raw prices
        # ----- Reversal features excluded from default set (available for --extra-features) -----
        # Adding these to the 33-feature baseline HURTS performance (curse of dimensionality):
        # The RF spreads its splitting budget over more features, diluting the signal from the
        # core features (vwap_slope, vol_pct_complete, etc.) without adding enough predictive lift.
        # All 31 new features are computed and stored in the DataFrame for future experimentation,
        # but excluded from the default feature set used by get_feature_columns().
        'bb_z_score', 'bb_width_atr',           # Bollinger Band context
        'vwap_sigma',                            # VWAP standard deviation bands
        'upper_wick_pct', 'lower_wick_pct', 'body_pct', 'rejection_wick_pct',  # Wicking
        'vol_zscore', 'vol_climax', 'vol_declining_3bar',  # Volume climax
        'atr_ratio_5', 'atr_ratio_20', 'atr_regime',      # ATR regime
        'ema60_slope_atr', 'price_vs_ema60_atr', 'htf_trend_aligned',  # Higher-TF trend
        'stoch_k', 'stoch_d', 'stoch_reversal_signal',    # Stochastic
        'reversion_velocity_1', 'reversion_velocity_3',    # Reversion velocity
        'pct_of_max_extension',                  # Max excursion
        'extension_accel', 'extension_decelerating',       # Extension deceleration
        'gap_atr', 'prior_day_range_atr', 'prior_close_vs_prior_vwap_atr',  # Prior day
        'vwap_slope_atr', 'vwap_stability',      # VWAP anchoring
        'consec_reverting',                       # Consecutive reverting bars
        'wap_vs_close_atr',                       # Spread proxy
        # S/R sub-components (keep composites, exclude granular)
        'strong_rejection_wick',                 # sub-component of sr_rejection_score
        'swept_pdh', 'swept_pdl',               # sub-components of swept_key_level
        'broke_swing_high', 'broke_swing_low',   # sub-components of sr_rejection_score
        'vwap_band_reclaim', 'engulfing_toward_vwap',  # sub-components of sr_trigger_score
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
        n_valid = int(df_labeled[label_col].notna().sum())
        # mean() on empty/NaN-safe
        win_rate = float(df_labeled[label_col].mean() * 100.0) if n_valid > 0 else float('nan')
        print(f"  Stop {stop_atr:4.2f} ATR: {n_valid:7,} valid labels ({win_rate:5.2f}% win rate)")

    return df_labeled


def train_rf_model(
    df,
    stop_atr,
    features,
    test_year=2024,
    train_start_year=None,
    train_end_year=None,
    test_end_year=None,
    label_mode='touch_vwap',
    sample_weight_mode='none',
    slippage_per_share=SLIPPAGE_PER_SHARE,
    min_net_r: float = 0.0,
    model_kind: str = 'classifier',
    regression_target: str = 'net_r',
    max_realized_target_samples: int = 20000,
    setup_filter: bool = False,
    min_dist_atr: float = 0.5,
    min_minutes_session: int = 15,
    max_minutes_session: int = 360,
    min_rr_setup: float = 1.0,
    calibrate: bool = False,
):
    """Train RF model for a single stop width.

    model_kind:
      - 'classifier' (existing behavior)
      - 'regressor'  (predict expected payoff; used for max-P&L selection)

    regression_target:
      - 'net_r': per-trade net R multiple = (gross_pnl - costs)/risk
      - 'net_pnl': per-trade net dollars (gross_pnl - costs)
      - 'realized_net_pnl': *realized-path* net dollars (VWAP/stop/EOD) via _simulate_trade_realized_path()

    Setup filter (when setup_filter=True):
      Only bars that qualify as a "proper reversion setup" are included in
      training and evaluation.  Non-setup bars are dropped entirely — they are
      not labeled 0, they simply do not exist in the dataset.  This focuses the
      model on realistic trade candidates instead of diluting it with 40K bars
      where no trader would ever enter.

    Notes:
      - For classifier mode, y_test_raw is preserved for P&L evaluation.
      - For regressor mode, y_test_raw is still preserved for realized *outcome* P&L evaluation,
        but selection is based on predicted payoff.
      - To be consistent with a conservative interpretation of OHLC ambiguity, the realized-path
        simulator assumes stop is hit before target if both are touched in the same bar.
        Label generation should use the same ordering to avoid fundamental differences.
    """
    label_col = f"label_s{stop_atr}".replace(".", "_")

    # Filter valid labels
    valid = df[label_col].notna()
    df_valid = df[valid].copy()

    # --- SETUP FILTER: only keep bars that qualify as proper reversion setups ---
    n_before_setup = len(df_valid)
    if setup_filter:
        setup_mask = apply_setup_filter(
            df_valid,
            stop_atr=stop_atr,
            min_dist_atr=min_dist_atr,
            min_minutes_session=min_minutes_session,
            max_minutes_session=max_minutes_session,
            min_rr_setup=min_rr_setup,
        )
        df_valid = df_valid[setup_mask].copy()
        n_after_setup = len(df_valid)
        pct_kept = n_after_setup / max(1, n_before_setup) * 100
        print(f"  [SETUP] {n_before_setup:,} -> {n_after_setup:,} bars ({pct_kept:.1f}% kept)"
              f" | min_dist={min_dist_atr} min_mins={min_minutes_session}"
              f" max_mins={max_minutes_session} min_rr={min_rr_setup}")
    else:
        n_after_setup = n_before_setup

    if len(df_valid) < 500:
        return None

    # Build features
    X = df_valid[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_raw = df_valid[label_col].astype(int)

    # Common per-trade economics (for regression targets and/or cost-aware labels)
    reward_per_share = df_valid['vwap_width_atr'] * df_valid['atr']
    reward = reward_per_share * SHARES_PER_TRADE
    risk = float(stop_atr) * df_valid['atr'] * SHARES_PER_TRADE
    costs = 2.0 * (COMMISSION_PER_SHARE + float(slippage_per_share)) * SHARES_PER_TRADE

    # Regression target computed on *realized label outcome*: if label touch then +reward else -risk
    gross_pnl = np.where(y_raw.values == 1, reward.values, -risk.values)
    net_pnl = gross_pnl - float(costs)
    risk_safe = risk.replace(0, np.nan)
    net_r = (net_pnl / risk_safe).fillna(0.0)

    # Labels for classifier / base outcomes for train/test split
    if label_mode in ('net_positive', 'net_positive_r'):
        net_if_win = reward - costs
        if label_mode == 'net_positive':
            y = ((y_raw == 1) & (net_if_win > 0)).astype(int)
        else:
            net_r_if_win = (net_if_win / risk_safe).fillna(-np.inf)
            y = ((y_raw == 1) & (net_r_if_win >= float(min_net_r))).astype(int)
    else:
        y = y_raw.copy()
    # Train/test split by year
    df_valid['year'] = pd.to_datetime(df_valid['datetime']).dt.year
    if train_start_year is None and train_end_year is None:
        train_mask = df_valid['year'] < test_year
    elif train_start_year is not None and train_end_year is not None:
        train_mask = (df_valid['year'] >= train_start_year) & (df_valid['year'] < train_end_year)
    elif train_start_year is not None:
        train_mask = (df_valid['year'] >= train_start_year) & (df_valid['year'] < test_year)
    else:
        train_mask = df_valid['year'] < train_end_year

    if test_end_year is not None:
        test_mask = (df_valid['year'] >= test_year) & (df_valid['year'] < test_end_year)
    else:
        test_mask = df_valid['year'] >= test_year

    # Optional: realized-path net P&L target (slower)
    realized_net_pnl = None
    if model_kind == 'regressor' and regression_target == 'realized_net_pnl':
        # Build a simulation table that preserves df_valid's original index for safe alignment
        # - sim_index: df_valid index (used to align back to X/y_raw/df_valid)
        # - bar_index: original bar index into df_full (used by the simulator)
        df_valid_sim = df_valid.reset_index().rename(columns={'index': 'bar_index'})
        df_valid_sim['sim_index'] = df_valid_sim['bar_index']


        # Downsample to keep runtime bounded
        n_all = len(df_valid_sim)
        n_max = int(max(1000, max_realized_target_samples))
        if n_all > n_max:
            df_valid_sim = df_valid_sim.sample(n=n_max, random_state=42)

        bar_idx = df_valid_sim['bar_index'].astype(int).to_numpy()
        # Use 'bar_index' (== original df_valid index labels) so the Series
        # aligns with X / y_raw / df_valid via .loc later.
        keep_idx = df_valid_sim['bar_index'].astype(int).to_numpy()

        realized_vals = []
        for b in bar_idx:
            t = _simulate_trade_realized_path(
                df_full=df,
                entry_bar_index=int(b),
                stop_atr=float(stop_atr),
                shares=int(SHARES_PER_TRADE),
                slippage_per_share=float(slippage_per_share),
            )
            realized_vals.append(float(t['net_pnl']))

        realized_net_pnl = pd.Series(realized_vals, index=keep_idx)

    if model_kind == 'regressor':
        if regression_target == 'net_pnl':
            y_reg = pd.Series(net_pnl, index=df_valid.index)
        elif regression_target == 'realized_net_pnl':
            if realized_net_pnl is None or len(realized_net_pnl) == 0:
                return None
            y_reg = realized_net_pnl

            # Align EVERYTHING to the sampled subset using df_valid indices (which equal original bar_index labels)
            X = X.loc[y_reg.index]
            y_raw = y_raw.loc[y_reg.index]
            df_valid = df_valid.loc[y_reg.index]
            # Recompute year + masks on the subset so splits match downstream evaluation
            df_valid['year'] = pd.to_datetime(df_valid['datetime']).dt.year
            if train_start_year is None and train_end_year is None:
                train_mask = df_valid['year'] < test_year
            elif train_start_year is not None and train_end_year is not None:
                train_mask = (df_valid['year'] >= train_start_year) & (df_valid['year'] < train_end_year)
            elif train_start_year is not None:
                train_mask = (df_valid['year'] >= train_start_year) & (df_valid['year'] < test_year)
            else:
                train_mask = df_valid['year'] < train_end_year

            if test_end_year is not None:
                test_mask = (df_valid['year'] >= test_year) & (df_valid['year'] < test_end_year)
            else:
                test_mask = df_valid['year'] >= test_year
        else:
            y_reg = pd.Series(net_r, index=df_valid.index)

        # Build X splits (must exist for ALL regression targets)
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_test_raw = y_raw[test_mask]

        # Build regression y splits
        y_train = y_reg[train_mask]
        y_test = y_reg[test_mask]

        if len(X_train) < 200 or len(X_test) < 50:
            return None

        rf = RandomForestRegressor(**RF_REG_PARAMS)
        rf.fit(X_train, y_train)

        pred_train = rf.predict(X_train)
        pred_test = rf.predict(X_test)
        importance = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)

        train_dates = df_valid.loc[train_mask, 'datetime']
        test_dates = df_valid.loc[test_mask, 'datetime']

        # y_train_raw = binary win/loss labels for training bars (for RF value-add table)
        y_train_raw = y_raw[train_mask]

        return {
            'model': rf,
            'model_kind': 'regressor',
            'X_train': X_train,
            'y_train': y_train,
            'y_train_raw': y_train_raw,
            'pred_train': pred_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_raw': y_test_raw,
            'proba_test': pred_test,  # NOTE: reuse key; now it's predicted payoff
            'train_index': X_train.index,  # original df row labels for train set alignment
            'test_index': X_test.index,  # original df row labels for test set alignment
            'importance': importance,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_before_setup': n_before_setup,
            'n_after_setup': n_after_setup,
            'train_date_range': (str(train_dates.min()), str(train_dates.max())),
            'test_date_range': (str(test_dates.min()), str(test_dates.max())),
            'train_win_rate': float(y_raw[train_mask].mean()),
            'test_win_rate': float(y_raw[test_mask].mean()),
            'features': features,
            'stop_atr': stop_atr,
            'rf_params': RF_REG_PARAMS,
            'pnl_metrics': {
                'total_gross_pnl': 0.0,
                'total_net_pnl': 0.0,
                'total_costs': 0.0,
                'avg_risk_dollars': 0.0,
                'avg_net_pnl_per_trade': 0.0,
                'capital_per_trade': 0.0,
                'return_pct_per_trade': 0.0,
            },
        }    # --- logistic regression path ---
    if model_kind == 'logistic':
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        if len(X_train) < 200 or len(X_test) < 50:
            return None

        # Scale features (logistic regression needs normalised inputs)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=features)

        # --- Balance classes via random oversampling of minority (positive) class ---
        pos_mask = y_train == 1
        neg_mask = ~pos_mask
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        if n_pos < n_neg and n_pos > 0:
            # Randomly duplicate positive samples to match negative count
            rng = np.random.RandomState(42)
            pos_indices = np.where(pos_mask)[0]
            extra_indices = rng.choice(pos_indices, size=(n_neg - n_pos), replace=True)
            X_train_bal = pd.concat([X_train_scaled, X_train_scaled.iloc[extra_indices]], ignore_index=True)
            y_train_bal = pd.concat([y_train, y_train.iloc[extra_indices]], ignore_index=True)
        else:
            X_train_bal = X_train_scaled
            y_train_bal = y_train

        lr = LogisticRegression(**LOGISTIC_PARAMS)
        lr.fit(X_train_bal, y_train_bal)

        proba_train = lr.predict_proba(X_train_scaled)[:, 1]
        proba_test = lr.predict_proba(X_test_scaled)[:, 1]

        # Coefficient-based importance (absolute value of standardised coefficients)
        coefs = np.abs(lr.coef_[0])
        importance = pd.DataFrame({'feature': features, 'importance': coefs}).sort_values('importance', ascending=False)

        train_dates = df_valid.loc[train_mask, 'datetime']
        test_dates = df_valid.loc[test_mask, 'datetime']

        return {
            'model': lr,
            'scaler': scaler,
            'model_kind': 'classifier',  # shares classifier eval path
            'X_train': X_train_scaled,
            'y_train': y_train,
            'y_train_raw': y_raw[train_mask],
            'pred_train': proba_train,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_test_raw': y_test,
            'proba_test': proba_test,
            'train_index': X_train.index,
            'test_index': X_test.index,
            'importance': importance,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_before_setup': n_before_setup,
            'n_after_setup': n_after_setup,
            'train_date_range': (str(train_dates.min()), str(train_dates.max())),
            'test_date_range': (str(test_dates.min()), str(test_dates.max())),
            'train_win_rate': float(y_train.mean()),
            'test_win_rate': float(y_test.mean()),
            'features': features,
            'stop_atr': stop_atr,
            'rf_params': LOGISTIC_PARAMS,
            'pnl_metrics': {
                'total_gross_pnl': 0.0,
                'total_net_pnl': 0.0,
                'total_costs': 0.0,
                'avg_risk_dollars': 0.0,
                'avg_net_pnl_per_trade': 0.0,
                'capital_per_trade': 0.0,
                'return_pct_per_trade': 0.0,
            },
        }

    # --- neural network (MLP) classifier path ---
    if model_kind == 'nn':
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        if len(X_train) < 200 or len(X_test) < 50:
            return None

        # Scale features (neural networks are very sensitive to input scale)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=features)

        # --- Balance classes via random oversampling of minority (positive) class ---
        pos_mask = y_train == 1
        neg_mask = ~pos_mask
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        if n_pos < n_neg and n_pos > 0:
            rng = np.random.RandomState(42)
            pos_indices = np.where(pos_mask)[0]
            extra_indices = rng.choice(pos_indices, size=(n_neg - n_pos), replace=True)
            X_train_bal = pd.concat([X_train_scaled, X_train_scaled.iloc[extra_indices]], ignore_index=True)
            y_train_bal = pd.concat([y_train, y_train.iloc[extra_indices]], ignore_index=True)
        else:
            X_train_bal = X_train_scaled
            y_train_bal = y_train

        nn = MLPClassifier(**NN_PARAMS)
        nn.fit(X_train_bal, y_train_bal)

        n_epochs = nn.n_iter_
        _best_loss = getattr(nn, 'best_loss_', None)
        val_loss_str = f" | best val loss: {_best_loss:.4f}" if _best_loss is not None else ""
        print(f"  [NN] Converged in {n_epochs} epochs{val_loss_str}")

        proba_train = nn.predict_proba(X_train_scaled)[:, 1]
        proba_test = nn.predict_proba(X_test_scaled)[:, 1]

        # Permutation importance (model-agnostic; works for any black-box model)
        # Use a small subsample of test set to keep runtime bounded
        n_perm_samples = min(2000, len(X_test_scaled))
        perm_result = permutation_importance(
            nn, X_test_scaled.iloc[:n_perm_samples], y_test.iloc[:n_perm_samples],
            n_repeats=5, random_state=42, scoring='roc_auc', n_jobs=-1,
        )
        importance = pd.DataFrame({
            'feature': features,
            'importance': perm_result.importances_mean,
        }).sort_values('importance', ascending=False)

        train_dates = df_valid.loc[train_mask, 'datetime']
        test_dates = df_valid.loc[test_mask, 'datetime']

        return {
            'model': nn,
            'scaler': scaler,
            'model_kind': 'classifier',  # shares classifier eval path
            'X_train': X_train_scaled,
            'y_train': y_train,
            'y_train_raw': y_raw[train_mask],
            'pred_train': proba_train,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_test_raw': y_test,
            'proba_test': proba_test,
            'train_index': X_train.index,
            'test_index': X_test.index,
            'importance': importance,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_before_setup': n_before_setup,
            'n_after_setup': n_after_setup,
            'train_date_range': (str(train_dates.min()), str(train_dates.max())),
            'test_date_range': (str(test_dates.min()), str(test_dates.max())),
            'train_win_rate': float(y_train.mean()),
            'test_win_rate': float(y_test.mean()),
            'features': features,
            'stop_atr': stop_atr,
            'rf_params': NN_PARAMS,
            'pnl_metrics': {
                'total_gross_pnl': 0.0,
                'total_net_pnl': 0.0,
                'total_costs': 0.0,
                'avg_risk_dollars': 0.0,
                'avg_net_pnl_per_trade': 0.0,
                'capital_per_trade': 0.0,
                'return_pct_per_trade': 0.0,            },
        }

    # --- nn_pnl: PyTorch profit-maximising neural network path ---
    if model_kind == 'nn_pnl':
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        if len(X_train) < 200 or len(X_test) < 50:
            return None

        # Per-bar net P&L (dollars) — this IS the training signal
        pnl_all = pd.Series(net_pnl, index=df_valid.index)
        pnl_train = pnl_all[train_mask].values.astype(np.float64)
        pnl_test = pnl_all[test_mask].values.astype(np.float64)

        # Scale features
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train)
        X_test_np = scaler.transform(X_test)

        # Train/val split for early stopping
        val_frac = NN_PNL_PARAMS.get('val_fraction', 0.1)
        rng = np.random.RandomState(NN_PNL_PARAMS.get('random_state', 42))
        n_train = len(X_train_np)
        indices = rng.permutation(n_train)
        n_val = max(1, int(n_train * val_frac))
        val_idx = indices[:n_val]
        trn_idx = indices[n_val:]

        X_trn, X_val = X_train_np[trn_idx], X_train_np[val_idx]
        pnl_trn, pnl_val = pnl_train[trn_idx], pnl_train[val_idx]
        y_trn, y_val = y_train.values[trn_idx], y_train.values[val_idx]

        # No class balancing needed — loss is weighted by P&L, not class frequency.
        # Winners already get positive signal, losers get negative signal.

        net = _train_nn_pnl(
            X_trn, pnl_trn, y_trn,
            X_val, pnl_val, y_val,
            n_features=len(features),
            params=NN_PNL_PARAMS,
        )

        # Wrap for sklearn-compatible predict_proba
        wrapper = PnLModelWrapper(net, scaler, features)

        # Produce probabilities on unscaled frames via wrapper
        X_train_df = pd.DataFrame(X_train_np, index=X_train.index, columns=features)
        X_test_df = pd.DataFrame(X_test_np, index=X_test.index, columns=features)

        net.eval()
        with torch.no_grad():
            proba_train = torch.sigmoid(
                net(torch.tensor(X_train_np, dtype=torch.float32))
            ).numpy()
            proba_test = torch.sigmoid(
                net(torch.tensor(X_test_np, dtype=torch.float32))
            ).numpy()

        # Permutation importance using custom PnL-based scorer
        def pnl_scorer(estimator, X_df, y_unused):
            """Score = sum(prob * net_pnl) on test set — expected P&L."""
            probs = estimator.predict_proba(X_df)[:, 1]
            # Retrieve corresponding pnl values
            idx = X_df.index if hasattr(X_df, 'index') else np.arange(len(X_df))
            # Use test set pnl aligned to same rows
            return float(np.sum(probs * pnl_test[:len(probs)]))

        n_perm_samples = min(2000, len(X_test_df))
        perm_result = permutation_importance(
            wrapper, X_test_df.iloc[:n_perm_samples], y_test.iloc[:n_perm_samples],
            n_repeats=5, random_state=42, scoring=pnl_scorer, n_jobs=1,
        )
        importance = pd.DataFrame({
            'feature': features,
            'importance': perm_result.importances_mean,
        }).sort_values('importance', ascending=False)

        train_dates = df_valid.loc[train_mask, 'datetime']
        test_dates = df_valid.loc[test_mask, 'datetime']

        return {
            'model': wrapper,
            'scaler': scaler,
            'model_kind': 'classifier',  # shares classifier eval path
            'X_train': X_train_df,
            'y_train': y_train,
            'y_train_raw': y_raw[train_mask],
            'pred_train': proba_train,
            'X_test': X_test_df,
            'y_test': y_test,
            'y_test_raw': y_test,
            'proba_test': proba_test,
            'train_index': X_train.index,
            'test_index': X_test.index,
            'importance': importance,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_before_setup': n_before_setup,
            'n_after_setup': n_after_setup,
            'train_date_range': (str(train_dates.min()), str(train_dates.max())),
            'test_date_range': (str(test_dates.min()), str(test_dates.max())),
            'train_win_rate': float(y_train.mean()),
            'test_win_rate': float(y_test.mean()),
            'features': features,
            'stop_atr': stop_atr,
            'rf_params': NN_PNL_PARAMS,
            'pnl_metrics': {
                'total_gross_pnl': 0.0,
                'total_net_pnl': 0.0,
                'total_costs': 0.0,
                'avg_risk_dollars': 0.0,
                'avg_net_pnl_per_trade': 0.0,
                'capital_per_trade': 0.0,
                'return_pct_per_trade': 0.0,
            },
        }

    # --- linear_reg path (Ridge regression on per-trade net P&L) ---
    if model_kind == 'linear_reg':
        # Target: net_pnl per trade (label-based, not realized-path)
        y_reg = pd.Series(net_pnl, index=df_valid.index)

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train_reg = y_reg[train_mask]
        y_test_reg = y_reg[test_mask]
        y_test_raw = y_raw[test_mask]

        if len(X_train) < 200 or len(X_test) < 50:
            return None

        # Scale features (Ridge benefits from normalised inputs)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=features)

        ridge = Ridge(**RIDGE_PARAMS)
        ridge.fit(X_train_scaled, y_train_reg)

        pred_train = ridge.predict(X_train_scaled)
        pred_test = ridge.predict(X_test_scaled)

        # Coefficient-based importance (absolute value of standardised coefficients)
        coefs = np.abs(ridge.coef_)
        importance = pd.DataFrame({'feature': features, 'importance': coefs}).sort_values('importance', ascending=False)

        train_dates = df_valid.loc[train_mask, 'datetime']
        test_dates = df_valid.loc[test_mask, 'datetime']

        # y_train_raw = binary win/loss labels for training bars (for value-add table)
        y_train_raw = y_raw[train_mask]

        return {
            'model': ridge,
            'scaler': scaler,
            'model_kind': 'regressor',  # shares regressor eval path
            'X_train': X_train_scaled,
            'y_train': y_train_reg,
            'y_train_raw': y_train_raw,
            'pred_train': pred_train,
            'X_test': X_test_scaled,
            'y_test': y_test_reg,
            'y_test_raw': y_test_raw,
            'proba_test': pred_test,  # predicted net P&L per trade
            'train_index': X_train.index,
            'test_index': X_test.index,
            'importance': importance,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_before_setup': n_before_setup,
            'n_after_setup': n_after_setup,
            'train_date_range': (str(train_dates.min()), str(train_dates.max())),
            'test_date_range': (str(test_dates.min()), str(test_dates.max())),
            'train_win_rate': float(y_raw[train_mask].mean()),
            'test_win_rate': float(y_raw[test_mask].mean()),
            'features': features,
            'stop_atr': stop_atr,
            'rf_params': RIDGE_PARAMS,
            'pnl_metrics': {
                'total_gross_pnl': 0.0,
                'total_net_pnl': 0.0,
                'total_costs': 0.0,
                'avg_risk_dollars': 0.0,
                'avg_net_pnl_per_trade': 0.0,
                'capital_per_trade': 0.0,
                'return_pct_per_trade': 0.0,
            },
        }

    # --- classifier path (existing behavior) ---
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    if len(X_train) < 200 or len(X_test) < 50:
        return None

    # Compute optional sample weights
    sample_weight = None
    if sample_weight_mode == 'rr_proportional':
        per_trade_rr = (df_valid['vwap_width_atr'] / stop_atr).values
        w = np.where(per_trade_rr > 0, per_trade_rr, 0.0)
        w = np.clip(w, 0.1, 100.0)
        sample_weight = np.where(y.values == 1, w, 1.0)[train_mask]

    rf = RandomForestClassifier(**RF_PARAMS)
    if sample_weight is None:
        rf.fit(X_train, y_train)
    else:
        rf.fit(X_train, y_train, sample_weight=sample_weight)

    # --- Optional probability calibration (isotonic regression) ---
    # RF probabilities are compressed into a narrow band (e.g. 0.4–0.7) because
    # shallow trees + class_weight='balanced' constrain leaf predictions.
    # Calibration maps them onto the full [0,1] range so that prob_weighted mode
    # can assign bigger positions to genuinely high-confidence trades.
    if calibrate:
        # CalibratedClassifierCV with cv=5 uses cross-val on training data
        # to learn a monotonic mapping (isotonic) from raw proba → calibrated proba.
        # 'prefit' would need a held-out set; cv=5 reuses training data more efficiently.
        cal = CalibratedClassifierCV(rf, cv=5, method='isotonic')
        if sample_weight is None:
            cal.fit(X_train, y_train)
        else:
            cal.fit(X_train, y_train, sample_weight=sample_weight)
        predict_model = cal
        # Feature importance still comes from the base RF
        print(f"  [CAL] Calibrated RF probabilities (isotonic, cv=5)")
    else:
        predict_model = rf

    proba_train = predict_model.predict_proba(X_train)[:, 1]
    proba_test = predict_model.predict_proba(X_test)[:, 1]

    if calibrate:
        # Show how calibration broadened the distribution
        p5, p50, p95 = np.percentile(proba_test, [5, 50, 95])
        print(f"  [CAL] Test proba distribution: p5={p5:.3f} p50={p50:.3f} p95={p95:.3f}")

    importance = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)

    train_dates = df_valid.loc[train_mask, 'datetime']
    test_dates = df_valid.loc[test_mask, 'datetime']

    return {
        'model': rf,
        'model_kind': 'classifier',
        'X_train': X_train,
        'y_train': y_train,
        'y_train_raw': y_raw[train_mask],
        'pred_train': proba_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_test_raw': y_test,
        'proba_test': proba_test,
        'train_index': X_train.index,
        'test_index': X_test.index,  # original df row labels for test set alignment
        'importance': importance,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_before_setup': n_before_setup,
        'n_after_setup': n_after_setup,
        'train_date_range': (str(train_dates.min()), str(train_dates.max())),
        'test_date_range': (str(test_dates.min()), str(test_dates.max())),
        'train_win_rate': float(y_train.mean()),
        'test_win_rate': float(y_test.mean()),
        'features': features,
        'stop_atr': stop_atr,
        'rf_params': RF_PARAMS,
        'pnl_metrics': {
            'total_gross_pnl': 0.0,
            'total_net_pnl': 0.0,
            'total_costs': 0.0,
            'avg_risk_dollars': 0.0,
            'avg_net_pnl_per_trade': 0.0,
            'capital_per_trade': 0.0,
            'return_pct_per_trade': 0.0,
        },
    }


# ============================================================================
# METRICS / P&L HELPERS (self-contained)
# ============================================================================

def calculate_ev_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    stop_atr: float,
    median_dist_atr: float,
    win_definition: str = 'label',
    atr_series=None,
    vwap_dist_atr_series=None,
    shares_per_trade: int = SHARES_PER_TRADE,
    commission_per_share: float = COMMISSION_PER_SHARE,
    slippage_per_share: float = SLIPPAGE_PER_SHARE,
    **kwargs,
):
    """Classifier-only EV metrics.

    Returns:
        (ev_df, rr, breakeven_wr)

    Notes:
      - EV here is in R units using headline rr = median_dist_atr/stop_atr.
      - This is kept for backward compatibility with existing report logic.
    """
    rr = float(median_dist_atr / float(stop_atr)) if stop_atr else float('nan')
    breakeven_wr = 1.0 / (1.0 + rr) if rr > 0 else float('nan')

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba)

    rows = []
    for t in ([0.0] + RF_THRESHOLDS):
        mask = p >= t
        n = int(mask.sum())
        if n <= 0:
            rows.append({'threshold': t, 'n': 0, 'wr': np.nan, 'ev_r': np.nan})
            continue
        wr = float(y[mask].mean())
        ev_r = wr * rr - (1.0 - wr)
        rows.append({'threshold': t, 'n': n, 'wr': wr, 'ev_r': ev_r})

    ev_df = pd.DataFrame(rows)
    return ev_df, rr, breakeven_wr


def calculate_dollar_pnl_with_capital_constraint(
    y_actual,
    stop_atr: float,
    atr_series,
    entry_price_series,
    vwap_dist_atr_series,
    datetime_series,
    slippage_per_share: float = SLIPPAGE_PER_SHARE,
    capital_cap: float = CAPITAL_CAP,
    is_long_series=None,
    ohlc_series=None,
    log_file_path: Path = None,
):
    """Compute label-based net P&L with a simple capital-cap concurrency model.

    Assumptions:
      - If y_actual==1: winner makes reward = vwap_dist_atr * atr * shares.
      - Else: loser loses risk = stop_atr * atr * shares.
      - Net subtracts 2*(commission+slippage)*shares per trade.
      - Capital usage per position = entry_price * shares.
      - Concurrency approximation: each trade holds 1 bar. This is consistent with the
        original pipeline's constraint intent (avoid impossible simultaneous exposure)
        without requiring full path simulation.
    """
    y = np.asarray(y_actual).astype(int)
    atr = np.asarray(atr_series, dtype=float)
    entry = np.asarray(entry_price_series, dtype=float)
    dist = np.asarray(vwap_dist_atr_series, dtype=float)
    dt = pd.to_datetime(np.asarray(datetime_series))

    n_signals = int(len(y))
    if n_signals == 0:
        return {
            'n_trades_executed': 0,
            'pct_signals_executed': 0.0,
            'total_gross_pnl': 0.0,
            'total_net_pnl': 0.0,
            'total_costs': 0.0,
            'avg_risk_dollars': 0.0,
            'avg_net_pnl_per_trade': 0.0,
            'capital_per_trade': 0.0,
            'return_pct_per_trade': 0.0,
            'total_return_pct': 0.0,
            'max_positions_held': 0,
            'avg_positions_held': 0.0,
        }

    shares = int(SHARES_PER_TRADE)
    costs_per_trade = 2.0 * (float(COMMISSION_PER_SHARE) + float(slippage_per_share)) * shares

    risk_dollars = float(stop_atr) * atr * shares
    reward_dollars = dist * atr * shares

    gross = np.where(y == 1, reward_dollars, -risk_dollars)
    net = gross - costs_per_trade

    capital_per_trade = entry * shares

    # 1-bar holding period concurrency model
    executed = np.zeros(n_signals, dtype=bool)
    open_positions = 0
    max_open = 0
    open_counts = []

    for i in range(n_signals):
        # close previous bar's positions (hold=1)
        open_positions = 0

        if capital_per_trade[i] <= 0:
            open_counts.append(open_positions)
            continue

        if capital_per_trade[i] <= float(capital_cap):
            # only one position can be held under 1-bar model
            executed[i] = True
            open_positions = 1

        max_open = max(max_open, open_positions)
        open_counts.append(open_positions)

    gross_exec = gross[executed]
    net_exec = net[executed]

    total_gross = float(np.sum(gross_exec))
    total_net = float(np.sum(net_exec))
    total_costs = float(costs_per_trade * int(executed.sum()))

    avg_risk = float(np.mean(risk_dollars[executed])) if executed.any() else 0.0
    avg_net = float(np.mean(net_exec)) if executed.any() else 0.0

    avg_capital = float(np.mean(capital_per_trade[executed])) if executed.any() else 0.0
    ret_per_trade = (avg_net / avg_capital) * 100.0 if avg_capital > 0 else 0.0
    total_ret = (total_net / float(capital_cap)) * 100.0 if float(capital_cap) > 0 else 0.0

    if log_file_path is not None:
        try:
            log_df = pd.DataFrame({
                'datetime': dt,
                'label': y,
                'executed': executed.astype(int),
                'vwap_dist_atr': dist,
                'atr': atr,
                'entry_price': entry,
                'risk_dollars': risk_dollars,
                'reward_dollars': reward_dollars,
                'gross_pnl': gross,
                'net_pnl': net,
                'costs': costs_per_trade,
                'per_trade_rr': np.where(risk_dollars > 0, reward_dollars / risk_dollars, np.nan),
            })
            log_df.to_csv(log_file_path, index=False)
        except Exception:
            pass

    return {
        'n_trades_executed': int(executed.sum()),
        'pct_signals_executed': float(executed.mean() * 100.0),
        'total_gross_pnl': total_gross,
        'total_net_pnl': total_net,
        'total_costs': total_costs,
        'avg_risk_dollars': avg_risk,
        'avg_net_pnl_per_trade': avg_net,
        'capital_per_trade': avg_capital,
        'return_pct_per_trade': ret_per_trade,
        'total_return_pct': total_ret,
        'max_positions_held': int(max_open),
        'avg_positions_held': float(np.mean(open_counts)) if len(open_counts) else 0.0,
    }


# ============================================================================
# TRADE-LEVEL SELECTION / REALIZED-PATH SIM
# ============================================================================

def _simulate_trade_realized_path(
    df_full: pd.DataFrame,
    entry_bar_index: int,
    stop_atr: float,
    shares: int,
    slippage_per_share: float,
    capital: float = CAPITAL_CAP,
):
    """Simulate a single trade from entry bar until VWAP touch, stop, or EOD.

    Returns dict with exit info and net_pnl.

    Consistency note (important):
      - Labels in `src/label_generator.py` define the VWAP target as the *entry bar's VWAP* (fixed),
        then scan forward starting at the *next bar* to see whether stop or target is hit first.
      - This simulator follows that same rule for touch/stop logic, but still allows an EOD exit
        at the session close for a realistic $ P&L when VWAP is never touched.

    Assumptions:
      - Entry at close of entry bar.
      - Direction based on is_long_setup at entry (close < vwap => long else short).
      - Target VWAP is the entry bar VWAP (fixed target; NOT bar-by-bar).
      - Stop distance is stop_atr * entry_atr.
      - Exit priority within a bar: stop hit first (conservative), else VWAP touch, else EOD close.
      - Touch/stop checks begin on the bar AFTER entry (to match label generation).
      - IBKR margin interest is charged on (notional - capital) if notional > capital,
        prorated to the actual hold duration using tiered rates.
    """
    row0 = df_full.loc[entry_bar_index]
    entry_dt = pd.to_datetime(row0['datetime'])
    entry_date = row0['date']

    entry_price = float(row0['close'])
    entry_atr = float(row0['atr'])
    is_long = bool(row0['is_long_setup'])

    # Fixed VWAP target = entry bar VWAP (match label_generator)
    target_vwap_price = float(row0['vwap'])

    stop_dist = float(stop_atr) * entry_atr
    if is_long:
        stop_price = entry_price - stop_dist
    else:
        stop_price = entry_price + stop_dist

    # Costs per round-trip
    costs = 2.0 * (float(COMMISSION_PER_SHARE) + float(slippage_per_share)) * int(shares)

    # Walk forward within the same session/day
    exit_bar_index = entry_bar_index
    exit_reason = 'eod'
    exit_price = float(row0['close'])
    # Only select columns needed for simulation (iterrows on wide DFs is very slow)
    df_day = df_full.loc[df_full['date'] == entry_date, ['datetime', 'high', 'low', 'close']]
    if not df_day.index.is_monotonic_increasing:
        df_day = df_day.sort_values('datetime')

    # Start evaluating from the bar AFTER entry (match label_generator: k=j+1)
    started = False
    first_check = True
    for idx, r in df_day.iterrows():
        if not started:
            if idx != entry_bar_index:
                continue
            started = True
            continue  # skip entry bar checks

        # From here on, idx is strictly after entry_bar_index (in iteration order)
        hi = float(r['high'])
        lo = float(r['low'])

        # Conservative ordering: stop first
        if is_long:
            if lo <= stop_price:
                exit_bar_index = idx
                exit_reason = 'stop'
                exit_price = stop_price
                break
            if hi >= target_vwap_price:
                exit_bar_index = idx
                exit_reason = 'vwap'
                exit_price = target_vwap_price
                break
        else:
            if hi >= stop_price:
                exit_bar_index = idx
                exit_reason = 'stop'
                exit_price = stop_price
                break
            if lo <= target_vwap_price:
                exit_bar_index = idx
                exit_reason = 'vwap'
                exit_price = target_vwap_price
                break

        # else keep going until EOD
        exit_bar_index = idx
        exit_price = float(r['close'])    # Gross P&L
    if is_long:
        gross = (exit_price - entry_price) * int(shares)
    else:
        gross = (entry_price - exit_price) * int(shares)

    # IBKR margin borrowing cost
    notional = entry_price * int(shares)
    borrowed = max(0.0, notional - float(capital))
    exit_dt = pd.to_datetime(df_full.loc[exit_bar_index, 'datetime'])
    hold_td = exit_dt - entry_dt
    hold_hours = max(hold_td.total_seconds() / 3600.0, 0.0)
    # Minimum 5 minutes (1 bar) even when entry == exit bar
    if hold_hours < (5.0 / 60.0):
        hold_hours = 5.0 / 60.0
    margin_cost = ibkr_margin_cost(borrowed, hold_hours)

    net = float(gross) - float(costs) - margin_cost

    return {
        'entry_bar_index': int(entry_bar_index),
        'exit_bar_index': int(exit_bar_index),
        'entry_datetime': entry_dt,
        'exit_datetime': exit_dt,
        'is_long': int(is_long),
        'entry_price': float(entry_price),
        'exit_price': float(exit_price),
        'exit_reason': str(exit_reason),
        'gross_pnl': float(gross),
        'net_pnl': float(net),
        'costs': float(costs),
        'margin_cost': float(margin_cost),
        'margin_borrowed': float(borrowed),
        'notional': float(notional),
        'hold_hours': float(hold_hours),
        'shares': int(shares),
        'risk_dollars': float(stop_dist * int(shares)),
        'vwap_dist_atr': float(row0['vwap_width_atr']),
        'per_trade_rr': float(row0['vwap_width_atr'] / float(stop_atr)) if float(stop_atr) > 0 else float('nan'),
    }


def simulate_trade_level_pnl(
    df_full: pd.DataFrame,
    df_signals: pd.DataFrame,
    score_array: np.ndarray,
    stop_atr: float,
    model_kind: str,
    threshold: float = 0.5,
    top_n: int = 1000,
    min_rr: float = 0.0,
    slippage_per_share: float = SLIPPAGE_PER_SHARE,
    log_file_path: Path | None = None,
    shares_array: np.ndarray | None = None,
    capital: float = CAPITAL_CAP,
    max_concurrent: int = 1,
):
    """Convert bar-level model outputs into realized-path trades.

    - When max_concurrent=1: flat-to-flat (original behavior).
    - When max_concurrent>1: allows overlapping trades up to the limit.
      Aggregate open notional is tracked dynamically — new trades are skipped
      when adding them would exceed total capital.  Each trade's margin cost
      is computed against the *remaining* capital at entry time.

    Signal selection:
      - classifier: score >= threshold
      - regressor: uses global top-N by score, then applies simulation in time order
    shares_array: if provided, per-signal shares (for prob_weighted sizing).
                  Must be same length as df_signals/score_array.
    capital: available cash capital; notional exceeding this incurs IBKR margin interest.

    Returns pnl_metrics dict aligned with calculate_dollar_pnl_with_capital_constraint keys.
    """
    _empty = {
        'n_trades_executed': 0,
        'pct_signals_executed': 0.0,
        'total_gross_pnl': 0.0,
        'total_net_pnl': 0.0,
        'total_costs': 0.0,
        'total_margin_cost': 0.0,
        'avg_margin_cost_per_trade': 0.0,
        'avg_notional': 0.0,
        'avg_margin_borrowed': 0.0,
        'avg_risk_dollars': 0.0,
        'avg_net_pnl_per_trade': 0.0,
        'capital_per_trade': 0.0,
        'return_pct_per_trade': 0.0,
        'total_return_pct': 0.0,
        'max_positions_held': 1,
        'avg_positions_held': 1.0,
        'skipped_capacity': 0,
        'skipped_capital': 0,
    }
    if len(df_signals) == 0:
        return _empty

    scores = np.asarray(score_array, dtype=float)
    # base mask: signals already pre-filtered by caller; accept all
    if model_kind == 'regressor':
        n = len(scores)
        top_n = int(max(1, min(int(top_n), n)))
        order = np.argsort(scores)[::-1]
        allowed = np.zeros(n, dtype=bool)
        allowed[order[:top_n]] = True
        base_mask = allowed
    else:
        # Classifier / prob_weighted: signals already masked by caller's threshold
        base_mask = np.ones(len(scores), dtype=bool)

    # min_rr filter
    if float(min_rr) > 0:
        rr_ok = (df_signals['vwap_width_atr'].values / float(stop_atr)) >= float(min_rr)
    else:
        rr_ok = np.ones(len(df_signals), dtype=bool)

    signal_mask = base_mask & rr_ok

    df_sig = df_signals.copy()
    df_sig['score'] = scores
    if shares_array is not None:
        df_sig['_shares'] = np.asarray(shares_array, dtype=int)
    else:
        df_sig['_shares'] = int(SHARES_PER_TRADE)
    df_sig = df_sig.loc[signal_mask].sort_values('datetime')

    trades = []
    # Track open positions: list of (exit_datetime, notional) for currently open trades
    open_positions = []  # [(exit_dt, notional), ...]
    max_open = 0
    open_counts = []  # per-signal open count for avg calculation
    skipped_capacity = 0
    skipped_capital = 0

    # We need the original bar index to simulate on df_full
    if 'bar_index' not in df_sig.columns:
        raise ValueError("df_signals must include bar_index")

    for _, sig in df_sig.iterrows():
        entry_idx = int(sig['bar_index'])
        entry_dt = pd.to_datetime(sig['datetime'])
        trade_shares = int(sig['_shares'])
        entry_price = float(sig['close'])
        trade_notional = entry_price * trade_shares

        # Expire closed positions
        open_positions = [(t, n) for t, n in open_positions if entry_dt <= t]

        # Check max concurrent capacity
        if len(open_positions) >= int(max_concurrent):
            open_counts.append(len(open_positions))
            skipped_capacity += 1
            continue

        # Check aggregate notional: would this trade exceed total capital?
        open_notional = sum(n for _, n in open_positions)
        if open_notional + trade_notional > float(capital):
            open_counts.append(len(open_positions))
            skipped_capital += 1
            continue

        # Remaining capital for margin cost calculation
        remaining_capital = float(capital) - open_notional

        t = _simulate_trade_realized_path(
            df_full=df_full,
            entry_bar_index=entry_idx,
            stop_atr=float(stop_atr),
            shares=trade_shares,
            slippage_per_share=float(slippage_per_share),
            capital=remaining_capital,
        )
        trades.append(t)
        open_positions.append((t['exit_datetime'], t['notional']))
        open_counts.append(len(open_positions))
        max_open = max(max_open, len(open_positions))

    if len(trades) == 0:
        return _empty

    trades_df = pd.DataFrame(trades)

    total_gross = float(trades_df['gross_pnl'].sum())
    total_net = float(trades_df['net_pnl'].sum())
    total_costs = float(trades_df['costs'].sum())
    total_margin_cost = float(trades_df['margin_cost'].sum())
    avg_margin_cost = float(trades_df['margin_cost'].mean())
    avg_notional = float(trades_df['notional'].mean())
    avg_margin_borrowed = float(trades_df['margin_borrowed'].mean())
    avg_risk = float(trades_df['risk_dollars'].mean())
    avg_net = float(trades_df['net_pnl'].mean())
    # Capital proxy for reporting (entry_price * shares)
    if 'shares' in trades_df.columns:
        capital_per_trade = float((trades_df['entry_price'] * trades_df['shares']).mean())
    else:
        capital_per_trade = float((trades_df['entry_price'] * SHARES_PER_TRADE).mean())
    ret_per_trade = (avg_net / capital_per_trade) * 100.0 if capital_per_trade > 0 else 0.0
    total_ret = (total_net / float(CAPITAL_CAP)) * 100.0 if float(CAPITAL_CAP) > 0 else 0.0

    if log_file_path is not None:
        try:
            trades_df.to_csv(log_file_path, index=False)
        except Exception:
            pass
    # Yearly P&L breakdown
    yearly_pnl = {}
    try:
        trades_df['_year'] = pd.to_datetime(trades_df['entry_datetime']).dt.year
        for yr, grp in trades_df.groupby('_year'):
            yr_net = float(grp['net_pnl'].sum())
            yr_n = len(grp)
            yr_wr = float((grp['net_pnl'] > 0).mean()) if yr_n > 0 else 0.0
            yr_avg = float(grp['net_pnl'].mean()) if yr_n > 0 else 0.0
            yearly_pnl[int(yr)] = {
                'n_trades': yr_n,
                'total_net_pnl': yr_net,
                'win_rate': yr_wr,
                'avg_net_pnl': yr_avg,
            }
        trades_df.drop(columns=['_year'], inplace=True)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Trade size distribution stats
    # ------------------------------------------------------------------
    share_dist = {}
    try:
        sh = trades_df['shares'].values.astype(float)
        nt = trades_df['notional'].values.astype(float)
        rk = trades_df['risk_dollars'].values.astype(float)
        share_dist = {
            'shares_mean': float(np.mean(sh)),
            'shares_median': float(np.median(sh)),
            'shares_std': float(np.std(sh)),
            'shares_min': float(np.min(sh)),
            'shares_max': float(np.max(sh)),
            'shares_p05': float(np.percentile(sh, 5)),
            'shares_p25': float(np.percentile(sh, 25)),
            'shares_p75': float(np.percentile(sh, 75)),
            'shares_p95': float(np.percentile(sh, 95)),
            'notional_mean': float(np.mean(nt)),
            'notional_median': float(np.median(nt)),
            'notional_min': float(np.min(nt)),
            'notional_max': float(np.max(nt)),
            'risk_mean': float(np.mean(rk)),
            'risk_median': float(np.median(rk)),
            'risk_min': float(np.min(rk)),
            'risk_max': float(np.max(rk)),
            'n_capital_capped': int((nt >= float(capital) * 0.999).sum()),
            'n_risk_sized': int((nt < float(capital) * 0.999).sum()),
            'pct_capital_capped': float((nt >= float(capital) * 0.999).sum() / max(1, len(nt)) * 100),
        }
        # Share size buckets
        bucket_edges = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 50000, 100000]
        bucket_labels = ['0-1K', '1K-2K', '2K-3K', '3K-4K', '4K-5K',
                         '5K-6K', '6K-7K', '7K-8K', '8K-10K', '10K-50K', '50K-100K']
        counts, _ = np.histogram(sh, bins=bucket_edges)
        share_dist['share_buckets'] = {
            lbl: int(c) for lbl, c in zip(bucket_labels, counts) if c > 0
        }
        # Long vs Short breakdown
        if 'is_long' in trades_df.columns:
            for direction, label in [(True, 'long'), (1, 'long'), (False, 'short'), (0, 'short')]:
                sub = trades_df[trades_df['is_long'] == direction]
                if len(sub) > 0:
                    share_dist[f'{label}_n_trades'] = int(len(sub))
                    share_dist[f'{label}_avg_shares'] = float(sub['shares'].mean())
                    share_dist[f'{label}_avg_notional'] = float(sub['notional'].mean())
                    share_dist[f'{label}_avg_net_pnl'] = float(sub['net_pnl'].mean())
                    share_dist[f'{label}_total_net_pnl'] = float(sub['net_pnl'].sum())
                    break  # found the right dtype for is_long
            # Also do the other direction
            for direction, label in [(False, 'short'), (0, 'short'), (True, 'long'), (1, 'long')]:
                sub = trades_df[trades_df['is_long'] == direction]
                if len(sub) > 0 and f'{label}_n_trades' not in share_dist:
                    share_dist[f'{label}_n_trades'] = int(len(sub))
                    share_dist[f'{label}_avg_shares'] = float(sub['shares'].mean())
                    share_dist[f'{label}_avg_notional'] = float(sub['notional'].mean())
                    share_dist[f'{label}_avg_net_pnl'] = float(sub['net_pnl'].mean())
                    share_dist[f'{label}_total_net_pnl'] = float(sub['net_pnl'].sum())
                    break
        # By exit reason
        if 'exit_reason' in trades_df.columns:
            exit_breakdown = {}
            for reason, grp in trades_df.groupby('exit_reason'):
                exit_breakdown[str(reason)] = {
                    'n_trades': int(len(grp)),
                    'avg_shares': float(grp['shares'].mean()),
                    'avg_notional': float(grp['notional'].mean()),
                    'total_net_pnl': float(grp['net_pnl'].sum()),
                }
            share_dist['exit_breakdown'] = exit_breakdown
    except Exception:
        pass

    return {
        'n_trades_executed': int(len(trades_df)),
        'pct_signals_executed': float(len(trades_df) / max(1, int(signal_mask.sum())) * 100.0),
        'total_gross_pnl': total_gross,
        'total_net_pnl': total_net,
        'total_costs': total_costs,
        'total_margin_cost': total_margin_cost,
        'avg_margin_cost_per_trade': avg_margin_cost,
        'avg_notional': avg_notional,
        'avg_margin_borrowed': avg_margin_borrowed,
        'avg_risk_dollars': avg_risk,
        'avg_net_pnl_per_trade': avg_net,
        'capital_per_trade': capital_per_trade,
        'return_pct_per_trade': ret_per_trade,
        'total_return_pct': total_ret,
        'max_positions_held': int(max_open),
        'avg_positions_held': float(np.mean(open_counts)) if len(open_counts) else 1.0,
        'skipped_capacity': int(skipped_capacity),
        'skipped_capital': int(skipped_capital),
        'yearly_pnl': yearly_pnl,
        'share_dist': share_dist,
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    global TEST_YEAR
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
        "--min-rr",
        type=float,
        default=0.0,
        help=(
            "Minimum per-trade R:R (vwap_width_atr / stop_atr) to accept a signal. "
            "Trades below this ratio are rejected even if RF proba >= threshold. "
            "E.g. --min-rr 0.75 filters out trades whose reward < 75%% of risk. (default: 0.0 = no filter)"
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=["touch_vwap", "net_positive", "net_positive_r"],
        default="touch_vwap",
        help=(
            "Labeling mode used for training: "
            "'touch_vwap' (win=touch VWAP before stop); "
            "'net_positive' (touch VWAP AND net win $ > 0 after costs); "
            "'net_positive_r' (touch VWAP AND net win in R >= --min-net-r; strong filter for low-payoff trades)."
        ),
    )
    parser.add_argument(
        "--min-net-r",
        type=float,
        default=0.25,
        help=(
            "Only used when --label-mode net_positive_r. "
            "Minimum net R multiple required for a touch_vwap to be labeled as 1. "
            "netR = (reward_dollars - costs) / risk_dollars. Default: 0.25."
        ),
    )
    parser.add_argument(
        "--sample-weight",
        choices=["none", "rr_proportional"],
        default="none",
        help=(
            "Optional sample weighting during RF training. 'rr_proportional' weights winning samples by their per-trade R:R, "
            "so the model prioritizes high reward trades. Default is 'none'."
        ),
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.01,
        help="Slippage per share in dollars (default: 0.01). Use this to model different slippage scenarios.",
    )
    parser.add_argument(
        "--win-definition",
        choices=["label", "net_pnl", "realized_net_pnl"],
        default="label",
        help=(
            "How to count 'wins' in reported win rate: "
            "'label' uses forward-looking label==1; "
            "'net_pnl' uses net P&L>0 after fees/slippage with ATR-based reward assumptions; "
            "'realized_net_pnl' uses net P&L>0 after costs computed from realized path (VWAP/stop/EOD close)."
        ),
    )
    parser.add_argument(
        "--pnl-definition",
        choices=["label_rr", "realized_path"],
        default="label_rr",
        help=(
            "How to compute dollar P&L: 'label_rr' uses label outcomes with ATR-based reward/risk; "
            "'realized_path' uses realized exit prices from VWAP/stop/EOD close and net after costs."
        ),
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help=(
            "Path to the 5-min bar CSV file (e.g. data/tqqq_5min_10years.csv). "
            "Overrides the default DATA_FILE constant. Ticker is inferred from the filename."
        ),
    )
    parser.add_argument(        "--model-kind",
        choices=["classifier", "regressor", "logistic", "linear_reg", "nn", "nn_pnl"],
        default="classifier",
        help=(
            "Model type: 'classifier' = RF classifier (default). "
            "'regressor' = RF regressor predicting expected payoff. "
            "'logistic' = Logistic Regression (calibrated probabilities, strong L2 regularisation). "
            "'linear_reg' = Ridge (L2-regularised linear regression on per-trade net P&L). "
            "'nn' = Neural Network (MLP) classifier — captures nonlinear feature interactions like RF "
            "but with better probability calibration; uses StandardScaler + random oversampling. "
            "'nn_pnl' = PyTorch NN with profit-maximising loss — directly optimises expected dollar P&L "
            "instead of classification accuracy. Loss = -mean(sigmoid(logit) * net_pnl_per_bar)."
        ),
    )
    parser.add_argument(
        "--regression-target",
        choices=["net_r", "net_pnl", "realized_net_pnl"],
        default="net_r",
        help=(
            "Only used when --model-kind regressor. "
            "Target to regress: 'net_r' (net R multiple) or 'net_pnl' (net dollars)."
        ),
    )
    parser.add_argument(
        "--select-mode",
        choices=["threshold", "top", "prob_weighted"],
        default="threshold",
        help=(
            "How to select trades from model outputs: "
            "'threshold' uses RF_THRESHOLDS for classifier proba (default). "
            "'top' selects top-N trades by predicted payoff (regressor), controlled by --top-n. "
            "'prob_weighted' uses classifier P(win) to scale position size (shares). "
            "Takes all trades with P(win) >= --prob-threshold, scales shares linearly."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5000,
        help="Only used when --select-mode top. Select the top N trades by predicted payoff (default: 5000).",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.50,
        help="Min P(win) to take a trade in prob_weighted mode (default: 0.50).",    )
    parser.add_argument(        "--prob-min-shares",
        type=int,
        default=100,
        help="Shares at the probability threshold (minimum sizing) in prob_weighted mode (default: 100).",
    )
    parser.add_argument(
        "--prob-max-shares",
        type=int,
        default=500,        help="Shares at P(win)=1.0 (maximum sizing) in prob_weighted mode (default: 500).",
    )
    parser.add_argument(
        "--prob-risk-pct",
        type=float,
        default=0.0,
        help=(
            "Risk-based position sizing: fraction of CAPITAL_CAP to risk per trade (e.g. 0.01 = 1%%). "
            "When > 0, overrides --prob-min/max-shares. Shares = (capital * risk_pct) / (stop_atr * ATR). "
            "The prob score still scales the size: min_frac at threshold, full size at P=1.0. "
            "Default: 0.0 (disabled, use fixed share limits)."
        ),
    )
    parser.add_argument(
        "--prob-scale-min",
        type=float,
        default=0.3,
        help=(
            "In prob_weighted + risk-based sizing: fraction of full risk-based shares at the "
            "probability threshold (lowest confidence). Scales linearly up to --prob-scale-max "
            "at P(win)=1.0. E.g. 0.3 means 30%% of full size at threshold. Default: 0.3."
        ),
    )
    parser.add_argument(
        "--prob-scale-max",
        type=float,
        default=1.0,
        help=(
            "In prob_weighted + risk-based sizing: fraction of full risk-based shares at "
            "P(win)=1.0 (highest confidence). E.g. 1.0 means 100%% of full size. "
            "Use <1.0 to cap max sizing below full risk allocation. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help=(
            "Maximum number of concurrent open positions. Default: 1 (flat-to-flat). "
            "Set higher to allow overlapping trades. Capital is split equally among "
            "max_concurrent slots, so each position's notional cap = CAPITAL / max_concurrent. "
            "E.g. --max-concurrent 3 with $1M capital => $333K notional cap per position."
        ),
    )
    parser.add_argument(
        "--max-realized-target-samples",
        type=int,
        default=20000,
        help=(
            "Only used when --model-kind regressor --regression-target realized_net_pnl. "
            "Caps the number of rows to simulate for the realized-path regression target to keep runtime reasonable. "
            "Default: 20000."
        ),
    )
    parser.add_argument(
        "--stream-results",
        action="store_true",
        help=(
            "Write results incrementally in real time as each stop_atr finishes. "
            "Creates data/master_pipeline_progress_<timestamp>.csv and writes one row per stop/selection."
        ),
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Calibrate RF classifier probabilities using isotonic regression (cv=5). "
            "Broadens the compressed RF probability distribution so that prob_weighted "
            "mode can assign larger positions to genuinely high-confidence trades. "
            "Only applies to --model-kind classifier (RF). Ignored for other model kinds."
        ),
    )

    # ---- Setup filter arguments ----
    parser.add_argument(
        "--setup-filter",
        action="store_true",
        help=(
            "Enable setup filter: only train/evaluate on bars that qualify as "
            "proper VWAP reversion setups (meaningful extension, VWAP stabilised, "
            "enough session time, minimum R:R).  Non-setup bars are dropped entirely."
        ),
    )
    parser.add_argument(
        "--min-dist-atr",
        type=float,
        default=SETUP_DEFAULTS['min_dist_atr'],
        help=(
            "Setup filter: minimum distance from VWAP in ATR units. "
            f"Default: {SETUP_DEFAULTS['min_dist_atr']}"
        ),
    )
    parser.add_argument(
        "--min-minutes-session",
        type=int,
        default=SETUP_DEFAULTS['min_minutes_session'],
        help=(
            "Setup filter: minimum minutes into session (VWAP needs time to stabilise). "
            f"Default: {SETUP_DEFAULTS['min_minutes_session']}"
        ),
    )
    parser.add_argument(
        "--max-minutes-session",
        type=int,
        default=SETUP_DEFAULTS['max_minutes_session'],
        help=(
            "Setup filter: maximum minutes into session (need time for reversion to play out). "
            f"Default: {SETUP_DEFAULTS['max_minutes_session']}"
        ),
    )
    parser.add_argument(
        "--min-rr-setup",
        type=float,
        default=SETUP_DEFAULTS['min_rr_setup'],
        help=(
            "Setup filter: minimum per-trade R:R (vwap_width_atr / stop_atr) for a bar "
            "to qualify as a setup. Bars below this are not trained on at all. "
            f"Default: {SETUP_DEFAULTS['min_rr_setup']}"
        ),
    )
    parser.add_argument(
        "--train-years",
        type=str,
        default=None,
        help=(
            "Training year range as START-END (inclusive). E.g. --train-years 2016-2020. "
            "If not set, trains on all years before --test-years start (or TEST_YEAR)."
        ),
    )
    parser.add_argument(
        "--test-years",
        type=str,
        default=None,
        help=(
            "Test year range as START-END (inclusive). E.g. --test-years 2021-2026. "
            "If not set, tests on all years >= TEST_YEAR (2024)."
        ),
    )
    parser.add_argument(
        "--indicators-file",
        type=str,
        default=None,
        help=(
            "Path to a pre-computed indicators Parquet file (e.g. from "
            "precompute_streaming_indicators.py). When provided, skips "
            "calculate_core_indicators() and uses these indicators instead. "
            "This allows training on streaming-style indicators that match "
            "live inference conditions."
        ),
    )

    args = parser.parse_args()

    # Parse year ranges
    train_start_year = None
    train_end_year = None  # exclusive upper bound (year < train_end_year)
    test_start_year = TEST_YEAR
    test_end_year = None  # None = no upper bound

    if args.train_years:
        parts = args.train_years.split('-')
        train_start_year = int(parts[0])
        train_end_year = int(parts[1]) + 1  # inclusive -> exclusive

    if args.test_years:
        parts = args.test_years.split('-')
        test_start_year = int(parts[0])
        test_end_year = int(parts[1]) + 1  # inclusive -> exclusive
    elif args.train_years and not args.test_years:
        # If only train specified, test starts right after train ends
        test_start_year = train_end_year

    # Store on args for downstream use
    args._train_start_year = train_start_year
    args._train_end_year = train_end_year
    args._test_start_year = test_start_year
    args._test_end_year = test_end_year
    # Logistic regression shares the classifier eval path;
    # Neural network shares the classifier eval path;
    # Linear regression shares the regressor eval path;
    # preserve original for display, normalise for branching.
    args._display_model_kind = args.model_kind  # 'classifier', 'regressor', 'logistic', 'linear_reg', 'nn', or 'nn_pnl'
    if args.model_kind in ('logistic', 'nn', 'nn_pnl'):
        args.model_kind = 'classifier'          # eval loop treats it like classifier
    elif args.model_kind == 'linear_reg':
        args.model_kind = 'regressor'           # eval loop treats it like regressor

    # Override global TEST_YEAR
    TEST_YEAR = test_start_year

    # Define timestamp early for logging filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Progress / streaming output file (optional)
    progress_csv: Path | None = None
    if args.stream_results:
        progress_csv = OUTPUT_DIR / f"master_pipeline_progress_{timestamp}.csv"
        try:
            # Truncate/create the file and write header later after first row (schema depends on results keys)
            if progress_csv.exists():
                progress_csv.unlink()
        except Exception:
            progress_csv = None

    # Resolve data file and infer ticker
    data_file = Path(args.data_file) if args.data_file else DATA_FILE
    ticker = data_file.stem.split("_")[0].upper()   # e.g. "tsla" from "tsla_5min_10years.csv"

    print("\n" + "="*80)
    print(f" MASTER PIPELINE: RF VWAP REVERSION ANALYSIS  [{ticker}]")
    print("="*80)
    print(f"  Data file: {data_file}")
    train_label = f"{args._train_start_year}-{args._train_end_year - 1}" if args._train_start_year and args._train_end_year else f"<{TEST_YEAR}"
    test_label = f"{args._test_start_year}-{args._test_end_year - 1}" if args._test_end_year else f"{TEST_YEAR}+"
    print(f"  Train: {train_label} | Test: {test_label}")
    if args.min_rr > 0.0:
        print(f"  [FILTER] Min R:R = {args.min_rr:.2f} (reject trades with vwap_dist/stop < {args.min_rr:.2f})")
    print(f"  Slippage: ${args.slippage:.3f}/share | Win def: {args.win_definition} | P&L def: {args.pnl_definition}")
    if args.setup_filter:
        print(f"  [SETUP FILTER] min_dist={args.min_dist_atr} ATR | min_mins={args.min_minutes_session}"
              f" | max_mins={args.max_minutes_session} | min_rr_setup={args.min_rr_setup}")
    if progress_csv is not None:
        print(f"  [STREAM] Writing incremental results to: {progress_csv}")
    if args.indicators_file:
        print(f"  [STREAMING] Using pre-computed indicators: {args.indicators_file}")

    # Step 1 & 2: Load data + indicators
    if args.indicators_file:
        # Load pre-computed streaming indicators (e.g. from precompute_streaming_indicators.py)
        ind_path = Path(args.indicators_file)
        print(f"\n{'='*80}")
        print("LOADING PRE-COMPUTED STREAMING INDICATORS")
        print(f"{'='*80}")
        df = pd.read_parquet(ind_path, engine="pyarrow")
        # Ensure datetime is tz-aware UTC
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        # Ensure date column is Python date objects (needed for groupby('date'))
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        # Add 'time' column if missing (some downstream code expects it)
        if "time" not in df.columns and "datetime" in df.columns:
            df["time"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"[OK] Loaded {len(df):,} bars with {len(df.columns)} columns from {ind_path.name}")
        print(f"[OK] Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    else:
        # Standard path: load raw CSV + compute vectorized indicators
        df = load_and_validate_data(data_file)
        df = calculate_core_indicators(df)

    # Step 3: Get feature columns
    features = get_feature_columns(df)
    print(f"\n[OK] Using {len(features)} features:")
    for i, feat in enumerate(features, 1):
        print(f"  {i:2d}. {feat}")

    # Step 4: Generate labels
    df = generate_all_labels(df)

    # Step 5: Train RF models and calculate metrics
    print(f"\n{'='*80}")
    print("TRAINING RF MODELS & CALCULATING METRICS")
    print(f"{'='*80}\n")

    all_results = []
    trained_models = {}  # Store models for optional saving

    def _append_stream_row(row: dict):
        """Append a single results row to the streaming CSV, if enabled."""
        if progress_csv is None:
            return
        try:
            df_row = pd.DataFrame([row])
            write_header = not progress_csv.exists() or progress_csv.stat().st_size == 0
            df_row.to_csv(progress_csv, mode="a", header=write_header, index=False)
        except Exception:
            # Never fail the pipeline due to streaming I/O
            pass

    for stop_atr in STOP_ATRS:
        print(f"\n--- Stop Width: {stop_atr} ATR ---")

        result = train_rf_model(
            df,
            stop_atr,
            features,
            TEST_YEAR,
            train_start_year=args._train_start_year,
            train_end_year=args._train_end_year,
            test_end_year=args._test_end_year,
            label_mode=args.label_mode,
            sample_weight_mode=args.sample_weight,
            slippage_per_share=args.slippage,
            min_net_r=args.min_net_r,
            model_kind=args._display_model_kind,
            regression_target=args.regression_target,
            max_realized_target_samples=int(args.max_realized_target_samples),
            setup_filter=args.setup_filter,
            min_dist_atr=args.min_dist_atr,
            min_minutes_session=args.min_minutes_session,
            max_minutes_session=args.max_minutes_session,
            min_rr_setup=args.min_rr_setup,
            calibrate=args.calibrate,
        )

        if result is None:
            print(f"  [WARNING] Insufficient data for {stop_atr} ATR")
            continue

        kind = result.get('model_kind', 'classifier')
        print(f"  [OK] Trained({kind}): {result['n_train']:,} train, {result['n_test']:,} test samples")

        trained_models[stop_atr] = result
        # Build aligned test-set frame for this stop
        # If the model was trained on a downsampled subset (e.g. realized_net_pnl),
        # result['test_index'] contains the original df row labels for the test set.
        # Use those to subset df_test_local so lengths match result['proba_test'].
        label_col = f"label_s{stop_atr}".replace(".", "_")
        test_index = result.get('test_index', None)
        if test_index is not None:
            # Use the exact rows the model was tested on
            df_test_local = df.loc[test_index].copy()
        else:
            valid_mask_local = df[label_col].notna()
            df_valid_local = df.loc[valid_mask_local].copy()
            df_valid_local['year'] = pd.to_datetime(df_valid_local['datetime']).dt.year
            test_mask_local = df_valid_local['year'] >= TEST_YEAR
            if args._test_end_year is not None:
                test_mask_local = test_mask_local & (df_valid_local['year'] < args._test_end_year)
            df_test_local = df_valid_local.loc[test_mask_local].copy()

        # Ensure alignment: result['proba_test'] and df_test_local must match
        if len(df_test_local) != len(result['proba_test']):
            print(f"  [WARN] test set length mismatch ({len(df_test_local)} vs {len(result['proba_test'])}), skipping stop={stop_atr} metrics")
            continue

        # Stable bar indices for realized-path (if enabled)
        df_test_local = df_test_local.reset_index().rename(columns={'index': 'bar_index'})

        # Median distance for headline R:R
        valid_mask_all = df[label_col].notna()
        median_dist = float(df.loc[valid_mask_all, 'vwap_width_atr'].median())

        # Selection logic
        scores = np.asarray(result['proba_test'])
        # Max-P&L selection: with regressor, select by predicted payoff
        shares_array = None  # default: uniform SHARES_PER_TRADE for all trades
        if args.model_kind == 'regressor' and args.select_mode == 'prob_weighted' and args._display_model_kind == 'linear_reg':
            # Ridge P&L-weighted sizing: scale shares by predicted P&L magnitude
            # Filter: only take trades with predicted P&L > 0
            min_shares = int(args.prob_min_shares)
            max_shares = int(args.prob_max_shares)
            pred_pos_mask = scores > 0
            # Scale shares: min_shares at pred=0, max_shares at pred=max(positive predictions)
            pos_scores = scores[pred_pos_mask]
            pred_max = float(pos_scores.max()) if len(pos_scores) > 0 else 1.0
            if pred_max <= 0:
                pred_max = 1.0
            raw_scale = np.clip(scores / pred_max, 0.0, 1.0)
            shares_per_signal = np.round(min_shares + raw_scale * (max_shares - min_shares)).astype(int)
            shares_per_signal = np.clip(shares_per_signal, min_shares, max_shares)

            n_above = int(pred_pos_mask.sum())
            avg_shares_above = float(shares_per_signal[pred_pos_mask].mean()) if n_above > 0 else 0

            print(f"  [PRED_WEIGHTED] pred>0 trades={n_above}, "
                  f"shares=[{min_shares}..{max_shares}], "
                  f"avg_shares={avg_shares_above:.0f}, "
                  f"pred_range=[{float(scores.min()):.1f}..{float(scores.max()):.1f}]")

            shares_array = shares_per_signal
            base_masks = [("pred_pos", pred_pos_mask)]
        elif args.model_kind == 'regressor':
            if args.select_mode != 'top':
                print("  [INFO] Regressor selected; forcing select-mode to 'top' (max P&L mode)")
            top_n = int(max(0, args.top_n))
            if top_n <= 0:
                print("  [WARN] --top-n <= 0; skipping")
                continue
            order = np.argsort(scores)[::-1]
            top_idx = order[:min(top_n, len(order))]
            top_mask = np.isin(np.arange(len(scores)), top_idx)
            # Also add "predicted > 0" selection (natural threshold for P&L regressors)
            pred_pos_mask = scores > 0
            base_masks = [
                (f"top_{top_n}", top_mask),
                ("pred_pos", pred_pos_mask),
            ]
        elif args.select_mode == 'prob_weighted':
            # Probability-weighted position sizing (meta-labeling approach)
            prob_thresh = float(args.prob_threshold)
            prob_range = 1.0 - prob_thresh
            if prob_range <= 0:
                prob_range = 0.01

            # Scale factor per signal: 0.0 at threshold, 1.0 at P=1.0
            raw_scale = (scores - prob_thresh) / prob_range  # 0..1
            raw_scale = np.clip(raw_scale, 0.0, 1.0)

            risk_pct = float(args.prob_risk_pct)
            if risk_pct > 0:
                # Risk-based sizing: shares = (capital * risk_pct) / (stop_atr * ATR)
                # Scale between min_frac at threshold and max_frac at P=1.0
                atr_arr = df_test_local['atr'].values
                risk_dollars = float(CAPITAL_CAP) * risk_pct
                stop_risk_per_share = stop_atr * atr_arr  # $ risk per share
                stop_risk_per_share = np.where(stop_risk_per_share > 0, stop_risk_per_share, 1.0)
                full_shares = np.floor(risk_dollars / stop_risk_per_share).astype(int)
                # Scale: min_frac of full size at threshold, max_frac at P=1.0
                min_frac = float(args.prob_scale_min)
                max_frac = float(args.prob_scale_max)
                scale_frac = min_frac + raw_scale * (max_frac - min_frac)
                shares_per_signal = np.round(full_shares * scale_frac).astype(int)
                shares_per_signal = np.clip(shares_per_signal, 1, 9999)
                # Notional cap: shares × entry_price ≤ full CAPITAL_CAP
                # (concurrent positions handled dynamically in simulator, not here)
                entry_prices = df_test_local['close'].values
                max_shares_by_capital = np.floor(float(CAPITAL_CAP) / entry_prices).astype(int)
                shares_per_signal = np.minimum(shares_per_signal, max_shares_by_capital)
                shares_per_signal = np.clip(shares_per_signal, 1, max_shares_by_capital.max())
                sizing_label = f"risk_{risk_pct:.1%}"
                print(f"  [PROB-RISK] threshold={prob_thresh:.2f}, "
                      f"risk_pct={risk_pct:.1%}, "
                      f"risk_$/trade=${risk_dollars:,.0f}, "
                      f"scale=[{min_frac:.0%}-{max_frac:.0%}], "
                      f"max_concurrent={int(args.max_concurrent)}, "
                      f"capital=${CAPITAL_CAP:,.0f} (dynamic tracking), "
                      f"full_shares_range=[{int(full_shares.min())}-{int(full_shares.max())}], "
                      f"after_notional_cap=[{int(shares_per_signal.min())}-{int(shares_per_signal.max())}]")
            else:
                # Fixed share limits
                min_shares = int(args.prob_min_shares)
                max_shares = int(args.prob_max_shares)
                shares_per_signal = np.round(min_shares + raw_scale * (max_shares - min_shares)).astype(int)
                shares_per_signal = np.clip(shares_per_signal, min_shares, max_shares)
                # Notional cap: shares × entry_price ≤ full CAPITAL_CAP
                # (concurrent positions handled dynamically in simulator, not here)
                entry_prices = df_test_local['close'].values
                max_shares_by_capital = np.floor(float(CAPITAL_CAP) / entry_prices).astype(int)
                shares_per_signal = np.minimum(shares_per_signal, max_shares_by_capital)
                sizing_label = f"shares_{min_shares}-{max_shares}"

            prob_mask = scores >= prob_thresh
            n_above = int(prob_mask.sum())
            avg_shares_above = float(shares_per_signal[prob_mask].mean()) if n_above > 0 else 0

            print(f"  [PROB] threshold={prob_thresh:.2f}, "
                  f"trades above={n_above}, "
                  f"avg_shares={avg_shares_above:.0f}, "
                  f"sizing={sizing_label}")

            shares_array = shares_per_signal
            base_masks = [(f"prob_{prob_thresh:.2f}", prob_mask)]
        else:
            base_masks = [(str(t), scores >= t) for t in ([0.0] + RF_THRESHOLDS)]

        # Compute EV metrics only for classifier
        if args.model_kind == 'classifier':
            ev_df, rr, breakeven_wr = calculate_ev_metrics(
                result['y_test'], result['proba_test'], stop_atr, median_dist,
                win_definition=args.win_definition,
                atr_series=df_test_local['atr'].values if args.win_definition == 'net_pnl' else None,
                vwap_dist_atr_series=df_test_local['vwap_width_atr'].values,
                shares_per_trade=SHARES_PER_TRADE if args.win_definition in ('net_pnl', 'realized_net_pnl') else None,
                commission_per_share=COMMISSION_PER_SHARE if args.win_definition in ('net_pnl', 'realized_net_pnl') else None,
                slippage_per_share=args.slippage if args.win_definition in ('net_pnl', 'realized_net_pnl') else None,
                entry_close_series=df_test_local['close'].values if args.win_definition == 'realized_net_pnl' else None,
                entry_vwap_series=df_test_local['vwap'].values if args.win_definition == 'realized_net_pnl' else None,
                entry_atr_series=df_test_local['atr'].values if args.win_definition == 'realized_net_pnl' else None,
                entry_date_series=df_test_local['date'].values if args.win_definition == 'realized_net_pnl' else None,
                entry_index_series=df_test_local['bar_index'].values if args.win_definition == 'realized_net_pnl' else None,
                high_series=df['high'].values if args.win_definition == 'realized_net_pnl' else None,
                low_series=df['low'].values if args.win_definition == 'realized_net_pnl' else None,
                close_series=df['close'].values if args.win_definition == 'realized_net_pnl' else None,
                full_date_series=df['date'].values if args.win_definition == 'realized_net_pnl' else None,
                min_rr=args.min_rr,
            )
            rr_val = float(rr)
            print(f"  [OK] R:R = {rr_val:.2f}:1, Breakeven WR = {breakeven_wr*100:.1f}%")
        else:
            rr_val = float(median_dist / stop_atr) if stop_atr else float('nan')
            breakeven_wr = 1.0 / (1.0 + rr_val) if rr_val > 0 else float('nan')
            print(f"  [OK] (regressor) Headline R:R = {rr_val:.2f}:1")

        # Precompute per-trade R:R mask for optional min-rr filter
        vwap_dist_test_array = df_test_local['vwap_width_atr'].values
        if args.min_rr > 0.0:
            rr_ok_test = (vwap_dist_test_array / stop_atr) >= args.min_rr
        else:
            rr_ok_test = np.ones(len(scores), dtype=bool)

        # Evaluate each selection mask
        stop_rows_for_stream = []
        for sel_label, base_mask in base_masks:
            mask = base_mask & rr_ok_test
            n_trades = int(mask.sum())

            y_selected = result['y_test_raw'][mask]
            atr_selected = df_test_local['atr'].values[mask]
            datetime_selected = df_test_local['datetime'].values[mask]

            log_ohlc = {
                'open': df_test_local['open'].values[mask],
                'high': df_test_local['high'].values[mask],
                'low': df_test_local['low'].values[mask],
                'close': df_test_local['close'].values[mask],
                'volume': df_test_local['volume'].values[mask]
            }
            log_is_long = df_test_local['is_long_setup'].values[mask]

            log_file = OUTPUT_DIR / f"trades_y{TEST_YEAR}_stop{stop_atr}_sel{sel_label}_k{args.model_kind}_{timestamp}.csv"

            # Default: fast label-based P&L (bar-level)
            pnl_metrics = calculate_dollar_pnl_with_capital_constraint(
                y_actual=y_selected,
                stop_atr=stop_atr,
                atr_series=atr_selected,
                entry_price_series=df_test_local['close'].values[mask],
                vwap_dist_atr_series=df_test_local['vwap_width_atr'].values[mask],
                datetime_series=datetime_selected,
                slippage_per_share=args.slippage,
                capital_cap=CAPITAL_CAP,
                is_long_series=log_is_long,
                ohlc_series=log_ohlc,
                log_file_path=log_file,
            )
            # Optional: realized-path trade simulation (concurrent positions supported)
            # Always use realized-path for prob_weighted (variable shares) and regressor top
            use_realized = (args.pnl_definition == 'realized_path'
                            or (args.model_kind == 'regressor' and args.select_mode == 'top')
                            or args.select_mode == 'prob_weighted')
            if use_realized:
                try:
                    trade_log_file = OUTPUT_DIR / f"trades_realized_y{TEST_YEAR}_stop{stop_atr}_sel{sel_label}_k{args.model_kind}_{timestamp}.csv"
                    masked_shares = shares_array[mask] if shares_array is not None else None
                    pnl_metrics = simulate_trade_level_pnl(
                        df_full=df,
                        df_signals=df_test_local.loc[mask].copy(),
                        score_array=scores[mask],
                        stop_atr=float(stop_atr),
                        model_kind=str(args.model_kind),
                        threshold=float(sel_label) if (args.model_kind == 'classifier' and sel_label.replace('.', '', 1).replace('_', '', 1).isdigit()) else float(args.wf_threshold),
                        top_n=int(min(args.top_n, n_trades)) if n_trades > 0 else 1,
                        min_rr=0.0,  # already applied via mask
                        slippage_per_share=float(args.slippage),
                        log_file_path=trade_log_file,
                        shares_array=masked_shares,
                        capital=float(CAPITAL_CAP),
                        max_concurrent=int(args.max_concurrent),
                    )
                except Exception as e:
                    print(f"  [WARN] realized-path simulation failed: {e }")

                # Print concurrency stats
                if int(args.max_concurrent) > 1:
                    sk_cap = pnl_metrics.get('skipped_capacity', 0)
                    sk_notional = pnl_metrics.get('skipped_capital', 0)
                    max_held = pnl_metrics.get('max_positions_held', 1)
                    avg_held = pnl_metrics.get('avg_positions_held', 1.0)
                    print(f"    [CONCURRENT] max_held={max_held}, avg_held={avg_held:.1f}, "
                          f"skipped_capacity={sk_cap}, skipped_capital={sk_notional}")

            touch_wr = float(np.asarray(result['y_test_raw'])[mask].mean()) if n_trades > 0 else float('nan')

            row = {
                'stop_atr': stop_atr,
                'rr': rr_val,
                'breakeven_wr': breakeven_wr,
                'rf_threshold': float(sel_label) if (args.model_kind == 'classifier' and sel_label.replace('.', '', 1).isdigit()) else np.nan,
                'selection': sel_label,
                'model_kind': args.model_kind,
                'regression_target': args.regression_target if args.model_kind == 'regressor' else '',
                'n_trades': n_trades,
                'n_trades_executed': pnl_metrics['n_trades_executed'],
                'pct_signals_executed': pnl_metrics['pct_signals_executed'],
                'win_rate': touch_wr,
                'label_win_rate': touch_wr,
                'ev': np.nan,
                'pct_filtered': np.nan,
                'raw_win_rate': np.nan,
                'raw_ev': np.nan,
                'avg_risk_dollars': pnl_metrics['avg_risk_dollars'],
                'total_gross_pnl': pnl_metrics['total_gross_pnl'],
                'total_net_pnl': pnl_metrics['total_net_pnl'],
                'total_costs': pnl_metrics['total_costs'],
                'total_margin_cost': pnl_metrics.get('total_margin_cost', 0.0),
                'avg_margin_cost_per_trade': pnl_metrics.get('avg_margin_cost_per_trade', 0.0),
                'avg_notional': pnl_metrics.get('avg_notional', 0.0),
                'avg_margin_borrowed': pnl_metrics.get('avg_margin_borrowed', 0.0),
                'avg_net_pnl_per_trade': pnl_metrics['avg_net_pnl_per_trade'],
                'capital_per_trade': pnl_metrics['capital_per_trade'],
                'return_pct_per_trade': pnl_metrics['return_pct_per_trade'],
                'total_return_pct': pnl_metrics['total_return_pct'],
                'max_positions_held': pnl_metrics['max_positions_held'],
                'avg_positions_held': pnl_metrics['avg_positions_held'],
                'skipped_capacity': pnl_metrics.get('skipped_capacity', 0),
                'skipped_capital': pnl_metrics.get('skipped_capital', 0),
                'share_dist': pnl_metrics.get('share_dist', {}),
            }
            all_results.append(row)
            stop_rows_for_stream.append(row)
            _append_stream_row(row)

            if args.model_kind == 'regressor':
                margin_str = f" MarginCost=${pnl_metrics.get('total_margin_cost', 0):,.0f}" if pnl_metrics.get('total_margin_cost', 0) > 0 else ""
                print(
                    f"  [MAX-PNL {sel_label}] Trades={n_trades:,} WR(touch_vwap)={touch_wr*100:.1f}% "
                    f"NetP&L=${pnl_metrics['total_net_pnl']:,.0f} ({pnl_metrics['total_return_pct']:+.1f}%){margin_str}"
                )
            elif sel_label.startswith('prob_') or sel_label == '0.5':
                margin_str = f" MarginCost=${pnl_metrics.get('total_margin_cost', 0):,.0f}" if pnl_metrics.get('total_margin_cost', 0) > 0 else ""
                print(
                    f"  [{sel_label}] Trades={n_trades:,} WR(touch_vwap)={touch_wr*100:.1f}% "
                    f"NetP&L=${pnl_metrics['total_net_pnl']:,.0f} ({pnl_metrics['total_return_pct']:+.1f}%){margin_str}"
                )

            # Print yearly breakdown if available
            yearly_pnl = pnl_metrics.get('yearly_pnl', {})
            if yearly_pnl:
                row['yearly_pnl'] = yearly_pnl
                parts = []
                for yr in sorted(yearly_pnl.keys()):
                    yp = yearly_pnl[yr]
                    parts.append(f"{yr}: ${yp['total_net_pnl']:+,.0f} ({yp['n_trades']}t, {yp['win_rate']*100:.0f}%WR)")
                print(f"    yearly: {' | '.join(parts)}")

        # Stop-level real-time summary (total_net_pnl by selection)
        try:
            if len(stop_rows_for_stream) > 0:
                # pick the most comparable selection for console summary
                if args.model_kind == 'classifier':
                    # Match prob_weighted label or standard threshold label
                    if args.select_mode == 'prob_weighted':
                        prob_label = f"prob_{args.prob_threshold:.2f}"
                        pick = next((r for r in stop_rows_for_stream if str(r.get('selection')) == prob_label), stop_rows_for_stream[0])
                    else:
                        pick = next((r for r in stop_rows_for_stream if str(r.get('selection')) == '0.5'), stop_rows_for_stream[0])
                    pick_label = str(pick.get('selection'))
                else:
                    # Prefer pred_pos for summary if available
                    pick = next((r for r in stop_rows_for_stream if str(r.get('selection')) == 'pred_pos'), stop_rows_for_stream[0])
                    pick_label = str(pick.get('selection'))

                margin_c = float(pick.get('total_margin_cost', 0.0))
                margin_part = f" margin_cost=${margin_c:,.0f}" if margin_c > 0 else ""
                print(
                    f"  [STOP SUMMARY] stop_atr={stop_atr:.2f} sel={pick_label} "
                    f"n_exec={int(pick.get('n_trades_executed', 0)):,} "
                    f"total_net_pnl=${float(pick.get('total_net_pnl', 0.0)):,.0f}{margin_part}"
                )
        except Exception:
            pass

    # ------------------------------------------------------------------------
    # SAVE RESULTS / REPORT
    # ------------------------------------------------------------------------
    if len(all_results) == 0:
        print("\n[ERROR] No results produced. Exiting.")
        return None

    results_df = pd.DataFrame(all_results)

    # Stable ordering / columns (be tolerant if some fields are missing)
    preferred_cols = [
        'stop_atr', 'model_kind', 'selection', 'rf_threshold', 'regression_target',
        'rr', 'breakeven_wr',
        'n_trades', 'n_trades_executed', 'pct_signals_executed',
        'win_rate',
        'ev',
        'avg_risk_dollars',
        'total_gross_pnl', 'total_net_pnl', 'total_costs', 'total_margin_cost',
        'avg_margin_cost_per_trade', 'avg_notional', 'avg_margin_borrowed',
        'avg_net_pnl_per_trade',
        'capital_per_trade', 'return_pct_per_trade', 'total_return_pct',        'max_positions_held', 'avg_positions_held',
        'skipped_capacity', 'skipped_capital',
    ]
    existing_cols = [c for c in preferred_cols if c in results_df.columns]
    remaining_cols = [c for c in results_df.columns if c not in existing_cols]
    results_df = results_df[existing_cols + remaining_cols]

    results_csv = OUTPUT_DIR / f"master_pipeline_results_{timestamp}.csv"
    results_df.to_csv(results_csv, index=False)

    # Simple markdown summary
    summary_md = OUTPUT_DIR / f"master_pipeline_summary_{ticker}_{timestamp}.md"

    # Recommended stop: maximize net P&L on a comparable selection
    if args.model_kind == 'classifier':
        if args.select_mode == 'prob_weighted':
            prob_label = f"prob_{args.prob_threshold:.2f}"
            cand = results_df[results_df['selection'].astype(str) == prob_label]
        else:
            # Use RF>=0.5 row
            cand = results_df[(results_df['model_kind'] == 'classifier') & (results_df['selection'].astype(str) == '0.5')]
    else:
        # For regressor, use the top-N selection label
        top_label = f"top_{int(max(0, args.top_n))}"
        cand = results_df[(results_df['model_kind'] == 'regressor') & (results_df['selection'].astype(str) == top_label)]

    recommended_stop = None
    if len(cand) > 0 and 'total_net_pnl' in cand.columns:
        cand2 = cand.sort_values('total_net_pnl', ascending=False)
        recommended_stop = float(cand2.iloc[0]['stop_atr'])

    with open(summary_md, 'w', encoding='utf-8') as f:
        f.write(f"# Master pipeline summary — {ticker} ({timestamp})\n\n")
        f.write(f"- Ticker: {ticker}\n")
        f.write(f"- Data file: {data_file}\n")
        train_label = f"{args._train_start_year}-{args._train_end_year - 1}" if args._train_start_year and args._train_end_year else f"<{TEST_YEAR}"
        test_label = f"{args._test_start_year}-{args._test_end_year - 1}" if args._test_end_year else f"{TEST_YEAR}+"
        f.write(f"- Train years: {train_label}\n")
        f.write(f"- Test years: {test_label}\n")
        f.write(f"- Model kind: {args._display_model_kind}\n")
        if args.model_kind == 'regressor':
            f.write(f"- Regression target: {args.regression_target}\n")
            f.write(f"- Selection mode: {args.select_mode} (top_n={args.top_n})\n")
        else:
            if args.select_mode == 'prob_weighted':
                f.write(f"- Selection mode: prob_weighted\n")
                f.write(f"- Prob threshold: {args.prob_threshold}\n")
                if float(args.prob_risk_pct) > 0:
                    f.write(f"- Prob risk pct: {args.prob_risk_pct:.1%} of ${CAPITAL_CAP:,.0f}\n")
                    f.write(f"- Prob scale range: {args.prob_scale_min:.0%} – {args.prob_scale_max:.0%}\n")
                else:
                    f.write(f"- Prob min shares: {args.prob_min_shares}\n")
                    f.write(f"- Prob max shares: {args.prob_max_shares}\n")
                f.write(f"- Max concurrent positions: {args.max_concurrent}\n")
                f.write(f"- Capital model: dynamic notional tracking (full ${CAPITAL_CAP:,.0f} per trade, aggregate cap)\n")
            else:
                f.write(f"- RF thresholds: {RF_THRESHOLDS}\n")
        f.write(f"- Label mode: {args.label_mode}\n")
        if args.label_mode == 'net_positive_r':
            f.write(f"- min_net_r: {args.min_net_r}\n")
        f.write(f"- min_rr (post-filter): {args.min_rr}\n")
        f.write(f"- Slippage: {args.slippage}\n\n")

        if recommended_stop is not None:
            f.write(f"## Recommended stop_atr (by max total_net_pnl)\n\n")
            f.write(f"- stop_atr = **{recommended_stop:.2f}**\n\n")

        f.write("## Results (sorted by total_net_pnl)\n\n")
        if 'total_net_pnl' in results_df.columns:
            view = results_df.sort_values('total_net_pnl', ascending=False)
        else:
            view = results_df

        show_cols = [c for c in [
            'stop_atr', 'model_kind', 'selection', 'rf_threshold', 'regression_target',
            'n_trades', 'n_trades_executed',
            'win_rate', 'rr',
            'total_net_pnl', 'total_return_pct',
            'avg_net_pnl_per_trade', 'avg_risk_dollars',
        ] if c in view.columns]
        f.write(view[show_cols].to_markdown(index=False))
        f.write("\n")

        # ----------------------------------------------------------------
        # TOTAL P&L ON TEST YEARS
        # ----------------------------------------------------------------
        f.write("\n## Total P&L on Test Years\n\n")
        f.write(f"Test period: **{test_label}**\n\n")
        # Build summary: one row per stop_atr at the primary selection
        if args.model_kind == 'classifier' and args.select_mode == 'prob_weighted':
            sel_filter = f"prob_{args.prob_threshold:.2f}"
        elif args.model_kind == 'classifier':
            sel_filter = '0.5'
        else:
            sel_filter = f"top_{int(max(0, args.top_n))}"
        pnl_view = results_df[results_df['selection'].astype(str) == sel_filter].copy()
        if len(pnl_view) > 0 and 'total_net_pnl' in pnl_view.columns:
            pnl_view = pnl_view.sort_values('stop_atr')
            pnl_cols = [c for c in ['stop_atr', 'n_trades_executed', 'win_rate',
                                     'total_net_pnl', 'avg_net_pnl_per_trade',
                                     'total_return_pct'] if c in pnl_view.columns]
            pnl_summary = pnl_view[pnl_cols].copy()
            # Format win_rate as percentage string
            if 'win_rate' in pnl_summary.columns:
                pnl_summary['win_rate'] = pnl_summary['win_rate'].apply(
                    lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "n/a")
            if 'total_net_pnl' in pnl_summary.columns:
                pnl_summary['total_net_pnl'] = pnl_summary['total_net_pnl'].apply(
                    lambda x: f"${x:,.0f}")
            if 'avg_net_pnl_per_trade' in pnl_summary.columns:
                pnl_summary['avg_net_pnl_per_trade'] = pnl_summary['avg_net_pnl_per_trade'].apply(
                    lambda x: f"${x:,.0f}")
            if 'total_return_pct' in pnl_summary.columns:
                pnl_summary['total_return_pct'] = pnl_summary['total_return_pct'].apply(
                    lambda x: f"{x:+.1f}%")
            f.write(pnl_summary.to_markdown(index=False))
            f.write("\n")

            # Grand total row
            total_pnl = pnl_view['total_net_pnl'].sum()
            total_trades = int(pnl_view['n_trades_executed'].sum())
            best_stop = pnl_view.sort_values('total_net_pnl', ascending=False).iloc[0]
            f.write(f"\n**Grand total across all stops**: ${total_pnl:,.0f} "
                    f"({total_trades:,} total trades)\n\n")
            f.write(f"**Best stop**: {best_stop['stop_atr']:.2f} ATR -> "
                    f"${best_stop['total_net_pnl']:,.0f} "
                    f"({int(best_stop['n_trades_executed']):,} trades, "
                    f"{best_stop['win_rate']*100:.1f}% WR)\n")

            # Margin cost breakdown (if present)
            if 'total_margin_cost' in pnl_view.columns and pnl_view['total_margin_cost'].sum() > 0:
                f.write(f"\n### IBKR Margin Costs\n\n")
                margin_cols = [c for c in ['stop_atr', 'n_trades_executed',
                                           'total_gross_pnl', 'total_costs', 'total_margin_cost',
                                           'total_net_pnl',
                                           'avg_margin_cost_per_trade', 'avg_notional',
                                           'avg_margin_borrowed'] if c in pnl_view.columns]
                m_view = pnl_view[margin_cols].copy().sort_values('stop_atr')
                for c in ['total_gross_pnl', 'total_costs', 'total_margin_cost', 'total_net_pnl',
                           'avg_margin_cost_per_trade', 'avg_notional', 'avg_margin_borrowed']:
                    if c in m_view.columns:
                        m_view[c] = m_view[c].apply(lambda x: f"${x:,.0f}")
                f.write(m_view.to_markdown(index=False))
                f.write("\n\n")
                total_margin = pnl_view['total_margin_cost'].sum()
                f.write(f"**Total margin cost across all stops**: ${total_margin:,.0f}\n\n")
        else:
            f.write("_No trades executed at this selection._\n")
        f.write("\n")

        # ----------------------------------------------------------------
        # TRAINING DATA METRICS  (base rates — NOT model accuracy)
        # ----------------------------------------------------------------
        if trained_models:
            f.write("\n## Data Metrics\n\n")
            train_rows = []
            for stop_atr_k in sorted(trained_models.keys()):
                res = trained_models[stop_atr_k]
                n_before = res.get('n_before_setup', 0)
                n_after = res.get('n_after_setup', 0)
                pct_kept = n_after / max(1, n_before) * 100
                train_dr = res.get('train_date_range', ('', ''))
                test_dr = res.get('test_date_range', ('', ''))
                train_rows.append({
                    'stop_atr': stop_atr_k,
                    'n_total_bars': n_before,
                    'n_setup_bars': n_after,
                    'pct_kept': round(pct_kept, 1),
                    'n_train': res.get('n_train', 0),
                    'n_test': res.get('n_test', 0),
                    'base_wr_train': round(res.get('train_win_rate', float('nan')) * 100, 2),
                    'base_wr_test': round(res.get('test_win_rate', float('nan')) * 100, 2),
                    'train_start': train_dr[0][:10] if train_dr[0] else '',
                    'train_end': train_dr[1][:10] if train_dr[1] else '',
                    'test_start': test_dr[0][:10] if test_dr[0] else '',
                    'test_end': test_dr[1][:10] if test_dr[1] else '',
                })
            train_df = pd.DataFrame(train_rows)
            f.write(train_df.to_markdown(index=False))
            f.write("\n")

        # ----------------------------------------------------------------
        # RF MODEL VALUE-ADD TABLE
        # ----------------------------------------------------------------
        if trained_models and len(all_results) > 0:
            f.write("\n## RF Model Value-Add\n\n")
            f.write("all_WR = blind win rate of all setup bars. sel_WR = win rate of model-selected bars.\n\n")
            va_rows = []
            results_df_local = pd.DataFrame(all_results)
            for stop_atr_k in sorted(trained_models.keys()):
                res = trained_models[stop_atr_k]
                base_wr_train = res.get('train_win_rate', float('nan'))
                base_wr_test = res.get('test_win_rate', float('nan'))

                # Model-selected WR on TRAIN set (matching test selection logic)
                pred_train = res.get('pred_train')
                y_train_raw = res.get('y_train_raw')
                rf_wr_train = float('nan')
                rf_n_train = 0
                if pred_train is not None and y_train_raw is not None and len(pred_train) > 0:
                    y_tr_arr = np.asarray(y_train_raw)
                    if args.select_mode == 'prob_weighted':
                        prob_thr = args.prob_threshold
                        sel_mask_tr = pred_train >= prob_thr
                        if sel_mask_tr.sum() > 0:
                            rf_wr_train = float(y_tr_arr[sel_mask_tr].mean())
                            rf_n_train = int(sel_mask_tr.sum())
                    else:
                        top_n_val = int(args.top_n)
                        order_tr = np.argsort(pred_train)[::-1]
                        n_sel_tr = min(top_n_val, len(order_tr))
                        top_idx_tr = order_tr[:n_sel_tr]
                        rf_wr_train = float(y_tr_arr[top_idx_tr].mean())
                        rf_n_train = n_sel_tr                # RF-selected WR on TEST set (from results table)
                rf_wr_test = float('nan')
                rf_n_test = 0
                match = results_df_local[results_df_local['stop_atr'] == stop_atr_k]
                if len(match) > 0:
                    rf_wr_test = float(match.iloc[0].get('win_rate', float('nan')))
                    rf_n_test = int(match.iloc[0].get('n_trades_executed', 0))

                # F1 and AUC on full train/test sets
                f1_tr_val = float('nan')
                f1_te_val = float('nan')
                auc_tr_val = float('nan')
                auc_te_val = float('nan')
                if pred_train is not None and y_train_raw is not None and len(pred_train) > 0:
                    y_tr_arr2 = np.asarray(y_train_raw)
                    y_pred_tr2 = (pred_train >= 0.5).astype(int)
                    try:
                        f1_tr_val = f1_score(y_tr_arr2, y_pred_tr2)
                    except Exception:
                        pass
                    try:
                        auc_tr_val = roc_auc_score(y_tr_arr2, pred_train)
                    except Exception:
                        pass
                proba_test_md = res.get('proba_test')
                y_test_md = res.get('y_test')
                if proba_test_md is not None and y_test_md is not None and len(proba_test_md) > 0:
                    y_te_arr2 = np.asarray(y_test_md)
                    y_pred_te2 = (np.asarray(proba_test_md) >= 0.5).astype(int)
                    try:
                        f1_te_val = f1_score(y_te_arr2, y_pred_te2)
                    except Exception:
                        pass
                    try:
                        auc_te_val = roc_auc_score(y_te_arr2, proba_test_md)
                    except Exception:
                        pass

                va_rows.append({
                    'stop_atr': stop_atr_k,
                    'base_wr_train': round(base_wr_train * 100, 2),
                    'rf_wr_train': round(rf_wr_train * 100, 2),
                    'lift_train': round((rf_wr_train - base_wr_train) * 100, 2),
                    'rf_n_train': rf_n_train,
                    'base_wr_test': round(base_wr_test * 100, 2),
                    'rf_wr_test': round(rf_wr_test * 100, 2),
                    'lift_test': round((rf_wr_test - base_wr_test) * 100, 2),
                    'rf_n_test': rf_n_test,
                    'f1_train': round(f1_tr_val, 3) if not np.isnan(f1_tr_val) else None,
                    'f1_test': round(f1_te_val, 3) if not np.isnan(f1_te_val) else None,
                    'auc_train': round(auc_tr_val, 3) if not np.isnan(auc_tr_val) else None,
                    'auc_test': round(auc_te_val, 3) if not np.isnan(auc_te_val) else None,
                })
            va_df = pd.DataFrame(va_rows)
            f.write(va_df.to_markdown(index=False))
            f.write("\n")

        # ----------------------------------------------------------------
        # YEARLY P&L BREAKDOWN
        # ----------------------------------------------------------------
        yearly_rows_all = []
        for r in all_results:
            yp = r.get('yearly_pnl', {})
            if yp:
                for yr in sorted(yp.keys()):                    yearly_rows_all.append({
                        'stop_atr': r['stop_atr'],
                        'selection': r.get('selection', ''),
                        'year': yr,
                        'n_trades': yp[yr]['n_trades'],
                        'total_net_pnl': round(yp[yr]['total_net_pnl'], 2),
                        'win_rate': round(yp[yr]['win_rate'] * 100, 1),
                        'avg_net_pnl': round(yp[yr]['avg_net_pnl'], 2),
                    })
        if yearly_rows_all:
            f.write("\n## Yearly P&L Breakdown\n\n")
            yearly_df = pd.DataFrame(yearly_rows_all)
            f.write(yearly_df.to_markdown(index=False))
            f.write("\n")

        # ----------------------------------------------------------------
        # TRADE SIZE DISTRIBUTION (best stop only)
        # ----------------------------------------------------------------
        if recommended_stop is not None:
            best_row = None
            for r in all_results:
                if r['stop_atr'] == recommended_stop and r.get('share_dist'):
                    best_row = r
                    break
            if best_row:
                sd = best_row['share_dist']
                f.write("\n## Trade Size Distribution\n\n")
                f.write(f"Analysis for best stop (**{recommended_stop:.2f} ATR**), "
                        f"{best_row.get('n_trades_executed', 0):,} executed trades. "
                        f"Capital = ${float(CAPITAL_CAP):,.0f}.\n\n")

                # Share size stats
                f.write("### Share Size Stats\n\n")
                f.write("| Stat | Value |\n")
                f.write("|:-----|------:|\n")
                for label, key, fmt in [
                    ('Mean', 'shares_mean', '{:,.0f}'),
                    ('Median', 'shares_median', '{:,.0f}'),
                    ('Std Dev', 'shares_std', '{:,.0f}'),
                    ('Min', 'shares_min', '{:,.0f}'),
                    ('Max', 'shares_max', '{:,.0f}'),
                    ('P05', 'shares_p05', '{:,.0f}'),
                    ('P25', 'shares_p25', '{:,.0f}'),
                    ('P75', 'shares_p75', '{:,.0f}'),
                    ('P95', 'shares_p95', '{:,.0f}'),
                ]:
                    if key in sd:
                        f.write(f"| {label} | {fmt.format(sd[key])} |\n")
                f.write("\n")

                # Share size buckets
                buckets = sd.get('share_buckets', {})
                if buckets:
                    n_total = sum(buckets.values())
                    f.write("### Share Size Buckets\n\n")
                    f.write("| Bucket | Count | Pct |\n")
                    f.write("|:-------|------:|----:|\n")
                    for bkt, cnt in sorted(buckets.items(), key=lambda x: int(x[0].split('-')[0].replace('K', '000').replace('k', '000'))):
                        pct = cnt / max(1, n_total) * 100
                        f.write(f"| {bkt} | {cnt:,} | {pct:.1f}% |\n")
                    f.write("\n")

                # Notional distribution
                f.write("### Notional Distribution\n\n")
                f.write("| Stat | Value |\n")
                f.write("|:-----|------:|\n")
                for label, key, fmt in [
                    ('Mean', 'notional_mean', '${:,.0f}'),
                    ('Median', 'notional_median', '${:,.0f}'),
                    ('Min', 'notional_min', '${:,.0f}'),
                    ('Max', 'notional_max', '${:,.0f}'),
                ]:
                    if key in sd:
                        f.write(f"| {label} | {fmt.format(sd[key])} |\n")
                f.write("\n")

                # Risk per trade
                f.write("### Risk Per Trade\n\n")
                f.write("| Stat | Value |\n")
                f.write("|:-----|------:|\n")
                for label, key, fmt in [
                    ('Mean', 'risk_mean', '${:,.0f}'),
                    ('Median', 'risk_median', '${:,.0f}'),
                    ('Min', 'risk_min', '${:,.0f}'),
                    ('Max', 'risk_max', '${:,.0f}'),
                ]:
                    if key in sd:
                        f.write(f"| {label} | {fmt.format(sd[key])} |\n")
                f.write("\n")

                # Capital cap vs risk sizing
                n_cap = sd.get('n_capital_capped', 0)
                n_risk = sd.get('n_risk_sized', 0)
                pct_cap = sd.get('pct_capital_capped', 0)
                if n_cap + n_risk > 0:
                    f.write("### Sizing Constraint Breakdown\n\n")
                    f.write(f"- **Capital-capped** (notional ≥ 99.9% of capital): "
                            f"{n_cap:,} trades ({pct_cap:.1f}%)\n")
                    f.write(f"- **Risk-sized** (notional < 99.9% of capital): "
                            f"{n_risk:,} trades ({100 - pct_cap:.1f}%)\n\n")

                # Long vs Short
                if sd.get('long_n_trades') or sd.get('short_n_trades'):
                    f.write("### Long vs Short\n\n")
                    f.write("| Direction | Trades | Avg Shares | Avg Notional | Avg P&L | Total P&L |\n")
                    f.write("|:----------|-------:|-----------:|-------------:|--------:|----------:|\n")
                    for direction in ['long', 'short']:
                        nt = sd.get(f'{direction}_n_trades', 0)
                        if nt > 0:
                            f.write(f"| {direction.title()} "
                                    f"| {nt:,} "
                                    f"| {sd.get(f'{direction}_avg_shares', 0):,.0f} "
                                    f"| ${sd.get(f'{direction}_avg_notional', 0):,.0f} "
                                    f"| ${sd.get(f'{direction}_avg_net_pnl', 0):,.0f} "
                                    f"| ${sd.get(f'{direction}_total_net_pnl', 0):,.0f} |\n")
                    f.write("\n")

                # Exit reason breakdown
                exit_bd = sd.get('exit_breakdown', {})
                if exit_bd:
                    f.write("### By Exit Reason\n\n")
                    f.write("| Exit | Trades | Avg Shares | Avg Notional | Total P&L |\n")
                    f.write("|:-----|-------:|-----------:|-------------:|----------:|\n")
                    for reason in sorted(exit_bd.keys()):
                        eb = exit_bd[reason]
                        f.write(f"| {reason} "
                                f"| {eb['n_trades']:,} "
                                f"| {eb['avg_shares']:,.0f} "
                                f"| ${eb['avg_notional']:,.0f} "
                                f"| ${eb['total_net_pnl']:,.0f} |\n")
                    f.write("\n")

        # ----------------------------------------------------------------
        # FEATURE IMPORTANCE RANKINGS
        # ----------------------------------------------------------------
        if trained_models:
            f.write("\n## Feature Importance Rankings\n")

            # Collect importance DataFrames keyed by stop_atr
            imp_by_stop = {}
            for stop_atr_k, res in sorted(trained_models.items()):
                imp_df = res.get('importance')
                if imp_df is not None and len(imp_df) > 0:
                    imp_by_stop[stop_atr_k] = imp_df.set_index('feature')['importance']

            if imp_by_stop:
                # Build a combined DataFrame: features × stops
                imp_all = pd.DataFrame(imp_by_stop)
                imp_all.columns = [f"stop_{c}" for c in imp_all.columns]
                imp_all['avg'] = imp_all.mean(axis=1)
                imp_all = imp_all.sort_values('avg', ascending=False)

                # --- Average ranking table ---
                f.write("\n### Average across all stops\n\n")
                f.write("| Rank | Feature | Avg Importance |\n")
                f.write("|-----:|:--------|---------------:|\n")
                for rank, (feat, row) in enumerate(imp_all.iterrows(), 1):
                    f.write(f"| {rank} | {feat} | {row['avg']:.4f} |\n")

                # --- Per-stop breakdown table ---
                f.write("\n### Per-stop breakdown\n\n")
                # Format as a wide table: Feature | avg | stop_0.25 | stop_0.35 | ...
                display_df = imp_all.copy()
                display_df.insert(0, 'feature', display_df.index)
                display_df = display_df.reset_index(drop=True)
                # Round for readability
                num_cols = [c for c in display_df.columns if c != 'feature']
                display_df[num_cols] = display_df[num_cols].round(4)
                f.write(display_df.to_markdown(index=False))
                f.write("\n")

                # --- Print to console as well ---
                print(f"\n{'='*70}")
                print("FEATURE IMPORTANCE RANKINGS (avg across all stops)")
                print(f"{'='*70}")
                for rank, (feat, row) in enumerate(imp_all.iterrows(), 1):
                    bar = '#' * min(int(row['avg'] * 300), 50)
                    print(f"  {rank:2d}. {feat:<28s} {row['avg']:.4f}  {bar}")
    # --- Print RF Value-Add table to console ---
    if trained_models and len(all_results) > 0:
        print(f"\n{'='*120}")
        print("RF MODEL VALUE-ADD (base WR vs RF-selected WR)")
        print(f"{'='*120}")
        # Header
        print(f"  {'stop':>5s}  {'all_tr':>7s}  {'sel_tr':>7s}  {'lift_tr':>8s}  {'n_tr':>5s}"
              f"  {'all_te':>7s}  {'sel_te':>7s}  {'lift_te':>8s}  {'n_te':>5s}"
              f"  {'f1_tr':>6s}  {'f1_te':>6s}  {'auc_tr':>6s}  {'auc_te':>6s}")
        print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*5}"
              f"  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*5}"
              f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
        results_df_local2 = pd.DataFrame(all_results)
        for stop_atr_k in sorted(trained_models.keys()):
            res = trained_models[stop_atr_k]
            base_tr = res.get('train_win_rate', float('nan')) * 100
            base_te = res.get('test_win_rate', float('nan')) * 100

            pred_train = res.get('pred_train')
            y_train_raw = res.get('y_train_raw')
            y_test_labels = res.get('y_test')
            proba_test = res.get('proba_test')
            rf_tr = float('nan')
            n_tr = 0
            f1_tr = float('nan')
            f1_te = float('nan')
            auc_tr = float('nan')
            auc_te = float('nan')

            if pred_train is not None and y_train_raw is not None and len(pred_train) > 0:
                y_arr_tr = np.asarray(y_train_raw)
                if args.select_mode == 'prob_weighted':
                    prob_thr = args.prob_threshold
                    sel_mask_tr = pred_train >= prob_thr
                    if sel_mask_tr.sum() > 0:
                        rf_tr = float(y_arr_tr[sel_mask_tr].mean()) * 100
                        n_tr = int(sel_mask_tr.sum())
                else:
                    top_n_val = int(args.top_n)
                    order_tr = np.argsort(pred_train)[::-1]
                    n_sel_tr = min(top_n_val, len(order_tr))
                    top_idx_tr = order_tr[:n_sel_tr]
                    rf_tr = float(y_arr_tr[top_idx_tr].mean()) * 100
                    n_tr = n_sel_tr

                # F1 and AUC on FULL train set (not just selected subset)
                y_pred_tr = (pred_train >= 0.5).astype(int)
                try:
                    f1_tr = f1_score(y_arr_tr, y_pred_tr)
                except Exception:
                    pass
                try:
                    auc_tr = roc_auc_score(y_arr_tr, pred_train)
                except Exception:
                    pass

            rf_te = float('nan')
            n_te = 0
            match = results_df_local2[results_df_local2['stop_atr'] == stop_atr_k]
            if len(match) > 0:
                rf_te = float(match.iloc[0].get('win_rate', float('nan'))) * 100
                n_te = int(match.iloc[0].get('n_trades_executed', 0))

            # F1 and AUC on FULL test set
            if proba_test is not None and y_test_labels is not None and len(proba_test) > 0:
                y_arr_te = np.asarray(y_test_labels)
                y_pred_te = (np.asarray(proba_test) >= 0.5).astype(int)
                try:
                    f1_te = f1_score(y_arr_te, y_pred_te)
                except Exception:
                    pass
                try:
                    auc_te = roc_auc_score(y_arr_te, proba_test)
                except Exception:
                    pass

            lift_tr = rf_tr - base_tr
            lift_te = rf_te - base_te
            sign_tr = '+' if lift_tr >= 0 else ''
            sign_te = '+' if lift_te >= 0 else ''
            f1_tr_s = f"{f1_tr:.3f}" if not np.isnan(f1_tr) else "  n/a"
            f1_te_s = f"{f1_te:.3f}" if not np.isnan(f1_te) else "  n/a"
            auc_tr_s = f"{auc_tr:.3f}" if not np.isnan(auc_tr) else "  n/a"
            auc_te_s = f"{auc_te:.3f}" if not np.isnan(auc_te) else "  n/a"
            print(f"  {stop_atr_k:5.2f}  {base_tr:6.1f}%  {rf_tr:6.1f}%  {sign_tr}{lift_tr:6.1f}%  {n_tr:5d}"
                  f"  {base_te:6.1f}%  {rf_te:6.1f}%  {sign_te}{lift_te:6.1f}%  {n_te:5d}"
                  f"  {f1_tr_s}  {f1_te_s}  {auc_tr_s}  {auc_te_s}")

    # --- Print Total P&L summary to console ---
    if args.model_kind == 'classifier' and args.select_mode == 'prob_weighted':
        _sel_f = f"prob_{args.prob_threshold:.2f}"
    elif args.model_kind == 'classifier':
        _sel_f = '0.5'
    else:
        _sel_f = f"top_{int(max(0, args.top_n))}"
    _pnl_v = results_df[results_df['selection'].astype(str) == _sel_f].copy()
    if len(_pnl_v) > 0 and 'total_net_pnl' in _pnl_v.columns:
        _pnl_v = _pnl_v.sort_values('stop_atr')
        print(f"\n{'='*70}")
        print(f"TOTAL P&L ON TEST YEARS ({test_label})")
        print(f"{'='*70}")
        for _, _r in _pnl_v.iterrows():
            _wr = f"{_r['win_rate']*100:.1f}%" if not pd.isna(_r.get('win_rate')) else "n/a"
            print(f"  stop={_r['stop_atr']:.2f}  trades={int(_r['n_trades_executed']):>5,}  "
                  f"WR={_wr:>6s}  P&L=${_r['total_net_pnl']:>10,.0f}  "
                  f"avg=${_r['avg_net_pnl_per_trade']:>8,.0f}/trade")
        _best = _pnl_v.sort_values('total_net_pnl', ascending=False).iloc[0]
        _total = _pnl_v['total_net_pnl'].sum()
        print(f"  {'-'*60}")
        print(f"  BEST STOP: {_best['stop_atr']:.2f} ATR -> ${_best['total_net_pnl']:,.0f}")
        print(f"  GRAND TOTAL (all stops): ${_total:,.0f}")

    print(f"\n[OK] Wrote results: {results_csv}")
    print(f"[OK] Wrote summary: {summary_md}")

    # ------------------------------------------------------------------------
    # SAVE MODELS (optional)
    # ------------------------------------------------------------------------
    try:
        for stop_atr, res in trained_models.items():
            model = res.get('model')
            if model is None:
                continue
            model_path = MODELS_DIR / f"rf_vwap_stop{stop_atr}_{timestamp}.pkl"
            save_model(
                model,
                str(model_path),
                metadata={
                    'stop_atr': float(stop_atr),
                    'model_kind': res.get('model_kind', args.model_kind),
                    'rf_params': res.get('rf_params', {}),
                    'features': res.get('features', []),
                    'label_mode': args.label_mode,
                    'min_net_r': float(args.min_net_r),
                    'sample_weight': args.sample_weight,
                    'slippage': float(args.slippage),
                    'test_year': int(TEST_YEAR),
                    'regression_target': args.regression_target if args.model_kind == 'regressor' else None,
                    'target_mode': 'vwap',   # target = VWAP touch (matches label_generator)
                },
            )
    except Exception as e:
        print(f"[WARN] Model saving failed: {e}")

    # ------------------------------------------------------------------------
    # OPTIONAL WALK-FORWARD
    # ------------------------------------------------------------------------
    if args.walk_forward:
        wf_fn = globals().get('walk_forward_resample_fixed_stop', None)
        if wf_fn is None:
            print("[WARN] walk_forward_resample_fixed_stop() not found; skipping walk-forward.")
        else:
            wf_stop = args.wf_stop_atr if args.wf_stop_atr is not None else recommended_stop
            if wf_stop is None:
                print("[WARN] No stop_atr found for walk-forward; skipping.")
            else:
                try:
                    wf_df = wf_fn(
                        df=df,
                        stop_atr=float(wf_stop),
                        features=features,
                        start_train_year=int(args.wf_start_train_year),
                        start_test_year=int(args.wf_start_test_year),
                        end_test_year=int(args.wf_end_test_year),
                        threshold=float(args.wf_threshold),
                        label_mode=args.label_mode,
                        sample_weight_mode=args.sample_weight,
                        slippage_per_share=float(args.slippage),
                        min_net_r=float(args.min_net_r),
                        model_kind=args._display_model_kind,
                        regression_target=args.regression_target,
                        select_mode=args.select_mode,
                        top_n=int(args.top_n),
                        min_rr=float(args.min_rr),
                    )
                    # Append to markdown
                    with open(summary_md, 'a', encoding='utf-8') as f:
                        f.write("\n\n## Walk-forward results\n\n")
                        f.write(wf_df.to_markdown(index=False))
                        f.write("\n")
                    print("[OK] Appended walk-forward results to summary")
                except Exception as e:
                    print(f"[WARN] Walk-forward failed: {e}")

    return results_df


if __name__ == "__main__":
    results = main()
