"""Sweep logistic C parameter to find optimal regularization strength.
Runs stop=1.50 only (fastest, best P&L) for each C value.
"""
import subprocess, sys, re

C_VALUES = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
PYTHON = r"c:\Users\Administrator\ibkr8\.venv\Scripts\python.exe"
SCRIPT = "scripts/master_pipeline.py"

results = []
for c_val in C_VALUES:
    # Patch C value in LOGISTIC_PARAMS before running
    # We'll use a wrapper approach: modify the constant at runtime via env or arg
    # Simpler: just run the full pipeline but only for stop=1.50
    # We need to add a way to pass C... let's just patch the file temporarily
    pass

# Actually, let's just read the pipeline code and do the core logic here
import os, importlib
os.chdir(r"c:\Users\Administrator\ibkr8")
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import pipeline functions
import master_pipeline as mp

# Load and prep data once
df = mp.load_and_validate_data(Path("data/tsla_5min_10years.csv"))
df = mp.calculate_core_indicators(df, verbose=False)
features = mp.get_feature_columns(df)

STOP = 1.50
TEST_YEAR = 2024

print(f"\n{'='*80}")
print(f"LOGISTIC C SWEEP — Stop {STOP} ATR, TSLA")
print(f"{'='*80}")
print(f"{'C':>8s}  {'tr_WR':>7s}  {'te_WR':>7s}  {'lift_tr':>8s}  {'lift_te':>8s}  {'base_tr':>8s}  {'base_te':>8s}")
print(f"{'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

for c_val in C_VALUES:
    from label_generator import LabelConfig, generate_labels
    
    label_col = f"label_{STOP}"
    config = LabelConfig(stop_atr=STOP, label_mode='touch_vwap')
    df_labeled = generate_labels(df, config)
    
    if label_col not in df_labeled.columns:
        print(f"  {c_val:8.4f}  SKIP (no label column)")
        continue
    
    # Setup filter
    setup_mask = mp.apply_setup_filter(df_labeled, STOP, **mp.SETUP_DEFAULTS)
    df_setup = df_labeled[setup_mask].copy()
    
    # Valid rows
    valid = df_setup[label_col].notna() & df_setup[features].notna().all(axis=1)
    df_valid = df_setup[valid].copy()
    
    X = df_valid[features]
    y = df_valid[label_col].astype(int)
    
    train_mask = df_valid['datetime'].dt.year < TEST_YEAR
    test_mask = ~train_mask
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    base_tr = y_train.mean() * 100
    base_te = y_test.mean() * 100
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    
    lr = LogisticRegression(
        C=c_val, penalty='l2', solver='lbfgs',
        max_iter=1000, random_state=42, class_weight='balanced'
    )
    lr.fit(X_tr_s, y_train)
    
    p_tr = lr.predict_proba(X_tr_s)[:, 1]
    p_te = lr.predict_proba(X_te_s)[:, 1]
    
    # Top 5000 training WR (same as value-add table)
    top_n = min(5000, len(p_tr))
    idx_tr = np.argsort(p_tr)[::-1][:top_n]
    rf_tr = float(np.asarray(y_train)[idx_tr].mean()) * 100
    
    # Test: prob > 0.50
    te_sel = p_te >= 0.50
    if te_sel.sum() > 0:
        rf_te = float(np.asarray(y_test)[te_sel].mean()) * 100
        n_te = int(te_sel.sum())
    else:
        rf_te = float('nan')
        n_te = 0
    
    lift_tr = rf_tr - base_tr
    lift_te = rf_te - base_te
    
    print(f"  {c_val:8.4f}  {rf_tr:6.1f}%  {rf_te:6.1f}%  {lift_tr:+7.1f}%  {lift_te:+7.1f}%  {base_tr:7.1f}%  {base_te:7.1f}%  n_te={n_te}")
