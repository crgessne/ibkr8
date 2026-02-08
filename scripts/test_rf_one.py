"""Test RF with just one stop width."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from label_generator import LabelConfig, generate_labels

print("Loading data...")
df = pd.read_csv("../data/tsla_5min_10years_indicators.csv", parse_dates=['time'], index_col='time')
print(f"Loaded {len(df):,} bars")

print("\nGenerating labels for stop=0.5 ATR...")
config = LabelConfig(stop_atrs=[0.5])
df = generate_labels(df, config)
print(f"Label column: label_s0_5")
print(f"Valid labels: {df['label_s0_5'].notna().sum():,}")

print("\nPreparing features...")
exclude = ['date', 'year', 'open', 'high', 'low', 'close', 'volume']
features = [c for c in df.columns if c not in exclude and not c.startswith('label_') and df[c].dtype in ['float64', 'int64']]
print(f"Using {len(features)} features")

print("\nFiltering valid data...")
valid = df['label_s0_5'].notna()
X = df.loc[valid, features].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df.loc[valid, 'label_s0_5'].astype(int)
print(f"Valid samples: {len(X):,}")

print("\nSplit train/test by year...")
df['date'] = pd.to_datetime(df.index).date
df['year'] = pd.to_datetime(df.index).year
train_mask = (df.loc[valid, 'year'] < 2024).values
test_mask = (df.loc[valid, 'year'] >= 2024).values

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

print("\nTraining RF...")
rf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("\nTest results:")
proba = rf.predict_proba(X_test)[:, 1]
raw_wr = y_test.mean()
print(f"Raw WR: {raw_wr:.1%}")

for thresh in [0.4, 0.5, 0.6]:
    mask = proba >= thresh
    n = mask.sum()
    if n > 0:
        wr = y_test[mask].mean()
        print(f"RF>={thresh}: WR={wr:.1%} (N={n})")

print("\nDone!")
