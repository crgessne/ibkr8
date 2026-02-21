"""
PyTorch NN PnL model classes.

Extracted from master_pipeline.py so they can be imported for pickle
deserialization from any script (not just master_pipeline as __main__).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as tnn
from sklearn.preprocessing import StandardScaler


class PnLNet(tnn.Module):
    """Small MLP that outputs a single logit (sigmoid -> trade probability).

    The network is trained with a *profit-aware* loss:
        loss = -mean( sigma(logit) * net_pnl_per_bar )  +  L2

    This directly optimises for expected dollar P&L, not classification accuracy.
    Bars with large positive net_pnl pull the output toward 1; bars with large
    negative net_pnl push it toward 0 -- weighted by magnitude.
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

    Exposes predict_proba(X) -> array of shape (n, 2) matching MLPClassifier API.
    Also exposes a dummy fit() so sklearn's permutation_importance accepts it.
    """

    def __init__(self, net, scaler, features):
        self.net = net
        self.scaler = scaler
        self.features = features

    def fit(self, X, y=None):
        """No-op -- model is already trained. Required by sklearn validation."""
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
