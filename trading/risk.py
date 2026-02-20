"""
Risk Management — position sizing, share calculation, margin cost.

Matches master_pipeline.py / simulate_streaming_clean.py exactly.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from trading.config import (
    CAPITAL,
    COMMISSION_PER_SHARE,
    MAX_POSITION_PCT,
    SLIPPAGE_PER_SHARE,
)

# ── IBKR margin tier table (same as master_pipeline.py) ─────────────────
IBKR_MARGIN_RATE_TIERS: List[Tuple[float, float]] = [
    (100_000,       0.0683),   # ≤ $100K:  BM + 1.50%
    (1_000_000,     0.0633),   # $100K–$1M: BM + 1.00%
    (50_000_000,    0.0608),   # $1M–$50M:  BM + 0.75%
    (200_000_000,   0.0583),   # $50M–$200M: BM + 0.50%
    (float('inf'),  0.0558),   # >$200M:    BM + 0.25%
]
IBKR_DAYS_PER_YEAR = 360       # IBKR interest day-count convention


def ibkr_margin_cost(borrowed: float, hold_hours: float) -> float:
    """Compute IBKR tiered margin interest.

    Identical to master_pipeline.py — prorates each tier's annual rate to the
    actual hold duration: tier_amount × rate × hold_hours / (360 × 24).
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
        total_interest += tier_amount * rate * hold_hours / (IBKR_DAYS_PER_YEAR * 24.0)
        remaining -= tier_amount
        prev_bound = upper_bound
    return total_interest


def compute_shares(
    prob: float,
    threshold: float,
    atr: float,
    stop_atr: float,
    risk_pct: float,
    min_frac: float,
    capital: float,
    entry_price: float,
) -> int:
    """Risk-based, probability-scaled share sizing with notional cap.

    Formula (matches master_pipeline.py ``prob_weighted`` mode):
        raw_scale  = clip((prob − threshold) / (1 − threshold), 0, 1)
        scale_frac = min_frac + raw_scale × (1 − min_frac)
        risk$      = capital × risk_pct
        full_shares = ⌊risk$ / (stop_atr × atr)⌋
        shares      = round(full_shares × scale_frac), clipped [1, 9999]
        shares      = min(shares, ⌊capital / entry_price⌋)   ← notional cap
    """
    prob_range = max(1.0 - threshold, 0.01)
    raw_scale = max(0.0, min(1.0, (prob - threshold) / prob_range))
    scale_frac = min_frac + raw_scale * (1.0 - min_frac)

    risk_dollars = capital * risk_pct
    stop_risk_per_share = stop_atr * atr
    if stop_risk_per_share <= 0:
        stop_risk_per_share = 1.0

    full_shares = int(math.floor(risk_dollars / stop_risk_per_share))
    shares = int(round(full_shares * scale_frac))

    # Cap notional to fraction of capital (avoid oversizing for account)
    if entry_price > 0:
        max_notional = capital * MAX_POSITION_PCT
        max_shares_by_capital = int(math.floor(max_notional / entry_price))
        shares = min(shares, max_shares_by_capital)

    return max(1, min(9999, shares))


def round_trip_cost(shares: int) -> float:
    """Commission + slippage for a full round-trip (entry + exit)."""
    return 2.0 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * shares
