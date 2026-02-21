# VWAP Reversion Strategy - Master Pipeline Results

**Generated**: 2026-02-07 14:12:04

**Dataset**: tsla_5min_10years.csv
**Train/Test Split**: Pre-2024 / 2024+
**Features Used**: 15

---

## Summary Table - RF Filtered (threshold ≥ 0.5)

| Stop ATR | R:R | BE WR | Raw WR | Raw EV | RF WR | RF EV | RF N | Net P&L |
|----------|-----|-------|--------|--------|-------|-------|------|----------|
| 0.25 | 6.05:1 | 14.2% | 15.3% | +0.079R | 26.4% | +0.858R | 16,222.0 | $821,677 |
| 0.35 | 4.32:1 | 18.8% | 20.1% | +0.069R | 34.2% | +0.817R | 16,860.0 | $1,154,660 |
| 0.40 | 3.78:1 | 20.9% | 22.3% | +0.067R | 37.7% | +0.800R | 17,033.0 | $1,311,132 |
| 0.50 | 3.02:1 | 24.9% | 26.2% | +0.056R | 43.8% | +0.763R | 17,397.0 | $1,607,626 |
| 0.60 | 2.52:1 | 28.4% | 29.6% | +0.041R | 48.8% | +0.717R | 17,669.0 | $1,847,517 |
| 0.75 | 2.02:1 | 33.2% | 33.7% | +0.016R | 54.4% | +0.641R | 18,066.0 | $2,118,434 |
| 1.00 | 1.51:1 | 39.8% | 38.8% | -0.026R | 61.1% | +0.536R | 18,445.0 | $2,415,363 |
| 1.25 | 1.21:1 | 45.3% | 42.4% | -0.063R | 65.6% | +0.449R | 18,727.0 | $2,570,279 |
| 1.50 | 1.01:1 | 49.8% | 45.2% | -0.093R | 67.9% | +0.363R | 19,615.0 | $2,613,986 |

---

## Top 10 Features (by importance)

Based on 0.25 ATR stop model:

1. `vwap_width_atr`
2. `min_rr`
3. `max_rr`
4. `avg_rr`
5. `bars_from_vwap`
6. `price_to_vwap_atr`
7. `rel_vol`
8. `bar_range_atr`
9. `crossed_vwap`
10. `vwap_slope`

---

## Key Findings

- **Best EV**: 0.25 ATR stop → +0.858R EV (26.4% WR)
- **Best P&L**: 1.50 ATR stop → $2,613,986 (19,615.0 trades)
- **Total Projected P&L**: $16,460,676 across 160,034.0 trades
- **Average EV**: +0.661R across all stop widths
- **Features**: 15 non-redundant indicators
