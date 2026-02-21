# RF VWAP Reversion Strategy - Analysis Results

**Generated**: 2026-02-11 13:55:21

**Dataset**: tsla_5min_10years.csv
**Test Period Definition**: year >= 2024 (aggregate across all years >= TEST_YEAR)
**Win Definition**: label
**P&L Definition**: label_rr

**Capital Constraint**: $1,000,000 (realistic capital-constrained simulation)
**Features Used**: 18
**Position Size**: 100 shares
**Commission**: $0.005/share
**Slippage**: $0.01/share
**Min R:R Filter**: 0.75 (reject trades with vwap_dist/stop < 0.75)

## Top Strategies (RF >= 0.50, Capital-Constrained @ $1M)

| Stop ATR | R:R | WR(selected) | WR(label) | EV | Signals | Executed | % Exec | Net P&L | Return % | Max Pos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 2.02 | 43.2% | 43.2% | +0.303 | 10,411 | 10,411 | 100.0% | $-71,182 | -7.1% | 0 |
| 1.00 | 1.51 | 46.7% | 46.7% | +0.173 | 8,017 | 8,017 | 100.0% | $-78,154 | -7.8% | 0 |
| 0.60 | 2.52 | 39.9% | 39.9% | +0.405 | 11,800 | 11,800 | 100.0% | $-79,013 | -7.9% | 0 |
| 0.50 | 3.02 | 36.9% | 36.9% | +0.486 | 12,869 | 12,869 | 100.0% | $-80,326 | -8.0% | 0 |
| 1.25 | 1.21 | 48.0% | 48.0% | +0.060 | 5,980 | 5,980 | 100.0% | $-91,328 | -9.1% | 0 |
| 0.40 | 3.78 | 32.7% | 32.7% | +0.562 | 13,789 | 13,789 | 100.0% | $-94,019 | -9.4% | 0 |
| 0.35 | 4.32 | 30.6% | 30.6% | +0.625 | 13,835 | 13,835 | 100.0% | $-95,313 | -9.5% | 0 |
| 0.25 | 6.05 | 24.2% | 24.2% | +0.709 | 14,890 | 14,890 | 100.0% | $-106,537 | -10.7% | 0 |
| 1.50 | 1.01 | 47.3% | 47.3% | -0.050 | 5,266 | 5,266 | 100.0% | $-128,708 | -12.9% | 0 |

## Top Features

1. `vwap_width_atr`
2. `price_to_vwap_atr`
3. `bars_from_vwap`
4. `crossed_vwap`
5. `rel_vol`
6. `vwap_slope`
7. `bar_range_atr`
8. `vwap_slope_5`
9. `vwap_helping`
10. `vol_at_extension`
11. `bar_count`
12. `rsi`
13. `close_position`
14. `wap`
15. `rsi_slope`

