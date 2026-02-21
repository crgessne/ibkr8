# RF VWAP Reversion Strategy - Analysis Results

**Generated**: 2026-02-11 15:02:10

**Dataset**: tsla_5min_10years.csv
**Test Period Definition**: year >= 2024 (aggregate across all years >= TEST_YEAR)
**Win Definition**: label
**P&L Definition**: label_rr

**Capital Constraint**: $1,000,000 (realistic capital-constrained simulation)
**Features Used**: 18
**Position Size**: 100 shares
**Commission**: $0.005/share
**Slippage**: $0.01/share
**Min R:R Filter**: none

## Top Strategies (RF >= 0.50, Capital-Constrained @ $1M)

| Stop ATR | R:R | WR(selected) | WR(label) | EV | Signals | Executed | % Exec | Net P&L | Return % | Max Pos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 6.05 | 22.3% | 22.3% | +0.568 | 22,359 | 22,359 | 100.0% | $-208,475 | -20.8% | 0 |
| 0.40 | 3.78 | 32.6% | 32.6% | +0.556 | 22,229 | 22,229 | 100.0% | $-235,378 | -23.5% | 0 |
| 0.35 | 4.32 | 29.2% | 29.2% | +0.551 | 22,357 | 22,357 | 100.0% | $-239,890 | -24.0% | 0 |
| 0.50 | 3.02 | 37.5% | 37.5% | +0.509 | 22,847 | 22,847 | 100.0% | $-252,879 | -25.3% | 0 |
| 0.60 | 2.52 | 41.7% | 41.7% | +0.466 | 23,247 | 23,247 | 100.0% | $-268,442 | -26.8% | 0 |
| 0.75 | 2.02 | 46.5% | 46.5% | +0.403 | 23,960 | 23,960 | 100.0% | $-283,807 | -28.4% | 0 |
| 1.00 | 1.51 | 52.6% | 52.6% | +0.320 | 24,633 | 24,633 | 100.0% | $-323,720 | -32.4% | 0 |
| 1.25 | 1.21 | 56.5% | 56.5% | +0.249 | 25,197 | 25,197 | 100.0% | $-410,334 | -41.0% | 0 |
| 1.50 | 1.01 | 59.9% | 59.9% | +0.202 | 25,382 | 25,382 | 100.0% | $-483,289 | -48.3% | 0 |

## Top Features

1. `vwap_width_atr`
2. `price_to_vwap_atr`
3. `bars_from_vwap`
4. `wap`
5. `bar_count`
6. `vwap_slope`
7. `rel_vol`
8. `vwap_slope_5`
9. `bar_range_atr`
10. `vol_at_extension`
11. `vwap_helping`
12. `close_position`
13. `crossed_vwap`
14. `rsi`
15. `vol_ratio`