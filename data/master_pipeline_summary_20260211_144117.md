# RF VWAP Reversion Strategy - Analysis Results

**Generated**: 2026-02-11 14:41:17

**Dataset**: tsla_5min_10years.csv
**Test Period Definition**: year >= 2024 (aggregate across all years >= TEST_YEAR)
**Win Definition**: label
**P&L Definition**: label_rr

**Capital Constraint**: $1,000,000 (realistic capital-constrained simulation)
**Features Used**: 18
**Position Size**: 100 shares
**Commission**: $0.005/share
**Slippage**: $0.01/share
**Min R:R Filter**: 1.50 (reject trades with vwap_dist/stop < 1.50)

## Top Strategies (RF >= 0.50, Capital-Constrained @ $1M)

| Stop ATR | R:R | WR(selected) | WR(label) | EV | Signals | Executed | % Exec | Net P&L | Return % | Max Pos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.51 | 32.0% | 32.0% | -0.196 | 25 | 25 | 100.0% | $-1,837 | -0.2% | 0 |
| 0.75 | 2.02 | 36.4% | 36.4% | +0.098 | 2,856 | 2,856 | 100.0% | $-13,918 | -1.4% | 0 |
| 0.60 | 2.52 | 33.6% | 33.6% | +0.183 | 5,408 | 5,408 | 100.0% | $-31,306 | -3.1% | 0 |
| 0.50 | 3.02 | 31.4% | 31.4% | +0.261 | 7,438 | 7,438 | 100.0% | $-42,512 | -4.3% | 0 |
| 0.40 | 3.78 | 28.5% | 28.5% | +0.363 | 9,359 | 9,359 | 100.0% | $-49,778 | -5.0% | 0 |
| 0.35 | 4.32 | 26.8% | 26.8% | +0.426 | 10,006 | 10,006 | 100.0% | $-57,051 | -5.7% | 0 |
| 0.25 | 6.05 | 22.1% | 22.1% | +0.555 | 12,154 | 12,154 | 100.0% | $-72,490 | -7.2% | 0 |

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

