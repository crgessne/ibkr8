# RF VWAP Reversion Strategy - Analysis Results

**Generated**: 2026-02-11 14:31:09

**Dataset**: tsla_5min_10years.csv
**Test Period Definition**: year >= 2024 (aggregate across all years >= TEST_YEAR)
**Win Definition**: label
**P&L Definition**: label_rr

**Capital Constraint**: $1,000,000 (realistic capital-constrained simulation)
**Features Used**: 18
**Position Size**: 100 shares
**Commission**: $0.005/share
**Slippage**: $0.01/share
**Min R:R Filter**: 1.00 (reject trades with vwap_dist/stop < 1.00)

## Top Strategies (RF >= 0.50, Capital-Constrained @ $1M)

| Stop ATR | R:R | WR(selected) | WR(label) | EV | Signals | Executed | % Exec | Net P&L | Return % | Max Pos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.50 | 1.01 | 42.7% | 42.7% | -0.142 | 1,191 | 1,191 | 100.0% | $-40,342 | -4.0% | 0 |
| 1.00 | 1.51 | 43.5% | 43.5% | +0.092 | 4,627 | 4,627 | 100.0% | $-45,095 | -4.5% | 0 |
| 1.25 | 1.21 | 42.4% | 42.4% | -0.063 | 2,147 | 2,147 | 100.0% | $-51,736 | -5.2% | 0 |
| 0.75 | 2.02 | 40.4% | 40.4% | +0.219 | 7,777 | 7,777 | 100.0% | $-55,666 | -5.6% | 0 |
| 0.60 | 2.52 | 37.5% | 37.5% | +0.321 | 9,578 | 9,578 | 100.0% | $-59,006 | -5.9% | 0 |
| 0.50 | 3.02 | 34.7% | 34.7% | +0.395 | 11,027 | 11,027 | 100.0% | $-69,002 | -6.9% | 0 |
| 0.40 | 3.78 | 31.1% | 31.1% | +0.488 | 12,315 | 12,315 | 100.0% | $-79,128 | -7.9% | 0 |
| 0.35 | 4.32 | 29.1% | 29.1% | +0.549 | 12,572 | 12,572 | 100.0% | $-82,815 | -8.3% | 0 |
| 0.25 | 6.05 | 23.5% | 23.5% | +0.658 | 13,962 | 13,962 | 100.0% | $-92,908 | -9.3% | 0 |

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

