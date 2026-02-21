# RF VWAP Reversion Strategy - Analysis Results

**Generated**: 2026-02-10 17:40:41

**Capital Constraint**: $1,000,000 (realistic capital-constrained simulation)
**Features Used**: 18
**Position Size**: 100 shares
**Commission**: $0.005/share
**Slippage**: $0.01/share

## Top Strategies (RF >= 0.50, Capital-Constrained @ $1M)

| Stop ATR | R:R | Win Rate | EV | Signals | Executed | % Exec | Net P&L | Return % | Max Pos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.50 | 1.01 | 66.4% | +0.334 | 20,884 | 20,884 | 100.0% | $1,583,733 | +158.4% | 1 |
| 1.25 | 1.21 | 64.8% | +0.433 | 19,257 | 19,257 | 100.0% | $1,582,482 | +158.2% | 1 |
| 1.00 | 1.51 | 60.9% | +0.529 | 18,712 | 18,712 | 100.0% | $1,507,371 | +150.7% | 1 |
| 0.75 | 2.02 | 54.0% | +0.628 | 18,472 | 18,472 | 100.0% | $1,326,815 | +132.7% | 1 |
| 0.60 | 2.52 | 48.3% | +0.702 | 18,177 | 18,177 | 100.0% | $1,168,354 | +116.8% | 1 |
| 0.50 | 3.02 | 43.2% | +0.740 | 18,131 | 18,131 | 100.0% | $1,035,702 | +103.6% | 1 |
| 0.40 | 3.78 | 37.0% | +0.770 | 17,955 | 17,955 | 100.0% | $849,422 | +84.9% | 1 |
| 0.35 | 4.32 | 33.7% | +0.793 | 17,453 | 17,453 | 100.0% | $739,607 | +74.0% | 1 |
| 0.25 | 6.05 | 25.6% | +0.802 | 17,403 | 17,403 | 100.0% | $528,391 | +52.8% | 1 |

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

