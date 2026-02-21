# master_pipeline2 summary

- Year (test): 2024
- Train years: [2016, 2024)
- Validation (Option A): 2023
- Stop (ATR): 1.5
- Proba threshold: 0.825 (auto(validation=2023))
- Sizing: 1.0% risk/trade
- Capital cap: 1,000,000
- Costs (round-trip): 2*(commission+slippage)*shares = 2*(0.005+0.01)*shares

## Semantics
- Model target: classification: P(net_pnl>0) under realized-path execution + costs
- P&L definition: realized_path_dollars
- Win definitions: WR(label)=target-first, WR(net)=net_pnl>0

## Model fit diagnostics (classification)
Metrics computed on `net_profitable = (net_pnl>0)` labels.
- AUC/logloss/Brier use probabilities; acc/confusion use threshold=0.5.

### train
- n: 156,359
- base_rate (P(y=1)): 0.4216
- AUC: 0.7813
- logloss: 0.5726
- brier: 0.1957
- acc@0.5: 0.7002
- confusion@0.5 (tn fp / fn tp): 64269 26173 / 20705 45212

### valid
- n: 19,428
- base_rate (P(y=1)): 0.5018
- AUC: 0.7543
- logloss: 0.6103
- brier: 0.2110
- acc@0.5: 0.6695
- confusion@0.5 (tn fp / fn tp): 5817 3862 / 2559 7190

### test
- n: 19,548
- base_rate (P(y=1)): 0.5163
- AUC: 0.6750
- logloss: 0.6433
- brier: 0.2268
- acc@0.5: 0.6123
- confusion@0.5 (tn fp / fn tp): 5037 4419 / 3160 6932

## Results
- Trades executed: 847
- WR(label): 88.67%
- WR(net): 88.67%
- Gross P&L: $102,228.71
- Net P&L: $881.42

## Exit reasons
- target: 751
- stop: 94
- eod: 2
## Test-year results by probability threshold
Test year = 2024. Backtest uses the same realized-path execution + costs + capital cap.

| Threshold | Executed | WR(net) | WR(label) | Gross P&L | Net P&L | Return % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 3,777 | 67.57% | 67.70% | $-677,249.59 | $-1,124,185.72 | -112.419% |
| 0.55 | 3,624 | 70.03% | 70.89% | $-594,677.74 | $-1,027,144.60 | -102.714% |
| 0.60 | 3,510 | 72.85% | 73.76% | $-607,682.00 | $-1,028,175.23 | -102.818% |
| 0.65 | 3,300 | 76.21% | 77.03% | $-408,881.15 | $-810,305.39 | -81.031% |
| 0.70 | 3,034 | 78.61% | 79.37% | $-277,922.87 | $-646,929.83 | -64.693% |
| 0.75 | 2,603 | 81.21% | 81.75% | $-67,295.71 | $-384,605.14 | -38.461% |
| 0.80 | 1,646 | 86.15% | 86.21% | $73,351.76 | $-129,944.17 | -12.994% |
