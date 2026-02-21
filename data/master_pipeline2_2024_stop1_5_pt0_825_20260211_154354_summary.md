# master_pipeline2 summary

- Year (test): 2024
- Train years: [2016, 2024)
- Validation (Option A): 2023
- Stop (ATR): 1.5
- Proba threshold: 0.825 (auto(validation=2023))
- Shares/trade: 100
- Capital cap: 1,000,000
- Costs (round-trip): 2*(commission+slippage)*shares = 2*(0.005+0.01)*100

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
- Trades executed: 935
- WR(label): 88.88%
- WR(net): 88.88%
- Gross P&L: $4,827.65
- Net P&L: $2,022.65

## Exit reasons
- target: 831
- stop: 102
- eod: 2
## Test-year results by probability threshold
Test year = 2024. Backtest uses the same realized-path execution + costs + capital cap.

| Threshold | Executed | WR(net) | WR(label) | Gross P&L | Net P&L | Return % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 11,351 | 61.07% | 60.05% | $-87,346.64 | $-121,399.64 | -12.140% |
| 0.55 | 9,346 | 65.63% | 65.63% | $-48,397.81 | $-76,435.81 | -7.644% |
| 0.60 | 7,679 | 69.75% | 70.00% | $-37,047.16 | $-60,084.16 | -6.008% |
| 0.65 | 6,009 | 73.99% | 74.42% | $-19,363.68 | $-37,390.68 | -3.739% |
| 0.70 | 4,860 | 77.41% | 77.86% | $-11,480.57 | $-26,060.57 | -2.606% |
| 0.75 | 3,847 | 80.19% | 80.53% | $-4,688.51 | $-16,229.51 | -1.623% |
| 0.80 | 2,003 | 86.27% | 86.27% | $5,720.06 | $-288.94 | -0.029% |
