# master_pipeline2 summary

- Year (test): 2024
- Train years: [2016, 2024)
- Validation (Option A): 2023
- Stop (ATR): 1.5
- Proba threshold: 0.825 (auto(validation=2023))
- Sizing: 1.0% risk/trade
- Capital cap: 4,000,000
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
- Gross P&L: $408,967.26
- Net P&L: $3,540.63

## Exit reasons
- target: 751
- stop: 94
- eod: 2
## Test-year results by probability threshold
Test year = 2024. Backtest uses the same realized-path execution + costs + capital cap.

| Threshold | Executed | WR(net) | WR(label) | Gross P&L | Net P&L | Return % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 3,780 | 67.59% | 67.72% | $-2,709,223.27 | $-4,497,118.00 | -112.428% |
| 0.55 | 3,629 | 70.05% | 70.90% | $-2,378,839.12 | $-4,108,845.64 | -102.721% |
| 0.60 | 3,514 | 72.91% | 73.82% | $-2,430,868.77 | $-4,112,978.79 | -102.824% |
| 0.65 | 3,299 | 76.24% | 77.05% | $-1,635,690.85 | $-3,241,521.07 | -81.038% |
| 0.70 | 3,033 | 78.64% | 79.39% | $-1,111,731.06 | $-2,587,880.85 | -64.697% |
| 0.75 | 2,602 | 81.25% | 81.78% | $-269,157.83 | $-1,538,501.81 | -38.463% |
| 0.80 | 1,646 | 86.15% | 86.21% | $293,488.24 | $-519,764.75 | -12.994% |
