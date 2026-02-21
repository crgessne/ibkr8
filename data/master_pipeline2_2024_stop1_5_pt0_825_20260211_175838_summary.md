# master_pipeline2 summary

- Year (test): 2024
- Train years: [2016, 2024)
- Validation (Option A): 2023
- Stop (ATR): 1.5
- Proba threshold: 0.825 (auto(validation=2023))
- Sizing: 1.5% risk/trade
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
- Trades executed: 817
- WR(label): 88.37%
- WR(net): 88.37%
- Gross P&L: $117,331.96
- Net P&L: $12,183.25

## Exit reasons
- target: 722
- stop: 93
- eod: 2
## Test-year results by probability threshold
Test year = 2024. Backtest uses the same realized-path execution + costs + capital cap.

| Threshold | Executed | WR(net) | WR(label) | Gross P&L | Net P&L | Return % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 3,375 | 67.88% | 68.18% | $-729,107.48 | $-1,181,002.61 | -118.100% |
| 0.55 | 3,271 | 70.65% | 71.57% | $-650,926.45 | $-1,088,759.41 | -108.876% |
| 0.60 | 3,169 | 73.34% | 74.38% | $-649,817.04 | $-1,076,109.60 | -107.611% |
| 0.65 | 3,038 | 76.53% | 77.45% | $-449,630.39 | $-857,626.82 | -85.763% |
| 0.70 | 2,792 | 79.05% | 79.87% | $-298,662.57 | $-673,719.09 | -67.372% |
| 0.75 | 2,414 | 81.52% | 82.10% | $-48,389.88 | $-371,388.60 | -37.139% |
| 0.80 | 1,569 | 85.98% | 86.04% | $74,961.78 | $-133,931.85 | -13.393% |
