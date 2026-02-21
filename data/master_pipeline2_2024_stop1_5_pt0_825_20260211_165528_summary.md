# master_pipeline2 summary

- Year (test): 2024
- Train years: [2016, 2024)
- Validation (Option A): 2023
- Stop (ATR): 1.5
- Proba threshold: 0.825 (auto(validation=2023))
- Sizing: 3.0% risk/trade
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
- Trades executed: 805
- WR(label): 88.45%
- WR(net): 88.45%
- Gross P&L: $107,560.61
- Net P&L: $1,778.93

## Exit reasons
- target: 712
- stop: 91
- eod: 2
## Test-year results by probability threshold
Test year = 2024. Backtest uses the same realized-path execution + costs + capital cap.

| Threshold | Executed | WR(net) | WR(label) | Gross P&L | Net P&L | Return % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 3,281 | 67.85% | 68.18% | $-746,876.64 | $-1,199,710.89 | -119.971% |
| 0.55 | 3,178 | 70.48% | 71.43% | $-664,933.10 | $-1,103,634.71 | -110.363% |
| 0.60 | 3,098 | 73.27% | 74.34% | $-670,252.89 | $-1,097,453.91 | -109.745% |
| 0.65 | 2,971 | 76.30% | 77.25% | $-492,668.30 | $-901,771.28 | -90.177% |
| 0.70 | 2,733 | 78.81% | 79.66% | $-335,536.68 | $-711,556.26 | -71.156% |
| 0.75 | 2,359 | 81.56% | 82.15% | $-50,770.43 | $-374,449.37 | -37.445% |
| 0.80 | 1,544 | 86.01% | 86.08% | $59,152.83 | $-150,564.36 | -15.056% |
