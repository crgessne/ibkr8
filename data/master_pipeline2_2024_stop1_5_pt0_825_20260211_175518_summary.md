# master_pipeline2 summary

- Year (test): 2024
- Train years: [2016, 2024)
- Validation (Option A): 2023
- Stop (ATR): 1.5
- Proba threshold: 0.825 (auto(validation=2023))
- Sizing: 0.5% risk/trade
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
- Trades executed: 913
- WR(label): 88.72%
- WR(net): 88.72%
- Gross P&L: $84,715.72
- Net P&L: $8,898.34

## Exit reasons
- target: 810
- stop: 101
- eod: 2
## Test-year results by probability threshold
Test year = 2024. Backtest uses the same realized-path execution + costs + capital cap.

| Threshold | Executed | WR(net) | WR(label) | Gross P&L | Net P&L | Return % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 5,030 | 66.56% | 66.46% | $-603,980.02 | $-1,004,652.49 | -100.465% |
| 0.55 | 4,771 | 69.25% | 69.92% | $-531,363.21 | $-918,417.48 | -91.842% |
| 0.60 | 4,494 | 72.10% | 72.85% | $-446,280.77 | $-819,713.69 | -81.971% |
| 0.65 | 4,100 | 75.49% | 76.22% | $-295,971.89 | $-646,010.06 | -64.601% |
| 0.70 | 3,672 | 78.05% | 78.70% | $-232,652.61 | $-552,564.36 | -55.256% |
| 0.75 | 3,090 | 80.68% | 81.13% | $-85,131.85 | $-356,103.40 | -35.610% |
| 0.80 | 1,844 | 86.06% | 86.12% | $83,318.37 | $-81,338.40 | -8.134% |
