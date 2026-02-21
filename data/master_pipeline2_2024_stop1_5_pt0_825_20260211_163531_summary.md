# master_pipeline2 summary

- Year (test): 2024
- Train years: [2016, 2024)
- Validation (Option A): 2023
- Stop (ATR): 1.5
- Proba threshold: 0.825 (cli)
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
- Trades executed: 1
- WR(label): 0.00%
- WR(net): 0.00%
- Gross P&L: $-29,995.32
- Net P&L: $-30,124.65

## Exit reasons
- target: 0
- stop: 1
- eod: 0
## Test-year results by probability threshold
Test year = 2024. Backtest uses the same realized-path execution + costs + capital cap.

| Threshold | Executed | WR(net) | WR(label) | Gross P&L | Net P&L | Return % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 3 | 66.67% | 66.67% | $12,667.86 | $12,350.61 | 1.235% |
| 0.55 | 3 | 66.67% | 66.67% | $-16,693.30 | $-17,003.74 | -1.700% |
| 0.60 | 3 | 66.67% | 66.67% | $-16,693.30 | $-17,003.74 | -1.700% |
| 0.65 | 3 | 66.67% | 66.67% | $-16,693.30 | $-17,003.74 | -1.700% |
| 0.70 | 3 | 66.67% | 66.67% | $-16,693.30 | $-17,003.74 | -1.700% |
| 0.75 | 2 | 50.00% | 50.00% | $-21,441.66 | $-21,682.32 | -2.168% |
| 0.80 | 1 | 0.00% | 0.00% | $-29,997.99 | $-30,129.87 | -3.013% |
