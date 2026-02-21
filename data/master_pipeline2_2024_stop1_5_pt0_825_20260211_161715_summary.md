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
- Trades executed: 244
- WR(label): 90.98%
- WR(net): 90.98%
- Gross P&L: $94,336.03
- Net P&L: $69,551.02

## Exit reasons
- target: 222
- stop: 21
- eod: 1
## Test-year results by probability threshold
Test year = 2024. Backtest uses the same realized-path execution + costs + capital cap.

| Threshold | Executed | WR(net) | WR(label) | Gross P&L | Net P&L | Return % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 746 | 71.72% | 71.31% | $-179,154.12 | $-258,165.27 | -25.817% |
| 0.55 | 724 | 73.62% | 73.62% | $-118,127.31 | $-194,209.41 | -19.421% |
| 0.60 | 704 | 76.14% | 76.14% | $-70,333.67 | $-144,332.84 | -14.433% |
| 0.65 | 667 | 79.46% | 79.46% | $-34,835.38 | $-104,561.17 | -10.456% |
| 0.70 | 640 | 81.72% | 81.72% | $11,456.91 | $-55,520.70 | -5.552% |
| 0.75 | 570 | 84.21% | 84.21% | $62,961.52 | $3,756.70 | 0.376% |
| 0.80 | 394 | 88.07% | 88.07% | $66,698.80 | $25,863.58 | 2.586% |
