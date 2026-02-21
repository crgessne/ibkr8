# Master pipeline summary (20260212_165137)

- Test year: 2024
- Model kind: regressor
- Regression target: net_pnl
- Selection mode: threshold (top_n=500)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.75**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.75 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.358 | 2.0157  |        7065.13  |          0.706513  |                14.1303  |           187.711  |
|       0.6  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.288 | 2.51962 |        7052.32  |          0.705232  |                14.1046  |           156.499  |
|       0.25 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.148 | 6.0471  |        4315.62  |          0.431562  |                 8.63124 |            58.4792 |
|       0.35 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.182 | 4.31936 |        3400.27  |          0.340027  |                 6.80054 |            80.2832 |
|       1    | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.402 | 1.51177 |        3341.61  |          0.334161  |                 6.68322 |           237.411  |
|       0.5  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.222 | 3.02355 |        -503.776 |         -0.0503776 |                -1.00755 |           118.106  |
|       0.4  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.166 | 3.77944 |       -4797.79  |         -0.479779  |                -9.59559 |            93.9914 |
|       1.25 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.392 | 1.20942 |      -13298.6   |         -1.32986   |               -26.5971  |           273.036  |
|       1.5  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.396 | 1.00785 |      -16437     |         -1.6437    |               -32.8741  |           307.661  |
