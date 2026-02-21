# Master pipeline summary (20260212_173458)

- Test year: 2024
- Model kind: regressor
- Regression target: net_pnl
- Selection mode: threshold (top_n=500)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.25**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.25 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |   0.152    | 6.0471  |        7271.57  |          0.727157  |               14.5431   |            58.2657 |
|       0.6  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |   0.266    | 2.51962 |        3609     |          0.3609    |                7.218    |           150.471  |
|       0.35 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |   0.176    | 4.31936 |        3249.9   |          0.32499   |                6.49981  |            81.0572 |
|       0.4  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |   0.194    | 3.77944 |        2446.5   |          0.24465   |                4.89301  |            96.1827 |
|       0.5  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |   0.232    | 3.02355 |        -489.128 |         -0.0489128 |               -0.978255 |           112.807  |
|       0.75 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |   0.254    | 2.0157  |       -8536.24  |         -0.853624  |              -17.0725   |           164.095  |
|       1.5  | regressor    | top_500     |            nan | net_pnl             |        215 |                 215 |   0.190698 | 1.00785 |      -16054.1   |         -1.60541   |              -74.6705   |           231.477  |
|       1    | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |   0.22     | 1.51177 |      -22932.6   |         -2.29326   |              -45.8653   |           169.687  |
|       1.25 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |   0.206    | 1.20942 |      -26913.7   |         -2.69137   |              -53.8274   |           199.259  |
