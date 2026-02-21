# Master pipeline summary (20260212_114432)

- Test year: 2024
- Model kind: regressor
- Regression target: realized_net_pnl
- Selection mode: threshold (top_n=100)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **1.50**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       1.5  | regressor    | top_100     |            nan | realized_net_pnl    |        100 |                 100 |       0.65 | 1.00785 |       1138.53   |         0.113853   |               11.3853   |           198.868  |
|       1.25 | regressor    | top_100     |            nan | realized_net_pnl    |        100 |                 100 |       0.59 | 1.20942 |        627.503  |         0.0627503  |                6.27503  |           149.119  |
|       1    | regressor    | top_100     |            nan | realized_net_pnl    |        100 |                 100 |       0.51 | 1.51177 |        -19.2202 |        -0.00192202 |               -0.192202 |           113.243  |
|       0.6  | regressor    | top_100     |            nan | realized_net_pnl    |        100 |                 100 |       0.34 | 2.51962 |       -476.781  |        -0.0476781  |               -4.76781  |            70.1246 |
|       0.25 | regressor    | top_100     |            nan | realized_net_pnl    |        100 |                 100 |       0.16 | 6.0471  |       -546.32   |        -0.054632   |               -5.4632   |            31.067  |
|       0.35 | regressor    | top_100     |            nan | realized_net_pnl    |        100 |                 100 |       0.24 | 4.31936 |       -768.44   |        -0.076844   |               -7.6844   |            48.1913 |
|       0.75 | regressor    | top_100     |            nan | realized_net_pnl    |        100 |                 100 |       0.39 | 2.0157  |       -889.985  |        -0.0889985  |               -8.89985  |            87.2555 |
|       0.5  | regressor    | top_100     |            nan | realized_net_pnl    |        100 |                 100 |       0.31 | 3.02355 |       -934.75   |        -0.093475   |               -9.3475   |            56.0586 |
|       0.4  | regressor    | top_100     |            nan | realized_net_pnl    |        100 |                 100 |       0.23 | 3.77944 |      -1465.74   |        -0.146574   |              -14.6574   |            52.2149 |
