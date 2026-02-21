# Master pipeline summary (20260212_165306)

- Test year: 2024
- Model kind: regressor
- Regression target: net_pnl
- Selection mode: threshold (top_n=200)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.60**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.6  | regressor    | top_200     |            nan | net_pnl             |        200 |                 200 |      0.295 | 2.51962 |        6966.32  |          0.696632  |                34.8316  |           173.325  |
|       0.75 | regressor    | top_200     |            nan | net_pnl             |        200 |                 200 |      0.355 | 2.0157  |        5786.34  |          0.578634  |                28.9317  |           215.06   |
|       0.25 | regressor    | top_200     |            nan | net_pnl             |        200 |                 200 |      0.125 | 6.0471  |        1705.63  |          0.170563  |                 8.52813 |            67.0604 |
|       0.5  | regressor    | top_200     |            nan | net_pnl             |        200 |                 200 |      0.225 | 3.02355 |        1378.04  |          0.137804  |                 6.8902  |           131.731  |
|       0.35 | regressor    | top_200     |            nan | net_pnl             |        200 |                 200 |      0.145 | 4.31936 |         699.814 |          0.0699814 |                 3.49907 |            85.8355 |
|       0.4  | regressor    | top_200     |            nan | net_pnl             |        200 |                 200 |      0.14  | 3.77944 |       -1481.36  |         -0.148136  |                -7.4068  |           106.676  |
|       1    | regressor    | top_200     |            nan | net_pnl             |        200 |                 200 |      0.35  | 1.51177 |       -3158.88  |         -0.315888  |               -15.7944  |           267.304  |
|       1.5  | regressor    | top_200     |            nan | net_pnl             |        200 |                 200 |      0.355 | 1.00785 |      -12694.7   |         -1.26947   |               -63.4736  |           362.853  |
|       1.25 | regressor    | top_200     |            nan | net_pnl             |        200 |                 200 |      0.345 | 1.20942 |      -12820.5   |         -1.28205   |               -64.1024  |           326.6    |
