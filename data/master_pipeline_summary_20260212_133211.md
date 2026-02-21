# Master pipeline summary (20260212_133211)

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
|       0.25 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.116 | 6.0471  |         1677.58 |           0.167758 |                 3.35516 |            55.6554 |
|       0.35 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.154 | 4.31936 |         1068.33 |           0.106833 |                 2.13666 |            71.2739 |
|       0.5  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.186 | 3.02355 |        -5603.36 |          -0.560336 |               -11.2067  |           116.363  |
|       0.4  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.148 | 3.77944 |        -6414.33 |          -0.641433 |               -12.8287  |            87.5189 |
|       0.6  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.164 | 2.51962 |       -18133.4  |          -1.81334  |               -36.2669  |           131.553  |
|       0.75 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.224 | 2.0157  |       -19132.4  |          -1.91324  |               -38.2648  |           173.567  |
|       1.5  | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.448 | 1.00785 |       -24282.7  |          -2.42827  |               -48.5653  |           309.77   |
|       1.25 | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.342 | 1.20942 |       -32458.2  |          -3.24582  |               -64.9164  |           264.069  |
|       1    | regressor    | top_500     |            nan | net_pnl             |        500 |                 500 |      0.232 | 1.51177 |       -37100.4  |          -3.71004  |               -74.2007  |           202.201  |
