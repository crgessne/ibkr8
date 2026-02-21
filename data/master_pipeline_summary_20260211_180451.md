# Master pipeline summary (20260211_180451)

- Test year: 2024
- Model kind: regressor
- Regression target: net_pnl
- Selection mode: top (top_n=2000)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **1.50**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       1.5  | regressor    | top_2000    |            nan | net_pnl             |       2000 |                 556 |     0.5075 | 1.00785 |        24170.5  |           2.41705  |                43.4721  |           338.829  |
|       1.25 | regressor    | top_2000    |            nan | net_pnl             |       2000 |                 497 |     0.439  | 1.20942 |        14835.3  |           1.48353  |                29.8498  |           277.903  |
|       1    | regressor    | top_2000    |            nan | net_pnl             |       2000 |                 516 |     0.373  | 1.51177 |         3759.86 |           0.375986 |                 7.28655 |           221.06   |
|       0.6  | regressor    | top_2000    |            nan | net_pnl             |       2000 |                 672 |     0.2625 | 2.51962 |        -1055.01 |          -0.105501 |                -1.56996 |           129.672  |
|       0.75 | regressor    | top_2000    |            nan | net_pnl             |       2000 |                 594 |     0.307  | 2.0157  |        -1488.5  |          -0.14885  |                -2.5059  |           163.645  |
|       0.25 | regressor    | top_2000    |            nan | net_pnl             |       2000 |                1496 |     0.169  | 6.0471  |       -16923.3  |          -1.69233  |               -11.3124  |            44.7356 |
|       0.5  | regressor    | top_2000    |            nan | net_pnl             |       2000 |                 796 |     0.2095 | 3.02355 |       -17480.5  |          -1.74805  |               -21.9604  |           101.565  |
|       0.4  | regressor    | top_2000    |            nan | net_pnl             |       2000 |                1093 |     0.18   | 3.77944 |       -18867.5  |          -1.88675  |               -17.2621  |            74.6818 |
|       0.35 | regressor    | top_2000    |            nan | net_pnl             |       2000 |                1212 |     0.1525 | 4.31936 |       -27398.3  |          -2.73983  |               -22.6059  |            58.4418 |
