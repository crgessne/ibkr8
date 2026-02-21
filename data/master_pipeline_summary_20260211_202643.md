# Master pipeline summary (20260211_202643)

- Test year: 2024
- Model kind: regressor
- Regression target: net_pnl
- Selection mode: top (top_n=5000)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.25**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.25 | regressor    | top_5000    |            nan | net_pnl             |       5000 |                2773 |     0.1716 | 6.0471  |        -9473.87 |          -0.947387 |                -3.41647 |            37.5351 |
|       0.5  | regressor    | top_5000    |            nan | net_pnl             |       5000 |                1656 |     0.2684 | 3.02355 |       -10861.7  |          -1.08617  |                -6.55903 |            85.6585 |
|       0.35 | regressor    | top_5000    |            nan | net_pnl             |       5000 |                2201 |     0.1884 | 4.31936 |       -11324.7  |          -1.13247  |                -5.14524 |            51.4296 |
|       1.25 | regressor    | top_5000    |            nan | net_pnl             |       5000 |                1082 |     0.5228 | 1.20942 |       -14158.7  |          -1.41587  |               -13.0856  |           232.674  |
|       1    | regressor    | top_5000    |            nan | net_pnl             |       5000 |                1188 |     0.4658 | 1.51177 |       -14762.5  |          -1.47625  |               -12.4263  |           183.698  |
|       0.75 | regressor    | top_5000    |            nan | net_pnl             |       5000 |                1312 |     0.4042 | 2.0157  |       -15022.9  |          -1.50229  |               -11.4504  |           135.815  |
|       0.4  | regressor    | top_5000    |            nan | net_pnl             |       5000 |                2252 |     0.2212 | 3.77944 |       -15315.8  |          -1.53158  |                -6.80098 |            60.0284 |
|       1.5  | regressor    | top_5000    |            nan | net_pnl             |       5000 |                1062 |     0.575  | 1.00785 |       -17975.2  |          -1.79752  |               -16.9258  |           282.884  |
|       0.6  | regressor    | top_5000    |            nan | net_pnl             |       5000 |                1444 |     0.3438 | 2.51962 |       -18124.7  |          -1.81247  |               -12.5517  |           107.663  |
