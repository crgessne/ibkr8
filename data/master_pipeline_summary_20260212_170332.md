# Master pipeline summary (20260212_170332)

- Test year: 2024
- Model kind: regressor
- Regression target: net_pnl
- Selection mode: threshold (top_n=1000)
- Label mode: touch_vwap
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.60**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.6  | regressor    | top_1000    |            nan | net_pnl             |       1000 |                1000 |      0.293 | 2.51962 |        4156.82  |          0.415682  |                4.15682  |           135.562  |
|       0.25 | regressor    | top_1000    |            nan | net_pnl             |       1000 |                1000 |      0.149 | 6.0471  |        1864.73  |          0.186473  |                1.86473  |            53.0467 |
|       0.35 | regressor    | top_1000    |            nan | net_pnl             |       1000 |                1000 |      0.19  | 4.31936 |        -260.469 |         -0.0260469 |               -0.260469 |            73.7452 |
|       0.75 | regressor    | top_1000    |            nan | net_pnl             |       1000 |                1000 |      0.351 | 2.0157  |        -372.56  |         -0.037256  |               -0.37256  |           160.902  |
|       1    | regressor    | top_1000    |            nan | net_pnl             |       1000 |                1000 |      0.384 | 1.51177 |       -3487.96  |         -0.348796  |               -3.48796  |           206.811  |
|       0.4  | regressor    | top_1000    |            nan | net_pnl             |       1000 |                1000 |      0.2   | 3.77944 |       -5225.11  |         -0.522511  |               -5.22511  |            85.1433 |
|       0.5  | regressor    | top_1000    |            nan | net_pnl             |       1000 |                1000 |      0.237 | 3.02355 |       -6715.96  |         -0.671596  |               -6.71596  |           106.214  |
|       1.25 | regressor    | top_1000    |            nan | net_pnl             |       1000 |                1000 |      0.381 | 1.20942 |      -23398.7   |         -2.33987   |              -23.3987   |           241.797  |
|       1.5  | regressor    | top_1000    |            nan | net_pnl             |       1000 |                1000 |      0.372 | 1.00785 |      -38435.8   |         -3.84358   |              -38.4358   |           266.265  |
