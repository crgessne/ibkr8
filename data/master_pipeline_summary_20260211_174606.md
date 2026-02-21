# Master pipeline summary (20260211_174606)

- Test year: 2024
- Model kind: regressor
- Regression target: net_r
- Selection mode: top (top_n=5000)
- Label mode: net_positive_r
- min_net_r: 0.25
- min_rr (post-filter): 0.0
- Slippage: 0.01

## Recommended stop_atr (by max total_net_pnl)

- stop_atr = **0.25**

## Results (sorted by total_net_pnl)

|   stop_atr | model_kind   | selection   |   rf_threshold | regression_target   |   n_trades |   n_trades_executed |   win_rate |      rr |   total_net_pnl |   total_return_pct |   avg_net_pnl_per_trade |   avg_risk_dollars |
|-----------:|:-------------|:------------|---------------:|:--------------------|-----------:|--------------------:|-----------:|--------:|----------------:|-------------------:|------------------------:|-------------------:|
|       0.25 | regressor    | top_5000    |            nan | net_r               |       5000 |                5000 |     0.1476 | 6.0471  |        -44892.9 |           -4.48929 |                -8.97857 |            51.1037 |
|       0.35 | regressor    | top_5000    |            nan | net_r               |       5000 |                5000 |     0.2054 | 4.31936 |        -54902.7 |           -5.49027 |               -10.9805  |            67.8068 |
|       0.4  | regressor    | top_5000    |            nan | net_r               |       5000 |                5000 |     0.222  | 3.77944 |        -69978.9 |           -6.99789 |               -13.9958  |            77.9479 |
|       0.5  | regressor    | top_5000    |            nan | net_r               |       5000 |                5000 |     0.2858 | 3.02355 |        -72189   |           -7.2189  |               -14.4378  |           100.061  |
|       0.75 | regressor    | top_5000    |            nan | net_r               |       5000 |                5000 |     0.4248 | 2.0157  |        -74847.6 |           -7.48476 |               -14.9695  |           150.627  |
|       0.6  | regressor    | top_5000    |            nan | net_r               |       5000 |                5000 |     0.3534 | 2.51962 |        -75836.2 |           -7.58362 |               -15.1672  |           122.185  |
|       1    | regressor    | top_5000    |            nan | net_r               |       5000 |                5000 |     0.5214 | 1.51177 |        -98512.8 |           -9.85128 |               -19.7026  |           192.85   |
|       1.25 | regressor    | top_5000    |            nan | net_r               |       5000 |                5000 |     0.5776 | 1.20942 |       -129230   |          -12.923   |               -25.8461  |           244.17   |
|       1.5  | regressor    | top_5000    |            nan | net_r               |       5000 |                5000 |     0.6738 | 1.00785 |       -140816   |          -14.0816  |               -28.1631  |           289.661  |
