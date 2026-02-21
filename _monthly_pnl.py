"""Monthly P&L breakdown for best strategy config over test period (2024+)."""
import sys, warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from master_pipeline import (
    load_and_validate_data, calculate_core_indicators, get_feature_columns,
    apply_setup_filter, train_rf_model, DATA_FILE, TEST_YEAR,
    SHARES_PER_TRADE, COMMISSION_PER_SHARE, SLIPPAGE_PER_SHARE, STOP_ATRS,
)
from label_generator import LabelConfig, generate_labels

print("Loading data...")
df = load_and_validate_data(DATA_FILE)
df = calculate_core_indicators(df, verbose=False)
features = get_feature_columns(df)
config = LabelConfig(stop_atrs=[0.6, 0.75, 1.0, 1.25])
df = generate_labels(df, config)

costs_per_trade = 2.0 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * SHARES_PER_TRADE

for stop_atr in [1.25, 0.6]:
    print(f"\n{'='*80}")
    print(f"  STOP = {stop_atr} ATR  |  Setup Filter  |  Top-500 Regressor")
    print(f"{'='*80}")    result = train_rf_model(
        df, stop_atr, features, TEST_YEAR,
        model_kind='regressor', regression_target='net_r',
        setup_filter=True, min_rr=0.3, min_minutes_session=15,
        max_minutes_session=360,
    )
    if result is None:
        print("  No result")
        continue

    # Reconstruct test set aligned with model output
    label_col = f"label_s{stop_atr}".replace(".", "_")
    test_idx = result['test_index']
    df_test = df.loc[test_idx].copy()
    scores = np.asarray(result['proba_test'])
    y_raw = result['y_test_raw']

    # Select top-500
    order = np.argsort(scores)[::-1]
    top_idx = order[:500]
    mask = np.zeros(len(scores), dtype=bool)
    mask[top_idx] = True

    df_sel = df_test.iloc[mask].copy()
    y_sel = y_raw.iloc[mask]
    scores_sel = scores[mask]

    # Per-trade P&L
    reward = df_sel['vwap_width_atr'].values * df_sel['atr'].values * SHARES_PER_TRADE
    risk = stop_atr * df_sel['atr'].values * SHARES_PER_TRADE
    gross_pnl = np.where(y_sel.values == 1, reward, -risk)
    net_pnl = gross_pnl - costs_per_trade

    df_sel = df_sel.copy()
    df_sel['net_pnl'] = net_pnl
    df_sel['gross_pnl'] = gross_pnl
    df_sel['win'] = (y_sel.values == 1).astype(int)
    df_sel['reward'] = reward
    df_sel['risk'] = risk
    df_sel['dt'] = pd.to_datetime(df_sel['datetime'])
    df_sel['month'] = df_sel['dt'].dt.to_period('M')
    df_sel['year'] = df_sel['dt'].dt.year

    # ── Overall summary ──
    total_net = df_sel['net_pnl'].sum()
    total_gross = df_sel['gross_pnl'].sum()
    total_costs = costs_per_trade * len(df_sel)
    n_trades = len(df_sel)
    n_wins = df_sel['win'].sum()
    wr = n_wins / n_trades * 100
    avg_win = df_sel.loc[df_sel['win']==1, 'net_pnl'].mean() if n_wins > 0 else 0
    avg_loss = df_sel.loc[df_sel['win']==0, 'net_pnl'].mean() if (n_trades - n_wins) > 0 else 0
    
    print(f"\n  OVERALL:")
    print(f"    Trades: {n_trades}  |  Wins: {n_wins}  |  WR: {wr:.1f}%")
    print(f"    Gross P&L: ${total_gross:,.0f}  |  Costs: ${total_costs:,.0f}  |  Net P&L: ${total_net:,.0f}")
    print(f"    Avg Win: ${avg_win:,.0f}  |  Avg Loss: ${avg_loss:,.0f}  |  Avg Net/Trade: ${total_net/n_trades:,.1f}")
    print(f"    Profit Factor: {abs(df_sel.loc[df_sel['win']==1,'net_pnl'].sum() / min(-1, df_sel.loc[df_sel['win']==0,'net_pnl'].sum())):.2f}")

    # ── Yearly summary ──
    print(f"\n  BY YEAR:")
    print(f"  {'Year':>6s}  {'Trades':>6s}  {'Wins':>5s}  {'WR':>6s}  {'Gross':>10s}  {'Costs':>8s}  {'Net P&L':>10s}  {'Avg/Trade':>10s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")
    for year in sorted(df_sel['year'].unique()):
        sub = df_sel[df_sel['year'] == year]
        n = len(sub)
        w = sub['win'].sum()
        wr_y = w / n * 100 if n > 0 else 0
        g = sub['gross_pnl'].sum()
        c = costs_per_trade * n
        net = sub['net_pnl'].sum()
        avg = net / n if n > 0 else 0
        print(f"  {year:>6d}  {n:>6d}  {w:>5d}  {wr_y:>5.1f}%  ${g:>9,.0f}  ${c:>7,.0f}  ${net:>9,.0f}  ${avg:>9,.1f}")

    # ── Monthly breakdown ──
    print(f"\n  BY MONTH:")
    print(f"  {'Month':>8s}  {'Trades':>6s}  {'Wins':>5s}  {'WR':>6s}  {'Gross':>10s}  {'Net P&L':>10s}  {'Cumul':>10s}  {'Avg/Tr':>8s}  {'Best$':>8s}  {'Worst$':>8s}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    
    cumul = 0.0
    months_pos = 0
    months_neg = 0
    for month in sorted(df_sel['month'].unique()):
        sub = df_sel[df_sel['month'] == month]
        n = len(sub)
        w = sub['win'].sum()
        wr_m = w / n * 100 if n > 0 else 0
        g = sub['gross_pnl'].sum()
        net = sub['net_pnl'].sum()
        cumul += net
        avg = net / n if n > 0 else 0
        best = sub['net_pnl'].max()
        worst = sub['net_pnl'].min()
        marker = "+" if net > 0 else "-"
        if net > 0:
            months_pos += 1
        else:
            months_neg += 1
        print(f"  {str(month):>8s}  {n:>6d}  {w:>5d}  {wr_m:>5.1f}%  ${g:>9,.0f}  ${net:>9,.0f}  ${cumul:>9,.0f}  ${avg:>7,.0f}  ${best:>7,.0f}  ${worst:>7,.0f}  {marker}")

    print(f"\n  Profitable months: {months_pos}/{months_pos+months_neg} ({months_pos/(months_pos+months_neg)*100:.0f}%)")

    # ── Drawdown analysis ──
    daily_pnl = df_sel.groupby(df_sel['dt'].dt.date)['net_pnl'].sum()
    cumul_daily = daily_pnl.cumsum()
    running_max = cumul_daily.cummax()
    drawdown = cumul_daily - running_max
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    print(f"\n  DRAWDOWN:")
    print(f"    Max drawdown: ${max_dd:,.0f} (at {max_dd_date})")
    print(f"    Final cumulative: ${cumul_daily.iloc[-1]:,.0f}")
    if max_dd < 0:
        print(f"    Recovery ratio (final/dd): {cumul_daily.iloc[-1] / abs(max_dd):.2f}x")

    # ── Win/Loss streaks ──
    wins = df_sel['win'].values
    max_win_streak = 0
    max_loss_streak = 0
    cur_win = 0
    cur_loss = 0
    for w in wins:
        if w == 1:
            cur_win += 1
            cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)
    print(f"\n  STREAKS:")
    print(f"    Max consecutive wins: {max_win_streak}")
    print(f"    Max consecutive losses: {max_loss_streak}")

    # ── Long vs Short ──
    print(f"\n  LONG vs SHORT:")
    for side, side_val in [("Long (buy)", 1), ("Short (sell)", 0)]:
        sub = df_sel[df_sel['is_long_setup'] == side_val]
        n = len(sub)
        if n == 0:
            continue
        w = sub['win'].sum()
        wr_s = w / n * 100
        net = sub['net_pnl'].sum()
        print(f"    {side:15s}: {n:>4d} trades, WR={wr_s:.1f}%, Net=${net:>8,.0f}")
