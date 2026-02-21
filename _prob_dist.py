"""Analyze the RF regressor's prediction distribution and calibration.

Key questions:
1. What does the distribution of predicted payoffs look like?
2. Are higher predictions actually more profitable? (calibration)
3. Can we find a threshold where RF predictions actually add value?
4. Can we scale position size by prediction confidence to boost P&L?
5. Where does marginal P&L per trade turn negative?
"""
import sys, warnings, argparse
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from master_pipeline import (
    load_and_validate_data, calculate_core_indicators, get_feature_columns,
    apply_setup_filter, train_rf_model, DATA_FILE, TEST_YEAR,
    SHARES_PER_TRADE, COMMISSION_PER_SHARE, SLIPPAGE_PER_SHARE,
)
from label_generator import LabelConfig, generate_labels

# ── CLI ──
parser = argparse.ArgumentParser(description="RF Prediction Distribution Analysis")
parser.add_argument("--train-years", type=str, default=None, help="e.g. 2016-2020")
parser.add_argument("--test-years", type=str, default=None, help="e.g. 2021-2026")
parser.add_argument("--stops", type=str, default="0.5,0.75,1.0",
                    help="Comma-separated stop ATR values")
args = parser.parse_args()

# Parse year ranges
train_start_year = None
train_end_year = None
test_start_year = TEST_YEAR
test_end_year = None

if args.train_years:
    parts = args.train_years.split('-')
    train_start_year = int(parts[0])
    train_end_year = int(parts[1])
    if not args.test_years:
        test_start_year = train_end_year  # test starts right after train

if args.test_years:
    parts = args.test_years.split('-')
    test_start_year = int(parts[0])
    test_end_year = int(parts[1])

stops_to_analyze = [float(x) for x in args.stops.split(',')]

train_label = f"{train_start_year}-{train_end_year - 1}" if train_start_year and train_end_year else f"<{test_start_year}"
test_label = f"{test_start_year}-{test_end_year - 1}" if test_end_year else f">={test_start_year}"

# Tee output
_f = open("_prob_dist_out.txt", "w", encoding="utf-8")
class Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for x in self.s:
            try:
                x.write(d); x.flush()
            except UnicodeEncodeError:
                x.write(d.encode('ascii', 'replace').decode('ascii')); x.flush()    def flush(self):
        for x in self.s:
            try:
                x.flush()
            except ValueError:
                pass
sys.stdout = Tee(sys.__stdout__, _f)

print("=" * 90)
print(f"  RF PREDICTION DISTRIBUTION ANALYSIS")
print(f"  Train: {train_label}  |  Test: {test_label}")
print(f"  Stops: {stops_to_analyze}")
print("=" * 90)

print("\nLoading data...")
df = load_and_validate_data(DATA_FILE)
df = calculate_core_indicators(df, verbose=False)
features = get_feature_columns(df)

config = LabelConfig(stop_atrs=stops_to_analyze)
df = generate_labels(df, config)

costs = 2.0 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * SHARES_PER_TRADE

for stop_atr in stops_to_analyze:
    print(f"\n{'='*90}")
    print(f"  STOP = {stop_atr} ATR  |  RF Regressor Prediction Analysis")
    print(f"{'='*90}")

    result = train_rf_model(
        df, stop_atr, features, test_year=test_start_year,
        train_start_year=train_start_year,
        train_end_year=train_end_year if train_end_year else None,
        test_end_year=test_end_year,
        model_kind='regressor', regression_target='net_r',
        setup_filter=True, min_dist_atr=0.5, min_minutes_session=15,
        max_minutes_session=360, min_rr_setup=1.0,
    )
    if result is None:
        print("  No result")
        continue

    test_idx = result['test_index']
    df_test = df.loc[test_idx].copy()
    scores = np.asarray(result['proba_test'])  # predicted net_r
    y_raw = result['y_test_raw'].values
    atr_vals = df_test['atr'].values
    vwap_dist = df_test['vwap_width_atr'].values

    # Also get train predictions for comparison
    train_scores = np.asarray(result['pred_train'])
    y_train_raw = result['y_train_raw'].values
    train_idx = result['train_index']
    df_train = df.loc[train_idx]
    atr_train = df_train['atr'].values
    vwap_dist_train = df_train['vwap_width_atr'].values

    # Per-trade actual P&L (test)
    reward = vwap_dist * atr_vals * SHARES_PER_TRADE
    risk = stop_atr * atr_vals * SHARES_PER_TRADE
    gross = np.where(y_raw == 1, reward, -risk)
    net = gross - costs

    # Per-trade actual P&L (train)
    reward_tr = vwap_dist_train * atr_train * SHARES_PER_TRADE
    risk_tr = stop_atr * atr_train * SHARES_PER_TRADE
    gross_tr = np.where(y_train_raw == 1, reward_tr, -risk_tr)
    net_tr = gross_tr - costs

    # ── 1. Prediction distribution: TRAIN vs TEST ──
    print(f"\n  PREDICTION DISTRIBUTION (predicted net_r):")
    print(f"  {'':>12s}  {'TRAIN':>10s}  {'TEST':>10s}")
    print(f"  {'count':>12s}  {len(train_scores):>10,d}  {len(scores):>10,d}")
    print(f"  {'mean':>12s}  {train_scores.mean():>+10.4f}  {scores.mean():>+10.4f}")
    print(f"  {'std':>12s}  {train_scores.std():>10.4f}  {scores.std():>10.4f}")
    print(f"  {'min':>12s}  {train_scores.min():>+10.4f}  {scores.min():>+10.4f}")
    print(f"  {'25%':>12s}  {np.percentile(train_scores, 25):>+10.4f}  {np.percentile(scores, 25):>+10.4f}")
    print(f"  {'50%':>12s}  {np.percentile(train_scores, 50):>+10.4f}  {np.percentile(scores, 50):>+10.4f}")
    print(f"  {'75%':>12s}  {np.percentile(train_scores, 75):>+10.4f}  {np.percentile(scores, 75):>+10.4f}")
    print(f"  {'max':>12s}  {train_scores.max():>+10.4f}  {scores.max():>+10.4f}")
    print(f"  {'% positive':>12s}  {(train_scores > 0).mean()*100:>9.1f}%  {(scores > 0).mean()*100:>9.1f}%")

    # Range overlap -- does RF compress test predictions?
    tr_range = train_scores.max() - train_scores.min()
    te_range = scores.max() - scores.min()
    print(f"\n  Range: train={tr_range:.4f}, test={te_range:.4f}, ratio={te_range/tr_range:.2f}")
    print(f"  => RF spreads predictions {'LESS' if te_range < tr_range * 0.8 else 'SIMILARLY'} on test (compression = overfit signal)")

    # ── 2. Decile analysis: calibration ──
    print(f"\n  DECILE ANALYSIS - TEST (sorted by predicted payoff):")
    print(f"  {'Decile':>7s}  {'PredMean':>9s}  {'PredRange':>16s}  {'ActWR':>6s}  {'NetPnL':>10s}  {'Avg$/Tr':>9s}  {'N':>5s}")
    print(f"  {'-'*7}  {'-'*9}  {'-'*16}  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*5}")

    order = np.argsort(scores)
    n = len(scores)
    decile_size = n // 10

    decile_wrs = []
    for d in range(10):
        start = d * decile_size
        end = (d + 1) * decile_size if d < 9 else n
        idx = order[start:end]
        pred_d = scores[idx]
        wr_d = y_raw[idx].mean()
        net_d = net[idx]
        total_net = net_d.sum()
        avg_net = net_d.mean()
        decile_wrs.append(wr_d)
        rng = f"[{pred_d.min():+.4f},{pred_d.max():+.4f}]"
        marker = " ***" if d == 9 else ""
        print(f"  D{d+1:>5d}  {pred_d.mean():>+9.4f}  {rng:>16s}  {wr_d*100:>5.1f}%  ${total_net:>9,.0f}  ${avg_net:>8,.1f}  {len(idx):>5d}{marker}")

    # Check monotonicity
    mono_violations = sum(1 for i in range(1, 10) if decile_wrs[i] < decile_wrs[i-1])
    print(f"\n  Monotonicity: {10 - mono_violations}/9 consecutive pairs increase WR")
    print(f"  D1 WR: {decile_wrs[0]*100:.1f}% -> D10 WR: {decile_wrs[9]*100:.1f}% "
          f"(spread: {(decile_wrs[9]-decile_wrs[0])*100:+.1f} pts)")
    if decile_wrs[9] > decile_wrs[0] + 0.03:
        print(f"  [OK] TOP decile clearly beats BOTTOM -- RF has SOME signal")
    else:
        print(f"  [!!] TOP decile barely beats BOTTOM -- RF has WEAK signal")

    # ── 2b. Same for TRAIN (to show overfit gap) ──
    print(f"\n  DECILE ANALYSIS - TRAIN (for overfit comparison):")
    print(f"  {'Decile':>7s}  {'PredMean':>9s}  {'ActWR':>6s}  {'NetPnL':>10s}  {'Avg$/Tr':>9s}  {'N':>5s}")
    print(f"  {'-'*7}  {'-'*9}  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*5}")

    order_tr = np.argsort(train_scores)
    n_tr = len(train_scores)
    decile_size_tr = n_tr // 10
    train_decile_wrs = []
    for d in range(10):
        start = d * decile_size_tr
        end = (d + 1) * decile_size_tr if d < 9 else n_tr
        idx = order_tr[start:end]
        pred_d = train_scores[idx]
        wr_d = y_train_raw[idx].mean()
        net_d = net_tr[idx]
        total_net = net_d.sum()
        avg_net = net_d.mean()
        train_decile_wrs.append(wr_d)
        print(f"  D{d+1:>5d}  {pred_d.mean():>+9.4f}  {wr_d*100:>5.1f}%  ${total_net:>9,.0f}  ${avg_net:>8,.1f}  {len(idx):>5d}")

    print(f"\n  OVERFIT GAP: Train D10 WR={train_decile_wrs[9]*100:.1f}% vs Test D10 WR={decile_wrs[9]*100:.1f}%"
          f" (gap: {(train_decile_wrs[9]-decile_wrs[9])*100:.1f} pts)")

    # ── 3. Top-N sweep with cumulative P&L ──
    print(f"\n  TOP-N CUMULATIVE P&L - TEST (selected by highest predicted payoff):")
    print(f"  {'TopN':>6s}  {'WR':>6s}  {'NetPnL':>10s}  {'Avg/Tr':>9s}  {'MargAvg':>9s}  {'PredCutoff':>10s}  {'BaseWR':>7s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*7}")

    desc_order = np.argsort(scores)[::-1]
    base_wr = y_raw.mean()
    prev_net = 0.0
    prev_n = 0
    best_pnl = -1e9
    best_n = 0
    peak_n = 0
    for top_n in [25, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, len(scores)]:
        if top_n > len(scores):
            top_n = len(scores)
        idx = desc_order[:top_n]
        wr = y_raw[idx].mean()
        total_net = net[idx].sum()
        avg = total_net / top_n
        marg_net = total_net - prev_net
        marg_n = top_n - prev_n
        marg_avg = marg_net / marg_n if marg_n > 0 else 0
        cutoff = scores[desc_order[top_n - 1]]
        if total_net > best_pnl:
            best_pnl = total_net
            best_n = top_n
            peak_n = top_n
        marker = ""
        if top_n == best_n:
            marker = " <-- peak"
        elif marg_avg < 0 and total_net < best_pnl:
            marker = " v"
        print(f"  {top_n:>6d}  {wr*100:>5.1f}%  ${total_net:>9,.0f}  ${avg:>8,.1f}  ${marg_avg:>8,.1f}  {cutoff:>+10.4f}  {base_wr*100:>5.1f}%{marker}")
        prev_net = total_net
        prev_n = top_n
        if top_n == len(scores):
            break

    print(f"\n  PEAK P&L at top-{peak_n}: ${best_pnl:,.0f}")
    print(f"  Base WR (all test): {base_wr*100:.1f}%")

    # ── 4. Threshold sweep: find where prediction > threshold is profitable ──
    print(f"\n  THRESHOLD ANALYSIS - TEST (take all trades with pred >= threshold):")
    print(f"  {'Threshold':>10s}  {'N':>6s}  {'WR':>6s}  {'NetPnL':>10s}  {'Avg$/Tr':>9s}  {'Lift':>6s}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*6}")

    percentiles = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    thresholds_from_pct = [np.percentile(scores, p) for p in percentiles]
    # Also add some fixed thresholds
    fixed_thresholds = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
    all_thresholds = sorted(set(thresholds_from_pct + fixed_thresholds), reverse=True)

    best_thresh_pnl = -1e9
    best_thresh = None
    for thresh in all_thresholds:
        mask = scores >= thresh
        n_sel = mask.sum()
        if n_sel < 10:
            continue
        wr = y_raw[mask].mean()
        total_net = net[mask].sum()
        avg = total_net / n_sel
        lift = (wr - base_wr) * 100
        if total_net > best_thresh_pnl:
            best_thresh_pnl = total_net
            best_thresh = thresh
        marker = " <-- best" if thresh == best_thresh and total_net == best_thresh_pnl else ""
        print(f"  {thresh:>+10.4f}  {n_sel:>6d}  {wr*100:>5.1f}%  ${total_net:>9,.0f}  ${avg:>8,.1f}  {lift:>+5.1f}%{marker}")

    print(f"\n  Best threshold: {best_thresh:+.4f} => ${best_thresh_pnl:,.0f}")

    # ── 5. Correlation between prediction and outcome ──
    print(f"\n  PREDICTION vs OUTCOME CORRELATION:")
    from scipy.stats import spearmanr, pearsonr
    r_pearson, p_pearson = pearsonr(scores, net)
    r_spearman, p_spearman = spearmanr(scores, net)
    print(f"    Pearson:  r={r_pearson:.4f}  (p={p_pearson:.4g})")
    print(f"    Spearman: r={r_spearman:.4f}  (p={p_spearman:.4g})")

    # Train correlation for comparison
    r_tr_p, _ = pearsonr(train_scores, net_tr)
    r_tr_s, _ = spearmanr(train_scores, net_tr)
    print(f"    Train Pearson:  r={r_tr_p:.4f}")
    print(f"    Train Spearman: r={r_tr_s:.4f}")
    print(f"    Gap: Pearson {r_tr_p - r_pearson:.4f}, Spearman {r_tr_s - r_spearman:.4f}")

    # ── 6. Scaled position sizing simulation ──
    print(f"\n  SCALED POSITION SIZING (shares proportional to prediction):")
    print(f"  Only trades with pred > 0. Base = {SHARES_PER_TRADE} shares.")

    pos_mask = scores > 0
    n_pos = pos_mask.sum()
    print(f"    Trades with pred > 0: {n_pos:,} ({n_pos/len(scores)*100:.1f}%)")

    if n_pos > 0:
        flat_pnl = net[pos_mask].sum()
        print(f"    Flat sizing (pred>0): ${flat_pnl:,.0f}")
        print(f"  {'Threshold':>10s}  {'Flat$':>10s}  {'Scaled$':>10s}  {'Delta$':>9s}  {'AvgShares':>10s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*10}")

        for threshold in [0.05, 0.1, 0.2, 0.3, 0.5]:
            scale = np.clip(scores[pos_mask] / threshold, 0.5, 3.0)
            shares_scaled = SHARES_PER_TRADE * scale

            reward_s = vwap_dist[pos_mask] * atr_vals[pos_mask] * shares_scaled
            risk_s = stop_atr * atr_vals[pos_mask] * shares_scaled
            costs_s = 2.0 * (COMMISSION_PER_SHARE + SLIPPAGE_PER_SHARE) * shares_scaled
            gross_s = np.where(y_raw[pos_mask] == 1, reward_s, -risk_s)
            net_s = gross_s - costs_s
            net_scaled = net_s.sum()
            avg_shares = shares_scaled.mean()
            print(f"  {threshold:>+10.2f}  ${flat_pnl:>9,.0f}  ${net_scaled:>9,.0f}  ${net_scaled-flat_pnl:>+8,.0f}  {avg_shares:>10.0f}")

    # ── 7. Kelly-style quintile analysis ──
    print(f"\n  QUINTILE KELLY ANALYSIS - TEST:")
    print(f"  {'Quintile':>8s}  {'N':>5s}  {'WR':>6s}  {'AvgWin$':>9s}  {'AvgLoss$':>9s}  {'Edge$/Tr':>9s}  {'Kelly%':>7s}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*7}")

    quintile_size = n // 5
    for q in range(5):
        start = q * quintile_size
        end = (q + 1) * quintile_size if q < 4 else n
        idx = order[start:end]
        y_q = y_raw[idx]
        net_q = net[idx]
        wr_q = y_q.mean()
        wins = net_q[y_q == 1]
        losses = net_q[y_q == 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
        edge = wr_q * avg_win - (1 - wr_q) * avg_loss
        kelly = (wr_q - (1 - wr_q) * (avg_loss / avg_win)) if avg_win > 0 else -1
        kelly_pct = max(0, kelly) * 100
        print(f"  Q{q+1:>6d}  {len(idx):>5d}  {wr_q*100:>5.1f}%  ${avg_win:>8,.0f}  ${avg_loss:>8,.0f}  ${edge:>8,.1f}  {kelly_pct:>6.1f}%")

    # ── 8. Year-by-year breakdown of top decile ──
    print(f"\n  TOP DECILE BY YEAR - TEST:")
    top_decile_idx = order[n - decile_size:]
    top_y = y_raw[top_decile_idx]
    top_net = net[top_decile_idx]    if 'year' in df_test.columns:
        all_years = df_test.iloc[top_decile_idx]['year'].values
    elif 'datetime' in df_test.columns:
        all_years = pd.to_datetime(df_test.iloc[top_decile_idx]['datetime']).dt.year.values
    else:
        all_years = pd.DatetimeIndex(df_test.index[top_decile_idx]).year.values

    print(f"  {'Year':>6s}  {'N':>5s}  {'WR':>6s}  {'NetPnL':>10s}  {'Avg$/Tr':>9s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*9}")
    for yr in sorted(set(all_years)):
        yr_mask = all_years == yr
        n_yr = yr_mask.sum()
        wr_yr = top_y[yr_mask].mean()
        pnl_yr = top_net[yr_mask].sum()
        avg_yr = pnl_yr / n_yr if n_yr > 0 else 0
        print(f"  {yr:>6d}  {n_yr:>5d}  {wr_yr*100:>5.1f}%  ${pnl_yr:>9,.0f}  ${avg_yr:>8,.1f}")

print(f"\n{'='*90}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*90}")
print("\nDone.")
_f.close()
