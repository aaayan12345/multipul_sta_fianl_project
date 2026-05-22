"""
Step 3: Feature Extraction — weak-supervision trading style features
Features:
  F1: Industry preference distribution
  F2: Annualized turnover rate
  F3: Average holding period (days, FIFO matched)
  F4: Holding concentration (avg HHI over time)
  F5: Buy-sell symmetry (0-1, 0.5=balanced)
  F6: Volatility preference (weighted avg stock-level price CoV)
  F7: Realized return preference (FIFO matched sell return)
  F8: Max drawdown preference (approximate position-cost equity curve)
  F9: Market state exposure (trade-amount weighted bull/flat/bear score)
  F10: Trade interval (average days between adjacent trades)
"""
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

NUMERIC_FEATURES = [
    'turnover',
    'holding_period_days',
    'concentration_hhi',
    'buy_sell_symmetry',
    'volatility_pref',
    'realized_return_pref',
    'max_drawdown_pref',
    'market_state_exposure',
    'trade_interval_days',
]

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print("Step 3: Feature Extraction")
print("=" * 60)

strategies = pd.read_csv('clean_strategies.csv', dtype={'stock_code': str})
accounts = pd.read_csv('clean_accounts.csv', dtype={'stock_code': str})


def load_industry_mapping():
    """Prefer manually reviewed mapping when present.

    Blank stock names are valid as long as the stock code exists. If
    review_industry is filled, it overrides the original industry column.
    """
    review_path = Path('stock_industry_mapping_review.csv')
    mapping_path = review_path if review_path.exists() else Path('stock_industry_mapping.csv')
    df = pd.read_csv(mapping_path, dtype={'stock_code': str}, encoding='utf-8-sig')
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
    if 'review_industry' in df.columns:
        reviewed = df['review_industry'].fillna('').astype(str).str.strip()
        df['industry_final'] = np.where(reviewed.ne(''), reviewed, df['industry'])
    else:
        df['industry_final'] = df['industry']
    print(f"Using industry mapping: {mapping_path} ({len(df)} rows)")
    return df


industry_map = load_industry_mapping()
strategies['stock_code'] = strategies['stock_code'].astype(str).str.zfill(6)
accounts['stock_code'] = accounts['stock_code'].astype(str).str.zfill(6)

# Parse dates
strategies['date'] = pd.to_datetime(strategies['datetime'])
accounts['date'] = pd.to_datetime(accounts['datetime'])

# Build industry lookup
code2ind = dict(zip(industry_map['stock_code'], industry_map['industry_final']))
all_industries = sorted(set(code2ind.values()))
print(f"\nTotal industries: {len(all_industries)}")
print(f"Strategies: {strategies['strategy_name'].nunique()}")
print(f"Accounts: {accounts['account_id'].nunique()}")

# Join industry
strategies['industry'] = strategies['stock_code'].map(code2ind).fillna('综合')
accounts['industry'] = accounts['stock_code'].map(code2ind).fillna('综合')

# ============================================================
# 2. Feature Functions
# ============================================================

def compute_industry_pref(df):
    """F1: Industry preference — probability vector over 31 industries w/ Laplace smoothing"""
    buys = df[df['action'] == 'BUY']
    if len(buys) == 0:
        buys = df  # sell-only fallback
    amounts = buys.groupby('industry')['amount'].sum()
    pref = {}
    for ind in all_industries:
        pref[ind] = amounts.get(ind, 0.0) + 1.0  # add-1 smoothing
    total = sum(pref.values())
    return {k: v / total for k, v in pref.items()}


def compute_turnover(df):
    """F2: Annualized turnover = total_buy / avg_position / years"""
    total_buy = df[df['action'] == 'BUY']['amount'].sum()
    if total_buy == 0:
        return 0.0

    days_span = (df['date'].max() - df['date'].min()).days
    years = max(days_span / 365.25, 1 / 252)  # min ~1 trading day

    # Reconstruct daily position values
    positions = defaultdict(float)
    position_snapshots = []
    sorted_df = df.sort_values('date')
    for _, row in sorted_df.iterrows():
        code = row['stock_code']
        if row['action'] == 'BUY':
            positions[code] += row['amount']
        else:
            positions[code] = max(0.0, positions[code] - row['amount'])
        position_snapshots.append(sum(positions.values()))

    avg_position = np.mean(position_snapshots) if position_snapshots else total_buy
    if avg_position < 1.0:
        return 0.0  # effectively no position

    return float(total_buy / avg_position / years)


def compute_holding_period(df):
    """F3: Volume-weighted avg holding days via FIFO matching"""
    all_days = []  # list of (days, volume)

    for code, group in df.groupby('stock_code'):
        group = group.sort_values('date')
        queue = deque()  # (buy_date, remaining_volume)

        for _, row in group.iterrows():
            if row['action'] == 'BUY':
                queue.append((row['date'], float(row['volume'])))
            else:
                remaining = float(row['volume'])
                sell_date = row['date']
                while remaining > 0 and queue:
                    buy_date, buy_vol = queue[0]
                    matched = min(buy_vol, remaining)
                    days = (sell_date - buy_date).days
                    if days >= 0:
                        all_days.append((days, matched))
                    if buy_vol <= remaining:
                        queue.popleft()
                    else:
                        queue[0] = (buy_date, buy_vol - matched)
                    remaining -= matched

    if not all_days:
        return 0.0
    total_vol = sum(v for _, v in all_days)
    return float(sum(d * v for d, v in all_days) / total_vol)


def compute_concentration(df):
    """F4: Average Herfindahl-Hirschman Index (HHI) over time"""
    positions = defaultdict(float)
    hhi_snapshots = []

    for _, row in df.sort_values('date').iterrows():
        code = row['stock_code']
        if row['action'] == 'BUY':
            positions[code] += float(row['amount'])
        else:
            positions[code] = max(0.0, positions[code] - float(row['amount']))

        total = sum(positions.values())
        if total > 0:
            hhi = sum((v / total) ** 2 for v in positions.values())
            hhi_snapshots.append(hhi)

    return float(np.mean(hhi_snapshots)) if hhi_snapshots else 0.0


def compute_buy_sell_symmetry(df):
    """F5: Buy-sell symmetry — 0.5 = balanced, >0.5 = net buyer"""
    total_buy = df[df['action'] == 'BUY']['amount'].sum()
    total_sell = df[df['action'] == 'SELL']['amount'].sum()
    total = total_buy + total_sell
    if total < 1.0:
        return 0.5
    return float(total_buy / total)


def compute_volatility_pref(df):
    """F6: Volatility preference — weighted avg CoV of stock prices (robust)"""
    vol_records = []

    for code, group in df.groupby('stock_code'):
        if len(group) < 2:
            continue
        prices = group.sort_values('date')['price'].values
        mean_p = np.mean(prices)
        if mean_p < 0.01:
            continue
        cov = np.std(prices) / mean_p  # coefficient of variation (0~1 typically)
        cov = np.clip(cov, 0, 2.0)     # cap extreme outliers
        weight = float(group['amount'].sum())
        if weight > 0:
            vol_records.append((cov, weight))

    if not vol_records:
        return 0.0
    total_w = sum(w for _, w in vol_records)
    return float(sum(v * w for v, w in vol_records) / total_w)


def compute_realized_return_pref(df):
    """F7: Amount-weighted realized return via FIFO matching."""
    matched_returns = []

    for code, group in df.groupby('stock_code'):
        group = group.sort_values('date')
        queue = deque()  # (buy_price, remaining_volume)

        for _, row in group.iterrows():
            price = float(row['price'])
            volume = float(row['volume'])
            if price <= 0 or volume <= 0:
                continue
            if row['action'] == 'BUY':
                queue.append((price, volume))
                continue

            remaining = volume
            while remaining > 0 and queue:
                buy_price, buy_vol = queue[0]
                matched = min(buy_vol, remaining)
                if buy_price > 0:
                    ret = (price - buy_price) / buy_price
                    matched_returns.append((np.clip(ret, -1.0, 3.0), matched * price))
                if buy_vol <= remaining:
                    queue.popleft()
                else:
                    queue[0] = (buy_price, buy_vol - matched)
                remaining -= matched

    if not matched_returns:
        return 0.0
    total_w = sum(w for _, w in matched_returns)
    return float(sum(r * w for r, w in matched_returns) / max(total_w, 1e-9))


def compute_max_drawdown_pref(df):
    """F8: Approximate max drawdown on a position-cost equity curve."""
    positions = defaultdict(lambda: {'volume': 0.0, 'price': 0.0})
    equity_curve = []

    for _, row in df.sort_values('date').iterrows():
        code = row['stock_code']
        price = float(row['price'])
        volume = float(row['volume'])
        if price <= 0 or volume <= 0:
            continue

        pos = positions[code]
        if row['action'] == 'BUY':
            new_vol = pos['volume'] + volume
            pos['price'] = (pos['price'] * pos['volume'] + price * volume) / max(new_vol, 1e-9)
            pos['volume'] = new_vol
        else:
            pos['volume'] = max(0.0, pos['volume'] - volume)
            pos['price'] = price

        equity = sum(v['volume'] * v['price'] for v in positions.values())
        if equity > 0:
            equity_curve.append(equity)

    if len(equity_curve) < 2:
        return 0.0
    curve = np.array(equity_curve, dtype=np.float64)
    peaks = np.maximum.accumulate(curve)
    drawdowns = 1.0 - curve / np.maximum(peaks, 1e-9)
    return float(np.clip(np.max(drawdowns), 0.0, 1.0))


def compute_market_state_exposure(df):
    """F9: Trade-weighted exposure to recent market state."""
    daily_price = df.pivot_table(index='date', columns='stock_code', values='price', aggfunc='last')
    if len(daily_price) < 2:
        return 0.0

    daily_ret = daily_price.sort_index().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    market_ret = daily_ret.mean(axis=1).rolling(5, min_periods=1).mean()
    market_state = pd.cut(
        market_ret,
        bins=[-np.inf, -0.005, 0.005, np.inf],
        labels=[-1.0, 0.0, 1.0],
    ).astype(float)

    weighted = []
    for _, row in df.iterrows():
        state = market_state.reindex([row['date']], method='ffill').iloc[0]
        if pd.notna(state):
            weighted.append((float(state), float(row['amount'])))

    if not weighted:
        return 0.0
    total_w = sum(w for _, w in weighted)
    return float(sum(s * w for s, w in weighted) / max(total_w, 1e-9))


def compute_trade_interval(df):
    """F10: Average days between adjacent trades."""
    dates = df.sort_values('date')['date'].drop_duplicates()
    if len(dates) < 2:
        return 0.0
    intervals = dates.diff().dt.days.dropna()
    return float(intervals.mean()) if len(intervals) else 0.0


# ============================================================
# 3. Compute All Features
# ============================================================

def compute_all_features(df):
    return {
        'industry_pref': compute_industry_pref(df),
        'turnover': compute_turnover(df),
        'holding_period_days': compute_holding_period(df),
        'concentration_hhi': compute_concentration(df),
        'buy_sell_symmetry': compute_buy_sell_symmetry(df),
        'volatility_pref': compute_volatility_pref(df),
        'realized_return_pref': compute_realized_return_pref(df),
        'max_drawdown_pref': compute_max_drawdown_pref(df),
        'market_state_exposure': compute_market_state_exposure(df),
        'trade_interval_days': compute_trade_interval(df),
    }


# --- Strategies ---
print("\n--- Strategy Features ---")
strategy_features = {}
for sname in sorted(strategies['strategy_name'].unique()):
    group = strategies[strategies['strategy_name'] == sname]
    feats = compute_all_features(group)
    strategy_features[sname] = feats
    print(f"  [{sname}]")
    print(f"    turnover={feats['turnover']:.3f}, holding={feats['holding_period_days']:.1f}d, "
          f"HHI={feats['concentration_hhi']:.4f}, buy_ratio={feats['buy_sell_symmetry']:.3f}, "
          f"vol={feats['volatility_pref']:.5f}, ret={feats['realized_return_pref']:.4f}, "
          f"mdd={feats['max_drawdown_pref']:.4f}, mkt={feats['market_state_exposure']:.3f}")

# --- Accounts ---
print("\n--- Account Features ---")
account_features = {}
for aid in sorted(accounts['account_id'].unique()):
    group = accounts[accounts['account_id'] == aid]
    feats = compute_all_features(group)
    account_features[aid] = feats
    print(f"  [Account {aid}]")
    print(f"    turnover={feats['turnover']:.3f}, holding={feats['holding_period_days']:.1f}d, "
          f"HHI={feats['concentration_hhi']:.4f}, buy_ratio={feats['buy_sell_symmetry']:.3f}, "
          f"vol={feats['volatility_pref']:.5f}, ret={feats['realized_return_pref']:.4f}, "
          f"mdd={feats['max_drawdown_pref']:.4f}, mkt={feats['market_state_exposure']:.3f}")

# ============================================================
# 4. Save Results
# ============================================================

def flatten_features(feats_dict, prefix='entity'):
    """Flatten industry_pref into separate columns"""
    rows = []
    for name, feats in feats_dict.items():
        row = {'name': name}
        for k in NUMERIC_FEATURES:
            row[k] = feats[k]
        for ind in all_industries:
            row[f'ind_{ind}'] = feats['industry_pref'].get(ind, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


strategy_df = flatten_features(strategy_features)
account_df = flatten_features(account_features)

strategy_df.to_csv('strategy_features.csv', index=False, encoding='utf-8-sig')
account_df.to_csv('account_features.csv', index=False, encoding='utf-8-sig')

# Also save as JSON for easy reload (preserves structure)
with open('strategy_features.json', 'w', encoding='utf-8') as f:
    json.dump(strategy_features, f, ensure_ascii=False, indent=2, default=str)
with open('account_features.json', 'w', encoding='utf-8') as f:
    json.dump(account_features, f, ensure_ascii=False, indent=2, default=str)

print(f"\nSaved: strategy_features.csv + .json ({len(strategy_df)} strategies)")
print(f"Saved: account_features.csv + .json ({len(account_df)} accounts)")

# ============================================================
# 5. Summary statistics
# ============================================================
print("\n" + "=" * 60)
print("Feature Distribution Summary")
print("=" * 60)

for label, feats_dict in [("Strategy", strategy_features), ("Account", account_features)]:
    print(f"\n--- {label} ---")
    all_feats = list(feats_dict.values())
    for key in NUMERIC_FEATURES:
        vals = [f[key] for f in all_feats]
        print(f"  {key:25s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

print("\nDone. Step 3 complete.")
