"""
Step 4: Token 构建 + Word2Vec 预训练 (PyTorch 实现)
4.1 每条交易 → 多维风格 token
    {行业}_{BUY/SELL}_A{金额}_H{持仓}_T{换手}_R{收益}_D{回撤}_M{市场状态}
4.2 构建词表 (token → id)
4.3 Skip-gram + 负采样训练 64 维词向量
"""
import pandas as pd
import numpy as np
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 4.1 加载数据
# ============================================================
print("=" * 60)
print("Step 4: Token 构建 + Word2Vec 预训练 (PyTorch)")
print("=" * 60)

strategies = pd.read_csv('clean_strategies.csv', dtype={'stock_code': str})
accounts = pd.read_csv('clean_accounts.csv', dtype={'stock_code': str})
strategies['stock_code'] = strategies['stock_code'].astype(str).str.zfill(6)
accounts['stock_code'] = accounts['stock_code'].astype(str).str.zfill(6)


def load_industry_mapping():
    """Prefer manually reviewed mapping when present.

    Empty stock names are kept; stock_code is the join key. A filled
    review_industry overrides industry.
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
code2ind = dict(zip(industry_map['stock_code'], industry_map['industry_final']))
strategies['industry'] = strategies['stock_code'].map(code2ind).fillna('综合')
accounts['industry'] = accounts['stock_code'].map(code2ind).fillna('综合')
strategies['date'] = pd.to_datetime(strategies['datetime'])
accounts['date'] = pd.to_datetime(accounts['datetime'])

# ============================================================
# 4.2 Token 构建
# ============================================================

def amount_bucket(value, q33, q66):
    if value <= q33:
        return 'S'
    if value <= q66:
        return 'M'
    return 'L'


def holding_bucket(days):
    if days is None or pd.isna(days):
        return 'UNK'
    if days <= 3:
        return 'D0_3'
    if days <= 10:
        return 'D4_10'
    if days <= 30:
        return 'D11_30'
    if days <= 90:
        return 'D31_90'
    return 'D90P'


def turnover_bucket(turnover):
    if turnover <= 5:
        return 'LOW'
    if turnover <= 30:
        return 'MID'
    if turnover <= 80:
        return 'HIGH'
    return 'ULTRA'


def return_bucket(ret):
    if ret is None or pd.isna(ret):
        return 'NA'
    if ret <= -0.05:
        return 'LOSS_L'
    if ret < 0:
        return 'LOSS_S'
    if ret < 0.05:
        return 'GAIN_S'
    return 'GAIN_L'


def drawdown_bucket(dd):
    if dd <= 0.03:
        return 'LOW'
    if dd <= 0.10:
        return 'MID'
    if dd <= 0.20:
        return 'HIGH'
    return 'EXTREME'


def market_bucket(state):
    if state <= -0.005:
        return 'BEAR'
    if state >= 0.005:
        return 'BULL'
    return 'FLAT'


def compute_entity_turnover(df):
    total_buy = df[df['action'] == 'BUY']['amount'].sum()
    if total_buy <= 0:
        return 0.0
    days_span = (df['date'].max() - df['date'].min()).days
    years = max(days_span / 365.25, 1 / 252)
    positions = defaultdict(float)
    snapshots = []
    for _, row in df.sort_values('date').iterrows():
        if row['action'] == 'BUY':
            positions[row['stock_code']] += float(row['amount'])
        else:
            positions[row['stock_code']] = max(0.0, positions[row['stock_code']] - float(row['amount']))
        snapshots.append(sum(positions.values()))
    avg_position = np.mean(snapshots) if snapshots else total_buy
    return float(total_buy / max(avg_position, 1.0) / years)


def add_trade_style_columns(df):
    """Add per-trade style labels used by the multi-dimensional token."""
    df = df.sort_values('date').copy()
    df['matched_holding_days'] = np.nan
    df['realized_return'] = np.nan
    df['running_drawdown'] = 0.0

    # FIFO holding period and realized return on sells.
    for code, idxs in df.groupby('stock_code').groups.items():
        queue = deque()  # (date, price, remaining_volume)
        for idx in df.loc[idxs].sort_values('date').index:
            row = df.loc[idx]
            price = float(row['price'])
            volume = float(row['volume'])
            if price <= 0 or volume <= 0:
                continue
            if row['action'] == 'BUY':
                queue.append((row['date'], price, volume))
                continue

            remaining = volume
            hold_parts = []
            ret_parts = []
            while remaining > 0 and queue:
                buy_date, buy_price, buy_vol = queue[0]
                matched = min(buy_vol, remaining)
                hold_parts.append(((row['date'] - buy_date).days, matched))
                ret_parts.append(((price - buy_price) / buy_price, matched * price))
                if buy_vol <= remaining:
                    queue.popleft()
                else:
                    queue[0] = (buy_date, buy_price, buy_vol - matched)
                remaining -= matched
            if hold_parts:
                total_v = sum(v for _, v in hold_parts)
                df.loc[idx, 'matched_holding_days'] = sum(d * v for d, v in hold_parts) / max(total_v, 1e-9)
            if ret_parts:
                total_w = sum(w for _, w in ret_parts)
                df.loc[idx, 'realized_return'] = sum(r * w for r, w in ret_parts) / max(total_w, 1e-9)

    # Approximate running drawdown using trade-price position values.
    positions = defaultdict(lambda: {'volume': 0.0, 'price': 0.0})
    peak_equity = 0.0
    for idx, row in df.iterrows():
        code = row['stock_code']
        price = float(row['price'])
        volume = float(row['volume'])
        if price > 0 and volume > 0:
            pos = positions[code]
            if row['action'] == 'BUY':
                new_vol = pos['volume'] + volume
                pos['price'] = (pos['price'] * pos['volume'] + price * volume) / max(new_vol, 1e-9)
                pos['volume'] = new_vol
            else:
                pos['volume'] = max(0.0, pos['volume'] - volume)
                pos['price'] = price
        equity = sum(v['volume'] * v['price'] for v in positions.values())
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            df.loc[idx, 'running_drawdown'] = np.clip(1.0 - equity / peak_equity, 0.0, 1.0)

    # Entity-local market proxy: equal-weighted 5-day average price change.
    daily_price = df.pivot_table(index='date', columns='stock_code', values='price', aggfunc='last')
    if len(daily_price) >= 2:
        daily_ret = daily_price.sort_index().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        market_ret = daily_ret.mean(axis=1).rolling(5, min_periods=1).mean()
        df['market_state_value'] = [market_ret.reindex([d], method='ffill').iloc[0] for d in df['date']]
        df['market_state_value'] = df['market_state_value'].fillna(0.0)
    else:
        df['market_state_value'] = 0.0

    return df


def build_tokens_for_entity(df):
    """Build multi-dimensional style tokens for one strategy/account."""
    if len(df) == 0:
        return []
    df = add_trade_style_columns(df)
    amounts = df['amount'].values
    q33, q66 = np.percentile(amounts, [33.33, 66.67])
    turnover_style = turnover_bucket(compute_entity_turnover(df))

    tokens = []
    for _, row in df.sort_values('date').iterrows():
        size = amount_bucket(float(row['amount']), q33, q66)
        hold = holding_bucket(row['matched_holding_days'])
        ret = return_bucket(row['realized_return'])
        dd = drawdown_bucket(float(row['running_drawdown']))
        mkt = market_bucket(float(row['market_state_value']))
        tokens.append(
            f"{row['industry']}_{row['action']}_A{size}_H{hold}_T{turnover_style}_R{ret}_D{dd}_M{mkt}"
        )
    return tokens


print("\n--- 构建 Token 序列 ---")

strategy_sequences = {}
for sname in sorted(strategies['strategy_name'].unique()):
    group = strategies[strategies['strategy_name'] == sname]
    strategy_sequences[sname] = build_tokens_for_entity(group)
    print(f"  [{sname}]: {len(strategy_sequences[sname])} tokens")

account_sequences = {}
for aid in sorted(accounts['account_id'].unique()):
    group = accounts[accounts['account_id'] == aid]
    account_sequences[aid] = build_tokens_for_entity(group)
    print(f"  [Account {aid}]: {len(account_sequences[aid])} tokens")

# ============================================================
# 4.3 构建词表
# ============================================================

print("\n--- 构建词表 ---")

all_sequences = list(strategy_sequences.values()) + list(account_sequences.values())
all_tokens = set()
for seq in all_sequences:
    all_tokens.update(seq)

sorted_tokens = sorted(all_tokens)
token2id = {tok: i for i, tok in enumerate(sorted_tokens)}
id2token = {i: tok for tok, i in token2id.items()}
VOCAB_SIZE = len(token2id)

print(f"  词表大小: {VOCAB_SIZE}")

ind_token_count = defaultdict(int)
for seq in all_sequences:
    for tok in seq:
        ind = tok.split('_', 1)[0]
        ind_token_count[ind] += 1

print("\n  Token 分布 (按行业):")
for ind in sorted(ind_token_count.keys()):
    print(f"    {ind:8s}: {ind_token_count[ind]:4d}")

# ============================================================
# 4.4 Skip-gram Word2Vec (PyTorch)
# ============================================================

print("\n--- Skip-gram Word2Vec 训练 (PyTorch) ---")

VECTOR_SIZE = 64
WINDOW = 5
NEG_SAMPLES = 5
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 0.002

print("  构建 Skip-gram 正样本对...")
pos_pairs = []
for seq in all_sequences:
    ids = [token2id[t] for t in seq]
    for i, center in enumerate(ids):
        start = max(0, i - WINDOW)
        end = min(len(ids), i + WINDOW + 1)
        for j in range(start, end):
            if i != j:
                pos_pairs.append((center, ids[j]))

pos_pairs = np.array(pos_pairs, dtype=np.int64)
num_pairs = len(pos_pairs)
print(f"  正样本对数: {num_pairs}")

token_counts = np.zeros(VOCAB_SIZE, dtype=np.float32)
for seq in all_sequences:
    for tok in seq:
        token_counts[token2id[tok]] += 1
noise_dist = token_counts ** 0.75
noise_dist /= noise_dist.sum()


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context, neg_samples):
        v_c = self.in_embed(center)
        u_pos = self.out_embed(context)
        u_neg = self.out_embed(neg_samples)
        pos_score = (v_c * u_pos).sum(dim=1)
        neg_score = (v_c.unsqueeze(1) * u_neg).sum(dim=2)
        return pos_score, neg_score


device = torch.device('cpu')
model = SkipGram(VOCAB_SIZE, VECTOR_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
print(f"  vector_size={VECTOR_SIZE}, window={WINDOW}, neg_samples={NEG_SAMPLES}, "
      f"epochs={EPOCHS}, batch={BATCH_SIZE}")

model.train()
total_batches = num_pairs // BATCH_SIZE

for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(num_pairs)
    epoch_loss = 0.0

    for b in range(total_batches):
        idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        center_b = torch.tensor(pos_pairs[idx, 0], dtype=torch.long, device=device)
        context_b = torch.tensor(pos_pairs[idx, 1], dtype=torch.long, device=device)
        neg_b = torch.multinomial(
            torch.tensor(noise_dist, dtype=torch.float32),
            BATCH_SIZE * NEG_SAMPLES,
            replacement=True,
        ).view(BATCH_SIZE, NEG_SAMPLES).to(device)

        pos_score, neg_score = model(center_b, context_b, neg_b)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10).mean()
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-10).mean()
        loss = pos_loss + neg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        avg_loss = epoch_loss / max(total_batches, 1)
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}")

embedding_matrix = model.in_embed.weight.detach().cpu().numpy()
print(f"  Embedding 矩阵: {embedding_matrix.shape}")

# ============================================================
# 4.5 保存
# ============================================================

print("\n--- 保存 ---")

with open('token_vocab.json', 'w', encoding='utf-8') as f:
    json.dump({'token2id': token2id, 'id2token': {str(k): v for k, v in id2token.items()}},
              f, ensure_ascii=False, indent=2)

np.save('word2vec_embeddings.npy', embedding_matrix)

tokenized = {
    'strategies': {s: [token2id[t] for t in seq] for s, seq in strategy_sequences.items()},
    'accounts': {a: [token2id[t] for t in seq] for a, seq in account_sequences.items()},
}
with open('tokenized_sequences.pkl', 'wb') as f:
    pickle.dump(tokenized, f)

seq_rows = []
for entity_type, entities in [('strategy', strategy_sequences), ('account', account_sequences)]:
    for name, tokens in entities.items():
        seq_rows.append({
            'entity_type': entity_type,
            'entity_name': name,
            'seq_length': len(tokens),
            'token_ids': ','.join(str(token2id[t]) for t in tokens),
            'tokens_preview': ' '.join(tokens[:20]),
        })
pd.DataFrame(seq_rows).to_csv('token_sequences.csv', index=False, encoding='utf-8-sig')

torch.save(model.state_dict(), 'word2vec_model.pt')

print(f"  token_vocab.json         — 词表 ({VOCAB_SIZE} tokens)")
print(f"  word2vec_embeddings.npy  — {embedding_matrix.shape} embedding 矩阵")
print(f"  tokenized_sequences.pkl  — token id 序列 (Step 6 用)")
print(f"  token_sequences.csv      — 可读版序列")
print(f"  word2vec_model.pt        — PyTorch 模型权重")

# ============================================================
# 4.6 语义验证：查询最相似 tokens
# ============================================================

print("\n--- 语义验证: 查询最相似 tokens ---")

norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
emb_norm = embedding_matrix / (norms + 1e-10)


def most_similar(token, topn=5):
    if token not in token2id:
        return []
    tid = token2id[token]
    vec = emb_norm[tid]
    sims = emb_norm @ vec
    sims[tid] = -1
    top_ids = np.argsort(-sims)[:topn]
    return [(id2token[i], float(sims[i])) for i in top_ids]


test_queries = sorted_tokens[:4]
for query in test_queries:
    print(f"\n  与 '{query}' 最相似:")
    similar = most_similar(query, topn=5)
    for tok, score in similar:
        print(f"    {tok:72s}  sim={score:.4f}")

print("\nDone. Step 4 完成.")
