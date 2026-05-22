"""
Step 5: 模拟数据生成
5.1 基于 34 策略特征 → 500 模拟策略特征 + token 序列
5.2 基于 3 账户特征 → 200 模拟客户特征 + token 序列
5.3 建立匹配标签 (正/负样本对)
"""
import pandas as pd
import numpy as np
import json
import pickle
import sys
from collections import defaultdict
from scipy.stats import dirichlet

sys.stdout.reconfigure(encoding='utf-8')

np.random.seed(42)

# ============================================================
# 5.1 加载数据
# ============================================================
print("=" * 60)
print("Step 5: 模拟数据生成")
print("=" * 60)

strategy_feats = pd.read_csv('strategy_features.csv')
account_feats = pd.read_csv('account_features.csv')

with open('tokenized_sequences.pkl', 'rb') as f:
    tokenized = pickle.load(f)

with open('token_vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

token2id = vocab['token2id']
id2token = {int(k): v for k, v in vocab['id2token'].items()}
VOCAB_SIZE = len(token2id)

# 特征列: 自动纳入 Step 3 输出的所有数值风格特征
ind_cols = [c for c in strategy_feats.columns if c.startswith('ind_')]
num_cols = [c for c in strategy_feats.columns
            if c not in ['name'] and not c.startswith('ind_')]
ALL_INDUSTRIES = [c.replace('ind_', '') for c in ind_cols]

print(f"  真实策略: {len(strategy_feats)}, 真实账户: {len(account_feats)}")
print(f"  词表大小: {VOCAB_SIZE}")

# ============================================================
# 5.2 特征扰动生成
# ============================================================

def extract_features(df_row):
    """从 DataFrame 行提取 5 维数值特征 + 31 维行业偏好"""
    num_feat = df_row[num_cols].values.astype(np.float64)
    ind_feat = df_row[ind_cols].values.astype(np.float64)
    ind_feat = ind_feat / ind_feat.sum()  # 确保归一化
    return num_feat, ind_feat


def perturb_features(num_feat, ind_feat, noise_scale=0.15):
    """
    扰动特征向量:
    - 非负尺度特征: 乘性对数正态噪声
    - 可正可负特征: 加性正态噪声
    - 行业偏好: Dirichlet 噪声
    """
    new_num = num_feat.astype(np.float64).copy()
    for j, col in enumerate(num_cols):
        value = float(num_feat[j])
        if col in ['realized_return_pref', 'market_state_exposure']:
            new_num[j] = value + np.random.normal(0, noise_scale * 0.25)
        else:
            new_num[j] = value * np.exp(np.random.normal(0, noise_scale))

        if col == 'buy_sell_symmetry':
            new_num[j] = np.clip(new_num[j], 0.0, 1.0)
        elif col == 'concentration_hhi':
            new_num[j] = np.clip(new_num[j], 0.0, 1.0)
        elif col == 'volatility_pref':
            new_num[j] = np.clip(new_num[j], 0.0, 2.0)
        elif col == 'realized_return_pref':
            new_num[j] = np.clip(new_num[j], -1.0, 3.0)
        elif col == 'max_drawdown_pref':
            new_num[j] = np.clip(new_num[j], 0.0, 1.0)
        elif col == 'market_state_exposure':
            new_num[j] = np.clip(new_num[j], -1.0, 1.0)
        else:
            new_num[j] = np.clip(new_num[j], 0.001, 300.0)

    alpha = ind_feat * (1.0 / noise_scale)
    alpha = np.maximum(alpha, 0.01)
    new_ind = dirichlet.rvs(alpha, size=1)[0]

    return new_num, new_ind


print("\n--- 5.2 生成模拟特征 ---")

# 提取所有真实实体的特征
real_strategy_num = []
real_strategy_ind = []
for _, row in strategy_feats.iterrows():
    n, i = extract_features(row)
    real_strategy_num.append(n)
    real_strategy_ind.append(i)

real_account_num = []
real_account_ind = []
for _, row in account_feats.iterrows():
    n, i = extract_features(row)
    real_account_num.append(n)
    real_account_ind.append(i)

real_strategy_num = np.array(real_strategy_num)
real_strategy_ind = np.array(real_strategy_ind)
real_account_num = np.array(real_account_num)
real_account_ind = np.array(real_account_ind)

# 生成模拟策略特征 (2000)
N_SIM_STRATEGY = 2000
sim_strategy_num = []
sim_strategy_ind = []
sim_strategy_parent = []  # 记录来自哪个真实策略
noise_scales = [0.08, 0.12, 0.16, 0.20]  # 多档噪声增加多样性

for i in range(N_SIM_STRATEGY):
    parent_idx = i % len(real_strategy_num)  # 轮流取真实策略作为模板
    noise_s = noise_scales[i % len(noise_scales)]  # 交替用不同噪声
    n, ind = perturb_features(real_strategy_num[parent_idx],
                               real_strategy_ind[parent_idx],
                               noise_scale=noise_s)
    sim_strategy_num.append(n)
    sim_strategy_ind.append(ind)
    sim_strategy_parent.append(strategy_feats.iloc[parent_idx]['name'])

sim_strategy_num = np.array(sim_strategy_num)
sim_strategy_ind = np.array(sim_strategy_ind)

# 生成模拟客户特征 (1000)
N_SIM_ACCOUNT = 1000
sim_account_num = []
sim_account_ind = []
sim_account_parent = []

for i in range(N_SIM_ACCOUNT):
    parent_idx = i % len(real_account_num)
    noise_s = noise_scales[i % len(noise_scales)]
    n, ind = perturb_features(real_account_num[parent_idx],
                               real_account_ind[parent_idx],
                               noise_scale=noise_s)
    sim_account_num.append(n)
    sim_account_ind.append(ind)
    sim_account_parent.append(account_feats.iloc[parent_idx]['name'])

sim_account_num = np.array(sim_account_num)
sim_account_ind = np.array(sim_account_ind)

print(f"  模拟策略: {N_SIM_STRATEGY}, 模拟客户: {N_SIM_ACCOUNT}")

# ============================================================
# 5.3 建立匹配标签
# ============================================================

print("\n--- 5.3 建立匹配标签 ---")

# 对每个模拟客户，找特征最相似的模拟策略作为正样本
# 同时在所有策略中做一个简单的特征距离计算

# 标准化数值特征用于距离计算
from sklearn.preprocessing import StandardScaler
all_num = np.vstack([sim_strategy_num, sim_account_num])
scaler = StandardScaler().fit(all_num)
strat_num_scaled = scaler.transform(sim_strategy_num)
acct_num_scaled = scaler.transform(sim_account_num)

# 用扩展数值风格特征 + 行业偏好前 5 维的余弦距离
# 这是弱监督伪标签，不是真实客户适配标签
top_indices = np.argsort(-sim_strategy_ind, axis=1)[:, :5]
strat_top5 = np.zeros((N_SIM_STRATEGY, 5))
for i in range(N_SIM_STRATEGY):
    strat_top5[i] = sim_strategy_ind[i, top_indices[i]]

top_indices_a = np.argsort(-sim_account_ind, axis=1)[:, :5]
acct_top5 = np.zeros((N_SIM_ACCOUNT, 5))
for i in range(N_SIM_ACCOUNT):
    acct_top5[i] = sim_account_ind[i, top_indices_a[i]]

strat_feat = np.hstack([strat_num_scaled, strat_top5])
acct_feat = np.hstack([acct_num_scaled, acct_top5])

# 余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(acct_feat, strat_feat)

# 困难负样本策略:
# - 正样本: rank-1 (最相似)
# - 硬负样本: rank-10~60 (相似但不完全匹配，迫使模型学精细决策)
# - 易负样本: bottom-20% (兜底，防止全部太难导致不收敛)
match_pairs = []  # list of (client_idx, strategy_idx, is_match)
N_NEG_PER_POS = 5        # 每个正样本配 5 个负样本
N_HARD_NEG = 3            # 其中 3 个是困难负样本
N_EASY_NEG = 2            # 其中 2 个是易负样本

strat_ranked = np.argsort(-sim_matrix, axis=1)  # 每个客户对各策略从高到低排序

for ci in range(N_SIM_ACCOUNT):
    pos_si = strat_ranked[ci, 0]  # 最相似 = 正样本
    match_pairs.append((ci, pos_si, 1))

    # 困难负样本: 从 rank-10 到 rank-60 中随机选
    hard_start, hard_end = 10, min(60, N_SIM_STRATEGY)
    hard_candidates = strat_ranked[ci, hard_start:hard_end]
    hard_samples = np.random.choice(hard_candidates, size=N_HARD_NEG, replace=False)

    # 易负样本: 从 bottom-20% 中随机选
    easy_start = int(N_SIM_STRATEGY * 0.8)
    easy_candidates = strat_ranked[ci, easy_start:]
    easy_samples = np.random.choice(easy_candidates, size=N_EASY_NEG, replace=False)

    for ns in hard_samples:
        match_pairs.append((ci, ns, 0))
    for ns in easy_samples:
        match_pairs.append((ci, ns, 0))

n_pos = N_SIM_ACCOUNT
n_neg = N_SIM_ACCOUNT * N_NEG_PER_POS
print(f"  匹配对总数: {len(match_pairs)} (正样本: {n_pos}, 硬负样本: {n_pos * N_HARD_NEG}, 易负样本: {n_pos * N_EASY_NEG})")
print(f"  正样本平均相似度: {sim_matrix[np.arange(N_SIM_ACCOUNT), strat_ranked[:, 0]].mean():.4f}")

# ============================================================
# 5.4 生成模拟交易序列
# ============================================================

print("\n--- 5.4 生成模拟交易序列 ---")

def get_seq_length(turnover, holding_period, base_min=50, base_max=3000):
    """
    根据换手率和持仓周期估算序列长度
    高换手率 + 短持仓 → 长序列 (高频交易)
    低换手率 + 长持仓 → 短序列
    """
    # 把 turnover 和 holding_period 映射到序列长度
    turnover_score = np.log1p(turnover) / np.log1p(200)  # 0~1
    holding_score = 1.0 - np.clip(np.log1p(holding_period) / np.log1p(100), 0, 1)  # 1~0

    # 综合分数: 两者平均
    activity = (turnover_score + holding_score) / 2
    seq_len = int(base_min + activity * (base_max - base_min))
    return seq_len


def generate_sequence(num_feat, ind_feat, all_real_sequences, token2id,
                      id2token, all_industries, min_len=50, max_len=3000):
    """
    为模拟实体生成 token 序列
    策略: 块采样 (block bootstrap) + 行业偏好重加权

    1. 从所有真实序列中提取 token 块 (长度 3~15)
    2. 按行业偏好概率采样块
    3. 拼接直到达到目标长度
    4. 微调 BUY/SELL 比例
    """
    turnover = num_feat[0]
    holding = num_feat[1]
    target_len = get_seq_length(turnover, holding, min_len, max_len)

    # 从所有真实序列中提取块
    all_blocks = []  # list of (list_of_token_ids, industry_of_block)
    for seq_tokens in all_real_sequences:
        if len(seq_tokens) < 3:
            continue
        # 滑动窗口提取块
        for start in range(0, len(seq_tokens) - 2, 3):
            block_len = min(np.random.randint(3, 12), len(seq_tokens) - start)
            block = seq_tokens[start:start + block_len]
            # 确定这个块的主要行业
            block_inds = set()
            for tid in block:
                token_str = id2token.get(tid, '')
                if token_str:
                    ind = token_str.split('_')[0]
                    block_inds.add(ind)
            all_blocks.append((block, block_inds))

    if not all_blocks:
        return np.random.randint(0, len(token2id), size=target_len).tolist()

    # 按行业偏好采样块
    ind_probs = dict(zip(all_industries, ind_feat))
    generated = []

    while len(generated) < target_len:
        # 选择块: 偏好与目标行业分布匹配的块
        if np.random.random() < 0.7:
            # 用行业偏好采样
            ind_weights = np.array([max(ind_probs.get(list(bi)[0], 0.001), 0.001)
                                     for _, bi in all_blocks])
            ind_weights /= ind_weights.sum()
            bidx = np.random.choice(len(all_blocks), p=ind_weights)
        else:
            # 随机块 (增加多样性)
            bidx = np.random.randint(len(all_blocks))

        block, _ = all_blocks[bidx]
        generated.extend(block)

    # 截断到目标长度
    generated = generated[:target_len]

    # 微调: 随机翻转一些 token 的 action (BUY ↔ SELL)
    # 使得最终 BUY 比例接近 buy_sell_symmetry
    buy_ratio_target = num_feat[3]  # buy_sell_symmetry
    # 统计当前 BUY 比例
    buy_count = 0
    total_count = 0
    for tid in generated:
        token_str = id2token.get(tid, '')
        if '_BUY_' in token_str:
            buy_count += 1
            total_count += 1
        elif '_SELL_' in token_str:
            total_count += 1

    if total_count > 0:
        current_buy_ratio = buy_count / total_count
        # 如果需要调整
        if abs(current_buy_ratio - buy_ratio_target) > 0.05:
            # 决定需要翻转多少个
            target_buy_count = int(total_count * buy_ratio_target)
            diff = target_buy_count - buy_count

            if diff > 0:
                # 需要更多 BUY: 随机选 |diff| 个 SELL 翻成 BUY
                sell_positions = [j for j, tid in enumerate(generated)
                                  if '_SELL_' in id2token.get(tid, '')]
                positions_to_flip = np.random.choice(
                    sell_positions, size=min(abs(diff), len(sell_positions)), replace=False)
                for pos in positions_to_flip:
                    old = id2token[generated[pos]]
                    new = old.replace('_SELL_', '_BUY_')
                    if new in token2id:
                        generated[pos] = token2id[new]
            else:
                # 需要更多 SELL
                buy_positions = [j for j, tid in enumerate(generated)
                                 if '_BUY_' in id2token.get(tid, '')]
                positions_to_flip = np.random.choice(
                    buy_positions, size=min(abs(diff), len(buy_positions)), replace=False)
                for pos in positions_to_flip:
                    old = id2token[generated[pos]]
                    new = old.replace('_BUY_', '_SELL_')
                    if new in token2id:
                        generated[pos] = token2id[new]

    return generated


# 准备所有真实序列
all_real_seqs = []
for seq in tokenized['strategies'].values():
    all_real_seqs.append(seq)
for seq in tokenized['accounts'].values():
    all_real_seqs.append(seq)

# 生成模拟策略的序列
print("  生成模拟策略序列...")
sim_strategy_seqs = {}
for i in range(N_SIM_STRATEGY):
    seq = generate_sequence(sim_strategy_num[i], sim_strategy_ind[i],
                            all_real_seqs, token2id, id2token, ALL_INDUSTRIES,
                            min_len=80, max_len=3000)
    sim_strategy_seqs[f"sim_strat_{i:04d}"] = seq
    if i % 100 == 0:
        print(f"    策略 {i}/{N_SIM_STRATEGY}... (len={len(seq)})")

# 生成模拟客户的序列
print("  生成模拟客户序列...")
sim_account_seqs = {}
for i in range(N_SIM_ACCOUNT):
    seq = generate_sequence(sim_account_num[i], sim_account_ind[i],
                            all_real_seqs, token2id, id2token, ALL_INDUSTRIES,
                            min_len=50, max_len=2000)
    sim_account_seqs[f"sim_acct_{i:04d}"] = seq
    if i % 50 == 0:
        print(f"    客户 {i}/{N_SIM_ACCOUNT}... (len={len(seq)})")

# ============================================================
# 5.5 保存
# ============================================================

print("\n--- 5.5 保存 ---")

# 特征向量保存为 CSV/JSON
def save_sim_features(num_arr, ind_arr, names, prefix):
    rows = []
    for i in range(len(names)):
        row = {'name': names[i]}
        for j, col in enumerate(num_cols):
            row[col] = num_arr[i, j]
        for j, ind_name in enumerate(ALL_INDUSTRIES):
            row[f'ind_{ind_name}'] = ind_arr[i, j]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'simulated_{prefix}_features.csv', index=False, encoding='utf-8-sig')
    return df

save_sim_features(sim_strategy_num, sim_strategy_ind,
                  [f"sim_strat_{i:04d}" for i in range(N_SIM_STRATEGY)],
                  'strategies')
save_sim_features(sim_account_num, sim_account_ind,
                  [f"sim_acct_{i:04d}" for i in range(N_SIM_ACCOUNT)],
                  'accounts')

# 完整模拟数据 (特征 + 序列)
simulated_data = {
    'strategies': {
        'features_num': sim_strategy_num,
        'features_ind': sim_strategy_ind,
        'sequences': sim_strategy_seqs,
        'names': [f"sim_strat_{i:04d}" for i in range(N_SIM_STRATEGY)],
    },
    'accounts': {
        'features_num': sim_account_num,
        'features_ind': sim_account_ind,
        'sequences': sim_account_seqs,
        'names': [f"sim_acct_{i:04d}" for i in range(N_SIM_ACCOUNT)],
    },
    'match_pairs': match_pairs,  # [(client_idx, strategy_idx, is_match)]
}

with open('simulated_data.pkl', 'wb') as f:
    pickle.dump(simulated_data, f)

# 匹配对保存为 CSV
pairs_df = pd.DataFrame(match_pairs, columns=['client_idx', 'strategy_idx', 'is_match'])
pairs_df.to_csv('train_pairs.csv', index=False)

# 序列保存为 CSV
seq_df_rows = []
for name, seq in sim_strategy_seqs.items():
    seq_df_rows.append({'entity_type': 'strategy', 'entity_name': name,
                        'seq_length': len(seq),
                        'token_ids': ','.join(str(t) for t in seq)})
for name, seq in sim_account_seqs.items():
    seq_df_rows.append({'entity_type': 'account', 'entity_name': name,
                        'seq_length': len(seq),
                        'token_ids': ','.join(str(t) for t in seq)})
pd.DataFrame(seq_df_rows).to_csv('simulated_sequences.csv', index=False, encoding='utf-8-sig')

print(f"  simulated_strategies_features.csv — {N_SIM_STRATEGY} 模拟策略特征")
print(f"  simulated_accounts_features.csv   — {N_SIM_ACCOUNT} 模拟客户特征")
print(f"  simulated_data.pkl                — 完整模拟数据 (特征+序列+匹配)")
print(f"  train_pairs.csv                   — {len(match_pairs)} 条训练标签")
print(f"  simulated_sequences.csv           — 模拟序列")

# 统计
strat_lens = [len(s) for s in sim_strategy_seqs.values()]
acct_lens = [len(s) for s in sim_account_seqs.values()]
print(f"\n  模拟策略序列长度: min={min(strat_lens)}, max={max(strat_lens)}, "
      f"mean={np.mean(strat_lens):.0f}")
print(f"  模拟客户序列长度: min={min(acct_lens)}, max={max(acct_lens)}, "
      f"mean={np.mean(acct_lens):.0f}")

print("\nDone. Step 5 完成.")
