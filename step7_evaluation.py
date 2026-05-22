"""
Step 7: 匹配评估 + 归因分析
7.1 Phase 1: 特征工程匹配（Baseline）
7.2 Phase 2: 深度学习匹配（LSTM Embedding）—— 从 Step 6 加载
7.3 两阶段对比（Spearman / Top-K 一致性）
7.4 SHAP 特征归因
"""
import pandas as pd
import numpy as np
import json
import sys
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

np.random.seed(42)

# ============================================================
# 7.1 加载数据
# ============================================================
print("=" * 60)
print("Step 7: 匹配评估 + 归因分析")
print("=" * 60)

strategy_feats = pd.read_csv('strategy_features.csv')
account_feats = pd.read_csv('account_features.csv')
sim_strat_feats = pd.read_csv('simulated_strategies_features.csv')
sim_acct_feats = pd.read_csv('simulated_accounts_features.csv')
train_pairs = pd.read_csv('train_pairs.csv')

# LSTM 向量
strat_vecs = np.load('strategy_embeddings.npy')   # (34, 128)
acct_vecs = np.load('account_embeddings.npy')      # (3, 128)

with open('embedding_meta.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

strategy_names = meta['strategy_names']
account_names = meta['account_names']

print(f"  真实策略: {len(strategy_names)}, 真实账户: {len(account_names)}")
print(f"  模拟策略: {len(sim_strat_feats)}, 模拟账户: {len(sim_acct_feats)}")
print(f"  训练标签对: {len(train_pairs)}")

# ============================================================
# 7.2 特征列定义
# ============================================================
ind_cols = [c for c in strategy_feats.columns if c.startswith('ind_')]
num_cols = [c for c in strategy_feats.columns
            if c not in ['name'] and not c.startswith('ind_')]
ALL_FEAT_COLS = num_cols + ind_cols

# ============================================================
# 7.3 Phase 1: 特征工程匹配（Baseline）
# ============================================================
print("\n--- 7.3 Phase 1: 特征工程匹配 ---")

# 提取特征矩阵
real_strat_feat = strategy_feats[ALL_FEAT_COLS].values.astype(np.float64)
real_acct_feat = account_feats[ALL_FEAT_COLS].values.astype(np.float64)

# 标准化
scaler_p1 = StandardScaler()
all_feat = np.vstack([real_strat_feat, real_acct_feat])
scaler_p1.fit(all_feat)
strat_feat_scaled = scaler_p1.transform(real_strat_feat)
acct_feat_scaled = scaler_p1.transform(real_acct_feat)

# 余弦相似度
sim_p1 = cosine_similarity(acct_feat_scaled, strat_feat_scaled)  # (3, 34)

df_p1 = pd.DataFrame(
    sim_p1,
    index=account_names,
    columns=strategy_names
)

print("\n  Phase 1 特征匹配矩阵:")
print(df_p1.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n  Phase 1 Top-5 (每个账户):")
p1_top5 = {}
for i, aname in enumerate(account_names):
    row = df_p1.loc[aname]
    top5 = row.sort_values(ascending=False).head(5)
    p1_top5[aname] = set(top5.index)
    print(f"\n  {aname}:")
    for j, (sname, score) in enumerate(top5.items()):
        print(f"    {j+1}. {sname:30s}  sim={score:.4f}")

# ============================================================
# 7.4 Phase 2: 深度学习匹配（LSTM）
# ============================================================
print("\n--- 7.4 Phase 2: 深度学习匹配 ---")

sim_p2 = cosine_similarity(acct_vecs, strat_vecs)  # (3, 34)

df_p2 = pd.DataFrame(
    sim_p2,
    index=account_names,
    columns=strategy_names
)

print("\n  Phase 2 LSTM匹配矩阵:")
print(df_p2.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n  Phase 2 Top-5 (每个账户):")
p2_top5 = {}
for i, aname in enumerate(account_names):
    row = df_p2.loc[aname]
    top5 = row.sort_values(ascending=False).head(5)
    p2_top5[aname] = set(top5.index)
    print(f"\n  {aname}:")
    for j, (sname, score) in enumerate(top5.items()):
        print(f"    {j+1}. {sname:30s}  sim={score:.4f}")

# ============================================================
# 7.5 两阶段对比
# ============================================================
print("\n" + "=" * 60)
print("--- 7.5 两阶段对比分析 ---")

# 7.5.1 Spearman 秩相关
print("\n  Spearman 秩相关系数 (每个账户, Phase 1 vs Phase 2):")
for i, aname in enumerate(account_names):
    rho, pval = spearmanr(sim_p1[i], sim_p2[i])
    print(f"    {aname}: ρ={rho:.4f} (p={pval:.4f})")

# 7.5.2 Top-K 重叠率
print("\n  Top-K 重叠率:")
for k in [1, 3, 5]:
    overlaps = []
    for aname in account_names:
        p1_set = set(df_p1.loc[aname].sort_values(ascending=False).head(k).index)
        p2_set = set(df_p2.loc[aname].sort_values(ascending=False).head(k).index)
        overlap = len(p1_set & p2_set) / k
        overlaps.append(overlap)
    print(f"    Top-{k}: {[f'{o:.0%}' for o in overlaps]}  mean={np.mean(overlaps):.1%}")

# 7.5.3 排名差异最大的策略
print("\n  排名差异最大的策略 (Phase 2 rank - Phase 1 rank):")
for i, aname in enumerate(account_names):
    rank_p1 = np.argsort(np.argsort(-sim_p1[i]))  # 0 = best
    rank_p2 = np.argsort(np.argsort(-sim_p2[i]))
    rank_diff = rank_p2 - rank_p1

    # Top-5 上升最多的
    top_risers = np.argsort(-rank_diff)[:3]
    top_fallers = np.argsort(rank_diff)[:3]

    print(f"\n  {aname}:")
    print(f"    上升最多 (Phase 2 排名更高):")
    for idx in top_risers:
        print(f"      {strategy_names[idx]:30s}  P1=#{rank_p1[idx]+1} → P2=#{rank_p2[idx]+1}  (Δ={-rank_diff[idx]})")
    print(f"    下降最多 (Phase 2 排名更低):")
    for idx in top_fallers:
        print(f"      {strategy_names[idx]:30s}  P1=#{rank_p1[idx]+1} → P2=#{rank_p2[idx]+1}  (Δ={-rank_diff[idx]})")

# 7.5.4 相似度分布对比
print("\n  相似度分布对比:")
print(f"    Phase 1 (特征):  mean={sim_p1.mean():.4f}, std={sim_p1.std():.4f}, "
      f"min={sim_p1.min():.4f}, max={sim_p1.max():.4f}")
print(f"    Phase 2 (LSTM):  mean={sim_p2.mean():.4f}, std={sim_p2.std():.4f}, "
      f"min={sim_p2.min():.4f}, max={sim_p2.max():.4f}")

# ============================================================
# 7.6 SHAP 特征归因
# ============================================================
print("\n" + "=" * 60)
print("--- 7.6 SHAP 特征归因 ---")

# 在模拟数据上：用扩展特征预测弱监督伪标签，再用 SHAP 解释
# 为每个 (account, strategy) 对构造特征：特征差的绝对值 + 特征积（交互）

sim_strat_feat_mat = sim_strat_feats[ALL_FEAT_COLS].values.astype(np.float64)
sim_acct_feat_mat = sim_acct_feats[ALL_FEAT_COLS].values.astype(np.float64)

# 从 train_pairs 构造特征矩阵
X_feat_list = []
y_match = []
pair_info = []  # (client_idx, strat_idx)

for _, row in train_pairs.iterrows():
    ci = int(row['client_idx'])
    si = int(row['strategy_idx'])
    is_match = int(row['is_match'])

    a_feat = sim_acct_feat_mat[ci]
    s_feat = sim_strat_feat_mat[si]

    # 特征工程: 差值 + 乘积 + 原始值
    diff = np.abs(a_feat - s_feat)
    prod = a_feat * s_feat
    combined = np.concatenate([diff, prod, a_feat, s_feat])

    X_feat_list.append(combined)
    y_match.append(is_match)
    pair_info.append((ci, si))

X_feat = np.array(X_feat_list)  # (1000, 144)
y_match = np.array(y_match)

# 标准化
scaler_shap = StandardScaler()
X_feat_scaled = scaler_shap.fit_transform(X_feat)

# 特征名
diff_names = [f"diff_{c}" for c in ALL_FEAT_COLS]
prod_names = [f"prod_{c}" for c in ALL_FEAT_COLS]
acct_names_feat = [f"acct_{c}" for c in ALL_FEAT_COLS]
strat_names_feat = [f"strat_{c}" for c in ALL_FEAT_COLS]
all_feat_names = diff_names + prod_names + acct_names_feat + strat_names_feat

# 训练 XGBoost 分类器
from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
)
xgb.fit(X_feat_scaled, y_match)

# 训练集准确率
y_pred = xgb.predict(X_feat_scaled)
train_acc = (y_pred == y_match).mean()
print(f"\n  XGBoost 训练准确率: {train_acc:.3f}")

# 特征重要性 (gain)
importance = xgb.get_booster().get_score(importance_type='gain')
print("\n  Top-15 特征重要性 (by gain):")
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
for feat, score in sorted_imp:
    # f0, f1, ... → 真实特征名
    idx = int(feat[1:])
    print(f"    {all_feat_names[idx]:40s}  gain={score:.1f}")

# SHAP 分析
import shap

print("\n  计算 SHAP 值...")
explainer = shap.TreeExplainer(xgb)
# 取子集计算 SHAP (全部 1000 条较慢，抽 300 条)
sample_idx = np.random.choice(len(X_feat_scaled), size=min(300, len(X_feat_scaled)), replace=False)
X_sample = X_feat_scaled[sample_idx]
shap_values = explainer.shap_values(X_sample)

# 汇总 SHAP (用简洁的数字表展示)
print("\n  SHAP 特征归因 (Top-15, 按 mean|SHAP|):")
shap_mean_abs = np.abs(shap_values).mean(axis=0)
shap_top_idx = np.argsort(-shap_mean_abs)[:15]

for rank, idx in enumerate(shap_top_idx):
    feat_name = all_feat_names[idx]
    mean_impact = shap_mean_abs[idx]
    # 区分正向/负向影响
    pos_impact = (shap_values[:, idx] > 0).mean()
    print(f"    {rank+1:2d}. {feat_name:40s}  |SHAP|={mean_impact:.4f}  "
          f"正向占比={pos_impact:.1%}")

# SHAP 按原始特征分组汇总
print("\n  按原始特征汇总 SHAP (合并 diff/prod/acct/strat):")
feature_group_impact = {}
for col in ALL_FEAT_COLS:
    group_indices = [i for i, name in enumerate(all_feat_names) if col in name]
    group_shap = shap_mean_abs[group_indices].sum()
    feature_group_impact[col] = group_shap

sorted_groups = sorted(feature_group_impact.items(), key=lambda x: x[1], reverse=True)
for col, impact in sorted_groups[:10]:
    print(f"    {col:35s}  combined|SHAP|={impact:.4f}")

# ============================================================
# 7.7 最终匹配推荐
# ============================================================
print("\n" + "=" * 60)
print("--- 7.7 最终匹配推荐 ---")

# 综合 Phase 1 和 Phase 2 的排名（平均排名）
rank_p1 = np.argsort(np.argsort(-sim_p1, axis=1), axis=1)  # (3, 34), 0=best
rank_p2 = np.argsort(np.argsort(-sim_p2, axis=1), axis=1)
avg_rank = (rank_p1 + rank_p2) / 2  # 平均排名
final_rank_idx = np.argsort(avg_rank, axis=1)

print("\n  综合推荐 (Phase 1 + Phase 2 平均排名 Top-5):")
for i, aname in enumerate(account_names):
    print(f"\n  {aname}:")
    for j in range(5):
        sidx = final_rank_idx[i, j]
        sname = strategy_names[sidx]
        print(f"    {j+1}. {sname:30s}  "
              f"P1_rank=#{rank_p1[i,sidx]+1}  P2_rank=#{rank_p2[i,sidx]+1}  "
              f"P1_sim={sim_p1[i,sidx]:.4f}  P2_sim={sim_p2[i,sidx]:.4f}")

# ============================================================
# 7.8 保存
# ============================================================
print("\n" + "=" * 60)
print("--- 7.8 保存 ---")

# 匹配矩阵
df_p1.to_csv('matching_phase1_features.csv', encoding='utf-8-sig')
df_p2.to_csv('matching_phase2_lstm.csv', encoding='utf-8-sig')

# 综合推荐
final_recs = []
for i, aname in enumerate(account_names):
    for j in range(5):
        sidx = final_rank_idx[i, j]
        final_recs.append({
            'account': aname,
            'rank': j + 1,
            'strategy': strategy_names[sidx],
            'phase1_rank': int(rank_p1[i, sidx] + 1),
            'phase2_rank': int(rank_p2[i, sidx] + 1),
            'phase1_similarity': float(sim_p1[i, sidx]),
            'phase2_similarity': float(sim_p2[i, sidx]),
        })
pd.DataFrame(final_recs).to_csv('final_recommendations.csv', index=False, encoding='utf-8-sig')

# SHAP
shap_report = {
    'top_features_by_gain': [(all_feat_names[int(k[1:])], float(v))
                              for k, v in sorted_imp],
    'top_features_by_shap': [(all_feat_names[idx], float(shap_mean_abs[idx]))
                              for idx in shap_top_idx],
    'grouped_shap': [(col, float(impact)) for col, impact in sorted_groups],
}
with open('shap_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(shap_report, f, ensure_ascii=False, indent=2)

print(f"  matching_phase1_features.csv  — Phase 1 特征匹配矩阵")
print(f"  matching_phase2_lstm.csv      — Phase 2 LSTM 匹配矩阵")
print(f"  final_recommendations.csv     — 综合推荐 (Top-5 × 3)")
print(f"  shap_analysis.json            — SHAP 特征归因结果")

print("\nDone. Step 7 完成. 全流程结束!")
