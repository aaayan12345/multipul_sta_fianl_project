# 机器学习方法对接说明

本文档用于说明如何把本项目的机器学习匹配结果接入统计方法仓库：

当前机器学习模块定位为“小样本弱监督辅助排序器”：它不替代统计方法，而是输出客户-策略相似度矩阵和 Top-N 候选推荐，供统计方法做融合、复核或加权排序。

## 1. 对接目标

队友的统计方法只需要接入两个层次的结果：

1. 机器学习相似度矩阵：每个账户对每个策略的 LSTM 序列风格相似度。
2. 综合推荐长表：每个账户的 Top-N 策略，以及 Phase 1 特征排名和 Phase 2 LSTM 排名。

推荐优先接入 `matching_phase2_lstm.csv`，因为它是最通用的矩阵格式；如果只想快速展示结果，可以直接接入 `final_recommendations.csv`。

## 2. 运行环境

本项目使用 conda 环境 `stock`。

```bash
conda run -n stock python step3_feature_extraction.py
conda run -n stock python step4_word2vec_pretrain.py
conda run -n stock python step5_simulate_data.py
conda run -n stock python step6_lstm_contrastive.py
conda run -n stock python step7_evaluation.py
```

如果行业映射没有改动，可从 Step 4 开始运行；如果人工修改了 `stock_industry_mapping_review.csv`，必须先重跑 Step 3。

## 3. 输入文件

机器学习模块依赖以下文件：

| 文件 | 说明 | 是否需要队友提供 |
|---|---|---|
| `clean_strategies.csv` | 清洗后的策略交易记录 | 否，当前项目生成 |
| `clean_accounts.csv` | 清洗后的账户交易记录 | 否，当前项目生成 |
| `stock_industry_mapping_review.csv` | 人工校验后的行业映射，优先使用 `review_industry` | 可选，人工校验后更新 |
| `strategy_features.csv` | 策略画像特征 | Step 3 生成 |
| `account_features.csv` | 账户画像特征 | Step 3 生成 |
| `tokenized_sequences.pkl` | 策略和账户 token 序列 | Step 4 生成 |
| `word2vec_embeddings.npy` | Word2Vec 初始向量 | Step 4 生成 |

注意：股票名为空但股票代码存在时保留，不作为无效记录。行业映射以 `stock_code` 为主键。

## 4. 输出文件

### 4.1 `matching_phase2_lstm.csv`

机器学习方法的核心输出。格式是账户 × 策略的相似度矩阵。

示例结构：

```csv
,策略1,策略2,策略3
Account_A,0.2219,-0.1133,0.4684
Account_B,0.5395,0.5260,0.8649
Account_C,...
```

读取方式：

```python
import pandas as pd

ml_score = pd.read_csv('matching_phase2_lstm.csv', index_col=0)
# 行: account，例如 Account_A
# 列: strategy，例如 中证1000增强
# 值: LSTM cosine similarity，越大表示交易序列风格越相似
```

### 4.2 `matching_phase1_features.csv`

本项目内部的特征工程基线，格式与 `matching_phase2_lstm.csv` 相同。它不是队友统计方法的替代品，只用于和 LSTM 排名对比。

```python
feature_score = pd.read_csv('matching_phase1_features.csv', index_col=0)
```

### 4.3 `final_recommendations.csv`

最终推荐长表，适合直接展示 Top-N。

字段说明：

| 字段 | 说明 |
|---|---|
| `account` | 账户 ID，例如 `Account_A` |
| `rank` | 综合推荐排名 |
| `strategy` | 策略名称 |
| `phase1_rank` | 特征基线排名 |
| `phase2_rank` | LSTM 排名 |
| `phase1_similarity` | 特征相似度 |
| `phase2_similarity` | LSTM 相似度 |

读取方式：

```python
recs = pd.read_csv('final_recommendations.csv')
account_a_top5 = recs[recs['account'] == 'Account_A'].sort_values('rank')
```

## 5. 推荐融合方式

建议队友的统计方法输出同样的矩阵格式：

```csv
,策略1,策略2,策略3
Account_A,0.31,0.12,0.45
Account_B,0.27,0.66,0.18
```

假设统计方法输出为 `stat_score.csv`，可以这样融合：

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

stat = pd.read_csv('stat_score.csv', index_col=0)
ml = pd.read_csv('matching_phase2_lstm.csv', index_col=0)

# 对齐账户和策略
common_accounts = stat.index.intersection(ml.index)
common_strategies = stat.columns.intersection(ml.columns)
stat = stat.loc[common_accounts, common_strategies]
ml = ml.loc[common_accounts, common_strategies]

# 每个矩阵内部归一化到 0-1，避免量纲不同
scaler = MinMaxScaler()
stat_norm = pd.DataFrame(
    scaler.fit_transform(stat.values),
    index=stat.index,
    columns=stat.columns,
)
ml_norm = pd.DataFrame(
    scaler.fit_transform(ml.values),
    index=ml.index,
    columns=ml.columns,
)

# 推荐初始权重：统计方法为主，机器学习辅助
alpha = 0.7
final_score = alpha * stat_norm + (1 - alpha) * ml_norm

# 输出每个账户 Top-5
rows = []
for account, row in final_score.iterrows():
    top = row.sort_values(ascending=False).head(5)
    for rank, (strategy, score) in enumerate(top.items(), start=1):
        rows.append({
            'account': account,
            'rank': rank,
            'strategy': strategy,
            'final_score': score,
            'stat_score': stat.loc[account, strategy],
            'ml_score': ml.loc[account, strategy],
        })

pd.DataFrame(rows).to_csv('merged_recommendations.csv', index=False, encoding='utf-8-sig')
```

## 6. 推荐的融合口径

小样本条件下，建议不要让机器学习结果单独决定最终推荐。更稳的默认权重是：

| 场景 | 统计方法权重 | 机器学习权重 |
|---|---:|---:|
| 稳健展示版 | 0.8 | 0.2 |
| 平衡实验版 | 0.7 | 0.3 |
| 强调序列风格版 | 0.6 | 0.4 |

建议论文/汇报中使用 `0.7 / 0.3`，并说明机器学习是“序列风格辅助项”。

## 7. 账户和策略名称对齐规则

机器学习输出中的账户名格式为：

```text
Account_A
Account_B
Account_C
```

策略名直接使用清洗后策略名称，例如：

```text
中证1000增强
行业etf增强
煤炭周期优选动态轮动策略
```

如果统计方法使用 `A/B/C`，需要做一次映射：

```python
account_map = {
    'A': 'Account_A',
    'B': 'Account_B',
    'C': 'Account_C',
}
```

策略名必须完全一致。若统计方法使用简写，建议维护一个 `strategy_name_mapping.csv`，包含：

```csv
stat_strategy_name,ml_strategy_name
中证1000,中证1000增强
行业ETF,行业etf增强
```

## 8. 方法边界

机器学习模块的训练标签来自弱监督伪标签：即用扩展交易画像相似度生成正负样本，再训练 LSTM 学习交易序列风格。因此它适合做：

- 候选策略排序
- 与统计方法互相补充
- 解释交易风格相似性
- 作为后续真实标签训练的模型框架

它不应被单独表述为已经验证的真实投资适配模型。最终推荐仍需结合风险承受能力、收益回撤约束、人工校验和真实客户反馈。

## 9. 最小接入清单

队友只需要拿到以下两个文件即可完成第一版接入：

```text
matching_phase2_lstm.csv
final_recommendations.csv
```

如果要融合排序，再额外提供他的统计得分矩阵：

```text
stat_score.csv
```

三者统一成“账户为行、策略为列、分数越大越推荐”的格式即可。
