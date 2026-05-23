# 小样本弱监督下的证券投资策略与客户交易画像匹配研究

## Phase 2: 深度学习辅助赛道

本项目目标不是在小样本下证明“精准匹配”已经成立，而是构建一条可行的弱监督匹配路线：
先用可解释交易画像刻画客户与策略风格，再用多维交易 token + LSTM 对比学习作为辅助排序器，
最终输出候选策略排名，供人工校验、风险约束和后续真实标签迭代。

---

## 项目结构

```
project/
│
├── README.md                                 # 本文件
├── .env.example                              # DeepSeek API Key 模板
├── ML_INTEGRATION_GUIDE.md                   # Phase 1/2 接入说明（队友撰写）
│
├── 量化策略绩效-1.xlsx                       # 原始数据：12 个策略交易记录
├── 量化策略绩效-2.xlsx                       # 原始数据：22 个策略交易记录
├── 带ETF的策略1.csv                          # 原始数据：国证2000ETF增强（GZ2000）
├── 带ETF策略2.csv                            # 原始数据：创业板300ETF增强（CYB300）
├── 朝花夕拾策略.csv                          # 原始数据：朝花夕拾策略（HX）
├── 模拟账户A的记录.xlsx                      # 原始数据：模拟账户 A 交易记录
├── 模拟账户B的记录.xlsx                      # 原始数据：模拟账户 B 交易记录
├── 模拟账户C的记录.xlsx                      # 原始数据：模拟账户 C 交易记录
│
│── step1_data_loader.py                      # Step 1: 数据加载与清洗
│── step2_industry_mapping.py                 # Step 2: 股票→行业映射
│── step3_feature_extraction.py               # Step 3: 10 维特征提取
│── step4_word2vec_pretrain.py                # Step 4: 多维Token构建 + Word2Vec
│── step5_simulate_data.py                    # Step 5: 模拟数据生成
│── step6_lstm_contrastive.py                 # Step 6: LSTM编码器 + 对比学习
│── step7_evaluation.py                       # Step 7: 匹配评估 + 归因分析
│
│── clean_strategies.csv                      # [产出 Step1] 清洗后策略交易记录
│── clean_accounts.csv                        # [产出 Step1] 清洗后账户交易记录
│── stock_industry_mapping.csv                # [产出 Step2] 股票代码→申万行业映射
│── stock_industry_mapping_review.csv         # [人工校验] review_industry 覆盖原 industry
│── stock_industry_mapping_priority_review.csv# [人工校验] 优先级审核版
│── strategy_features.csv/json                # [产出 Step3] 37 策略特征向量
│── account_features.csv/json                 # [产出 Step3] 3 账户特征向量
│
│── token_vocab.json                          # [产出 Step4] Token→ID 映射 (~19K tokens)
│── word2vec_embeddings.npy                   # [产出 Step4] vocab×64 词向量矩阵
│── tokenized_sequences.pkl                   # [产出 Step4] Token ID 序列
│── token_sequences.csv                       # [产出 Step4] 可读版序列
│── word2vec_model.pt                         # [产出 Step4] PyTorch 模型权重
│
│── simulated_strategies_features.csv         # [产出 Step5] 2000 模拟策略特征
│── simulated_accounts_features.csv           # [产出 Step5] 1000 模拟客户特征
│── simulated_data.pkl                        # [产出 Step5] 完整模拟数据
│── train_pairs.csv                           # [产出 Step5] 6000 条训练标签
│── simulated_sequences.csv                   # [产出 Step5] 模拟序列
│
│── models/lstm_encoder.pt                    # [产出 Step6] 训练好的 LSTM 编码器
│── strategy_embeddings.npy                   # [产出 Step6] 37×128 策略向量
│── account_embeddings.npy                    # [产出 Step6] 3×128 账户向量
│── embedding_meta.json                       # [产出 Step6] 向量名映射
│── training_history.csv                      # [产出 Step6] 训练 loss/acc 日志
│
│── similarity_matrix.csv                     # [产出 Step6] 3×37 匹配相似度矩阵
│── matching_phase1_features.csv              # [产出 Step7] Phase 1 特征匹配矩阵
│── matching_phase2_lstm.csv                  # [产出 Step7] Phase 2 LSTM 匹配矩阵
│── final_recommendations.csv                 # [产出 Step7] 综合推荐 Top-5
│── shap_analysis.json                        # [产出 Step7] SHAP 特征归因结果
│
│── _step1_result.txt                         # Step 1 运行摘要
│── _step2_final.txt                          # Step 2 运行摘要
│── _step2_test_noapi.py                      # Step 2 纯规则测试版
│
└── models/                                   # 模型保存目录
```

---

## 七步流程

### Step 1 — 数据加载与清洗

**做什么**：合并 `量化策略绩效-1.xlsx`（12 个策略）、`量化策略绩效-2.xlsx`（22 个策略）及 3 个 CSV 策略文件（国证2000ETF增强、创业板300ETF增强、朝花夕拾策略），共 37 个策略；加载 3 个模拟账户 A/B/C 交易记录。统一列名、处理缺失值、过滤无效记录。

**关键处理**：
- 列名映射统一（`datetime`, `stock_code`, `action`, `volume`, `price`, `amount`）
- CSV 策略文件：`product_id`→`strategy_name`, `symbol`→`stock_code`, `side`→`action`, `qty`→`volume`, `trade_time`→`datetime`
- 排除无交易记录的策略（策略ETF, 策略etf2, 全球etf增强, 百亿etf等）
- 过滤非主动买卖事件（配售股份、中签下账、新股入账）
- 股票代码标准化（去前缀 SHSE/SZSE，补零至 6 位）
- 金额列类型转换（混合字符串/浮点数 → 统一 float）
- 排除可转债/债券/B股（代码 12xxxx, 11xxxx, 10xxxx, 9xxxxx 等）

**产出文件**：
| 文件 | 内容 | 行数 |
|------|------|------|
| `clean_strategies.csv` | 清洗后 37 策略交易记录 | ~33K |
| `clean_accounts.csv` | 清洗后 3 账户交易记录 | ~2.1K |

**脚本**：`step1_data_loader.py`

---

### Step 2 — 股票代码 → 申万一级行业映射

**做什么**：将 ~2,850 只去重股票映射到 31 个申万一级行业，为 Step 3 行业偏好特征提供基础。

**五轮分类策略**：

| 轮次 | 方法 | 匹配数 | 说明 |
|------|------|--------|------|
| 第一轮 | 关键词规则 | ~600 | 正则匹配股票名称（如"煤业"→煤炭、"半导体"→电子） |
| 第二轮 | 策略名推断 | ~300 | 从策略名推断行业（如"军工etf增强"→国防军工） |
| 第三轮 | ETF代码段 | — | ETF代码前缀匹配（15xxxx/51xxxx 等→综合） |
| 第四轮 | DeepSeek API（未匹配） | ~99 | 批量查询规则未覆盖股票 |
| 第五轮 | DeepSeek API（综合类） | ~1,800 | 对"综合"类宽基策略个股细分 |

**最终覆盖**：~2,850 只股票 100% 映射，0 只遗漏。

**行业分布 Top 5**：电子(377)、医药生物(234)、基础化工(173)、机械设备(171)、计算机(167)

**人工校验**：产出 `stock_industry_mapping_review.csv`，可通过 `review_industry` 列覆盖自动映射结果。

**产出文件**：
| 文件 | 内容 | 列 |
|------|------|-----|
| `stock_industry_mapping.csv` | 股票→行业映射 | `stock_code`, `stock_name`, `industry`, `source` |

**脚本**：`step2_industry_mapping.py`（需 DeepSeek API）、`_step2_test_noapi.py`（纯规则测试版）

---

### Step 3 — 扩展交易风格特征提取（10 维数值特征）

**做什么**：把每个策略和账户的交易行为抽象为行业偏好 + 9 个可量化数值特征，构建固定维度的交易画像向量，作为弱监督匹配的主干。

**特征集合**：

| 特征 | 维度 | 捕捉什么 | 计算方式 |
|------|------|---------|---------|
| F1 行业偏好 | 31 | 钱投向哪些行业 | 买入金额按行业分布 + Laplace 平滑 |
| F2 年化换手率 | 1 | 交易频率 | 总买入 / 平均持仓市值 / 时间跨度(年) |
| F3 持仓周期 | 1 | 买入到卖出多久 | FIFO 匹配 → 成交量加权平均持有天数 |
| F4 持股集中度 | 1 | 分散还是集中 | 时序仓位 HHI 指数均值 |
| F5 买卖对称性 | 1 | 建仓还是清仓 | 总买入 / (总买入+总卖出)，0.5=平衡 |
| F6 波动偏好 | 1 | 喜欢高波动还是低波动 | 各股价格变异系数 CoV 加权平均 |
| F7 实现收益偏好 | 1 | 交易是否偏向盈利/亏损兑现 | FIFO 匹配卖出，按成交金额加权收益 |
| F8 回撤承受风格 | 1 | 持仓曲线的回撤暴露 | 用成交价近似持仓市值曲线，计算最大回撤 |
| F9 市场状态暴露 | 1 | 偏好顺势/震荡/逆势交易 | 个体交易股票等权 5 日价格变化状态，加权平均 |
| F10 交易间隔 | 1 | 交易节奏疏密 | 相邻交易日期平均间隔 |

**最终向量**：每个策略/账户 → **40 维向量**（31 行业概率 + 9 数值特征）

**三个账户画像**：

| 特征 | 账户 A | 账户 B | 账户 C |
|------|--------|--------|--------|
| 换手率 | 5.9 | 89.2 | 36.9 |
| 持仓周期 | 56.8 天 | 3.1 天 | 7.3 天 |
| 集中度 HHI | 0.47 | 0.29 | 0.38 |
| 买卖对称 | 0.54（净买） | 0.50（平衡） | 0.50（平衡） |
| 波动偏好 | 0.07 | 0.13 | 0.04 |
| 实现收益偏好 | 0.016 | 0.002 | ~0 |
| 最大回撤 | 0.69 | 0.95 | ~0 |
| 市场状态暴露 | 0.31 | 0.23 | ~0 |
| 交易间隔 | 5.0 天 | 1.6 天 | ~0 天 |

**产出文件**：
| 文件 | 内容 |
|------|------|
| `strategy_features.csv` | 37 策略特征 |
| `strategy_features.json` | 同上 JSON 格式 |
| `account_features.csv` | 3 账户特征 |
| `account_features.json` | 同上 JSON 格式 |

**脚本**：`step3_feature_extraction.py`

---

### Step 4 — 多维交易 Token + Word2Vec 预训练

**状态**：✅ 已完成

**做什么**：
- 每条交易记录 → `{行业}_{BUY/SELL}_A{金额}_H{持仓}_T{换手}_R{收益}_D{回撤}_M{市场状态}` token
- 金额按实体内三分位数分桶；持仓周期、换手、实现收益、运行回撤、市场状态分别离散化为风格桶
- 这样 token 不再只表达“哪个行业买卖多少钱”，还包含交易节奏、盈利/亏损兑现、回撤和市场环境
- PyTorch 从零实现 Skip-gram + 负采样 Word2Vec（非 gensim，避免 Windows C++ 编译器依赖）
- 64 维词向量，窗口=5，负采样=5，30 epochs，Adam lr=0.002

**语义验证**：训练后脚本会从当前词表中抽取样例 token，输出 Top-N 余弦相似 token。
由于 token 现在包含多维风格桶，词表大小会随人工校验行业和交易风格分布变化。

**产出文件**：
| 文件 | 内容 |
|------|------|
| `token_vocab.json` | Token→ID 映射（多维 token，词表大小动态生成） |
| `word2vec_embeddings.npy` | vocab_size×64 词向量矩阵 |
| `tokenized_sequences.pkl` | 各策略/账户的 token ID 序列 |
| `token_sequences.csv` | 可读版序列 |
| `word2vec_model.pt` | PyTorch 模型权重 |

**脚本**：`step4_word2vec_pretrain.py`

---

### Step 5 — 模拟数据生成

**状态**：✅ 已完成

**做什么**：
- 基于真实策略 + 3 账户的扩展特征分布，扰动生成模拟数据用于对比学习训练
- **特征扰动**：数值特征用对数正态噪声（σ=0.12~0.15），行业偏好用 Dirichlet 噪声
- **序列生成**：Block bootstrap 从真实序列采样 token 块（3~12 tokens），按行业偏好加权拼接，并微调 BUY/SELL 比例
- **弱监督伪标签**：用扩展交易画像的余弦相似度生成正负样本；top-1 作为正样本，困难负样本和易负样本混合采样。该标签用于训练辅助排序器，不代表真实客户适配标签。

**生成规模**：

| 实体 | 数量 | 序列长度 (mean/min/max) |
|------|------|------------------------|
| 模拟策略 | 2000 | 1,344 / 542 / 2,867 |
| 模拟客户 | 1000 | 1,105 / 407 / 1,610 |
| 正样本平均相似度 | — | 0.894 |

**产出文件**：
| 文件 | 内容 |
|------|------|
| `simulated_strategies_features.csv` | 2000 模拟策略扩展特征 |
| `simulated_accounts_features.csv` | 1000 模拟客户扩展特征 |
| `simulated_data.pkl` | 完整模拟数据 (特征 + 序列 + 匹配标签) |
| `train_pairs.csv` | 弱监督伪标签 (client_idx, strategy_idx, is_match) |
| `simulated_sequences.csv` | 模拟 token 序列 |

**脚本**：`step5_simulate_data.py`

---

### Step 6 — LSTM 编码器 + 对比学习训练

**状态**：✅ 已完成

**做什么**：
- 构建序列编码器：`Word2Vec Embedding(vocab_size×64) → BiLSTM(2层, hidden=128) → Mean Pooling → Linear(256→128) → L2 归一化`
- 参数量随词表大小变化，Word2Vec 预训练权重初始化 Embedding 层
- 对比学习 Triplet Loss：`max(0, d(anchor, pos) - d(anchor, neg) + 0.2)`
- 训练时随机截取 512-token 子序列（数据增强）
- 200 epochs, batch=128, Adam lr=0.0005, CosineAnnealingWarmRestarts 调度
- 训练集/验证集按模拟客户 80/20 划分，验证集固定负样本消除随机性

**训练结果**：最佳 val_acc=0.82 (Epoch 13), val_loss=0.086, 共训练 43 epochs

**产出文件**：
| 文件 | 内容 |
|------|------|
| `models/lstm_encoder.pt` | 训练好的编码器 (含配置+训练历史) |
| `strategy_embeddings.npy` | 37×128 真实策略向量 |
| `account_embeddings.npy` | 3×128 真实账户向量 |
| `similarity_matrix.csv` | 3×37 余弦相似度矩阵 |
| `training_history.csv` | 43 轮训练 loss/acc |

**脚本**：`step6_lstm_contrastive.py`

---

### Step 7 — 匹配评估 + 归因分析

**状态**：✅ 已完成

**做什么**：
- **Phase 1 Baseline**：40 维扩展特征余弦相似度匹配，作为对照基线
- **Phase 2 深度学习**：LSTM 128 维向量余弦相似度匹配
- **两阶段对比**：Spearman 秩相关、Top-K 重叠率、排名变化分析
- **SHAP 归因**：在弱监督伪标签上训练 XGBoost，用 SHAP TreeExplainer 解释“规则生成标签”的特征贡献
- **结论边界**：当前评估用于检验方法链路和候选排序稳定性，不能替代真实客户反馈、收益回撤表现或适当性标签验证

**关键发现**：

| 指标 | Phase 1 (特征工程) | Phase 2 (LSTM) |
|------|-------------------|----------------|
| 相似度范围 | -0.41 ~ 0.53 | -0.25 ~ 1.00 |
| 相似度标准差 | 0.20 | 0.34 |
| Spearman ρ (A/B/C) | — | 0.49 / 0.60 / 0.40 |

Phase 2 区分度约为 Phase 1 的 1.7 倍（按标准差），两阶段排名中等相关（ρ≈0.4~0.6），说明 LSTM 学习了互补的序列风格信号。

**SHAP 解释口径**：SHAP 解释的是弱监督伪标签的生成逻辑。扩展后重点观察持仓周期、换手率、实现收益、最大回撤、市场状态、集中度和行业偏好的相对贡献。

**综合推荐** (Phase 1 + Phase 2 平均排名)：

| 排名 | Account A | Account B | Account C |
|------|-----------|-----------|-----------|
| 1 | 煤炭周期优选动态轮动 | 动量趋势策略 | 综合拆分1 |
| 2 | 成长红利量化选股 | 综合全 | 综合全 |
| 3 | 杠铃 | 综合拆分1 | 综合拆分2 |
| 4 | 旅游etf增强 | 行业etf增强 | 行业etf增强 |
| 5 | 国企etf增强 | 朝花夕拾策略 | 朝花夕拾策略 |

**产出文件**：
| 文件 | 内容 |
|------|------|
| `matching_phase1_features.csv` | Phase 1 特征匹配矩阵 |
| `matching_phase2_lstm.csv` | Phase 2 LSTM 匹配矩阵 |
| `final_recommendations.csv` | 综合推荐 Top-5 × 3 账户 |
| `shap_analysis.json` | SHAP 特征归因结果 |

**脚本**：`step7_evaluation.py`

---

## 方法边界与可行路线

- 当前数据只有 3 个模拟客户账户和少量策略，因此模型目标是候选排序，不是直接证明真实投资适配。
- 行业映射优先读取 `stock_industry_mapping_review.csv`；如果存在 `review_industry` 且非空，则覆盖原始 `industry`。股票名为空但股票代码存在时保留，不作为无效记录。
- 深度学习部分使用弱监督伪标签训练，核心价值是学习交易序列风格相似性；最终推荐仍需结合可解释特征、风险等级、收益回撤约束和人工校验。
- 后续若获得真实客户选择、满意度、持有后收益/回撤或人工专家标注，可直接替换伪标签，升级为监督学习排序。

## 技术栈

- **数据**：pandas, numpy
- **API**：DeepSeek API (OpenAI SDK, model: deepseek-v4pro)
- **深度学习**：PyTorch (Word2Vec Skip-gram, BiLSTM Encoder)
- **评估**：scikit-learn, SciPy, SHAP, XGBoost
- **语言**：Python 3.x
- **GPU 支持**：代码自动检测 `cuda` 设备，可部署至 GPU 服务器重新训练

---

## 运行方式

```bash
# Step 1-3: 数据准备
conda run -n stock python step1_data_loader.py
conda run -n stock python step2_industry_mapping.py    # 需要 DeepSeek API key
conda run -n stock python step3_feature_extraction.py

# Step 4-7: 模型训练与评估
conda run -n stock python step4_word2vec_pretrain.py
conda run -n stock python step5_simulate_data.py
conda run -n stock python step6_lstm_contrastive.py
conda run -n stock python step7_evaluation.py
```

## Phase 说明

| Phase | 方法 | 负责人 | 状态 |
|-------|------|--------|------|
| Phase 1 | 传统统计匹配（特征工程 + 聚类 + 多度量） | 队友 | 进行中 |
| Phase 2 | 深度学习匹配（Word2Vec + BiLSTM + 对比学习） | 本项目 | ✅ 已完成 |
