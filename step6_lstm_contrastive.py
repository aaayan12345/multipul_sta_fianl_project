"""
Step 6: LSTM 编码器 + 对比学习训练
6.1 构建 LSTM Encoder（Word2Vec Embedding + BiLSTM + Projection）
6.2 对比学习训练（Triplet Loss）
6.3 编码真实策略和账户
6.4 计算匹配相似度矩阵
"""
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

sys.stdout.reconfigure(encoding='utf-8')

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 6.1 加载数据
# ============================================================
print("=" * 60)
print("Step 6: LSTM 编码器 + 对比学习训练")
print("=" * 60)

# 加载模拟数据
with open('simulated_data.pkl', 'rb') as f:
    sim_data = pickle.load(f)

sim_strat_seqs = sim_data['strategies']['sequences']
sim_acct_seqs = sim_data['accounts']['sequences']
match_pairs = sim_data['match_pairs']  # [(client_idx, strategy_idx, is_match)]

# 加载真实序列
with open('tokenized_sequences.pkl', 'rb') as f:
    real_tokenized = pickle.load(f)

# 加载预训练 Word2Vec
word2vec_emb = np.load('word2vec_embeddings.npy')
VOCAB_SIZE, EMBED_DIM = word2vec_emb.shape

# 加载词表
with open('token_vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
PAD_TOKEN = '<PAD>'
if PAD_TOKEN not in vocab['token2id']:
    PAD_IDX = VOCAB_SIZE  # 用 vocab_size 作为 padding index
else:
    PAD_IDX = vocab['token2id'][PAD_TOKEN]

# 加载真实特征(用于最终匹配报告)
strategy_feats = pd.read_csv('strategy_features.csv')
account_feats = pd.read_csv('account_features.csv')

print(f"  词表大小: {VOCAB_SIZE}, Embedding维度: {EMBED_DIM}")
print(f"  模拟策略数: {len(sim_strat_seqs)}, 模拟客户数: {len(sim_acct_seqs)}")
print(f"  真实策略数: {len(strategy_feats)}, 真实账户数: {len(account_feats)}")

# ============================================================
# 6.2 准备训练数据
# ============================================================
print("\n--- 6.2 准备训练数据 ---")

# 组织匹配关系
pos_pairs_dict = {}  # client_idx -> strategy_idx
neg_pairs_dict = defaultdict(list)  # client_idx -> [strategy_idx, ...]
for ci, si, is_match in match_pairs:
    if is_match == 1:
        pos_pairs_dict[ci] = si
    else:
        neg_pairs_dict[ci].append(si)

N_CLIENTS = len(sim_acct_seqs)
N_STRATEGIES = len(sim_strat_seqs)
print(f"  正样本对: {len(pos_pairs_dict)}, 负样本对总数: {sum(len(v) for v in neg_pairs_dict.values())}")

# 划分训练/验证集 (80/20)
client_ids = list(range(N_CLIENTS))
np.random.shuffle(client_ids)
split = int(0.8 * N_CLIENTS)
train_clients = client_ids[:split]
val_clients = client_ids[split:]

print(f"  训练客户: {len(train_clients)}, 验证客户: {len(val_clients)}")

# 将序列转为 list of tensors
sim_strat_tensors = [torch.tensor(s, dtype=torch.long) for s in sim_strat_seqs.values()]
sim_acct_tensors = [torch.tensor(s, dtype=torch.long) for s in sim_acct_seqs.values()]

# ============================================================
# 6.3 定义模型
# ============================================================
print("\n--- 6.3 构建 LSTM Encoder ---")

LSTM_HIDDEN = 128
LSTM_LAYERS = 2
OUTPUT_DIM = 128
DROPOUT = 0.25
MAX_SEQ_LEN = 512  # 训练时截断长度（增大以覆盖更完整的交易节奏）


class LSTMEncoder(nn.Module):
    """Word2Vec Embedding + BiLSTM + Projection → 固定维度向量"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 num_layers=2, dropout=0.3, pretrained_emb=None, pad_idx=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim,
                                      padding_idx=pad_idx)
        if pretrained_emb is not None:
            # 用预训练权重初始化，padding token 随机初始化
            self.embedding.weight.data[:vocab_size] = torch.tensor(
                pretrained_emb, dtype=torch.float32)
            # 冻结 embedding（可选，这里先冻结再微调）
            # self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x, lengths):
        """
        x: (batch, max_seq_len) token ids
        lengths: (batch,) 每个序列的实际长度
        Returns: (batch, output_dim) L2归一化向量
        """
        # Embedding: (batch, seq_len, embed_dim)
        emb = self.embedding(x)

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # BiLSTM
        lstm_out, _ = self.lstm(packed)

        # Unpack: (batch, seq_len, hidden*2)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Mean pooling over time (only over valid positions)
        mask = (x != self.embedding.padding_idx).unsqueeze(-1).float()  # (batch, seq, 1)
        pooled = (unpacked * mask).sum(dim=1) / lengths.unsqueeze(-1).float().clamp(min=1)

        # Projection + L2 normalize
        out = self.projection(pooled)
        out = nn.functional.normalize(out, p=2, dim=1)

        return out


# 初始化模型
model = LSTMEncoder(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=LSTM_HIDDEN,
    output_dim=OUTPUT_DIM,
    num_layers=LSTM_LAYERS,
    dropout=DROPOUT,
    pretrained_emb=word2vec_emb,
    pad_idx=PAD_IDX,
)

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  总参数: {n_params:,}, 可训练: {n_trainable:,}")
print(f"  LSTM hidden={LSTM_HIDDEN}, layers={LSTM_LAYERS}, output={OUTPUT_DIM}")
print(f"  max_seq_len={MAX_SEQ_LEN}, dropout={DROPOUT}")


# ============================================================
# 6.4 训练准备
# ============================================================
print("\n--- 6.4 训练准备 ---")

MARGIN = 0.4          # 增大 margin 防止 embedding collapse
BATCH_SIZE = 128       # GPU 12G 可以吃更大 batch
EPOCHS = 200           # 足够长的训练周期
LEARNING_RATE = 0.0005 # 更低初始 lr，配合长周期
WEIGHT_DECAY = 1e-4    # 更强的 L2 正则化
HARD_NEG_REFRESH = 5   # 每 N epoch 刷新困难负样本
early_stop_patience = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"  设备: {device}")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                       weight_decay=WEIGHT_DECAY)
# ReduceLROnPlateau: val_loss 不降时自动衰减 lr，避免 WarmRestarts 剧烈扰动
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

criterion = nn.TripletMarginLoss(margin=MARGIN, p=2.0)

# 创建时间戳运行目录，避免覆盖历史产出
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = f'models/runs/{RUN_TIMESTAMP}'
os.makedirs(RUN_DIR, exist_ok=True)
print(f"  运行目录: {RUN_DIR}")

# 保存训练参数（训练前快照，防止训练中断丢失配置）
os.makedirs('models', exist_ok=True)
pre_config = {
    'model': {
        'architecture': 'Word2Vec_Embedding + BiLSTM + MeanPooling + Projection',
        'vocab_size': VOCAB_SIZE,
        'embed_dim': EMBED_DIM,
        'lstm_hidden': LSTM_HIDDEN,
        'lstm_layers': LSTM_LAYERS,
        'output_dim': OUTPUT_DIM,
        'dropout': DROPOUT,
        'max_seq_len': MAX_SEQ_LEN,
        'pad_idx': PAD_IDX,
        'total_params': n_params,
        'trainable_params': n_trainable,
    },
    'training': {
        'loss': 'TripletMarginLoss',
        'margin': MARGIN,
        'batch_size': BATCH_SIZE,
        'max_epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'optimizer': 'Adam',
        'grad_clip_norm': 1.0,
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_mode': 'min',
        'scheduler_factor': 0.5,
        'scheduler_patience': 10,
        'scheduler_min_lr': 1e-6,
        'early_stop_patience': early_stop_patience,
        'hard_neg_refresh': HARD_NEG_REFRESH,
        'hard_neg_pool_size': 20,
        'hard_neg_ratio': 0.8,
        'device': str(device),
    },
    'data': {
        'sim_strategies': N_STRATEGIES,
        'sim_clients': N_CLIENTS,
        'train_clients': len(train_clients),
        'val_clients': len(val_clients),
    },
}
with open('models/training_config.json', 'w', encoding='utf-8') as f:
    json.dump(pre_config, f, ensure_ascii=False, indent=2)
with open(f'{RUN_DIR}/training_config.json', 'w', encoding='utf-8') as f:
    json.dump(pre_config, f, ensure_ascii=False, indent=2)


def prepare_batch(client_indices, max_len=MAX_SEQ_LEN, training=True):
    """
    为一个 batch 的客户准备 (anchor, positive, negative) 三元组
    training=True: 随机截取子序列（数据增强）
    training=False: 取前 max_len 个 token
    """
    anchor_seqs = []
    pos_seqs = []
    neg_seqs = []

    for ci in client_indices:
        a_seq = sim_acct_tensors[ci]
        p_seq = sim_strat_tensors[pos_pairs_dict[ci]]

        # 负样本: 优先用困难负样本池，否则随机采样 (80% hard / 20% random)
        if ci in hard_neg_pool and np.random.random() < 0.8:
            n_idx = np.random.choice(hard_neg_pool[ci])
            n_seq = sim_strat_tensors[n_idx]
        else:
            neg_pool = list(range(N_STRATEGIES))
            neg_pool.remove(pos_pairs_dict[ci])
            n_idx = np.random.choice(neg_pool)
            n_seq = sim_strat_tensors[n_idx]

        if training:
            # 随机截取子序列
            for seq in [a_seq, p_seq, n_seq]:
                if len(seq) > max_len:
                    start = np.random.randint(0, len(seq) - max_len)
                    # 需要截取
                    if seq is a_seq:
                        a_seq = seq[start:start + max_len]
                    elif seq is p_seq:
                        p_seq = seq[start:start + max_len]
                    else:
                        n_seq = seq[start:start + max_len]

        anchor_seqs.append(a_seq[:max_len])
        pos_seqs.append(p_seq[:max_len])
        neg_seqs.append(n_seq[:max_len])

    # 记录长度
    a_lengths = torch.tensor([len(s) for s in anchor_seqs], dtype=torch.long)
    p_lengths = torch.tensor([len(s) for s in pos_seqs], dtype=torch.long)
    n_lengths = torch.tensor([len(s) for s in neg_seqs], dtype=torch.long)

    # Pad
    anchor_pad = pad_sequence(anchor_seqs, batch_first=True,
                              padding_value=PAD_IDX).to(device)
    pos_pad = pad_sequence(pos_seqs, batch_first=True,
                           padding_value=PAD_IDX).to(device)
    neg_pad = pad_sequence(neg_seqs, batch_first=True,
                           padding_value=PAD_IDX).to(device)

    return (anchor_pad, a_lengths.to(device),
            pos_pad, p_lengths.to(device),
            neg_pad, n_lengths.to(device))


# 困难负样本池（每 HARD_NEG_REFRESH epoch 刷新一次）
hard_neg_pool = {}  # client_idx -> [strategy_idx, ...]


@torch.no_grad()
def refresh_hard_negatives(client_indices, max_len=MAX_SEQ_LEN):
    """用当前 encoder 编码所有 client 和 strategy，为每个 client 选 hardest negatives"""
    model.eval()

    # 编码所有策略（只取前 max_len 个 token）
    all_strat_vecs = []
    for si in range(N_STRATEGIES):
        seq = sim_strat_tensors[si][:max_len].unsqueeze(0).to(device)
        length = torch.tensor([min(len(sim_strat_tensors[si]), max_len)], device=device)
        all_strat_vecs.append(model(seq, length))
    all_strat_vecs = torch.cat(all_strat_vecs, dim=0)  # (N_STRATEGIES, 128)
    all_strat_vecs = nn.functional.normalize(all_strat_vecs, p=2, dim=1)

    # 编码所有需要刷新的 client
    client_vecs = []
    for ci in client_indices:
        seq = sim_acct_tensors[ci][:max_len].unsqueeze(0).to(device)
        length = torch.tensor([min(len(sim_acct_tensors[ci]), max_len)], device=device)
        client_vecs.append(model(seq, length))
    client_vecs = torch.cat(client_vecs, dim=0)
    client_vecs = nn.functional.normalize(client_vecs, p=2, dim=1)

    # 每个 client 选 top-20 hardest negatives（距离最近的负样本）
    sim_all = client_vecs @ all_strat_vecs.T  # (n_clients, N_STRATEGIES)

    for i, ci in enumerate(client_indices):
        pos_si = pos_pairs_dict[ci]
        # 排除正样本
        sim_all[i, pos_si] = -float('inf')
        # 选相似度最高（距离最近）的 20 个作为困难负样本
        _, hardest = torch.topk(sim_all[i], k=min(20, N_STRATEGIES - 1))
        hard_neg_pool[ci] = hardest.cpu().numpy().tolist()

    model.train()
    avg_pool_size = np.mean([len(v) for v in hard_neg_pool.values()])
    print(f"    困难负样本刷新: {len(hard_neg_pool)} clients, avg pool={avg_pool_size:.1f}")


def encode_sequence(seq_tensor, max_len=MAX_SEQ_LEN):
    """编码单条序列为向量（推理用），取多个窗口的均值"""
    model.eval()
    if len(seq_tensor) <= max_len:
        x = seq_tensor.unsqueeze(0).to(device)
        length = torch.tensor([len(seq_tensor)], device=device)
        with torch.no_grad():
            vec = model(x, length)
        return vec.squeeze(0).cpu().numpy()

    # 长序列：取 5 个均匀分布窗口的均值
    windows = []
    step = (len(seq_tensor) - max_len) // 4 if len(seq_tensor) > max_len else 0
    for i in range(5):
        start = i * step if step > 0 else 0
        start = min(start, len(seq_tensor) - max_len)
        chunk = seq_tensor[start:start + max_len]
        x = chunk.unsqueeze(0).to(device)
        length = torch.tensor([len(chunk)], device=device)
        with torch.no_grad():
            v = model(x, length)
        windows.append(v.squeeze(0))

    # 平均并重新归一化
    avg_vec = torch.stack(windows).mean(dim=0)
    avg_vec = nn.functional.normalize(avg_vec, p=2, dim=0)
    return avg_vec.cpu().numpy()


# ============================================================
# 6.5 训练准备 (续) — 固定验证集负样本
# ============================================================

# 为每个 val client 预选固定的负样本策略（消除 val_loss 随机波动）
val_neg_strategies = {}
for ci in val_clients:
    pos_si = pos_pairs_dict[ci]
    # 固定选 4 个负样本（2 个困难 + 2 个随机）用于整个训练过程的验证
    neg_pool = list(range(N_STRATEGIES))
    neg_pool.remove(pos_si)
    # 随机固定选 4 个
    val_neg_strategies[ci] = np.random.choice(neg_pool, size=4, replace=False)

print(f"  验证集每个客户固定 4 个负样本 (消除 val_loss 随机性)")

# ============================================================
# 6.5 训练循环
# ============================================================
print("\n--- 6.5 训练 (Triplet Loss) ---")
print(f"  Epochs={EPOCHS}, Batch={BATCH_SIZE}, Margin={MARGIN}, LR={LEARNING_RATE}")

train_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}

best_val_loss = float('inf')
best_val_acc = 0.0
best_state = None
best_epoch = 0
no_improve = 0


def prepare_val_batch(client_indices, max_len=MAX_SEQ_LEN):
    """验证用：使用固定负样本"""
    anchor_seqs, pos_seqs, neg_seqs = [], [], []

    for ci in client_indices:
        a_seq = sim_acct_tensors[ci]
        p_seq = sim_strat_tensors[pos_pairs_dict[ci]]
        # 使用固定的负样本
        n_idx = val_neg_strategies[ci][np.random.randint(4)]
        n_seq = sim_strat_tensors[n_idx]

        anchor_seqs.append(a_seq[:max_len])
        pos_seqs.append(p_seq[:max_len])
        neg_seqs.append(n_seq[:max_len])

    a_lengths = torch.tensor([len(s) for s in anchor_seqs], dtype=torch.long)
    p_lengths = torch.tensor([len(s) for s in pos_seqs], dtype=torch.long)
    n_lengths = torch.tensor([len(s) for s in neg_seqs], dtype=torch.long)

    anchor_pad = pad_sequence(anchor_seqs, batch_first=True, padding_value=PAD_IDX).to(device)
    pos_pad = pad_sequence(pos_seqs, batch_first=True, padding_value=PAD_IDX).to(device)
    neg_pad = pad_sequence(neg_seqs, batch_first=True, padding_value=PAD_IDX).to(device)

    return (anchor_pad, a_lengths.to(device),
            pos_pad, p_lengths.to(device),
            neg_pad, n_lengths.to(device))


for epoch in range(1, EPOCHS + 1):
    # ---- 训练 ----
    model.train()
    train_loss = 0.0
    n_batches = 0
    np.random.shuffle(train_clients)

    for b_start in range(0, len(train_clients), BATCH_SIZE):
        batch_clients = train_clients[b_start:b_start + BATCH_SIZE]

        a_x, a_l, p_x, p_l, n_x, n_l = prepare_batch(batch_clients, training=True)

        anchor_vec = model(a_x, a_l)
        pos_vec = model(p_x, p_l)
        neg_vec = model(n_x, n_l)

        loss = criterion(anchor_vec, pos_vec, neg_vec)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        n_batches += 1

    train_loss /= max(1, n_batches)

    # ---- 验证 ----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    n_val_batches = 0

    with torch.no_grad():
        for b_start in range(0, len(val_clients), BATCH_SIZE):
            batch_clients = val_clients[b_start:b_start + BATCH_SIZE]

            a_x, a_l, p_x, p_l, n_x, n_l = prepare_val_batch(batch_clients)

            anchor_vec = model(a_x, a_l)
            pos_vec = model(p_x, p_l)
            neg_vec = model(n_x, n_l)

            loss = criterion(anchor_vec, pos_vec, neg_vec)
            val_loss += loss.item()
            n_val_batches += 1

            d_pos = torch.norm(anchor_vec - pos_vec, p=2, dim=1)
            d_neg = torch.norm(anchor_vec - neg_vec, p=2, dim=1)
            val_correct += (d_pos < d_neg).sum().item()
            val_total += len(d_pos)

    val_loss /= max(1, n_val_batches)
    val_acc = val_correct / max(1, val_total)

    # ReduceLROnPlateau: 用 val_loss 驱动 lr 衰减
    scheduler.step(val_loss)

    # 定期刷新困难负样本
    if epoch % HARD_NEG_REFRESH == 0:
        refresh_hard_negatives(train_clients)

    train_history['epoch'].append(epoch)
    train_history['train_loss'].append(train_loss)
    train_history['val_loss'].append(val_loss)
    train_history['val_acc'].append(val_acc)

    # Early stopping + 保存最佳模型
    improved = False
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
        improved = True
    if val_acc > best_val_acc:
        best_val_acc = val_acc

    if improved:
        no_improve = 0
    else:
        no_improve += 1

    # 日志
    if epoch % 10 == 1 or epoch == 1 or epoch == EPOCHS or improved:
        marker = " *" if improved else ""
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.3f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}{marker}")

    # 每 40 epoch 保存 checkpoint → run_dir
    if epoch % 40 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, f'{RUN_DIR}/lstm_encoder_epoch{epoch}.pt')

    # Early stop
    if no_improve >= early_stop_patience:
        print(f"\n  Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
        break

print(f"\n  最佳模型: Epoch {best_epoch} | val_loss={best_val_loss:.4f} | best_val_acc={best_val_acc:.3f}")

# 加载最佳模型
model.load_state_dict(best_state)

# ============================================================
# 6.6 编码真实策略和账户
# ============================================================
print("\n--- 6.6 编码真实实体 ---")

# 真实策略
real_strategy_names = sorted(real_tokenized['strategies'].keys())
print(f"  编码 {len(real_strategy_names)} 个真实策略...")
strategy_embeddings = {}
for sname in real_strategy_names:
    seq = torch.tensor(real_tokenized['strategies'][sname], dtype=torch.long)
    vec = encode_sequence(seq)
    strategy_embeddings[sname] = vec
    print(f"    [{sname}]: seq_len={len(seq)} → vec[{len(vec)}]")

# 真实账户
real_account_names = sorted(real_tokenized['accounts'].keys())
print(f"  编码 {len(real_account_names)} 个真实账户...")
account_embeddings = {}
for aname in real_account_names:
    seq = torch.tensor(real_tokenized['accounts'][aname], dtype=torch.long)
    vec = encode_sequence(seq)
    account_embeddings[aname] = vec
    print(f"    [Account {aname}]: seq_len={len(seq)} → vec[{len(vec)}]")

# ============================================================
# 6.7 匹配矩阵
# ============================================================
print("\n--- 6.7 匹配相似度矩阵 ---")

strat_vecs = np.stack(list(strategy_embeddings.values()))  # (34, 128)
acct_vecs = np.stack(list(account_embeddings.values()))    # (3, 128)

# Cosine similarity
strat_norm = strat_vecs / (np.linalg.norm(strat_vecs, axis=1, keepdims=True) + 1e-10)
acct_norm = acct_vecs / (np.linalg.norm(acct_vecs, axis=1, keepdims=True) + 1e-10)
sim_matrix = acct_norm @ strat_norm.T  # (3, 34)

# 构建 DataFrame
sim_df = pd.DataFrame(
    sim_matrix,
    index=[f"Account_{a}" for a in real_account_names],
    columns=real_strategy_names
)

print("\n  Cosine Similarity Matrix (账户 × 策略):")
print(sim_df.to_string(float_format=lambda x: f"{x:.4f}"))

# 每个账户的 Top-5 匹配策略
print("\n  Top-5 匹配 (每个账户):")
for aname in real_account_names:
    row = sim_df.loc[f"Account_{aname}"]
    top5 = row.sort_values(ascending=False).head(5)
    print(f"\n  Account {aname}:")
    for i, (sname, score) in enumerate(top5.items()):
        print(f"    {i+1}. {sname:30s}  sim={score:.4f}")

# ============================================================
# 6.8 保存
# ============================================================
print(f"\n--- 6.8 保存 ---")
print(f"  运行目录: {RUN_DIR}")

os.makedirs('models', exist_ok=True)

model_config = {
    'vocab_size': VOCAB_SIZE,
    'embed_dim': EMBED_DIM,
    'hidden_dim': LSTM_HIDDEN,
    'output_dim': OUTPUT_DIM,
    'num_layers': LSTM_LAYERS,
    'dropout': DROPOUT,
    'max_seq_len': MAX_SEQ_LEN,
    'pad_idx': PAD_IDX,
}

# 模型 → models/ + run_dir
torch.save({
    'model_state_dict': best_state,
    'config': model_config,
    'train_history': train_history,
}, 'models/lstm_encoder.pt')
torch.save({
    'model_state_dict': best_state,
    'config': model_config,
    'train_history': train_history,
}, f'{RUN_DIR}/lstm_encoder.pt')

# Embedding 向量 → 根目录 + run_dir
np.save('strategy_embeddings.npy', strat_vecs)
np.save(f'{RUN_DIR}/strategy_embeddings.npy', strat_vecs)
np.save('account_embeddings.npy', acct_vecs)
np.save(f'{RUN_DIR}/account_embeddings.npy', acct_vecs)

# 匹配矩阵 → 根目录 + run_dir
sim_df.to_csv('similarity_matrix.csv', encoding='utf-8-sig')
sim_df.to_csv(f'{RUN_DIR}/similarity_matrix.csv', encoding='utf-8-sig')

# 向量名映射 → 根目录 + run_dir
embed_meta = {
    'strategy_names': real_strategy_names,
    'account_names': [f"Account_{a}" for a in real_account_names],
    'vector_dim': OUTPUT_DIM,
}
with open('embedding_meta.json', 'w', encoding='utf-8') as f:
    json.dump(embed_meta, f, ensure_ascii=False, indent=2)
with open(f'{RUN_DIR}/embedding_meta.json', 'w', encoding='utf-8') as f:
    json.dump(embed_meta, f, ensure_ascii=False, indent=2)

# 训练配置 + 结果摘要 → models/ + run_dir
training_config = {
    'run_timestamp': RUN_TIMESTAMP,
    'model': {
        'architecture': 'Word2Vec_Embedding + BiLSTM + MeanPooling + Projection',
        **model_config,
        'total_params': n_params,
        'trainable_params': n_trainable,
    },
    'training': {
        'loss': 'TripletMarginLoss',
        'margin': MARGIN,
        'batch_size': BATCH_SIZE,
        'epochs': len(train_history['epoch']),
        'max_epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'optimizer': 'Adam',
        'grad_clip_norm': 1.0,
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_mode': 'min',
        'scheduler_factor': 0.5,
        'scheduler_patience': 10,
        'scheduler_min_lr': 1e-6,
        'early_stop_patience': early_stop_patience,
        'hard_neg_refresh': HARD_NEG_REFRESH,
        'hard_neg_pool_size': 20,
        'hard_neg_ratio': 0.8,
        'device': str(device),
    },
    'data': {
        'sim_strategies': N_STRATEGIES,
        'sim_clients': N_CLIENTS,
        'train_clients': len(train_clients),
        'val_clients': len(val_clients),
        'train_pos_pairs': sum(1 for ci in train_clients if ci in pos_pairs_dict),
    },
    'results': {
        'best_epoch': best_epoch,
        'best_val_loss': round(best_val_loss, 6),
        'best_val_acc': round(best_val_acc, 4),
        'final_train_loss': round(train_history['train_loss'][-1], 6),
        'final_val_loss': round(train_history['val_loss'][-1], 6),
        'final_val_acc': round(train_history['val_acc'][-1], 4),
        'real_strategies_encoded': len(real_strategy_names),
        'real_accounts_encoded': len(real_account_names),
    },
}
with open('models/training_config.json', 'w', encoding='utf-8') as f:
    json.dump(training_config, f, ensure_ascii=False, indent=2)
with open(f'{RUN_DIR}/training_config.json', 'w', encoding='utf-8') as f:
    json.dump(training_config, f, ensure_ascii=False, indent=2)

# 训练历史 → 根目录 + models/ + run_dir
history_df = pd.DataFrame(train_history)
history_df.to_csv('training_history.csv', index=False)
history_df.to_csv('models/training_history.csv', index=False)
history_df.to_csv(f'{RUN_DIR}/training_history.csv', index=False)

print(f"  根目录输出 (step7 兼容):")
print(f"    strategy_embeddings.npy     — {strat_vecs.shape} 策略向量")
print(f"    account_embeddings.npy      — {acct_vecs.shape} 账户向量")
print(f"    similarity_matrix.csv       — 匹配相似度矩阵")
print(f"    embedding_meta.json         — 向量名映射")
print(f"    training_history.csv        — 训练历史 ({len(history_df)} epochs)")
print(f"  models/ 输出:")
print(f"    models/lstm_encoder.pt      — 训练好的 LSTM 编码器")
print(f"    models/training_config.json — 训练参数 + 结果摘要")
print(f"    models/training_history.csv — 训练历史")
print(f"  {RUN_DIR}/ (完整归档):")
print(f"    包含以上所有文件的时间戳副本")

print("\nDone. Step 6 完成.")
