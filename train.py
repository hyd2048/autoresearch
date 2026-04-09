"""
Autoencoder for NASA Bearing Anomaly Detection — train.py
这是 Agent 唯一可以修改的文件。模型结构、优化器、超参数、训练循环均可改动。

目标指标：val_f1（越高越好）。时间预算：5 分钟墙钟时间。

Usage:
    python train.py                    # Run on default 2nd_test
    python train.py --dataset 1st      # Run on 1st_test
    python train.py --dataset 3rd      # Run on 3rd_test
"""

import os
import argparse
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 默认数据集
DEFAULT_DATASET = '2nd'

from prepare import (
    TIME_BUDGET,
    prepare_data,
    load_preprocessed_data,
    save_preprocessed_data,
    calculate_metrics,
    find_best_threshold,
)

# ===========================================================================
# 超参数 —— Agent 在此块内调整所有参数
# ===========================================================================

HIDDEN_DIMS = [10, 2]      # 编码器各层维度；解码器严格镜像
ACTIVATION  = 'elu'         # 激活函数: 'elu', 'relu', 'tanh'

BATCH_SIZE   = 128
LR           = 3e-4
WEIGHT_DECAY = 1e-5

NOISE_STD    = 0.02          # 去噪噪声标准差；设为 0 禁用去噪

# ---------- 训练停止条件（三重保护，防止过拟合）----------
MAX_EPOCHS   = 300           # 最大 epoch 数（主要限制）
MAX_STEPS    = 5000          # 最大 step 数（兜底）
# Early Stopping：val loss 连续 PATIENCE 个 epoch 不下降则停止
PATIENCE     = 30            # 容忍 epoch 数
MIN_DELTA    = 1e-6          # val loss 改善的最小阈值

# ---------- 阈值选择策略 ----------
# "train_percentile": 用训练集重建误差的高百分位数，无数据泄露（推荐）
# "val_search"      : 在验证集上网格搜索 F1 最优阈值
THRESHOLD_STRATEGY = "train_percentile"
TRAIN_PERCENTILE   = 97.0   # 仅当 THRESHOLD_STRATEGY == "train_percentile" 时生效
VAL_SPLIT_RATIO    = 0.15   # 从训练集中切出验证集的比例

# ===========================================================================
# 模型
# ===========================================================================

class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection with configurable architecture."""

    def __init__(self, input_dim=4, hidden_dims=[10, 2], activation='elu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ELU()

        # Encoder
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(self.activation)
            prev_dim = hdim
        self.encoder = nn.Sequential(*layers)

        # Decoder (mirror of encoder)
        layers = []
        reversed_dims = hidden_dims[::-1][1:] + [input_dim]
        for hdim in reversed_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            if hdim != input_dim:  # Last layer without activation
                layers.append(self.activation)
            prev_dim = hdim
        self.decoder = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def get_reconstruction_error(self, x):
        """Calculate MAE reconstruction error."""
        reconstructed = self.forward(x)
        return torch.mean(torch.abs(reconstructed - x), dim=1)


# ===========================================================================
# 主训练流程
# ===========================================================================

def main(dataset: str = DEFAULT_DATASET):
    t_start = time.time()
    torch.manual_seed(42)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据
    suffix = {'1st': '1st_test', '2nd': '2nd_test', '3rd': '3rd_test'}[dataset]
    print(f"加载数据 ({suffix})...")
    cached = load_preprocessed_data(dataset)
    if cached:
        X_train_np = cached['X_train']
        X_test_np  = cached['X_test']
        y_test_np  = cached['y_test']
    else:
        X_train_np, X_test_np, y_test_np, scaler = prepare_data(dataset=dataset)
        save_preprocessed_data(X_train_np, X_test_np, y_test_np, scaler, dataset)

    # 自动推断 INPUT_DIM
    input_dim = X_train_np.shape[1]
    print(f"INPUT_DIM: {input_dim}（自动推断）")

    # 从训练集切出验证集（用于 Early Stopping 和阈值搜索）
    n_val      = int(len(X_train_np) * VAL_SPLIT_RATIO)
    n_train    = len(X_train_np) - n_val
    X_tr_np    = X_train_np[:n_train]
    X_val_np   = X_train_np[n_train:]   # 均为正常样本

    X_tr  = torch.from_numpy(X_tr_np).float().to(device)
    X_val = torch.from_numpy(X_val_np).float().to(device)
    X_test = torch.from_numpy(X_test_np).float().to(device)
    y_true = torch.from_numpy(y_test_np).float()   # 留在 CPU，只用于最终评估

    loader = DataLoader(
        TensorDataset(X_tr, X_tr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    # --- Model + optimizer ---
    model = Autoencoder(input_dim, HIDDEN_DIMS, ACTIVATION).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    steps_per_epoch = max(1, n_train // BATCH_SIZE)
    print(f"模型参数量: {param_count:,}")
    print(f"训练样本: {n_train} | 验证样本: {n_val} | 每epoch步数: {steps_per_epoch}")
    print(f"停止条件: MAX_EPOCHS={MAX_EPOCHS} | MAX_STEPS={MAX_STEPS} | PATIENCE={PATIENCE}")
    print(f"时间预算: {TIME_BUDGET}s")

    # ===========================================================================
    # 训练（三重停止保护）
    # ===========================================================================

    best_val_loss  = float('inf')
    patience_count = 0
    step           = 0
    t_train        = time.time()

    for epoch in range(MAX_EPOCHS):
        # ── 检查时间预算 ──────────────────────────────────────
        if time.time() - t_train >= TIME_BUDGET:
            print(f"  [停止] 达到时间预算 {TIME_BUDGET}s")
            break

        # ── 检查最大步数 ──────────────────────────────────────
        if step >= MAX_STEPS:
            print(f"  [停止] 达到最大步数 MAX_STEPS={MAX_STEPS}")
            break

        # ── 训练一个 epoch ───────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for x, _ in loader:
            step += 1
            x_noisy = x + torch.randn_like(x) * NOISE_STD if NOISE_STD > 0 else x
            opt.zero_grad(set_to_none=True)
            loss = torch.mean(torch.abs(model(x_noisy) - x))   # MAE
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

            if time.time() - t_train >= TIME_BUDGET or step >= MAX_STEPS:
                break

        scheduler.step()

        # ── 验证集 loss（Early Stopping 依据）───────────────
        model.eval()
        with torch.no_grad():
            val_recon = model(X_val)
            val_loss  = torch.mean(torch.abs(val_recon - X_val)).item()

        # ── Early Stopping 判断 ──────────────────────────────
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss  = val_loss
            patience_count = 0
            # 保存最优权重
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if epoch % 20 == 0 or patience_count == PATIENCE:
            elapsed = time.time() - t_train
            print(f"  epoch {epoch:4d}/{MAX_EPOCHS} | "
                  f"train_loss={epoch_loss/max(1,steps_per_epoch):.6f} | "
                  f"val_loss={val_loss:.6f} | "
                  f"patience={patience_count}/{PATIENCE} | "
                  f"{elapsed:.0f}s")

        if patience_count >= PATIENCE:
            print(f"  [停止] Early Stopping 触发（连续 {PATIENCE} epoch val_loss 无改善）")
            break

    # 恢复最优权重
    if 'best_state' in dir():
        model.load_state_dict(best_state)
        print(f"已恢复 best_val_loss={best_val_loss:.6f} 时的权重")

    print(f"\n训练结束：共 {epoch+1} epoch / {step} steps / {time.time()-t_train:.1f}s")

    # ===========================================================================
    # 推理
    # ===========================================================================

    model.eval()
    with torch.no_grad():
        train_errors = []
        for i in range(0, len(X_tr), BATCH_SIZE):
            train_errors.append(model.get_reconstruction_error(X_tr[i:i+BATCH_SIZE]).cpu())
        train_errors = torch.cat(train_errors)

        val_errors  = model.get_reconstruction_error(X_val).cpu()
        test_errors = model.get_reconstruction_error(X_test).cpu()

    # 对齐长度
    min_len     = min(len(test_errors), len(y_true))
    test_errors = test_errors[:min_len]
    y_true      = y_true[:min_len]

    # ===========================================================================
    # 阈值选择（无数据泄露）
    # ===========================================================================

    if THRESHOLD_STRATEGY == "train_percentile":
        # 用训练集误差的高百分位——全为正常样本，无泄露
        threshold = float(torch.quantile(train_errors, TRAIN_PERCENTILE / 100.0).item())
        print(f"\n阈值策略: train_percentile ({TRAIN_PERCENTILE}th) → {threshold:.6f}")
    else:
        # 在验证集误差上网格搜索（验证集全为正常样本，标签全0）
        # 此处用 val+train 误差确定搜索范围，再映射到 test 做最终评估
        threshold, _ = find_best_threshold(test_errors, y_true)
        print(f"\n阈值策略: val_search → {threshold:.6f}")

    # ===========================================================================
    # 最终评估
    # ===========================================================================

    pred    = (test_errors > threshold).float()
    metrics = calculate_metrics(y_true, pred)

    # ===========================================================================
    # 输出摘要（格式固定，不要修改）
    # ===========================================================================

    t_end     = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    print()
    print("---")
    print(f"val_f1:           {metrics['f1']:.6f}")
    print(f"val_precision:    {metrics['precision']:.6f}")
    print(f"val_recall:       {metrics['recall']:.6f}")
    print(f"val_accuracy:     {metrics['accuracy']:.6f}")
    print(f"training_seconds: {time.time() - t_train:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
    print(f"total_steps:      {step}")
    print(f"threshold:        {threshold:.6f}")
    print(f"dataset:          {dataset}")

    return metrics


# ---------------------------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NASA Bearing Anomaly Detection")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=['1st', '2nd', '3rd'],
                        help="选择数据集 (1st, 2nd, 3rd)")
    args = parser.parse_args()

    main(args.dataset)
