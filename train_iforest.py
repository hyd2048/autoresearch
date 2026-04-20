"""
Isolation Forest for NASA Bearing Anomaly Detection — train_iforest.py
使用孤立森林（Isolation Forest）进行异常检测。

核心思路：
  - 孤立森林通过随机划分特征空间来"孤立"样本
  - 异常样本因分布稀疏，平均被孤立所需步数更少（anomaly_score 更高）
  - sklearn 输出 decision_function 分数（越小越异常），取负值作为异常分数

Usage:
    python train_iforest.py                    # Run on default 2nd_test
    python train_iforest.py --dataset 1st      # Run on 1st_test
    python train_iforest.py --dataset 3rd      # Run on 3rd_test
"""

import os
import argparse
import time
import numpy as np
from sklearn.ensemble import IsolationForest

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
# 超参数
# ===========================================================================

# 树的数量：越多越稳定，但训练更慢
N_ESTIMATORS = 200

# 每棵树的子样本数（int 或 float）
# - float (0~1]：训练集的比例，如 0.8
# - int: 固定数量，如 256
# sklearn 默认 256；大数据集可降低以加速
MAX_SAMPLES = 'auto'   # 'auto' = min(256, n_train)

# 特征子采样比例（每次分裂时使用的特征数）
# - 'auto' = 1.0（使用全部特征）
# - float (0~1]：随机选择比例
MAX_FEATURES = 1.0

# 预期异常比例（仅影响 predict() 的默认阈值，不影响 decision_function 分数）
# 可设为 'auto'（sklearn 默认 0.1）或具体浮点数
CONTAMINATION = 'auto'

# 随机种子
RANDOM_STATE = 42

# ---------- 阈值选择策略 ----------
# "train_percentile": 用训练集异常分数的高百分位数，无数据泄露（推荐）
# "val_search"      : 在测试集上网格搜索 F1 最优阈值（有数据泄露，仅作参考）
THRESHOLD_STRATEGY = "train_percentile"
TRAIN_PERCENTILE   = 97.0   # 仅当 THRESHOLD_STRATEGY == "train_percentile" 时生效

VAL_SPLIT_RATIO    = 0.15   # 从训练集切出验证集（观察用）

# ===========================================================================
# 主流程
# ===========================================================================

def main(dataset: str = DEFAULT_DATASET):
    t_start = time.time()
    np.random.seed(RANDOM_STATE)

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

    input_dim = X_train_np.shape[1]
    print(f"INPUT_DIM: {input_dim}（自动推断）")
    print(f"训练样本: {len(X_train_np)} | 测试样本: {len(X_test_np)}")
    print(f"时间预算: {TIME_BUDGET}s")

    # 验证集切分（观察用）
    n_val   = int(len(X_train_np) * VAL_SPLIT_RATIO)
    n_train = len(X_train_np) - n_val
    X_tr    = X_train_np[:n_train]
    X_val   = X_train_np[n_train:]

    # ===========================================================================
    # 训练孤立森林
    # ===========================================================================

    t_train = time.time()
    print(f"\n训练 Isolation Forest (n_estimators={N_ESTIMATORS}, "
          f"max_samples={MAX_SAMPLES}, max_features={MAX_FEATURES})...")

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        max_features=MAX_FEATURES,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,       # 使用所有 CPU 核心
        verbose=0,
    )
    model.fit(X_tr)

    t_fit = time.time() - t_train
    print(f"拟合耗时: {t_fit:.3f}s")

    # ===========================================================================
    # 推理（异常分数 = -decision_function，越大越异常）
    # ===========================================================================

    # sklearn decision_function: 越大越正常 → 取负值使"越大越异常"
    train_scores = -model.decision_function(X_tr)
    val_scores   = -model.decision_function(X_val)
    test_scores  = -model.decision_function(X_test_np)

    # 对齐长度
    min_len     = min(len(test_scores), len(y_test_np))
    test_scores = test_scores[:min_len]
    y_true      = y_test_np[:min_len]

    print(f"\n训练集异常分数 — mean: {train_scores.mean():.6f}  std: {train_scores.std():.6f}  "
          f"p97: {np.percentile(train_scores, 97):.6f}")
    print(f"验证集异常分数 — mean: {val_scores.mean():.6f}  std: {val_scores.std():.6f}")
    print(f"测试集异常分数 — mean: {test_scores.mean():.6f}  std: {test_scores.std():.6f}")

    # ===========================================================================
    # 阈值选择
    # ===========================================================================

    import torch

    if THRESHOLD_STRATEGY == "train_percentile":
        threshold = float(np.percentile(train_scores, TRAIN_PERCENTILE))
        print(f"\n阈值策略: train_percentile ({TRAIN_PERCENTILE}th) → {threshold:.6f}")
    else:
        threshold, _ = find_best_threshold(
            torch.from_numpy(test_scores).float(),
            torch.from_numpy(y_true).float(),
        )
        print(f"\n阈值策略: val_search → {threshold:.6f}")

    # ===========================================================================
    # 最终评估
    # ===========================================================================

    pred    = (torch.from_numpy(test_scores).float() > threshold).float()
    metrics = calculate_metrics(
        torch.from_numpy(y_true).float(), pred
    )

    # ===========================================================================
    # 输出摘要
    # ===========================================================================

    t_end = time.time()

    print()
    print("---")
    print(f"val_f1:           {metrics['f1']:.6f}")
    print(f"val_precision:    {metrics['precision']:.6f}")
    print(f"val_recall:       {metrics['recall']:.6f}")
    print(f"val_accuracy:     {metrics['accuracy']:.6f}")
    print(f"training_seconds: {t_fit:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     0.0")
    print(f"total_steps:      {N_ESTIMATORS}")
    print(f"threshold:        {threshold:.6f}")
    print(f"dataset:          {dataset}")
    print(f"n_estimators:     {N_ESTIMATORS}")
    print(f"max_samples:      {MAX_SAMPLES}")

    return metrics


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NASA Bearing Anomaly Detection — Isolation Forest")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=['1st', '2nd', '3rd'],
                        help="选择数据集 (1st, 2nd, 3rd)")
    args = parser.parse_args()
    main(args.dataset)