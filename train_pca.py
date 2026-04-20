"""
PCA for NASA Bearing Anomaly Detection — train_pca.py
使用主成分分析（PCA）进行异常检测。

核心思路：
  - 用正常样本拟合 PCA，学习"正常子空间"
  - 推理时将样本投影到该子空间再重建
  - 重建误差（MAE）作为异常分数；正常样本误差低，异常样本误差高

Usage:
    python train_pca.py                    # Run on default 2nd_test
    python train_pca.py --dataset 1st      # Run on 1st_test
    python train_pca.py --dataset 3rd      # Run on 3rd_test
"""

import os
import argparse
import time
import numpy as np
from sklearn.decomposition import PCA

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

# PCA 保留的成分数（int）或解释方差比（0~1 的 float）
# - int：直接指定主成分数，如 2
# - float：自动选择使累计解释方差 >= 该值的最少成分数，如 0.95
N_COMPONENTS = 0.95

# ---------- 阈值选择策略 ----------
# "train_percentile": 用训练集重建误差的高百分位数，无数据泄露（推荐）
# "val_search"      : 在测试集上网格搜索 F1 最优阈值（有数据泄露，仅作参考）
THRESHOLD_STRATEGY = "train_percentile"
TRAIN_PERCENTILE   = 97.0   # 仅当 THRESHOLD_STRATEGY == "train_percentile" 时生效

VAL_SPLIT_RATIO    = 0.15   # 从训练集中切出验证集（仅用于观察，PCA 本身不使用）

# ===========================================================================
# 工具函数
# ===========================================================================

def reconstruction_error(X: np.ndarray, pca: PCA) -> np.ndarray:
    """MAE 重建误差：X → 投影到主成分子空间 → 重建 → |X - X_hat| 均值。"""
    X_hat = pca.inverse_transform(pca.transform(X))
    return np.mean(np.abs(X - X_hat), axis=1)


# ===========================================================================
# 主流程
# ===========================================================================

def main(dataset: str = DEFAULT_DATASET):
    t_start = time.time()
    np.random.seed(42)

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
    # 训练 PCA
    # ===========================================================================

    t_train = time.time()
    print(f"\n训练 PCA (N_COMPONENTS={N_COMPONENTS})...")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    pca.fit(X_tr)

    n_comp = pca.n_components_
    explained = pca.explained_variance_ratio_.sum()
    print(f"实际使用主成分数: {n_comp} / {input_dim}")
    print(f"累计解释方差: {explained:.4f}")
    t_fit = time.time() - t_train
    print(f"拟合耗时: {t_fit:.3f}s")

    # ===========================================================================
    # 推理（重建误差）
    # ===========================================================================

    train_errors = reconstruction_error(X_tr, pca)
    val_errors   = reconstruction_error(X_val, pca)
    test_errors  = reconstruction_error(X_test_np, pca)

    # 对齐长度
    min_len     = min(len(test_errors), len(y_test_np))
    test_errors = test_errors[:min_len]
    y_true      = y_test_np[:min_len]

    print(f"\n训练集误差 — mean: {train_errors.mean():.6f}  std: {train_errors.std():.6f}  "
          f"p97: {np.percentile(train_errors, 97):.6f}")
    print(f"验证集误差 — mean: {val_errors.mean():.6f}  std: {val_errors.std():.6f}")
    print(f"测试集误差 — mean: {test_errors.mean():.6f}  std: {test_errors.std():.6f}")

    # ===========================================================================
    # 阈值选择
    # ===========================================================================

    if THRESHOLD_STRATEGY == "train_percentile":
        threshold = float(np.percentile(train_errors, TRAIN_PERCENTILE))
        print(f"\n阈值策略: train_percentile ({TRAIN_PERCENTILE}th) → {threshold:.6f}")
    else:
        threshold, _ = find_best_threshold(
            __import__('torch').from_numpy(test_errors).float(),
            __import__('torch').from_numpy(y_true).float(),
        )
        print(f"\n阈值策略: val_search → {threshold:.6f}")

    # ===========================================================================
    # 最终评估
    # ===========================================================================

    import torch
    pred    = (torch.from_numpy(test_errors).float() > threshold).float()
    metrics = calculate_metrics(
        torch.from_numpy(y_true).float(), pred
    )

    # ===========================================================================
    # 输出摘要
    # ===========================================================================

    t_end     = time.time()

    print()
    print("---")
    print(f"val_f1:           {metrics['f1']:.6f}")
    print(f"val_precision:    {metrics['precision']:.6f}")
    print(f"val_recall:       {metrics['recall']:.6f}")
    print(f"val_accuracy:     {metrics['accuracy']:.6f}")
    print(f"training_seconds: {t_fit:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     0.0")
    print(f"total_steps:      1")
    print(f"threshold:        {threshold:.6f}")
    print(f"dataset:          {dataset}")
    print(f"n_components:     {n_comp}")
    print(f"explained_var:    {explained:.4f}")

    return metrics


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NASA Bearing Anomaly Detection — PCA")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=['1st', '2nd', '3rd'],
                        help="选择数据集 (1st, 2nd, 3rd)")
    args = parser.parse_args()
    main(args.dataset)