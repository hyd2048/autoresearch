# autoresearch

NASA IMS Bearing 异常检测实验平台。基于自编码器，对轴承振动数据进行无监督异常检测。

## 项目简介

使用 NASA IMS Bearing Dataset，通过自编码器学习正常轴承的振动模式，利用重建误差检测异常。训练仅使用正常数据，测试时重建误差异常增大则判定为异常。

支持三个数据集：

| 数据集 | 传感器数 | 时间点 | 训练样本 | 测试样本 | 异常占比 |
|--------|---------|--------|---------|---------|---------|
| 1st_test | 8 | 2,156 | 1,113 | 1,043 | 69.1% |
| 2nd_test | 4 | 984 | 225 | 760 | 43.1% |
| 3rd_test | 4 | 6,324 | 2,323 | 4,001 | 60.4% |

## How it works

仓库核心文件：

- **`prepare.py`** — 数据加载、预处理、缓存管理、评估工具。从 NASA Bearing 原始文件提取均值绝对值特征，按时间切分训练/测试集，MinMaxScaler 归一化（仅训练集 fit，防泄露）。
- **`train.py`** — 自编码器模型、训练循环、阈值选择、评估。Agent 可修改此文件优化模型。
- **`program.md`** — Agent 指令文件。

训练时间预算 5 分钟，目标指标 **val_f1**（越高越好）。

## Quick start

**Requirements:** Python 3.10+, PyTorch, scikit-learn, pandas, numpy。

```bash
# 1. 安装依赖
uv sync

# 2. 准备数据（默认 2nd_test，首次需下载 NASA Bearing 数据到 dataset/）
python prepare.py

# 3. 训练
python train.py

# 4. 指定数据集
python train.py --dataset 1st   # 1st_test
python train.py --dataset 2nd   # 2nd_test（默认）
python train.py --dataset 3rd   # 3rd_test
```

## 数据准备

将 NASA IMS Bearing Dataset 解压到 `dataset/` 目录，结构如下：

```
dataset/
├── 1st_test/1st_test/    # 8通道振动数据
├── 2nd_test/2nd_test/    # 4通道振动数据
└── 3rd_test/4th_test/txt # 4通道振动数据
```

预处理支持缓存（`~/.cache/autoanomaly/`），可用 `--force` 强制重新处理：

```bash
python prepare.py --force              # 重新处理默认数据集
python prepare.py --dataset 1st --force  # 重新处理 1st_test
python prepare.py --features           # 启用特征工程（滚动std + 差分，INPUT_DIM=12）
```

## 模型

当前使用对称自编码器，支持可配置隐藏层维度和激活函数：

```python
# 超参数（train.py 顶部）
HIDDEN_DIMS = [10, 2]    # 编码器各层维度
ACTIVATION  = 'elu'      # 激活函数: 'elu', 'relu', 'tanh'
```

阈值策略：
- **`train_percentile`**（默认）：取训练集重建误差的 97 百分位，无数据泄露
- **`val_search`**：验证集网格搜索最优 F1 阈值

## Project structure

```
prepare.py      — 数据预处理 + 评估工具（不要修改）
train.py        — 模型、训练循环（Agent 修改此文件）
program.md      — Agent 指令
dataset/        — 原始数据（不纳入版本控制）
```

## Baseline 结果

| 数据集 | F1 | Precision | Recall | 训练时间 |
|--------|------|-----------|--------|---------|
| 1st_test | 0.764 | 0.983 | 0.626 | 5.8s |
| 2nd_test | 0.850 | 0.740 | 1.000 | 1.1s |
| 3rd_test | 0.754 | 0.605 | 1.000 | 7.5s |

## Design choices

- **INPUT_DIM 自动推断**：从数据维度自动获取，无需手动设置。
- **无数据泄露**：scaler 仅在训练集 fit；阈值选择基于训练集重建误差百分位。
- **三重停止保护**：时间预算 + 最大步数 + Early Stopping，防止过拟合。
- **去噪自编码器**：训练时添加高斯噪声（`NOISE_STD=0.02`），提升模型鲁棒性。

## License

MIT
