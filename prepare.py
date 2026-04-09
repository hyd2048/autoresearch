"""
Data preparation for NASA Bearing anomaly detection experiments.
Loads bearing data, preprocesses it, and provides dataloaders.

Usage:
    python prepare.py                  # Run data prep (default: 2nd_test)
    python prepare.py --dataset 1st    # Use 1st_test dataset
    python prepare.py --dataset 2nd    # Use 2nd_test dataset
    python prepare.py --dataset 3rd    # Use 3rd_test dataset
    python prepare.py --force          # Ignore cache, re-prepare from scratch
    python prepare.py --features       # Enable feature engineering (INPUT_DIM=12)
"""

import os
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch

# ---------------------------------------------------------------------------
# 数据集配置
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    '1st': {
        'dir': '1st_test/1st_test',
        'num_sensors': 8,  # 1st_test 有8个通道（每个轴承2个）
        'anomaly_start_date': '2003-11-19 00:00:00',  # 故障发生在11月19日后
        'train_end_time': '2003-11-15 23:59:59',      # 训练截止到11月15日
    },
    '2nd': {
        'dir': '2nd_test/2nd_test',
        'num_sensors': 4,
        'anomaly_start_date': '2004-02-17 00:00:00',
        'train_end_time': '2004-02-13 23:52:39',
    },
    '3rd': {
        'dir': '3rd_test/4th_test/txt',
        'num_sensors': 4,
        'anomaly_start_date': '2004-04-01 00:00:00',  # 故障发生在4月1日后
        'train_end_time': '2004-03-20 23:59:59',      # 训练截止到3月20日
    },
}

DEFAULT_DATASET = '2nd'

# ---------------------------------------------------------------------------
# 常量（固定，不要修改）
# ---------------------------------------------------------------------------

TIME_BUDGET  = 300   # 训练时间预算，秒（5分钟）

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

DATA_DIR      = os.path.join(os.path.dirname(__file__), "dataset")
CACHE_DIR     = os.path.join(os.path.expanduser("~"), ".cache", "autoanomaly")
CACHE_VERSION = 1   # 修改数据处理逻辑时请递增此值，旧缓存会自动失效

# 全局变量：当前使用的数据集
_current_dataset = DEFAULT_DATASET

# ---------------------------------------------------------------------------
# 异常标签
# ---------------------------------------------------------------------------

def get_anomaly_labels(merged_data: pd.DataFrame, dataset: str = None) -> np.ndarray:
    """
    根据领域知识生成异常标签。
    异常起始时间之后的样本标记为异常（1），之前为正常（0）。
    """
    if dataset is None:
        dataset = _current_dataset
    config = DATASET_CONFIGS[dataset]
    anomaly_start = pd.Timestamp(config['anomaly_start_date'])
    labels = (merged_data.index >= anomaly_start).astype(int)
    return labels.values if hasattr(labels, 'values') else np.array(labels)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_bearing_data(dataset: str = None) -> pd.DataFrame:
    """
    从指定数据集目录加载所有轴承数据文件。

    Args:
        dataset: 数据集名称 ('1st', '2nd', '3rd')

    每个文件对应一个时间点，包含 4 列（4个传感器）的原始振动信号。
    取每列的均值绝对值作为特征，得到 shape=(N, 4) 的 DataFrame。

    Returns:
        merged_data: 以时间戳为索引、4列传感器特征的 DataFrame
    """
    if dataset is None:
        dataset = _current_dataset

    config = DATASET_CONFIGS[dataset]
    data_dir = os.path.join(DATA_DIR, config['dir'])

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"数据目录不存在: {data_dir}\n"
            f"请将 NASA IMS Bearing Dataset 的 {dataset}st_test 文件夹放到 dataset/ 下。"
        )

    num_sensors = config['num_sensors']
    rows = []
    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            df = pd.read_csv(filepath, sep='\t', header=None)
            if df.shape[1] < num_sensors:
                print(f"  跳过 {filename}：列数不足 ({df.shape[1]})")
                continue
            # 取前 num_sensors 列
            mean_abs = df.iloc[:, :num_sensors].abs().mean().values
            rows.append((filename, mean_abs))
        except Exception as e:
            print(f"  警告：无法读取 {filename}: {e}")

    if not rows:
        raise RuntimeError(f"未能加载任何数据文件，请检查 {data_dir} 目录。")

    filenames   = [r[0] for r in rows]
    values      = np.stack([r[1] for r in rows])   # (N, num_sensors)
    merged_data = pd.DataFrame(
        values, index=filenames,
        columns=[f'Bearing {i+1}' for i in range(num_sensors)]
    )

    # 将文件名解析为时间戳（格式：2004.02.12.10.32.39）
    merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
    merged_data.sort_index(inplace=True)

    return merged_data


# ---------------------------------------------------------------------------
# 特征工程（可选，需同步修改 train.py 中的 INPUT_DIM）
# ---------------------------------------------------------------------------

def engineer_features(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    在原始均值特征基础上添加衍生特征，帮助模型捕捉趋势变化。

    新增特征：
    - 滚动标准差（窗口=5）：捕捉局部波动幅度
    - 一阶差分：捕捉相邻时间点的变化速率

    所有新特征与原始特征拼接，NaN 用前向填充 + 0 补齐。
    启用后 INPUT_DIM = 4 * 3 = 12，请同步修改 train.py。
    """
    original    = merged_data.copy()
    rolling_std = merged_data.rolling(window=5, min_periods=1).std()
    rolling_std.columns = [f'{c}_std5' for c in rolling_std.columns]
    diff1 = merged_data.diff().fillna(0)
    diff1.columns = [f'{c}_diff1' for c in diff1.columns]

    enriched = pd.concat([original, rolling_std, diff1], axis=1)
    enriched.ffill(inplace=True)
    enriched.fillna(0, inplace=True)
    return enriched


# ---------------------------------------------------------------------------
# 数据准备主函数
# ---------------------------------------------------------------------------

def prepare_data(use_feature_engineering: bool = False, dataset: str = None):
    """
    完整数据准备流程：加载 → （可选特征工程）→ 时间切分 → 归一化。

    Args:
        use_feature_engineering: 是否启用滚动统计等衍生特征（默认关闭）。
            若开启，INPUT_DIM 从 4 变为 12，请同步修改 train.py。
        dataset: 数据集名称 ('1st', '2nd', '3rd')

    Returns:
        X_train (np.ndarray): 训练特征，shape=(N_train, D)，归一化至 [0,1]
        X_test  (np.ndarray): 测试特征，shape=(N_test, D)，归一化至 [0,1]
        y_test  (np.ndarray): 测试集异常标签，0=正常 1=异常
        scaler  (MinMaxScaler): 仅在训练集上拟合的归一化器

    数据无泄露说明：
        scaler 仅在 X_train 上 fit，X_test 用 transform。
        阈值选择应使用训练集重建误差，不应直接用测试集标签搜索（详见 train.py）。
    """
    if dataset is not None:
        _current_dataset = dataset

    if dataset is None:
        dataset = _current_dataset

    suffix = {'1st': '1st_test', '2nd': '2nd_test', '3rd': '3rd_test'}[dataset]
    print(f"加载轴承数据 ({suffix})...")
    merged_data = load_bearing_data(dataset)
    print(f"  共加载 {len(merged_data)} 个时间点")

    if use_feature_engineering:
        print("  应用特征工程（滚动std + 差分）...")
        merged_data = engineer_features(merged_data)
        print(f"  特征维度扩展为 {merged_data.shape[1]}")

    all_labels = get_anomaly_labels(merged_data, dataset)

    config = DATASET_CONFIGS[dataset]
    train_end     = pd.Timestamp(config['train_end_time'])
    dataset_train = merged_data[:train_end]
    dataset_test  = merged_data[train_end:]
    y_test        = all_labels[len(dataset_train):]

    print(f"  训练样本：{len(dataset_train)}（均为正常）")
    print(f"  测试样本：{len(dataset_test)}")
    print(f"  测试集异常：{y_test.sum()} 个（{100 * y_test.sum() / len(y_test):.1f}%）")

    # 归一化：仅在训练集上 fit，防止测试集信息泄露
    scaler  = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(dataset_train).astype(np.float32)
    X_test  = scaler.transform(dataset_test).astype(np.float32)

    return X_train, X_test, y_test, scaler


# ---------------------------------------------------------------------------
# 缓存读写
# ---------------------------------------------------------------------------

def save_preprocessed_data(X_train, X_test, y_test, scaler, dataset: str = None):
    """将预处理结果缓存到磁盘。"""
    if dataset is None:
        dataset = _current_dataset
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"data_{dataset}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test':  X_test,
            'y_test':  y_test,
            'scaler':  scaler,
            'version': CACHE_VERSION,
            'dataset': dataset,
        }, f)
    print(f"数据已缓存至 {cache_file}（版本 {CACHE_VERSION}）")


def load_preprocessed_data(dataset: str = None):
    """
    从磁盘加载缓存数据。
    若缓存不存在或版本不匹配，返回 None（调用方需重新调用 prepare_data）。
    """
    if dataset is None:
        dataset = _current_dataset
    cache_file = os.path.join(CACHE_DIR, f"data_{dataset}.pkl")
    if not os.path.exists(cache_file):
        return None
    with open(cache_file, 'rb') as f:
        cached = pickle.load(f)
    if cached.get('version', 0) != CACHE_VERSION:
        print(f"缓存版本不匹配（当前={cached.get('version',0)}，期望={CACHE_VERSION}），将重新处理。")
        return None
    return cached


# ---------------------------------------------------------------------------
# Dataset / DataLoader 工具
# ---------------------------------------------------------------------------

class BearingDataset(torch.utils.data.Dataset):
    """轴承数据的简单 Dataset 包装。"""

    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def make_dataloader(X: np.ndarray, batch_size: int,
                    shuffle: bool = True, drop_last: bool = False):
    """从 numpy 数组创建 DataLoader。"""
    return torch.utils.data.DataLoader(
        BearingDataset(X),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# 评估工具（供 train.py import）
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred) -> dict:
    """
    计算二分类指标：precision / recall / f1 / accuracy。

    Args:
        y_true: numpy array 或 torch.Tensor，shape=(N,)
        y_pred: numpy array 或 torch.Tensor，shape=(N,)，二值 0/1
    """
    if not torch.is_tensor(y_true):
        y_true = torch.from_numpy(np.array(y_true)).float()
    if not torch.is_tensor(y_pred):
        y_pred = torch.from_numpy(np.array(y_pred)).float()

    tp = ((y_true == 1) & (y_pred == 1)).sum().float()
    tn = ((y_true == 0) & (y_pred == 0)).sum().float()
    fp = ((y_true == 0) & (y_pred == 1)).sum().float()
    fn = ((y_true == 1) & (y_pred == 0)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        'precision': precision.item(),
        'recall':    recall.item(),
        'f1':        f1.item(),
        'accuracy':  accuracy.item(),
    }


def find_best_threshold(errors: torch.Tensor, labels: torch.Tensor,
                        n_thresholds: int = 4000) -> tuple:
    """
    网格搜索使 F1 最大的阈值。

    ⚠️  应传入验证集误差与标签，避免用测试集搜索阈值（数据泄露）。

    Args:
        errors       : 1-D tensor，重建误差
        labels       : 1-D tensor，真实标签（0/1）
        n_thresholds : 候选阈值数量

    Returns:
        (best_threshold: float, best_f1: float)
    """
    thresholds = torch.linspace(errors.min(), errors.max(), n_thresholds)
    best_f1, best_t = 0.0, thresholds[0].item()

    for t in thresholds:
        pred = (errors > t).float()
        tp = ((labels == 1) & (pred == 1)).sum()
        fp = ((labels == 0) & (pred == 1)).sum()
        fn = ((labels == 1) & (pred == 0)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = (2 * precision * recall / (precision + recall + 1e-8)).item()
        if f1 > best_f1:
            best_f1, best_t = f1, t.item()

    return best_t, best_f1


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备 NASA Bearing 数据")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=['1st', '2nd', '3rd'],
                        help="选择数据集 (1st, 2nd, 3rd)")
    parser.add_argument("--force",    action="store_true", help="忽略缓存，强制重新处理")
    parser.add_argument("--features", action="store_true", help="启用特征工程（INPUT_DIM=12）")
    args = parser.parse_args()

    _current_dataset = args.dataset

    cached = None if args.force else load_preprocessed_data(args.dataset)

    if cached is not None:
        suffix = {'1st': '1st_test', '2nd': '2nd_test', '3rd': '3rd_test'}[args.dataset]
        print(f"使用缓存数据 ({suffix})")
        X_train = cached['X_train']
        X_test  = cached['X_test']
        y_test  = cached['y_test']
    else:
        X_train, X_test, y_test, scaler = prepare_data(use_feature_engineering=args.features, dataset=args.dataset)
        save_preprocessed_data(X_train, X_test, y_test, scaler, args.dataset)

    print(f"\n数据维度：")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  特征数  : {X_train.shape[1]}  ← 请确认 train.py 中 INPUT_DIM 与此一致")
    print(f"  测试异常: {y_test.sum()} / {len(y_test)}")
    print("\n数据准备完成，可以开始训练。")