# autoresearch

这是一个让大语言模型（LLM）自主进行研究的实验。

## Setup

要设置一个新的实验，请与用户一起完成以下步骤：

1. **确定运行标签（run tag）**：根据今天的日期和使用的数据集提出一个标签（例如 `mar5-1st`）。分支 `ae/<tag>` 必须不存在——这是一个全新的实验。
2. **创建分支**：先 `git checkout autoanomaly`  到 autoanomaly 分支，再执行 `git checkout -b ae/<tag>`。
3. **阅读范围内的文件**：该仓库较小，请阅读以下文件以获取完整上下文：

   * `README.md` —— 仓库背景信息。
   * `prepare.py` —— 数据准备、特征处理、评估函数（**不要修改**）。
   * `train_ae.py` —— 需要修改的文件（模型结构、训练逻辑等）。
4. **验证数据存在**：确保数据目录 `dataset/` 存在且可读取（或已有缓存 `~/.cache/autoanomaly/`）。
5. **初始化 results.tsv**：对每个数据集创建仅包含表头的 `results.tsv` 文件（例如`results-1st.tsv`）。基线结果将在第一次运行后记录。
6. **确认并开始**：确认设置无误。

在获得确认后，启动实验流程。

---

## Experimentation

每个实验在单张 GPU 上运行。训练脚本运行**固定 5 分钟时间预算**（由 `prepare.py` 中 `TIME_BUDGET` 控制）。

运行方式：

```
uv run train_ae.py --dataset 数据集编号
```

**你可以做的：**

* 修改 `train_ae.py` —— 唯一允许编辑的文件（模型结构、损失函数、训练策略、超参数等）

**你不能做的：**

* 修改 `prepare.py`
* 修改数据处理逻辑或标签生成方式
* 安装新依赖
* 修改评估函数（`calculate_metrics`、`find_best_threshold`）

---

### 目标

**最大化 `val_f1`（越高越好）**

---

### 约束补充

* 这是**无监督异常检测任务**
* 训练集**全部为正常样本**
* 测试集包含异常标签（仅用于最终评估）
* 阈值选择必须避免数据泄露（优先使用 `train_percentile`）

---

### 简洁性原则：在其他条件相同情况下，越简单越好。

* 小幅提升但增加复杂代码，不值得。
* 删除代码却效果更好，是优秀改进。
* 提升很小但复杂度大幅增加，不建议保留。
* 提升不明显但代码更简单，应保留。

### 第一次运行: 必须先运行原始代码，建立 baseline。


---

## Output format

脚本输出如下：

```
---
val_f1:           0.85
val_precision:    0.90
val_recall:       0.80
val_accuracy:     0.88
training_seconds: 300.0
total_seconds:    320.0
peak_vram_mb:     2000.0
total_steps:      1000
threshold:        0.123456
dataset:          2nd
```

提取关键指标：

```
grep "^val_f1:" run.log
```

---

## Logging results

TSV 格式：

```
commit	val_f1	memory_gb	status	description
```

字段说明：

1. commit： git commit hash
2. val_f1——崩溃填 0.000000
3. memory_gb：（GB，保留1位小数，例如 12.3，计算方式：peak_vram_mb ÷ 1024）——崩溃填 0.0
4. status: `keep` / `discard` / `crash`
5. description：实验简要描述

示例：

```
commit	val_f1	memory_gb	status	description
a1b2c3d	0.812300	2.1	keep	baseline
b2c3d4e	0.845000	2.2	keep	将学习率提高到 0.04
c3d4e5f	0.790000	2.1	discard	切换为 GeLU 激活函数
d4e5f6g	0.000000	0.0	crash	OOM
```

---

## The experiment loop

实验在专用分支上进行（例如 `ae/mar5-1st`）。

**无限循环执行：**

1. 查看当前 git 状态（分支/提交）
2. 修改 `train_ae.py`，实现新的实验想法
3. 提交 git
4. 运行实验：

```
uv run train_ae.py --dataset 数据集编号 > run.log 2>&1
```

（重定向所有输出，不要使用 tee）

5. 提取结果：

```
grep "^val_f1:\|^peak_vram_mb:" run.log
```

6. 若无输出，说明崩溃：

   * 使用 `tail -n 50 run.log` 查看错误
   * 尝试修复
   * 若多次失败，放弃该思路

7. 记录结果到 tsv（注意：**不要提交 results.tsv 到 git**）

8. 如果 **val_f1 提升（更高）** → 保留

9. 否则 → 回滚

核心思想：
你是一个完全自主的研究员，不断尝试新想法。

* 有效 → 保留
* 无效 → 丢弃

**超时规则**：

* 正常实验约 5 分钟
* 超过 10 分钟 → 强制终止并视为失败

**崩溃处理：**

* 简单问题 → 修复后重试
* 思路本身有问题 → 标记 crash 并跳过

**永不停止：**
一旦开始实验循环：

* 不要询问用户是否继续
* 不要暂停
* 持续运行直到人为停止

典型场景：
用户睡觉时运行实验。

* 每 5 分钟一次
* 每小时约 12 次
* 一晚可运行约 100 次

用户醒来时，将看到完整实验结果。
