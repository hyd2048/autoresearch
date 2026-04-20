# autoresearch

这是一个让大语言模型（LLM）自主进行研究的实验。

本实验同时对三种无监督异常检测方法进行优化：

| 方法 | 训练脚本 | 子程序文件 | 分支前缀 |
|------|----------|------------|----------|
| Autoencoder | `train_ae.py` | `program_ae.md` | `ae/` |
| PCA | `train_pca.py` | `program_pca.md` | `pca/` |
| Isolation Forest | `train_iforest.py` | `program_iforest.md` | `iforest/` |

---

## Setup

1. **确定运行标签（run tag）**：根据今天的日期提出一个标签（例如 `mar5-1st`）。三个分支 `ae/<tag>`、`pca/<tag>`、`iforest/<tag>` 必须均不存在。

2. **为每种方法执行各自的 Setup**：读取对应子程序文件，按其 Setup 章节完成初始化：
   - 读取 `program_ae.md` → 完成 AE 的 Setup
   - 读取 `program_pca.md` → 完成 PCA 的 Setup
   - 读取 `program_iforest.md` → 完成 IForest 的 Setup

3. **确认并开始**：三种方法均初始化完毕后，确认设置无误。

在获得确认后，启动实验循环。

---

## The experiment loop

三种方法按 `ae → pca → iforest` 顺序**轮流**推进，无限循环直到人为停止。

每轮对当前方法：

1. 切换到对应分支（例如 `git checkout ae/<tag>`）
2. 读取该方法的子程序文件，按其 **Experimentation** 和 **The experiment loop** 章节执行一次完整实验
3. 轮转至下一种方法

**永不停止**：不要询问用户是否继续，不要暂停。