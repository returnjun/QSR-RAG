<div align="center">

# QSR-RAG

### 让问题随已验证的证据一起演化

**Question-State Rewriting for Multi-Hop Retrieval-Augmented Generation**

[中文论文 PDF](paper/qsr_rag_acl.pdf) · [论文源码](paper/qsr_rag_acl.tex) · [主实验复现](#复现主实验) · [结果](#主实验结果)

</div>

> **仓库范围**：这里发布论文主实验的完整 QSR-RAG 路径。三个数据集、两个主干模型、固定的 500 题清单和统一的稠密检索设置均已给出。论文中的对照方法仅用于结果解读，本仓库不提供其运行入口。

![QSR-RAG 的证据验证与问题状态更新流程](assets/qsr_workflow.png)

## 目录

- [研究问题与方法](#研究问题与方法)
- [主实验结果](#主实验结果)
- [复现主实验](#复现主实验)
- [输出与核验](#输出与核验)
- [仓库结构与论文](#仓库结构与论文)

## 研究问题与方法

多跳问答不是一次检索就能完成的任务。随着中间事实被找到，系统需要知道 **下一步要查什么**，也需要知道 **原问题现在应该如何表达**。QSR-RAG 将这两种状态分开：

| 对象 | 作用 | 生命周期 |
| --- | --- | --- |
| 目标问题 $Q^{(0)}$ | 锚定原始任务和最终答案空间 | 始终不变 |
| 推理问题 $Q^{(t)}$ | 表示当前尚待解决的完整问题 | 经验证后跨轮更新 |
| 已验证解答记忆 $M^{(t)}$ | 保存有直接证据支持的中间事实 | 跨轮累积 |
| 局部问题 $q^{(t)}$ | 驱动当轮的一次具体检索 | 当轮使用 |

每一轮先生成局部问题，检索证据并读取局部答案；**答案适配器**检查答案是否解决当前目标、证据是否直接支持该关系。通过检查的事实进入记忆，再由状态更新器选择：

1. **替换（Substitution）**：把已解析的隐藏实体写入问题。例如知道 Tokarev 任教于 Moscow State University 后，把“Tokarev 任教的大学何时建立？”改写为“莫斯科国立大学何时建立？”。
2. **增强（Augmentation）**：当比较、选择、布尔等问题需要保留原候选项与答案空间时，将已验证事实作为前提加入问题，而不直接替换候选项。

更新后的问题进入下一轮；目标问题保持原样，最终阅读器仍面向原始任务作答。系统最多执行 **4 轮**，每轮最多生成 **2 个**同层级局部问题。论文还讨论了状态重写可能造成关系语义漂移的失败案例与局限。

![不同推理状态载体的概念比较](assets/qsr_state_carriers.png)

## 主实验结果

以下为[中文论文](paper/qsr_rag_acl.pdf)主结果表：每个数据集 **500 题**，固定 **seed 43**，指标为答案精确匹配率（EM）与 token 级 F1，单位均为百分比。

| 主干模型 | 数据集 | QSR-RAG EM | QSR-RAG F1 | 该设置最强对照 F1 |
| :--- | :--- | ---: | ---: | ---: |
| GPT-4o-mini | HotpotQA | **57.60** | **73.65** | 70.47 |
| GPT-4o-mini | 2WikiMultiHopQA | **64.80** | **74.19** | 72.08 |
| GPT-4o-mini | MuSiQue | **42.00** | **55.35** | 51.02 |
| Qwen3-8B（非思考） | HotpotQA | **57.40** | **72.25** | 66.98 |
| Qwen3-8B（非思考） | 2WikiMultiHopQA | **62.60** | **71.88** | 69.99 |
| Qwen3-8B（非思考） | MuSiQue | **35.20** | **48.13** | 42.24 |

![三个数据集上 QSR-RAG 与各设置最强对照的 F1](assets/main_results_f1.png)

QSR-RAG 在六种设置中均取得最高的 EM/F1。相对各设置最强对照，六种设置平均提升 **4.63 EM 点**和 **3.78 F1 点**。GPT-4o-mini 的跨数据集平均 F1 为 **67.73**；平均每题使用 **20.47k tokens**。这些数值来自论文中共享检索后端的受控比较，不应与采用不同语料或检索器的公开榜单直接比较。图可用 `python scripts/plot_main_results.py` 重新生成。

## 复现主实验

以下命令在仓库根目录运行，以 **Python 3.10+** 为例。完整六组实验需要下载三个开发集、两个检索模型，并为两个语言模型提供可用的兼容 Chat Completions 接口；实际耗时与费用取决于硬件和服务商。建议先运行 5 题检查环境，再运行 500 题。断点文件会保存在 `outputs/`。

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS 将激活命令改为 `source .venv/bin/activate`。

### 2. 准备开发集

| 数据集 | 官方来源 | 放置位置 |
| --- | --- | --- |
| HotpotQA distractor dev | [HotpotQA 下载页](https://hotpotqa.github.io/) | `data/raw/hotpotqa/hotpot_dev_distractor_v1.json` |
| 2WikiMultiHopQA dev | [2WikiMultiHopQA 官方仓库](https://github.com/Alab-NII/2wikimultihop) | 先放 `data/raw/2wikimultihopqa/dev.json` |
| MuSiQue-Ans dev | [MuSiQue 官方仓库](https://github.com/stonybrooknlp/musique) | `data/raw/musique/musique_ans_v1.0_dev.jsonl` |

2Wiki 的官方 `dev.json` 放好后，执行一次格式转换；脚本保留原始样本顺序：

```powershell
python scripts/prepare_2wiki.py
```

论文主实验使用的三份开发集分别包含约 **66,581 / 56,686 / 21,098** 篇索引文档（按 HotpotQA / 2Wiki / MuSiQue 顺序）。`manifests/` 中只保存与论文主表对应的 **500 个题目 ID 及其顺序**，不包含数据集内容。若使用其他镜像或版本，请核对题目 ID、顺序与数据格式；不同 Parquet 写入版本可能产生不同的文件哈希，即使行内容相同。

<details>
<summary>本地论文实验数据文件的 SHA-256（用于核对下载版本）</summary>

| 文件 | SHA-256 |
| --- | --- |
| `hotpot_dev_distractor_v1.json` | `E3DA074DF24E8369009918AA5CDBDD254DADCDE4C63F7569D36AFD6F2268CAA8` |
| `dev.parquet` | `C0D8B60B9026B728FB07AD74C5252A0F188F6942E8BA5C02DF4DFA369502EA8D` |
| `musique_ans_v1.0_dev.jsonl` | `15FA63794D18A94CE12411ACA6E2327E65B6E83B0B1490EFAB3F1962E48ABF3B` |

</details>

### 3. 下载检索模型并建立索引

论文采用 [BGE-M3](https://huggingface.co/BAAI/bge-m3) 稠密编码和 [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) 重排序。以下命令将模型放到代码默认路径：

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='models/bge-m3')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='models/bge-reranker-v2-m3')"
python prepare_indexes.py
```

索引由**完整开发集**中的文档并集构建，而非只用选出的 500 题。主实验逐次检索使用 BGE-M3 + FAISS 稠密 top-200 → 文档重排 top-60 → 半径 1 的句子窗口重排 → top-20 窗口。索引文件会保存在 `indexes/`，不纳入 Git。

### 4. 配置语言模型

复制模板，按实际服务商调整模型名和接口地址；密钥只从环境变量读取，`config.json` 已被忽略：

```powershell
Copy-Item config.example.json config.json
$env:OPENAI_API_KEY = "你的 GPT-4o-mini 密钥"
$env:DASHSCOPE_API_KEY = "你的 Qwen3-8B 密钥"
```

`config.example.json` 中 Qwen3-8B 已设置 `enable_thinking: false`。四个阶段——局部问题生成、局部阅读、答案适配、状态更新——在同一组实验中使用同一个主干模型。若只运行一种模型，只需配置相应的密钥。

### 5. 运行六组主实验

```powershell
# 先检查一组 5 题：
python reproduce.py --models gpt4o_mini --datasets hotpotqa --num-questions 5

# 正式运行 3 数据集 × 2 主干模型，每组 500 题：
python reproduce.py
```

可以用 `--models`、`--datasets` 选择一部分任务，例如 `python reproduce.py --models qwen3_8b_non_thinking --datasets musique`。运行中断后再次执行相同命令会继续读取已有的逐题结果；若从 5 题扩展到 500 题，保留原输出即可继续。服务商暂时故障、模型版本变化、数值精度和推理环境都可能使重新运行的分数与论文略有差异。

## 输出与核验

每组运行会生成：

```text
outputs/main/<模型配置名>/<数据集>/
├── <数据集>_<模型配置名>_seed43_details.jsonl  # 逐题预测、证据、轨迹和 EM/F1
└── <数据集>_<模型配置名>_seed43_summary.json  # 汇总 EM/F1 与资源指标
```

摘要中的 `em`、`f1` 为 **0–1 比例**，乘以 100 后与上表的百分数比较。请同时核对 `completed_questions = 500`、`retrieval.policy_compliant = true`，以及每组详情中题目 ID 与 `manifests/<dataset>_seed43_500.jsonl` 一致。网络或模型调用在重试后仍失败的题目会保留在分母中，并计为 EM/F1=0。

## 仓库结构与论文

```text
QSR-RAG/
├── README.md                     # 中文项目主页与复现步骤
├── reproduce.py                  # 六组主实验入口
├── prepare_indexes.py            # 三个完整开发集的稠密索引
├── config.example.json           # 无密钥的模型配置模板
├── manifests/                   # 论文主表固定的题目 ID 顺序
├── data/                        # 数据解析器；原始数据自行下载
├── retriever/                   # 检索、阅读、适配、状态更新的核心实现
├── scripts/                     # 主评测与预处理脚本
├── assets/                      # GitHub 可直接显示的论文图与结果图
└── paper/                       # 中文 ACL 版式 PDF、TeX、参考文献和矢量图
```

中文论文为匿名审稿稿件；[PDF](paper/qsr_rag_acl.pdf) 可直接阅读，[TeX](paper/qsr_rag_acl.tex) 可用 XeLaTeX → BibTeX → XeLaTeX ×2 编译。论文中另有补充分析、消融与失败案例；本仓库的运行入口聚焦上表六组主实验。引用信息将在论文正式发表后补充，现阶段请引用论文标题与本仓库链接。
