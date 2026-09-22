# Knowledge-Routing-RAG

> **面向复杂法规检索的受约束知识路由架构**  
> *探索将知识网络与元数据作为“导航平面”而非直接填充“证据上下文”的研究原型。*

[English](README.md) | [中文说明](README_CN.md)

[![Release](https://img.shields.io/badge/release-v3--research--final-blue.svg)](https://github.com/bilppppp/Knowledge-Routing-RAG/releases)
[![Status](https://img.shields.io/badge/status-frozen__research-success.svg)](#10-局限性与有效性边界)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](requirements.txt)

---

## 项目核心摘要

**Knowledge-Routing-RAG** 探索了一种受约束的检索架构，将图谱与元数据作为**导航平面**而非直接充当证据上下文。在独立的 250 题 Holdout 测试集上，冻结的 V3 系统相比纯 Vector RAG 基线将回答准确率从 **67.6% 提升至 73.2%**，同时仅增加低毫秒级本地路由开销，且**无需任何额外的在线 LLM 调用**。规模鲁棒性与 Hub 鲁棒性**未得到确认**。

---

## 1. 这是什么？

本仓储包含 `Knowledge-Routing-RAG` 项目已全部完成并永久冻结的研究代码、实验协议、原始证据产物与最终科学结论。

当前实验阶段已经全部结束。系统架构、Prompt 及评测基准已依据 [`V3_FREEZE.md`](V3_FREEZE.md) 进行密码学哈希封存，**不再进行算法迭代或调优**。

### 核心痛点 (The Problem)
标准稠密向量检索（Vector RAG）依赖 Query 与切片 Embedding 之间的语义相似度。在复杂的专业合规领域（如卫生行政法规、技术标准、法律条款）中，向量检索常面临以下困难：
- **跨文档依赖 (Cross-Document Dependencies)**：正确答案需要多部不同法律法规条文的协同支撑。
- **多跳证据链 (Multi-Hop Evidence Chains)**：下位规章从上位母法衍生授权依据（`BASED_ON`），或将具体执行细则转引至配套文件（`REFERENCES`）。
- **时效与修订冲突 (Temporal / Version Relations)**：新法规修正或废止旧法规条款（`AMENDS`、`SUPERSEDES`）。

在这些场景下，Vector RAG 极易陷入“局部相似性陷阱”，仅检索到单一文档中的表层相似段落，而系统性遗漏分散在其他法规中的关键证据。

### 核心设计哲学 (The Core Idea)
朴素图检索（Naive Graph RAG）往往直接将多跳遍历到的邻居节点全量塞入 LLM 上下文，极易引发**图洪泛 (Graph Flooding)** 和上下文稀释。

Knowledge Routing 将**候选发现**与**最终证据上下文**彻底解耦：
- **图谱与元数据仅用于导航 (Navigation Plane)**：控制平面的拓扑网络与阴影候选池只负责探路并定位目标文档。
- **证据上下文严格预算物理隔离 (Evidence Context)**：进入生成模型的切片数量被硬性锁死在 $\le 5$ 块（$\le 4000$ Tokens），并通过覆盖度保持组合算法防止无关干扰。

> **核心原则**：
> 1. **候选空间 $\ne$ 证据上下文 (Candidate Space $\ne$ Evidence Context)**
> 2. **全局路由，局部下潜，全局组合 (Route globally, retrieve locally, compose globally)**

---

## 2. 最终冻结架构 (V3-Frozen)

经过多轮消融与去污染检验，最终系统收敛并冻结为 **V3-Frozen**（候选代号 `E2-Lite`）：

```text
                           用户提问 (Question)
                                  │
                                  ▼
                        向量入口 (Vector Entrance)
                                  │
              ┌───────────────────┴───────────────────┐
              │ 快速通道判定 (Fast Path: Sim ≥ 0.85)   │
              ▼                                       ▼
       [直通 Fast Path]                        [激活路由通道]
              │                                       │
              │                      ┌────────────────┴────────────────┐
              │                      │   控制平面阴影前缀池 (Top-20)     │
              │                      └────────────────┬────────────────┘
              │                                       │
              │                      ┌────────────────┴────────────────┐
              │                      │   目标法规前缀解析与上位法抬升   │
              │                      └────────────────┬────────────────┘
              │                                       │
              │                      ┌────────────────┴────────────────┐
              │                      │ E2-Lite 槽位条件词法下潜 (BM25)  │
              │                      └────────────────┬────────────────┘
              │                                       │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │  E1 覆盖度保持组合器        │ (Coverage-Preserving Composer)
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │  最终证据上下文 (≤ 5 块切片) │ (预算严格物理受控)
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │  生成模型 (DeepSeek-Chat)   │ (零工程偏置，采用原始 B0 Prompt)
                   └─────────────────────────────┘
```

### 关键组件说明：
1. **FIB 向量入口**：稠密向量初检 Top-5 切片（`google/embeddinggemma-300m`）。
2. **快速通道 (Fast Path)**：若首位匹配度 $\ge 0.85$，直接透传，确保单跳简单问题不受任何额外逻辑干扰。
3. **阴影前缀知识路由 (Shadow Knowledge Routing)**：在控制平面检查 Top-20 阴影候选池（RIB），汇聚法规前缀，沿拓扑图谱解析立法依据（`BASED_ON`）与引用关系（`REFERENCES`）。
4. **E2-Lite 局部词法下潜 (Lexical Targeted Descent)**：精准锁定缺失目标法规后，仅针对未闭合的语义槽位执行 BM25 词法全文检索与标题层级加权。（*注：经消融实验证明，目标法规内的稠密向量下潜及 RRF 排序融合贡献为零且稀释首位，已被彻底剔除*）。
5. **E1 覆盖度保持证据组合器 (Coverage-Aware Composer)**：依据边际覆盖效用进行准入判定，**彻底消除了将原有用基线证据误挤出的缺陷**。
6. **B0 原始提示词**：合成阶段采用 100% 原始未修改的基准 Prompt，杜绝任何提示词微调带来的虚假增益。

---

## 3. 核心实验结果：独立 Holdout-2

所有结论以预注册的 **Independent Holdout-2**（$N = 250$ 道全新题目，全量 $D_{100}$ 干扰语料库，3 轮独立随机种子生成，全部分歧案例双盲人工仲裁）为最终黄金标准：

| 评估指标 (Metric) | 纯向量基线 (Vector B0) | V3-Frozen 知识路由 | 增量 (Delta) | 显著性检验与机制判定 |
|:---|:---:|:---:|:---:|:---|
| **3-Run Majority 准确率** | **67.60%** (169/250) | **73.20%** (183/250) | **+5.60pp** | **McNemar 精确检验 $p = 0.0043$** |
| 95% 成对 Bootstrap 置信区间 | — | — | **[+2.00pp, +9.20pp]** | 严格高于 0，统计显著 |
| 稳定救回题数 (Stable Rescues) | — | — | **18 题** | $B0$ 错而 $V3$ 对 |
| 稳定退步题数 (Stable Regressions) | — | — | **4 题** | $B0$ 对而 $V3$ 错 |
| **净稳定救回 (Net Stable Rescue)** | — | — | **+14 题** | 二项检验 $p = 0.0044$ |
| **黄金法规召回率 (Gold Doc Recall)** | 88.93% | 98.53% | **+9.60pp** | 文档级导航机制完全证实 |
| **黄金切片召回率 (Gold Chunk Recall)**| 75.73% | 76.53% | **+0.80pp** | 切片级微观证据增益 |
| **完整证据链闭合率 (Chain Completion)**| 54.40% | 57.60% | **+3.20pp** (+8题) | 多法规证据链条成功闭合 |
| **多跳问题准确率 (2-Hop+ Acc)** | 48.72% | 57.69% | **+8.97pp** | 收益高度集中于复杂多跳 |
| **单跳简单题准确率 (1-Hop Acc)** | 98.94% | 98.94% | **+0.00pp** | **简单题零退步 (0 Regressions)** |

---

## 4. 已确认与未确认的结论

为保持学术与工程诚实，本研究对实验结论划定明确边界：

### 已确认的结论 (CONFIRMED)
- [x] **端到端准确率提升**：已证实（Holdout-2 上取得可重复且统计显著的 $+5.60\text{pp}$ 增益，$p=0.0043$）。
- [x] **文档导航能力增益**：已证实（黄金法规召回率提升 $+9.60\text{pp}$，从 $88.93\%$ 升至 $98.53\%$）。
- [x] **多跳复杂推理增益**：已证实（多跳题准确率大幅提升 $+8.97\text{pp}$，且单跳题保持 $0$ 退步）。
- [x] **证据组合器与局部下潜的必要性**：已证实（E1 组合器将历史版本的有用证据挤出率从 $9.5\%$ 彻底降至 $0.0\%$）。
- [x] **极高的工程落地价值**：已证实（在当前硬件与任务规模下，本地 P50 延迟仅增加 $+0.58\text{ ms}$，平均增加 $+3.63\text{ ms}$；**完全零新增在线 LLM 调用**，**零新增 Query Embedding**，输入 Token 仅微增 $+1.96\%$）。

### 未确认与负面结果 (NOT CONFIRMED)
- [ ] **规模鲁棒性优势**：**`未确认 (NOT CONFIRMED)`**。在 Phase C 评测中（$N=80$, $D_{20} \to D_{100}$ 语料扩张），V3 退化了 $+2.50\text{pp}$，而 B0 退化为 $+0.00\text{pp}$，规模鲁棒性优势 $\text{RA} = -2.50\text{pp}$（95% 置信区间 `[-6.25pp, 0.00pp]`）。**因此严禁声称“V3 比纯 Vector RAG 更能抵抗知识库扩张带来的干扰”。**
- [ ] **Hub 节点稳定性**：**`未确认 (NOT CONFIRMED)`**。在 Phase D 高度数测试中，度数大于 20 的节点引发了 V3 内部候选切片扩张（P95 达到 27.1），证据链完整度从 $60\%$ 骤降至 $42.5\%$。
- [ ] **图洪泛抑制**：**`未确认 / 不可测 (NOT TESTABLE)`**。由于历史无约束自由图遍历基线（Legacy Graph baseline）在代码库中未留存，缺乏同场对照，因此无法声称 V3 已证实解决了图洪泛。
- [ ] **全局普遍优越性**：**`未声称`**。收益仅在具有显式法律效力层级与条文引用的法规语料中得到验证。

---

## 5. 必须明确的方法学披露：Dev-216 的地位

在早期研发过程中，项目曾使用包含 120 道题目跨 $D_{20}/D_{50}/D_{100}$ 展开的 216 个评测实例来迭代候选系统 C1～C7。

随着研发推进：
- 该数据集被频繁用于错题人工归因、Prompt 调整及针对性路由逻辑修改；
- **Dev-216 事实上已转变为“开发/优化集 (Optimization / Development Set)”**；
- 早期 C7 在该集合上取得的 $80.09\%$ 成绩部分受特定启发式规则影响，**绝不可作为外推泛化的有效学术证据**；
- 项目随后开展全面去污染审计，构建了通用干净的 `C7-Clean`，并最终以完全物理隔离且冻结的 **Holdout-2 ($73.20\%$)** 作为唯一终审依据。

---

## 6. 快速开始 (Quick Start)

### 环境依赖
- Python 3.12+
- Docker & Docker Compose（用于本地 Qdrant 与 Embedding 容器）

### 1. 克隆代码与配置 Python 虚拟环境
```bash
git clone https://github.com/bilppppp/Knowledge-Routing-RAG.git
cd Knowledge-Routing-RAG

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 环境变量配置
```bash
cp .env.example .env
# 若需运行在线 LLM 生成实验，请在 .env 中填入有效的 DEEPSEEK_API_KEY
```

### 3. 启动本地容器化向量与嵌入服务
```bash
docker compose up -d
```

### 4. 运行发布冒烟测试 (Smoke Test)
```bash
python scripts/release_smoke_test.py
```

预期输出：
```text
======================================================
 Knowledge-Routing-RAG — Release Smoke Test 
======================================================
  [PASS] Core modules imported successfully.
  [PASS] Runtime configuration files loaded and validated.
  [PASS] Knowledge LSDB loaded (100 documents, 2862 chunks, 1397 routing edges).
  [PASS] B0 single query executed (retrieved 5 chunks).
  [PASS] V3-Frozen single query executed (retrieved 5 chunks, budget constraint <= 5 chunks satisfied).

All smoke tests passed successfully! Release candidate is ready.
```

---

## 7. 实验复现指南 (Reproduction Protocols)

完整参数、密码学哈希清单及说明详见 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)。

- **Level 1 — 基础冒烟测试 (极简验证)**：
  ```bash
  python scripts/release_smoke_test.py
  ```
- **Level 2 — 确定性检索机制复现 (零在线 LLM 消耗)**：
  ```bash
  python scripts/run_holdout2_experiment.py --retrieval-only
  ```
- **Level 3 — 全量科研实验端到端复现 (需配置 API 密钥)**：
  ```bash
  # 1. 复现核心结论：Holdout-2 终审评测 (N=250, 3-run 多数表决)
  python scripts/run_holdout2_experiment.py

  # 2. 复现规模鲁棒性评测 (Phase C: N=80)
  python scripts/run_scale1_experiment.py

  # 3. 复现 Hub 压力评测 (Phase D: N=100)
  python scripts/run_hub1_experiment.py

  # 4. 复现工程延迟与复杂度基准测试 (Phase E)
  python scripts/benchmark_phase_e_cost.py
  ```

---

## 8. 仓储目录结构

```text
Knowledge-Routing-RAG/
├── README.md                      # 英文项目主文档
├── README_CN.md                   # 中文项目主文档 (本文档)
├── REPRODUCIBILITY.md             # 硬件、随机种子、哈希与 3 级复现指南
├── RELEASE_NOTES.md               # GitHub Release 发版说明
├── V3_FREEZE.md                   # V3 架构与代码哈希冻结清单
├── LICENSE                        # MIT 开源许可证
├── requirements.txt               # 运行环境依赖 (Python 3.12+)
├── docker-compose.yml             # Qdrant 与本地 Embedding 容器编排
├── configs/
│   ├── b0_baseline.yaml           # 纯 Vector RAG 运行配置
│   ├── v3_frozen.yaml             # 冻结的 V3 Knowledge Routing 运行配置
│   └── runtime_config.yaml        # 历史通用运行时配置
├── data/
│   ├── knowledge_lsdb.sqlite      # SQLite Link-State 数据库 (FTS5 + 拓扑有向图)
│   ├── chunks.jsonl               # 2,862 条法规切片数据
│   ├── documents/                 # 100 篇卫生健康法规全文 (doc001–doc100)
│   └── manifests/                 # D20 / D50 / D100 语料子集划分清单
├── benchmark/
│   ├── README.md                  # 数据集使用说明与暴露状态清单
│   ├── questions.jsonl            # Dev-216 (优化/开发集，STATUS: EXPOSED)
│   ├── confirmation/              # Holdout-1 (N=200，STATUS: EXPOSED AFTER CONFIRMATION)
│   ├── holdout2/                  # Holdout-2 (N=250，STATUS: FINAL INDEPENDENT CONFIRMATION)
│   ├── scale1/                    # ScaleSet-1 (N=80，STATUS: SCALE CONFIRMATION)
│   └── hub1/                      # HubSet-1 (N=100，STATUS: HUB CHARACTERIZATION)
├── src/
│   ├── retrieval/                 # B0 向量检索实现
│   ├── routing/                   # C7-Clean、E1 组合路由器与 E2-Lite 下潜路由器
│   ├── composition/               # E1 覆盖度保持组合器与 E2 词法下潜算法
│   ├── graph/                     # SQLite Knowledge LSDB 拓扑引擎
│   ├── services/                  # 向量检索、本地嵌入与模型服务接口
│   └── evaluation/                # 评估指标计算模块
├── scripts/
│   ├── release_smoke_test.py      # 发布冒烟验证测试
│   ├── run_holdout2_experiment.py # Holdout-2 终审实验运行脚本
│   ├── run_scale1_experiment.py   # Phase C 规模鲁棒性实验运行脚本
│   ├── run_hub1_experiment.py     # Phase D Hub 压力实验运行脚本
│   └── benchmark_phase_e_cost.py  # Phase E 延迟与复杂度评测脚本
└── reports/
    ├── README.md                  # 01～09 全研究阶段学术报告索引
    ├── FINAL_REPORT.md            # 终审综合科研实验报告 (完整论述)
    └── final_repository_audit.md  # 仓储与发布就绪状态审计报告
```

---

## 9. 核心报告与科学证据链索引

全流程研究报告与原始 JSON 评测记录已全部收录于 [`reports/README.md`](reports/README.md)：
- **权威科研总报告**：[`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md)
- **Holdout-2 终审确认报告**：[`reports/v3_holdout2_confirmation.md`](reports/v3_holdout2_confirmation.md)
- **Phase C 规模鲁棒性报告**：[`reports/v3_scale_robustness_confirmation.md`](reports/v3_scale_robustness_confirmation.md)
- **Phase D Hub 压力评测报告**：[`reports/v3_hub_stress_confirmation.md`](reports/v3_hub_stress_confirmation.md)
- **Phase E 成本与复杂度审计报告**：[`reports/v3_cost_complexity_audit.md`](reports/v3_cost_complexity_audit.md)
- **去污染与规则清洗审计报告**：[`reports/decontamination_rule_audit.md`](reports/decontamination_rule_audit.md)

---

## 10. 局限性与有效性边界

1. **语料领域边界**：评测语料局限于 100 篇中国卫生医疗行政法规（共 2,862 条切片）。系统设计深度依托于法规之间显式的条文援引与立法授权层级（`BASED_ON`、`REFERENCES`、`SUPERSEDES`、`AMENDS`）。**在未经领域适配前，严禁将本架构结论直接推广至非结构化叙事文本、开放领域网页或代码库检索。**
2. **语料规模边界**：实验测试上限为 100 篇法规（$D_{100}$）。Phase C 的规模退化表明，在更大规模知识库（1,000+ 文档）中，前缀探索需要更严格的自适应剪枝。
3. **模型依赖性**：所有生成与裁判均基于 `deepseek-chat`（Greedy 贪婪解码，温度 $T=0.0$）。
4. **科研原型状态**：本仓储是用于验证检索机制机理的**冻结科研原型**，并非高可用生产微服务。

---

## 11. 开源协议

本项目采用 [MIT 许可证](LICENSE)。
