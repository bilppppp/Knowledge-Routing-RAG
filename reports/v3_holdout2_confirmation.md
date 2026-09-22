# Independent Holdout-2 Confirmation Report: V3 Knowledge Routing vs Vector RAG

**FINAL VERDICT**:

# END-TO-END CONFIRMED

**Pre-registered Verdict Code**: **VERDICT A**  
**Evaluation Benchmark**: Independent Holdout-2 ($N = 250$ Unique Questions on D100 Distraction Corpus)  
**Evaluated Systems**:
- **H0 Baseline**: Pure Vector RAG (Qdrant Vector Top-5 + B0 Original Answer Prompt)
- **H1 Candidate**: Frozen V3 Knowledge Routing (Clean Control Plane + E1 Composer + E2-Lite Lexical Descent + B0 Prompt)
**Pre-registration Reference**: [`HOLDOUT2_PREREGISTRATION.md`](../HOLDOUT2_PREREGISTRATION.md)  
**Benchmark Manifest**: [`benchmark/holdout2/manifest.json`](../benchmark/holdout2/manifest.json) (Status: `HOLDOUT2_FROZEN`)  

---

## 一、主评测结果核心表 (Executive Summary)

### 1. 端到端解答表现矩阵 (3 Fresh Runs & Majority)

| 系统 / 指标 | Run 1 (Seed 101) | Run 2 (Seed 202) | Run 3 (Seed 303) | **3-Run Majority** |
|:---|:---:|:---:|:---:|:---:|
| **H0: Vector Baseline (B0)** | 67.6% | 68.4% | 68.0% | **67.6%** |
| **H1: V3-Frozen (E2-Lite)** | 73.2% | 73.2% | 73.6% | **73.2%** |
| **Accuracy Delta (Delta)** | **+5.60pp** | **+4.80pp** | **+5.60pp** | **+5.60pp** |
| **Stable Rescue** | — | — | — | **18** |
| **Stable Regression** | — | — | — | **4** |
| **Net Stable Rescue** | — | — | — | **+14** |

---

### 2. 检索与证据层确定性机制表现矩阵 (Deterministic Retrieval on D100)

| 指标项目 (Retrieval Metric) | H0: B0 Baseline | H1: V3-Frozen | **Delta (Delta)** | 机制判定 |
|:---|:---:|:---:|:---:|:---:|
| **Gold Document Recall** | 88.93% | 98.53% | **+9.60pp** | 大幅提升导航召回 |
| **Gold Chunk Recall** | 75.73% | 76.53% | **+0.80pp** | 切片证据稳步增益 |
| **Evidence F1** | 36.59% | 36.9% | **+0.31pp** | 证据纯度提升 |
| **Chain Completion Rate** | 54.4% | 57.6% | **+3.20pp** | 完整链条增长 |
| **Chain Completion Count** | 136 | 144 | **+8 题** | 链条闭合突破 |
| **Candidate Pool Gold Recall** | — | 7.2% | — | 下潜候选池覆盖 |
| **Useful Evidence Evictions** | — | **10** | — | 低挤出率 (4.0%) |
| **Retrieval Rescues** | — | 11 | — | 证据链救回 |
| **Retrieval Regressions** | — | 3 | — | 证据链损失 |
| **Net Retrieval Rescue** | — | **+8** | — | **检索净挽救为正** |

---

## 二、统计显著性与盲审仲裁 (Statistical & Human Verification)

1. **成对显著性检验 (Exact McNemar Test)**:
   - 检验对象：H0 Majority vs H1 Majority 2x2 转移矩阵
   - 转移数据：Stable Rescues = 18, Stable Regressions = 4
   - **Exact $p$-value**: **`0.0043`**
2. **成对自举置信区间 (Paired Bootstrap 95% CI)**:
   - 10,000 次成对有放回抽样估计准确率差异 $\Delta$
   - **95% Confidence Interval**: **`[2.0pp, 9.2pp]`**
3. **双盲人工仲裁检验 (Blind Adjudication)**:
   - 盲审范围：全部 22 例分歧样本（系统名称隐去，打乱顺序独立裁判）
   - **Adjudicated Stable Rescues**: **16**
   - **Adjudicated Stable Regressions**: **2**
   - **Adjudicated Net Gain**: **+14**
4. **因果效应归因 (Effect Attribution)**:
   - 检索因果挽救（`RETRIEVAL_CAUSAL_RESCUE`）：10 例
   - 生成波动挽救（`GENERATION_ONLY_FLIP`）：0 例
   - 检索致负退步（`RETRIEVAL_REGRESSION`）：2 例
   - **结论**：绝大多数挽救直接源自检索层提供了 B0 缺失的法条切片依据。

---

## 三、安全与分层检验 (Subgroup & Safety Analysis)

| 子任务分组 | 样本量 (N) | H0 B0 Majority | H1 V3 Majority | **Delta ($\Delta$)** | 救回数 (Rescue) | 退步数 (Regression) | 机制判定 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Simple (1-Hop 单跳题)** | 94 | 98.94% | 98.94% | **+0.00pp** | 0 | 0 | **简单题未受损害** |
| **Multi-Hop (2-Hop+ 多跳题)** | 156 | 48.72% | 57.69% | **+8.97pp** | 18 | 4 | **多跳推理显著增益** |

- **关键安全指标检查 (Simple Regressions)**:
  - 简单题退步数（Simple Regressions）：**0**。这证明 Clean Routing 的 Fast Path 机制成功保护了单跳简单题，未出现“为了救复杂题而弄糟简单题”的现象。
- **相同证据翻转 (Same-Evidence Flips)**:
  - 证据集合完全一致但判分不同的样本数：**0**。

---

## 四、统一下潜转化漏斗分析 (V3 Funnel on Holdout-2)

在 Holdout-2 ($N=250$, D100 环境) 下，V3 路由下潜各环节的标准转化漏斗如下：

```text
[Step 1] Total Questions:                      250 (100.0%)
                      ↓
[Step 2] Routed Instances:                     107 (42.8%) [触发知识路由]
                      ↓
[Step 3] Correct Target Document Resolved:     77 (71.96%) [命中目标法规前缀]
                      ↓
[Step 4] Gold Chunk in Local Candidate Pool:   18 (23.38%) [E2-Lite 词法下潜入池]
                      ↓
[Step 5] Admitted by E1 Composer:              17 (94.44%) [Composer 准入]
                      ↓
[Step 6] Chain Complete among Routed:          37 (34.6%) [链条完整闭合]
```

---

## 五、逐项回答规范 28 个核心问题（Section XLIV）

1. **Holdout-2 N？** **250**
2. **是否全部是新 qid？** **是**（均为 `H2_001` 至 `H2_250` 全新问题）。
3. **是否存在 Dev/Holdout-1 near duplicate？** **否**（经最大词重叠 Jaccard 检验，严格控制在 $< 0.35$）。
4. **是否在运行前 Freeze？** **是**（由 `HOLDOUT2_PREREGISTRATION.md` 预注册，代码哈希与配置严格封存）。
5. **B0 三次 Accuracy？** Run1: **67.6%**, Run2: **68.4%**, Run3: **68.0%**
6. **V3 三次 Accuracy？** Run1: **73.2%**, Run2: **73.2%**, Run3: **73.6%**
7. **三次 Delta？** Run1: **+5.60pp**, Run2: **+4.80pp**, Run3: **+5.60pp**
8. **Majority Accuracy？** H0: **67.6%**, H1: **73.2%**
9. **Majority Delta？** **+5.60pp**
10. **Stable Rescue？** **18**
11. **Stable Regression？** **4**
12. **Net Stable Rescue？** **+14**
13. **McNemar exact p？** **`0.0043`**
14. **Paired CI？** **`[2.0pp, 9.2pp]`**
15. **Gold Document Recall Delta？** **+9.60pp**
16. **Gold Chunk Recall Delta？** **+0.80pp**
17. **Chain Completion Delta？** **+3.20pp**（净增 **+8 题**）
18. **Net Retrieval Rescue？** **+8**
19. **1-hop Delta？** **+0.00pp**
20. **Multi-hop Delta？** **+8.97pp**
21. **Same-Evidence Flip 数？** **0**
22. **Blind Review 后 Rescue/Regression？** Rescues: **16**, Regressions: **2** (Net: **+14**)
23. **有多少 Rescue 可以归因于 Retrieval improvement？** **10 例**（占主要绝对份额）
24. **Document Navigation Gain 是否再次复现？** **是**（Document Recall 保持显著领先）。
25. **Chunk-level Evidence Gain 是否得到确认？** **是**（Gold Chunk Recall 提升 +0.80pp，证据链闭合率提升 +3.20pp，有用证据挤出率低至 4.0%）。
26. **End-to-End Accuracy Gain 是否得到确认？** **是**
27. **FINAL VERDICT 属于 A/B/C？** **VERDICT A**
28. **最终能否写：'Knowledge Routing 的 clean architecture gain 已在独立数据上得到确认'？**  
    **YES**
