# Phase C — Scale & Distractor Robustness Confirmation Report

**Date**: 2026-09-22 15:17:03  
**Evaluated Benchmark**: `ScaleSet-1` ($N = 80$ unique QIDs)  
**Corpus Hierarchy**: $D_{20} \subset D_{50} \subset D_{100}$  
**Systems**: S0 (B0: Vector Top-5 + B0 Prompt) vs S1 (V3-Frozen: Clean Global Routing + E1 Composer + E2-Lite Lexical + B0 Prompt)  

---

# FINAL SCALE VERDICT:

```text
VERDICT S-C:
SCALE ROBUSTNESS NOT CONFIRMED
```

> **当前没有证据表明 Clean Knowledge Routing 比 Vector RAG 更能抵抗知识库扩张带来的干扰。**

---

## 一、核心表 1：端到端准确率与退化（Accuracy & Degradation）

*评测基于 3 轮独立生成多数表决（3-Run Majority），每轮包含独立的系统调用与判定。*

| 系统 | $D_{20}$ 准确率 | $D_{50}$ 准确率 | $D_{100}$ 准确率 | $D_{20} \to D_{50}$ Drop | $D_{50} \to D_{100}$ Drop | $D_{20} \to D_{100}$ 总退化 |
|---|---|---|---|---|---|---|
| **S0 — B0 (Vector RAG)** | 53.75% | 55.0% | 53.75% | -1.25pp | +1.25pp | **+0.00pp** |
| **S1 — V3-Frozen** | 58.75% | 55.0% | 56.25% | +3.75pp | -1.25pp | **+2.50pp** |
| **Delta (V3 - B0)** | +5.00pp | +0.00pp | +2.50pp | — | — | — |

### 核心规模鲁棒性收益（Robustness Advantage）:
$$\text{RA} = Drop_{B0} - Drop_{V3} = 0.00\text{pp} - 2.50\text{pp} = \mathbf{-2.50\text{pp}}$$

- **QID-Level Paired Bootstrap 95% CI** (10,000 resamples): **`[-6.25pp, +0.00pp]`**
- **3 轮单次独立 Run 结果**:
  - Run 1 (Seed 101): B0 Drop = +1.25pp, V3 Drop = +6.25pp, RA = -5.00pp
  - Run 2 (Seed 202): B0 Drop = +1.25pp, V3 Drop = +1.25pp, RA = +0.00pp
  - Run 3 (Seed 303): B0 Drop = +1.25pp, V3 Drop = +2.50pp, RA = -1.25pp

---

## 二、核心表 2：检索层指标与退化（Retrieval Degradation & Pollution）

| 评测指标 | 系统 | $D_{20}$ | $D_{50}$ | $D_{100}$ | $D_{20} \to D_{100}$ Drop |
|---|---|---|---|---|---|
| **Gold Document Recall** | B0 | 85.21% | 84.79% | 84.38% | +0.83pp |
| | V3 | 95.83% | 92.5% | 91.67% | +4.16pp |
| **Gold Chunk Recall** | B0 | 71.88% | 71.46% | 71.46% | +0.42pp |
| | V3 | 76.46% | 74.38% | 74.79% | +1.67pp |
| **Chain Completion Rate** | B0 | 52.5% | 52.5% | 52.5% | +0.00pp |
| | V3 | 60.0% | 57.5% | 57.5% | +2.50pp |
| **Evidence F1** | B0 | 36.16% | 35.85% | 35.85% | +0.31pp |
| | V3 | 39.15% | 37.81% | 38.12% | +1.03pp |
| **Context Pollution Rate (CPR Chunk)** | B0 | 74.75% | 75.0% | 75.0% | +0.25pp |
| | V3 | 72.5% | 73.5% | 73.25% | +0.75pp |
| **Context Pollution Rate (CPR Doc)** | B0 | 7.0% | 10.75% | 12.0% | +5.00pp |
| | V3 | 7.25% | 13.5% | 14.75% | +7.50pp |

---

## 三、核心表 3：规模转移分类（Scale Transition Matrix: $D_{20} \to D_{100}$）

| 转移类型 | 定义 | S0 — B0 计数 | S1 — V3 计数 | 差异 (V3 vs B0) |
|---|---|---|---|---|
| **Stable Correct** | $D_{20}$ 对 $\to D_{100}$ 对 | 43 | 45 | +2 |
| **Scale Regression** (关键安全指标) | $D_{20}$ 对 $\to D_{100}$ 错 | **0** | **2** | **+2 (V3 发生 2 例退化，B0 为 0 例)** |
| **Scale Rescue** | $D_{20}$ 错 $\to D_{100}$ 对 | 0 | 0 | +0 |
| **Stable Wrong** | $D_{20}$ 错 $\to D_{100}$ 错 | 37 | 33 | -4 |

---

## 四、候选空间与证据预算隔离分析（Candidate Space vs Evidence Context）

*验证架构设计承诺：内部候选空间允许扩展，但最终送入生成器的证据预算严格物理受控。*

| 指标 | 系统 | $D_{20}$ | $D_{50}$ | $D_{100}$ | 趋势说明 |
|---|---|---|---|---|---|
| **B0 检索候选数** | B0 | 5.0 | 5.0 | 5.0 | 固定 Top-5 |
| **V3 内部 Shadow 候选数** | V3 | 0.84 | 1.11 | 1.44 | 随规模自适应扩展 |
| **V3 路由候选池数** | V3 | 1.11 | 1.38 | 1.35 | 控制平面有序探索 |
| **最终证据块数 (Final Evidence)** | **B0 / V3** | **5.00 / 5.00** | **5.00 / 5.00** | **5.00 / 5.00** | **100% 严格一致** |

---

## 五、分层退化分析（Subgroup Degradation）

| 题目子集 | 样本数 $N$ | B0 $D_{20}$ | B0 $D_{100}$ | B0 Drop | V3 $D_{20}$ | V3 $D_{100}$ | V3 Drop | Subgroup RA |
|---|---|---|---|---|---|---|---|---|
| **1-Hop (Simple)** | 24 | 100.0% | 100.0% | +0.00pp | 100.0% | 100.0% | +0.00pp | **+0.00pp** |
| **Multi-Hop (2-Hop+)** | 56 | 33.93% | 33.93% | +0.00pp | 41.07% | 37.5% | +3.57pp | **-3.57pp** |
| **Hub-Adjacent** | 43 | 32.56% | 32.56% | +0.00pp | 41.86% | 39.53% | +2.33pp | **-2.33pp** |
| **Non-Hub** | 37 | 78.38% | 78.38% | +0.00pp | 78.38% | 75.68% | +2.70pp | **-2.70pp** |

---

## 六、延时与系统开销监测（Latency & Resource Consumption）

| 系统与 Corpus | 检索 P50 (ms) | 检索 P95 (ms) | 生成 P50 (ms) | 输入 Token 总计 | 输出 Token 总计 |
|---|---|---|---|---|---|
| **B0 × D20** | 161.73 | 247.09 | 4337.87 | 66334 | 55887 |
| **B0 × D50** | 3.71 | 5.09 | 4201.94 | 67038 | 57239 |
| **B0 × D100** | 3.3 | 4.86 | 4352.04 | 68494 | 57364 |
| **V3 × D20** | 3.66 | 37.66 | 4416.92 | 64293 | 58085 |
| **V3 × D50** | 0.52 | 17.86 | 4089.31 | 63838 | 58135 |
| **V3 × D100** | 0.47 | 17.04 | 4498.83 | 65682 | 58922 |

---

## 七、协议三十九问逐一明确回答（27 Mandatory Questions）

1. **ScaleSet N？**  
   答：**80** 个唯一 QID。
2. **是否所有 Gold 位于 D20？**  
   答：**是**，100% 的 Gold Documents 与 Gold Chunks 严格位于 D20。
3. **D50/D100 是否改变任何正确答案？**  
   答：**否**，经审核 D21–D100 中无任何改变或推翻 D20 结论的法规条文。
4. **是否在运行前冻结？**  
   答：**是**，生成并经完整数据审核后已固化哈希进入 `SCALESET1_FROZEN`。
5. **B0 D20 Majority Accuracy？**  
   答：**53.75%**。
6. **B0 D50？**  
   答：**55.0%**。
7. **B0 D100？**  
   答：**53.75%**。
8. **V3 D20？**  
   答：**58.75%**。
9. **V3 D50？**  
   答：**55.0%**。
10. **V3 D100？**  
   答：**56.25%**。
11. **B0 D20→D100 Drop？**  
   答：**+0.00pp**。
12. **V3 Drop？**  
   答：**+2.50pp**。
13. **Robustness Advantage (RA)？**  
   答：**-2.50pp**。
14. **95% CI？**  
   答：**`[-6.25pp, +0.00pp]`**。
15. **B0 Scale Regression 数？**  
   答：**0** 个。
16. **V3 Scale Regression 数？**  
   答：**2** 个。
17. **Gold Doc Recall degradation？**  
   答：B0 退化 **+0.83pp**（85.21%→84.38%），V3 退化 **+4.16pp**（95.83%→91.67%），V3 召回降幅大于 B0。
18. **Gold Chunk Recall degradation？**  
   答：B0 退化 **+0.42pp**，V3 退化 **+1.67pp**。
19. **Chain Completion degradation？**  
   答：B0 退化 **+0.00pp**，V3 退化 **+2.50pp**。
20. **CPR growth？**  
   答：B0 CPR 上升 **+0.25pp**，V3 CPR 上升 **+0.75pp**。
21. **Multi-hop degradation？**  
   答：Multi-hop 上 B0 退化 **+0.00pp**，V3 退化 **+3.57pp**，Multi-hop RA 为 **-3.57pp**。
22. **Simple-question degradation？**  
   答：1-hop Simple 上 B0 退化 **+0.00pp**，V3 退化 **+0.00pp**，1-hop RA 为 **+0.00pp**。
23. **Candidate count 是否随规模爆炸？**  
   答：**否**。内部 Candidate 虽有所扩展，但受限在受控常数范围，且最终证据块数被严格截断在 5 块。
24. **Final Evidence budget 是否始终一致？**  
   答：**是**，在所有 6 个实验条件下均为恰好 5 个 chunks、上限 4000 tokens。
25. **Blind adjudication 是否支持主要 Scale transitions？**  
   答：**是**，双盲复核确认 Scale Regression 为真实答案失真，而非裁判噪声。
26. **RQ2 最终 Verdict？**  
   答：**`VERDICT S-C`**。
27. **能否写“Clean Knowledge Routing 比纯 Vector RAG 更抗知识库扩张和 distractor pollution”？**  
   答：**`NO`**。
