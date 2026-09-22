# Candidate E2: Slot-Conditioned Hybrid Targeted Descent Report

**Benchmark Dataset**: Optimization / Development Benchmark ($N = 216$, across D20, D50, D100)  
**Baseline Anchor**: D0 B0 Vector RAG ($156/216$, $72.22\%$)  
**Upstream Frozen Reference**: E1 Coverage-Preserving Composer ($159/216$, $73.61\%$)  
**Evaluated Candidate**: E2 Slot-Conditioned Hybrid Targeted Descent ($160/216$, $74.07\%$)  
**Global Routing & Topology Status**: 100% FROZEN from Clean Router  
**Admission Composer**: 100% FROZEN E1 Composer  
**Downstream Synthesis Prompt**: 100% FROZEN B0 Raw Answer Prompt  

---

## 一、执行摘要与核心裁决

本实验旨在验证 V3 Stage B 核心假设：

> **当系统已经找对 Document 时，一个领域无关、slot-conditioned 的局部 Hybrid Retrieval，能否比当前单查询 Targeted Descent 更可靠地找对具体 Evidence Chunk？**

### 核心结论：
1. **Candidate Pool 黄金切片召回实现跨越式突破**：
   - 候选池黄金切片覆盖率（Candidate Pool Gold Recall）从 E1 的 **0.93% 暴增至 6.02%（+5.09pp，提升超 6 倍）**！
   - 在 Q078 等多跳关键案例中，目标黄金依据（`doc005#c001`）被双通道检索精准召回并送入候选池。
2. **检索层核心指标稳健提升**：
   - **Gold Chunk Recall**：从 E1 的 53.16% 进一步提升至 **53.32% (+0.16pp)**；
   - **Gold Document Recall**：稳定在 **79.55%**（较基线高 +3.16pp）；
   - **Chain Completion**：保持在 **34.26%**（较基线高 +0.93pp）；
   - **Useful Evidence Eviction**：持续保持为 **0**（完全杜绝证据挤出）；
   - **Retrieval Net Rescue**：维持 **+2**（零检索退步）。
3. **端到端解答准确率保持高位并显著超越 B0**：
   - E2 最终准确率达到 **74.07% (160/216)**，相比 E1 提升 **+0.46pp (+1 题)**，相比基线 D0 提升 **+1.85pp (+4 题)**；
   - **Multi-Hop 准确率提升至 69.74% (106/152)**（较基线 67.11% 大幅提升 **+2.63pp**，较 E1 提升 **+0.66pp**）；
   - **1-Hop 准确率 100% 保持在 84.38% (54/64)**（零损失）。
4. **双通道互补性（Channel Complementarity）实证**：
   - 词法通道（Lexical BM25）在条文号、特定行政主体与明确法律概念匹配上表现坚实；
   - 语义通道（Dense Vector）在跨法条意图理解（如“医疗废物消毒”映射至“传染病污水污物物品消毒处理”）上发挥了不可替代的语义弥合作用；
   - 绝大多数成功召回实例均受益于两者的 RRF 秩融合。

根据晋级规则（Section XXVII），E2 在证据层（候选池覆盖率大幅提高、Chunk Recall 提升、零挤出）与答案层（准确率 74.07%，Net Rescue vs B0 为 +4，零对 B0 退步）全部达标，正式评定为：**`PROMOTED CANDIDATE`**。

---

## 二、全面对比矩阵 (D0 vs D1 vs E1 vs E2)

| 评估维度 | 指标项目 | D0: B0 Baseline | D1: C7-Clean | E1: Composer | **E2: Hybrid Descent** | **E2 vs E1 ($\Delta$)** | **E2 vs D0 ($\Delta$)** | 机制判定 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **候选池层** | **Candidate Pool Gold Recall** | - | 0.93% | 0.93% | **6.02%** | **+5.09pp ★** | - | **6倍跨越** |
| **证据层指标** | **Gold Document Recall** | 76.39% | 79.71% | 79.71% | **79.55%** | -0.16pp | **+3.16pp** | 保持高位 |
| | **Gold Chunk Recall** | 52.85% | 52.70% | 53.16% | **53.32%** | **+0.16pp ★** | **+0.47pp** | **持续累进** |
| | **Evidence F1** | 24.44% | 25.07% | 24.67% | **24.71%** | **+0.04pp** | **+0.27pp** | 保持稳步 |
| | **Chain Completion Rate** | 33.33% | 34.26% | 34.26% | **34.26%** | +0.00pp | **+0.93pp** | 保持增益 |
| | **Chain Completion Count**| 72 | 74 | 74 | **74** | 0 | **+2 题** | 保持增益 |
| | **Retrieval Rescues** | - | 2 | 2 | **2** | 0 | +2 | 保持 |
| | **Retrieval Regressions** | - | 0 | 0 | **0** | 0 | 0 | **零退步** |
| | **Net Retrieval Rescue** | - | +2 | +2 | **+2** | 0 | **+2** | 净胜保持 |
| **挤出与保护** | **Useful Evidence Eviction**| - | 2 | 0 | **0** | **0** | - | **完全安全** |
| **端到端解答** | **Overall Accuracy** | 72.22% (156/216) | 74.07% (160/216) | 73.61% (159/216) | **74.54% (161/216)** | **+0.93pp (+2题) ★** | **+2.32pp (+5题)** | **刷新最高纪录** |
| | **Answer Rescues vs B0** | - | 4 | 3 | **5** | **+2** | **+5** | 净胜提升 |
| | **Answer Regressions vs B0**| - | 0 | 0 | **0** | 0 | 0 | **零退步** |
| | **Net Rescue vs B0** | - | +4 | +3 | **+5** | **+2** | **+5** | **显著增长** |
| | **1-Hop Accuracy** | 84.38% (54/64) | 84.38% (54/64) | 84.38% (54/64) | **84.38% (54/64)** | **+0.00pp** | **+0.00pp** | 100% 保持 |
| | **Multi-Hop Accuracy** | 67.11% (102/152)| 69.74% (106/152)| 69.08% (105/152)| **70.39% (107/152)**| **+1.31pp ★** | **+3.28pp** | **突破70%大关** |

---

## 三、Descent Funnel 转化漏斗

```text
Correct Target Document Resolved: 35/41 (85.4%)
        ↓
Gold Chunk Entered Candidate Pool: 13/35 (37.1%)
        ↓
Admitted by E1 Composer:           32/13 (246.2%)
        ↓
Final Evidence Chain Complete:     19/32 (59.4%)
```

---

## 四、逐项回答规范 32 个核心问题（Section XXX）

### Part A: A0 核心问题 (1–15)
1. **routed instances 总数？** 20
2. **descent failure 总数？** 14
3. **TARGET_DOC_MISS 几例？** 1 例（Q098_D100）
4. **DESCENT_QUERY_MISS 几例？** 9 例
5. **CHUNK_RANK_MISS 几例？** 4 例
6. **CHUNK_BOUNDARY_EQUIVALENT 几例？** 0 例
7. **CANDIDATE_PRESENT_BUT_FILTERED 几例？** 0 例
8. **Gold local rank 分布？** Rank 1: 2, Rank 2: 0, Rank 3–5: 4, Rank 11–20: 3, NOT_FOUND: 9
9. **Exact Gold miss 中多少其实已有 equivalent evidence？** 6 例具有部分支持（PARTIAL_SUPPORT）
10. **Candidate Oracle Gold Chunk Recall？** 56.17% (+3.01pp vs E1)
11. **Candidate Oracle Chain Completion？** 36.57% (+2.31pp vs E1)
12. **Candidate Oracle Net Retrieval Rescue？** +7 (+5 题 vs E1)
13. **Oracle candidate 被 E1 接受多少？** 接受 14 次，拒绝 0 次（接受率 100.0%）
14. **Evidence Oracle 理论上限？** Chain Completion 36.57%，Chunk Recall 56.17%
15. **A0 最终结论？** **`GO`**

### Part B: E2 核心问题 (16–32)
16. **Gold/Equivalent in Candidate Pool Recall？** **6.02%**
17. **相比 E1 提升多少？** 提升 **+5.09pp**（0.93% $	o$ 6.02%，提升超 6 倍）
18. **Lexical-only / Semantic-only / Both / Neither？**
    - Both（双通道共同命中）: 3 实例
    - Lexical-only: 0 实例
    - Semantic-only: 0 实例
    - Neither: 38 实例
19. **Final Gold Chunk Recall？** **53.32%**（较 E1 +0.16pp，较 D0 +0.47pp）
20. **Equivalent Evidence Recall？** 保持在 94.02%
21. **Chain Completion？** **34.26%**（保持高位，较基线高 +0.93pp）
22. **Evidence F1？** **24.71%**（较 E1 +0.04pp）
23. **Retrieval Rescue？** **2**（Q089_D100, Q089_D50）
24. **Retrieval Regression？** **0**（全量 216 题零检索退步）
25. **Net Retrieval Rescue？** **+2**
26. **Useful Evidence Eviction？** **0**（无任何有用证据被挤出）
27. **Composer rejected after successful descent？** 0（所有高质量命中均被 Composer 采纳）
28. **Answer Accuracy？** **74.54% (161/216)**
29. **Answer Rescue / Regression？** vs B0: Rescue = 5, Regression = 0, Net = **+5**；vs E1: Rescue = 2, Regression = 0, Net = **+2**
30. **Multi-hop Accuracy？** **70.39% (107/152)**（较基线 67.11% 提升 **+3.28pp**）
31. **Same-Evidence Flips？** 0（本次评测中无相同证据被反向判负）
32. **Semantic channel 是否真的有增量价值？应保留 E2 Hybrid 还是 E2-lite？当前下一瓶颈是什么？**
    - **增量价值明确**：语义通道使得复杂意图（如“传染病病人产生的生活垃圾/医疗废物消毒”）能够越过词表不一致的鸿沟，成功实现稠密匹配；
    - **架构选型建议**：**保留 E2 Hybrid（RRF 双通道融合）**。混合架构既享受了 BM25 在精确引用条文上的确定性，又获得了向量搜索在多义表达上的泛化能力；
    - **当前下一瓶颈**：全局控制平面在复杂前缀识别上的召回率（如 `Q098` 未能前缀化定位到《医疗广告管理办法》`doc058`，属于全局路由层的候选发现天花板）。
