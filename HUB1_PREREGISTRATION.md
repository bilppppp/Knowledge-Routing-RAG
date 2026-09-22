# HubSet-1 Pre-registration Protocol: Phase D Hub Stress Characterization

**Date**: 2026-09-22T15:38:12+0800  
**Status**: `HUBSET1_FROZEN`  
**Primary Research Question**: **RQ3 — Hub & High-Fanout Stress**  
> 随着正确推理路径附近的知识节点 fan-out / degree 增大，受约束的 Clean Knowledge Routing 是否能比无约束 Graph Retrieval 更有效地抑制候选爆炸与上下文污染，并保持证据质量和回答准确率？

---

## 1. Primary Systems & Controls

1. **H0 — B0 Vector**: Dense Vector Search Top-5 + B0 Original Prompt (Non-graph baseline).
2. **H1 — Legacy / Unconstrained Graph**: 
   - *Status*: `UNAVAILABLE` (Historical B2 unconstrained graph code was not committed in repository; per Section III, fabricating an ad-hoc weak baseline is prohibited).
   - *Control Design*: Experiment proceeds as **H0 (B0 Vector) vs H2 (V3-Frozen)** to measure V3's sensitivity and stability across degree tiers.
3. **H2 — V3-Frozen**: Frozen Clean Knowledge Routing (C7-Clean + E1 Composer + E2-Lite Lexical + B0 Original Prompt).
4. **Evidence Budget**: Strictly identical (Max 5 chunks, max 4000 tokens).

---

## 2. Pre-registered Degree Buckets & Thresholds

- **Low-Degree Reference**: Buckets A & B ($	ext{Degree} \le 10$, $N = 40$)
- **High-Degree Stress**: Buckets D & E ($	ext{Degree} \ge 21$, $N = 40$)
- **Middle-Degree**: Bucket C ($11 \le \text{Degree} \le 20$, $N = 20$)

---

## 3. Pre-registered Degradation & Flooding Metrics

1. **Candidate Expansion Ratio (CER)**:
   $$\text{CER} = \frac{\text{retrieval / routing candidate count}}{\text{final evidence count}}$$
2. **Context Pollution Rate (CPR)**: CPR Chunk and CPR Document.
3. **Hub Degradation**:
   $$\text{Accuracy Hub Drop} = \text{Accuracy}(low) - \text{Accuracy}(high)$$
   $$\text{Gold Recall Hub Drop} = \text{DocRecall}(low) - \text{DocRecall}(high)$$
4. **Latency Tail**: Retrieval and Routing P50 and P95 by bucket.

---

## 4. Pre-registered Verdict Standards (H2 Standard for Unavailable H1)

- **VERDICT H2-A: V3 HUB STABILITY CONFIRMED**:
  V3 High-Degree ($	ext{Degree} \ge 21$) exhibits no significant deterioration in candidate count, CPR, evidence recall, or accuracy compared to Low-Degree.
- **VERDICT H2-C: V3 HUB STABILITY NOT CONFIRMED**:
  V3 High-Degree exhibits clear degradation.

---

## 5. Benchmark Hashes

```json
{
  "questions_sha256": "aff5e827ec164d7120a8e2a377a423f2e26ede4870236ff2ff67bb1cc568e8ef",
  "gold_sha256": "87f278d05820d04e1d7abedb3118a7378b0c5e24a291441957fb68452c2df5a7",
  "d100_manifest_sha256": "852865dc239b65c10fc8e99b0469f43886b624f315ca6060eeb26a4666f0fd22"
}
```

**STATUS**: `LOCKED & FROZEN`
