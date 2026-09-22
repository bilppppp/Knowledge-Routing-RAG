# ScaleSet-1 Pre-registration Protocol: Phase C Robustness Confirmation

**Date**: 2026-09-22T15:05:33+0800  
**Status**: `SCALESET1_FROZEN`  
**Primary Research Question**: **RQ2 — Scale & Distractor Robustness Confirmation**  
> 当答案和 Gold Evidence 完全保持不变，仅扩大知识库并增加无关/近邻干扰文档时（$D_{20} \subset D_{50} \subset D_{100}$），Frozen V3 是否比纯 Vector RAG (B0) 更抗退化？

---

## 1. Experimental Setup & Frozen Conditions

### 1.1 Evaluated Systems (100% Permanently Frozen)
- **S0 (B0)**: Dense Vector Search (Top-5) + Raw B0 Answer Prompt.
- **S1 (V3-Frozen)**: Clean Global Routing + E1 Coverage Composer + E2-Lite Lexical Descent + Raw B0 Answer Prompt.
- **Evidence Budget**: Identical across systems and corpora (Max 5 chunks, max 4000 tokens).

### 1.2 Corpora Suite
- $D_{20}$: 20 Core Health & Medicine statutes (1,865 chunks).
- $D_{50}$: 50 Statutes & regulations ($D_{20} + 30$ distractors, 2,807 chunks).
- $D_{100}$: 100 Statutes, regulations & administrative replies ($D_{50} + 50$ distractors, 2,862 chunks).
- **Invariance Rule**: All $N=80$ ScaleSet-1 questions have their gold evidence strictly in $D_{20}$. Question, gold evidence, and gold answer are $100\%$ invariant. Only distractor scale expands.

---

## 2. Primary Metrics & Statistical Protocol

### 2.1 Primary Degradation Metric
- **$Drop_{B0} = \text{Accuracy}_{B0}(D_{20}) - \text{Accuracy}_{B0}(D_{100})$**
- **$Drop_{V3} = \text{Accuracy}_{V3}(D_{20}) - \text{Accuracy}_{V3}(D_{100})$**
- **Robustness Advantage (RA)**:
  $$\text{RA} = Drop_{B0} - Drop_{V3}$$
- **Primary Statistical Test**: QID-level Paired Difference-in-Differences Bootstrap (10,000 resamples), computing the two-sided 95% Confidence Interval.

### 2.2 Scale Transitions
For each unique QID, correctness transition across $D_{20} \to D_{100}$:
- **Stable Correct**: Correct in $D_{20}$ and correct in $D_{100}$.
- **Scale Regression**: Correct in $D_{20}$ but wrong in $D_{100}$ (Primary Failure Mode).
- **Scale Rescue**: Wrong in $D_{20}$ but correct in $D_{100}$.
- **Stable Wrong**: Wrong in both $D_{20}$ and $D_{100}$.

### 2.3 Retrieval Degradation Metrics
- Gold Document Recall degradation ($D_{20} \to D_{100}$)
- Gold Chunk Recall degradation ($D_{20} \to D_{100}$)
- Chain Completion degradation ($D_{20} \to D_{100}$)
- Context Pollution Rate (CPR) growth:
  $$\text{CPR} = \frac{\text{distractor evidence chunks}}{\text{total final evidence chunks}}$$

---

## 3. Pre-registered Verdict Rules

Only one of the following three verdicts will be rendered:

### VERDICT S-A: SCALE ROBUSTNESS CONFIRMED
Requires:
1. $\text{RA} > 0$ and Paired Bootstrap 95% CI strictly $> 0$.
2. $\text{V3 Scale Regressions} < \text{B0 Scale Regressions}$.
3. At least one primary retrieval degradation metric (Gold Doc Recall, Gold Chunk Recall, Chain Completion, or CPR slope) confirms smaller degradation for V3.

### VERDICT S-B: END-TO-END ROBUSTNESS NOT CONFIRMED, RETRIEVAL ROBUSTNESS CONFIRMED
Triggered if:
- Retrieval degradation is consistently smaller for V3, BUT
- Accuracy RA 95% CI crosses zero.

### VERDICT S-C: SCALE ROBUSTNESS NOT CONFIRMED
Triggered if:
- $\text{RA} \le 0$, OR
- $\text{V3 Scale Regressions} \ge \text{B0 Scale Regressions}$, with no consistent retrieval improvement.

---

## 4. Benchmark Hashes & Execution Integrity

```json
{
  "questions_sha256": "510ad0768746069a58729619bace988d53252b3c535f9ed71522eba38df396c8",
  "gold_sha256": "1fd9c31a6521c4db76cb8ffcba150b6ec279da232dc30160b727774f3fec652f",
  "d20_manifest_sha256": "e0e6b19097853c2856d86bc0b3ee8ac982a589ca1b6a256f0941f05b3f8e97e4",
  "d50_manifest_sha256": "9673555854eb93180fbb530af50cd7a97b17c3ac25d7b5ae6071744b79c9a807",
  "d100_manifest_sha256": "852865dc239b65c10fc8e99b0469f43886b624f315ca6060eeb26a4666f0fd22"
}
```

**PRE-REGISTRATION STATUS**: `LOCKED & FROZEN`
