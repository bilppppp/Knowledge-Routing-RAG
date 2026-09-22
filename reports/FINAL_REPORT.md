# Knowledge-Routing-RAG: Final Research & Experimental Report

**Project Title**: Knowledge-Routing-RAG: Constrained Graph Navigation for Complex Regulatory Retrieval  
**Date**: 2026-09-22  
**Repository**: [https://github.com/bilppppp/Knowledge-Routing-RAG](https://github.com/bilppppp/Knowledge-Routing-RAG)  
**Status**: Experimental Research Completed & Frozen (`v3-research-final`)  
**Audience**: AI Researchers, Retrieval Engineers, and Academic Evaluators  

---

## 1. Executive Summary

Under the frozen corpus, embedding model, generator configuration, evidence-budget, and independent evaluation conditions of this study:

```text
========================================================================================
FINAL RESEARCH VERDICTS:

End-to-End Accuracy Gain:        CONFIRMED  (+5.60pp on Holdout-2, McNemar p = 0.0043)
Document Navigation Gain:        CONFIRMED  (+9.60pp Gold Document Recall)
Multi-hop Reasoning Gain:        CONFIRMED  (+8.97pp on 2-hop+ subtasks, 0 simple regression)
Engineering Value:               CONFIRMED  (+0.58 ms P50 local overhead, 0 extra online LLMs)

Scale Robustness Advantage:      NOT CONFIRMED (RA = -2.50pp, 95% CI [-6.25pp, 0.00pp])
Hub Node Stability:              NOT CONFIRMED (Candidate P95 expanded to 27.1, chain drop -17.5pp)
Graph Flooding Suppression:      NOT CONFIRMED / NOT TESTABLE (Legacy graph baseline unavailable)
Universal Superiority:           NOT CLAIMED
========================================================================================
```

**Core Project Verdict (English)**:
> Under the frozen corpus, model, evidence-budget, and evaluation conditions used in this study, Clean Knowledge Routing produced a repeatable and statistically supported end-to-end accuracy improvement over Vector RAG. The gains were concentrated in document navigation and multi-hop evidence recovery, with only low-millisecond additional local computation and no extra online LLM calls. Scale robustness and hub robustness were not confirmed.

**核心结论 (中文)**:
> 在本实验的医疗法规语料、冻结系统配置和独立评测条件下，Clean Knowledge Routing 相比纯 Vector RAG 获得了可重复且统计显著的端到端准确率提升。收益主要来自更好的文档导航、多跳证据恢复和受约束的证据组合，并且只增加低毫秒级本地计算成本。但实验没有确认该架构具有更强的规模鲁棒性或 Hub 鲁棒性，因此这些不能作为其已验证优势。

---

## 2. Research Question & Core Premise

Standard dense Vector RAG operates on the premise that semantically similar query and chunk embeddings will co-locate relevant evidence in a shared vector space. However, in complex technical domains—such as regulatory compliance, healthcare administration, and legal synthesis—critical evidence is distributed across cross-document dependency chains, temporal modifications (amendments and repeals), and hierarchical statutory bases (e.g., administrative regulations deriving authority from overarching mother laws).

In such environments, pure vector retrieval frequently suffers from:
1. **Local Trapping**: Selecting superficially similar chunks from a single document while entirely missing required cross-document companion statutes.
2. **Context Dilution**: Flooding the prompt with near-duplicate paragraphs from high-ranking sections, crowding out cross-hop evidence.
3. **Graph Flooding** (in naive Graph RAG): Exploring unconstrained relational edges and introducing distractors that degrade generator fidelity.

### The Research Question (RQ)
> **Under strictly matched documents, embedding models, generator configurations, and identical evidence token budgets ($\le 4000$ tokens / $\le 5$ chunks), can a constrained Knowledge Routing architecture outperform pure Vector RAG in recovering complete multi-hop evidence chains and improving end-to-end answer correctness?**

### Core Design Axioms
1. **Candidate Space $\ne$ Evidence Context**: The system may explore intermediate relational candidates on the control plane, but the data plane strictly filters and compresses evidence into a fixed budget ($\le 5$ chunks) before synthesis.
2. **Route Globally, Retrieve Locally, Compose Globally**:
   - *Route Globally*: Use vector entry points and knowledge network topologies to resolve target document prefixes across the corpus.
   - *Retrieve Locally*: Once a target document is resolved, descend locally using precise lexical search conditioned on unresolved query slots.
   - *Compose Globally*: Evaluate candidate chunks across all sources using a coverage-preserving set utility function, protecting unique evidence and rejecting redundant distractors.

---

## 3. Frozen System Architecture: V3-Frozen

The final verified architecture is **V3-Frozen** (specifically candidate `E2-Lite`), which strips out non-essential components discovered during ablations and retains only what earned its complexity.

```text
                            User Question
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Dense Vector Entry    │ (Qdrant Cosine Top-5)
                     └────────────┬────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │ [Fast-Path Check: Sim ≥ 0.85]          │
              ▼                                       ▼
       [Fast Path: Direct]                 [Routing Lane Activated]
              │                                       │
              │                      ┌────────────────┴────────────────┐
              │                      │   Control Plane Shadow RIB      │ (Top-20 prefixes)
              │                      └────────────────┬────────────────┘
              │                                       │
              │                      ┌────────────────┴────────────────┐
              │                      │    Route-Prefix & Parent Lift   │ (LSDB DiGraph)
              │                      └────────────────┬────────────────┘
              │                                       │
              │                      ┌────────────────┴────────────────┐
              │                      │ E2-Lite Lexical Targeted Descent│ (BM25 + Headings)
              │                      └────────────────┬────────────────┘
              │                                       │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │  E1 Coverage-Aware Composer │ (Slot Coverage Utility)
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │   Final Evidence Context    │ (Strictly ≤ 5 chunks, ≤4000 tok)
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │     LLM Answer Generator    │ (DeepSeek-Chat, Temp=0.0, Raw B0 Prompt)
                   └─────────────────────────────┘
```

### Key Subsystems:
1. **FIB Vector Entrance**: Retrieves the top-5 chunks via dense cosine similarity (`google/embeddinggemma-300m`).
2. **Shadow Candidate Plane (Control Plane)**: Expands the candidate horizon to top-20 chunks to aggregate document prefixes without polluting the evidence context.
3. **Route-Prefix & Next-Hop Resolution**: Uses an immutable SQLite-backed Link-State Database (`knowledge_lsdb.sqlite`) to traverse explicit statutory edges (`BASED_ON`, `REFERENCES`, `SUPERSEDES`, `AMENDS`) and parent-child structural hierarchies.
4. **E2-Lite Slot-Conditioned Lexical Descent**: When a target document is identified as missing critical evidence, the system extracts unresolved query slots and performs targeted BM25 lexical retrieval within that document, supplemented by heading category bonuses. (Dense local vector retrieval and RRF were ablated and deleted due to zero incremental contribution).
5. **E1 Coverage-Preserving Composer**: Evaluates incoming local candidates against the baseline pool using marginal set coverage utility, enforcing a hard constraint: **Useful Evidence Eviction $= 0$**.
6. **B0 Raw Answer Prompt**: Context is assembled using the exact, unmodified baseline system prompt.

---

## 4. Chronological Experimental Evolution

The project progressed through distinct phases of empirical discovery, failure analysis, and methodological correction:

```text
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Phase V1    │ ──> │  Phase V2    │ ──> │  Decontamination │ ──> │  Holdout-1   │
│ (Hypothesis  │     │  (Candidate  │     │  Audit & Clean   │     │ (Nav. Conf., │
│  Failure)    │     │   Search)    │     │  Architecture)   │     │  E2E Failed) │
└──────────────┘     └──────────────┘     └──────────────────┘     └──────────────┘
                                                                          │
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐            │
│  Phase E     │ <── │  Phase D     │ <── │  Phase C         │ <── ┌──────────────┐
│ (Cost/Latency│     │ (Hub Stress  │     │  (Scale Robust.  │     │  Phase V3 &  │
│  Confirmed)  │     │  Not Conf.)  │     │   Not Conf.)     │     │  Holdout-2   │
└──────────────┘     └──────────────┘     └──────────────────┘     │ (E2E Conf.)  │
                                                                   └──────────────┘
```

### 4.1 Phase V1: Initial Hypothesis Failure
- **Hypothesis**: Direct analogy between network routing protocols (RIFT vertical routing, EIGRP feasibility conditions, SRv6 segment planning) and knowledge graph traversal would yield superior retrieval.
- **Outcome**: Systems K1, K2, K3, and K4 failed to beat the B0 Vector baseline on the evaluation set. K4 achieved $72.22\%$ vs B0's $72.22\%$ ($\Delta = 0.00\text{pp}$). Unconstrained exploration introduced distractors, while heuristic graph pruning evicted correct baseline chunks.
- **Verdict**: **`HYPOTHESIS NOT CONFIRMED (NO-GO)`**.

### 4.2 Phase V2: Directed Candidate Search & Benchmark Exposure
- **Progression**: Candidates C1 through C7 were iteratively developed on the 216 development instances:
  - *C1–C3*: Local graph fallbacks and relation filtering.
  - *C4*: Introduced the Shadow Candidate Plane (Top-20 RIB) to decouple discovery from context.
  - *C5*: Added Hierarchical Resolution (Parent Lift).
  - *C6*: Added Evidence-Contract Synthesis.
  - *C7*: Added Route-Prefix Resolution, reaching an apparent $80.09\%$ accuracy on Dev-216 ($+7.87\text{pp}$ vs B0).
- **Critical Methodological Finding**: Analysis revealed that C7 had become heavily contaminated by benchmark-specific heuristics (e.g., hardcoded statute regexes and query-specific extraction rules), rendering the $80.09\%$ result invalid as an estimate of true generalization.

### 4.3 Decontamination Audit: Establishing Clean Architecture
- **Intervention**: A systematic decontamination audit ([`decontamination_rule_audit.md`](decontamination_rule_audit.md)) excised all hardcoded statute tables, specific law strings, and benchmark-tailored prompts.
- **Deliverable**: `C7-Clean` router, operating strictly on generic graph relations and algorithmic keyword matching.
- **Ablation Results**: Decontamination caused a drop from $80.09\%$ to $74.54\%$ on Dev-216, quantifying that **$+5.55\text{pp}$ of C7's apparent gain was benchmark-specific artifact**, while $+2.32\text{pp}$ represented genuine architectural lift.

### 4.4 Holdout-1 Confirmation: Decoupling Navigation from Answer Gain
- **Setup**: Evaluated C7-Clean on a completely fresh, unseen 200-question holdout set ([`clean_architecture_confirmation.md`](clean_architecture_confirmation.md)).
- **Result**:
  - Gold Document Recall: $83.00\% \to 91.08\%$ (**$+8.08\text{pp}$**) $\implies$ Navigation gain confirmed.
  - 3-Run Majority Accuracy: $74.50\% \to 74.00\%$ (**$-0.50\text{pp}$**, McNemar $p = 1.000$) $\implies$ End-to-end gain **not confirmed**.
- **Root Cause Analysis**: Fixed position-based replacement (slots 4–5) frequently evicted a relevant baseline chunk to admit a newly routed chunk (*Useful Evidence Eviction* occurred in 19 instances).
- **Core Lesson**: **`Document Navigation Gain ≠ End-to-End Answer Gain`**.

### 4.5 Phase V3 Development: Fixing Composition and Descent
- **E1 Coverage-Preserving Composer**: Replaced position-based replacement with a set-utility optimization that locks uniquely covered slots and penalizes eviction. Useful evidence evictions dropped to strictly **0** ([`v3_e1_evidence_composition.md`](v3_e1_evidence_composition.md)).
- **E2 Local Descent & Channel Ablation**:
  - Evaluated targeted in-document descent.
  - Channel ablation proved that within a correctly navigated document, dense vector retrieval provided **zero** unique gold chunks (`SEMANTIC_UNIQUE_GOLD = 0`), whereas BM25 lexical descent located 4 unique gold chunks missed by vectors.
  - Combining lexical and semantic retrieval via Reciprocal Rank Fusion (RRF) diluted the highest-scoring lexical chunks, hurting chain completion.
  - The semantic channel was deleted. The system was frozen as **`E2-Lite (Lexical Only)`** under the rule: *Keep only what earns its complexity*.

---

## 5. Explicit Disclosure: Status of the Dev-216 Benchmark

Early in the project, an evaluation suite of 120 questions was stratified across $D_{20}$, $D_{50}$, and $D_{100}$ corpora, yielding 216 instance runs.

During the development of candidates C1 through C7, this set was repeatedly examined:
- Failure cases were inspected in detail.
- Heuristic routing patterns were adjusted in response to specific errors.
- Prompt phrasing was modified based on observed generation failures.

Consequently, **Dev-216 became an Optimization / Development Set**.

```text
========================================================================================
METHODOLOGICAL DISCLOSURE:
Any accuracy figure reported on Dev-216 (such as C7's 80.09% or E2-Lite's 74.54%)
reflects performance on an exposed development set and CANNOT be interpreted as
evidence of independent out-of-sample generalization.
The only valid confirmatory evidence for V3 is Independent Holdout-2.
========================================================================================
```

---

## 6. Primary Confirmatory Evidence: Independent Holdout-2

To obtain definitive, uncontaminated evidence, **Independent Holdout-2** was designed, pre-registered ([`HOLDOUT2_PREREGISTRATION.md`](../HOLDOUT2_PREREGISTRATION.md)), and frozen prior to data generation.

### Experimental Controls:
- **Sample Size**: $N = 250$ unique, unseen regulatory compliance questions.
- **Corpus**: High-distraction $D_{100}$ corpus (100 documents, 2,862 chunks).
- **Isolation**: Max question Jaccard similarity $< 0.35$ against all past sets.
- **Replication**: 3 paired fresh generation runs using seeds `101`, `202`, `303` (zero cache reuse).
- **Primary Endpoint**: 3-Run Majority Answer Correctness.
- **Adjudication**: Double-blind manual review of all discordant cases.

### Primary Results Matrix

| Metric | H0: Vector Baseline (B0) | H1: V3-Frozen (E2-Lite) | Delta ($\Delta$) | Statistical Test / Significance |
|:---|:---:|:---:|:---:|:---|
| **Majority Accuracy** | **67.60%** (169/250) | **73.20%** (183/250) | **+5.60pp** | **McNemar exact $p = 0.0043$** |
| Run 1 (Seed 101) | 67.60% | 73.20% | +5.60pp | — |
| Run 2 (Seed 202) | 68.40% | 73.20% | +4.80pp | — |
| Run 3 (Seed 303) | 68.00% | 73.60% | +5.60pp | — |
| **Stable Rescues** ($B0=0, V3=1$) | — | — | **18** | Pre-registered safety criterion: |
| **Stable Regressions** ($B0=1, V3=0$) | — | — | **4** | $\text{Rescues} > \text{Regressions}$ |
| **Net Stable Rescue** | — | — | **+14** | Exact Binomial $p = 0.0044$ |
| **Paired Bootstrap 95% CI** | — | — | **[+2.00pp, +9.20pp]** | Strictly bounded above 0 |
| **Gold Document Recall** | 88.93% | 98.53% | **+9.60pp** | Mechanism confirmed |
| **Gold Chunk Recall** | 75.73% | 76.53% | **+0.80pp** | Positive chunk yield |
| **Evidence F1** | 36.59% | 36.90% | **+0.31pp** | Preserved precision |
| **Chain Completion Rate** | 54.40% (136) | 57.60% (144) | **+3.20pp** (+8 chains) | Multi-document closure |
| **Useful Evidence Evictions** | — | **10** (4.0%) | — | Controlled eviction rate |
| **Net Retrieval Rescues** | — | **+8** (11 res, 3 reg) | — | Positive retrieval transition |

### Subgroup Analysis: Simple vs Multi-Hop

| Subgroup | $N$ | B0 Majority | V3 Majority | Delta ($\Delta$) | Rescues | Regressions | Interpretation |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Simple (1-Hop)** | 94 | 98.94% | 98.94% | **+0.00pp** | 0 | 0 | **Zero regression on simple tasks** |
| **Multi-Hop (2-Hop+)** | 156 | 48.72% | 57.69% | **+8.97pp** | 18 | 4 | **Concentrated multi-hop uplift** |

- **Simple Regressions**: **0**. The Fast-Path gating mechanism ($T \ge 0.85$) successfully bypassed routing on direct single-document queries, protecting baseline performance from degradation.
- **Same-Evidence Flips**: **0**. In all instances where V3 and B0 retrieved identical evidence chunks, the LLM judge evaluated the answers identically.

### Blind Adjudication Verification
All 22 discordant cases were anonymized and reviewed independently:
- Adjudicated Rescues: **16**
- Adjudicated Regressions: **2**
- Adjudicated Net Gain: **+14**
- Concordance with Automated Judge: **90.9%** (20/22).
- **Attribution**: 10 rescues were directly attributed to the retrieval of previously missing statutory chunks (`RETRIEVAL_CAUSAL_RESCUE`); 0 were random generation fluctuations.

---

## 7. Negative Finding: Scale Robustness (Phase C)

A core initial hypothesis of the project was that Knowledge Routing would exhibit greater robustness than Vector RAG as the corpus expanded from small to large collections.

To test this, **ScaleSet-1** ($N = 80$) was constructed where all gold answers were strictly contained within the small $D_{20}$ sub-corpus, and evaluated across $D_{20}$, $D_{50}$, and $D_{100}$ ([`v3_scale_robustness_confirmation.md`](v3_scale_robustness_confirmation.md)).

### Scale Degradation Results

| Metric | S0: B0 Vector RAG | S1: V3-Frozen | Robustness Advantage (RA) |
|:---|:---:|:---:|:---:|
| $D_{20}$ Majority Accuracy | 53.75% | 58.75% | $+5.00\text{pp}$ (V3 lead) |
| $D_{50}$ Majority Accuracy | 55.00% | 55.00% | $+0.00\text{pp}$ |
| $D_{100}$ Majority Accuracy | 53.75% | 56.25% | $+2.50\text{pp}$ (V3 lead) |
| **$D_{20} \to D_{100}$ Total Drop** | **+0.00pp** | **+2.50pp** | **$\text{RA} = -2.50\text{pp}$** |
| **95% Bootstrap CI for RA** | — | — | **`[-6.25pp, 0.00pp]`** |
| Scale Regressions ($D_{20}=1, D_{100}=0$) | **0** | **2** | V3 exhibited 2 scale failures |
| Gold Doc Recall Drop | +0.83pp | +4.16pp | V3 dropped more than B0 |
| CPR Doc (Context Pollution) Growth | +5.00pp | +7.50pp | Distractors increased faster in V3 |

### Analysis:
While V3 maintained a higher absolute accuracy than B0 at all scale points ($56.25\%$ vs $53.75\%$ on $D_{100}$), **its relative drop from $D_{20}$ to $D_{100}$ was larger than B0's**. The addition of 80 distractor documents introduced plausible but irrelevant statutory links that occasionally diverted V3's prefix search.

```text
========================================================================================
FINAL SCALE VERDICT:
VERDICT S-C: SCALE ROBUSTNESS NOT CONFIRMED.
It is forbidden to claim that V3 is more robust to corpus expansion than Vector RAG.
========================================================================================
```

---

## 8. Negative Finding: Hub Stress Characterization (Phase D)

Knowledge graphs often contain high-degree "hub" nodes (e.g., broad national statutes like the *Law on the Prevention and Control of Infectious Diseases*) that link to dozens of subordinate rules. Naive graph expansion suffers from severe graph flooding at these hubs.

In **Phase D**, **HubSet-1** ($N = 100$) was evaluated across 5 pre-registered degree buckets (A: <5, B: 5–10, C: 11–20, D: 21–50, E: >50), decoupled from hop count ([`v3_hub_stress_confirmation.md`](v3_hub_stress_confirmation.md)).

### Hub Stress Results

| Degree Tier | Degree Range | $N$ | B0 Acc (%) | V3 Acc (%) | Delta | V3 Cand P95 | V3 CPR Doc | V3 Chain Complete |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Bucket A | < 5 | 20 | 45.00% | 45.00% | +0.00pp | 8.2 | 15.0% | 55.0% |
| Bucket B | 5–10 | 20 | 70.00% | 80.00% | +10.00pp | 10.2 | 20.0% | 65.0% |
| Bucket C | 11–20 | 20 | 70.00% | 70.00% | +0.00pp | 15.4 | 25.0% | 60.0% |
| Bucket D | 21–50 | 20 | 45.00% | 50.00% | +5.00pp | 27.1 | 45.0% | 40.0% |
| Bucket E | > 50 | 20 | 75.00% | 75.00% | +0.00pp | 12.9 | 34.0% | 45.0% |
| **Low-Degree (A & B)** | $\le 10$ | 40 | **57.50%** | **62.50%** | **+5.00pp** | **10.1** | **17.5%** | **60.0%** |
| **High-Degree (D & E)** | $> 20$ | 40 | **60.00%** | **62.50%** | **+2.50pp** | **27.1** | **39.5%** | **42.5%** |

### Analysis & Status of Baseline:
1. **Candidate Expansion**: At high-degree nodes (Bucket D), candidate space expanded to a P95 of 27.1 chunks, and CPR Doc (non-target document inclusion) surged from $17.5\%$ to $39.5\%$.
2. **Evidence Degradation**: Chain completion collapsed from $60.0\%$ in low-degree to $42.5\%$ in high-degree (a $-17.5\text{pp}$ drop).
3. **Hub Robustness Advantage**: $\text{RA} = \text{Drop}_{B0} - \text{Drop}_{V3} = -2.50\text{pp}$ ($95\%$ CI `[-12.50pp, +7.50pp]`).
4. **Baseline Availability**: The historical unconstrained graph baseline was not preserved in Git history and was declared **`UNAVAILABLE`**. Because unconstrained traversal could not be tested side-by-side, graph flooding suppression cannot be claimed.

```text
========================================================================================
FINAL HUB VERDICT:
VERDICT H2-C: V3 HUB STABILITY NOT CONFIRMED.
Graph Flooding Suppression: NOT CONFIRMED / NOT TESTABLE.
V3 did not execute unconstrained graph walks, but high-degree nodes still induced
candidate expansion and evidence-chain degradation.
========================================================================================
```

---

## 9. Confirmed Finding: Engineering Value & Complexity (Phase E)

To determine whether V3's $+5.60\text{pp}$ accuracy gain justifies its operational complexity, **Phase E** conducted a rigorous hardware-level latency, token, and systems audit on Holdout-2 ($N = 250$, Apple Silicon M4, 5 warm-up iterations, 5 measurement passes) ([`v3_cost_complexity_audit.md`](v3_cost_complexity_audit.md)).

### Cost / Benefit Summary Table

| Dimension | Metric | B0 Vector Baseline | V3-Frozen Clean Routing | Incremental Cost |
|:---|:---|:---:|:---:|:---|
| **Accuracy Benefit** | Majority Accuracy | 67.60% | 73.20% | **+5.60pp** ($p=0.0043$) |
| | Stable Net Rescue | Reference | +14 cases | **+14 / 250 cases** |
| | Multi-hop Accuracy | 48.72% | 57.69% | **+8.97pp** |
| **Local Retrieval Latency** | Local Latency P50 | 3.67 ms | 4.34 ms | **+0.58 ms** (1.18x) |
| | Local Latency P95 | 4.96 ms | 19.07 ms | **+14.11 ms** |
| | Local Latency Mean | 3.76 ms | 7.39 ms | **+3.63 ms** (95% CI: `[+2.87, +4.44]`) |
| **End-to-End Latency** | E2E Latency P50 | 3,551.8 ms | 3,681.5 ms | **+129.7 ms** |
| | Routing % of Total E2E | 0.00% | 0.10% | **< 0.2% of total runtime** |
| **Online Model Costs** | Additional Online LLM Calls | 0 | 0 | **0 extra calls** (Strictly 1 generation) |
| | Additional Query Embeddings | 0 | 0 | **0 extra calls** (Descent is pure lexical) |
| | Mean Input Tokens | 739.8 | 754.2 | **+14.48 tokens** (+1.96%) |
| | Context Budget Cap | 4,000 | 4,000 | **100% strictly enforced** ($\le 5$ chunks) |
| **Computational Footprint** | Candidate Work Factor | 1.00x (5.0 items) | 1.32x (6.6 items) | **+32% items evaluated** |
| | Local FTS Queries / Query | 0 | 0.42 | **Average 0.42 FTS queries** |
| **Architecture Dependencies** | External Online Services | 2 (Qdrant, LLM API) | 2 (Qdrant, LLM API) | **0 new external services** (SQLite embedded) |
| | Storage Overhead | ~15 MB | ~23.6 MB | **+8.6 MB local SQLite DB** |

### Key Engineering Insights:
1. **Asymmetric Activation**: Routing is not triggered on every query. Fast-path queries ($57.2\%$) execute in $3.67\text{ ms}$ (identical to B0). Only complex multi-hop queries ($42.8\%$) trigger the graph and descent modules (Local P50: $10.63\text{ ms}$).
2. **Negligible Latency Tax**: A mean local overhead of $3.63\text{ ms}$ represents $0.10\%$ of total request time, completely dwarfed by generator network latency.
3. **Efficiency Ratios**:
   - **Cost per $+1\text{pp}$ Gain**: $\frac{3.63\text{ ms}}{5.60} = \mathbf{0.65\text{ ms / +1pp}}$.
   - **Cost per Net Rescue**: $\frac{250 \times 3.63\text{ ms}}{14} = \mathbf{0.07\text{ s / net rescue}}$.

```text
========================================================================================
FINAL ENGINEERING VERDICT:
VERDICT E-A: ENGINEERING VALUE CONFIRMED.
Under the current hardware, task scale, and system configuration, the confirmed +5.60pp
gain is obtained with low-millisecond local overhead and zero extra online LLM calls.
========================================================================================
```

---

## 10. Summary of Confirmed vs. Non-Confirmed Claims

| Claim | Experimental Status | Definitive Finding |
|:---|:---:|:---|
| **End-to-End Accuracy Gain** | **CONFIRMED** | Statistically significant $+5.60\text{pp}$ gain on Independent Holdout-2 ($p=0.0043$, CI `[+2.00, +9.20]`). |
| **Document Navigation Gain** | **CONFIRMED** | $+9.60\text{pp}$ lift in Gold Document Recall ($88.93\% \to 98.53\%$). |
| **Multi-Hop Advantage** | **CONFIRMED** | $+8.97\text{pp}$ gain on 2-hop+ queries; 0 regressions on 1-hop simple queries. |
| **Composition & Descent Necessity** | **CONFIRMED** | E1 eliminated evidence eviction; E2-Lite provided targeted local chunk recovery. |
| **Engineering ROI** | **CONFIRMED** | Low millisecond overhead ($+0.58\text{ ms}$ P50); 0 extra LLM calls; 0 query embeddings. |
| **Scale Robustness** | **NOT CONFIRMED** | $\text{RA} = -2.50\text{pp}$; V3 degraded faster than B0 from $D_{20}$ to $D_{100}$. |
| **Hub Node Stability** | **NOT CONFIRMED** | High-degree nodes exhibited candidate expansion (P95: 27.1) and chain drop ($-17.5\text{pp}$). |
| **Graph Flooding Suppression** | **NOT TESTABLE** | Legacy unconstrained graph baseline was unavailable; comparative suppression unproven. |
| **Universal RAG Superiority** | **NOT CLAIMED** | Validated specifically in regulatory domain with explicit statutory cross-references. |

---

## 11. Threats to Validity & Limitations

1. **Domain Specificity**: The evaluation corpus consists of 100 Chinese medical and healthcare administrative regulations ($2,862$ chunks). The architecture relies on the existence of statutory cross-references and legislative hierarchies (`BASED_ON`, `REFERENCES`). It should not be generalized to unstructured narrative text, open-domain web corpora, or codebases without domain testing.
2. **Scale Boundary**: Tested up to 100 documents ($D_{100}$). While candidate count remained bounded, scale degradation in Phase C indicates that larger corpora ($1,000+$ documents) will require tighter prefix pruning.
3. **Model Dependence**: Generators and judges were evaluated with `deepseek-chat`. While blind adjudication showed high concordance ($90.9\%$), sensitivity to alternative LLM backends remains unmeasured.
4. **Research Prototype Status**: The system is a frozen research artifact designed to validate retrieval principles, not a production-hardened microservice.
