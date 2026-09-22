# Knowledge-Routing-RAG: Final Research & Experimental Report

**Project Title**: Knowledge-Routing-RAG: Empirical Study of Structured Retrieval and Graph Navigation for Regulatory RAG  
**Date**: 2026-09-22 / 2026-09-23  
**Repository**: [https://github.com/bilppppp/Knowledge-Routing-RAG](https://github.com/bilppppp/Knowledge-Routing-RAG)  
**Status**: Experimental Research Cycle Completed & Frozen (`v1.0-research-final`)  
**Audience**: AI Researchers, Retrieval Engineers, and Academic Evaluators  

---

## 1. Executive Summary

Under the corrected frozen corpus, matched embedding models, fixed evidence budget ($\le 4000$ tokens / $\le 5$ chunks), independent evaluation protocols, and causal mechanism ablations of this study:

```text
========================================================================================
FINAL RESEARCH VERDICTS:

Structured Retrieval Gain:          CONFIRMED (+6.67pp on Holdout-3 over Dense Top-5, p = 0.0033)
Metadata / Title Routing Gain:      CONFIRMED (+6.25pp on Holdout-3 over Dense Top-5, p = 0.0051)
Local Lexical Descent Value:        SUPPORTED (FTS5 BM25 recovers statutory article slots)
Coverage-Aware Composition Value:   SUPPORTED (Multi-statute slot preservation)

Graph Incremental Value:            NOT CONFIRMED (Negative increment: -5.42pp vs NoGraph, p = 0.0059)
Graph Necessity:                    REJECTED FOR CURRENT TASK (All historical gains explained by S2/S3)
True Graph vs Shuffled Graph:       STATISTICALLY INDISTINGUISHABLE (+0.83pp, p = 0.7728)
True Graph Causal Rescues:          0 (Zero questions rescued by graph edges)
Graph Causal Regressions:           9 (Nine questions degraded by graph distractor injection)
Net Graph Causal Gain:              -9

Scale Robustness:                   NOT CONFIRMED (RA = -2.50pp, 95% CI [-6.25pp, 0.00pp])
Hub Node Stability:                 NOT CONFIRMED (Candidate P95 rose to 27.1, chain drop -17.5pp)
Graph Flooding Suppression:         NOT CONFIRMED / NOT TESTABLE (Legacy graph baseline unavailable)
Universal Superiority:              NOT CLAIMED

RECOMMENDED FINAL ARCHITECTURE:     V3-NoGraph / Structured Retrieval Final (Graph-Free)
========================================================================================
```

**Core Project Verdict (English)**:
> In this structured regulatory corpus, document/title-aware retrieval, local lexical descent, and coverage-aware evidence composition significantly outperform a simple Dense Top-5 baseline. Knowledge graph relations were not required for this gain and caused measurable evidence displacement and answer regressions in the final causal ablation. The strongest validated configuration is a simpler graph-free structured retrieval pipeline.

**核心结论 (中文)**:
> 在本实验的结构化医疗法规语料中，文档/标题感知检索、局部词法下潜与覆盖感知证据组合相比简单 Dense Top-5 能显著改善检索和回答表现；知识图关系不是获得这一收益的必要条件，并在最终因果消融实验中降低了系统表现。历史上完整 V3 相比 Dense Top-5 的全部收益均可由非图结构化检索组件完全解释。最终推荐系统为移除了图游走的纯结构化检索流水线（V3-NoGraph）。

---

## 2. Original Hypothesis: Network-Routing-Inspired RAG

The project was originally conceived on an analogy between IP network routing and multi-hop knowledge retrieval:
- In IP networks, routers do not forward packets based solely on payload contents; they use a structured routing information base (RIB) and forwarding information base (FIB) to direct packets along known topology paths.
- Similarly, we hypothesized that an LLM-based autonomous routing controller could navigate an interconnected knowledge network of statutes, amendments, and derived regulations using discrete control-plane routing protocols (K1 Prefix Planner, K2 Forwarding Engine, K3 Segment Router, K4 Gap Detector).

Under this initial vision:
1. Dense vector search would only act as a local entry point.
2. Knowledge graph edges (e.g., `REFERENCES`, `DERIVED_FROM`, `REVISES`) would serve as the autonomous routing plane to discover cross-statute dependencies.
3. An LLM agent would iteratively hop through the graph until all missing evidence slots were filled.

---

## 3. V1 Failure: Autonomous Routing NO-GO

The initial V1 implementation was evaluated against a standard Vector RAG baseline on the development corpus ($D_{20} \subset D_{50} \subset D_{100}$). The empirical results rejected the autonomous routing hypothesis:

```text
V1 Exploration Results (Dev Benchmark):
- Vector RAG Baseline (B0):         68.5% Accuracy
- Autonomous Knowledge Routing (V1): 58.3% Accuracy (Delta = -10.2pp)
- Unconstrained Graph Traversal:     P95 Candidate Count > 45 chunks
- Context Contamination:             Evidence context filled with irrelevant neighbor nodes
```

**Root Causes of V1 Failure**:
1. **Unconstrained Graph Drift**: Following relational edges without strict boundary constraints rapidly flooded the candidate pool with legally irrelevant cross-references (e.g., citing a general administrative penalty law when the question asked about a specific medical licensing exemption).
2. **LLM Controller Latency & Hallucination**: The autonomous agent made 3–5 sequential online LLM calls per query, introducing multi-second latency and cumulative reasoning errors.
3. **Evidence Displacement**: When graph neighbors were forced into the generator context, they evicted the true target chunks found by the initial vector search.

**Verdict**: The original autonomous network-routing hypothesis was **NOT CONFIRMED** and declared a complete **NO-GO**.

---

## 4. V2/V3 Development: Shift to Structural Constraints

Following the V1 failure, the research shifted from autonomous agentic graph hopping to a deterministic two-plane architecture:
- **Control Plane vs Data Plane Separation**: The control plane searches broadly across candidate spaces, while the data plane strictly filters and compresses evidence into a fixed budget ($\le 5$ chunks, $\le 4000$ tokens).
- **Directed Search (C1–C7)**: Explored structural heuristics:
  - *C4 Shadow Prefix Plane*: Expanding initial vector recall from Top-5 to Top-20 candidate documents without feeding them to the generator.
  - *C5 Parent Lift & Title Resolution*: Lifting chunk-level hits to document-level metadata to enable cross-document awareness.
  - *C6 Local Lexical Descent*: Using FTS5 BM25 search restricted strictly within candidate documents to find specific statutory articles.
  - *C7 Decontamination*: Removing hardcoded statute lookup tables and regex rules to establish a generic, scalable pipeline (`C7-Clean`).

Through these phases, the architecture progressively de-emphasized deep graph exploration in favor of robust document-level identification, local lexical retrieval, and evidence set composition.

---

## 5. Benchmark Contamination: The Fall of Dev-216

Early development relied on a 216-instance benchmark across D20, D50, and D100 (`benchmark/questions.jsonl`). As the engineering team analyzed failure cases and iteratively tuned heuristic thresholds (e.g., fast-path thresholds, heading bonus weights, BM25 multipliers), this dataset became progressively contaminated:
- The system achieved **80.09%** on Dev-216 under C7, but manual inspection revealed that prompts, heuristics, and lane thresholds had been implicitly tailored to the idiosyncrasies of those specific 216 questions.
- **Scientific Decision**: Dev-216 was formally stripped of its evaluation status and reclassified as **DEVELOPMENT / CONTAMINATED BY ITERATIVE OPTIMIZATION**. All subsequent confirmations required strictly sealed, unseen holdout datasets with automated lexical decontamination ($Jaccard < 0.35$ against all historical queries).

---

## 6. Holdout-1: Navigation Gain $\ne$ Answer Gain

The decontaminated `C7-Clean` architecture was subjected to its first independent test on **Holdout-1** ($N = 200$, unseen questions):

```text
Holdout-1 Results (N = 200):
- Gold Document Recall:    83.00% (B0) -> 91.08% (C7-Clean)  [+8.08pp, CONFIRMED]
- End-to-End Accuracy:     74.50% (B0) -> 74.00% (C7-Clean)  [-0.50pp, NOT CONFIRMED]
```

**The Discovery of Useful Evidence Eviction**:
While `C7-Clean` successfully navigated to the correct target documents, its unconstrained candidate replacement logic replaced high-relevance chunks from the initial vector search with lower-relevance chunks found during document exploration.
- This proved the fundamental methodological lesson: **Document Navigation Gain $\ne$ Answer Generation Gain**.
- This failure led directly to the development of **V3**, specifically introducing:
  1. **E1 Coverage-Preserving Evidence Composer**: A submodular-inspired set selector that only admits replacement candidates if they cover genuinely unresolved query slots without evicting core evidence.
  2. **E2-Lite Slot-Conditioned Lexical Descent**: Restricting local descent to pure lexical BM25 over structural headings, discarding complex semantic neural channels.

---

## 7. Holdout-2: Historical Full-Stack Confirmation & Its Boundaries

On **Holdout-2** ($N = 250$, unseen questions evaluated across 3 independent runs with random seeds 101, 202, 303), the frozen **V3** stack was evaluated against Dense Top-5 ($B_0$):

```text
Historical Holdout-2 Results (Old Corpus Snapshot, N = 250):
- B0 (Dense Top-5 Vector RAG):   67.60%
- V3 (Full Clean Routing Stack):  73.20%
- Delta:                         +5.60pp (p = 0.0043, 95% CI [+2.00pp, +9.20pp])
- Cleaned (N = 248, Gate I):     67.74% vs 73.39% (Delta = +5.65pp)
```

### The Re-Defined Role of Holdout-2:
Holdout-2 successfully confirmed that the **full V3 retrieval stack significantly outperformed Dense Top-5**.
However, as revealed by subsequent integrity audits and causal ablations:
1. **Holdout-2 cannot attribute gains to graph routing**: It tested full V3 (which bundled shadow recall, metadata title resolution, BM25, coverage composition, AND graph edges) against pure vector Top-5.
2. **Explicitness Bias**: Multi-hop questions in Holdout-2 were overwhelmingly explicit (94.87% E2/E3), meaning the benchmark primarily measured explicit multi-statute retrieval rather than implicit relational discovery.
3. **Same-Evidence Flip Shortcut**: The runner reused baseline answers when evidence was identical (`h1_ans = h0_ans`), meaning Same-Evidence Flips = 0 was a structural artifact of the code rather than empirical stability.
4. **Useful Evidence Eviction**: The eviction rate on Holdout-2 was 4.0% (10 / 250), not 0% (which was true only on the development set).

---

## 8. Scale & Hub Negative Results

To stress-test V3 under expanding corpus size and high-degree graph topology, two dedicated pre-registered experiments were conducted:

### Phase C: Scale Robustness (ScaleSet-1, $N = 80$)
- Questions answerable strictly within $D_{20}$ were evaluated as the corpus expanded to $D_{50}$ and $D_{100}$.
- **Result**: V3 accuracy degraded by $+2.50\text{pp}$ ($88.75\% \to 86.25\%$), while B0 experienced zero degradation ($83.75\% \to 83.75\%$).
- Relative Advantage: $\text{RA} = -2.50\text{pp}$ (95% CI `[-6.25pp, 0.00pp]`).
- **Verdict**: **SCALE ROBUSTNESS NOT CONFIRMED**.

### Phase D: Hub Node Stress (HubSet-1, $N = 100$)
- Questions were stratified across 5 knowledge graph degree tiers (Bucket A: degree $<5$, to Bucket E: degree $>50$).
- **Result**: In high-degree tiers, P95 candidate count expanded to 27.1, and chain completion dropped from 60.0% to 42.5%.
- **Verdict**: **HUB STABILITY NOT CONFIRMED; GRAPH FLOODING SUPPRESSION NOT TESTABLE**.

---

## 9. Gate I Integrity Audit: Data & Runner Errata

Prior to final causal confirmation, a 100% full-corpus data integrity audit was conducted across all 100 documents and 2,862 chunks, yielding verdict **`I-B — CORRECTABLE INTEGRITY ISSUES`**:

```text
Gate I Verification Summary:
- Total Documents Audited:         100 / 100 (100%)
- Total Chunks Audited:            2862 / 2862 (100% verified against raw text)
- Total Gold Spans Audited:        1425 / 1425 (100% verified verbatim in corpus)

Identified Metadata Misplacements:
1. doc034: Chunk titles and graph nodes carried title of doc033 ("红十字标志使用办法") 
           instead of canonical title ("中华人民共和国红十字会法"). (32 chunks affected)
2. doc086: Manifest title erroneously contained doc087 title. (14 chunks affected)

Total Affected Benchmark Questions:
- Dev-216: 6 questions
- Holdout-1: 2 questions
- Holdout-2: 2 questions (H2_017, H2_086)
```

**Decontamination & Freeze Action**:
- Metadata, SQLite FTS5 tables, and Qdrant payloads were corrected and permanently frozen with SHA256 verification in [`CORPUS_CORRECTED_FREEZE.md`](../CORPUS_CORRECTED_FREEZE.md).
- Removing the 2 affected questions from Holdout-2 resulted in $N=248$: $B_0 = 67.74\%$, $V_3 = 73.39\%$ ($\Delta = +5.65\text{pp}$), proving the historical full-stack gain was not manufactured by data errors.

---

## 10. Mechanism Holdout-3: Definitive Causal Ablation

To resolve the decisive question—*Does the knowledge graph provide an independent causal increment, or do strong non-graph baselines explain the entire historical gain?*—we designed and executed **Mechanism Holdout-3**:

- **Benchmark**: $N = 240$ unseen questions constructed on the Corrected Frozen Corpus, equally partitioned into 3 explicitness strata:
  - **Q-E (Explicit)**: 80 questions (statute names explicitly stated in query).
  - **Q-P (Partial)**: 80 questions (one statute explicit, secondary statute implicit).
  - **Q-I (Implicit)**: 80 questions (purely situational queries; no statute names).
- **Execution Protocol**: One complete independent end-to-end generation and judging run (`holdout3_run1.json`, $N=240$), evaluated with Gemini 3.8 Flash (`gemini-3.8-flash`) with dynamic 4096-token reasoning headroom.
- **6 Homogeneous Systems**:
  - $S_0$: Dense Top-5 (Historical Baseline)
  - $S_1$: Dense Top-20 + E1 Coverage-Aware Composer
  - $S_2$: Metadata / Title Routing + Local BM25
  - $S_3$: V3-NoGraph (Matched Structured Ablation without Graph Traversal)
  - $S_4$: V3-TrueGraph (Frozen V3 with Real Graph Traversal)
  - $S_5$: V3-ShuffledGraph (Control Baseline with Randomly Shuffled Neighbors)

### Overall Results ($N = 240$)

| System | Architecture | Retrieval Chain Comp. | Precision | Recall | Accuracy ($N=240$) | vs $S_0$ (Dense) | vs $S_3$ (NoGraph) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$S_0$** | Dense Top-5 | 40.42% | 27.08% | 47.92% | **62.92%** | — | -6.67pp |
| **$S_1$** | Dense Top-20 + Composer | 42.50% | 26.67% | 48.75% | **65.00%** | +2.08pp | -4.58pp |
| **$S_2$** | Metadata / Title + BM25 | 47.08% | 30.25% | 54.17% | **69.17%** | **+6.25pp** | -0.42pp |
| **$S_3$** | **V3-NoGraph (Matched Structured)** | **47.92%** | **30.67%** | **55.42%** | $\mathbf{69.58\%}$ | $\mathbf{+6.67\text{pp}}$ | **0.00pp** |
| **$S_4$** | **V3-TrueGraph (Real Graph)** | 44.17% | 28.58% | 52.08% | **64.17%** | +1.25pp | $\mathbf{-5.42\text{pp}}$ |
| **$S_5$** | **V3-ShuffledGraph (Random Graph)** | 45.00% | 29.08% | 52.92% | **63.33%** | +0.41pp | -6.25pp |

### Results Across Explicitness Strata ($N = 80$ each)

| System | Q-E: Explicit ($N=80$) | Q-P: Partial ($N=80$) | Q-I: Implicit ($N=80$) |
| :--- | :---: | :---: | :---: |
| **$S_0$ (Dense Top-5)** | 70.00% | 62.50% | 56.25% |
| **$S_1$ (Dense Top-20 + Composer)** | 72.50% | 63.75% | 58.75% |
| **$S_2$ (Metadata + BM25)** | $\mathbf{83.75\%}$ | 63.75% | 60.00% |
| **$S_3$ (V3-NoGraph)** | 82.50% | $\mathbf{65.00\%}$ | $\mathbf{61.25\%}$ |
| **$S_4$ (V3-TrueGraph)** | 75.00% | 60.00% | 57.50% |
| **$S_5$ (V3-ShuffledGraph)** | 73.75% | 62.50% | 53.75% |

### Pre-Registered Primary Contrasts & Statistical Significance

| Contrast | Comparison | $\Delta$ | McNemar $p$-value | 95% Bootstrap CI | Scientific Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **C1** | $S_1 - S_0$ (Dense Expansion) | +2.08pp | $p = 0.5322$ | [-2.93pp, +7.08pp] | Dense expansion alone does not explain gains |
| **C2** | $S_2 - S_0$ (Metadata/BM25 vs Dense) | **+6.25pp** | $\mathbf{p = 0.0051^{**}}$ | [+2.50pp, +10.42pp] | **Metadata & lexical retrieval drive large gains** |
| **C3** | $S_4 - S_3$ (Graph over Matched) | **-5.42pp** | $\mathbf{p = 0.0059^{**}}$ | [-9.17pp, -2.08pp] | **Enabling graph significantly reduces accuracy** |
| **C4** | $S_4 - S_5$ (True vs Shuffled Graph) | +0.83pp | $p = 0.7728$ | [-2.08pp, +3.34pp] | Real topology is indistinguishable from random |
| **Aux** | $S_3 - S_0$ (Structured NoGraph vs Dense)| **+6.67pp** | $\mathbf{p = 0.0033^{**}}$ | [+2.50pp, +10.83pp] | **Full structured stack (no graph) is superior** |

### Same-Evidence Flip Rate (Independent Generation Verification)
- $S_4$ and $S_3$ shared identical final evidence contexts on **212 / 240 questions (88.33%)**.
- Across these 212 questions, the independent generation flip rate was **9 / 212 (4.25%)**.
- This proves the evaluation runner executed strictly independent generation without shortcuts, separating generator stochasticity from retrieval differences.

---

## 11. Causal Attribution: Graph Rescues vs Regressions

For all 19 discordant cases between $S_4$ (V3-TrueGraph) and $S_3$ (V3-NoGraph):

```text
Rescues (S4 Correct, S3 Wrong): Total = 3
- TRUE_GRAPH_CAUSAL_RESCUE:           0  (Zero questions were rescued by graph edges)
- GRAPH_ADDED_NON_GOLD_BUT_FLIP:       1  (Graph added non-gold chunk, but LLM guessed correctly)
- SAME_EVIDENCE_FLIP:                 2  (Generator variance on identical evidence)

Regressions (S3 Correct, S4 Wrong): Total = 16
- GRAPH_CAUSAL_REGRESSION:             9  (Graph neighbor evicted or diluted gold statutory text)
- GRAPH_DISTRACTOR:                   0
- SAME_EVIDENCE_FLIP:                 7  (Generator variance on identical evidence)

NET GRAPH CAUSAL GAIN: 0 - 9 = -9
```

**Causal Mechanism Summary**:
The knowledge graph did not perform a single genuine causal rescue across 240 questions. Instead, unconstrained neighbor expansion pulled in distractor chunks from related but non-applicable regulations, causing 9 regressions where gold statutory articles were evicted or diluted from the prompt.

---

## 12. Final Architecture: Graph-Free Structured Retrieval

The final recommended and empirically validated architecture is **V3-NoGraph (Structured Retrieval Final)**. The knowledge graph is completely removed from the online retrieval path.

```text
                             User Question
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     Dense Vector Entrance    │ (Top-5 chunks)
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │    Shadow Candidate Recall   │ (Top-20 chunk prefixes)
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │ Document / Title Resolution  │ (Match canonical titles from metadata)
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │ Slot-Conditioned BM25 Descent│ (Targeted FTS5 search inside resolved docs)
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  Coverage-Aware Composer E1  │ (Slot coverage maximization, budget ≤ 5)
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                       Final Evidence Context (≤ 5 Chunks)
                                   │
                                   ▼
                                  LLM
```

**Key Architectural Takeaway**:
- **Candidate Space $\ne$ Evidence Context**: Searching broadly across documents using metadata and BM25 before compressing into 5 chunks is highly effective.
- **No Graph Required**: Canonical document titles and hierarchical FTS5 indexing capture statutory structures without the noise, complexity, and failure modes of graph traversal.

---

## 13. Final Conclusions: Confirmed vs Not Confirmed

| Claim | Experimental Status | Definitive Evidence |
| :--- | :---: | :--- |
| **Structured retrieval beats Dense Top-5** | **CONFIRMED** | $S_3$ achieved 69.58% vs 62.92% ($+6.67\text{pp}$, $p = 0.0033^{**}$). |
| **Metadata / title + local lexical retrieval helps** | **CONFIRMED** | $S_2$ achieved 69.17% vs 62.92% ($+6.25\text{pp}$, $p = 0.0051^{**}$). |
| **Coverage-aware evidence composition helps** | **SUPPORTED** | Preserves multi-statute evidence under a strict 5-chunk limit. |
| **Wider dense recall alone explains the gain** | **NOT CONFIRMED** | $S_1$ achieved 65.00% ($+2.08\text{pp}$, $p = 0.5322$, not significant). |
| **Graph relations add independent value** | **NOT CONFIRMED** | $S_4$ TrueGraph lagged $S_3$ NoGraph by $-5.42\text{pp}$ ($p = 0.0059^{**}$). |
| **Graph relations are necessary for gains** | **NO** | $S_2$ and $S_3$ completely capture and exceed the historical gain. |
| **True graph outperforms shuffled graph** | **NO** | $S_4$ vs $S_5$ difference is $+0.83\text{pp}$ ($p = 0.7728$, indistinguishable). |
| **Scale robustness confirmed** | **NOT CONFIRMED** | Phase C: $\text{RA} = -2.50\text{pp}$ ($D_{20} \to D_{100}$). |
| **Hub node robustness confirmed** | **NOT CONFIRMED** | Phase D: High-degree chain completion dropped from 60% to 42.5%. |
| **Graph flooding suppression confirmed** | **NOT CONFIRMED** | Legacy unconstrained graph baseline was unavailable for formal testing. |
| **Historical full V3 beats Dense Top-5** | **CONFIRMED** | Holdout-2: 73.20% vs 67.60% ($+5.60\text{pp}$, $p = 0.0043^{**}$). |
| **Historical gain can be attributed to graph** | **NO** | Disproven by Mechanism Holdout-3 causal ablations. |

---

## 14. Remaining Known Limitations

1. **Regulatory Domain Specificity**: All experiments were conducted on a 100-document Chinese healthcare administrative regulatory corpus. Generalization to other domains (e.g., medical clinical guidelines, open-domain web corpora) is not established.
2. **Implicit Query Retrieval Ceiling**: Across all tested systems on Holdout-3, implicit multi-hop questions (Q-I, where statute names are omitted) plateaued at ~61% accuracy. Resolving implicit statutory references remains an open technical challenge.
3. **Single Evaluation Run on Holdout-3**: Due to external API quota limits, Holdout-3 was evaluated on one complete independent end-to-end generation run (`holdout3_run1.json`, $N=240$). While retrieval traces are fully deterministic ($N=240$) and McNemar tests are highly significant ($p < 0.01$), 3-run majority voting was not completed.
4. **Graph Construction Density**: The knowledge graph was constructed using explicit statutory citations and administrative hierarchies. A different, more densely curated graph ontology might behave differently, but within this benchmark, relational expansion produced negative net utility.
