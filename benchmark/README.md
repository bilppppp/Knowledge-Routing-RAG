# Benchmark Datasets & Exposure Manifest

This directory contains the benchmarks used across all experimental phases of the `Knowledge-Routing-RAG` project, from initial development through the Gate I integrity audit and the final Mechanism Holdout-3 causal evaluation.

> [!IMPORTANT]
> **Data Integrity & Exposure Classification**:
> To ensure complete scientific transparency, every benchmark suite in this repository is explicitly classified by its exposure, contamination status, corpus version, and exact role in the scientific conclusions.

---

## Benchmark Suites Overview

| Benchmark Suite | Directory Path | Sample Size ($N$) | Target Corpus | Primary Purpose | Exposure & Scientific Status |
|:---|:---|:---:|:---:|:---|:---|
| **Dev-216** | `benchmark/` (`questions.jsonl`, `gold.jsonl`, `split.json`) | 216 instance runs<br>(120 unique QIDs across D20/D50/D100) | $D_{20} \subset D_{50} \subset D_{100}$ | Initial algorithm exploration & iterative heuristic development (C1–C7, E1–E2). | **`STATUS: DEVELOPMENT / CONTAMINATED BY ITERATIVE OPTIMIZATION`**<br>*Subjected to failure analysis, prompt tweaks, and rule refinements. Results on this set (e.g. C7 80.09%) cannot serve as independent generalization evidence.* |
| **Holdout-1** | `benchmark/confirmation/` | 200 unique QIDs (`CONF_Q001`–`CONF_Q200`) | $D_{100}$ (Old Snapshot) | Initial clean architecture confirmation following C7 decontamination. | **`STATUS: INDEPENDENT AT CREATION / NOW EXPOSED`**<br>*Confirmed navigation gain (+8.08pp doc recall), but revealed end-to-end majority gap (-0.50pp), demonstrating Navigation Gain $\ne$ Answer Gain.* |
| **Holdout-2** | `benchmark/holdout2/` | 250 unique QIDs (`H2_001`–`H2_250`) | $D_{100}$ (Old Snapshot) | Historical confirmation of full V3 stack vs Dense Top-5 baseline. | **`STATUS: HISTORICAL FULL-STACK CONFIRMATION / OLD CORPUS SNAPSHOT / NOW EXPOSED`**<br>*Confirmed full V3 stack (73.20%) > Dense Top-5 (67.60%, +5.60pp). Does NOT confirm graph edges as the cause or graph routing as a necessary component.* |
| **ScaleSet-1** | `benchmark/scale1/` | 80 unique QIDs (`SCALE_001`–`SCALE_080`) | $D_{20} \subset D_{50} \subset D_{100}$ | Scale robustness & distractor stress characterization across expanding corpora. | **`STATUS: SCALE CHARACTERIZATION / NOW EXPOSED`**<br>*Gold answers strictly located in $D_{20}$. Evaluated degradation under corpus expansion. Concluded: Scale robustness is NOT CONFIRMED.* |
| **HubSet-1** | `benchmark/hub1/` | 100 unique QIDs (`HUB_001`–`HUB_100`) | $D_{100}$ (Old Snapshot) | Knowledge graph high-degree / hub stress characterization across 5 degree tiers. | **`STATUS: HUB CHARACTERIZATION / NOW EXPOSED`**<br>*Stratified by graph degree (Buckets A–E: <5, 5–10, 11–20, 21–50, >50). Concluded: Hub stability is NOT CONFIRMED; Flooding suppression is NOT TESTABLE.* |
| **Mechanism Holdout-3** | `benchmark/holdout3/` | 240 unique QIDs (`H3_001`–`H3_240`) | $D_{100}$ (Corrected Frozen Corpus) | Final causal mechanism evaluation: separating graph edges from metadata, lexical, and composition. | **`STATUS: FINAL CAUSAL MECHANISM EVALUATION / CORRECTED CORPUS / NOW EXPOSED`**<br>*Stratified across Q-E (80), Q-P (80), Q-I (80). Proved structured retrieval explains gains; graph routing is not necessary and causes regressions.* |

---

## Detailed Suite Specifications

### 1. Dev-216 (`benchmark/`)
- **Files**: `questions.jsonl`, `gold.jsonl`, `split.json`
- **Questions**: 120 total questions (24 dev, 96 test in original protocol; expanded to 216 test runs across D20/D50/D100).
- **Status**: **DEVELOPMENT / CONTAMINATED BY ITERATIVE OPTIMIZATION**.
- **Role in Research**: Used for iterative routing experiments. Through error analysis and prompt adjustments, this benchmark effectively became an optimization set. It must not be cited as independent confirmation.

### 2. Holdout-1 (`benchmark/confirmation/`)
- **Files**: `questions.jsonl`, `gold.jsonl`, `manifest.json`
- **Questions**: 200 unseen questions constructed under strict lexical dissimilarity ($Jaccard < 0.35$).
- **Status**: **INDEPENDENT AT CREATION / NOW EXPOSED**.
- **Role in Research**: Used to test C7-Clean. Document recall improved significantly ($83.00\% \to 91.08\%$), but answer majority accuracy did not ($74.50\% \to 74.00\%$). Proved *Navigation Gain $\ne$ Answer Gain*, identifying the "useful evidence eviction" flaw that led directly to V3.

### 3. Holdout-2 (`benchmark/holdout2/`)
- **Files**: `questions.jsonl`, `gold.jsonl`, `manifest.json`
- **Questions**: 250 questions constructed strictly on the original D100 corpus snapshot.
- **Status**: **HISTORICAL FULL-STACK CONFIRMATION / OLD CORPUS SNAPSHOT / NOW EXPOSED**.
- **Role in Research**: Historical confirmation that the full V3 stack (73.20%) outperformed Dense Top-5 (67.60%, $+5.60\text{pp}$; cleaned $N=248$: 73.39% vs 67.74%, $+5.65\text{pp}$). 
- **Methodological Limitations Revealed by Gate I**:
  1. *Explicitness Bias*: 94.87% (148 / 156) of multi-hop questions explicitly gave all target statute titles (E2/E3), and 100% of the net multi-hop rescues (+14) occurred in E2/E3.
  2. *Forced Same-Evidence Flip*: The runner code structurally copied baseline answers when final evidence was identical (`h1_ans = h0_ans`), meaning Same-Evidence Flips = 0 was structurally forced rather than empirically measured.
  3. *Useful Evidence Eviction*: Measured at 4.0% (10 / 250), not 0% (which applied only to the development ablation).
  4. *Causal Confound*: It tested full V3 vs B0, leaving graph contribution confounded with metadata routing, BM25, and coverage composition.

### 4. ScaleSet-1 (`benchmark/scale1/`)
- **Files**: `questions.jsonl`, `gold.jsonl`, `manifest.json`
- **Questions**: 80 questions whose ground truth documents reside strictly in $D_{20}$, evaluated in $D_{20}$, $D_{50}$, and $D_{100}$.
- **Status**: **SCALE CHARACTERIZATION / NOW EXPOSED**.
- **Role in Research**: Demonstrated that V3 degradation ($D_{20} \to D_{100}$) was $+2.50\text{pp}$ while B0 was $+0.00\text{pp}$ ($\text{RA} = -2.50\text{pp}$, 95% CI `[-6.25pp, 0.00pp]`). Concluded: *Scale robustness is NOT CONFIRMED*.

### 5. HubSet-1 (`benchmark/hub1/`)
- **Files**: `questions.jsonl`, `gold.jsonl`, `manifest.json`
- **Questions**: 100 questions stratified across 5 graph degree buckets (20 questions each in Buckets A, B, C, D, E).
- **Status**: **HUB CHARACTERIZATION / NOW EXPOSED**.
- **Role in Research**: Evaluated degree sensitivity. High-degree buckets exhibited candidate expansion (P95: 27.1 candidates) and chain degradation ($60\% \to 42.5\%$). Legacy unconstrained baseline was unavailable. Concluded: *Hub stability is NOT CONFIRMED; Graph flooding suppression is NOT TESTABLE*.

### 6. Mechanism Holdout-3 (`benchmark/holdout3/`)
- **Files**: `questions.jsonl`, `gold.jsonl`, `manifest.json`
- **Questions**: Exactly 240 questions constructed on the Corrected Frozen Corpus, equally partitioned into 3 explicitness strata:
  - **Q-E (Explicit)**: 80 questions (all target statute titles explicitly given).
  - **Q-P (Partial)**: 80 questions (one target statute explicit, others implicit).
  - **Q-I (Implicit)**: 80 questions (purely situational queries with no statute titles).
- **Status**: **FINAL CAUSAL MECHANISM EVALUATION / CORRECTED CORPUS / NOW EXPOSED**.
- **Role in Research**: The definitive causal ablation test isolating graph edges from non-graph structured baselines across 6 systems ($S_0 \sim S_5$). Proved that:
  - Non-graph structured retrieval ($S_2$: 69.17%, $S_3$: 69.58%) significantly beats Dense Top-5 ($S_0$: 62.92%, $p < 0.01$).
  - Graph traversal ($S_4$: 64.17%) significantly underperforms the matched non-graph system $S_3$ ($-5.42\text{pp}$, $p = 0.0059$).
  - True graph ($S_4$) is statistically indistinguishable from a randomly shuffled graph ($S_5$: 63.33%, $p = 0.7728$).
  - True Graph Causal Rescues = 0, Graph Regressions = 9, Net Graph Causal Gain = -9.
