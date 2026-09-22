# Benchmark Datasets & Exposure Manifest

This directory contains the stratified benchmarks used across all experimental phases of the `Knowledge-Routing-RAG` project.

> [!IMPORTANT]
> **Data Integrity & Exposure Classification**:
> To ensure scientific rigor and avoid circular reasoning, each benchmark suite is explicitly classified by its exposure and contamination status.

---

## Benchmark Suites Overview

| Benchmark Suite | Directory Path | Sample Size ($N$) | Target Corpus | Primary Purpose | Exposure & Contamination Status |
|:---|:---|:---:|:---:|:---|:---|
| **Dev-216** | `benchmark/` (`questions.jsonl`, `gold.jsonl`, `split.json`) | 216 instance runs<br>(120 unique QIDs across D20/D50/D100) | $D_{20} \subset D_{50} \subset D_{100}$ | Initial algorithm exploration & iterative heuristic development (C1–C7, E1–E2). | **`STATUS: DEVELOPMENT / EXPOSED`**<br>*Subjected to failure analysis, prompt tweaks, and rule refinements. Results on this set (e.g. C7 80.09%) cannot serve as independent generalization evidence.* |
| **Holdout-1** | `benchmark/confirmation/` | 200 unique QIDs (`CONF_Q001`–`CONF_Q200`) | $D_{100}$ | Initial clean architecture confirmation following C7 decontamination. | **`STATUS: EXPOSED AFTER CONFIRMATION`**<br>*Confirmed navigation gain (+8.08pp doc recall), but revealed end-to-end majority gap (-0.50pp), prompting V3 E1/E2 development.* |
| **Holdout-2** | `benchmark/holdout2/` | 250 unique QIDs (`H2_001`–`H2_250`) | $D_{100}$ | Final pre-registered independent confirmation of frozen V3 vs B0 Vector RAG. | **`STATUS: FINAL INDEPENDENT CONFIRMATION SET`**<br>*100% physically isolated and frozen prior to evaluation. Zero question overlap or parameter exposure. Serves as primary scientific proof.* |
| **ScaleSet-1** | `benchmark/scale1/` | 80 unique QIDs (`SCALE_001`–`SCALE_080`) | $D_{20} \subset D_{50} \subset D_{100}$ | Phase C: Scale robustness & distractor stress characterization across expanding corpora. | **`STATUS: SCALE CONFIRMATION SET`**<br>*Gold answers strictly located in $D_{20}$. Evaluates degradation under corpus expansion.* |
| **HubSet-1** | `benchmark/hub1/` | 100 unique QIDs (`HUB_001`–`HUB_100`) | $D_{100}$ | Phase D: Knowledge graph high-degree / hub stress characterization across 5 degree tiers. | **`STATUS: HUB CHARACTERIZATION SET`**<br>*Stratified by graph degree (Buckets A–E: <5, 5–10, 11–20, 21–50, >50), decoupled from hop count.* |

---

## Detailed Suite Specifications

### 1. Dev-216 (`benchmark/`)
- **Files**: `questions.jsonl`, `gold.jsonl`, `split.json`
- **Questions**: 120 total questions (24 dev, 96 test in original protocol; expanded to 216 test runs across D20/D50/D100).
- **Status**: **EXPOSED / OPTIMIZATION SET**.
- **Role in Research**: Early in the project, these 216 instances were used for iterative routing experiments. Through error analysis and prompt adjustments, this benchmark effectively became an optimization set. It must not be cited as independent confirmation.

### 2. Holdout-1 (`benchmark/confirmation/`)
- **Files**: `questions.jsonl`, `gold.jsonl`, `manifest.json`
- **Questions**: 200 unseen questions constructed under strict lexical dissimilarity ($Jaccard < 0.35$).
- **Status**: **EXPOSED AFTER CONFIRMATION**.
- **Role in Research**: Used to test C7-Clean. It showed that document recall improved significantly ($83.00\% \to 91.08\%$), but answer majority accuracy did not ($74.50\% \to 74.00\%$). This critical negative result demonstrated that *Navigation Gain $\ne$ End-to-End Gain*, identifying the "useful evidence eviction" flaw that led directly to V3.

### 3. Holdout-2 (`benchmark/holdout2/`)
- **Files**: `questions.jsonl`, `gold.jsonl`, `manifest.json`
- **Questions**: 250 fresh, independent questions constructed strictly on the D100 corpus.
- **Status**: **FINAL INDEPENDENT CONFIRMATION SET (FROZEN)**.
- **Role in Research**: The definitive test set for V3-Frozen. Evaluated across 3 fresh runs with random seeds (101, 202, 303) and blinded adjudication. Confirmed $+5.60\text{pp}$ accuracy gain ($p = 0.0043$, 95% CI `[+2.00pp, +9.20pp]`).

### 4. ScaleSet-1 (`benchmark/scale1/`)
- **Files**: `questions.jsonl`, `gold.jsonl`, `manifest.json`
- **Questions**: 80 questions whose ground truth documents reside strictly in $D_{20}$, evaluated in $D_{20}$, $D_{50}$, and $D_{100}$.
- **Status**: **SCALE CONFIRMATION SET**.
- **Role in Research**: Demonstrated that V3 degradation ($D_{20} \to D_{100}$) was $+2.50\text{pp}$ while B0 was $+0.00\text{pp}$ ($\text{RA} = -2.50\text{pp}$, 95% CI `[-6.25pp, 0.00pp]`). Concluded: *Scale robustness is NOT CONFIRMED*.

### 5. HubSet-1 (`benchmark/hub1/`)
- **Files**: `questions.jsonl`, `gold.jsonl`, `manifest.json`
- **Questions**: 100 questions stratified across 5 graph degree buckets (20 questions each in Buckets A, B, C, D, E).
- **Status**: **HUB CHARACTERIZATION SET**.
- **Role in Research**: Evaluated degree sensitivity. High-degree buckets exhibited candidate expansion (P95: 27.1 candidates) and chain degradation ($60\% \to 42.5\%$). Legacy unconstrained baseline was unavailable. Concluded: *Hub stability is NOT CONFIRMED; Graph flooding suppression is NOT TESTABLE*.
