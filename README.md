# Knowledge-Routing-RAG

> **An Empirical Study of Structured Retrieval for Multi-Document Regulatory RAG, Including a Negative Result on Graph-Based Routing**  
> *A frozen research repository detailing initial hypothesis failure, heuristic development, data integrity audits, and causal mechanism evaluation.*

[English](README.md) | [中文说明](README_CN.md)

[![Release](https://img.shields.io/badge/release-v1.0--research--final-blue.svg)](https://github.com/bilppppp/Knowledge-Routing-RAG/releases)
[![Status](https://img.shields.io/badge/status-research__frozen-success.svg)](#10-limitations--scope-of-validity)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](requirements.txt)

---

## Executive Project Summary

**Knowledge-Routing-RAG began as an experiment in applying network-routing and graph-navigation ideas to retrieval-augmented generation. The final causal experiments did not support graph routing as the source of the observed gains. Instead, the strongest validated system is a simpler graph-free structured retrieval pipeline combining document/title resolution, local lexical descent, wider candidate recall, and coverage-aware evidence composition.**

---

## What This Project Ultimately Found

1. **Dense Top-5 alone was insufficient** for multi-document regulatory queries (achieving only 40.42% retrieval chain completion and 62.92% answer accuracy on the Mechanism Holdout-3 benchmark).
2. **Metadata/title resolution + local lexical retrieval significantly improved performance**, lifting accuracy to **69.17%** ($+6.25\text{pp}$, $p = 0.0051$) and chain completion to 47.08%.
3. **Graph relations were not necessary for that improvement**. The matched non-graph structured stack (**V3-NoGraph**) achieved the highest accuracy across all tested systems at **69.58%** ($+6.67\text{pp}$ over Dense Top-5, $p = 0.0033$).
4. **In the final causal ablation, enabling the true graph reduced answer accuracy** relative to the matched NoGraph system (**64.17% vs 69.58%**, $\Delta = -5.42\text{pp}$, $p = 0.0059$). Graph neighbor expansion introduced distractor chunks that evicted or diluted gold statutory text (0 causal rescues vs 9 causal regressions, net gain: **-9**).
5. **A real graph topology was statistically indistinguishable from a randomly shuffled graph** (TrueGraph 64.17% vs ShuffledGraph 63.33%, $p = 0.7728$).
6. **Scale robustness and hub robustness were not confirmed**.

---

## Primary Results: Mechanism Holdout-3 ($N = 240$, Corrected Frozen Corpus)

The definitive mechanism evaluation isolates the causal contribution of graph relations from non-graph structured baselines across 6 systems on the Corrected Frozen Corpus ($N = 240$, equally partitioned into 80 Explicit, 80 Partial, and 80 Implicit questions; evaluated with Gemini 3.8 Flash, Temperature = 0.0, 4096-token ceiling):

| System | Architecture Description | Retrieval Chain Comp. | Precision | Recall | Accuracy ($N=240$) | vs Dense ($S_0$) | vs NoGraph ($S_3$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$S_0$** | **Dense Top-5** (Historical Baseline) | 40.42% | 27.08% | 47.92% | **62.92%** | — | -6.67pp |
| **$S_1$** | **Dense Top-20 + Composer** | 42.50% | 26.67% | 48.75% | **65.00%** | +2.08pp | -4.58pp |
| **$S_2$** | **Metadata / Title + Local BM25** | 47.08% | 30.25% | 54.17% | **69.17%** | **+6.25pp** ($p=0.0051$) | -0.42pp |
| **$S_3$** | **V3-NoGraph (Recommended Final)** | **47.92%** | **30.67%** | **55.42%** | $\mathbf{69.58\%}$ | $\mathbf{+6.67\text{pp}}$ ($p=0.0033$) | **0.00pp** |
| **$S_4$** | **V3-TrueGraph (Frozen V3 Real Graph)** | 44.17% | 28.58% | 52.08% | **64.17%** | +1.25pp | $\mathbf{-5.42\text{pp}}$ ($p=0.0059$) |
| **$S_5$** | **V3-ShuffledGraph (Random Graph Control)**| 45.00% | 29.08% | 52.92% | **63.33%** | +0.41pp | -6.25pp |

### Pre-Registered Primary Contrasts:
- **C2 ($S_2 - S_0$, Metadata/Lexical vs Dense)**: $\mathbf{+6.25\text{pp}}$ ($p = 0.0051^{**}$, 95% CI `[+2.50pp, +10.42pp]`) $\to$ **Strong structured baseline explains historical gains.**
- **C3 ($S_4 - S_3$, True Graph over Matched NoGraph)**: $\mathbf{-5.42\text{pp}}$ ($p = 0.0059^{**}$, 95% CI `[-9.17pp, -2.08pp]`) $\to$ **Graph expansion significantly degrades answer accuracy.**
- **C4 ($S_4 - S_5$, True Graph vs Shuffled Graph)**: $\mathbf{+0.83\text{pp}}$ ($p = 0.7728$, 95% CI `[-2.08pp, +3.34pp]`) $\to$ **Real graph topology provides zero unique causal advantage over random noise.**

---

## Historical Full-Stack Result: Holdout-2 ($N = 250$, Old Corpus Snapshot)

Before the Gate I audit and the Holdout-3 mechanism isolation, the full V3 stack was confirmed against Dense Top-5 on the original corpus snapshot:

| Benchmark Metric | Vector RAG Baseline ($B_0$) | Full V3 Retrieval Stack | Delta ($\Delta$) | Statistical Test |
| :--- | :---: | :---: | :---: | :--- |
| **Majority Accuracy ($N=250$)** | 67.60% (169/250) | 73.20% (183/250) | **+5.60pp** | Exact McNemar $p = 0.0043$ |
| Cleaned ($N=248$, Gate I) | 67.74% (168/248) | 73.39% (182/248) | **+5.65pp** | Robust to data errata |
| Net Stable Rescues | — | — | **+14** | Primary endpoint confirmed |
| Multi-Hop (2-Hop+) Accuracy | 48.72% | 57.69% | **+8.97pp** | Multi-statute completion |

> [!WARNING]
> **Status of Holdout-2**:
> This result confirms that the **full retrieval stack outperformed Dense Top-5**, but later causal ablations on Holdout-3 proved that **the gain should not be attributed to graph routing**.
> - **Explicitness Bias**: 94.87% (148/156) of Holdout-2 multi-hop queries explicitly named all target statutes (E2/E3), and 100% of the net multi-hop rescues occurred on these explicit queries. Holdout-2 measured explicit multi-statute retrieval, not implicit graph discovery.
> - **Same-Evidence Flips = 0 was Structurally Forced**: The historical runner code copied baseline answers when final evidence chunk IDs were identical (`h1_ans = h0_ans`). Same-evidence flip rate was therefore not empirically measured in Holdout-2.
> - **Useful Evidence Eviction**: Eviction on Holdout-2 was 4.0% (10/250), not 0% (which was true only on the development set).

---

## Recommended Architecture: Structured Retrieval Final (Graph-Free)

The final validated configuration is **V3-NoGraph (Structured Retrieval Final)**. The knowledge graph is completely removed from the online path:

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
                       Final Evidence Context (≤ 5 Chunks, ≤ 4000 tok)
                                   │
                                   ▼
                                  LLM
```

### Enduring Engineering Axioms:
- **Candidate Space $\ne$ Evidence Context**: Broad candidate exploration (Top-20 shadow recall, metadata resolution, targeted BM25) followed by strict set compression ($\le 5$ chunks) is highly effective.
- **Title & Metadata Resolution > Graph Traversal**: Recognizing statutory titles in queries and descending into specific articles via lexical search achieves $+6.67\text{pp}$ over dense search without the distractor risks of relational graph expansion.

---

## Final Research Verdict Matrix

| Scientific Claim | Final Status | Empirical Evidence |
| :--- | :---: | :--- |
| **Structured retrieval beats Dense Top-5** | **CONFIRMED** | $S_3$ achieved 69.58% vs 62.92% ($+6.67\text{pp}$, $p = 0.0033^{**}$). |
| **Metadata / title + local lexical retrieval helps** | **CONFIRMED** | $S_2$ achieved 69.17% vs 62.92% ($+6.25\text{pp}$, $p = 0.0051^{**}$). |
| **Coverage-aware evidence composition helps** | **SUPPORTED** | Solves multi-statute slot preservation under a 5-chunk limit. |
| **Wider dense recall alone explains the gain** | **NOT CONFIRMED** | $S_1$ achieved 65.00% ($+2.08\text{pp}$, $p = 0.5322$, not significant). |
| **Graph relations add independent value** | **NOT CONFIRMED** | $S_4$ TrueGraph lagged $S_3$ NoGraph by $-5.42\text{pp}$ ($p = 0.0059^{**}$). |
| **Graph relations are necessary for gains** | **REJECTED** | $S_2$ and $S_3$ completely capture and exceed historical gains. |
| **True graph outperforms shuffled graph** | **NO** | $S_4$ vs $S_5$ difference is $+0.83\text{pp}$ ($p = 0.7728$, indistinguishable). |
| **Scale robustness confirmed** | **NOT CONFIRMED** | Phase C: $\text{RA} = -2.50\text{pp}$ ($D_{20} \to D_{100}$). |
| **Hub node robustness confirmed** | **NOT CONFIRMED** | Phase D: High-degree chain completion dropped from 60% to 42.5%. |
| **Graph flooding suppression confirmed** | **NOT CONFIRMED** | Legacy unconstrained baseline was unavailable for formal testing. |
| **Historical full V3 beats Dense Top-5** | **CONFIRMED** | Holdout-2: 73.20% vs 67.60% ($+5.60\text{pp}$, $p = 0.0043^{**}$). |
| **Historical gain can be attributed to graph** | **NO** | Disproven by Mechanism Holdout-3 causal ablations. |

---

## Quick Start (Recommended Configuration)

The default runnable pipeline is **Structured Retrieval Final** (`configs/structured_retrieval_final.yaml`).

### 1. Prerequisites & Environment
```bash
git clone https://github.com/bilppppp/Knowledge-Routing-RAG.git
cd Knowledge-Routing-RAG

# Create virtual environment (Python 3.12+)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Smoke Test (Zero API Cost)
```bash
python scripts/release_smoke_test.py
```

### 3. Run Structured Retrieval on Holdout-3 (Zero LLM API Cost)
```bash
# Evaluates deterministic retrieval across Holdout-3 (S0 through S5)
python scripts/run_holdout3_experiment.py
```

---

## Repository Structure & Research Index

```text
├── configs/
│   ├── structured_retrieval_final.yaml    # RECOMMENDED VALIDATED CONFIGURATION (S3)
│   ├── dense_baseline.yaml                # Standard Dense Top-5 baseline (S0)
│   └── v3_truegraph_experimental.yaml     # Historical graph-routing configuration (S4)
├── benchmark/
│   ├── README.md                          # Exposure status across all 6 benchmark suites
│   ├── holdout3/                          # FINAL CAUSAL MECHANISM BENCHMARK (N=240)
│   ├── holdout2/                          # Historical full-stack confirmation (N=250)
│   ├── scale1/                            # Scale robustness stress test (N=80)
│   └── hub1/                              # Knowledge graph hub stress test (N=100)
├── reports/
│   ├── FINAL_REPORT.md                    # Comprehensive authoritative research report
│   ├── MECHANISM_HOLDOUT3_REPORT.md       # Mechanism Holdout-3 statistical report
│   ├── README.md                          # Chronological 12-phase reports index
│   └── integrity/                         # Gate I 100% corpus integrity audit reports
├── CORPUS_CORRECTED_FREEZE.md             # Hash-verified corrected frozen corpus manifest
└── REPRODUCIBILITY.md                     # Step-by-step reproduction guide and checksums
```

---

## Citations & Research History

This repository is frozen as a permanent scientific record. For complete details of each phase, consult:
- [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md): Complete research report detailing failures, ablations, audits, and conclusions.
- [`reports/MECHANISM_HOLDOUT3_REPORT.md`](reports/MECHANISM_HOLDOUT3_REPORT.md): Mechanism Holdout-3 statistical tests and causal attribution.
- [`CORPUS_CORRECTED_FREEZE.md`](CORPUS_CORRECTED_FREEZE.md): Cryptographic manifest of the corrected frozen corpus.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md): Exact environments, hashes, and execution commands.
