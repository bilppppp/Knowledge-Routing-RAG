# Knowledge-Routing-RAG

> **Constrained Graph Navigation for Complex Regulatory Retrieval**  
> *A research prototype exploring knowledge routing as a navigation plane rather than evidence context.*

[English](README.md) | [中文说明](README_CN.md)

[![Release](https://img.shields.io/badge/release-v3--research--final-blue.svg)](https://github.com/bilppppp/Knowledge-Routing-RAG/releases)
[![Status](https://img.shields.io/badge/status-frozen__research-success.svg)](#10-limitations--scope-of-validity)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](requirements.txt)

---

## Project Summary

**Knowledge-Routing-RAG** explores a constrained retrieval architecture where graphs and metadata are used as a **navigation plane** rather than evidence context. On an independent 250-question holdout, the frozen V3 system improved end-to-end answer accuracy from **67.6% to 73.2%** over a Vector-RAG baseline while adding only low-millisecond local routing overhead and **no additional online LLM calls**. Scale robustness and hub robustness were **not confirmed**.

---

## 1. What is This?

This repository contains the complete, frozen research code, experimental protocols, evaluation artifacts, and definitive conclusions of the **Knowledge-Routing-RAG** project.

The experimental phase is complete. The system architecture, prompts, and evaluation datasets are permanently frozen under [`V3_FREEZE.md`](V3_FREEZE.md).

### The Problem
Standard dense Vector RAG relies on semantic similarity between query and chunk embeddings. In complex technical domains—such as regulatory compliance, healthcare administration, and legal synthesis—vector search struggles with:
- **Cross-document dependencies**: Critical answers depend on interactions between separate statutes.
- **Multi-hop evidence chains**: A rule derives authority from an overarching law (`BASED_ON`), or delegates details to implementation rules (`REFERENCES`).
- **Temporal modifications**: New rules amend or supersede older versions (`AMENDS`, `SUPERSEDES`).

In these cases, Vector RAG frequently retrieves superficially similar paragraphs from a single document while missing companion evidence from related documents.

### The Idea: Candidate Space $\ne$ Evidence Context
Naive Graph RAG often dumps entire graph neighborhoods into the LLM prompt, causing **graph flooding** and context dilution.

Knowledge Routing decouples discovery from synthesis:
- **Metadata and Graphs are for Navigation**: Relational topologies and shadow candidate pools are used exclusively on the control plane to discover target documents.
- **Context is Strictly Budgeted**: The generation context is limited to $\le 5$ chunks ($\le 4000$ tokens), composed via coverage-preserving set utility.

> **Principle**: *Route globally, retrieve locally, compose globally.*

---

## 2. Frozen Architecture (V3-Frozen)

The verified system is **V3-Frozen** (candidate `E2-Lite`), which pairs global routing with local lexical descent and coverage-preserving composition:

```text
Question
   │
   ▼
Vector Entrance
   │
   ├── Fast Path (Sim ≥ 0.85) ─────────────┐
   │                                       │
   ▼                                       │
Shadow Knowledge Routing                   │
   │                                       │
   ▼                                       │
Target Document Resolution                 │
   │                                       │
   ▼                                       │
Lexical Targeted Descent (BM25)            │
   │                                       │
   ▼                                       │
Coverage-Aware Composer ◄──────────────────┘
   │
   ▼
Final Evidence (≤ 5 chunks, ≤ 4000 tokens)
   │
   ▼
LLM Generator (DeepSeek-Chat, Temp=0.0, Raw B0 Prompt)
```

1. **FIB Vector Entrance**: Top-5 dense vector retrieval (`google/embeddinggemma-300m`).
2. **Fast Path**: Bypasses routing if top vector match similarity $\ge 0.85$ (preserves single-hop performance).
3. **Shadow Knowledge Routing**: Inspects a Top-20 shadow candidate plane (RIB) to aggregate document prefixes, resolving cross-statute edges (`BASED_ON`, `REFERENCES`) and parent lifts.
4. **Targeted Local Descent (E2-Lite)**: Once a target document is resolved, executes generic BM25 lexical descent conditioned on unresolved query slots. *(Dense local vectors and RRF were ablated and deleted due to zero incremental yield).*
5. **Coverage-Aware Composer (E1)**: Optimizes marginal set coverage to prevent useful evidence eviction.
6. **B0 Raw Prompt**: Passes evidence to the LLM using the exact baseline prompt template (zero prompt engineering artifacts).

---

## 3. Key Results: Independent Holdout-2

The primary claim rests on **Independent Holdout-2** ($N = 250$ unseen questions on the full $D_{100}$ corpus, evaluated across 3 fresh runs with random seeds 101, 202, 303, and verified via double-blind human adjudication):

| Evaluation Metric | Vector RAG Baseline (B0) | V3-Frozen Clean Routing | Delta ($\Delta$) | Statistical Test |
|:---|:---:|:---:|:---:|:---|
| **Majority Accuracy** | **67.60%** (169/250) | **73.20%** (183/250) | **+5.60pp** | **Exact McNemar $p = 0.0043$** |
| 95% Paired Bootstrap CI | — | — | **[+2.00pp, +9.20pp]** | Strictly bounded above 0 |
| Stable Rescues ($B0=0, V3=1$) | — | — | **18** | Exact Binomial $p = 0.0044$ |
| Stable Regressions ($B0=1, V3=0$) | — | — | **4** | Pre-registered criterion met |
| **Net Stable Rescue** | — | — | **+14** | Primary endpoint confirmed |
| **Gold Document Recall** | 88.93% | 98.53% | **+9.60pp** | Navigation mechanism confirmed |
| **Gold Chunk Recall** | 75.73% | 76.53% | **+0.80pp** | Chunk evidence yield |
| **Chain Completion Rate** | 54.40% | 57.60% | **+3.20pp** (+8 chains) | Complete cross-statute chains |
| **Multi-Hop (2-Hop+) Accuracy** | 48.72% | 57.69% | **+8.97pp** | Concentrated multi-hop gain |
| **Simple (1-Hop) Accuracy** | 98.94% | 98.94% | **+0.00pp** | **Zero simple regression** |

---

## 4. What Was Confirmed vs. What Was NOT Confirmed

### Confirmed Findings
- [x] **End-to-End Accuracy Gain**: Confirmed ($+5.60\text{pp}$, McNemar $p = 0.0043$, 95% CI `[+2.00pp, +9.20pp]`).
- [x] **Document Navigation Gain**: Confirmed ($+9.60\text{pp}$ Gold Document Recall).
- [x] **Multi-Hop Reasoning Gain**: Confirmed ($+8.97\text{pp}$ on multi-hop tasks; 0 regressions on simple tasks).
- [x] **Evidence Composition Necessity**: Confirmed (E1 Composer eliminated useful evidence evictions from $9.5\%$ to $0.0\%$).
- [x] **Low Engineering Overhead**: Confirmed (Mean local routing overhead $+3.63\text{ ms}$, P50 $+0.58\text{ ms}$; 0 extra online LLM calls; 0 query embeddings; $+1.96\%$ input tokens).

### NOT Confirmed / Negative Findings
- [ ] **Scale Robustness**: **`NOT CONFIRMED`**. In Phase C ($N=80$, $D_{20} \to D_{100}$), V3 degraded $+2.50\text{pp}$ while B0 was flat ($+0.00\text{pp}$), yielding a Robustness Advantage of $\text{RA} = -2.50\text{pp}$ (95% CI `[-6.25pp, 0.00pp]`). *We do NOT claim V3 is more robust to corpus expansion than Vector RAG.*
- [ ] **Hub Node Stability**: **`NOT CONFIRMED`**. In Phase D ($N=100$), high-degree nodes ($>20$ connections) caused V3 candidate pool P95 to expand to 27.1 chunks, and chain completion dropped from $60.0\%$ to $42.5\%$.
- [ ] **Graph Flooding Suppression**: **`NOT TESTABLE`**. The historical unconstrained graph baseline was unavailable in Git history; comparative flooding suppression was therefore not testable.
- [ ] **Universal Superiority**: **`NOT CLAIMED`**. Gains are established only within a structured statutory regulatory domain.

---

## 5. Explicit Methodological Disclosure: Dev-216

Early in development, 216 instance runs (120 unique questions) were used to test candidates C1 through C7.

Because this set was repeatedly inspected during error analysis and prompt adjustments:
- **Dev-216 became an Optimization / Development Set**.
- C7's apparent $80.09\%$ score on this set was partially inflated by benchmark-specific heuristics.
- Decontamination removed these rules, establishing the generic `C7-Clean` architecture.
- **The only valid confirmatory evidence for out-of-sample generalization is Holdout-2 ($73.20\%$).**

---

## 6. Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (for Qdrant & Embedding server)

### 1. Clone & Setup Environment
```bash
git clone https://github.com/bilppppp/Knowledge-Routing-RAG.git
cd Knowledge-Routing-RAG

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env to provide your DEEPSEEK_API_KEY if running generation experiments
```

### 3. Launch Vector & Embedding Services
```bash
docker compose up -d
```

### 4. Run Release Smoke Test
```bash
python scripts/release_smoke_test.py
```

Expected output:
```text
[PASS] Core modules imported successfully.
[PASS] Runtime configuration files loaded and validated.
[PASS] Knowledge LSDB loaded (100 documents, 2862 chunks, 1397 routing edges).
[PASS] B0 single query executed (retrieved 5 chunks).
[PASS] V3-Frozen single query executed (retrieved 5 chunks, budget constraint <= 5 chunks satisfied).
All smoke tests passed successfully! Release candidate is ready.
```

---

## 7. Reproduction Protocols

Detailed reproduction steps, cryptographic hashes, and parameter manifests are provided in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

- **Level 1 — Smoke Test**:
  ```bash
  python scripts/release_smoke_test.py
  ```
- **Level 2 — Deterministic Retrieval Reproduction (Zero LLM API cost)**:
  ```bash
  python scripts/run_holdout2_experiment.py --retrieval-only
  ```
- **Level 3 — Full Experimental Suite (Requires DeepSeek API Key)**:
  ```bash
  # Holdout-2 Primary Confirmation (N=250, 3 runs)
  python scripts/run_holdout2_experiment.py

  # Phase C: Scale Robustness (N=80)
  python scripts/run_scale1_experiment.py

  # Phase D: Hub Stress (N=100)
  python scripts/run_hub1_experiment.py

  # Phase E: Cost & Complexity Benchmark
  python scripts/benchmark_phase_e_cost.py
  ```

---

## 8. Repository Structure

```text
Knowledge-Routing-RAG/
├── README.md                      # Main project documentation (this file)
├── REPRODUCIBILITY.md             # Environment, seeds, hashes & reproduction steps
├── V3_FREEZE.md                   # Frozen implementation manifest & parameter hashes
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── docker-compose.yml             # Qdrant vector database & local embedding service
├── configs/
│   ├── b0_baseline.yaml           # Pure Vector RAG configuration
│   ├── v3_frozen.yaml             # Frozen Knowledge Routing configuration
│   └── runtime_config.yaml        # Legacy runtime configuration
├── data/
│   ├── knowledge_lsdb.sqlite      # SQLite Knowledge LSDB (FTS5 + Relational DiGraph)
│   ├── chunks.jsonl               # 2,862 text chunks across 100 documents
│   ├── documents/                 # Raw document texts (doc001–doc100)
│   └── manifests/                 # Corpus subsets (D20, D50, D100)
├── benchmark/
│   ├── README.md                  # Benchmark exposure & contamination manifest
│   ├── questions.jsonl            # Dev-216 (Optimization set, EXPOSED)
│   ├── confirmation/              # Holdout-1 (N=200, EXPOSED AFTER CONFIRMATION)
│   ├── holdout2/                  # Holdout-2 (N=250, FINAL INDEPENDENT CONFIRMATION)
│   ├── scale1/                    # ScaleSet-1 (N=80, SCALE ROBUSTNESS SET)
│   └── hub1/                      # HubSet-1 (N=100, HUB STRESS SET)
├── src/
│   ├── retrieval/                 # B0 Vector RAG implementation
│   ├── routing/                   # Knowledge routers (C7-Clean, E1, E2-Lite)
│   ├── composition/               # E1 Coverage-preserving composer & E2 descent
│   ├── graph/                     # Knowledge LSDB graph interface
│   ├── services/                  # Search, embedding, and LLM clients
│   └── evaluation/                # Metrics and evaluation logic
├── scripts/
│   ├── release_smoke_test.py      # Minimum release verification test
│   ├── run_holdout2_experiment.py # Holdout-2 confirmation runner
│   ├── run_scale1_experiment.py   # Phase C scale runner
│   ├── run_hub1_experiment.py     # Phase D hub stress runner
│   └── benchmark_phase_e_cost.py  # Phase E cost & latency benchmark
└── reports/
    ├── README.md                  # Chronological research reports index
    ├── FINAL_REPORT.md            # Comprehensive, definitive research report
    └── final_repository_audit.md  # Release readiness & code audit report
```

---

## 9. Key Reports & Scientific Artifacts

All experimental reports, blind review records, and raw JSON outputs are cataloged in [`reports/README.md`](reports/README.md):
- **Definitive Synthesis**: [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md)
- **Holdout-2 Confirmation**: [`reports/v3_holdout2_confirmation.md`](reports/v3_holdout2_confirmation.md)
- **Phase C Scale Robustness**: [`reports/v3_scale_robustness_confirmation.md`](reports/v3_scale_robustness_confirmation.md)
- **Phase D Hub Stress**: [`reports/v3_hub_stress_confirmation.md`](reports/v3_hub_stress_confirmation.md)
- **Phase E Cost & Latency**: [`reports/v3_cost_complexity_audit.md`](reports/v3_cost_complexity_audit.md)
- **Decontamination Audit**: [`reports/decontamination_rule_audit.md`](reports/decontamination_rule_audit.md)

---

## 10. Limitations & Scope of Validity

1. **Domain Boundary**: All evaluations were conducted on a corpus of 100 Chinese medical and healthcare administrative regulations ($2,862$ chunks). Findings specifically exploit statutory cross-references (`BASED_ON`, `REFERENCES`, `SUPERSEDES`, `AMENDS`) and should not be extrapolated to unstructured narrative corpora or open-domain web data without domain adaptation.
2. **Corpus Scale Boundary**: Evaluated up to 100 documents ($D_{100}$). Scale degradation in Phase C indicates that larger corpora ($1,000+$ documents) will require tighter prefix pruning.
3. **Model Configuration**: Tested with `deepseek-chat` under greedy decoding ($T=0.0$).
4. **Status**: This codebase is a **frozen research prototype** designed to validate retrieval principles, not a production-hardened microservice.

---

## 11. License

This project is licensed under the [MIT License](LICENSE).
