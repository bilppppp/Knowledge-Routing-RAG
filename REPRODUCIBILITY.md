# Reproducibility Guide & Experimental Manifest

This document provides exact system requirements, cryptographic hashes, random seeds, benchmark manifests, and step-by-step commands to reproduce all evaluations in `Knowledge-Routing-RAG`.

---

## 1. Frozen Experimental Environment

### Hardware Environment (Audit Reference)
- **Architecture**: Apple Silicon M4 (arm64, 10-core CPU)
- **RAM**: 24.0 GB Unified Memory
- **Operating System**: macOS Sequoia (Darwin 25.6.0)

### Software & Service Stack
- **Python**: 3.12.9 (managed via `.venv` or `uv`)
- **Vector Database**: Qdrant Docker `v1.8.2` (Host port `55001` or `6333`)
- **Embedding Backend**: Infinity / TEI Docker container serving `BAAI/bge-large-zh-v1.5` (1024d) at `http://localhost:8000/v1`
- **Graph & FTS5 Engine**: Embedded SQLite 3 with FTS5 and `jieba` tokenization (`data/knowledge_lsdb.sqlite`)
- **LLM Generators & Judges**:
  - Holdout-1 & Holdout-2: DeepSeek API (`deepseek-chat`, Temperature = 0.0)
  - Mechanism Holdout-3: Gemini API (`gemini-3.8-flash`, Temperature = 0.0, 4096-token ceiling)

---

## 2. Cryptographic Integrity Signatures (SHA-256)

All critical code modules and dataset artifacts are frozen. Verify checksums using:
```bash
shasum -a 256 <file_path>
```

| Component / Artifact | Relative Path | Expected SHA-256 Checksum |
|:---|:---|:---|
| **V3 Router Implementation** | `src/routing/e2_descent_router.py` | `946e59c96acc9c572f6edf95a99851b6d65e66c3b2e8f51b91270bf240e7fec5` |
| **E1 Composer Module** | `src/composition/composer.py` | `224ef0e88364a969a7c919a4195d4a26ab7f86373a33d1c0625c7f12fa09291d` |
| **E2 Descent Module** | `src/composition/descent.py` | `001497c30cc07d51995771834d2d3e0c98738ddf4517aadfafab6943d8b7fa0c` |
| **Slot Decomposition** | `src/composition/slots.py` | `e6d5b960c924e7d350d9cc8bb9eba8aaf070e3b4419f4ec0dbd8648efde9953f` |
| **Coverage Estimator** | `src/composition/coverage.py` | `e75d4af5cae647c61fcfebea2b2ad3d657dca634c9f1273e3e47944fbdd3b7c5` |
| **B0 Raw Prompt Module** | `src/common/prompt.py` | `cc9c8ad349c642275b233db5f9e823751735377c80e30f62cd3bf7c16d023961` |
| **B0 Baseline Retrieval** | `src/retrieval/vector_rag.py` | `9da475a64725684a9e63e0b799bd15b913747fc7935c44bc0262891d29fef6fd` |
| **Corrected Knowledge LSDB** | `data/knowledge_lsdb.sqlite` | `a1f5881acf21d0db2b3803f1a16f1341e01324164bec16448bf018ab4446ebb3` |
| **Corrected Corpus Chunks** | `data/chunks.jsonl` | `d3c054be36cd5c805b9d6f0612aa2771990aa3bd890551ca72579d9d3f32513c` |
| **Corrected Corpus Manifest** | `data/manifests/corpus_manifest.json` | `bb21f21fb3f4785c23ac82e18d926e5185252d498d1f0a1da26dad8325add47f` |
| **Holdout-3 Gold Benchmark** | `benchmark/holdout3/gold.jsonl` | `a8c9cc2329a8853901ca12c4b8ed8f8c322e196c0a9d572b67d33bfc9ae66051` |
| **Holdout-3 Questions** | `benchmark/holdout3/questions.jsonl` | `0f4193f8d580b272331d8c2b6c28f444a360849eba8315a8ebddecd4c8a0a232` |

---

## 3. Pre-Registered Random Seeds & Determinism

- **Temperature**: `0.0` (Strict deterministic decoding)
- **Pre-registered Seeds (3-Run Protocol)**:
  - Run 1: `seed = 101`
  - Run 2: `seed = 202`
  - Run 3: `seed = 303`

---

## 4. Benchmark Manifests & Metadata

| Suite | Questions Path | Gold Truth Path | Manifest Path | Sample Size ($N$) | Scientific Role |
|:---|:---|:---|:---|:---:|:---|
| **Mechanism Holdout-3** | `benchmark/holdout3/questions.jsonl` | `benchmark/holdout3/gold.jsonl` | `benchmark/holdout3/manifest.json` | 240 | **Final Causal Mechanism Evaluation (Corrected Corpus)** |
| **Holdout-2** | `benchmark/holdout2/questions.jsonl` | `benchmark/holdout2/gold.jsonl` | `benchmark/holdout2/manifest.json` | 250 | Historical Full-Stack Confirmation (Old Snapshot) |
| **Holdout-1** | `benchmark/confirmation/questions.jsonl` | `benchmark/confirmation/gold.jsonl` | `benchmark/confirmation/manifest.json` | 200 | Initial Decontaminated Architecture Confirmation |
| **ScaleSet-1** | `benchmark/scale1/questions.jsonl` | `benchmark/scale1/gold.jsonl` | `benchmark/scale1/manifest.json` | 80 | Scale Robustness Stress Characterization |
| **HubSet-1** | `benchmark/hub1/questions.jsonl` | `benchmark/hub1/gold.jsonl` | `benchmark/hub1/manifest.json` | 100 | Knowledge Graph Hub Node Stress Characterization |
| **Dev-216** | `benchmark/questions.jsonl` | `benchmark/gold.jsonl` | `benchmark/split.json` | 216 runs | Development / Contaminated Optimization Set |

---

## 5. Reproduction Protocols: Three Verification Levels

### Level 1 — Release Smoke Test (Zero API Calls)
Validates that the Python environment, database connections, all configurations, B0 baseline, and V3 router execute correctly and adhere to the $\le 5$ chunks budget.

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run release smoke test
python scripts/release_smoke_test.py
```

**Expected Output**:
```text
======================================================
 Knowledge-Routing-RAG — Release Smoke Test 
======================================================
  [PASS] Core modules imported successfully.
  [PASS] Runtime configuration files loaded and validated (B0, S0, V3, S3-Final, S4-TrueGraph).
  [PASS] Knowledge LSDB loaded (100 documents, 2862 chunks, 1397 routing edges).
  [PASS] B0 single query executed (retrieved 5 chunks).
  [PASS] V3-Frozen single query executed (retrieved 5 chunks, budget constraint <= 5 chunks satisfied).

All smoke tests passed successfully! Release candidate is ready.
```

---

### Level 2 — Deterministic Retrieval Reproduction (Zero LLM API Cost)
Reproduces the exact deterministic retrieval outputs for all 6 systems ($S_0 \sim S_5$) across Mechanism Holdout-3 ($N=240$) using local SQLite and Qdrant:

```bash
# Inspect pre-computed retrieval results
cat reports/holdout3_retrieval.json | grep -A 25 '"summary"'
```

**Key Verification Targets**:
- $S_0$ (Dense Top-5) Chain Completion: **40.42%**
- $S_2$ (Metadata + BM25) Chain Completion: **47.08%**
- $S_3$ (V3-NoGraph) Chain Completion: **47.92%**
- $S_4$ (V3-TrueGraph) Chain Completion: **44.17%**
- $S_5$ (V3-ShuffledGraph) Chain Completion: **45.00%**

---

### Level 3 — Full Mechanism Evaluation Run
Runs the complete independent generation and judging evaluation on Mechanism Holdout-3 ($N=240$):

```bash
# Configure API keys in .env
# DEEPSEEK_API_KEY=... (or GEMINI_API_KEY)
python scripts/run_holdout3_experiment.py
```

**Key Verification Targets (`reports/holdout3_run1.json` / `reports/MECHANISM_HOLDOUT3_REPORT.md`)**:
- $S_0$ (Dense Top-5): **62.92%**
- $S_2$ (Metadata + BM25): **69.17%** ($\Delta = +6.25\text{pp}$, $p = 0.0051$)
- $S_3$ (V3-NoGraph): **69.58%** ($\Delta = +6.67\text{pp}$, $p = 0.0033$)
- $S_4$ (V3-TrueGraph): **64.17%** ($\Delta = -5.42\text{pp}$ vs $S_3$, $p = 0.0059$)
- $S_5$ (V3-ShuffledGraph): **63.33%** ($\Delta = +0.83\text{pp}$ vs $S_4$, $p = 0.7728$)
- True Graph Rescues = **0**, Graph Regressions = **9**, Net Graph Causal Gain = **-9**
