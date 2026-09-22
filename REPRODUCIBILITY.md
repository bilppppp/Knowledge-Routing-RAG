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
- **Embedding Backend**: Infinity / TEI Docker container serving `BAAI/bge-large-zh-v1.5` or `google/embeddinggemma-300m` (768d / 1024d)
- **Graph & FTS5 Engine**: Embedded SQLite 3 with FTS5 and `jieba` tokenization (`data/knowledge_lsdb.sqlite`)
- **LLM Generator & Judge**: DeepSeek API (`deepseek-chat`, Temperature = 0.0)

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
| **Knowledge LSDB Database** | `data/knowledge_lsdb.sqlite` | `5495e97f18f2e414622cca162f8f6051cf88b77e8d3aefd565dcb7950ec9bfa8` |
| **Corpus Chunks** | `data/chunks.jsonl` | `7763192631df6bd55a0db1188ae8fcf3e7ff212304a0a5e75605d4539efeca7a` |
| **D100 Manifest** | `data/manifests/d100.json` | `852865dc239b65c10fc8e99b0469f43886b624f315ca6060eeb26a4666f0fd22` |

---

## 3. Pre-Registered Random Seeds & Determinism

- **Generator Model**: `deepseek-chat`
- **Temperature**: `0.0` (Strict deterministic decoding)
- **Pre-registered Seeds (3-Run Protocol)**:
  - Run 1: `seed = 101`
  - Run 2: `seed = 202`
  - Run 3: `seed = 303`

---

## 4. Benchmark Manifests & Metadata

| Suite | Questions Path | Gold Truth Path | Manifest Path | Sample Size ($N$) |
|:---|:---|:---|:---|:---:|
| **Holdout-2** | `benchmark/holdout2/questions.jsonl` | `benchmark/holdout2/gold.jsonl` | `benchmark/holdout2/manifest.json` | 250 |
| **Holdout-1** | `benchmark/confirmation/questions.jsonl` | `benchmark/confirmation/gold.jsonl` | `benchmark/confirmation/manifest.json` | 200 |
| **ScaleSet-1** | `benchmark/scale1/questions.jsonl` | `benchmark/scale1/gold.jsonl` | `benchmark/scale1/manifest.json` | 80 |
| **HubSet-1** | `benchmark/hub1/questions.jsonl` | `benchmark/hub1/gold.jsonl` | `benchmark/hub1/manifest.json` | 100 |
| **Dev-216** | `benchmark/questions.jsonl` | `benchmark/gold.jsonl` | `benchmark/split.json` | 216 runs |

---

## 5. Reproduction Protocols: Three Verification Levels

### Level 1 — Smoke Test & Single Query Verification
Validates that Python environment, database connections, B0 baseline, and V3 router execute correctly without needing a full benchmark evaluation or significant API costs.

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run release smoke test
python scripts/release_smoke_test.py
```

**Expected Output**:
```text
[PASS] Dependencies and modules imported successfully
[PASS] Runtime configs loaded (B0 & V3)
[PASS] SQLite Knowledge LSDB verified (100 docs, 2862 chunks)
[PASS] B0 Vector search returned 5 chunks
[PASS] V3 Knowledge Routing completed
[PASS] Final evidence budget constraint verified (≤ 5 chunks, ≤ 4000 tokens)
```

---

### Level 2 — Deterministic Retrieval Reproduction (Zero LLM API Cost)
Evaluates retrieval and evidence composition deterministically over Holdout-2 ($N=250$) without making external LLM generation calls.

```bash
# Run deterministic retrieval evaluation on Holdout-2
python scripts/run_holdout2_experiment.py --retrieval-only
```

**Expected Metrics (D100)**:
- B0 Gold Document Recall: **$88.93\%$**
- V3 Gold Document Recall: **$98.53\%$** ($\Delta = \mathbf{+9.60\text{pp}}$)
- B0 Gold Chunk Recall: **$75.73\%$**
- V3 Gold Chunk Recall: **$76.53\%$** ($\Delta = \mathbf{+0.80\text{pp}}$)
- B0 Chain Completion Rate: **$54.40\%$**
- V3 Chain Completion Rate: **$57.60\%$** ($\Delta = \mathbf{+3.20\text{pp}}$)
- Useful Evidence Evictions: **10 / 250** ($4.0\%$)

---

### Level 3 — Full Experimental Reproduction (End-to-End LLM Generation)
Reproduces the complete multi-run experiment suite. Requires valid `DEEPSEEK_API_KEY`.

#### Step 3.1: Reproduce Holdout-2 (The Confirmed Core Result)
```bash
python scripts/run_holdout2_experiment.py
```
- **Runs Executed**: 3 fresh runs for B0 and V3 with seeds 101, 202, 303.
- **Expected Outcome**:
  - B0 Majority Accuracy: **$67.60\%$**
  - V3 Majority Accuracy: **$73.20\%$**
  - Delta: **$+5.60\text{pp}$**
  - Stable Rescues: **18**, Stable Regressions: **4**, Net: **+14**
  - Exact McNemar $p = 0.0043$

#### Step 3.2: Reproduce Phase C (Scale Robustness)
```bash
python scripts/run_scale1_experiment.py
```
- **Expected Outcome**:
  - B0 $D_{20} \to D_{100}$ Drop: $+0.00\text{pp}$
  - V3 $D_{20} \to D_{100}$ Drop: $+2.50\text{pp}$
  - Robustness Advantage (RA): **$-2.50\text{pp}$** (`VERDICT S-C: NOT CONFIRMED`)

#### Step 3.3: Reproduce Phase D (Hub Stress)
```bash
python scripts/run_hub1_experiment.py
```
- **Expected Outcome**:
  - Low-Degree Accuracy (Buckets A & B): B0 $57.50\%$, V3 $62.50\%$
  - High-Degree Accuracy (Buckets D & E): B0 $60.00\%$, V3 $62.50\%$
  - V3 Candidate P95: expands from $10.1$ to $27.1$ (`VERDICT H2-C: NOT CONFIRMED`)

#### Step 3.4: Reproduce Phase E (Cost & Latency Benchmark)
```bash
python scripts/benchmark_phase_e_cost.py
```
- **Expected Outcome**:
  - B0 Local Latency P50: $3.67\text{ ms}$
  - V3 Local Latency P50: $4.34\text{ ms}$
  - P50 Overhead: $+0.58\text{ ms}$
  - Additional LLM Calls: **0**
  - Additional Query Embeddings: **0**
